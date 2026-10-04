#!/usr/bin/env python3
"""High-N test of PCAone --robust: do relatives distort the PCs at biobank-like
sample sizes, and in which PCs?

Simulated data: Balding-Nichols populations (K = 5, F_ST from 0.05 down to
0.002, i.e. strong to very weak structure) plus a group admixed between
populations 0 and 1; unlinked SNPs. Families are built from founders by
Mendelian transmission: MZ twins, sib pairs, parent-offspring trios, nuclear
families (2 parents + 2 children), three-generation pedigrees (a grandparent,
two of their children and two grandchildren) and first-cousin pairs (below the
2nd-degree threshold). A fraction --rel of the sample is in a family.

Truth: PCA of a separate reference panel of unrelated individuals, onto which
the analysis sample is projected (relatives at their realised position).
Scores per method and -k:
  minR2     min over the top K-1 true PCs of the R2 on the method's top K-1 PCs
            (unrelated individuals only)
  err       relatives' placement error relative to the spread
  fam_max   for the extra PCs (K .. k): the largest share of a PC's variance
            explained by family membership (eta^2; 1 = a pure family axis)
  fam_n50   number of extra PCs with eta^2 > 0.5
"""
import argparse
import os
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import write_bed  # noqa: E402

FST = [0.05, 0.03, 0.01, 0.005, 0.002]


class Sim:
    def __init__(self, M, rng):
        self.rng = rng
        p = rng.uniform(0.05, 0.95, M)
        P = []
        for f in FST:
            a, b = p * (1 - f) / f, (1 - p) * (1 - f) / f
            P.append(rng.beta(a, b))
        self.P = np.clip(np.array(P), 0.01, 0.99)
        self.M = M

    def ancestry(self):
        """a random ancestry vector: 85% single population, 15% admixed 0/1"""
        q = np.zeros(len(FST))
        if self.rng.random() < 0.15:
            x = self.rng.random()
            q[0], q[1] = x, 1 - x
        else:
            q[self.rng.integers(len(FST))] = 1
        return q

    def founder(self, q=None):
        q = self.ancestry() if q is None else q
        pi = q @ self.P
        r = self.rng.random
        return ((r(self.M) < pi).astype(np.int8) + (r(self.M) < pi).astype(np.int8)), q

    def child(self, ga, gb):
        r = self.rng.random
        return ((r(self.M) < ga / 2).astype(np.int8) + (r(self.M) < gb / 2).astype(np.int8))


def make_families(sim, n_target):
    """families until n_target individuals; returns list of genotype lists"""
    fams = []
    n = 0
    types = ["mz", "sibs", "trio", "nuclear", "pedigree", "cousins"]
    wts = np.array([0.05, 0.25, 0.2, 0.2, 0.1, 0.2])
    while n < n_target:
        t = types[sim.rng.choice(len(types), p=wts)]
        a, q = sim.founder()
        b, _ = sim.founder(q)  # partner of the same ancestry
        if t == "mz":
            m = [a, a.copy()]
        elif t == "sibs":
            m = [sim.child(a, b), sim.child(a, b)]
        elif t == "trio":
            m = [a, b, sim.child(a, b)]
        elif t == "nuclear":
            m = [a, b, sim.child(a, b), sim.child(a, b)]
        elif t == "pedigree":
            c1, c2 = sim.child(a, b), sim.child(a, b)
            sp, _ = sim.founder(q)
            m = [a, c1, c2, sim.child(c1, sp), sim.child(c1, sp)]
        else:  # first cousins: children of two sibs with unrelated partners
            c1, c2 = sim.child(a, b), sim.child(a, b)
            s1, _ = sim.founder(q)
            s2, _ = sim.founder(q)
            m = [sim.child(c1, s1), sim.child(c2, s2)]
        fams.append((t, m))
        n += len(m)
    return fams


def simulate(N, rel, M, n_ref, seed):
    sim = Sim(M, np.random.default_rng(seed))
    fams = make_families(sim, int(round(rel * N)))
    n_fam = sum(len(m) for _, m in fams)
    G, famid, ftype = [], [], []
    for f, (t, m) in enumerate(fams):
        G += m
        famid += [f + 1] * len(m)
        ftype += [t] * len(m)
    for _ in range(N - n_fam):
        G.append(sim.founder()[0])
        famid.append(0)
        ftype.append("unrelated")
    R = np.array([sim.founder()[0] for _ in range(n_ref)])
    return np.array(G), np.array(famid), np.array(ftype), R


def reference_truth(R, G, k):
    """PCA of the reference panel R, analysis sample G projected; unit-length columns"""
    f = R.mean(0) / 2
    keep = (f > 0.01) & (f < 0.99)
    sd = np.sqrt(2 * f[keep] * (1 - f[keep]))
    Xr = (R[:, keep] - 2 * f[keep]) / sd
    _, _, Vt = np.linalg.svd(Xr, full_matrices=False)
    T = ((G[:, keep] - 2 * f[keep]) / sd) @ Vt[:k].T
    return T / np.linalg.norm(T, axis=0)


def score(U, T, base, rel, kk):
    U = U[:, :kk]
    Z = np.c_[U, np.ones(len(U))]
    B, *_ = np.linalg.lstsq(Z[base], T[base, :kk], rcond=None)
    P = Z @ B
    Tb = T[base, :kk]
    r2 = 1 - ((Tb - P[base]) ** 2).sum(0) / ((Tb - Tb.mean(0)) ** 2).sum(0)
    spread = np.sqrt(((Tb - Tb.mean(0)) ** 2).sum(1).mean())
    err = np.sqrt(((T[rel, :kk] - P[rel]) ** 2).sum(1).mean()) / spread if len(rel) else np.nan
    return r2, err


def family_eta2(u, famid):
    """share of the variance of u explained by family membership (unrelated
    individuals form one group)"""
    u = u - u.mean()
    tot = (u ** 2).sum()
    between = 0.0
    for f in np.unique(famid[famid > 0]):
        x = u[famid == f]
        between += len(x) * x.mean() ** 2
    x = u[famid == 0]
    between += len(x) * x.mean() ** 2
    return between / tot


def run(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"failed: {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return time.time() - t0, r.stdout + r.stderr


def one_dataset(a, N, rel, rep):
    K = len(FST)  # populations; the admixed group lies between 0 and 1
    kk = K - 1
    seed = zlib.crc32(f"{N}|{rel}|{rep}".encode())
    pre = os.path.join(a.work, f"hn_N{N}_rel{int(rel * 100)}_r{rep}")
    G, famid, ftype, R = simulate(N, rel, a.M, a.nref, seed)
    keep = G.min(0) != G.max(0)
    G, R = G[:, keep], R[:, keep]
    T = reference_truth(R, G, kk)
    bim = [f"1\tsnp{i}\t0\t{i + 1}\tA\tC" for i in range(G.shape[1])]
    write_bed(pre, [[str(famid[i]) if famid[i] else f"U{i}", f"I{i}", "0", "0", "0", "-9"] for i in range(N)],
              bim, G)
    del R
    base = np.where(famid == 0)[0]
    relidx = np.where(famid > 0)[0]
    tk, _ = run(f"{a.plink2} --bfile {pre} --make-king-table --king-table-filter 0.04 "
                f"--threads {a.threads} --out {pre}_king")
    rows = [dict(N=N, rel=rel, rep=rep, method="plink2-KING", k="", sec=round(tk, 1))]
    for k in [int(x) for x in a.ks.split(",")]:
        for method in a.methods.split(","):
            out = f"{pre}_{method}_k{k}"
            if method == "standard":
                cmd = f"{a.pcaone} -b {pre} -k {k} -n {a.threads} -v 1 -o {out}"
            else:
                cmd = (f"{a.pcaone} -b {pre} -k {k} -n {a.threads} -v 1 --robust {method} "
                       f"--kinship {pre}_king.kin0 -o {out}")
            try:
                t, log = run(cmd)
            except RuntimeError as e:
                print(e, file=sys.stderr, flush=True)
                continue
            U = np.loadtxt(out + ".eigvecs", ndmin=2)
            r2, err = score(U, T, base, relidx, kk)
            eta = [family_eta2(U[:, j], famid) for j in range(kk, U.shape[1])]
            rank = ""
            for line in log.split("\n"):
                if "fit, rank" in line:
                    rank = line.split("fit, rank")[1].split()[0].split("(")[0]
            rows.append(dict(N=N, rel=rel, rep=rep, method=method, k=k, sec=round(t, 1),
                             minR2=round(float(r2.min()), 4), r2=";".join(f"{x:.4f}" for x in r2),
                             err=round(float(err), 4), fam_max=round(max(eta), 3) if eta else "",
                             fam_n50=int(sum(e > 0.5 for e in eta)) if eta else "", rank=rank,
                             n_rel=len(relidx)))
    # small per-dataset family summary
    types, counts = np.unique(ftype, return_counts=True)
    print(f"# N={N} rel={rel} rep={rep}: " + ", ".join(f"{t} {c}" for t, c in zip(types, counts)),
          file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--work", required=True)
    ap.add_argument("--Ns", default="2000,5000,20000")
    ap.add_argument("--rels", default="0.1,0.3")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ks", default="4,10,20")
    ap.add_argument("--methods", default="standard,detect-white,cswhite")
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--nref", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    jobs = [(N, rel, rep) for N in map(int, a.Ns.split(",")) for rel in map(float, a.rels.split(","))
            for rep in range(a.reps)]
    cols = ["N", "rel", "rep", "method", "k", "sec", "minR2", "r2", "err", "fam_max", "fam_n50", "rank", "n_rel"]
    with open(a.out, "w") as fo:
        fo.write("\t".join(cols) + "\n")
        with ThreadPoolExecutor(a.jobs) as ex:
            for rows in ex.map(lambda j: one_dataset(a, *j), jobs):
                for r in rows:
                    fo.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
                fo.flush()


if __name__ == "__main__":
    main()
