#!/usr/bin/env python3
"""--robust on synthetic biobank genotypes with realistic LD and ancestry
(HAPNEST, chromosomes 12-22; hapnest_prep.py), N of a few thousand.

Families are built from HAPNEST founders by transmission with recombination
(sim.Dropper, 1 cM/Mb): MZ twins, sibs, trios, nuclear families,
three-generation pedigrees and first cousins within an ancestry, and
relatives of different ancestry: a child of parents of two ancestries (with
both parents), half-sibs whose mothers differ in ancestry, a grandparent and
a grandchild of another ancestry, an aunt and a nephew of mixed ancestry.

Truth: PCA of a disjoint reference panel (600 per ancestry) with the sample
projected (as highn_test.py). Scores per method:
  minR2   min over the top K-1 true axes of R2 on the method's top K-1 PCs
          (unrelated individuals)
  err     relatives' placement error relative to the spread
  fam_n50 / fam_max   family axes among the PCs beyond K-1 (k = 20)
  recall  true pairs with pedigree kinship >= tau among the reported pairs
  false   reported pairs that are not related (pedigree kinship < 2^-4.5)
  and the mean final k0/k1/k2 per relationship type.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from highn_test import family_eta2  # noqa: E402
from sim import Dropper, read_bed, write_bed  # noqa: E402

TAU, TAU3 = 2 ** -3.5, 2 ** -4.5
EXPECT = {"MZ": (0.0, 0.0, 1.0), "parent-offspring": (0.0, 1.0, 0.0), "full sibs": (0.25, 0.5, 0.25),
          "2nd degree": (0.5, 0.5, 0.0)}


class Pedigree:
    """genotypes and pedigree kinship of the individuals made so far"""

    def __init__(self, G, anc, drop, rng, track=False):
        self.G, self.anc, self.drop, self.rng = G, anc, drop, rng
        self.haps, self.parents, self.kind, self.ancs = [], [], [], []
        self.K = {}  # (a, b) -> kinship for a < b, plus (a, a) -> self-kinship
        # track: founder-haplotype labels (2x, 2x + 1 for founder x) carried through the same
        # crossovers, for the realized IBD of any pair
        self.track, self.labs = track, []

    def kin(self, a, b):
        if a == b:
            return self.K.get((a, a), 0.5)
        return self.K.get((min(a, b), max(a, b)), 0.0)

    def founder(self, gi):
        x = len(self.haps)
        self.haps.append(self.drop.phase(self.G[gi]))
        if self.track:
            m = self.G.shape[1]
            self.labs.append(np.stack([np.full(m, 2 * x, np.int16), np.full(m, 2 * x + 1, np.int16)]))
        self.parents.append(None)
        self.kind.append("founder")
        self.ancs.append(self.anc[gi])
        self.K[(x, x)] = 0.5
        return x

    def child(self, f, m):
        x = len(self.haps)
        if self.track:
            hf, lf = self.drop.gamete(self.haps[f], self.labs[f])
            hm, lm = self.drop.gamete(self.haps[m], self.labs[m])
            self.haps.append(np.stack([hf, hm]))
            self.labs.append(np.stack([lf, lm]))
        else:
            self.haps.append(np.stack([self.drop.gamete(self.haps[f]), self.drop.gamete(self.haps[m])]))
        self.parents.append((f, m))
        self.kind.append("child")
        self.ancs.append(f"{self.ancs[f]}x{self.ancs[m]}" if self.ancs[f] != self.ancs[m] else self.ancs[f])
        for y in range(x):
            k = 0.5 * (self.kin(f, y) + self.kin(m, y))
            if k > 0:
                self.K[(y, x)] = k
        self.K[(x, x)] = 0.5 + 0.5 * self.kin(f, m)
        return x

    def copy(self, src):
        x = len(self.haps)
        self.haps.append(self.haps[src])
        if self.track:
            self.labs.append(self.labs[src])
        self.parents.append(("=", src))
        self.kind.append("mz")
        self.ancs.append(self.ancs[src])
        for y in range(x):
            if self.kin(src, y) > 0:
                self.K[(y, x)] = self.kin(src, y)
        self.K[(src, x)] = self.kin(src, src)
        self.K[(x, x)] = self.kin(src, src)
        return x


def simulate(G, anc, bim, n_target, rel_frac, seed, track=False):
    """track: also return the founder-haplotype labels (2 x M, int16) of every individual in the
    data, for the realized IBD of a pair (realized_ibd); the draws and the data are the same"""
    rng = np.random.default_rng(seed)
    drop = Dropper(bim, rng)
    P = Pedigree(G, anc, drop, rng, track)
    pool = {a: list(rng.permutation(np.where(anc == a)[0])) for a in np.unique(anc)}
    main = ["eur", "eas", "afr", "amr", "csa", "mid"]
    wts = np.array([0.25, 0.25, 0.25, 0.1, 0.1, 0.05])

    def take(a):
        if not pool[a]:
            raise ValueError(f"pool {a} exhausted")
        return P.founder(pool[a].pop())

    def rand_anc(excl=()):
        while True:
            a = main[rng.choice(len(main), p=wts)]
            if a not in excl:
                return a

    in_data, famid, ftype = [], [], []
    types = ["mz", "sibs", "trio", "nuclear", "pedigree", "cousins", "childx", "halfsibx", "grandx", "avuncx"]
    tw = np.array([0.04, 0.16, 0.14, 0.14, 0.08, 0.12, 0.08, 0.08, 0.08, 0.08])
    fam = 0
    while len(in_data) < rel_frac * n_target:
        t = types[rng.choice(len(types), p=tw)]
        a = rand_anc()
        fam += 1
        if t == "mz":
            x = take(a)
            m = [x, P.copy(x)]
        elif t == "sibs":
            f, mo = take(a), take(a)
            m = [P.child(f, mo), P.child(f, mo)]
        elif t == "trio":
            f, mo = take(a), take(a)
            m = [f, mo, P.child(f, mo)]
        elif t == "nuclear":
            f, mo = take(a), take(a)
            m = [f, mo, P.child(f, mo), P.child(f, mo)]
        elif t == "pedigree":
            f, mo = take(a), take(a)
            c1, c2 = P.child(f, mo), P.child(f, mo)
            sp = take(a)
            m = [f, c1, c2, P.child(c1, sp), P.child(c1, sp)]
        elif t == "cousins":
            f, mo = take(a), take(a)
            c1, c2 = P.child(f, mo), P.child(f, mo)
            m = [P.child(c1, take(a)), P.child(c2, take(a))]
        elif t == "childx":
            b = rand_anc((a,))
            f, mo = take(a), take(b)
            m = [f, mo, P.child(f, mo)]
        elif t == "halfsibx":
            b = rand_anc((a,))
            c = rand_anc((a, b))
            f = take(a)
            m = [P.child(f, take(b)), P.child(f, take(c))]
        elif t == "grandx":
            b = rand_anc((a,))
            gp = take(a)
            c1 = P.child(gp, take(b))
            m = [gp, P.child(c1, take(b))]
        else:  # avuncx
            b = rand_anc((a,))
            f, mo = take(a), take(a)
            aunt, sib = P.child(f, mo), P.child(f, mo)
            m = [aunt, P.child(sib, take(b))]
        in_data += m
        famid += [fam] * len(m)
        ftype += [t] * len(m)
    while len(in_data) < n_target:
        in_data.append(take(rand_anc()))
        famid.append(0)
        ftype.append("unrelated")
    Gs = np.array([P.haps[x].sum(0) for x in in_data], dtype=np.int8)
    ix = {x: i for i, x in enumerate(in_data)}
    truth = {}
    for (a, b), k in P.K.items():
        if a != b and a in ix and b in ix and k > 0:
            i, j = sorted((ix[a], ix[b]))
            truth[(i, j)] = k
    ancs = [P.ancs[x] for x in in_data]
    if track:
        return Gs, np.array(famid), np.array(ftype), truth, ancs, [P.labs[x] for x in in_data]
    return Gs, np.array(famid), np.array(ftype), truth, ancs


def realized_ibd(li, lj, keep=None):
    """realized (k0, k1, k2) and kinship k1/4 + k2/2 of a pair from their founder-haplotype labels
    (2 x M each): at each SNP the number of i's haplotypes IBD with j's (0, 1 or 2; no inbreeding)"""
    if keep is not None:
        li, lj = li[:, keep], lj[:, keep]
    n = ((li[0] == lj[0]) | (li[0] == lj[1])).astype(np.int8) + ((li[1] == lj[0]) | (li[1] == lj[1]))
    k = np.bincount(n, minlength=3)[:3] / n.size
    return k[0], k[1], k[2], k[1] / 4 + k[2] / 2


def relation(i, j, k, ftype):
    if k > 0.354:
        return "MZ"
    if k > 0.177:
        return "1st degree"
    if k > 0.0884:
        return "2nd degree"
    return "3rd degree"


def reference_truth(R, G, kk):
    f = R.mean(0) / 2
    keep = (f > 0.01) & (f < 0.99)
    sd = np.sqrt(2 * f[keep] * (1 - f[keep]))
    Xr = (R[:, keep] - 2 * f[keep]) / sd
    C = Xr @ Xr.T / keep.sum()
    w, v = np.linalg.eigh(C)
    w, v = w[::-1], v[:, ::-1]
    T = ((G[:, keep] - 2 * f[keep]) / sd) @ (Xr.T @ v[:, :kk])
    return T / np.linalg.norm(T, axis=0), w[:12]


def score(U, T, base, rel, kk):
    U = U[:, :kk]
    Z = np.c_[U, np.ones(len(U))]
    B, *_ = np.linalg.lstsq(Z[base], T[base, :kk], rcond=None)
    P = Z @ B
    Tb = T[base, :kk]
    r2 = 1 - ((Tb - P[base]) ** 2).sum(0) / ((Tb - Tb.mean(0)) ** 2).sum(0)
    spread = np.sqrt(((Tb - Tb.mean(0)) ** 2).sum(1).mean())
    err = np.sqrt(((T[rel, :kk] - P[rel]) ** 2).sum(1).mean()) / spread if len(rel) else np.nan
    return r2.min(), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="hapnest_prep.py output prefix (hapnest_pool)")
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--rel", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=6, help="true number of ancestry axes + 1 (scored axes: K-1); "
                    "HAPNEST chromosomes 12-22 has 5 axes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="", help="comma-separated subset of the methods (default: all)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    fam, bim, G = read_bed(a.pool)
    G = G.astype(np.int8)
    info = {l.split()[0]: l.split()[1:] for l in list(open(os.path.join(os.path.dirname(a.pool), "individuals.tsv")))[1:]}
    ids = [r[1] for r in fam]
    anc = np.array([info[x][0] for x in ids])
    sset = np.array([info[x][1] for x in ids])
    pool_idx, ref_idx = np.where(sset == "pool")[0], np.where(sset == "ref")[0]
    t0 = time.time()
    Gs, famid, ftype, truth, ancs = simulate(G[pool_idx], anc[pool_idx], bim, a.N, a.rel, a.seed)
    keep = Gs.min(0) != Gs.max(0)
    Gs = Gs[:, keep]
    bim_k = [bim[i] for i in np.where(keep)[0]]
    R = G[ref_idx][:, keep].astype(float)
    T, wref = reference_truth(R, Gs.astype(float), a.K - 1)
    pre = os.path.join(a.work, f"hap_N{a.N}_s{a.seed}")
    write_bed(pre, [[str(f) if f else f"U{i}", f"I{i}", "0", "0", "0", "-9"] for i, f in enumerate(famid)], bim_k, Gs)
    print(f"# N={a.N} M={Gs.shape[1]} in families {int((famid > 0).sum())} true pairs (kin >= tau) "
          f"{sum(k >= TAU for k in truth.values())}; reference eigenvalues {np.round(wref[:8], 3)}; "
          f"simulated in {time.time() - t0:.0f} s", flush=True)
    base = np.where(famid == 0)[0]
    rel = np.where(famid > 0)[0]
    kk = a.K - 1
    # dwg-sketch: the very-large-N path (KING and the evalAdmix screen on each person's sketch
    # neighbours) forced at this N
    methods = {"standard": "", "detect-white": "--robust detect-white", "dwg": "--robust",
               "dwg-sketch": "--robust --king-search sketch --ea-search sketch",
               "dwg-dense": "--robust dwg --robust-engine dense"}
    if a.methods:
        methods = {m: methods[m] for m in a.methods.split(",")}
    rows = []
    for name, extra in methods.items():
        for k in (kk, 20):
            out = f"{pre}_{name}_k{k}"
            t0 = time.time()
            r = subprocess.run(f"{a.pcaone} -b {pre} -k {k} -n {a.threads} {extra} -o {out}", shell=True,
                               capture_output=True, text=True)
            sec = time.time() - t0
            if r.returncode:
                print(r.stdout[-1500:], r.stderr[-1500:])
                continue
            U = np.loadtxt(out + ".eigvecs", ndmin=2)
            mr2, err = score(U, T, base, rel, kk)
            row = dict(method=name, k=k, sec=round(sec, 1), minR2=round(mr2, 4), err=round(err, 3))
            if k > kk:
                eta = [family_eta2(U[:, j], famid) for j in range(kk, U.shape[1])]
                row.update(fam_max=round(max(eta), 3), fam_n50=int(sum(e > 0.5 for e in eta)))
            if name != "standard" and os.path.exists(out + ".relpairs"):
                rp = [l.split() for l in list(open(out + ".relpairs"))[1:]]
                det = {}
                for t in rp:
                    i, j = sorted((int(t[0][1:]), int(t[1][1:])))
                    det[(i, j)] = list(map(float, t[2:]))
                tp = {p for p, kv in truth.items() if kv >= TAU}
                row.update(pairs=len(det), recall=round(len(tp & set(det)) / max(len(tp), 1), 4),
                           false=sum(1 for p in det if truth.get(p, 0) < TAU3))
                # recall by family type for the cross-ancestry relatives
                for t in ("childx", "halfsibx", "grandx", "avuncx"):
                    tt = {p for p in tp if ftype[p[0]] == t}
                    row[f"rec_{t}"] = round(len(tt & set(det)) / len(tt), 3) if tt else ""
                if k == kk:
                    by = {}
                    for p, v in det.items():
                        kv = truth.get(p, 0)
                        if kv < TAU:
                            continue
                        lab = relation(p[0], p[1], kv, ftype)
                        if lab == "1st degree":
                            lab = "parent-offspring" if v[3] < 0.1 else "full sibs"
                        by.setdefault(lab, []).append(v[2:6])
                    row["ibd"] = {lab: np.round(np.mean(vs, 0), 3).tolist() + [len(vs)] for lab, vs in by.items()}
            rows.append(row)
            print(row, flush=True)
    import json
    with open(a.out, "w") as f:
        json.dump(dict(N=a.N, M=int(Gs.shape[1]), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
