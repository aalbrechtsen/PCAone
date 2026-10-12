#!/usr/bin/env python3
"""PCP-pi benchmark (report/inbreeding/PLAN_pcp_pi.md, steps 3-4): inbreeding F
and relatedness (kinship, k0/k1/k2) from individual allele frequencies pi of
several estimators, against the realised truth of a simulation with linkage.

Simulation: admixTjeck2 founders (n per population of CEU, CHB, MXL, YRI;
random phase), recombination at 1 cM/Mb (inbreed_true_hap.gamete), founder
haplotype labels give the realised F and IBD of everyone. Relatives (14):
  CEU  2 founders -> 3 full sibs; child of two of the sibs (F 1/4)
  CHB  MZ copy of a founder; half sibs (one shared parent); their child (F 1/8)
  MXL  2 founders -> 2 sibs; each sib x a founder -> a child (first cousins;
       avuncular with the other sib); child of the cousins (F 1/16)
  CEU x YRI founder -> a child
Scenarios:
  rel    as above
  null   founders only
  small  rel, YRI with only 3 founders (a small drifted population)
  het    rel, plus 20% of a CHB founder's homozygous sites made heterozygous
Giant families (step 5; no other relatives):
  giant<T>     one multi-generation CEU family of T (2 founders, their children,
               spouses from CEU, grandchildren, ...) besides the n unrelated CEU
  giantinb<T>  the same, children of the second generation on mate with a
               cousin when one is free (inbred family)
  giantpop<T>  CEU is only the family (no unrelated CEU)

Methods (pi -> F of everyone; kinship and IBD of the evaluated pairs):
  pca<k>     top-k PCs of PCAone aarobust-kin+diag (-k 10), intercept; F: plain;
             R: both families left out (as PCAone for a called pair), k = 3, 6, 10;
             pca<k>-own: each without its own family
  pcp-*      PCP smoother, candidates = the screen; leave-out none/self/pair/family/own
             (own: pi_i without i's family, pi_j without j's)
  pcpT-*     PCP with the true pairs as candidates
  dwg-*      dwg fit smoother, same candidates
  PCAone aad / dwg   PCAone's own reported kinship and IBD (called pairs only; aad -k 10,
             dwg -k 3: its fitted rank is limited to k + 1)
  KING       KING-robust kinship (R only)
Screen (candidates): pairs called by PCAone aarobust-kin+diag or dwg, or with
KING > 2^-4.5.
Evaluated pairs: all true pairs (realised kinship > 1/64), all screen pairs, and
100 random unrelated pairs within populations.

PCs: standard PCAone, aarobust-kin+diag and dwg (top 3) against the reference
panel truth (benchmark.truth_reference): min R^2, and each group's error in
SDs of its own population's founders.

usage: pcp_pi_bench.py <workdir> <outprefix> [workers] [scen,...] [n,...] [reps]
writes <outprefix>_F.tsv, <outprefix>_R.tsv and <outprefix>_P.tsv
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import inbreed_smooth_sim as SIM  # noqa: E402
import pcp_pi as Q  # noqa: E402
from inbreed_true_hap import gamete  # noqa: E402
from inbreeding_bench import PCAONE  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

KS = (3, 6, 10)
LEAVE = ("none", "self", "pair", "family", "own")


def build(seed, n, scen):
    """haplotypes N x 2 x M, labels, pops, group, relationship names"""
    if not SIM._DATA:
        fam, _, G = read_bed(SIM.BFILE)
        SIM._DATA.update(G=G, lab=np.array([r[1] for r in fam]))
    G, lab = SIM._DATA["G"], SIM._DATA["lab"]
    rng = np.random.default_rng(seed)
    giant = scen.startswith("giant")
    T = int("".join(c for c in scen if c.isdigit())) if giant else 0
    nn = {p: (3 if (scen == "small" and p == "YRI") else n) for p in SIM.POPS}
    if scen.startswith("giantpop"):
        nn["CEU"] = 0
    pool = {p: rng.permutation(np.flatnonzero(lab == p)) for p in SIM.POPS}
    idx = np.concatenate([pool[p][:nn[p]] for p in SIM.POPS])
    spare = list(pool["CEU"][nn["CEU"]:])  # CEU founders for the family
    X = G[idx].astype(np.int8)
    N0, M = X.shape
    h1 = np.where(X == 1, rng.integers(0, 2, X.shape), X // 2)
    hap = [np.stack([a, g - a]).astype(np.int8) for a, g in zip(h1, X)]
    labs = [np.stack([np.full(M, 2 * i), np.full(M, 2 * i + 1)]).astype(np.int32) for i in range(N0)]
    pops = list(lab[idx])
    group = ["founder"] * N0
    rows = list(idx)

    def child(a, b, pop, g):
        ga, la = gamete(hap[a], labs[a], rng)
        gb, lb = gamete(hap[b], labs[b], rng)
        hap.append(np.stack([ga, gb]))
        labs.append(np.stack([la, lb]))
        pops.append(pop)
        group.append(g)
        rows.append(-1)
        return len(hap) - 1

    def founder(r):
        """a new CEU founder (row r of the panel) in the family"""
        g = G[r].astype(np.int8)
        a = np.where(g == 1, rng.integers(0, 2, M), g // 2)
        hap.append(np.stack([a, g - a]).astype(np.int8))
        nl = 2 * (len(labs) + 10 ** 6)
        labs.append(np.stack([np.full(M, nl), np.full(M, nl + 1)]).astype(np.int32))
        pops.append("CEU")
        group.append("family")
        rows.append(r)
        return len(hap) - 1

    if giant:
        a, b = founder(spare.pop()), founder(spare.pop())
        gen = [child(a, b, "CEU", "family") for _ in range(min(4, T - 2))]
        par = {g: (a, b) for g in gen}
        while len(hap) - N0 < T:
            nxt = []
            free = list(gen)
            rng.shuffle(free)
            while free and len(hap) - N0 < T:
                x = free.pop()
                mate = None
                if scen.startswith("giantinb"):
                    cous = [y for y in free if set(par.get(y, ())) != set(par.get(x, ()))]
                    if cous:
                        mate = cous[0]
                        free.remove(mate)
                if mate is None:
                    if len(hap) - N0 >= T - 1:
                        break
                    mate = founder(spare.pop())
                for _ in range(2):
                    if len(hap) - N0 >= T:
                        break
                    c = child(x, mate, "CEU", "family")
                    par[c] = (x, mate)
                    nxt.append(c)
            if not nxt:
                break
            gen = nxt
    elif scen != "null":
        def f(p, k):
            return [i for i in range(N0) if pops[i] == p][k]

        a, b = f("CEU", 0), f("CEU", 1)
        s = [child(a, b, "CEU", "relative") for _ in range(3)]
        child(s[0], s[1], "CEU", "inbred")
        c = [f("CHB", k) for k in range(4)]
        hap.append(hap[c[0]].copy())
        labs.append(labs[c[0]].copy())
        pops.append("CHB")
        group.append("relative")
        h1_ = child(c[1], c[2], "CHB", "relative")
        h2_ = child(c[1], c[3], "CHB", "relative")
        child(h1_, h2_, "CHB", "inbred")
        m = [f("MXL", k) for k in range(4)]
        k1 = child(m[0], m[1], "MXL", "relative")
        k2 = child(m[0], m[1], "MXL", "relative")
        g1 = child(k1, m[2], "MXL", "relative")
        g2 = child(k2, m[3], "MXL", "relative")
        child(g1, g2, "MXL", "inbred")
        child(f("CEU", 2), f("YRI", 0), "CEU", "relative")
    clean = np.stack(hap).sum(1).astype(np.int8)
    if scen == "het":
        i = [k for k in range(N0) if pops[k] == "CHB"][4 % n]
        h = hap[i]
        hit = (h[0] == h[1]) & (rng.random(M) < 0.2)
        h[1, hit] = 1 - h[1, hit]
        group[i] = "excess het"
    H2, L2 = np.stack(hap), np.stack(labs)
    Xs = H2.sum(1)
    keep = Xs.min(0) != Xs.max(0)
    ex = dict(rows=np.array(rows), cols=np.flatnonzero(keep), clean=clean[:, keep])
    return H2[:, :, keep], L2[:, :, keep], np.array(pops), np.array(group), ex


def truth(L2):
    """realised F (N), kinship (N x N) and k0/k1/k2 (N x N x 3) from the labels"""
    N = len(L2)
    F = (L2[:, 0] == L2[:, 1]).mean(1)
    K = np.zeros((N, N))
    IBD = np.zeros((N, N, 3))
    for i in range(N):
        a0, a1 = L2[i, 0], L2[i, 1]
        for j in range(i + 1, N):
            b0, b1 = L2[j, 0], L2[j, 1]
            m00, m01, m10, m11 = a0 == b0, a0 == b1, a1 == b0, a1 == b1
            K[i, j] = K[j, i] = (m00.mean() + m01.mean() + m10.mean() + m11.mean()) / 4
            two = (m00 & m11) | (m01 & m10)
            one = (m00 | m01 | m10 | m11) & ~two
            IBD[i, j] = IBD[j, i] = [1 - two.mean() - one.mean(), one.mean(), two.mean()]
    return F, K, IBD


def read_pairs(path, ids):
    if not os.path.exists(path):
        return {}
    rows = list(open(path))
    h = rows[0].split()
    out = {}
    for l in rows[1:]:
        x = l.split()
        i, j = sorted((ids[x[0]], ids[x[1]]))
        out[(i, j)] = [float(x[h.index(c)]) for c in ("KINSHIP", "K0", "K1", "K2")]
    return out


def leave_set(how, i, j, comp):
    if how == "none":
        return ([], [])
    if how == "self":
        return ([i], [j])
    if how == "pair":
        return ([i, j], [i, j])
    if how == "own":
        # each without its own family: a called pair (one family) as "family",
        # an uncalled pair each without its own relatives
        return (list(np.flatnonzero(comp == comp[i])), list(np.flatnonzero(comp == comp[j])))
    # both families: for a pair no method called, i and j are in different ones
    fam = list(np.flatnonzero((comp == comp[i]) | (comp == comp[j])))
    return (fam, fam)


def lin_pair(Gh, cs, Lt, C, i, j, how, comp):
    oa, ob = leave_set(how, i, j, comp)
    if oa == ob:
        P = Q.pi_lin(Gh, Lt, C, [i, j], out=oa, colsum=cs)
        return P[0], P[1]
    return (Q.pi_lin(Gh, Lt, C, [i], out=oa, colsum=cs)[0], Q.pi_lin(Gh, Lt, C, [j], out=ob, colsum=cs)[0])


def lin_F(G, Lt, C, loo):
    """F of everyone from pi = f + W (g/2 - f); loo: own genotype left out"""
    W = Lt @ np.linalg.inv(C)
    f = G.mean(0) / 2
    x = G / 2.0 - f
    P = W @ x
    if loo:
        h = np.diag(W)[:, None]
        P = (P - h * x) / (1 - h)
    return Q.inbreeding_from_pi(G, f + P)


def one(job):
    scen, n, seed, workdir = job
    t0 = time.time()
    H2, L2, pops, group, ex = build(seed, n, scen)
    G = H2.sum(1).astype(np.int8)
    N = len(G)
    Ft, Kt, IBDt = truth(L2)
    d = os.path.join(workdir, f"pp_{scen}_n{n}_s{seed}")
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, "g")
    ids = [f"I{i}" for i in range(N)]
    write_bed(pre, [[pops[i], ids[i], "0", "0", "0", "-9"] for i in range(N)],
              [f"1\ts{j}\t0\t{j + 1}\tA\tC" for j in range(G.shape[1])], G)
    for name, extra in (("std", []), ("aad", ["--robust", "aarobust-kin", "--impute-diag"]),
                        ("dwg", ["--robust", "dwg"])):
        subprocess.run([PCAONE, "-b", pre, "-k", "3" if name == "dwg" else "10", "-n", "1", "-o", os.path.join(d, name)] + extra,
                       capture_output=True, text=True)
    idmap = {x: i for i, x in enumerate(ids)}
    pc = {m: read_pairs(os.path.join(d, m + ".relpairs"), idmap) for m in ("aad", "dwg")}
    U = np.loadtxt(os.path.join(d, "aad.eigvecs"))
    Us = {m: np.loadtxt(os.path.join(d, m + ".eigvecs")) for m in ("std", "aad", "dwg")}
    shutil.rmtree(d)
    # PCs against the reference-panel truth
    Gall, lab = SIM._DATA["G"], SIM._DATA["lab"]
    used = ex["rows"][ex["rows"] >= 0]
    ref = np.setdiff1d(np.flatnonzero(np.isin(lab, SIM.POPS)), used)
    Tr = B.truth_reference(dict(G=ex["clean"].astype(float), ref_idx=ref, cols=ex["cols"]), Gall, 3)
    Prows = []
    for m, Um in Us.items():
        r2, _ = B.score(Um, Tr, N, [], 3)
        Z = np.c_[Um[:, :3], np.ones(N)]
        e = np.sqrt(((Z @ np.linalg.lstsq(Z, Tr, rcond=None)[0] - Tr) ** 2).sum(1))
        z = np.zeros(N)
        for p in np.unique(pops):
            k = np.flatnonzero(pops == p)
            base = [i for i in k if group[i] == "founder"] or list(k)
            z[k] = e[k] / np.sqrt(((Tr[base] - Tr[base].mean(0)) ** 2).sum(1).mean())
        Prows.append(dict(scen=scen, n=n, seed=seed, method=m, minR2=r2.min(), pairs=len(pc.get(m, {})),
                          **{f"err_{x.replace(' ', '_')}": float(np.sqrt((z[group == x] ** 2).mean()))
                             for x in np.unique(group)}))
    Kk = B.king_robust(G)
    iu = np.triu_indices(N, 1)
    screen = set(pc["aad"]) | set(pc["dwg"]) | {(int(i), int(j)) for i, j in zip(*iu) if Kk[i, j] > B.TAU3}
    tp = {(int(i), int(j)) for i, j in zip(*iu) if Kt[i, j] > 1 / 64}
    rng = np.random.default_rng(seed + 1)
    unrel = [(int(i), int(j)) for i, j in zip(*iu) if pops[i] == pops[j] and Kt[i, j] == 0 and
             group[i] == group[j] == "founder"]
    unrel = [unrel[k] for k in rng.choice(len(unrel), min(100, len(unrel)), replace=False)]
    ev = sorted(tp | screen | set(unrel))

    def cand_of(P):
        c = np.zeros((N, N), bool)
        for i, j in P:
            c[i, j] = c[j, i] = True
        return c

    Gf = G.astype(float)
    Lp, Cp, _ = Q.pcp_parts(Gf, cand=cand_of(screen))
    LpT, CpT, _ = Q.pcp_parts(Gf, cand=cand_of(tp))
    Ld, Cd, _ = Q.dwg_parts(Gf, cand=cand_of(screen))
    comp_aad = Q.families(list(pc["aad"]), N)
    comp_dwg = Q.families(list(pc["dwg"]), N)
    comp_T = Q.families(list(tp), N)

    # ---- F
    Fr = {}
    Gh = G / 2.0
    cs = Gh.sum(0)
    for k in KS:
        V = np.hstack([U[:, :k], np.ones((N, 1))])
        H = V @ np.linalg.pinv(V)
        P = H @ Gh
        Fr[f"pca{k}"] = Q.inbreeding_from_pi(G, P)
        h = np.diag(H)[:, None]
        Fr[f"pca{k}-self"] = Q.inbreeding_from_pi(G, (P - h * Gh) / (1 - h))
    for nm, (Lt, C) in (("pcp", (Lp, Cp)), ("pcpT", (LpT, CpT)), ("dwg", (Ld, Cd))):
        Fr[f"{nm}-none"] = lin_F(G, Lt, C, False)
        Fr[f"{nm}-self"] = lin_F(G, Lt, C, True)
    Frows = [dict(scen=scen, n=n, seed=seed, ind=i, pop=pops[i], group=group[i], F_true=Ft[i],
                  **{m: v[i] for m, v in Fr.items()}) for i in range(N)]

    # ---- R
    Rrows = []
    for i, j in ev:
        kind = "true" if (i, j) in tp else ("unrel" if Kt[i, j] == 0 else "distant")
        row = dict(scen=scen, n=n, seed=seed, i=i, j=j, pop=pops[i] if pops[i] == pops[j] else "cross",
                   kind=kind, kin_true=Kt[i, j], k0_true=IBDt[i, j, 0], k1_true=IBDt[i, j, 1],
                   k2_true=IBDt[i, j, 2], screen=(i, j) in screen, KING=Kk[i, j])
        for k in KS:
            out = np.flatnonzero((comp_aad == comp_aad[i]) | (comp_aad == comp_aad[j]))
            P = Q.pi_pca(Gh, U[:, :k], [i, j], out=out)
            kin, kk = Q.pair_rel(G, i, j, P[0], P[1])
            row[f"pca{k}"], row[f"pca{k}_k0"], row[f"pca{k}_k2"] = kin, kk[0], kk[2]
            pa = Q.pi_pca(Gh, U[:, :k], [i], out=np.flatnonzero(comp_aad == comp_aad[i]))[0]
            pb = Q.pi_pca(Gh, U[:, :k], [j], out=np.flatnonzero(comp_aad == comp_aad[j]))[0]
            kin, kk = Q.pair_rel(G, i, j, pa, pb)
            row[f"pca{k}-own"], row[f"pca{k}-own_k0"], row[f"pca{k}-own_k2"] = kin, kk[0], kk[2]
        for nm, (Lt, C), comp in (("pcp", (Lp, Cp), comp_aad), ("pcpT", (LpT, CpT), comp_T),
                                  ("dwg", (Ld, Cd), comp_dwg)):
            for how in LEAVE:
                pa, pb = lin_pair(Gh, cs, Lt, C, i, j, how, comp)
                kin, kk = Q.pair_rel(G, i, j, pa, pb)
                m = f"{nm}-{how}"
                row[m], row[m + "_k0"], row[m + "_k2"] = kin, kk[0], kk[2]
        for m in ("aad", "dwg"):
            v = pc[m].get((i, j))
            row[f"PCAone-{m}"] = v[0] if v else np.nan
            row[f"PCAone-{m}_k0"] = v[1] if v else np.nan
            row[f"PCAone-{m}_k2"] = v[3] if v else np.nan
        Rrows.append(row)
    print(f"{scen} n={n} seed={seed}: N={N}, {len(ev)} pairs, {time.time() - t0:.0f}s", flush=True)
    return Frows, Rrows, Prows


def main():
    workdir, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    scen = sys.argv[4].split(",") if len(sys.argv) > 4 else ["rel", "null", "small", "het"]
    ns = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [5, 10, 20, 50]
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    jobs = [(s, n, seed, workdir) for s in scen for n in ns for seed in range(100, 100 + reps)]
    import pandas as pd
    F, R, P = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for a, b, c in ex.map(one, jobs):
            F += a
            R += b
            P += c
    pd.DataFrame(F).to_csv(out + "_F.tsv", sep="\t", index=False)
    pd.DataFrame(R).to_csv(out + "_R.tsv", sep="\t", index=False)
    pd.DataFrame(P).to_csv(out + "_P.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
