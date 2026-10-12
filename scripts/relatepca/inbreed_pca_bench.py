#!/usr/bin/env python3
"""PCA accuracy when relatives and inbred (or excess-heterozygous) individuals
are in the same data, alone or together (inbreed_smooth_sim.build scenarios
rel, inb, both, relinb), n = 5 and 10 per population, 10 replicates each.

Methods: PCAone -k 3 standard, --robust aarobust-kin, --impute-diag
(aarobust-diag), --robust aarobust-kin --impute-diag, --robust dwg, the
top 3 eigenvectors of the structure part L of the Python PCP fit that the
inbreeding smoother uses (pcp_L), and aarobust-kin+refit: the pairs called by
aarobust-kin --impute-diag, then a rank-limited fit of the GRM with the diagonal
and these pairs left out (dw_grm.fit, rank from the noise edge, at most k + 1,
as dwg), PCs = top k eigenvectors of its centred structure part, not whitened.
Truth: reference panel (admixTjeck2 individuals not in the data), the
unchanged genotypes of the data projected (benchmark.truth_reference).
Scores: min R2 (benchmark.score), MEV, each group's error in SDs of its own
population (relatives, inbred, excess het, others; RMS over the group), true
related pairs found (of 10; relinb 15) and pairs called; and F of the inbred and the
relatives from the PCP smoother and from 3 PCs.

usage: inbreed_pca_bench.py <workdir> <out.tsv> [workers] [scenarios, comma-separated]
"""
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as DW  # noqa: E402
import inbreed_smooth as IS  # noqa: E402
import inbreed_smooth_sim as SIM  # noqa: E402
from inbreeding_bench import METHODS, PCAONE, unit  # noqa: E402
from sim import write_bed  # noqa: E402

K = 3
SCEN = ["rel", "inb", "both", "relinb"]
GROUPS = ["relative", "inbred", "excess het", "other"]


def true_pairs(group, pops, n0):
    """the simulated pairs: parents (first two CEU), sibs (n0..n0+2), CHB
    source and its MZ copy (n0+3); relinb: also the child of sibs n0, n0+1
    (n0+4) with its parents, its aunt/uncle and its grandparents (15 pairs)"""
    ceu = [i for i, p in enumerate(pops[:n0]) if p == "CEU"]
    a, b = ceu[0], ceu[1]
    sibs = [n0, n0 + 1, n0 + 2]
    P = {(min(x, y), max(x, y)) for x in (a, b) for y in sibs}
    P |= {(sibs[i], sibs[j]) for i in range(3) for j in range(i + 1, 3)}
    src = list(pops).index("CHB")
    P.add((src, n0 + 3))
    if len(pops) > n0 + 4:
        c = n0 + 4
        P |= {(x, c) for x in (a, b) + tuple(sibs)}
    return P


def scores(U, T, pops, group):
    r2, _ = B.score(U, T, len(T), [], K)
    Qu, Qt = np.linalg.qr(unit(U))[0], np.linalg.qr(unit(T))[0]
    mev = np.mean([np.linalg.norm(Qu.T @ Qt[:, j]) for j in range(K)])
    Z = np.c_[U, np.ones(len(U))]
    P = Z @ np.linalg.lstsq(Z, T, rcond=None)[0]
    e = np.sqrt(((P - T) ** 2).sum(1))
    # each individual's error in SDs of its own population (spread of the
    # population's unaffected individuals in the truth)
    z = np.zeros(len(e))
    for p in np.unique(pops):
        k = np.flatnonzero(pops == p)
        ref = [i for i in k if group[i] == "other"]
        spread = np.sqrt(((T[ref] - T[ref].mean(0)) ** 2).sum(1).mean())
        z[k] = e[k] / spread
    g = {x: (np.sqrt((z[group == x] ** 2).mean()) if (group == x).any() else np.nan) for x in GROUPS}
    return r2.min(), mev, g


def one(job):
    scen, n, seed, workdir = job
    X, pops, group, ex = SIM.build(seed, n, scen, full=True)
    Gall, lab = SIM._DATA["G"], SIM._DATA["lab"]
    used = ex["idx"][ex["idx"] >= 0]
    ref = np.setdiff1d(np.flatnonzero(np.isin(lab, SIM.POPS)), used)
    T = B.truth_reference(dict(G=ex["clean"].astype(float), ref_idx=ref, cols=ex["cols"]), Gall, K)
    n0 = 4 * n
    tp = true_pairs(group, pops, n0) if scen not in ("inb", "none") else set()
    d = os.path.join(workdir, f"{scen}_n{n}_s{seed}")
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, "g")
    fam = [[pops[i], f"I{i}", "0", "0", "0", "-9"] for i in range(len(X))]  # IID: used in .relpairs
    bim = [f"1\ts{j}\t0\t{j + 1}\tA\tC" for j in range(X.shape[1])]
    write_bed(pre, fam, bim, X)
    rows = []

    def add(m, U, called):
        r2, mev, g = scores(U, T, pops, group)
        found = len(called & tp)
        rows.append(dict(scen=scen, n=n, seed=seed, method=m, minR2=r2, mev=mev, pairs=len(called),
                         true_found=found, **{f"err_{x.replace(' ', '_')}": g[x] for x in GROUPS}))

    called_by = {}
    for m, extra in METHODS.items():
        out = os.path.join(d, m)
        r = subprocess.run([PCAONE, "-b", pre, "-k", str(K), "-n", "1", "-o", out] + extra,
                           capture_output=True, text=True)
        if r.returncode != 0:
            continue
        called = set()
        if os.path.exists(out + ".relpairs"):
            for l in list(open(out + ".relpairs"))[1:]:
                x, y = (int(w[1:]) for w in l.split()[:2])
                called.add((min(x, y), max(x, y)))
        called_by[m] = called
        add(m, np.loadtxt(out + ".eigvecs")[:, :K], called)
    # refit: aarobust's pairs, rank-limited structure fit (as dwg), not whitened
    A, f, w = IS.scaled_uncentred(X)
    N = len(A)
    cand = np.zeros((N, N), bool)
    for x, y in called_by.get("aarobust-kin+diag", set()):
        cand[x, y] = cand[y, x] = True
    Lr, _, rr, pr = DW.fit(A, B.TAU, cand, DW.noise_edge(f, w, N), K + 1, None)
    Pc = np.eye(N) - 1 / N
    add("aarobust-kin+refit", np.linalg.eigh(Pc @ Lr @ Pc)[1][:, ::-1][:, :K],
        {(min(i, j), max(i, j)) for i, j, _ in pr})
    # the PCP fit of the smoother: its structure part and its pairs
    W, info = IS.smoother(X, "pcp")
    L, S = info["parts"]["L"], info["parts"]["S"]
    called = {(i, j) for i, j in zip(*np.nonzero(np.triu(np.abs(S) > 1e-12, 1)))}
    add("pcp_L", np.linalg.eigh(L)[1][:, ::-1][:, :K], called)
    # F of the inbred individual and the relatives: smoother and 3 PCs
    F_s = IS.inbreeding(X, W)[0]
    Wk, _ = IS.smoother(X, "k3")
    F_k = IS.inbreeding(X, Wk)[0]
    for r_ in rows:
        r_.update(F_inb_smooth=F_s[group == "inbred"].mean() if (group == "inbred").any() else np.nan,
                  F_inb_k3=F_k[group == "inbred"].mean() if (group == "inbred").any() else np.nan,
                  F_rel_smooth=F_s[group == "relative"].mean() if (group == "relative").any() else np.nan,
                  F_rel_k3=F_k[group == "relative"].mean() if (group == "relative").any() else np.nan)
    shutil.rmtree(d)
    return rows


def main():
    workdir, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    scen = sys.argv[4].split(",") if len(sys.argv) > 4 else SCEN
    jobs = [(s, n, seed, workdir) for s in scen for n in (5, 10) for seed in range(100, 110)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(out, sep="\t", index=False)
    for v in ("minR2", "mev", "err_relative", "err_inbred", "err_excess_het", "err_other", "true_found", "pairs"):
        print("==", v)
        print(d.pivot_table(index=["scen", "method"], columns="n", values=v).round(3).to_string())


if __name__ == "__main__":
    main()
