#!/usr/bin/env python3
"""Does centring hurt aarobust? PCP with the kinship rule and the diagonal
unobserved (illustrate.pcp_kin) on the centred standardised GRM (as
aarobust-kin) and on the SNP-scaled uncentred GRM (as dwg; the mean is then
part of the structure, and the PCs are the top k eigenvectors of the centred
structure part), with every pair, KING-robust > 0.04 (as PCAone) or the true
pairs as candidates. Same
simulated designs, truth and scores as inbreed_pca_bench.py.

usage: inbreed_pca_centre.py <out.tsv> [workers]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
import inbreed_smooth as IS  # noqa: E402
import inbreed_smooth_sim as SIM  # noqa: E402
from inbreed_pca_bench import K, scores, true_pairs  # noqa: E402

SCEN = ["none", "inb", "rel", "both", "rel_sib4", "relinb"]


def one(job):
    scen, n, seed = job
    X, pops, group, ex = SIM.build(seed, n, scen, full=True)
    Gall, lab = SIM._DATA["G"], SIM._DATA["lab"]
    ref = np.setdiff1d(np.flatnonzero(np.isin(lab, SIM.POPS)), ex["idx"][ex["idx"] >= 0])
    T = B.truth_reference(dict(G=ex["clean"].astype(float), ref_idx=ref, cols=ex["cols"]), Gall, K)
    N = len(X)
    tp = true_pairs(group, pops, 4 * n) if scen not in ("inb", "none") else set()
    tcand = np.zeros((N, N), bool)
    for i, j in tp:
        tcand[i, j] = tcand[j, i] = True
    P = np.eye(N) - 1 / N
    C = IS.centred_grm(X)
    A, _, _ = IS.scaled_uncentred(X)
    rows = []
    kcand = (B.king_robust(X) > 0.04)
    for cname, cand in (("all pairs", None), ("KING", kcand), ("true pairs", tcand)):
        for gname, M, centre_after in (("centred", C, False), ("uncentred", A, True)):
            L, S, _ = I.pcp_kin(M, B.TAU, free_diag=True, cand=cand)
            Lc = P @ L @ P if centre_after else L
            U = np.linalg.eigh(Lc)[1][:, ::-1][:, :K]
            called = {(i, j) for i, j in zip(*np.nonzero(np.triu(np.abs(S) > 1e-12, 1)))}
            r2, mev, g = scores(U, T, pops, group)
            rows.append(dict(scen=scen, n=n, seed=seed, grm=gname, cand=cname, minR2=r2, pairs=len(called),
                             true_found=len(called & tp), **{f"err_{x.replace(' ', '_')}": g[x] for x in g}))
    return rows


def main():
    out = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jobs = [(s, n, seed) for s in SCEN for n in (5, 10) for seed in range(100, 110)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(out, sep="\t", index=False)
    d["wrong"] = d.minR2 < 0.95
    for v in ("minR2", "wrong", "pairs", "true_found", "err_other"):
        print("==", v)
        print(d.pivot_table(index=["cand", "grm"], columns=["n", "scen"], values=v).round(2).to_string())


if __name__ == "__main__":
    main()
