#!/usr/bin/env python3
"""Why dwg beats aarobust-kin+diag at small n (inbreed_pca_bench.py): on the
replicates of one scenario, min R2 of the PCs from dwg's structure fit
(dw_grm.fit, rank from the noise edge, at most 4) not whitened, whitened by the
noise only and by noise + pairs, with every pair or only the true pairs as
candidates; and of PCP's structure part with the true pairs as candidates.

usage: inbreed_pca_why.py <n per population> <scenario>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import benchmark as B
import illustrate as I
import dw_grm as DW
import inbreed_smooth as IS
import inbreed_smooth_sim as SIM
import inbreed_pca_bench as PB

n, scen = int(sys.argv[1]), sys.argv[2]
res = {}
def add(k, U, T):
    res.setdefault(k, []).append(B.score(U, T, len(T), [], 3)[0].min())
npairs = []
for seed in range(100, 110):
    X, pops, group, ex = SIM.build(seed, n, scen, full=True)
    Gall, lab = SIM._DATA["G"], SIM._DATA["lab"]
    ref = np.setdiff1d(np.flatnonzero(np.isin(lab, SIM.POPS)), ex["idx"][ex["idx"] >= 0])
    T = B.truth_reference(dict(G=ex["clean"].astype(float), ref_idx=ref, cols=ex["cols"]), Gall, 3)
    N = len(X)
    tp = PB.true_pairs(group, pops, 4 * n)
    cand = np.zeros((N, N), bool)
    for i, j in tp:
        cand[i, j] = cand[j, i] = True
    A, f, w = IS.scaled_uncentred(X)
    for cname, cm in (("all pairs", np.ones((N, N), bool)), ("true pairs", cand)):
        Ld, Sd, rd, pd_ = DW.fit(A, B.TAU, cm, DW.noise_edge(f, w, N), 4, None)
        if cname == "all pairs":
            npairs.append(len(pd_))
        P = np.eye(N) - 1 / N
        v = np.maximum(np.diag(A) - np.diag(Ld), 1e-6)
        add(f"dwg py, cand {cname}: L not whitened", I.top_eig(P @ Ld @ P, 3)[1], T)
        add(f"dwg py, cand {cname}: noise only whitened", I.whitened_noise(A, v, [], 3, centre=True), T)
        add(f"dwg py, cand {cname}: noise+pairs whitened", I.whitened_noise(A, v, pd_, 3, centre=True), T)
    C = IS.centred_grm(X)
    L, S, _ = I.pcp_kin(C, B.TAU, free_diag=True, cand=cand)
    add("PCP, cand true pairs", np.linalg.eigh(L)[1][:, ::-1][:, :3], T)
print("pairs called by dwg py with all candidates:", npairs)
for k, v in res.items():
    v = np.array(v)
    print(f"{k:46s} min R2 mean {v.mean():.3f}  wrong {np.mean(v < 0.95):.1f}")
