#!/usr/bin/env python3
"""Run time of the Python prototypes (same inputs, 16 BLAS threads): matrix
construction + fit + PCs; the KING screen (shared) is excluded."""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed  # noqa: E402

_, _, Gall = read_bed(sys.argv[1])
k, tau = 4, B.TAU
print("N\tmethod\tseconds\tpairs")
for N in map(int, sys.argv[2].split(",")):
    G = Gall[:N].astype(float)
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    t0 = time.time()
    U, r, pp = W.dwg(G, k, tau, cand, "v")
    print(f"{N}\tdwg_v\t{time.time() - t0:.2f}\t{len(pp)}", flush=True)
    t0 = time.time()
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L, S, _, _ = I.lr_kin_fd_auto(H, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
    pp = I.lr_kin_pairs(H, D, L, S)
    AM = G @ G.T / G.shape[1]
    I.whitened_noise(AM, np.diag(AM) - np.diag(L), pp, k)
    print(f"{N}\tdetect_white\t{time.time() - t0:.2f}\t{len(pp)}", flush=True)
    t0 = time.time()
    _, C = B.grm(G)
    Lp, Sp, _ = I.pcp_kin(C, tau, True, cand=cand)
    I.top_eig(Lp, k)
    npair = int((np.abs(np.triu(Sp, 1)) > 1e-12).sum())
    print(f"{N}\taarobust_kin\t{time.time() - t0:.2f}\t{npair}", flush=True)
