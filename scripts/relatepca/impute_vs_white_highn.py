#!/usr/bin/env python3
"""Large N, many PCs: PCs of the Gram with the related pairs and the diagonal
imputed from the detection fit (imp_diagL) against whitening (detect-white)
and standard PCA. Family axes among the PCs beyond the ancestry axes:
eta^2 of a PC explained by family membership (highn_test.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed  # noqa: E402


from highn_test import family_eta2 as eta2  # noqa: E402


bfile, k, Ka = sys.argv[1], int(sys.argv[2]), 4
fam, _, G = read_bed(bfile)
G = G.astype(float)
famid = np.array([int(r[0]) if r[0][0] != "U" else 0 for r in fam])
N, M = G.shape
tau = B.TAU
king = B.king_robust(G)
cand = king > B.KING_SCREEN
np.fill_diagonal(cand, False)
H = I.cs_matrix(G)
D = (G * (2 - G)).mean(1)
L, S, _, r = I.lr_kin_fd_auto(H, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
pp = I.lr_kin_pairs(H, D, L, S)
AM = G @ G.T / M
res = {}
_, C = B.grm(G)
res["standard"] = I.top_eig(C, k)[1]
res["detect_white"] = I.whitened_noise(AM, np.diag(AM) - np.diag(L), pp, k)
off = (np.abs(S) > 1e-12) & ~np.eye(N, dtype=bool)
Mi = np.where(off, L, H)
np.fill_diagonal(Mi, np.diag(L))
res["imp_diagL"] = I.cs_pcs(Mi, k)[1]
print(f"N={N} rank={r} pairs={len(pp)}; PCs {Ka + 1}..{k} (beyond the {Ka} ancestry axes)")
for m, U in res.items():
    e = np.array([eta2(U[:, j], famid) for j in range(Ka, k)])
    print(f"{m:13s} family axes (eta2>0.5): {int((e > 0.5).sum()):2d} of {k - Ka}; max eta2 {e.max():.2f}; "
          f"median eta2 {np.median(e):.3f}")
