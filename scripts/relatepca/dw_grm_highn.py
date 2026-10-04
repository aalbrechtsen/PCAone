#!/usr/bin/env python3
"""detect-white-GRM at N = 2000, k = 20: family axes beyond the ancestry axes
and relatives' placement against the reference truth (highn_test.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm_bench as DB  # noqa: E402
from highn_test import family_eta2  # noqa: E402
from sim import read_bed  # noqa: E402

bfile, k, Ka = sys.argv[1], int(sys.argv[2]), 4
fam, _, G = read_bed(bfile)
G = G.astype(float)
famid = np.array([int(r[0]) if r[0][0] != "U" else 0 for r in fam])
king = B.king_robust(G)
cand = king > B.KING_SCREEN
np.fill_diagonal(cand, False)
out, ranks = DB.methods(G, k, B.TAU, cand)
_, C = B.grm(G)
out["standard"] = np.linalg.eigh(C)[1][:, ::-1][:, :k]
print(f"{os.path.basename(bfile)}: ranks {ranks}")
for m, U in out.items():
    e = np.array([family_eta2(U[:, j], famid) for j in range(Ka, k)])
    print(f"  {m:13s} family axes {int((e > 0.5).sum()):2d}/{k - Ka}  max eta2 {e.max():.2f}  median {np.median(e):.3f}")
