#!/usr/bin/env python3
"""Large simulated data set with families for timing the relative search
(same model as highn_test.py: Balding-Nichols populations, unlinked SNPs)."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from highn_test import simulate  # noqa: E402
from sim import write_bed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--M", type=int, default=20000)
ap.add_argument("--rel", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
G, famid, ftype, _ = simulate(a.N, a.rel, a.M, 2, a.seed)
keep = G.min(0) != G.max(0)
G = G[:, keep]
bim = [f"1\tsnp{i}\t0\t{i + 1}\tA\tC" for i in range(G.shape[1])]
write_bed(a.out, [[str(famid[i]) if famid[i] else f"U{i}", f"I{i}", "0", "0", "0", "-9"] for i in range(a.N)], bim, G)
print("wrote", a.out, G.shape)
