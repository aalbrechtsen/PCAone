#!/usr/bin/env python3
"""Why are true close pairs missed by dwg in the HAPNEST test? For every true
pair (pedigree kinship >= tau): relationship type, pedigree kinship, KING,
evalAdmix kinship and the k0 moment from the dwg PCs, and a PC-Relate style
realised kinship (individual allele frequencies from the PCs)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from hapnest_test import simulate  # noqa: E402
from sim import read_bed  # noqa: E402

pool, pre, seed, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fam, bim, G = read_bed(pool)
info = {l.split()[0]: l.split()[1:] for l in list(open(os.path.join(os.path.dirname(pool), "individuals.tsv")))[1:]}
ids = [r[1] for r in fam]
anc = np.array([info[x][0] for x in ids])
sset = np.array([info[x][1] for x in ids])
pidx = np.where(sset == "pool")[0]
Gs, famid, ftype, truth, ancs = simulate(G.astype(np.int8)[pidx], anc[pidx], bim, N, 0.3, seed)
keep = Gs.min(0) != Gs.max(0)
Gs = Gs[:, keep].astype(float)
U = np.loadtxt(pre + "_dwg_k5.eigvecs")
det = {tuple(sorted((int(l.split()[0][1:]), int(l.split()[1][1:])))) for l in list(open(pre + "_dwg_k5.relpairs"))[1:]}
tp = [p for p, k in truth.items() if k >= B.TAU]
king = B.king_robust(Gs)
ea = I.evaladmix_kin(Gs, U, intercept=True)
V = np.c_[U, np.ones(N)]
Pi = np.clip(0.5 * V @ np.linalg.lstsq(V, Gs, rcond=None)[0], 1e-3, 1 - 1e-3)
rows = []
for i, j in tp:
    ri, rj = Gs[i] - 2 * Pi[i], Gs[j] - 2 * Pi[j]
    phi_pc = (ri * rj).sum() / (4 * np.sqrt(Pi[i] * (1 - Pi[i]) * Pi[j] * (1 - Pi[j])).sum())
    obs = np.sum((Gs[i] == 0) & (Gs[j] == 2)) + np.sum((Gs[i] == 2) & (Gs[j] == 0))
    exp = np.sum(Pi[i] ** 2 * (1 - Pi[j]) ** 2 + (1 - Pi[i]) ** 2 * Pi[j] ** 2)
    rows.append((ftype[i], ancs[i], ancs[j], truth[(i, j)], king[i, j], ea[i, j], obs / exp, phi_pc, (i, j) in det))
import pandas as pd
d = pd.DataFrame(rows, columns=["type", "anc_i", "anc_j", "ped", "king", "evaladmix", "k0", "phi_pc", "detected"])
pd.set_option("display.width", 200)
print(d.groupby(["type", "detected"])[["ped", "king", "evaladmix", "k0", "phi_pc"]].agg(["count", "mean", "min", "max"]).round(3).to_string())
m = d[~d.detected]
print("\nmissed pairs:", len(m), "of", len(d))
print(m.sort_values("phi_pc").round(3).to_string())
