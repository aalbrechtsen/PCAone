#!/usr/bin/env python3
"""KING-robust for the true close pairs (pedigree kinship >= 2^-3.5) of the
benchmark scenarios: how far does structure/admixture pull it, and do any
fall below the 0.04 screen?"""
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402

Gb.init("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2")
pops = "CEU,CHB,MXL,YRI".split(",")
rows = []
for scen in B.SCENARIOS:
    for n in (5, 10, 20):
        for rep in range(3):
            seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
            try:
                ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
            except (ValueError, KeyError):
                continue
            K = ds["Kped"]
            king = B.king_robust(ds["G"].astype(float))
            for i, j in zip(*np.where(np.triu(K >= B.TAU, 1))):
                rows.append((scen, n, rep, K[i, j], king[i, j]))
import pandas as pd
d = pd.DataFrame(rows, columns=["scen", "n", "rep", "ped", "king"])
d["bias"] = d.king - d.ped
d.to_csv(sys.argv[1], sep="\t", index=False)
g = d.groupby(["scen", "ped"]).agg(pairs=("king", "size"), king_min=("king", "min"), king_mean=("king", "mean"),
                                   bias_min=("bias", "min"), bias_max=("bias", "max"),
                                   below_screen=("king", lambda x: int((x <= 0.04).sum())))
pd.set_option("display.width", 200)
print(g.round(3).to_string())
print("\npairs below the 0.04 screen:", int((d.king <= 0.04).sum()), "of", len(d))
print("2nd-degree pairs (ped 0.125): KING min", d[d.ped.round(3) == 0.125].king.min().round(3))
