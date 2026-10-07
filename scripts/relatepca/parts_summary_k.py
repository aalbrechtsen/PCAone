#!/usr/bin/env python3
"""Summary of parts_bench.py --design S1k: per-axis R2 against the truth.
axes recovered = R2 >= 0.95; 'lost vs best' = fewer axes recovered than the
best combination on the same data set; mean R2 per axis.
usage: parts_summary_k.py results/parts_S1k.tsv"""
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1], sep="\t")
r2 = d.r2_axes.str.split(",", expand=True).astype(float)
d["n_ax"] = r2.notna().sum(1)
d["rec"] = (r2 >= 0.95).sum(1)
for j in range(5):
    d[f"ax{j + 1}"] = r2[j] if j in r2 else np.nan
ids = ["src", "mode", "n", "rep", "k"]
d["best"] = d.groupby(ids).rec.transform("max")
d["lost"] = d.rec < d.best
order = list(dict.fromkeys(d.combo))
pd.set_option("display.width", 250)
for (mode, k), x in d.groupby(["mode", "k"]):
    print(f"\n######## mode={mode} k={k}  (axes scored: {x.n_ax.iloc[0]})")
    t = x.groupby(["combo", "n"]).agg(rec=("rec", "mean"), lost=("lost", "mean")).unstack("n")
    print(t.reindex(order).round(2).to_string())
    ax = [c for c in ["ax1", "ax2", "ax3", "ax4", "ax5"] if x[c].notna().any()]
    print(x.groupby("combo")[ax].mean().reindex(order).round(3).to_string())
    if x.signal_ratio.notna().any() and isinstance(x.signal_ratio.iloc[0], str):
        sr = x[x.combo == "standard"].groupby("n").signal_ratio.first()
        print("signal / BBP threshold per axis:", sr.to_dict())
    rk = x[x.combo.str.startswith("topr")].groupby("n")["rank"].mean().round(2).to_dict()
    print("mean top-r rank:", rk)
