#!/usr/bin/env python3
"""S2 summary: by scenario group (relatives, cross-ancestry, HWE-breaking,
small population), n and k: runs wrong, mean min R2, relatives' error,
recall, false pairs per run, time.
usage: parts_summary_s2.py results/parts_S2.tsv"""
import sys

import pandas as pd

XANC = {"halfsibx", "grandx", "avuncx", "xanc", "childx", "aswx"}
HWE = {"inbredpop_mz2", "err1_mz2"}


def group(s):
    if ":" in s:
        return "small_pop"
    if s in XANC:
        return "cross_anc"
    if s in HWE:
        return "hwe"
    return "relatives"


d = pd.read_csv(sys.argv[1], sep="\t")
d["fail"] = d.minR2 < 0.95
d["grp"] = d.scen.map(group)
order = list(dict.fromkeys(d.combo))
pd.set_option("display.width", 250)
for v, lab in [("fail", "runs wrong"), ("minR2", "mean min R2"), ("err", "relatives' error"),
               ("recall", "recall"), ("false_pairs", "false pairs per run")]:
    print(f"\n== {lab}: group x n (k = true K-1)")
    x = d[d.k == d.k.min()]
    print(x.pivot_table(index="combo", columns=["grp", "n"], values=v).reindex(order).round(3).to_string())
print("\n== runs wrong at k = 10 vs true k (all groups, all n)")
print(d.pivot_table(index="combo", columns="k", values="fail").reindex(order).round(3).to_string())
print("\n== recall at k = 10 vs true k")
print(d.pivot_table(index="combo", columns="k", values="recall").reindex(order).round(3).to_string())
print("\n== mean seconds by n")
print(d.pivot_table(index="combo", columns="n", values="sec").reindex(order).round(2).to_string())
