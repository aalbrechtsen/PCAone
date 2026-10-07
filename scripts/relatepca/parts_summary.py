#!/usr/bin/env python3
"""Summaries of parts_bench.py output: runs wrong (min R2 < 0.95) per
combination by n (and by M, k, scenario), mean min R2, relatives' error,
recall, false pairs, and a speed class relative to standard PCA on the same
data (very fast <= 2x, fast <= 10x, slow > 10x); "relatively slow" when a
combination takes > 3x the fastest robust combination on the same data.

usage: parts_summary.py results/parts_S1.tsv [more.tsv ...]"""
import sys

import numpy as np
import pandas as pd


def speed_class(x):
    return "very fast" if x <= 2 else "fast" if x <= 10 else "slow"


def main(files):
    d = pd.concat([pd.read_csv(f, sep="\t") for f in files])
    d["fail"] = d.minR2 < 0.95
    order = list(dict.fromkeys(d.combo))
    pd.set_option("display.width", 250)
    for key, x in d.groupby(["design"] + (["panel"] if "panel" in d else [])):
        print(f"\n######## {key}")
        tab = lambda v, cols: x.pivot_table(index="combo", columns=cols, values=v).reindex(order)  # noqa: E731
        print("\n== runs wrong (fraction) by n")
        print(tab("fail", "n").round(3).to_string())
        print("\n== runs wrong by M, k (all n)")
        print(tab("fail", ["M", "k"]).round(3).to_string())
        print("\n== runs wrong by scenario (all n)")
        print(tab("fail", "scen").round(3).to_string())
        print("\n== mean min R2 by n")
        print(tab("minR2", "n").round(3).to_string())
        if x.n_true.sum() > 0:
            print("\n== relatives' error / recall by n")
            print(tab("err", "n").round(3).to_string())
            print(tab("recall", "n").round(3).to_string())
        print("\n== false pairs per run by n")
        print(tab("false_pairs", "n").round(3).to_string())
        # time relative to standard PCA on the same data set
        ids = ["n", "M", "scen", "rep", "k"]
        st = x[x.combo == "standard"].set_index(ids).sec.rename("sec_std")
        y = x.join(st, on=ids)
        y["ratio"] = y.sec / np.maximum(y.sec_std, 1e-4)
        rob = y[y.combo != "standard"]
        fastest = rob.groupby(ids).sec.min().rename("sec_fast")
        y = y.join(fastest, on=ids)
        y["rel"] = y.sec / np.maximum(y.sec_fast, 1e-4)
        sp = y.groupby("combo").agg(sec=("sec", "mean"), ratio=("ratio", "median"), rel=("rel", "median")).reindex(order)
        sp["class"] = sp.ratio.map(speed_class) + np.where(sp.rel > 3, ", relatively slow", "")
        print("\n== time: mean seconds, median ratio to standard PCA and to the fastest robust combination, class")
        print(sp.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
