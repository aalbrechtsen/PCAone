#!/usr/bin/env python3
"""Tables of the PCP-pi benchmark (pcp_pi_bench.py): F and kinship bias and
RMSE against the realised truth, false relatedness and degree classification.

  F:  by scenario x n, for founders (not inbred), relatives, inbred, excess het
  R:  true pairs by degree (realised kinship: MZ > 0.354, 1st > 0.177, 2nd
      > 0.088, 3rd > 0.044, else distant), and the 100 random unrelated pairs
      per data set (mean kinship, share above 2^-4.5)
  degree: share of true pairs (and unrelated pairs) given the right degree

usage: pcp_pi_summary.py <prefix> [methods]
"""
import sys

import numpy as np
import pandas as pd

FM = ["pca3", "pca6", "pca10", "pca3-self", "pcp-none", "pcp-self", "pcpT-self", "dwg-none", "dwg-self"]
RM = ["KING", "pca3", "pca3-own", "pca6", "pca10", "PCAone-aad", "PCAone-dwg", "pcp-none", "pcp-self", "pcp-pair",
      "pcp-family", "pcp-own", "pcpT-own", "dwg-self", "dwg-pair", "dwg-family", "dwg-own"]
EDGES = [2 ** -1.5, 2 ** -2.5, 2 ** -3.5, 2 ** -4.5]
NAMES = ["MZ", "1st", "2nd", "3rd", "unrel/distant"]


def degree(k):
    k = np.asarray(k)
    out = np.full(k.shape, 4)
    for d, e in reversed(list(enumerate(EDGES))):
        out[k > e] = d
    return out


def br(e):
    return f"{e.mean():+.3f}/{np.sqrt((e ** 2).mean()):.3f}"


def main():
    pre = sys.argv[1]
    F = pd.read_csv(pre + "_F.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    R = pd.read_csv(pre + "_R.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    fm = [m for m in FM if m in F]
    rm = [m for m in RM if m in R]
    print("== F: bias/RMSE against realised F")
    rows = []
    for (s, n, g), d in F.groupby(["scen", "n", "group"]):
        rows.append(dict(scen=s, n=n, group=g, count=len(d), **{m: br(d[m] - d.F_true) for m in fm}))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== kinship: bias/RMSE, true pairs by degree")
    R["deg"] = degree(R.kin_true)
    rows = []
    for (s, n, k), d in R[R.kind == "true"].groupby(["scen", "n", "deg"]):
        rows.append(dict(scen=s, n=n, deg=NAMES[k], count=len(d), **{m: br(d[m] - d.kin_true) for m in rm}))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== unrelated pairs: mean kinship / share > 2^-4.5")
    rows = []
    for (s, n), d in R[R.kind == "unrel"].groupby(["scen", "n"]):
        rows.append(dict(scen=s, n=n, count=len(d),
                         **{m: f"{d[m].mean():+.3f}/{(d[m] > 2 ** -4.5).mean():.2f}" for m in rm if
                            d[m].notna().any()}))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== degree right (true pairs, 3rd or closer) / (unrelated pairs called unrelated)")
    rows = []
    for (s, n), d in R.groupby(["scen", "n"]):
        t = d[(d.kind == "true") & (d.deg < 4)]
        u = d[d.kind == "unrel"]
        rows.append(dict(scen=s, n=n, **{m: f"{(degree(t[m]) == t.deg).mean():.2f}/{(degree(u[m]) == 4).mean():.2f}"
                                         for m in rm if t[m].notna().all()}))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== k0, k2 RMSE (true pairs, 3rd or closer)")
    rows = []
    for (s, n), d in R[(R.kind == "true") & (R.deg < 4)].groupby(["scen", "n"]):
        r = dict(scen=s, n=n)
        for m in rm:
            if m + "_k0" in d and d[m + "_k0"].notna().all():
                r[m] = (f"{np.sqrt(((d[m + '_k0'] - d.k0_true) ** 2).mean()):.3f}/"
                        f"{np.sqrt(((d[m + '_k2'] - d.k2_true) ** 2).mean()):.3f}")
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
