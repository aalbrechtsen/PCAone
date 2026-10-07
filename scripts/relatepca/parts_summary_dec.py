#!/usr/bin/env python3
"""S2 decision-rule table (section 3 of S2_small_related.md): per combination
wrong (%) with cap 3 (no small population / small population), wrong with cap
10, recall same / different ancestry, relatives' error, false pairs.
usage: parts_summary_dec.py results/parts_S2n.tsv [more.tsv ...]"""
import sys

import pandas as pd

XANC = {"halfsibx", "grandx", "avuncx", "xanc", "childx", "aswx"}
d = pd.concat([pd.read_csv(f, sep="\t") for f in sys.argv[1:]])
d = d[d.combo != "standard"]
d["fail"] = 100 * (d.minR2 < 0.95)
d["small"] = d.scen.str.contains(":")
d["xanc"] = d.scen.isin(XANC)
k3, k10 = d[d.k == d.k.min()], d[d.k == 10]
rows = []
for c in dict.fromkeys(d.combo):
    a, b = k3[k3.combo == c], k10[k10.combo == c]
    same = a[~a.xanc & ~a.small & (a.n_true > 0)]
    rows.append(dict(combo=c, wrong_nosmall=a[~a.small].fail.mean(), wrong_small=a[a.small].fail.mean(),
                     wrong_cap10=b.fail.mean(), rec_same=same.recall.mean(),
                     rec_x3=a[a.xanc].recall.mean(), rec_x10=b[b.xanc].recall.mean(),
                     err=a[~a.small].err.mean(), fp_all=a[~a.small].false_pairs.mean(),
                     fp_small=a[a.small].false_pairs.mean(), sec=a.sec.mean(), n=len(a)))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
