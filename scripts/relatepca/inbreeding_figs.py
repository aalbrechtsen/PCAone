#!/usr/bin/env python3
"""Summary figures of inbreeding_bench.py: for inbreeding (inb_*) or excess
heterozygosity (het_*), one column per number of affected individuals (1, 5,
all CEU) and one row per score against the number per population:
  min R2, MEV, minSSE (whole-sample scores), the affected individuals' error in
  units of the spread of their own population (err_pop) and the largest
  individual error over the median one (max_med).
Grey: the same scores on the data without any change (dashed standard PCAone,
dotted dwg); in the err_pop row, the unaffected CEU of that data. The methods
are shifted sideways (and have their own line style) so that lines that
coincide stay visible. aarobust-kin --impute-diag is left out: without
relatives it equals aarobust-diag.

usage: inbreeding_figs.py <inbreeding.tsv> inb|het <out.pdf> [slide]
  slide: rows min R2 and err_pop only, larger font
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"
# (method, label, colour, marker, line style, sideways offset)
METH = [("standard", "standard PCAone", "#0072B2", "o", "-", -0.18),
        ("aarobust-kin", "aarobust-kin", "#56B4E9", "s", "--", -0.06),
        ("aarobust-diag", "aarobust-diag", "#E69F00", "D", "-", 0.06),
        ("dwg", "dwg", "#009E73", "^", ":", 0.18)]
NS = [5, 10, 20, 40, 0]  # 0: the full data
# (label, log scale, y range); full data: the truth is standard PCA of the unchanged
# genotypes, so standard without change scores 0 there (off the log axes)
ROWS = {"minR2": ("min $R^2$", False, (-0.02, 1.02)), "mev": ("MEV", False, (0.6, 1.01)),
        "minsse": ("minSSE", True, (1e-3, 5)), "err_pop": ("affected error\n(own-population SD)", True, (0.05, 60)),
        "max_med": ("largest / median\nindividual error", True, (2, 600))}


def main():
    d = pd.read_csv(sys.argv[1], sep="\t")
    kind, out = sys.argv[2], sys.argv[3]
    slide = len(sys.argv) > 4 and sys.argv[4] == "slide"
    rows = ["minR2", "err_pop"] if slide else list(ROWS)
    plt.rcParams.update({"font.size": 17 if slide else 13})
    cols = [(f"{kind}_1", "1 individual"), (f"{kind}_5", "5 individuals"), (f"{kind}_pop", "all CEU")]
    x = np.arange(len(NS))
    fig, ax = plt.subplots(len(rows), 3, figsize=(14, 3.3 * len(rows) + 0.8) if slide else (13, 2.9 * len(rows)),
                           sharex=True, sharey="row")
    none = d[d.scen == "none"].groupby(["method", "n"])
    for c, (scen, title) in enumerate(cols):
        for r, v in enumerate(rows):
            a = ax[r, c]
            for m, label, col, mk, ls, dx in METH:
                g = d[(d.scen == scen) & (d.method == m)].groupby("n")[v].mean()
                a.plot(x + dx, [g[n] for n in NS], marker=mk, ls=ls, color=col, lw=2, ms=7, mec="white",
                       label=label)
            # reference: no change (err_pop: the unaffected CEU)
            ref = "err_pop_ref" if v == "err_pop" else v
            for m, ls, lab in (("standard", "--", "standard, no change"), ("dwg", ":", "dwg, no change")):
                a.plot(x, [none[ref].mean()[(m, n)] for n in NS], ls=ls, color=MUTED, lw=1.4, zorder=1, label=lab)
            if ROWS[v][1]:
                a.set_yscale("log")
            a.set_ylim(*ROWS[v][2])
            a.grid(axis="y", color=GRID, lw=0.5)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            if c == 0:
                a.set_ylabel(ROWS[v][0])
        ax[0, c].set_title(title, color=INK)
        ax[-1, c].set_xticks(x, [str(n) if n else "all" for n in NS])
        ax[-1, c].set_xlabel("individuals per population")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(0.995, 0.5), frameon=False)
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
