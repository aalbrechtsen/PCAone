#!/usr/bin/env python3
"""Scree plots for a reindeer data set: the top 10 eigenvalues of standard
PCAone, aarobust-kin, aarobust-kin --impute-diag (aad_all) and dwg (the eigenvalues of its whitened fit), all SNPs (full/) next to LD-pruned (ld/).

usage: scree_reindeer.py <dir> <out.pdf>
"""
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, GRID = "#222222", "#bdbdbd"
# (run, label, colour, marker, line style, sideways offset): offsets and styles keep
# lines that coincide (standard and aarobust-kin) visible
RUNS = (("std_all", "standard PCAone", "#0072B2", "o", "-", -0.2), ("aar_all", "aarobust-kin", "#D55E00", "s", "--", -0.07),
        ("aad_all", "aarobust-kin+diag", "#CC79A7", "D", "-.", 0.07), ("dwg_all", "dwg", "#009E73", "^", ":", 0.2))


def main():
    d, out = sys.argv[1], sys.argv[2]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for a, sub, title in zip(ax, ("full", "ld"), ("all SNPs", "LD-pruned")):
        for run, label, col, m, ls, dx in RUNS:
            e = np.loadtxt(os.path.join(d, sub, run + ".eigvals"))[:10]
            a.plot(np.arange(1, len(e) + 1) + dx, e, marker=m, ls=ls, color=col, lw=2, ms=7, mec="white", mew=0.8,
                   label=label)
        a.set_yscale("log")
        a.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 50], ["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xticks(range(1, 11))
        a.set_xlabel("PC")
        a.set_title(title, color=INK)
        a.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("eigenvalue (log scale)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
