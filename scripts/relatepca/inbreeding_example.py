#!/usr/bin/env python3
"""Example data set of inbreeding_bench.py: PC1 vs PC2 and PC2 vs PC3 for
standard PCAone, aarobust-kin, aarobust-diag (--impute-diag) and dwg. Stars:
the affected (inbred or excess-heterozygous) individuals. Prints min R2, the
affected individuals' GRM diagonal relative to the median of the others and
dwg's noise ratio.

usage: inbreeding_example.py <bfile> <workdir> <n> <scen> <rep> <out.pdf> [slide]
  slide: larger font
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import inbreeding_bench as T  # noqa: E402
from sim import read_bed  # noqa: E402

INK, GRID = "#222222", "#bdbdbd"
PCOL = {"CEU": "#0072B2", "CHB": "#E69F00", "MXL": "#009E73", "YRI": "#CC79A7"}
SHOW = [("standard", "standard PCAone"), ("aarobust-kin", "aarobust-kin"), ("aarobust-diag", "aarobust-diag"),
        ("dwg", "dwg")]


def main():
    bfile, workdir, n, scen, rep, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), \
        sys.argv[6]
    plt.rcParams.update({"font.size": 18 if len(sys.argv) > 7 and sys.argv[7] == "slide" else 13})
    T.init(bfile)
    rows = T.one((n, scen, rep, workdir, True))
    d = os.path.join(workdir, f"n{n}_{scen}_r{rep}")
    aff = np.atleast_1d(np.loadtxt(os.path.join(d, "affected.txt"), dtype=int))
    truth = np.loadtxt(os.path.join(d, "truth.txt"))
    fam, _, G = read_bed(os.path.join(d, "g"))
    pops = np.array([r[1] for r in fam])
    G = G.astype(float)
    f = G.mean(0) / 2
    diag = (((G - 2 * f) ** 2) / (2 * f * (1 - f))).mean(1)
    oth = np.setdiff1d(np.arange(len(G)), aff)
    print("GRM diagonal, affected / median of the others:", np.round(diag[aff] / np.median(diag[oth]), 2))
    for r in rows:
        print(r[3], "min R2", round(r[4], 3), "affected error", round(r[5], 3), "own-population SD", round(r[11], 2),
              "unaffected CEU", round(r[12], 2), "dwg noise ratio", r[8])
    is_aff = np.zeros(len(G), bool)
    is_aff[aff] = True
    fig, ax = plt.subplots(2, len(SHOW), figsize=(15, 7.2))
    for c, (m, title) in enumerate(SHOW):
        U = np.loadtxt(os.path.join(d, m + ".eigvecs"))[:, :3]
        s = np.sign(np.sum(U * truth, 0))  # orientation of the truth
        U = U * np.where(s == 0, 1, s)
        r2 = [r[4] for r in rows if r[3] == m][0]
        for r, (x, y) in enumerate(((0, 1), (1, 2))):
            a = ax[r, c]
            for p, col in PCOL.items():
                k = (pops == p) & ~is_aff
                a.scatter(U[k, x], U[k, y], s=55, color=col, edgecolors="white", lw=0.6, zorder=3)
                k = (pops == p) & is_aff
                a.scatter(U[k, x], U[k, y], s=260, marker="*", color=col, edgecolors=INK, lw=0.8, zorder=5)
            a.set_xlabel(f"PC{x + 1}")
            a.set_ylabel(f"PC{y + 1}")
            a.axhline(0, color=GRID, lw=0.7, ls="--", zorder=1)
            a.axvline(0, color=GRID, lw=0.7, ls="--", zorder=1)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if r == 0:
                a.set_title(f"{title}\nmin $R^2$ = {r2:.2f}", color=INK)
    h = [Line2D([], [], ls="", marker="o", ms=8, mfc=col, mec="white", label=p) for p, col in PCOL.items()]
    h.append(Line2D([], [], ls="", marker="*", ms=15, mfc="#999999", mec=INK, label="affected"))
    fig.tight_layout()
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.995, 0.5), frameon=False)
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
