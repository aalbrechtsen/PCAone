#!/usr/bin/env python3
"""Slide figure: a small population (admixTjeck2, 5 CEU, CHB, MXL and only 2
YRI, no relatives; the S1s example of report/scenarios, n=5, rep 5). Truth
and the four methods of the slides (four_methods_bench.run_methods), every
PC flipped to agree with the truth.

usage: slides_yri2.py <bfile> <out.pdf> [rep]
"""
import os
import sys
import zlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import four_methods_bench as FM  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import methods_figs as F  # noqa: E402
import slides_hook as S  # noqa: E402
from methods_figs import INK, MUTED, PCOL  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
POPS = ["CEU", "CHB", "MXL", "YRI"]


def dataset(rep, n=5, small=("YRI", 2)):
    """as parts_bench.one: the first 2 base individuals of YRI kept"""
    seed = zlib.crc32(f"{','.join(POPS)}|{n}|0|none|{rep}".encode())
    ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, POPS, n, 0, B.SCENARIOS["none"], seed)
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    bp = np.array(ds["base_pop"])
    n0 = ds["n0"]
    keep = [i for i in range(n0) if bp[i] != small[0] or (bp[:i] == small[0]).sum() < small[1]]
    return ds["G"].astype(float)[keep], T[keep], bp[keep]


def r2_min(U, T):
    Z = np.c_[U, np.ones(len(U))]
    b, *_ = np.linalg.lstsq(Z, T, rcond=None)
    return (1 - ((T - Z @ b) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)).min()


def main():
    bfile, out = sys.argv[1], sys.argv[2]
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    Gb.init(bfile)
    G, T, pop = dataset(rep)
    N = len(G)
    T = S.orient_truth(T, pop)
    res = FM.run_methods(G)
    panels = [("truth (reference panel)", T, None)]
    for m, name in [("standard", "standard PCA"), ("detect_white", "detect-white"), ("dwg", "dwg"),
                    ("aarobust_kin", "aarobust-kin")]:
        U = S.align_signs(res[m][0][:, :3], T, np.arange(N))
        panels.append((name, U, r2_min(U, T)))
    fig, axs = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 5.4))
    no = np.zeros(N, bool)
    for c, (name, U, r2) in enumerate(panels):
        bad = r2 is not None and r2 < 0.95
        title = name if r2 is None else f"{name}\nmin $R^2$ = {r2:.2f}"
        for row, (a, b) in enumerate([(0, 1), (1, 2)]):
            ax = axs[row, c]
            S.scatter(ax, U, pop, no, title if row == 0 else "", a=a, b=b)
            yri = pop == "YRI"
            Un = U / np.abs(U).max(0)
            ax.scatter(Un[yri, a], Un[yri, b], s=70, color=PCOL["YRI"], edgecolor=INK, linewidth=0.8, zorder=3)
            if row == 0:
                ax.title.set_color("#c62828" if bad else INK)
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white" if p != "YRI" else INK,
                     markersize=8 if p != "YRI" else 10) for p in POPS]
    fig.legend(hs, POPS[:3] + ["YRI (only 2)"], loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "N", N, {p[0]: (None if p[2] is None else round(p[2], 3)) for p in panels})


if __name__ == "__main__":
    main()
