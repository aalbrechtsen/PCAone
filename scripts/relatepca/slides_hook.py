#!/usr/bin/env python3
"""Hook figures for the slides: a population tree, the reference truth and
standard PCA, for (a) 10 per population + 8 relatives and (b) 5 per
population without relatives (admixTjeck2)."""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods_figs as F  # noqa: E402
from methods_figs import INK, MUTED, PCOL, POPS  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def tree(ax):
    """((CEU, (CHB, Native American)), YRI); MXL = CEU x Native American"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lw = 2.2
    # root at the bottom; tips at the top
    tips = {"YRI": 0.1, "CEU": 0.38, "MXL": 0.6, "CHB": 0.9}
    y_tip = 0.82
    ax.plot([0.1, 0.1], [0.1, y_tip], color=PCOL["YRI"], lw=lw)
    ax.plot([0.1, 0.62], [0.1, 0.1], color=MUTED, lw=lw)  # root bar
    ax.plot([0.62, 0.62], [0.1, 0.42], color=MUTED, lw=lw)  # out-of-Africa
    ax.plot([0.38, 0.84], [0.42, 0.42], color=MUTED, lw=lw)
    ax.plot([0.38, 0.38], [0.42, y_tip], color=PCOL["CEU"], lw=lw)
    ax.plot([0.84, 0.84], [0.42, 0.55], color=MUTED, lw=lw)  # East Asia / Americas
    ax.plot([0.74, 0.92], [0.55, 0.55], color=MUTED, lw=lw)
    ax.plot([0.92, 0.92], [0.55, y_tip], color=PCOL["CHB"], lw=lw)
    ax.plot([0.74, 0.74], [0.55, 0.66], color="#999999", lw=lw, ls=(0, (2, 1.5)))  # Native American (unsampled)
    # admixture into MXL
    ax.annotate("", xy=(0.6, 0.72), xytext=(0.74, 0.66), arrowprops=dict(arrowstyle="-|>", color=PCOL["MXL"], lw=1.6))
    ax.annotate("", xy=(0.6, 0.72), xytext=(0.38, 0.62), arrowprops=dict(arrowstyle="-|>", color=PCOL["MXL"], lw=1.6))
    ax.plot([0.6, 0.6], [0.72, y_tip], color=PCOL["MXL"], lw=lw)
    for p, x in tips.items():
        ax.text(x, y_tip + 0.04, p, ha="center", va="bottom", fontsize=12, weight="bold", color=PCOL[p])
    ax.text(0.74, 0.69, "Native\nAmerican", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(0.5, 0.0, "MXL: admixed (European + Native American)", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("the populations", color=INK)


def scatter(ax, U, pop, rel, title, a=1, b=2):
    U = U / np.abs(U).max(0)
    for p in POPS:
        idx = (pop == p) & ~rel
        ax.scatter(U[idx, a], U[idx, b], s=34, color=PCOL[p], edgecolor="white", linewidth=0.5)
    if rel.any():
        ax.scatter(U[rel, a], U[rel, b], s=110, marker="*", color=[PCOL[p] for p in pop[rel]], edgecolor=INK,
                   linewidth=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC{a + 1}", color=MUTED)
    ax.set_ylabel(f"PC{b + 1}", color=MUTED)
    ax.set_title(title, color=INK)


def legend(fig, rel):
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    labs = list(POPS)
    if rel:
        hs.append(plt.Line2D([], [], marker="*", ls="", color="#bbbbbb", markeredgecolor=INK, markersize=12))
        labs.append("relative")
    fig.legend(hs, labs, loc="lower center", ncol=len(labs), frameon=False, bbox_to_anchor=(0.62, -0.02),
               fontsize=10)


def hook(bfile, n, scen, out, std_title):
    ds, T = F.dataset(bfile, n=n, scen=scen)
    c = F.compute(ds)
    N = len(ds["G"])
    pop = F.family_pop(ds)
    rel = np.arange(N) >= ds["n0"]
    # align the standard PCs to the truth by least squares on the unrelated (as the score)
    base = np.where(~rel)[0]
    Z = np.c_[c["U_std"], np.ones(N)]
    Bc, *_ = np.linalg.lstsq(Z[base], T[base], rcond=None)
    A = Z @ Bc
    r2 = 1 - ((T[base] - A[base]) ** 2).sum(0) / ((T[base] - T[base].mean(0)) ** 2).sum(0)
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_axes([0.0, 0.1, 0.27, 0.78])
    ax1 = fig.add_axes([0.33, 0.14, 0.3, 0.72])
    ax2 = fig.add_axes([0.69, 0.14, 0.3, 0.72])
    tree(ax0)
    scatter(ax1, T, pop, rel, "the truth")
    scatter(ax2, A, pop, rel, f"{std_title}\nPC3 recovered: $R^2$ = {r2[2]:.2f}")
    legend(fig, rel.any())
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "R2 per axis", np.round(r2, 3), "N", N)


if __name__ == "__main__":
    bfile, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    hook(bfile, 10, "many", os.path.join(outdir, "hook_relatives.pdf"), "standard PCA, 10 per population + 8 relatives")
    hook(bfile, 5, "none", os.path.join(outdir, "hook_smalln.pdf"), "standard PCA, 5 per population, no relatives")
