#!/usr/bin/env python3
"""Figures of the PCP smoother for per-sample inbreeding (inbreed_smooth.py) on
the simulated data of inbreed_smooth_sim.py:

  smooth_method.pdf  one data set (10 per population): the centred GRM, its
                     structure part L and the rest S (noise diagonal + related
                     pairs) from PCP, and the weight each direction gets in the
                     prediction of the allele frequencies (smoother: max(0,
                     1 - 1/lambda) of the S-whitened GRM; k PCs: 1 for the top k)
  smooth_F.pdf       F per individual in that data set, from 3, 6, 10 PCs and
                     the smoother
Prints mean (sd) F per group over replicates (10 per n = 5, 10) as LaTeX rows.

usage: inbreed_smooth_figs.py <outdir> [reps]
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inbreed_smooth as IS  # noqa: E402
import inbreed_smooth_sim as SIM  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"
METH = [("k3", "3 PCs", "#56B4E9"), ("k6", "6 PCs", "#0072B2"), ("k10", "10 PCs", "#332288"),
        ("pcp", "PCP smoother", "#D55E00")]
GROUPS = [("other", "others"), ("relative", "relatives"), ("inbred", "inbred\n(F = 0.5)"),
          ("excess het", "excess\nhet.")]
PCOL = {"CEU": "#0072B2", "CHB": "#E69F00", "MXL": "#009E73", "YRI": "#CC79A7"}


def heat(a, M, title, order, vmax):
    a.imshow(M[np.ix_(order, order)], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    a.set_title(title, color=INK)
    a.set_xticks([])
    a.set_yticks([])


def method_fig(X, pops, group, out):
    W, info = IS.smoother(X, "pcp")
    P = info["parts"]
    # order: populations, the relatives and special individuals last within them
    rank = {"other": 0, "relative": 1, "inbred": 2, "excess het": 2}
    order = sorted(range(len(pops)), key=lambda i: (SIM.POPS.index(pops[i]), rank[group[i]], i))
    fig = plt.figure(figsize=(11.5, 10.5))
    gs = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.3)
    cells = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
    vmax = 0.25
    titles = ("GRM", "structure $L$ (PCP)", "rest $S$: noise + pairs")
    for j, (M, t) in enumerate(zip((P["C"], P["L"], P["S"]), titles)):
        a = fig.add_subplot(cells[j])
        heat(a, M, t, order, vmax)
        if j == 0:
            # population bars along the left edge
            for p in SIM.POPS:
                k = [n for n, i in enumerate(order) if pops[i] == p]
                a.plot([-2.2, -2.2], [k[0] - 0.4, k[-1] + 0.4], color=PCOL[p], lw=5, solid_capstyle="butt",
                       clip_on=False)
                a.text(-5.5, np.mean(k), p, ha="right", va="center", fontsize=11, color=INK, clip_on=False)
        if j == 2:
            # mark the special individuals
            # labels to the right of the panel, lines to their diagonal entries
            for n, i in enumerate(order):
                if group[i] in ("inbred", "excess het"):
                    a.annotate(group[i], (n, n), xytext=(1.04, 1 - (n + 0.5) / len(order)),
                               textcoords="axes fraction", fontsize=10, color=INK, va="center",
                               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
            # the related blocks (CEU family, CHB MZ pair): one label, a line to each
            for p in ("CEU", "CHB"):
                k = [n for n, i in enumerate(order) if pops[i] == p and group[i] == "relative"]
                c = np.mean(k)
                a.annotate("relatives" if p == "CEU" else "", (c, c), xytext=(1.04, 0.93),
                           textcoords="axes fraction", fontsize=10, color=INK, va="center",
                           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a = fig.add_subplot(cells[3])
    w = np.sort(P["wts"])[::-1]
    x = np.arange(1, len(w) + 1)
    for k, ls in ((3, ":"), (6, "--"), (10, "-.")):
        a.step(x, (x <= k).astype(float), where="mid", ls=ls, color=MUTED, lw=1.4, label=f"top {k} PCs")
    a.plot(x, w, "o", color="#D55E00", ms=5, mec="white", mew=0.5, label="PCP smoother", zorder=3)
    a.set_xlim(0, 16.5)
    a.set_ylim(-0.04, 1.08)
    a.set_xlabel("direction (largest first)")
    a.set_ylabel("weight in $\\hat\\pi$")
    a.set_title(f"weights (smoother: {info['eff_rank']:.1f} in total)", color=INK)
    a.legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    a.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return info


def F_all(X):
    res = {}
    for m, _, _ in METH:
        W, info = IS.smoother(X, m)
        res[m] = IS.inbreeding(X, W)[0]
    return res


def F_fig(X, pops, group, out):
    res = F_all(X)
    fig, ax = plt.subplots(1, len(METH), figsize=(15, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for a, (m, lab, col) in zip(ax, METH):
        for g_, (gk, _) in enumerate(GROUPS):
            k = np.flatnonzero(group == gk)
            xj = g_ + (rng.uniform(-0.18, 0.18, len(k)) if len(k) > 1 else np.zeros(1))
            a.scatter(xj, res[m][k], s=46, color=[PCOL[pops[i]] for i in k], edgecolors="white", lw=0.6, zorder=3)
        # truth: 0 for others and relatives, 0.5 for the inbred
        a.hlines(0, -0.4, 1.4, color=INK, lw=1.2, ls="--", zorder=2)
        a.hlines(0.5, 1.6, 2.4, color=INK, lw=1.2, ls="--", zorder=2)
        a.axhline(0, color=GRID, lw=0.6, zorder=1)
        a.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], fontsize=11)
        a.set_title(lab, color=col, fontweight="bold")
        a.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("estimated F")
    h = [Line2D([], [], ls="", marker="o", ms=8, mfc=c, mec="white", label=p) for p, c in PCOL.items()]
    h.append(Line2D([], [], ls="--", color=INK, label="true F"))
    fig.tight_layout()
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.995, 0.5), frameon=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return res


def main():
    outdir = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    X, pops, group = SIM.build(1, 10)
    info = method_fig(X, pops, group, os.path.join(outdir, "smooth_method.pdf"))
    print("example: pairs", info["pairs"], "effective rank", round(info["eff_rank"], 2))
    res = F_fig(X, pops, group, os.path.join(outdir, "smooth_F.pdf"))
    for m, lab, _ in METH:
        print(lab, {g: np.round(res[m][group == g], 3).tolist() for g, _ in GROUPS if g != "other"})
    # replicates
    for n in (10, 5):
        acc = {m: {g: [] for g, _ in GROUPS} for m, _, _ in METH}
        for seed in range(100, 100 + reps):
            X, pops, group = SIM.build(seed, n)
            r = F_all(X)
            for m in acc:
                for g, _ in GROUPS:
                    acc[m][g].extend(r[m][group == g])
        for m, lab, _ in METH:
            cells = [f"{np.mean(acc[m][g]):+.3f} ({np.std(acc[m][g]):.3f})" for g, _ in GROUPS]
            print(f"{n} & {lab} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
