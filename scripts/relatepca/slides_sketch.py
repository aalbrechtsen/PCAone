#!/usr/bin/env python3
"""Slide figure: the sketch search for relatives (the first panels of
regime_figs.fig_verylarge, larger): (1) count sketch, (2) correlation of
sketch rows for relatives and random pairs (simulated N = 20,000, s = 2048),
(3) nearest neighbours, (4) recall against KING over all pairs.

usage: slides_sketch.py <N20k bfile prefix (with _king.kin0)> <out.pdf>
"""
import os
import sys

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regime_figs as R  # noqa: E402
from methods_figs import INK, MUTED  # noqa: E402

plt.rcParams.update({"font.size": 13})
# degree of relatedness: one purple ramp, light (3rd) to dark (MZ); not the ancestry colours
KCOL = {"3rd": "#bcbddc", "2nd": "#9e9ac8", "1st": "#6a51a3", "MZ": "#3f007d"}


def sketch_panel(a):
    """rows: individuals; columns: SNPs. Above each SNP: the sketch column it is added to, with its sign"""
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis("off")
    rng = np.random.default_rng(3)
    m, s_ = 6, 3
    Gs = rng.integers(0, 3, (10, m))
    cols = rng.integers(0, s_, m)
    sgn = rng.choice(["+", "\u2212"], m)
    cm = matplotlib.colors.ListedColormap(["#f4f4f4", "#a0a0a0", "#303030"])
    x0, x1, y0, y1 = 0.06, 0.56, 0.16, 0.72
    a.imshow(Gs, cmap=cm, vmin=-0.5, vmax=2.5, extent=(x0, x1, y0, y1), aspect="auto", interpolation="nearest")
    for j in range(m):
        a.text(x0 + (x1 - x0) * (j + 0.5) / m, y1 + 0.03, f"{sgn[j]}{cols[j] + 1}", fontsize=10, ha="center",
               va="bottom", color=INK)
    Ys = rng.normal(size=(10, s_))
    u0, u1 = 0.74, 0.98
    a.imshow(Ys, cmap="Greys", extent=(u0, u1, y0, y1), aspect="auto", interpolation="nearest")
    for q in range(s_):
        a.text(u0 + (u1 - u0) * (q + 0.5) / s_, y1 + 0.03, f"{q + 1}", fontsize=11, ha="center", va="bottom",
               color=INK)
    a.annotate("", xy=(u0 - 0.01, 0.44), xytext=(x1 + 0.02, 0.44),
               arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    a.text(0.02, (y0 + y1) / 2, "individuals ($N$)", rotation=90, ha="center", va="center", fontsize=11,
           color=MUTED)
    a.text((x0 + x1) / 2, 0.05, "SNPs ($M$)", ha="center", fontsize=11, color=MUTED)
    a.text((u0 + u1) / 2, 0.05, "sketch ($s$)", ha="center", fontsize=11, color=MUTED)
    a.set_title("1  count sketch (pass 1):\n    SNP $\\to$ $\\pm$ one column", color=INK, loc="left",
                fontsize=13.5)


def hist_panel(a, sk, Nsk):
    bins = np.linspace(-0.12, 1.02, 115)
    order = ["random pairs", "3rd degree", "2nd degree", "1st degree", "MZ/dup (>0.354)"]
    lab = {"MZ/dup (>0.354)": "MZ", "1st degree": "1st degree", "2nd degree": "2nd degree",
           "3rd degree": "3rd degree", "random pairs": "unrelated"}
    colr = {"MZ/dup (>0.354)": KCOL["MZ"], "1st degree": KCOL["1st"], "2nd degree": KCOL["2nd"],
            "3rd degree": KCOL["3rd"], "random pairs": "#BBBBBB"}
    for name in order:
        if name in sk:
            a.hist(sk[name], bins=bins, color=colr[name], alpha=0.85, label=lab[name], density=True,
                   histtype="stepfilled", lw=0)
    sd = sk["random pairs"].std()
    a.set_yscale("log")
    a.set_ylim(5e-3, 1e4)
    a.set_yticks([])
    a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.tick_params(labelsize=12, colors=MUTED)
    a.set_xlabel("correlation of sketch rows ($\\approx 2\\phi$)", color=MUTED)
    for s in ["top", "right", "left"]:
        a.spines[s].set_visible(False)
    a.legend(fontsize=10, frameon=False, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2)
    a.set_title(f"2  relatives stand out\n    unrelated: $0 \\pm 1/\\sqrt{{s}}$ = {sd:.3f}", color=INK, loc="left",
                fontsize=13.5)


def nn_panel(a):
    """each person's nearest neighbours in the sketch: the relatives"""
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis("off")
    rng = np.random.default_rng(7)
    P = rng.uniform(0.08, 0.92, (40, 2))
    P[:, 1] = 0.3 + 0.65 * (P[:, 1] - 0.08) / 0.84
    a.scatter(P[:, 0], P[:, 1], s=18, color="#BBBBBB", lw=0)
    me = np.array([0.45, 0.62])
    rel = me + np.array([[0.07, 0.04], [-0.05, 0.06], [0.03, -0.08]])
    a.scatter(*rel.T, s=40, color=KCOL["1st"], lw=0, zorder=3)
    a.scatter(*me, s=80, marker="*", color=INK, zorder=4)
    a.add_patch(plt.Circle(me, 0.13, fill=False, ec=INK, lw=1.2, ls="--"))
    for r in rel:
        a.plot([me[0], r[0]], [me[1], r[1]], color=KCOL["1st"], lw=0.8, zorder=2)
    a.text(0.5, 0.0, "top 5 neighbors per person:\n$N^2 s$, not $N^2 M$; then exact KING", ha="center",
           va="bottom", fontsize=12.2, color=INK)
    a.set_title("3  nearest neighbors\n    = candidate pairs", color=INK, loc="left", fontsize=13.5)


def recall_panel(a):
    names = ["MZ", "1st", "2nd", "3rd"]
    rec = [1.0, 1.0, 1.0, 0.988]
    a.bar(range(4), rec, color=[KCOL[n] for n in names], width=0.6)
    for x, v in enumerate(rec):
        a.text(x, v + 0.01, f"{100 * v:.1f}%" if v < 1 else "100%", ha="center", va="bottom", fontsize=9.5,
               color=INK)
    a.set_xticks(range(4))
    a.set_xticklabels(names, fontsize=12.2)
    a.set_yticks([])
    a.set_ylim(0, 1.18)
    for s in ["top", "right", "left"]:
        a.spines[s].set_visible(False)
    a.set_title("4  recall vs KING\n    over all pairs", color=INK, loc="left", fontsize=13.5)


def main():
    pre, out = sys.argv[1], sys.argv[2]
    sk, Nsk = R.sketch_corr(pre, pre + "_king.kin0")
    fig = plt.figure(figsize=(11, 3.6))
    xs, ws = [0.0, 0.27, 0.58, 0.83], [0.23, 0.27, 0.2, 0.17]
    ax = [fig.add_axes([x, 0.16, w, 0.62]) for x, w in zip(xs, ws)]
    sketch_panel(ax[0])
    hist_panel(ax[1], sk, Nsk)
    nn_panel(ax[2])
    recall_panel(ax[3])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "N", Nsk, {k: len(v) for k, v in sk.items()})


if __name__ == "__main__":
    main()
