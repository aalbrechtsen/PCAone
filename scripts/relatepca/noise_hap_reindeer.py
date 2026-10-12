#!/usr/bin/env python3
"""Experimental: the dwg weights of each animal, 1 / sqrt(v_i) relative to the
median, from diploid dwg (PCAone, <dir>/full|ld/dwg_all.noise) and haploid dwg
(reindeer_hap.py, hapdwg_all.noise), grouped by site (log2 scale). Lines: site
medians; stars: related samples (KING list as slides_reindeer.py).

usage: noise_hap_reindeer.py <dir> <full|ld> <out.pdf>
"""
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slides_reindeer import COLS, MARKS, find, read_fam  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"


def main():
    d, sub, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ids, pops = read_fam(os.path.join(d, "rd_full.fam" if sub == "full" else "rd_ld.fam"))
    pops = np.array(pops)
    removed = {w for l in list(open(find(d, "king.king.cutoff.out.id")))[1:] for w in l.split()}
    rel = np.array([i in removed for i in ids])
    W = {}
    for run, lab in (("dwg_all", "diploid dwg"), ("hapdwg_all", "haploid dwg")):
        v = np.loadtxt(os.path.join(d, sub, run + ".noise"))
        w = 1 / np.sqrt(v)
        W[lab] = w / np.median(w)
    order = sorted(set(pops))
    idx = np.concatenate([np.flatnonzero(pops == p) for p in order])
    x = np.arange(len(idx))
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True, sharey=True)
    for a, (lab, r) in zip(ax, W.items()):
        for c, p in enumerate(order):
            k = np.flatnonzero(pops[idx] == p)
            sel = idx[k]
            a.scatter(x[k][~rel[sel]], r[sel][~rel[sel]], s=60, marker=MARKS[c % 4], facecolors=COLS[c],
                      edgecolors="white", lw=0.6, zorder=3)
            a.scatter(x[k][rel[sel]], r[sel][rel[sel]], s=190, marker="*", facecolors=COLS[c],
                      edgecolors="white", lw=0.7, zorder=4)
            a.hlines(np.median(r[sel]), x[k].min() - 0.4, x[k].max() + 0.4, color=COLS[c], lw=1.5, zorder=2)
        a.axhline(1, color=MUTED, lw=0.9, ls="--", zorder=1)
        a.set_yscale("log", base=2)
        a.set_yticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_ylabel(f"{lab} weight\n$1/\\sqrt{{v_i}}$ / median", fontsize=12)
        a.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    centres = [np.flatnonzero(pops[idx] == p).mean() for p in order]
    ax[-1].set_xticks(centres, [p.replace("_", " ") for p in order], rotation=90)
    for b in np.cumsum([np.sum(pops == p) for p in order])[:-1]:
        for a in ax:
            a.axvline(b - 0.5, color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    a_, b_ = W["diploid dwg"], W["haploid dwg"]
    print(sub, "cor(log weights) diploid vs haploid", round(float(np.corrcoef(np.log(a_), np.log(b_))[0, 1]), 3),
          "range diploid", round(float(a_.min()), 2), round(float(a_.max()), 2),
          "haploid", round(float(b_.min()), 2), round(float(b_.max()), 2))
    for p in order:
        k = pops == p
        print(f"  {p:24s} diploid {np.median(a_[k]):.2f}  haploid {np.median(b_[k]):.2f}")


if __name__ == "__main__":
    main()
