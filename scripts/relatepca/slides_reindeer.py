#!/usr/bin/env python3
"""Slide figures for the Greenland reindeer panel (111 animals, 12 sampling
sites, chromosomes 1-3, MAF >= 0.05, no missing genotypes): PC1 vs PC2 of
standard PCAone next to aarobust-kin.

  all    all 111; stars: related samples, removed by KING (plink2 --king-cutoff 0.0354)
  unrel  the 73 KING-unrelated; aarobust-diag (--impute-diag: only the diagonal imputed)
  high   all 111, PC3 vs PC4 and PC5 vs PC6 (rows: standard, aarobust-kin)
  unrelhigh  the same for the 73 unrelated
  three  all, rows PC1-2, PC3-4, PC5-6; columns standard, aarobust-kin, dwg
  four   columns standard, aarobust-kin, aarobust-kin --impute-diag (aad_all), dwg; rows PC1-2 to
         PC9-10; legend above
  refit  experimental: columns aarobust-kin+diag, its rank-limited refit (refit_all), the refit
         whitened (refitw_all; reindeer_refit.py), dwg; rows PC1-2 to PC5-6; legend above
  hap    experimental: columns aarobust-kin+diag, haploid aarobust (hapaar_all), dwg, haploid dwg
         (hapdwg_all; reindeer_hap.py); rows PC1-2 to PC5-6; legend above

usage: slides_reindeer.py <dir> all|unrel|high|unrelhigh <out.pdf>
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"
# 12 sites: distinct hues, and the marker shape alternates so that sites with
# similar hues still differ
COLS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9",
        "#222222", "#999933", "#882255", "#44AA99", "#AA4499", "#332288"]
MARKS = ["o", "s", "^", "D"]


def read_fam(f):
    """sample IDs and populations; the population is whichever of the first two
    columns has fewer distinct values (the course files put it second, the new
    data first)"""
    rows = [l.split() for l in open(f)]
    a, b = [r[0] for r in rows], [r[1] for r in rows]
    return (a, b) if len(set(b)) < len(set(a)) else (b, a)


def fam_of(d, run):
    """the PLINK prefix the run read (-b in its log)"""
    for l in open(os.path.join(d, run + ".log")):
        w = l.split()
        if "-b" in w:
            return os.path.join(d, w[w.index("-b") + 1] + ".fam")
    raise SystemExit(f"no -b in {run}.log")


def find(d, name):
    """name in d, else in its parent (the KING list of the LD-pruned runs)"""
    f = os.path.join(d, name)
    return f if os.path.exists(f) else os.path.join(d, "..", name)


def read_vecs(f):
    return np.loadtxt(f)[:, :10]


def align(ref, x):
    """flip the sign of each column of x to agree with ref"""
    s = np.sign(np.sum(ref * x, axis=0))
    s[s == 0] = 1
    return x * s


def panel(a, pcs, pops, order, hollow, title, x=0, y=1, size=1.0):
    # white outlines keep overlapping points apart; related samples are stars
    # in their population's colour, drawn on top
    for i, p in enumerate(order):
        k = np.array([q == p for q in pops])
        a.scatter(pcs[k & ~hollow, x], pcs[k & ~hollow, y], s=70 * size, marker=MARKS[i % 4], lw=0.6,
                  facecolors=COLS[i], edgecolors="white", zorder=3)
    for i, p in enumerate(order):
        k = np.array([q == p for q in pops]) & hollow
        if k.any():
            a.scatter(pcs[k, x], pcs[k, y], s=230 * size, marker="*", lw=0.7,
                      facecolors=COLS[i], edgecolors="white", zorder=5)
    a.axhline(0, color=GRID, lw=0.8, ls="--", zorder=1)
    a.axvline(0, color=GRID, lw=0.8, ls="--", zorder=1)
    a.set_xlabel(f"PC{x + 1}")
    a.set_ylabel(f"PC{y + 1}")
    a.set_title(title, fontsize=plt.rcParams["font.size"] + 0.5, color=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def main():
    d, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "given":
        # one run (sys.argv[4]) on its own, PC1 vs PC2: the check against a plot made elsewhere
        run = sys.argv[4]
        ids, pops = read_fam(fam_of(d, run))
        X = read_vecs(os.path.join(d, run + ".eigvecs"))
        south = np.array([p == "South_Greenland" for p in pops])
        if X[south, 0].mean() > 0:
            X[:, 0] *= -1
        order = sorted(set(pops))
        fig, a = plt.subplots(figsize=(6.4, 4.5))
        panel(a, X, pops, order, np.zeros(len(ids), bool), f"standard PCAone (N = {len(ids)})")
        h = [Line2D([], [], ls="", marker=MARKS[i % 4], ms=7.5, mfc=COLS[i], mec="white", mew=0.6,
                    label=p.replace("_", " ")) for i, p in enumerate(order)]
        fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False, fontsize=11,
                   handletextpad=0.3, labelspacing=0.45)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        return
    if mode in ("all", "high", "three", "four", "refit", "hap"):
        std, aar = "std_all", "aar_all"
        ids, pops = read_fam(fam_of(d, std))
        removed = {w for l in list(open(find(d, "king.king.cutoff.out.id")))[1:] for w in l.split()}
        hollow = np.array([i in removed for i in ids])
        titles = (f"standard PCAone (N = {len(ids)})", f"aarobust-kin (N = {len(ids)})")
    else:
        std, aar = "std_unrel", "aar_unrel"
        ids, pops = read_fam(fam_of(d, std))
        hollow = np.zeros(len(ids), bool)
        titles = (f"standard PCAone (N = {len(ids)})", "aarobust-diag")
    S = read_vecs(os.path.join(d, std + ".eigvecs"))
    # the course plot's orientation: South Greenland on the left of PC1
    south = np.array([p == "South_Greenland" for p in pops])
    if S[south, 0].mean() > 0:
        S[:, 0] *= -1
    R = align(S, read_vecs(os.path.join(d, aar + ".eigvecs")))
    order = sorted(set(pops))

    def lims(a, pcs, x, y):
        """each panel on its own scale: the methods spread the PCs differently"""
        for j, setl in ((x, a.set_xlim), (y, a.set_ylim)):
            lo, hi = pcs[:, j].min(), pcs[:, j].max()
            setl(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))

    if mode == "three":
        # columns: standard, aarobust-kin, dwg; rows: PC1-2, PC3-4, PC5-6
        W = align(S, read_vecs(os.path.join(d, "dwg_all.eigvecs")))
        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(3, 3, figsize=(13.5, 11.5))
        for c, (pcs, name) in enumerate(zip((S, R, W), ("standard PCAone", "aarobust-kin", "dwg"))):
            for r, (x, y) in enumerate(((0, 1), (2, 3), (4, 5))):
                panel(ax[r, c], pcs, pops, order, hollow, name if r == 0 else "", x, y, size=1.5)
                lims(ax[r, c], pcs, x, y)
        C = np.abs(np.corrcoef(S[:, :6].T, W[:, :6].T)[:6, 6:])
        print("|cor(std PC i, dwg PC j)|:\n", np.round(C, 3))
    elif mode == "four":
        # columns: standard, aarobust-kin, aarobust-kin+diag, dwg; rows: PC1-2, ..., PC9-10
        D = align(S, read_vecs(os.path.join(d, "aad_all.eigvecs")))
        W = align(S, read_vecs(os.path.join(d, "dwg_all.eigvecs")))
        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(5, 4, figsize=(18, 20.4))
        for c, (pcs, name) in enumerate(zip((S, R, D, W), ("standard PCAone", "aarobust-kin", "aarobust-kin+diag",
                                                          "dwg"))):
            for r, (x, y) in enumerate(((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))):
                panel(ax[r, c], pcs, pops, order, hollow, name if r == 0 else "", x, y, size=1.5)
                lims(ax[r, c], pcs, x, y)
        for nm, X in (("aarobust-kin+diag", D), ("dwg", W)):
            print(f"|cor(std PC i, {nm} PC j)|:\n", np.round(np.abs(np.corrcoef(S[:, :6].T, X[:, :6].T)[:6, 6:]), 3))
    elif mode in ("refit", "hap"):
        runs = ((("aad_all", "aarobust-kin+diag"), ("refit_all", "refit"), ("refitw_all", "refit, whitened"),
                 ("dwg_all", "dwg")) if mode == "refit" else
                (("aad_all", "aarobust-kin+diag"), ("hapaar_all", "haploid aarobust"), ("dwg_all", "dwg"),
                 ("hapdwg_all", "haploid dwg")))
        cols = [(run, align(S, read_vecs(os.path.join(d, run + ".eigvecs"))), name) for run, name in runs]
        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(3, 4, figsize=(18, 12.6))
        for c, (_, pcs, name) in enumerate(cols):
            for r, (x, y) in enumerate(((0, 1), (2, 3), (4, 5))):
                panel(ax[r, c], pcs, pops, order, hollow, name if r == 0 else "", x, y, size=1.5)
                lims(ax[r, c], pcs, x, y)
    elif mode in ("high", "unrelhigh"):
        plt.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots(2, 2, figsize=(11.0, 6.9))
        for r, (pcs, name) in enumerate(zip((S, R), ("standard PCAone", titles[1].split(" (")[0]))):
            for c, (x, y) in enumerate(((2, 3), (4, 5))):
                panel(ax[r, c], pcs, pops, order, hollow, name, x, y, size=1.6)
                lims(ax[r, c], pcs, x, y)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.5))
        for a, pcs, t in zip(ax, (S, R), titles):
            panel(a, pcs, pops, order, hollow, t)
            lims(a, pcs, 0, 1)
    h = [Line2D([], [], ls="", marker=MARKS[i % 4], ms=7.5, mfc=COLS[i], mec="white", mew=0.6, label=p.replace("_", " "))
         for i, p in enumerate(order)]
    if hollow.any():
        h.append(Line2D([], [], ls="", marker="*", ms=13, mfc=MUTED, mec="white", mew=0.7,
                        label="related sample"))
    if mode in ("four", "refit", "hap"):
        # above the panels, so that four fit in a row
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False, fontsize=17,
                   handletextpad=0.2, columnspacing=1.0)
    else:
        fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False,
                   fontsize={"three": 17, "high": 14, "unrelhigh": 14}.get(mode, 11), handletextpad=0.3,
                   labelspacing=0.45)
        fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    C = np.abs(np.corrcoef(S[:, :6].T, R[:, :6].T)[:6, 6:])
    print("|cor(std PC i, aar PC j)|:\n", np.round(C, 3))


if __name__ == "__main__":
    main()
