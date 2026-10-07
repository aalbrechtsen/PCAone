#!/usr/bin/env python3
"""Slide figure: HAPNEST N = 5000 (hapnest_test.py, seed 1). True realized
kinship of the 2nd-degree pairs (pedigree kinship 0.125), from the founder-
haplotype labels carried through the simulated meioses (simulate(track=True),
realized_ibd), stacked by relationship; left of tau = missed by dwg. The
PC-adjusted (PC-Relate style) estimate is printed for comparison.

usage: slides_hapnest_missed.py <pool prefix> <run prefix> <seed> <N> <out.pdf>
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
from hapnest_test import realized_ibd, simulate  # noqa: E402
from sim import read_bed  # noqa: E402

plt.rcParams.update({"font.size": 12})
INK, MUTED = "#222222", "#666666"


def main():
    pool, pre, seed, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    fam, bim, G = read_bed(pool)
    info = {l.split()[0]: l.split()[1:] for l in list(open(os.path.join(os.path.dirname(pool), "individuals.tsv")))[1:]}
    ids = [r[1] for r in fam]
    anc = np.array([info[x][0] for x in ids])
    sset = np.array([info[x][1] for x in ids])
    pidx = np.where(sset == "pool")[0]
    Gs, famid, ftype, truth, ancs, labs = simulate(G.astype(np.int8)[pidx], anc[pidx], bim, N, 0.3, seed,
                                                   track=True)
    keep = Gs.min(0) != Gs.max(0)
    Gs = Gs[:, keep].astype(float)
    U = np.loadtxt(pre + "_dwg_k5.eigvecs")
    det = {tuple(sorted((int(l.split()[0][1:]), int(l.split()[1][1:])))) for l in list(open(pre + "_dwg_k5.relpairs"))[1:]}
    tp = [p for p, k in truth.items() if k >= B.TAU]
    V = np.c_[U, np.ones(N)]
    Pi = np.clip(0.5 * V @ np.linalg.lstsq(V, Gs, rcond=None)[0], 1e-3, 1 - 1e-3)
    phi, phi_pc, found, ped = [], [], [], []
    for i, j in tp:
        ri, rj = Gs[i] - 2 * Pi[i], Gs[j] - 2 * Pi[j]
        phi_pc.append((ri * rj).sum() / (4 * np.sqrt(Pi[i] * (1 - Pi[i]) * Pi[j] * (1 - Pi[j])).sum()))
        phi.append(realized_ibd(labs[i], labs[j], keep)[3])
        found.append((i, j) in det)
        ped.append(truth[(i, j)])
    phi, phi_pc, found, ped = np.array(phi), np.array(phi_pc), np.array(found), np.array(ped)
    print("true vs PC-adjusted realized kinship: corr", round(float(np.corrcoef(phi, phi_pc)[0, 1]), 3),
          "missed: true", np.round(np.sort(phi[~found]), 3))
    print("found with true realized kinship < tau:", int((found & (phi < B.TAU)).sum()),
          "missed with true realized kinship >= tau:", int((~found & (phi >= B.TAU)).sum()))
    print("true pairs", len(tp), "missed", int((~found).sum()), "missed phi", np.round(np.sort(phi[~found]), 3),
          "missed pedigree kinship", np.unique(np.round(ped[~found], 4), return_counts=True))
    sec = np.isclose(ped, 0.125)  # all missed pairs are 2nd degree: show the 2nd-degree pairs
    # relationship of each 2nd-degree pair: "pedigree" families are [grandparent, c1, c2, g1, g2]
    # (same ancestry); halfsibx / grandx / avuncx are the cross-ancestry families
    first = {}
    for x, f in enumerate(famid):
        first.setdefault(f, x)
    rel = []
    for (i, j) in tp:
        t = ftype[i]
        if t == "pedigree":
            rel.append("grandparent" if first[famid[i]] in (i, j) else "aunt/uncle")
        else:
            rel.append({"halfsibx": "half sibs, cross-ancestry", "grandx": "grandparent, cross-ancestry",
                        "avuncx": "aunt/uncle, cross-ancestry"}.get(t, t))
    rel = np.array(rel)
    cats = [("grandparent", "#cccccc"), ("aunt/uncle", "#8c8c8c"), ("half sibs, cross-ancestry", "#cbc9e2"),
            ("grandparent, cross-ancestry", "#9e9ac8"), ("aunt/uncle, cross-ancestry", "#54278f")]
    for c, _ in cats:
        m = sec & (rel == c)
        print(f"{c:30s} pairs {m.sum():3d} missed {(m & ~found).sum():2d}")
    print("other 2nd-degree labels", set(rel[sec]) - {c for c, _ in cats})
    # found and missed 2nd-degree pairs in separate panels, on the true realized kinship
    fig, axs = plt.subplots(2, 1, figsize=(8, 3.9), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
    bins = np.linspace(0.04, 0.22, 46)
    for ax, sel, col, lab in [(axs[0], sec & found, "#BBBBBB", "found"), (axs[1], sec & ~found, "#6a51a3", "missed")]:
        ax.hist(phi[sel], bins=bins, color=col)
        ax.axvline(B.TAU, color=INK, ls="--", lw=1.2)
        ax.axvline(0.125, color=MUTED, ls=":", lw=1.2)
        ax.text(0.215, ax.get_ylim()[1] * 0.92, f"{lab} by dwg ({sel.sum()} of {sec.sum()})", ha="right", va="top",
                fontsize=11, color=INK)
        ax.set_ylabel("pairs", color=MUTED)
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)
    top = axs[0].get_ylim()[1]
    axs[0].text(B.TAU - 0.002, top * 0.97, "$\\tau = 2^{-3.5}$\n(cutoff on the\nestimate)", fontsize=9.5,
                color=INK, va="top", ha="right")
    axs[0].text(0.127, top * 0.97, "expected", fontsize=9.5, color=MUTED, va="top")
    axs[1].set_xlabel("true realized kinship of the 2nd-degree pairs (IBD from the simulation)", color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
