#!/usr/bin/env python3
"""Per-animal noise for a reindeer data set, grouped by site: the diagonal of
the standard GRM (what standard PCA sees) and the noise v_i that dwg fits
(<run>.noise; dwg weights each animal by 1/sqrt(v_i)). Both relative to the
median over all animals; animals more than 2-fold from their site's median
noise are labelled. Bottom: the inbreeding coefficient F of PCAone --inbreed 2 with 4
PCs (<dir>/pc4inb.inbred: expected heterozygosity from each animal's own
allele frequencies, i.e. corrected for the structure of the top 4 PCs), and
the same F from the top 6 PCs of aarobust-kin --impute-diag (<dir>/aad6inb.inbred,
inbreed_from_pcs.py), and F from the PCP smoother, which needs no number of PCs
(<dir>/smooth_pcp.inbred, inbreed_smooth.py); animals with F < -0.1 are labelled. Stars: related
samples (KING list as slides_reindeer.py).

usage: noise_reindeer.py <dir> <out.pdf>     (dir holds rd_full.*, pc4inb.inbred and full/dwg_all.noise)
"""
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import read_bed  # noqa: E402
from slides_reindeer import COLS, MARKS, find, read_fam  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"


def grm_diag(prefix, block=100000):
    """diagonal of the standardised GRM, sum_j (g_ij - 2f_j)^2 / (2f_j(1-f_j)) / M"""
    _, _, G = read_bed(prefix)
    s, m = np.zeros(G.shape[0]), 0
    for a in range(0, G.shape[1], block):
        g = G[:, a:a + block].astype(np.float32)
        f = g.mean(0) / 2
        w = 2 * f * (1 - f)
        ok = w > 1e-6
        s += (((g[:, ok] - 2 * f[ok]) ** 2) / w[ok]).sum(1)
        m += ok.sum()
    return s / m


def main():
    d, out = sys.argv[1], sys.argv[2]
    ids, pops = read_fam(os.path.join(d, "rd_full.fam"))
    pops = np.array(pops)
    diag = grm_diag(os.path.join(d, "rd_full"))
    v = np.loadtxt(os.path.join(d, "full", "dwg_all.noise"))
    removed = {w for l in list(open(find(d, "king.king.cutoff.out.id")))[1:] for w in l.split()}
    rel = np.array([i in removed for i in ids])
    def read_F(name):
        het = {l.split()[1]: float(l.split()[-1]) for l in list(open(os.path.join(d, name)))[1:]}
        return np.array([het[i] for i in ids])

    F = read_F("pc4inb.inbred")
    F6 = read_F("aad6inb.inbred")
    Fp = read_F("smooth_pcp.inbred")
    order = sorted(set(pops))
    idx = np.concatenate([np.flatnonzero(pops == p) for p in order])
    x = np.arange(len(idx))
    fig, ax = plt.subplots(5, 1, figsize=(12, 15.2), sharex=True)
    for a, val, lab in ((ax[0], diag, "GRM diagonal\n(standard PCA)"), (ax[1], v, "dwg noise $v_i$\n(weight $1/\\sqrt{v_i}$)"),
                        (ax[2], F, "inbreeding F\n(PCAone, 4 PCs)"),
                        (ax[3], F6, "inbreeding F\n(aarobust-kin+diag,\n6 PCs)"),
                        (ax[4], Fp, "inbreeding F\n(PCP smoother,\nno k)")):
        isf = lab.startswith("inbreeding")
        r = val if isf else val / np.median(val)
        for c, p in enumerate(order):
            k = np.flatnonzero(pops[idx] == p)
            sel = idx[k]
            a.scatter(x[k][~rel[sel]], r[sel][~rel[sel]], s=60, marker=MARKS[c % 4], facecolors=COLS[c],
                      edgecolors="white", lw=0.6, zorder=3)
            a.scatter(x[k][rel[sel]], r[sel][rel[sel]], s=190, marker="*", facecolors=COLS[c],
                      edgecolors="white", lw=0.7, zorder=4)
            a.hlines(np.median(r[sel]), x[k].min() - 0.4, x[k].max() + 0.4, color=COLS[c], lw=1.5, zorder=2)
            if isf:
                # strongly negative F
                for j, i in zip(x[k], sel):
                    if r[i] < -0.1:
                        a.annotate(ids[i], (j, r[i]), xytext=(4, -2), textcoords="offset points", fontsize=10,
                                   color=INK)
            if lab.startswith("dwg"):
                # animals far from their site
                for j, i in zip(x[k], sel):
                    if r[i] > 2 * np.median(r[sel]) or r[i] < 0.5 * np.median(r[sel]):
                        a.annotate(ids[i], (j, r[i]), xytext=(4, 4), textcoords="offset points", fontsize=10,
                                   color=INK)
        a.axhline(0 if isf else 1, color=MUTED, lw=0.9, ls="--", zorder=1)
        if not isf:
            a.set_yscale("log", base=2)
            a.set_yticks([0.25, 0.5, 1, 2, 4, 8], ["1/4", "1/2", "1", "2", "4", "8"])
            a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_ylabel(lab if isf else lab + "\n/ median", fontsize=12)
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
    for p in order:
        k = pops == p
        print(f"{p:24s} diag {np.median(diag[k] / np.median(diag)):.2f}  v {np.median(v[k] / np.median(v)):.2f}  F {np.median(F[k]):.3f}")
    print("cor(F PCP, log v)", round(float(np.corrcoef(Fp, np.log(v))[0, 1]), 3))
    print("cor(F 6 PCs, log v)", round(float(np.corrcoef(F6, np.log(v))[0, 1]), 3))
    print("cor(F, log v)", round(float(np.corrcoef(F, np.log(v))[0, 1]), 3), "cor(F, log diag)",
          round(float(np.corrcoef(F, np.log(diag))[0, 1]), 3))


if __name__ == "__main__":
    main()
