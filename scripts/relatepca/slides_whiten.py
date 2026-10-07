#!/usr/bin/env python3
"""Slide figure: whiten, don't just remove the pairs. The largen data set
(N=2000, 30% in families; slides_largen.py), k=20: the family share eta^2 of
every PC beyond the 4 ancestry axes for standard PCA, the Gram with the
related pairs and the diagonal imputed (imp_diagL) and whitening
(detect-white), as impute_vs_white_highn.py.

usage: slides_whiten.py <bfile> <out.pdf>
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from highn_test import family_eta2 as eta2  # noqa: E402
from sim import read_bed  # noqa: E402

plt.rcParams.update({"font.size": 11})
INK, MUTED = "#222222", "#666666"


def main():
    bfile, out = sys.argv[1], sys.argv[2]
    k, Ka = 20, 4
    fam, _, G = read_bed(bfile)
    G = G.astype(float)
    famid = np.array([int(r[0]) if r[0][0] != "U" else 0 for r in fam])
    N, M = G.shape
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L, S, _, r = I.lr_kin_fd_auto(H, B.TAU, D, I.cs_noise_edge(G), cand, rmax=k + 1)
    pp = I.lr_kin_pairs(H, D, L, S)
    AM = G @ G.T / M
    _, C = B.grm(G)
    off = (np.abs(S) > 1e-12) & ~np.eye(N, dtype=bool)
    Mi = np.where(off, L, H)
    np.fill_diagonal(Mi, np.diag(L))
    res = [("standard PCA", I.top_eig(C, k)[1], "#D55E00"),
           ("related pairs imputed", I.cs_pcs(Mi, k)[1], "#E69F00"),
           ("whitening", I.whitened_noise(AM, np.diag(AM) - np.diag(L), pp, k), "#0072B2")]
    x = np.arange(Ka + 1, k + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, U, col in res:
        e = np.array([eta2(U[:, j], famid) for j in range(Ka, k)])
        ax.plot(x, e, "-o", color=col, lw=1.6, ms=5)
        above = name != "standard PCA"  # whitening: the expected line is below it
        ax.text(x.mean(), e.max() + 0.03 if above else e.min() - 0.03, name, ha="center",
                va="bottom" if above else "top", fontsize=10, color=INK)
        print(name, "median", round(float(np.median(e)), 3), "max", round(float(e.max()), 2))
    # expected eta^2 with no family signal: the same family sizes on random unrelated people
    # (family labels permuted), on the whitening PCs
    rng = np.random.default_rng(1)
    Uw = res[-1][1]
    e0 = np.array([np.mean([eta2(Uw[:, j], rng.permutation(famid)) for _ in range(100)]) for j in range(Ka, k)])
    ax.plot(x, e0, "--", color=MUTED, lw=1.4)
    ax.text(x.mean(), e0.max() - 0.03, "expected if families were unrelated", ha="center",
            va="top", fontsize=10, color=MUTED)
    print("null median", round(float(np.median(e0)), 3))
    ax.set_xticks(x)
    ax.set_xlabel("PC (beyond the 4 ancestry axes)", color=MUTED)
    ax.set_ylabel("family share $\\eta^2$", color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_xlim(Ka + 0.5, k + 0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "N", N, "rank", r, "pairs", len(pp))


if __name__ == "__main__":
    main()
