#!/usr/bin/env python3
"""Graphical abstract of dwg (detect-white on the GRM scale), drawn from the
admixTjeck2 example of methods_figs.py."""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import methods_figs as F  # noqa: E402
from methods_figs import INK  # noqa: E402


def main(bfile, out):
    ds, T = F.dataset(bfile)
    G = ds["G"].astype(float)
    N = len(G)
    n0 = ds["n0"]
    pop = F.family_pop(ds)
    rel = np.arange(N) >= n0
    o = F.order_rows(pop, n0, ds["Kped"])
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    po, ro = pop[o], rel[o]
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    # centred GRM, restored uncentred Gram, fit, whitening
    A, f, w, Gs = W.grm_scaled_uncentred(G)
    mu = 2 * f / np.sqrt(w)
    Xc = Gs / np.sqrt(w) - mu
    M = len(f)
    C = Xc @ Xc.T / M
    u = Xc @ mu / M
    As = C + u[:, None] + u[None, :] + (mu @ mu) / M
    assert np.abs(As - A).max() < 1e-8
    L, S, r, pairs = W.fit(As, B.TAU, cand, W.noise_edge(f, w, N), 4, None)
    U, _, _ = W.dwg(G, 3, B.TAU, cand, "v")
    v = np.maximum(np.diag(As) - np.diag(L), 1e-6)
    Sig = F.sigma(v, pairs, N)

    fig = plt.figure(figsize=(11.5, 3.0))
    ax = F.panel_axes(fig, [1.0, 1, 1, 1, 1, 1.05], 0.85)
    F.heat(ax[0], P(C), "centred GRM $C$\n(diagonal grey)", pop=po, mask_diag=True)
    F.heat(ax[1], P(As - As[~np.eye(N, dtype=bool)].mean()), "uncentred $A_s$, restored\n(overall mean removed)",
           pop=po, mask_diag=True)
    F.heat(ax[2], P(L), f"$L$: rank {r}, diag. free", pop=po, centre=True)
    Sd = P(S).copy()
    np.fill_diagonal(Sd, 0)
    F.heat(ax[3], Sd, "$S$: pairs, $\\hat\\phi>\\tau$\n(scaled by fitted $v$)", sparse=True, pop=po)
    F.heat(ax[4], P(Sig), "$\\Sigma$ (families)", sparse=True, pop=po, diag_dots=True)
    F.pcs(ax[5], U[o], po, ro, "PCs (whitened)", 1, 2)
    fig.canvas.draw()
    F.arrow(fig, ax[0], ax[1], "$+u1'+1u'$\n$+\\mu'\\mu$")
    F.arrow(fig, ax[1], ax[2], "rank $r$\n+ kinship")
    F.plus(fig, ax[2], ax[3])
    F.arrow(fig, ax[3], ax[4], "families,\nnoise $v$")
    F.arrow(fig, ax[4], ax[5], "whiten,\n$\\times\\Sigma^{1/2}$")
    fig.text(0.5, 0.96, "dwg: detect-white on the GRM scale, no heterozygosity (prototype)", ha="center",
             fontsize=11, color=INK, weight="bold")
    F.legend_row(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("rank", r, "pairs", len(pairs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
