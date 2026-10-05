#!/usr/bin/env python3
"""Graphical abstract of dwg as implemented in PCAone (Robust.cpp), drawn from
the admixTjeck2 example of methods_figs.py (10 per population + 8 relatives):

  genotypes -> GRM A (SNP-standardised, not centred) -> A = L + S
  (rank r from the noise edge; S = the diagonal + the related pairs, kinship
  scaled by the fitted noise v; L = A - S, its diagonal and pair entries
  imputed by the top r eigenvectors) -> Sigma = v^1/2 (I + 2 Psi) v^1/2 ->
  whitened Sigma^-1/2 A Sigma^-1/2 -> centred by projecting out Sigma^-1/2 1
  (the mean direction) -> PCs = top k eigenvectors, mapped back by Sigma^1/2.

Candidates here: the KING screen (the evalAdmix + k0 screen adds nothing on
this example).

usage: dwg_figs.py <admixTjeck2 bfile> <out.pdf>
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import dwg_loc as DL  # noqa: E402
import illustrate as I  # noqa: E402
import methods_figs as F  # noqa: E402
from methods_figs import INK, MUTED  # noqa: E402

K = 3


def dwg_centred(A, v, pairs, k, floor=0.1):
    """PCs of the whitened Gram, centred by projecting out Sigma^-1/2 1"""
    N = len(A)
    d = np.sqrt(np.maximum(v, 1e-12))
    R_ = np.eye(N)
    for i, j, phi in pairs:
        R_[i, j] = R_[j, i] = 2 * phi
    Sig = d[:, None] * R_ * d[None, :]
    ev, Q = np.linalg.eigh(Sig)
    ev = np.maximum(ev, floor * d.min() ** 2)
    Wm, Wh = (Q / np.sqrt(ev)) @ Q.T, (Q * np.sqrt(ev)) @ Q.T
    Mw = Wm @ A @ Wm
    u = Wm @ np.ones(N)
    u /= np.linalg.norm(u)
    P = np.eye(N) - np.outer(u, u)
    Mc = P @ Mw @ P
    _, V = I.top_eig(Mc, k)
    return Wh @ V, Sig, Mw, Mc


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
    A, f, w, _ = W.grm_scaled_uncentred(G)
    L, S, r, pairs, _ = DL.fit_loc(A, B.TAU, cand, W.noise_edge(f, w, N), K + 1, 4, kin_check=True)
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    U, Sig, Mw, Mc = dwg_centred(A, v, pairs, K)
    Limp = A - S  # L as eigendecomposed: A with the diagonal and the pair entries imputed
    Zt = np.c_[U, np.ones(N)]
    Bc, *_ = np.linalg.lstsq(Zt[~rel], T[~rel], rcond=None)
    r2 = 1 - ((T[~rel] - (Zt @ Bc)[~rel]) ** 2).sum(0) / ((T[~rel] - T[~rel].mean(0)) ** 2).sum(0)

    fig = plt.figure(figsize=(9.6, 5.8))
    top = F.panel_axes(fig, [1.0, 1, 1, 1], 0.85, bottom=0.56, height=0.31)
    bot = F.panel_axes(fig, [1, 1, 1, 1.05], 0.85, bottom=0.10, height=0.29)
    F.geno(top[0], G[o][:, :160], po)
    F.heat(top[1], P(A), "GRM $A$\n(scaled, not centred)", pop=po, centre=True)
    F.heat(top[2], P(Limp), f"$L$: diag. + pairs imputed\nby top {r - 1} PCs (rank {r})", pop=po, centre=True)
    F.heat(top[3], P(S), "$S$: diagonal + pairs\n$\\hat\\phi>\\tau$ (scaled by $v$)", sparse=True, pop=po,
           diag_dots=True)
    F.heat(bot[0], P(Sig), "$\\Sigma$: noise $v$\n+ families", sparse=True, pop=po, diag_dots=True)
    Mwo = P(Mw)
    bot[1].imshow(Mwo, cmap="Reds", vmin=Mwo.min(), vmax=np.percentile(Mwo, 99), interpolation="nearest")
    for i, p in enumerate(po):
        bot[1].add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=F.PCOL[p], clip_on=False, lw=0))
    bot[1].set_xticks([])
    bot[1].set_yticks([])
    bot[1].set_title("whitened $\\Sigma^{-1/2}A\\,\\Sigma^{-1/2}$\n(not centred: all positive)", color=INK)
    F.heat(bot[2], P(Mc), "centred: $\\Sigma^{-1/2}\\mathbf{1}$\nprojected out", pop=po)
    F.pcs(bot[3], U[o], po, ro, f"PCs: PC2 vs PC3\nMXL axis $R^2$ = {r2[2]:.2f}", 1, 2)
    fig.canvas.draw()
    F.arrow(fig, top[0], top[1], "standardise,\nnot centred")
    F.arrow(fig, top[1], top[2], "fit\n$A = L + S$")
    F.plus(fig, top[2], top[3])
    # elbow from S down to Sigma
    b3, b4 = top[3].get_position(), bot[0].get_position()
    xs, xb, ym = (b3.x0 + b3.x1) / 2, (b4.x0 + b4.x1) / 2, b3.y0 - 0.035
    fig.add_artist(plt.Line2D([xs, xs, xb], [b3.y0 - 0.01, ym, ym], transform=fig.transFigure, color=MUTED, lw=1.0))
    fig.add_artist(FancyArrowPatch((xb, ym), (xb, b4.y1 + 0.085), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color=MUTED, lw=1.0))
    fig.text((xs + xb) / 2, ym + 0.008, "noise $v$ = diagonal of $S$; families from the pairs", ha="center",
             va="bottom", fontsize=8, color=MUTED)
    F.arrow(fig, bot[0], bot[1], "whiten")
    F.arrow(fig, bot[1], bot[2], "centre")
    F.arrow(fig, bot[2], bot[3], "top $k$,\n$\\times\\Sigma^{1/2}$")
    fig.text(0.5, 0.965, "dwg: detect related pairs on the GRM, then whiten and centre", ha="center", fontsize=11,
             color=INK, weight="bold")
    F.legend_row(fig, 0.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("rank", r, "pairs", len(pairs), "R2", np.round(r2, 3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
