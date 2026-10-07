#!/usr/bin/env python3
"""Slide figures: one graphical abstract per small-N method (no relatives,
5 per population, admixTjeck2), in the style of the methods document:
genotypes -> matrix -> L + S (S = the diagonal) -> PCs.

  ga_small_fit.pdf        uncentred GRM, rank from the noise edge, diagonal free
  ga_small_aarobust.pdf   aarobust-kin: robust PCA of the GRM, diagonal unobserved
  ga_small_dw.pdf         detect-white: Chen & Storey matrix, diagonal free, whitened
  ga_small_dwg.pdf        dwg: uncentred GRM, diagonal free, whitened by the noise v

usage: slides_methods.py <admixTjeck2 bfile> <outdir>
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import dwg_loc as DL  # noqa: E402
import illustrate as I  # noqa: E402
import methods_figs as F  # noqa: E402
import slides_hook as S  # noqa: E402
from methods_figs import INK, MUTED, PCOL, POPS  # noqa: E402

K = 3


def diag_heat(ax, d, title, vmax, pop):
    """a matrix that is only its diagonal"""
    N = len(d)
    M = np.full((N, N), np.nan)
    M[np.diag_indices(N)] = d
    cm = plt.get_cmap("Reds").copy()
    cm.set_bad("white")
    ax.imshow(M, cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
    for i, p in enumerate(pop):
        ax.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#BBBBBB")
    ax.set_title(title, color=INK)


def imputed(M, Lr):
    """the matrix that is eigendecomposed: M off the diagonal, the diagonal
    predicted by the rank-r fit Lr (so M = L + S exactly, S = diag(M - Lr))"""
    Mi = M.copy()
    np.fill_diagonal(Mi, np.diag(Lr))
    return Mi


def r2_of(U, T):
    Z = np.c_[U, np.ones(len(U))]
    Bc, *_ = np.linalg.lstsq(Z, T, rcond=None)
    return 1 - ((T - Z @ Bc) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)


def draw(G, pop, T, mat, mat_title, mat_kw, L, L_title, s_diag, s_title, U, title, out, steps, sigma=None,
         shift=1, raw=None, sigma_title="$\\Sigma$ = diag($v$)", centre=False, eig=None):
    """shift=1 for uncentred matrices: their PC1 is the mean, so the ancestry
    PCs are labelled PC1+1, PC2+1, ... (PCk+1 corresponds to standard PCk)"""
    N = len(G)
    if sigma is None and eig is None:
        # one row, tight gaps (bigger panels at slide width)
        fig = plt.figure(figsize=(11.5, 3.0))
        ax = F.panel_axes(fig, [1.0, 1, 1, 1, 1.05], 0.62, left=0.01, bottom=0.15, height=0.66)
    elif sigma is None:
        # two rows: genotypes -> matrix -> L + S, then eigenvalues -> PCs
        fig = plt.figure(figsize=(12, 4.8))
        ax = F.panel_axes(fig, [1.0, 1, 1, 1], 1.1, left=0.06, bottom=0.50, height=0.30)
        if eig is not None:
            ax += F.panel_axes(fig, [1.3, 1.05], 1.1, left=0.25, bottom=0.12, height=0.25)
        else:
            ax += F.panel_axes(fig, [1.05], 1.1, left=0.40, bottom=0.08, height=0.29)
    else:
        # two rows: genotypes -> matrix -> L + S, then Sigma -> whitened -> PCs
        fig = plt.figure(figsize=(9.2, 5.6))
        ax = F.panel_axes(fig, [1.0, 1, 1, 1], 0.85, bottom=0.56, height=0.32)
        bw = [1, 1, 1, 1.05] if centre else [1, 1, 1.05]
        ax += F.panel_axes(fig, bw, 0.85, left=0.015 if centre else 0.14, bottom=0.10, height=0.30)
    F.geno(ax[0], G[:, :160], pop)
    F.heat(ax[1], mat, mat_title, pop=pop, **mat_kw)
    # PCs of L: S + L, so L sits next to the arrow to the PCs; whitening uses
    # S (Sigma = S): L + S, so S sits next to the arrow to Sigma
    iL, iS = (3, 2) if sigma is None else (2, 3)
    F.heat(ax[iL], L, L_title, pop=pop, positive=mat_kw.get("positive", False),
           vmax=mat_kw.get("vmax"))
    diag_heat(ax[iS], s_diag, s_title, np.abs(s_diag).max(), pop)
    j = 4
    if eig is not None:
        lam, edge, r = eig
        x = np.arange(1, len(lam) + 1)
        ax[4].bar(x, lam, color=["#0072B2" if i < r else "#BBBBBB" for i in range(len(lam))])
        ax[4].axhline(edge, color="#D55E00", lw=1, ls="--")
        ax[4].text(len(lam) + 0.4, edge * 1.4, "noise edge", ha="right", fontsize=8, color="#D55E00")
        ax[4].set_yscale("log")
        ax[4].set_xticks(x)
        ax[4].tick_params(labelsize=7)
        ax[4].set_xlabel("eigenvalue", color=MUTED, fontsize=8)
        ax[4].set_title(f"rank {r}: eigenvalues above\nthe noise edge (blue)", color=INK)
        for sp in ["top", "right"]:
            ax[4].spines[sp].set_visible(False)
        j = 5
    if sigma is not None:
        diag_heat(ax[4], sigma, sigma_title, np.abs(sigma).max(), pop)
        Wm = raw / np.sqrt(np.outer(sigma, sigma))  # whitened: every entry A_ij / sqrt(v_i v_j)
        if centre:
            # dwg: the whitened matrix is not centred (all positive); project out Sigma^-1/2 1
            ax[5].imshow(Wm, cmap="Reds", vmin=Wm.min(), vmax=np.percentile(Wm, 99), interpolation="nearest")
            for i, p in enumerate(pop):
                ax[5].add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p],
                                              clip_on=False, lw=0))
            ax[5].set_xticks([])
            ax[5].set_yticks([])
            ax[5].set_title("whitened: $A_{ij}/\\sqrt{v_i v_j}$\n(not centered: all positive)", color=INK)
            u = 1 / np.sqrt(sigma)
            u /= np.linalg.norm(u)
            Pu = np.eye(N) - np.outer(u, u)
            F.heat(ax[6], Pu @ Wm @ Pu, "centered: $\\Sigma^{-1/2}\\mathbf{1}$\nprojected out", pop=pop)
            j = 7
        else:
            F.heat(ax[5], Wm, "whitened:\n$A_{ij}/\\sqrt{v_i v_j}$", pop=pop, positive=True)
            j = 6
    U = S.align_signs(U, T, np.arange(N))
    r2 = r2_of(U, T)
    lab = (lambda k: f"PC{k}+1") if shift else (lambda k: f"PC{k}")
    S.scatter(ax[j], U, pop, np.zeros(N, bool), f"PCs: {lab(2)} vs {lab(3)}\nMXL axis $R^2$ = {r2[2]:.2f}",
              a=1, b=2)
    ax[j].set_xlabel(lab(2), color=MUTED)
    ax[j].set_ylabel(lab(3), color=MUTED)
    for t in ax[j].texts:
        t.set_fontsize(9)
    ax[j].title.set_fontsize(9)
    ax[j].xaxis.label.set_fontsize(9)
    ax[j].yaxis.label.set_fontsize(9)
    for c in ax[j].collections:
        c.set_sizes([16])
    fig.canvas.draw()
    F.arrow(fig, ax[0], ax[1], steps[0])
    F.arrow(fig, ax[1], ax[2], steps[1])
    F.plus(fig, ax[2], ax[3])
    if sigma is not None:
        b3, b4 = ax[3].get_position(), ax[4].get_position()
        xs, xb, ym = (b3.x0 + b3.x1) / 2, (b4.x0 + b4.x1) / 2, b3.y0 - 0.035
        # elbow: down from S, left along the gap between the rows, down into Sigma
        fig.add_artist(plt.Line2D([xs, xs, xb], [b3.y0 - 0.01, ym, ym], transform=fig.transFigure, color=MUTED,
                                  lw=1.0))
        fig.add_artist(FancyArrowPatch((xb, ym), (xb, b4.y1 + 0.085), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=10, color=MUTED, lw=1.0))
        fig.text((xs + xb) / 2, ym + 0.008, steps[2].replace("\n", " "), ha="center", va="bottom", fontsize=8,
                 color=MUTED)
        F.arrow(fig, ax[4], ax[5], "$\\Sigma^{-1/2} A\\, \\Sigma^{-1/2}$")
        if centre:
            F.arrow(fig, ax[5], ax[6], "center")
            F.arrow(fig, ax[6], ax[7], steps[3])
        else:
            F.arrow(fig, ax[5], ax[6], steps[3])
    elif eig is None:
        F.arrow(fig, ax[3], ax[4], steps[2])
    else:
        # elbow from L down to the bottom row
        b3, b4 = ax[3].get_position(), ax[4].get_position()
        xs, xb, ym = (b3.x0 + b3.x1) / 2, (b4.x0 + b4.x1) / 2, b3.y0 - 0.035
        fig.add_artist(plt.Line2D([xs, xs, xb], [b3.y0 - 0.01, ym, ym], transform=fig.transFigure, color=MUTED,
                                  lw=1.0))
        fig.add_artist(FancyArrowPatch((xb, ym), (xb, b4.y1 + 0.085), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.0))
        fig.text((xs + xb) / 2, ym + 0.008, steps[2].replace("\n", " "), ha="center", va="bottom", fontsize=8,
                 color=MUTED)
        if eig is not None:
            F.arrow(fig, ax[4], ax[5], steps[3])
    if not (sigma is None and eig is None):  # one-row figures: the slide title says it
        fig.text(0.5, 0.96, title, ha="center", fontsize=11, color=INK, weight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "R2", np.round(r2, 3))


def masked_heat(ax, M, title, pop, cmap="RdBu_r", positive=False):
    """matrix with the diagonal shown as unknown (grey '?'); positive: an
    uncentred matrix, white -> red"""
    N = len(M)
    M = np.array(M, float)
    off = ~np.eye(N, dtype=bool)
    shown = M.copy()
    np.fill_diagonal(shown, np.nan)
    if positive:
        cm = plt.get_cmap("Reds").copy()
        lo, hi = M[off].min(), np.percentile(M[off], 99)
    else:
        cm = plt.get_cmap(cmap).copy()
        hi = np.percentile(np.abs(M[off]), 99)
        lo = -hi
    cm.set_bad("#d0d0d0")
    ax.imshow(shown, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    for i in range(N):
        ax.text(i, i, "?", ha="center", va="center", fontsize=6, color=INK)
    for i, p in enumerate(pop):
        ax.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK)


def impute_topr_fig(bfile, out, n=5, r=4, iters=500):
    """imputing the diagonal with the top r PCs (scaled, uncentred GRM)"""
    ds, _ = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    A, f, w, _ = W.grm_scaled_uncentred(G)
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    d = np.array([A[i, off[i]].mean() for i in range(N)])
    At = A.copy()
    for _ in range(iters):
        np.fill_diagonal(At, d)
        lam, V = np.linalg.eigh(At)
        Lr = (V[:, -r:] * lam[-r:]) @ V[:, -r:].T
        if np.max(np.abs(np.diag(Lr) - d)) < 1e-10:
            break
        d = np.diag(Lr).copy()
    np.fill_diagonal(At, d)
    lam = np.linalg.eigvalsh(At)[::-1]
    edge = W.noise_edge(f, w, N)
    fig = plt.figure(figsize=(12, 3.8))
    ax0 = fig.add_axes([0.03, 0.12, 0.22, 0.74])
    ax1 = fig.add_axes([0.36, 0.16, 0.27, 0.70])
    ax2 = fig.add_axes([0.72, 0.16, 0.27, 0.70])
    masked_heat(ax0, A, "GRM: diagonal unknown", pop, positive=True)
    x = np.arange(1, 11)
    ax1.bar(x, lam[:10], color=["#0072B2" if i < r else "#BBBBBB" for i in range(10)])
    ax1.axhline(edge, color="#D55E00", lw=1, ls="--")
    ax1.text(10.4, edge * 1.3, "noise edge", ha="right", fontsize=9, color="#D55E00")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xlabel("eigenvalue", color=MUTED)
    ax1.set_title(f"keep the top {r} (blue), drop the rest", color=INK)
    x2 = np.arange(N)
    ax2.scatter(x2, np.diag(A), s=30, color=[PCOL[p] for p in pop], edgecolor="white", lw=0.4, label="observed")
    ax2.scatter(x2, d, s=50, marker="_", color=INK, lw=1.6, label=f"imputed: diag. of top {r}")
    ax2.set_xticks([])
    ax2.set_xlabel("individuals (by population)", color=MUTED)
    ax2.set_ylabel("diagonal", color=MUTED)
    ax2.set_ylim(0, np.diag(A).max() * 1.2)
    ax2.set_title("the diagonal", color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (ax1, ax2):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "eig 1-6", np.round(lam[:6], 3), "edge", round(edge, 3))


def mushroom(ax, title=True):
    """a cartoon nuclear explosion"""
    from matplotlib.patches import Circle, Ellipse, Polygon
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Ellipse((0.5, 0.08), 0.95, 0.12, color="#8C510A", alpha=0.6))  # dust ring
    ax.add_patch(Polygon([[0.42, 0.1], [0.58, 0.1], [0.54, 0.55], [0.46, 0.55]], color="#E66101"))  # stem
    ax.add_patch(Polygon([[0.45, 0.1], [0.55, 0.1], [0.52, 0.5], [0.48, 0.5]], color="#FDB863"))
    ax.add_patch(Ellipse((0.5, 0.52), 0.34, 0.08, color="#B35806", alpha=0.8))  # collar
    for (x, y, r, c) in [(0.5, 0.72, 0.2, "#B35806"), (0.33, 0.68, 0.13, "#E66101"), (0.67, 0.68, 0.13, "#E66101"),
                         (0.42, 0.8, 0.14, "#F1A340"), (0.58, 0.8, 0.14, "#F1A340"), (0.5, 0.85, 0.12, "#FEE0B6"),
                         (0.5, 0.7, 0.1, "#FFF7BC")]:
        ax.add_patch(Circle((x, y), r, color=c))
    ax.text(0.5, 0.71, r"$\|L\|_*$", ha="center", va="center", fontsize=18, color=INK, weight="bold")
    for (x, y, rot) in [(0.08, 0.92, 20), (0.92, 0.9, -20)]:
        ax.text(x, y, "BOOM!", ha="center", va="center", fontsize=13, color="#D7191C", weight="bold", rotation=rot)
    if title:
        ax.set_title("the nuclear norm", color=INK)


def mushroom_fig(out):
    """the cartoon alone, for a corner of the aarobust-kin slides"""
    fig = plt.figure(figsize=(2.2, 2.0))
    mushroom(fig.add_axes([0, 0, 1, 1]), title=False)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)


def nuclear_fig(bfile, out, n=5):
    """imputing the diagonal with the nuclear norm (PCP, centred, scaled GRM)"""
    ds, _ = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    _, C = B.grm(G)
    N = len(C)
    Lp, _, _ = I.pcp_kin(C, B.TAU, True, cand=np.zeros((N, N), bool))
    nn = lambda M: np.abs(np.linalg.eigvalsh(M)).sum()  # noqa: E731
    ex = [("observed", np.diag(C)), ("half the observed", np.diag(C) / 2), ("zero", np.zeros(N)),
          ("PCP: the minimum", np.diag(Lp))]
    off = ~np.eye(N, dtype=bool)
    vmax = np.percentile(np.abs(C[off]), 99)
    fig = plt.figure(figsize=(12, 3.6))
    ax0 = fig.add_axes([0.0, 0.05, 0.22, 0.85])
    mushroom(ax0)
    fig.text(0.62, 0.95, r"the same GRM off the diagonal, four diagonals: $\|L\|_* = \sum_k |\lambda_k|$",
             ha="center", fontsize=11, color=INK)
    for i, (name, d) in enumerate(ex):
        ax = fig.add_axes([0.26 + 0.185 * i, 0.16, 0.16, 0.62])
        M = C.copy()
        np.fill_diagonal(M, d)
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        for j, p in enumerate(pop):
            ax.add_patch(plt.Rectangle((-0.07 * N - 0.5, j - 0.5), 0.05 * N, 1, color=PCOL[p], clip_on=False, lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        best = i == len(ex) - 1
        ax.set_title(f"diagonal: {name}", color=INK, fontsize=9, weight="bold" if best else "normal")
        ax.set_xlabel(f"$\\|L\\|_* = {nn(M):.1f}$", fontsize=12, color="#0072B2" if best else INK)
        for sp in ax.spines.values():
            sp.set_edgecolor("#0072B2" if best else "#BBBBBB")
            sp.set_linewidth(2.5 if best else 0.8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, {name: round(nn(np.where(np.eye(N, dtype=bool), np.diag(d), C)), 2) for name, d in ex})


def whitening_fig(bfile, out, n=5):
    """what whitening does (no relatives): every entry A_ij is divided by
    sqrt(v_i v_j), so every person's noise on the diagonal becomes 1"""
    ds, _ = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    A, f, w, _ = W.grm_scaled_uncentred(G)
    N = len(A)
    L, _, _, _, _ = DL.fit_loc(A, B.TAU, np.zeros((N, N), bool), W.noise_edge(f, w, N), K + 1, 4, kin_check=True)
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    Wm = A / np.sqrt(np.outer(v, v))
    fig = plt.figure(figsize=(12, 3.8))
    ax0 = fig.add_axes([0.03, 0.12, 0.22, 0.74])
    ax1 = fig.add_axes([0.36, 0.12, 0.22, 0.74])
    ax2 = fig.add_axes([0.70, 0.16, 0.29, 0.70])
    for ax, M, t in [(ax0, A, "GRM $A$"), (ax1, Wm, "whitened: $A_{ij}/\\sqrt{v_i v_j}$")]:
        ax.imshow(M, cmap="Reds", vmin=M.min(), vmax=M.max(), interpolation="nearest")
        for i, p in enumerate(pop):
            ax.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t, color=INK)
    F.arrow(fig, ax0, ax1, "divide row $i$ and\ncolumn $i$ by $\\sqrt{v_i}$")
    x = np.arange(N)
    ax2.scatter(x, v, s=34, color=[PCOL[p] for p in pop], edgecolor="white", lw=0.4, label="before: $v_i$")
    ax2.scatter(x, np.ones(N), s=60, marker="_", color=INK, lw=1.8, label="after: $v_i / v_i = 1$")
    ax2.set_ylim(0, max(1.2, v.max() * 1.15))
    ax2.set_xticks([])
    ax2.set_xlabel("individuals (by population)", color=MUTED)
    ax2.set_ylabel("noise on the diagonal", color=MUTED)
    ax2.set_title("everyone's noise becomes 1", color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "v range", np.round([v.min(), v.max()], 3))


def trap_fig(bfile, out, n=5):
    """the centering trap on the standard GRM C (= J A J): with the diagonal
    hidden C looks fine, but its rows sum to zero, so every person's diagonal
    (mostly noise) is spread over their row: mean off-diagonal = -C_ii/(N-1)"""
    ds, _ = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    _, C = B.grm(G)
    N = len(C)
    off = ~np.eye(N, dtype=bool)
    fig = plt.figure(figsize=(9, 3.8))
    ax0 = fig.add_axes([0.04, 0.12, 0.30, 0.74])
    ax1 = fig.add_axes([0.50, 0.16, 0.48, 0.70])
    masked = C.copy()
    np.fill_diagonal(masked, np.nan)
    vmax = np.percentile(np.abs(C[off]), 99)
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d0d0d0")
    ax0.imshow(masked, cmap=cm, vmin=-vmax, vmax=vmax, interpolation="nearest")
    for i, p in enumerate(pop):
        ax0.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("GRM $C$ (centered), diagonal gray:\nhidden, but not gone", color=INK)
    rowmean = np.array([C[i, off[i]].mean() for i in range(N)])
    ax1.bar(np.arange(N), rowmean, color=[PCOL[p] for p in pop])
    ax1.axhline(0, color=MUTED, lw=0.6)
    ax1.set_xticks([])
    ax1.set_xlabel("individuals (by population)", color=MUTED)
    ax1.set_ylabel("mean off-diagonal entry\n$= -C_{ii}/(N-1)$", color=MUTED)
    ax1.set_title("rows sum to zero: each person's diagonal\nis spread over their row", color=INK)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    hs = [plt.Line2D([], [], marker="s", ls="", color=PCOL[p], markersize=8) for p in POPS]
    ax1.legend(hs, POPS, frameon=False, fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "row means", np.round([rowmean.min(), rowmean.max()], 4),
          "max |rowmean + C_ii/(N-1)|", np.abs(rowmean + np.diag(C) / (N - 1)).max())


def hwe_fig(results, out):
    """HWE within individuals: runs wrong (min R2 < 0.95) for the four
    methods (results/four_methods.tsv, 10 replicates per n = 5-40) with
    standard PCA and the HWE-based Chen & Storey diagonal as references
    (results/hwe.tsv, the same datasets: replicates 0-9; k = 3)"""
    import pandas as pd
    d4 = pd.read_csv(os.path.join(results, "four_methods.tsv"), sep="\t")
    d4 = d4[d4.group == "hwe"]
    dh = pd.read_csv(os.path.join(results, "hwe.tsv"), sep="\t")
    dh = dh[(dh.rep < 10) & (dh.method == "lrkin_cs_kc")].assign(method="cs_diag")
    d = pd.concat([d4[["scen", "n", "rep", "method", "minR2"]], dh[["scen", "n", "rep", "method", "minR2"]]])
    d["fail"] = d.minR2 < 0.95
    scen = [("inbred1", "one inbred\nindividual"), ("inbredpop", "inbred\npopulation"), ("err1", "1% genotype\nerrors")]
    meth = [("standard", "standard PCA", "#BBBBBB"),
            ("cs_diag", "Chen & Storey diagonal (HWE)", "#E69F00"),
            ("unc_fit", "uncentered GRM fit", "#56B4E9"),
            ("aarobust_kin", "aarobust-kin", "#CC79A7"),
            ("detect_white", "detect-white", "#009E73"),
            ("dwg", "dwg", "#0072B2")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(meth)
    for i, (m, lab, c) in enumerate(meth):
        y = [100 * d[(d.method == m) & (d.scen == s_)].fail.mean() for s_, _ in scen]
        ax.bar(np.arange(len(scen)) - 0.4 + w * (i + 0.5), y, w, color=c, label=lab)
        if m in FOUR_KEYS:
            for x_, y_ in zip(np.arange(len(scen)) - 0.4 + w * (i + 0.5), y):
                if y_ < 0.5:
                    ax.text(x_, 0.6, "0", ha="center", fontsize=7, color=c)
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels([t for _, t in scen])
    ax.set_ylabel("runs wrong (%)", color=MUTED)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for t in leg.get_texts():
        if t.get_text() in ("uncentered GRM fit", "aarobust-kin", "detect-white", "dwg"):
            t.set_fontweight("bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    for m, lab, _ in meth:
        print(lab, [round(100 * d[(d.method == m) & (d.scen == s_)].fail.mean(), 1) for s_, _ in scen])


FOUR_KEYS = {"unc_fit", "aarobust_kin", "detect_white", "dwg"}


def wigner_fig(out, N=600, seed=7):
    """Wigner's semicircle: eigenvalues of a symmetric noise matrix with
    independent entries (sd sigma) fill [-2 sigma sqrt N, 2 sigma sqrt N];
    added structure (spikes theta) gives eigenvalues beyond the edge only if
    theta > sigma sqrt N (then at theta + sigma^2 N / theta)"""
    rng = np.random.default_rng(seed)
    sigma = 1 / np.sqrt(N)  # edge at 2
    X = rng.normal(0, sigma, (N, N))
    E = (X + X.T) / np.sqrt(2)
    lam0 = np.linalg.eigvalsh(E)
    thetas = [3.0, 2.0, 0.7]
    Q, _ = np.linalg.qr(rng.normal(size=(N, len(thetas))))
    lam1 = np.linalg.eigvalsh(E + (Q * thetas) @ Q.T)
    x = np.linspace(-2, 2, 400)
    dens = np.sqrt(np.maximum(4 - x ** 2, 0)) / (2 * np.pi)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, lam, t in [(axs[0], lam0, "pure noise"), (axs[1], lam1, "noise + 3 structure axes")]:
        ax.hist(lam, bins=60, density=True, color="#BBBBBB", label="eigenvalues")
        ax.plot(x, dens, color="#0072B2", lw=2, label="semicircle")
        ax.axvline(2, color="#D55E00", ls="--", lw=1)
        ax.text(2.05, 0.33, "edge\n$2\\sigma\\sqrt{N}$", color="#D55E00", fontsize=9, va="top")
        ax.set_xlim(-2.6, 4.0)
        ax.set_ylim(0, 0.36)
        ax.set_yticks([])
        ax.set_xlabel("eigenvalue (units of $\\sigma\\sqrt{N}$)", color=MUTED)
        ax.set_title(t, color=INK)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
    out_l = lam1[lam1 > 2.05]
    axs[1].scatter(out_l, np.full(len(out_l), 0.02), marker="v", s=80, color="#0072B2", zorder=3)
    axs[1].text(out_l.mean(), 0.07, "strong axes:\nbeyond the edge", ha="center", fontsize=9, color="#0072B2")
    axs[1].text(-2.5, 0.33, "weak axis (strength 0.7 < 1):\nhidden in the noise", fontsize=9, color=MUTED, va="top")
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "max noise eig", round(lam0.max(), 3), "outliers", np.round(out_l, 3))


FOUR = {"dwg", "uncentered GRM fit", "detect-white", "aarobust-kin"}


def compare_fig(results, out):
    """accuracy without relatives (scenario "none", 5 replicates per n, k=3)
    and run time of the C++ implementations"""
    import pandas as pd
    rc = pd.read_csv(os.path.join(results, "restore_centring_test.tsv"), sep="\t")
    gf = pd.read_csv(os.path.join(results, "grm_fit_bench.tsv"), sep="\t")
    dl = pd.read_csv(os.path.join(results, "dwg_loc.tsv"), sep="\t")
    sel = lambda d, m: d[(d.scen == "none") & (d.k == 3) & (d.method == m)]  # noqa: E731
    acc = [("dwg", sel(dl, "loc4_ea_iter"), "#0072B2", "-"),
           ("uncentered GRM fit", sel(rc, "restored_fit"), "#56B4E9", "--"),
           ("detect-white", sel(rc, "detect_white"), "#009E73", "-"),
           ("aarobust-kin", sel(rc, "aarobust_kin"), "#E69F00", "-"),
           ("Chen & Storey diagonal", sel(rc, "imp_diagCS"), "#999999", ":"),
           ("GRM (standardized), diagonal free", sel(gf, "grm_fit"), "#CC79A7", "-"),
           ("standard PCA", sel(rc, "standard"), "#D55E00", "-")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for name, d, c, ls in acc:
        m = d.groupby("n").minR2.mean()
        main = name in FOUR
        a1.plot(m.index, m.values, ls=ls, color=c, lw=2.6 if main else 1.2, marker="o", ms=6 if main else 4,
                label=name, zorder=3 if main else 2)
    a1.set_xscale("log")
    a1.set_xticks([5, 10, 20, 40])
    a1.set_xticklabels(["5", "10", "20", "40"])
    a1.minorticks_off()
    a1.set_ylim(-0.03, 1.05)
    a1.set_xlabel("individuals per population")
    a1.set_ylabel("mean min $R^2$ (3 ancestry axes)")
    a1.set_title("accuracy, no relatives", color=INK)
    leg = a1.legend(frameon=False, fontsize=8, loc="lower right", title="bold: the four methods", title_fontsize=8)
    for t in leg.get_texts():
        if t.get_text() in FOUR:
            t.set_fontweight("bold")
    # run time, C++ dense engine (results/robust_speed_dense_cpp.txt, dwg_speed_cpp.txt)
    # and dwg in its default operator engine (methods document)
    t = {"aarobust-kin": ([500, 1000, 2000], [4.71, 30.71, 247.66], "#E69F00", "-"),
         "detect-white": ([500, 1000, 2000], [1.48, 3.90, 9.00], "#009E73", "-"),
         "dwg (dense engine)": ([500, 1000, 2000], [1.79, 4.51, 10.69], "#0072B2", "-"),
         "dwg (default engine)": ([1000, 5000, 20000], [1.1, 4.7, 26.5], "#0072B2", "--")}
    for name, (x, y, c, ls) in t.items():
        a2.plot(x, y, ls=ls, color=c, lw=2, marker="o", ms=5, label=name)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("$N$")
    a2.set_ylabel("seconds")
    a2.set_title("run time (PCAone)", color=INK)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (a1, a2):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    for name, d, _, _ in acc:
        print(name, d.groupby("n").minR2.mean().round(3).to_dict(), "reps", d.groupby("n").size().to_dict())


def aar_pairs_fig(bfile, out):
    """aarobust-kin with relatives (10 per population + 8 relatives): how the
    related pairs are found (KING screen, then the kinship of the PCP residual
    C - L above tau), the entries that are imputed (diagonal + pairs, grey) and
    L with them imputed"""
    ds, _ = F.dataset(bfile)
    c = F.compute(ds)
    pop = F.family_pop(ds)
    N = len(pop)
    o = F.order_rows_spread(pop, ds["Kped"])
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    C, Lc, Sc, king = P(c["C"]), P(c["Lc"]), P(c["Sc"]), P(c["king"])
    pop = pop[o]
    R = C - Lc
    phi = I.kin_scale(R, np.diag(R))
    off = ~np.eye(N, dtype=bool)
    sel = (Sc != 0) & off
    fig = plt.figure(figsize=(12, 4.0))
    a0 = fig.add_axes([0.06, 0.10, 0.24, 0.76])
    a1 = fig.add_axes([0.40, 0.10, 0.24, 0.76])
    a2 = fig.add_axes([0.74, 0.10, 0.24, 0.76])
    iu = np.triu_indices(N, 1)
    cand, s, py = king[iu] > B.KING_SCREEN, sel[iu], phi[iu]
    # KING kinship matrix, diagonal left out; the pairs found (residual kinship > tau) boxed
    ck = plt.get_cmap("RdBu_r").copy()
    ck.set_bad("white")
    kshow = king.copy()
    np.fill_diagonal(kshow, np.nan)
    a0.imshow(kshow, cmap=ck, vmin=-0.25, vmax=0.25, interpolation="nearest")
    for i, j in zip(*np.where(sel)):
        a0.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#D55E00", lw=1.6))
    a0.set_title("1. KING kinship: related\npairs found (orange)", color=INK)
    vmax = np.percentile(np.abs(C[off]), 99)
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d0d0d0")
    shown = C.copy()
    mask = sel | ~off
    shown[mask] = np.nan
    a1.imshow(shown, cmap=cm, vmin=-vmax, vmax=vmax, interpolation="nearest")
    for i, j in zip(*np.where(mask)):
        a1.text(j, i, "?", ha="center", va="center", fontsize=5, color=INK)
    a1.set_title("2. GRM $C$: diagonal + related\npairs to impute (gray)", color=INK)
    F.heat(a2, Lc, "3. $L$: imputed by PCP\n(min $\\sum|\\lambda|$)", vmax=vmax, pop=pop)
    for ax in (a0, a1, a2):
        for i, p in enumerate(pop):
            ax.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
    F.arrow(fig, a0, a1, "set to\nmissing")
    F.arrow(fig, a1, a2, "PCP")
    for t in fig.texts:
        t.set_fontsize(9)
    S.legend(fig, True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "candidates", int(cand.sum()), "selected", int(s.sum()),
          "selected kinship", np.round(np.sort(py[s]), 3), "max non-selected candidate", np.round(py[cand & ~s].max(), 3)
          if (cand & ~s).any() else None)


def aar_flow_fig(bfile, out, extra=2):
    """aarobust-kin step by step (10 per population + 8 relatives): KING
    candidates, then PCP iterates L (low-rank fit of C - S, imputing the
    diagonal and the pairs) and S (diagonal + candidates with phi_S > tau),
    PCs of L. No population bars on the matrices. For illustration `extra`
    unrelated pairs (the highest KING below the screen, >= 4 rows apart) are added as
    candidates: a candidate only enters S if phi_S > tau, so they stay out."""
    ds, _ = F.dataset(bfile)
    c = F.compute(ds)
    pop = F.family_pop(ds)
    N = len(pop)
    rel = np.arange(N) >= ds["n0"]
    o = F.order_rows_spread(pop, ds["Kped"])
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    tau = B.TAU
    cand0 = c["cand"].copy()
    np.fill_diagonal(cand0, False)
    pos = np.argsort(o)  # row of each person in the plot: keep the added pairs off the diagonal
    far = np.abs(pos[:, None] - pos[None, :]) >= 4
    kk = np.where(np.triu(~cand0 & far, 1), c["king"], -np.inf)
    for q in np.argsort(kk, axis=None)[::-1][:extra]:
        i, j = np.unravel_index(q, kk.shape)
        cand0[i, j] = cand0[j, i] = True
    Lc0, Sc0, _ = I.pcp_kin(c["C"], tau, True, cand=cand0)
    king, Lc, Sc, C = P(c["king"]), P(Lc0), P(Sc0), P(c["C"])
    pop, rel, U = pop[o], rel[o], I.top_eig(Lc0, 3)[1][o]
    off = ~np.eye(N, dtype=bool)
    cand = P(cand0) & off
    fig = plt.figure(figsize=(10, 3.6))
    ax = F.panel_axes(fig, [1.0, 1, 1, 1, 1.05], 0.8, bottom=0.17, height=0.56)
    F.geno(ax[0], c["G"][o][:, :160], pop)
    for pt in list(ax[0].patches):
        pt.remove()
    ck = plt.get_cmap("RdBu_r").copy()
    ck.set_bad("white")
    kshow = king.copy()
    np.fill_diagonal(kshow, np.nan)
    ax[1].imshow(kshow, cmap=ck, vmin=-0.25, vmax=0.25, interpolation="nearest")
    for i, j in zip(*np.where(cand)):
        ax[1].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#D55E00", lw=1.4))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("$\\phi_{\\mathrm{KING}}$:\ncandidates (orange)", color=INK)
    vmax = np.percentile(np.abs(C[off]), 99)
    t = int(round(np.log2(tau) * 2))
    F.heat(ax[2], Sc, f"$S$: diagonal + candidates\nwith $\\phi_S > \\tau = 2^{{{t / 2:g}}}$", sparse=True,
           diag_dots=True)
    for i, j in zip(*np.where(cand & (Sc == 0))):  # candidates left out of S
        ax[2].add_patch(plt.Rectangle((j - 0.9, i - 0.9), 1.8, 1.8, fill=False, ec="#D55E00", lw=1.0))
    F.heat(ax[3], Lc, "$L$: low-rank fit of $C - S$\n(diagonal, pairs imputed)", vmax=vmax)
    F.pcs(ax[4], U, pop, rel, "PCs of $L$", 1, 2)
    ax[4].set_xlim(-1.25, 1.25)
    ax[4].set_ylim(-1.25, 1.25)
    for a_ in ax:
        a_.title.set_fontsize(9.5)
    fig.canvas.draw()
    F.arrow(fig, ax[0], ax[1], "KING")
    F.arrow(fig, ax[1], ax[2], "candidates")
    b3, b4 = ax[3].get_position(), ax[4].get_position()
    y4 = (b3.y0 + b3.y1) / 2
    fig.add_artist(FancyArrowPatch((b3.x1 + 0.004, y4), (b4.x0 - 0.03, y4), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.0))
    fig.text((b3.x1 + b4.x0 - 0.026) / 2, y4 + 0.05, "eigen-\nvectors", ha="center", va="bottom", fontsize=7.5,
             color=MUTED, transform=fig.transFigure, linespacing=1.1)
    # the PCP loop between S and L
    b2, b3 = ax[2].get_position(), ax[3].get_position()
    x0, x1 = b2.x1 + 0.006, b3.x0 - 0.006
    ym = (b2.y0 + b2.y1) / 2
    for ya, yb, rad, txt, va in [(ym + 0.1, ym + 0.1, -0.5, "impute $L$", "bottom"),
                                 (ym - 0.1, ym - 0.1, -0.5, "$\\phi_S > \\tau$", "top")]:
        a, b = ((x0, ya), (x1, yb)) if va == "bottom" else ((x1, ya), (x0, yb))
        fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                                       color=MUTED, lw=1.0, connectionstyle=f"arc3,rad={rad}"))
        fig.text((x0 + x1) / 2, ym + (0.06 if va == "bottom" else -0.06), txt, ha="center", va="center",
                 fontsize=8.5, color=INK, transform=fig.transFigure)
    fig.text((x0 + x1) / 2, ym, "iterate", ha="center", va="center", fontsize=8, color=MUTED,
             transform=fig.transFigure)
    F.legend_row(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "candidates", int(cand.sum() // 2), "in S", int(((Sc != 0) & off).sum() // 2))


def main():
    bfile, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    if len(sys.argv) > 3 and sys.argv[3] == "mushroom":
        mushroom_fig(os.path.join(outdir, "mushroom.pdf"))
        return
    if len(sys.argv) > 3 and sys.argv[3] == "aarflow":
        aar_flow_fig(bfile, os.path.join(outdir, "aar_flow.pdf"))
        return
    if len(sys.argv) > 3 and sys.argv[3] == "aarpairs":
        aar_pairs_fig(bfile, os.path.join(outdir, "aar_pairs.pdf"))
        return
    if len(sys.argv) > 4 and sys.argv[3] == "hwe":
        hwe_fig(sys.argv[4], os.path.join(outdir, "hwe.pdf"))
        return
    if len(sys.argv) > 3 and sys.argv[3] == "trap":
        trap_fig(bfile, os.path.join(outdir, "trap.pdf"))
        return
    if len(sys.argv) > 3 and sys.argv[3] == "wigner":
        wigner_fig(os.path.join(outdir, "wigner.pdf"))
        return
    if len(sys.argv) > 3 and sys.argv[3] == "impute":
        impute_topr_fig(bfile, os.path.join(outdir, "impute_topr.pdf"))
        nuclear_fig(bfile, os.path.join(outdir, "impute_nuclear.pdf"))
        whitening_fig(bfile, os.path.join(outdir, "whitening.pdf"))
        return
    if len(sys.argv) > 3:
        compare_fig(sys.argv[3], os.path.join(outdir, "smalln_compare.pdf"))
        return
    ds, T = F.dataset(bfile, n=5, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    T = S.orient_truth(T[o], pop)
    N = len(G)
    tau = B.TAU
    none = np.zeros((N, N), bool)

    # uncentred GRM (SNP-standardised, not centred); L with the diagonal free,
    # rank stepped up from 1 while the next eigenvalue is above the noise edge
    A, f, w, _ = W.grm_scaled_uncentred(G)
    L, Sm, r, pairs, _ = DL.fit_loc(A, tau, none, W.noise_edge(f, w, N), K + 1, 4, kin_check=True)
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    off = np.percentile(np.abs((A - A.mean())[~np.eye(N, dtype=bool)]), 99)
    kw = dict(positive=True)
    draw(G, pop, T, A, "uncentered GRM $A$\n(scaled)", kw, imputed(A, L), f"$L$: diag. imputed by\ntop {r - 1} PCs (rank {r})",
         np.diag(A - L),
         "$S$: the diagonal", I.cs_pcs(L, K)[1],
         "uncentered GRM fit: low rank, diagonal free", os.path.join(outdir, "ga_small_fit.pdf"),
         ["scale,\nnot centered", "fit\n$A = S + L$", "eigen-\nvectors"])
    draw(G, pop, T, A, "uncentered GRM $A$\n(scaled)", kw, imputed(A, L), f"$L$: diag. imputed by\ntop {r - 1} PCs (rank {r})",
         np.diag(A - L),
         "$S$: the diagonal", I.whitened_noise(A, v, pairs, K, centre=True),
         "detect-white-GRM (dwg): uncentered GRM, diagonal free, then whitened and centered",
         os.path.join(outdir, "ga_small_dwg.pdf"),
         ["scale,\nnot centered", "fit\n$A = L + S$", "$\\Sigma = S$", "top $k$,\n$\\times\\Sigma^{1/2}$"],
         sigma=v, raw=A, centre=True, shift=0, sigma_title="$\\Sigma = S$\n(diagonal $v$)")

    # aarobust-kin: PCP of the centred, scaled GRM
    _, C = B.grm(G)
    Lp, Sp, _ = I.pcp_kin(C, tau, True, cand=none)
    vmax = np.percentile(np.abs(C[~np.eye(N, dtype=bool)]), 99)
    # L = C off the diagonal; its diagonal is the one with the smallest sum of
    # |eigenvalues| (nuclear-norm completion, all eigenvectors); S = the rest
    draw(G, pop, T, C, "GRM $C$\n(standardized)", dict(vmax=vmax), Lp,
         "$L$: diag. imputed,\nmin $\\sum_k|\\lambda_k|$ (all eigenvectors)", np.diag(C - Lp), "$S$: the diagonal",
         I.top_eig(Lp, K)[1],
         "aarobust-kin: robust PCA of the GRM",
         os.path.join(outdir, "ga_small_aarobust.pdf"),
         ["standardize", "PCP\n$C = S + L$", "eigen-\nvectors"], shift=0)

    # detect-white: raw Gram G G'/M, diagonal free, whitened. PCAone fits the
    # Chen & Storey matrix H; with the diagonal free H and the raw Gram give
    # the same L (they differ only on the diagonal), so S = diag(v)
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L2, S2, _, r2 = I.lr_kin_fd_auto(H, tau, D, I.cs_noise_edge(G), none, rmax=K + 1)
    pp = I.lr_kin_pairs(H, D, L2, S2)
    AM = G @ G.T / G.shape[1]
    v2 = np.diag(AM) - np.diag(L2)
    draw(G, pop, T, AM, "raw Gram $GG^\\top/M$\n(not centered, not scaled)", dict(positive=True), imputed(AM, L2),
         f"$L$: diag. imputed by\ntop {r2 - 1} PCs (rank {r2})",
         v2, "$S$: the diagonal $v$", I.whitened_noise(AM, v2, pp, K),
         "detect-white: raw Gram, diagonal free, then whitened",
         os.path.join(outdir, "ga_small_dw.pdf"),
         ["$GG^\\top/M$", "fit\n$GG^\\top/M = L + S$", "$\\Sigma = S$", "eigen-\nvectors,\n$\\times\\Sigma^{1/2}$"],
         sigma=v2, raw=AM, sigma_title="$\\Sigma = S$\n(diagonal $v$)")


if __name__ == "__main__":
    main()
