#!/usr/bin/env python3
"""Figures for the PCAone --robust methods document: one graphical abstract per
method, drawn from a real example (admixTjeck2, 10 per population, scenario
"many"), plus an overview of the PCs and a schematic of the engines.

The matrices are computed with the Python prototypes, which PCAone reproduces
exactly (check_pcaone.py)."""
import argparse
import os
import sys
import zlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed  # noqa: E402

POPS = ["CEU", "CHB", "MXL", "YRI"]
# Okabe-Ito, colour-blind safe
PCOL = {"CEU": "#0072B2", "CHB": "#E69F00", "MXL": "#009E73", "YRI": "#CC79A7"}
INK, MUTED = "#222222", "#666666"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})


def dataset(bfile, n=10, scen="many"):
    fam, bim, Gall = read_bed(bfile)
    lab = np.array([r[1] for r in fam])
    seed = zlib.crc32(f"{','.join(POPS)}|{n}|0|{scen}|0".encode())
    ds = B.make_dataset(Gall, lab, bim, POPS, n, 0, B.SCENARIOS[scen], seed)
    T = B.truth_reference(ds, Gall, 3)
    return ds, T


def family_pop(ds):
    """population of every row, from the pedigree nodes (children: the father's)"""
    nodes = ds["nodes"]

    def npop(x):
        kind, info, p = nodes[x]
        if kind in ("base", "ext"):
            return p
        if kind == "copy":
            return npop(info)
        return npop(info[0])

    return np.array([npop(x) for x in ds["rows"]])


def order_rows(pop, n0, K):
    """rows by population, each relative right after its closest placed kin;
    a relative with no placed kin starts at the end of its population's block
    (its own relatives then follow it)"""
    N = len(pop)
    order = [i for p in POPS for i in range(n0) if pop[i] == p]
    left = list(range(n0, N))
    while left:
        kin = [max(order, key=lambda j: K[r, j]) for r in left]
        best = max(range(len(left)), key=lambda a: K[left[a], kin[a]])
        r = left.pop(best)
        if K[r, kin[best]] > 0:
            pos = order.index(kin[best]) + 1
        else:
            same = [a for a, j in enumerate(order) if pop[j] == pop[r]]
            pos = same[-1] + 1 if same else len(order)
        order.insert(pos, r)
    return np.array(order)


def order_rows_spread(pop, K):
    """rows by population, the members of each family spread over their
    population's block (not next to each other), so a related pair sits away
    from the diagonal and stands out in S"""
    N = len(pop)
    fam = np.arange(N)  # connected components of K > 0
    for i, j in zip(*np.where(np.triu(K, 1) > 0)):
        a, b = fam[i], fam[j]
        fam[fam == b] = a
    order = []
    for p in POPS:
        idx = np.where(pop == p)[0]
        fams = [idx[fam[idx] == f] for f in dict.fromkeys(fam[idx])]
        multi = [m for m in fams if len(m) > 1]
        single = [m[0] for m in fams if len(m) == 1]
        t = {i: (k + 0.5) / max(len(single), 1) for k, i in enumerate(single)}
        for f, m in enumerate(multi):  # member j near (j + 0.5) / size, families staggered
            for j, i in enumerate(m):
                t[i] = (j + 0.5 + 0.6 * (f + 0.5) / len(multi) - 0.3) / len(m)
        order += sorted(idx, key=lambda i: t[i])
    return np.array(order)


def heat(ax, M, title, vmax=None, mask_diag=False, sparse=False, cmap="RdBu_r", pop=None, positive=False,
         diag_dots=False):
    M = np.array(M, dtype=float)
    n = len(M)
    off = M[~np.eye(n, dtype=bool)]
    if positive:  # an uncentred matrix (no negative entries): white -> red from its smallest off-diagonal entry
        lo, hi = off.min(), np.percentile(off, 99)
        cm = plt.get_cmap("Reds").copy()
        cm.set_bad("#d0d0d0")
        shown = M.copy()
        if mask_diag:
            np.fill_diagonal(shown, np.nan)
        ax.imshow(shown, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        if pop is not None:
            for i, p in enumerate(pop):
                ax.add_patch(plt.Rectangle((-0.06 * n - 0.5, i - 0.5), 0.04 * n, 1, color=PCOL[p], clip_on=False,
                                           lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=INK)
        return
    if vmax is None:
        vmax = np.abs(off).max() if sparse else np.percentile(np.abs(off), 99)
    cm = plt.get_cmap(cmap).copy()
    if sparse:
        # nonzero off-diagonal cells as markers (single pixels are invisible)
        ax.imshow(np.full((n, n), np.nan), cmap=cm, vmin=-vmax, vmax=vmax)
        ii, jj = np.where((np.abs(M) > 1e-12) & ~np.eye(n, dtype=bool))
        ax.scatter(jj, ii, s=16, marker="s", c=M[ii, jj], cmap=cm, vmin=-vmax, vmax=vmax, linewidths=0)
        if diag_dots:  # the diagonal, in colour (it saturates: it is much larger than the pairs)
            d = np.arange(n)
            ax.scatter(d, d, s=16, marker="s", c=np.diag(M), cmap=cm, vmin=-vmax, vmax=vmax, linewidths=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
    else:
        shown = M.copy()
        if mask_diag:
            np.fill_diagonal(shown, np.nan)
        cm.set_bad("#d0d0d0")
        ax.imshow(shown, cmap=cm, vmin=-vmax, vmax=vmax, interpolation="nearest")
    if pop is not None:
        for i, p in enumerate(pop):
            ax.add_patch(plt.Rectangle((-0.06 * n - 0.5, i - 0.5), 0.04 * n, 1, color=PCOL[p], clip_on=False, lw=0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK)


def geno(ax, G, pop, title="genotypes $G$"):
    cm = matplotlib.colors.ListedColormap(["#f4f4f4", "#9ecae1", "#08519c"])
    ax.imshow(G, cmap=cm, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
    for i, p in enumerate(pop):
        ax.add_patch(plt.Rectangle((-0.06 * G.shape[1], i - 0.5), 0.04 * G.shape[1], 1, color=PCOL[p],
                                   clip_on=False, lw=0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK)


def pcs(ax, U, pop, rel, title, a=0, b=1):
    U = U / np.abs(U).max(0)
    for p in POPS:
        idx = (pop == p) & ~rel
        ax.scatter(U[idx, a], U[idx, b], s=14, color=PCOL[p], edgecolor="white", linewidth=0.4, label=p)
    idx = rel
    ax.scatter(U[idx, a], U[idx, b], s=34, marker="*", color=[PCOL[p] for p in pop[idx]], edgecolor=INK,
               linewidth=0.5, label="relative")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC{a + 1}", color=MUTED, labelpad=1)
    ax.set_ylabel(f"PC{b + 1}", color=MUTED, labelpad=1)
    ax.set_title(title, color=INK)


def arrow(fig, ax1, ax2, text=""):
    b1, b2 = ax1.get_position(), ax2.get_position()
    y = (b1.y0 + b1.y1) / 2
    x0, x1 = b1.x1 + 0.004, b2.x0 - 0.004
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color=MUTED, lw=1.0))
    if text:
        fig.text((x0 + x1) / 2, y + 0.05, text, ha="center", va="bottom", fontsize=7.5, color=MUTED,
                 transform=fig.transFigure, linespacing=1.1)


def plus(fig, ax1, ax2, sym="+"):
    b1, b2 = ax1.get_position(), ax2.get_position()
    fig.text((b1.x1 + b2.x0) / 2, (b1.y0 + b1.y1) / 2, sym, ha="center", va="center", fontsize=13,
             color=INK, transform=fig.transFigure)


def legend_row(fig, y=0.02):
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=6) for p in POPS]
    hs.append(plt.Line2D([], [], marker="*", ls="", color="#bbbbbb", markeredgecolor=INK, markersize=9))
    fig.legend(hs, POPS + ["relative"], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, y),
               fontsize=7.5)


def panel_axes(fig, widths, gap, left=0.015, bottom=0.17, height=0.62):
    tot = sum(widths) + gap * (len(widths) - 1)
    scale = (1 - 2 * left) / tot
    axes, x = [], left
    for w in widths:
        axes.append(fig.add_axes([x, bottom, w * scale, height]))
        x += (w + gap) * scale
    return axes


def compute(ds):
    G = ds["G"]
    N, M = G.shape
    k = 3
    tau = B.TAU
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    D = (G * (2 - G)).mean(1)
    AM = G @ G.T / M
    H = I.cs_matrix(G)
    out = dict(G=G, king=king, cand=cand, D=D, AM=AM, H=H)
    # aarobust-kin
    _, C = B.grm(G)
    Lc, Sc, _ = I.pcp_kin(C, tau, True, cand=cand)
    out.update(C=C, Lc=Lc, Sc=Sc, U_aar=I.top_eig(Lc, k)[1])
    # detect-white (noise-edge rank, at most k + 1): raw Gram, diagonal free
    L, S, _, r = I.lr_kin_fd_auto(AM, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
    pp = I.lr_kin_pairs(AM, D, L, S)
    v = np.diag(AM) - np.diag(L)
    out.update(L=L, S=S, rank=r, pairs=pp, noise=v, U_dw=I.whitened_noise(AM, v, pp, k))
    # frkin (fixed rank k + 1)
    Lf, Sf, _ = I.lr_kin_fd(AM, k + 1, tau, D, cand)
    out.update(Lf=Lf, Sf=Sf, U_fr=I.cs_pcs(Lf, k)[1])
    # cswhite
    ppk = B.pairs_above(king, tau)
    out.update(pairs_king=ppk, U_cs=I.whitened_cs(G, ppk, k))
    # standard PCA
    X = I.standardize(G)
    X = X[:, np.isfinite(X).all(0)]
    out.update(U_std=I.top_eig(X @ X.T / X.shape[1], k)[1])
    return out


def sigma(noise, pairs, N):
    d = np.sqrt(np.maximum(noise, 1e-12))
    R = np.eye(N)
    for i, j, phi in pairs:
        R[i, j] = R[j, i] = 2 * phi
    return d[:, None] * R * d[None, :]


def fig_aarobust(c, o, pop, rel, out):
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    fig = plt.figure(figsize=(10, 3.0))
    ax = panel_axes(fig, [1.0, 1, 1, 1, 1.05], 0.85)
    geno(ax[0], c["G"][o][:, :160], pop)
    vmax = np.percentile(np.abs(c["C"][~np.eye(len(o), dtype=bool)]), 99)
    heat(ax[1], P(c["C"]), "GRM $C$\n(standardized)", vmax=vmax, pop=pop)
    heat(ax[2], P(c["Sc"]), "$S$: diagonal\n+ related pairs", sparse=True, pop=pop, diag_dots=True)
    heat(ax[3], P(c["Lc"]), "$L$: diagonal and\npairs imputed", vmax=vmax, pop=pop)
    pcs(ax[4], c["U_aar"][o], pop, rel, "PCs of $L$", 1, 2)
    fig.canvas.draw()
    arrow(fig, ax[0], ax[1], "standardize")
    arrow(fig, ax[1], ax[2], "PCP\n$C = S + L$")
    plus(fig, ax[2], ax[3])
    arrow(fig, ax[3], ax[4], "eigen-\nvectors")
    fig.text(0.5, 0.96, "aarobust-kin: robust PCA (PCP) of the GRM with a kinship threshold",
             ha="center", fontsize=11, color=INK, weight="bold")
    legend_row(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_detect_white(c, o, pop, rel, out):
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    N = len(o)
    fig = plt.figure(figsize=(11.5, 3.0))
    ax = panel_axes(fig, [1.0, 1, 1, 1, 1, 1.05], 0.85)
    geno(ax[0], c["G"][o][:, :160], pop)
    # the raw Gram G G'/M: PCAone fits the Chen & Storey matrix H, which differs
    # from it only on the (free) diagonal, so L is the same and S gains diag(D)
    heat(ax[1], P(c["AM"]), "raw Gram $GG^\\top/M$", pop=pop, positive=True)
    heat(ax[2], P(c["L"]), f"$L$: rank {c['rank']}, diag. free", pop=pop, positive=True)
    heat(ax[3], P(c["S"]), "$S$: diagonal\n+ pairs $\\hat\\phi>\\tau$", sparse=True, pop=pop,
         diag_dots=True)
    Sig = sigma(c["noise"], c["pairs"], N)
    heat(ax[4], P(Sig), "$\\Sigma$ (families)", sparse=True, pop=pop, diag_dots=True)
    pcs(ax[5], c["U_dw"][o], pop, rel, "PCs (whitened)", 1, 2)
    fig.canvas.draw()
    arrow(fig, ax[0], ax[1], "one pass")
    arrow(fig, ax[1], ax[2], "fixed rank\n+ kinship")
    plus(fig, ax[2], ax[3])
    arrow(fig, ax[3], ax[4], "families,\nnoise $v$")
    arrow(fig, ax[4], ax[5], "whiten,\n$\\times\\Sigma^{1/2}$")
    fig.text(0.5, 0.96, "detect-white: detect related pairs on the raw Gram, then whitening",
             ha="center", fontsize=11, color=INK, weight="bold")
    legend_row(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cswhite(c, o, pop, rel, out):
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    N = len(o)
    fig = plt.figure(figsize=(10, 3.0))
    ax = panel_axes(fig, [1.0, 1, 1, 1, 1.05], 0.85)
    geno(ax[0], c["G"][o][:, :160], pop)
    Kt = np.where(c["king"] >= B.TAU, c["king"], 0)
    np.fill_diagonal(Kt, 0)
    heat(ax[1], P(Kt), "KING kinship $\\geq\\tau$", sparse=True, pop=pop)
    Sig = sigma(c["D"], c["pairs_king"], N)
    heat(ax[2], P(Sig), "$\\Sigma$ (families)", sparse=True, pop=pop, diag_dots=True)
    w, v = np.linalg.eigh(Sig)
    w = np.maximum(w, 0.1 * np.sqrt(c["D"]).min() ** 2)
    Wm = (v / np.sqrt(w)) @ v.T
    Bw = Wm @ c["AM"] @ Wm
    Bw = Bw - Bw.mean()  # the mean component dominates; centred for display
    heat(ax[3], P(Bw), "$\\Sigma^{-1/2}GG'\\Sigma^{-1/2}$", pop=pop)
    pcs(ax[4], c["U_cs"][o], pop, rel, "PCs (whitened)", 1, 2)
    fig.canvas.draw()
    arrow(fig, ax[0], ax[1], "KING-\nrobust")
    arrow(fig, ax[1], ax[2], "families,\n$v = D$")
    arrow(fig, ax[2], ax[3], "whiten")
    arrow(fig, ax[3], ax[4], "eigvecs,\n$\\times\\Sigma^{1/2}$")
    fig.text(0.5, 0.96, "cswhite: Chen & Storey whitening with KING kinship (assumes HWE)", ha="center",
             fontsize=10, color=INK, weight="bold")
    legend_row(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_frkin(c, o, pop, rel, out):
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    N = len(o)
    fig = plt.figure(figsize=(10, 3.0))
    ax = panel_axes(fig, [1.0, 1, 1, 1, 1.05], 0.85)
    geno(ax[0], c["G"][o][:, :160], pop)
    heat(ax[1], P(c["AM"]), "raw Gram $GG^\\top/M$", pop=pop, positive=True)
    heat(ax[2], P(c["Sf"]), "$S$: diagonal\n+ related pairs", sparse=True, pop=pop, diag_dots=True)
    heat(ax[3], P(c["Lf"]), "$L$: rank $k+1$, diag. free", pop=pop, positive=True)
    pcs(ax[4], c["U_fr"][o], pop, rel, "PCs of $L$", 1, 2)
    fig.canvas.draw()
    arrow(fig, ax[0], ax[1], "one pass")
    arrow(fig, ax[1], ax[2], "fixed rank\n+ kinship")
    plus(fig, ax[2], ax[3])
    arrow(fig, ax[3], ax[4], "eigen-\nvectors")
    fig.text(0.5, 0.96, "frkin: fixed rank + kinship threshold on the raw Gram", ha="center", fontsize=10,
             color=INK, weight="bold")
    legend_row(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_overview(c, T, pop, rel, out):
    items = [("reference truth", T), ("standard PCA", c["U_std"]), ("aarobust-kin +impute-diag", c["U_aar"]),
             ("detect-white", c["U_dw"]), ("cswhite", c["U_cs"]), ("frkin", c["U_fr"])]
    base = np.where(~rel)[0]
    fig, axs = plt.subplots(2, 6, figsize=(13, 4.4))
    for j, (name, U) in enumerate(items):
        # align to the truth (regression on the unrelated) so that panels are comparable
        Z = np.c_[U, np.ones(len(U))]
        Bc, *_ = np.linalg.lstsq(Z[base], T[base], rcond=None)
        A = Z @ Bc if name != "reference truth" else T
        r2 = 1 - ((T[base] - A[base]) ** 2).sum(0) / ((T[base] - T[base].mean(0)) ** 2).sum(0)
        sub = "" if name == "reference truth" else f"\nmin $R^2$ = {r2.min():.3f}"
        pcs(axs[0, j], A, pop, rel, name + sub, 0, 1)
        pcs(axs[1, j], A, pop, rel, "", 1, 2)
    legend_row(fig, -0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f2f6fb", ec="#7a9cc6", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else None,
            linespacing=1.25)


def link(ax, a, b, text="", dx=0.0, dy=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=0.9))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, fontsize=7, color=MUTED, ha="center",
                va="center")


def fig_engines(out):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.5, 0.93, 0.22, 0.07, "PCAone -b data -k K-1 --robust", fc="#ffffff", ec=MUTED, bold=True)
    box(ax, 0.5, 0.80, 0.34, 0.07, "--robust auto (default): dwg;  engine: in-core or N > 5000?",
        fc="#fff7e6", ec="#e0a030")
    link(ax, (0.5, 0.895), (0.5, 0.835))
    box(ax, 0.2, 0.64, 0.25, 0.09, "dense engine\n(dwg out-of-core, N $\\leq$ 5000;\naarobust-kin with --impute-diag)",
        fc="#eef7ee", ec="#5aa35a", bold=False)
    box(ax, 0.8, 0.64, 0.25, 0.09, "operator engine\n(dwg in-core at every N,\nand out-of-core above 5000)",
        fc="#eef7ee", ec="#5aa35a")
    link(ax, (0.36, 0.80), (0.2, 0.685), "no", dy=0.03)
    link(ax, (0.64, 0.80), (0.8, 0.685), "yes", dy=0.03)
    ax.text(0.5, 0.64, "--impute-diag: aarobust-kin\nexplicit modes: aarobust-kin | dwg |\ndetect-white | cswhite | frkin",
            ha="center", va="center", fontsize=7, color=MUTED)
    # engines
    ax.text(0.2, 0.52, "dense engine  (aarobust-kin always; dwg out-of-core and others for N $\\leq$ 5000)",
            ha="center", fontsize=8, color=INK, weight="bold")
    ax.text(0.73, 0.52, "operator engine  (dwg in-core; other modes N > 5000)", ha="center", fontsize=8,
            color=INK, weight="bold")
    box(ax, 0.2, 0.42, 0.34, 0.09, "one pass over the genotypes\n$A = GG'$, $D_i$, KING counts (het-het, IBS0)"
        "\n[dwg: also $A_s = GW^{-1}G'$; aarobust-kin: the GRM $C$]  -  N $\\times$ N")
    box(ax, 0.2, 0.28, 0.34, 0.09, "fit on the N $\\times$ N matrix\nwarm-started subspace iteration "
        "(top $r$ only)\n[aarobust-kin: full eigendecomposition per iteration]")
    box(ax, 0.2, 0.14, 0.34, 0.07, "whitening / PCs; final relatedness (IAF, 1 pass)\n"
        "->  .eigvecs  .eigvals  .relpairs")
    link(ax, (0.2, 0.375), (0.2, 0.325))
    link(ax, (0.2, 0.235), (0.2, 0.175))
    box(ax, 0.73, 0.42, 0.42, 0.09, "pass 1: $D_i$, diag($GG'$), noise edge, bit-packed genotypes [+ count sketch]\n"
        "KING-robust over all pairs (N $\\leq$ 20,000) or on sketch neighbors (N > 20,000)\n"
        "->  candidate pairs > 0.04   [or --kinship: predetermined pairs, no search]")
    box(ax, 0.73, 0.28, 0.42, 0.09, "fit: each iteration = one pass  $Z = G\\,W^{-1}(G'V)$ (dwg)  (SNPs over threads)\n"
        "Rayleigh-Ritz on (A - S), update L and S on the candidates; family axes;\n"
        "evalAdmix screen from the residual sketch + $k_0$  -  nothing N $\\times$ N")
    box(ax, 0.73, 0.14, 0.42, 0.07, "whitening: block-diagonal $\\Sigma$ (families), warm start from the fit;"
        " final relatedness\n->  .eigvecs  .eigvals  .relpairs   (in-core or out-of-core -m)")
    link(ax, (0.73, 0.375), (0.73, 0.325))
    link(ax, (0.73, 0.235), (0.73, 0.175))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_smalln(bfile, out):
    """the small-N problem at n = 5 per population: the same 20 unrelated
    individuals without and with two MZ pairs"""
    ds, T = dataset(bfile, n=5, scen="mz2")
    G, n0 = ds["G"], ds["n0"]
    N = len(G)
    pop = family_pop(ds)
    rel = np.arange(N) >= n0
    k = 3
    # unrelated only: the base individuals, GRM recomputed on them
    _, Cb = B.grm(G[:n0])
    Lb, _, _ = I.pcp_kin(Cb, B.TAU, True, cand=np.zeros((n0, n0), bool))
    # with relatives
    _, C = B.grm(G)
    cand = B.king_robust(G) > B.KING_SCREEN
    L, S, _ = I.pcp_kin(C, B.TAU, True, cand=cand)
    So = S.copy()
    np.fill_diagonal(So, 0)
    rows = [
        ("unrelated individuals only", np.arange(n0),
         [("reference truth", None), ("standard PCA", I.top_eig(Cb, k)[1]),
          ("standard PCA\n+ --impute-diag", I.top_eig(Lb, k)[1])]),
        ("with relatives (two MZ pairs)", np.arange(N),
         [("reference truth", None), ("standard PCA", I.top_eig(C, k)[1]),
          ("aarobust-kin\n(diagonal kept)", I.top_eig(C - So, k)[1]),
          ("aarobust-kin\n+ --impute-diag", I.top_eig(L, k)[1])]),
    ]
    fig = plt.figure(figsize=(10, 10.4))
    subs = fig.subfigures(2, 1, hspace=0.06)
    stats = {}
    for r, (label, idx, items) in enumerate(rows):
        sf = subs[r]
        sf.suptitle(label, fontsize=11, weight="bold", color=INK)
        axs = sf.subplots(2, 4, gridspec_kw=dict(hspace=0.12, wspace=0.18))
        sf.subplots_adjust(top=0.80, bottom=0.04)
        Tr = T[idx]
        base = np.where(~rel[idx])[0]
        for j, (name, U) in enumerate(items):
            if U is None:
                A, sub = Tr, "\n"
            else:
                Z = np.c_[U, np.ones(len(U))]
                Bc, *_ = np.linalg.lstsq(Z[base], Tr[base], rcond=None)
                A = Z @ Bc
                r2 = 1 - ((Tr[base] - A[base]) ** 2).sum(0) / ((Tr[base] - Tr[base].mean(0)) ** 2).sum(0)
                sub = f"\nmin $R^2$ = {r2.min():.3f}"
                stats[(label, name)] = r2.min()
            pcs(axs[0, j], A, pop[idx], rel[idx], name + sub, 0, 1)
            pcs(axs[1, j], A, pop[idx], rel[idx], "", 1, 2)
        if len(items) == 3:
            # the cause: the observed GRM diagonal against the imputed one
            gs = axs[0, 3].get_gridspec()
            axs[0, 3].remove()
            axs[1, 3].remove()
            ax = sf.add_subplot(gs[:, 3])
            o = np.argsort([POPS.index(p) for p in pop[:n0]], kind="stable")
            x = np.arange(n0)
            ax.scatter(x, np.diag(Cb)[o], s=22, color=[PCOL[p] for p in pop[:n0][o]], edgecolor="white", lw=0.4,
                       label="observed")
            ax.scatter(x, np.diag(Lb)[o], s=22, marker="_", color=INK, lw=1.2, label="imputed (structure part)")
            ax.axhline(np.mean(np.diag(Lb)), color="#bbbbbb", lw=0.6, zorder=0)
            ax.set_xticks([])
            ax.set_xlabel("individuals (by population)", color=MUTED)
            ax.set_ylabel("GRM diagonal $C_{ii}$", color=MUTED, labelpad=2)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_title("why: the observed diagonal\nreflects heterozygosity", color=INK)
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    legend_row(fig, -0.025)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return stats


def fig_smalln_fail(bfile, out):
    """what "fail" means at n = 5: raw PCs, and the worst true axis against
    its best reconstruction from the method's PCs"""
    ds, T = dataset(bfile, n=5, scen="mz2")
    G, n0 = ds["G"], ds["n0"]
    N = len(G)
    pop = family_pop(ds)
    rel = np.arange(N) >= n0
    k = 3
    _, Cb = B.grm(G[:n0])
    Lb, _, _ = I.pcp_kin(Cb, B.TAU, True, cand=np.zeros((n0, n0), bool))
    _, C = B.grm(G)
    cand = B.king_robust(G) > B.KING_SCREEN
    L, S, _ = I.pcp_kin(C, B.TAU, True, cand=cand)
    So = S.copy()
    np.fill_diagonal(So, 0)
    cases = [
        ("A. 20 unrelated individuals (5 per population)", np.arange(n0),
         [("reference truth", None), ("standard PCA", I.top_eig(Cb, k)[1]),
          ("standard PCA + --impute-diag", I.top_eig(Lb, k)[1])]),
        ("B. the same 20 plus two MZ twins (stars)", np.arange(N),
         [("reference truth", None), ("standard PCA", I.top_eig(C, k)[1]),
          ("aarobust-kin, diagonal kept", I.top_eig(C - So, k)[1]),
          ("aarobust-kin + --impute-diag", I.top_eig(L, k)[1])]),
    ]
    names = ["PC1", "PC2", "PC3"]
    fig = plt.figure(figsize=(12, 12.5))
    subs = fig.subfigures(2, 1, hspace=0.05)
    for r, (label, idx, items) in enumerate(cases):
        sf = subs[r]
        sf.suptitle(label, fontsize=12, weight="bold", color=INK, x=0.02, ha="left")
        axs = sf.subplots(2, 4, gridspec_kw=dict(hspace=0.45, wspace=0.25))
        sf.subplots_adjust(top=0.86, bottom=0.07, left=0.05, right=0.98)
        Tr = T[idx]
        pp, rr = pop[idx], rel[idx]
        base = np.where(~rr)[0]
        for j in range(4):
            if j >= len(items):
                axs[0, j].axis("off")
                axs[1, j].axis("off")
                continue
            name, U = items[j]
            # row 1: the raw PCs as the method returns them
            raw = Tr if U is None else U
            pcs(axs[0, j], raw, pp, rr, name, 1, 2)
            ax = axs[1, j]
            if U is None:
                # the three true axes, for reference
                ax.axis("off")
                ax.text(0.0, 0.95, "How to read the lower row\n\n"
                        "x: an individual's position on\n   a true ancestry axis\n"
                        "y: the best reconstruction of\n   that axis from the method's\n   3 PCs (any rotation/scale)\n\n"
                        "on the line  = axis recovered\n"
                        "scattered   = axis lost\n\n"
                        "shown: the method's worst axis\n"
                        "fail = $R^2$ < 0.95 for any axis",
                        transform=ax.transAxes, va="top", fontsize=8.5, color=INK, family="DejaVu Sans")
                continue
            Z = np.c_[U, np.ones(len(U))]
            Bc, *_ = np.linalg.lstsq(Z[base], Tr[base], rcond=None)
            P = Z @ Bc
            r2 = 1 - ((Tr[base] - P[base]) ** 2).sum(0) / ((Tr[base] - Tr[base].mean(0)) ** 2).sum(0)
            a = int(np.argmin(r2))
            lo, hi = Tr[:, a].min(), Tr[:, a].max()
            pad = 0.1 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#bbbbbb", lw=0.8, zorder=0)
            for p_ in POPS:
                m = (pp == p_) & ~rr
                ax.scatter(Tr[m, a], P[m, a], s=26, color=PCOL[p_], edgecolor="white", lw=0.4)
            if rr.any():
                ax.scatter(Tr[rr, a], P[rr, a], s=60, marker="*", color=[PCOL[q] for q in pp[rr]], edgecolor=INK,
                           lw=0.5)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(min(lo, P[:, a].min()) - pad, max(hi, P[:, a].max()) + pad)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"true {names[a]}", color=MUTED)
            ax.set_ylabel(f"reconstructed {names[a]}", color=MUTED)
            ok = r2.min() >= 0.95
            ax.set_title(f"worst axis: {names[a]},  $R^2$ = {r2.min():.3f}\n" + ("recovered" if ok else "FAILS"),
                         color="#1a7f37" if ok else "#c62828", weight="bold")
    legend_row(fig, -0.015)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds, T = dataset(a.bfile)
    c = compute(ds)
    n0, N = ds["n0"], len(ds["G"])
    pop = family_pop(ds)
    rel = np.arange(N) >= n0
    o = order_rows(pop, n0, ds["Kped"] + np.eye(N) * 0)
    po, ro = pop[o], rel[o]
    o2 = order_rows_spread(pop, ds["Kped"])  # related pairs off the diagonal, visible in S
    fig_aarobust(c, o2, pop[o2], rel[o2], os.path.join(a.out, "ga_aarobust.pdf"))
    fig_detect_white(c, o, po, ro, os.path.join(a.out, "ga_detect_white.pdf"))
    fig_cswhite(c, o, po, ro, os.path.join(a.out, "ga_cswhite.pdf"))
    fig_frkin(c, o, po, ro, os.path.join(a.out, "ga_frkin.pdf"))
    fig_overview(c, T, pop, rel, os.path.join(a.out, "overview_pcs.pdf"))
    fig_engines(os.path.join(a.out, "engines.pdf"))
    st = fig_smalln(a.bfile, os.path.join(a.out, "smalln.pdf"))
    fig_smalln_fail(a.bfile, os.path.join(a.out, "smalln_fail.pdf"))
    with open(os.path.join(a.out, "smalln.txt"), "w") as f:
        for (lab, name), v in st.items():
            f.write(f"{lab} | {name.replace(chr(10), ' ')} | {v:.3f}\n")
    with open(os.path.join(a.out, "example.txt"), "w") as f:
        f.write(f"N={N} n0={n0} M={ds['G'].shape[1]} rank={c['rank']} pairs_detect={len(c['pairs'])} "
                f"pairs_king={len(c['pairs_king'])}\n")
        for i, j, phi in c["pairs"]:
            f.write(f"{i} {j} {pop[i]} {pop[j]} phi={phi:.3f} ped={ds['Kped'][i, j]:.3f}\n")


if __name__ == "__main__":
    main()
