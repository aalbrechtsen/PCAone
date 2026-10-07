#!/usr/bin/env python3
"""Hook figures for the slides: a population tree, the reference truth and
standard PCA, for (a) 10 per population + 8 relatives and (b) 5 per
population without relatives (admixTjeck2)."""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods_figs as F  # noqa: E402
from methods_figs import INK, MUTED, PCOL, POPS  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def tree(ax):
    """((CEU, (CHB, Native American)), YRI); MXL = CEU x Native American"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lw = 2.2
    # root at the bottom; tips at the top
    tips = {"YRI": 0.1, "CEU": 0.38, "MXL": 0.6, "CHB": 0.9}
    y_tip = 0.82
    ax.plot([0.1, 0.1], [0.1, y_tip], color=PCOL["YRI"], lw=lw)
    ax.plot([0.1, 0.62], [0.1, 0.1], color=MUTED, lw=lw)  # root bar
    ax.plot([0.62, 0.62], [0.1, 0.42], color=MUTED, lw=lw)  # out-of-Africa
    ax.plot([0.38, 0.84], [0.42, 0.42], color=MUTED, lw=lw)
    ax.plot([0.38, 0.38], [0.42, y_tip], color=PCOL["CEU"], lw=lw)
    ax.plot([0.84, 0.84], [0.42, 0.55], color=MUTED, lw=lw)  # East Asia / Americas
    ax.plot([0.74, 0.92], [0.55, 0.55], color=MUTED, lw=lw)
    ax.plot([0.92, 0.92], [0.55, y_tip], color=PCOL["CHB"], lw=lw)
    ax.plot([0.74, 0.74], [0.55, 0.66], color="#999999", lw=lw, ls=(0, (2, 1.5)))  # Native American (unsampled)
    # admixture into MXL
    ax.annotate("", xy=(0.6, 0.72), xytext=(0.74, 0.66), arrowprops=dict(arrowstyle="-|>", color=PCOL["MXL"], lw=1.6))
    ax.annotate("", xy=(0.6, 0.72), xytext=(0.38, 0.62), arrowprops=dict(arrowstyle="-|>", color=PCOL["MXL"], lw=1.6))
    ax.plot([0.6, 0.6], [0.72, y_tip], color=PCOL["MXL"], lw=lw)
    for p, x in tips.items():
        ax.text(x, y_tip + 0.04, p, ha="center", va="bottom", fontsize=12, weight="bold", color=PCOL[p])
    ax.text(0.74, 0.69, "Native\nAmerican", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(0.5, 0.0, "MXL: admixed (European + Native American)", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("the populations", color=INK)


def scatter(ax, U, pop, rel, title, a=1, b=2):
    U = U / np.abs(U).max(0)
    for p in POPS:
        idx = (pop == p) & ~rel
        ax.scatter(U[idx, a], U[idx, b], s=34, color=PCOL[p], edgecolor="white", linewidth=0.5)
    if rel.any():
        ax.scatter(U[rel, a], U[rel, b], s=110, marker="*", color=[PCOL[p] for p in pop[rel]], edgecolor=INK,
                   linewidth=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC{a + 1}", color=MUTED)
    ax.set_ylabel(f"PC{b + 1}", color=MUTED)
    ax.set_title(title, color=INK)


def orient_truth(T, pop):
    """the same orientation on every slide: YRI positive on PC1, CHB on PC2,
    MXL on PC3 (population mean minus overall mean)"""
    T = T.copy()
    for j, p in enumerate(["YRI", "CHB", "MXL"][:T.shape[1]]):
        if T[pop == p, j].mean() < T[:, j].mean():
            T[:, j] *= -1
    return T


def align_signs(U, T, base):
    """flip each PC to agree with the same-numbered truth axis (unrelated only)"""
    U = U.copy()
    for j in range(min(U.shape[1], T.shape[1])):
        if np.corrcoef(U[base, j], T[base, j])[0, 1] < 0:
            U[:, j] *= -1
    return U


def legend(fig, rel):
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    labs = list(POPS)
    if rel:
        hs.append(plt.Line2D([], [], marker="*", ls="", color="#bbbbbb", markeredgecolor=INK, markersize=12))
        labs.append("relative")
    fig.legend(hs, labs, loc="lower center", ncol=len(labs), frameon=False, bbox_to_anchor=(0.62, -0.02),
               fontsize=10)


def hook(bfile, n, scen, out, std_title):
    ds, T = F.dataset(bfile, n=n, scen=scen)
    c = F.compute(ds)
    N = len(ds["G"])
    pop = F.family_pop(ds)
    rel = np.arange(N) >= ds["n0"]
    T = orient_truth(T, pop)
    # R2: the standard PCs mapped onto the truth by least squares on the unrelated (as the score);
    # plotted: the raw PCs, signs matched to the truth
    base = np.where(~rel)[0]
    Z = np.c_[c["U_std"], np.ones(N)]
    Bc, *_ = np.linalg.lstsq(Z[base], T[base], rcond=None)
    A = Z @ Bc
    r2 = 1 - ((T[base] - A[base]) ** 2).sum(0) / ((T[base] - T[base].mean(0)) ** 2).sum(0)
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_axes([0.0, 0.1, 0.27, 0.78])
    ax1 = fig.add_axes([0.33, 0.14, 0.3, 0.72])
    ax2 = fig.add_axes([0.69, 0.14, 0.3, 0.72])
    tree(ax0)
    scatter(ax1, T, pop, rel, "the truth")
    scatter(ax2, align_signs(c["U_std"], T, base), pop, rel, f"{std_title}\nPC3 recovered: $R^2$ = {r2[2]:.2f}")
    legend(fig, rel.any())
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "R2 per axis", np.round(r2, 3), "N", N)


def truth_fig(bfile, out, n=10, scen="many"):
    """the opening slide: the populations and the true PC1/2 and PC2/3 (the
    unrelated individuals of the first hook)"""
    ds, T = F.dataset(bfile, n=n, scen=scen)
    pop = F.family_pop(ds)
    T = orient_truth(T, pop)
    keep = np.arange(len(T)) < ds["n0"]
    T, pop = T[keep], pop[keep]
    none = np.zeros(len(T), bool)
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_axes([0.0, 0.1, 0.27, 0.78])
    ax1 = fig.add_axes([0.33, 0.14, 0.3, 0.72])
    ax2 = fig.add_axes([0.69, 0.14, 0.3, 0.72])
    tree(ax0)
    scatter(ax1, T, pop, none, "the truth: PC1 vs PC2", a=0, b=1)
    scatter(ax2, T, pop, none, "the truth: PC2 vs PC3", a=1, b=2)
    legend(fig, False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "N", len(T))


def cs_fig(bfile, out, n=5, scen="none"):
    """Chen & Storey: uncentred G G'/M with the heterozygosity removed from the
    diagonal. PC1 is the mean (every individual ~ 1/sqrt(N)); the ancestry
    axes are PC2-4"""
    import illustrate as I
    ds, T = F.dataset(bfile, n=n, scen=scen)
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop, T = G[o], pop[o], orient_truth(T[o], pop[o])
    N = len(G)
    H = I.cs_matrix(G)
    _, V = I.top_eig(H, 4)
    V[:, 0] *= np.sign(V[:, 0].sum())
    S = align_signs(V[:, 1:4], T, np.arange(N))
    Z = np.c_[V[:, 1:4], np.ones(N)]
    Bc, *_ = np.linalg.lstsq(Z, T, rcond=None)
    r2 = 1 - ((T - Z @ Bc) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
    fig = plt.figure(figsize=(12, 4.0))
    axh = fig.add_axes([0.03, 0.12, 0.24, 0.74])
    ax0 = fig.add_axes([0.37, 0.14, 0.27, 0.72])
    ax1 = fig.add_axes([0.73, 0.14, 0.26, 0.72])
    axh.imshow(H, cmap="Reds", vmin=H.min(), vmax=H.max(), interpolation="nearest")
    for i, p in enumerate(pop):
        axh.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
    axh.set_xticks([])
    axh.set_yticks([])
    axh.set_title("CS covariance $H$\n(not centered: all positive)", color=INK)
    ax0.bar(np.arange(N), V[:, 0], color=[PCOL[p] for p in pop], width=0.8)
    ax0.axhline(1 / np.sqrt(N), color=MUTED, lw=0.8, ls=":")
    ax0.text(-0.5, 1.2 / np.sqrt(N), r"dotted: $1/\sqrt{N}$, the same for everyone", ha="left", va="top",
             fontsize=9, color=MUTED)
    ax0.set_ylim(0, 1.25 / np.sqrt(N))
    ax0.set_xticks([])
    ax0.set_xlabel("individuals (by population)", color=MUTED)
    ax0.set_ylabel("PC1", color=MUTED)
    ax0.set_title("PC1: the mean, not ancestry", color=INK)
    for s in ["top", "right"]:
        ax0.spines[s].set_visible(False)
    none = np.zeros(N, bool)
    scatter(ax1, np.c_[V[:, 0], S], pop, none, "PC1+1 vs PC2+1: the ancestry", a=1, b=2)
    ax1.set_xlabel("PC1+1", color=MUTED)
    ax1.set_ylabel("PC2+1", color=MUTED)
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    fig.legend(hs, POPS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "R2 from PC2-4", np.round(r2, 3), "PC1 range", np.round([V[:, 0].min(), V[:, 0].max()], 3))


def ls_fig(bfile, out, n=5, scen="none"):
    """small N as L + S on the GRM C (standardized: centered and scaled): L is
    C with its diagonal imputed by PCP (smallest nuclear norm; it works on a
    centered matrix), S = diag(C - L), the heterozygosity noise. C = L + S exactly"""
    import benchmark as B
    import illustrate as I
    ds, _ = F.dataset(bfile, n=n, scen=scen)
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    _, C = B.grm(G)
    N = len(C)
    L, _, _ = I.pcp_kin(C, B.TAU, True, cand=np.zeros((N, N), bool))
    S = np.diag(np.diag(C - L))
    vmax = np.abs(L).max()  # the structure's scale: C's diagonal saturates (it is mostly noise)
    fig = plt.figure(figsize=(12, 4.0))
    axs = [fig.add_axes([0.03 + 0.335 * i, 0.08, 0.26, 0.78]) for i in range(3)]
    shown = S.copy()
    shown[~np.eye(N, dtype=bool)] = np.nan
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("white")
    for ax, M, t, vm in zip(axs, [C, L, shown], ["GRM $C$ (standardized)", "$L$: diagonal imputed",
                                                 "$S$: diagonal only (heterozygosity)"],
                            [vmax, vmax, np.nanmax(np.abs(shown))]):
        ax.imshow(M, cmap=cm, vmin=-vm, vmax=vm, interpolation="nearest")
        for i, p in enumerate(pop):
            ax.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t, color=INK)
        for s in ax.spines.values():
            s.set_edgecolor("#BBBBBB")
    for a, b, sym in [(axs[0], axs[1], "="), (axs[1], axs[2], "+")]:
        b1, b2 = a.get_position(), b.get_position()
        fig.text((b1.x1 + b2.x0) / 2, (b1.y0 + b1.y1) / 2, sym, ha="center", va="center", fontsize=26, color=INK)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "mean diag C", round(np.diag(C).mean(), 3), "L", round(np.diag(L).mean(), 3),
          "S", round(np.diag(S).mean(), 3), "max |C - L - S|", np.abs(C - L - S).max())


def grm_diag(bfile, out, n=5, scen="none"):
    """the small-N GRM (the individuals of the second hook): the heatmap by
    population, and its diagonal against the structure part (imputed)"""
    import benchmark as B
    import illustrate as I
    ds, _ = F.dataset(bfile, n=n, scen=scen)
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    _, C = B.grm(G)
    N = len(C)
    L, _, _ = I.pcp_kin(C, B.TAU, True, cand=np.zeros((N, N), bool))
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    C, L, pop = C[np.ix_(o, o)], L[np.ix_(o, o)], pop[o]
    fig = plt.figure(figsize=(11, 4.4))
    ax0 = fig.add_axes([0.04, 0.08, 0.36, 0.8])
    ax1 = fig.add_axes([0.53, 0.14, 0.44, 0.72])
    vmax = np.abs(C).max()
    ax0.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for i, p in enumerate(pop):
        ax0.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
        ax0.add_patch(plt.Rectangle((i - 0.5, N - 0.5 + 0.02 * N), 1, 0.04 * N, color=PCOL[p], clip_on=False, lw=0))
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(f"GRM, {n} per population", color=INK)
    x = np.arange(N)
    ax1.scatter(x, np.diag(C), s=46, color=[PCOL[p] for p in pop], edgecolor="white", lw=0.5,
                label="observed diagonal")
    ax1.scatter(x, np.diag(L), s=60, marker="_", color=INK, lw=1.6,
                label="ancestry part (predicted from the off-diagonal)")
    ax1.set_xticks([])
    ax1.set_xlabel("individuals (by population)", color=MUTED)
    ax1.set_ylabel("GRM diagonal $C_{ii}$", color=MUTED)
    ax1.set_ylim(min(0, np.diag(L).min()) - 0.05, np.diag(C).max() * 1.15)
    ax1.set_title("the diagonal is mostly heterozygosity", color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    hs = [plt.Line2D([], [], marker="s", ls="", color=PCOL[p], markersize=8) for p in POPS]
    fig.legend(hs, POPS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "mean diag observed", round(np.diag(C).mean(), 3), "structure", round(np.diag(L).mean(), 3),
          "max |off|", round(np.abs(C[~np.eye(N, dtype=bool)]).max(), 3))


def pca_intro(bfile, out, n=10, nsnp=150):
    """PCA in three panels: genotypes -> covariance (GRM) -> PCs, unrelated
    individuals sorted by population"""
    import benchmark as B
    import illustrate as I
    ds, T = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    T = orient_truth(T[o], pop)
    _, C = B.grm(G)
    U = align_signs(I.top_eig(C, 2)[1], T, np.arange(len(G)))
    rng = np.random.default_rng(3)
    poly = np.where(G.std(0) > 0)[0]
    cols = np.sort(rng.choice(poly, nsnp, replace=False))
    fig = plt.figure(figsize=(12, 4.0))
    ax0 = fig.add_axes([0.03, 0.12, 0.27, 0.74])
    ax1 = fig.add_axes([0.40, 0.12, 0.22, 0.74])
    ax2 = fig.add_axes([0.75, 0.12, 0.24, 0.74])
    F.geno(ax0, G[:, cols], pop, f"genotypes (0/1/2), {nsnp} of {G.shape[1] // 1000}k SNPs")
    ax0.set_xlabel("SNPs", color=MUTED)
    ax0.set_ylabel("individuals", color=MUTED, labelpad=12)
    F.heat(ax1, C, "covariance between individuals (GRM)", pop=pop)
    scatter(ax2, U, pop, np.zeros(len(pop), bool), "PCs: top eigenvectors", a=0, b=1)
    for a, b, t, pad in [(ax0, ax1, "standardize,\n$XX^\\top/M$", 0), (ax1, ax2, "eigen-\ndecomposition", 0.025)]:
        F.arrow(fig, a, b, t)
        if pad:  # keep the arrow off the PC2 label
            p = fig.patches[-1] if fig.patches else fig.artists[-1]
            (x0, y), (x1, _) = p._posA_posB
            p.set_positions((x0, y), (x1 - pad, y))
    for t in fig.texts:
        t.set_fontsize(10)
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    fig.legend(hs, POPS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out, "N", len(G), "M", G.shape[1])


def pca_predict(bfile, outdir, n=5, rs=(2, 4, 6, 8), centred=True):
    """one figure per r: the uncentred GRM A, its prediction from the top r
    PCs (sum of lambda_k v_k v_k^T) and PC1/2 .. PC7/8, the PCs used in
    colour; the sample of the imputation slides. Two versions: the diagonal
    used (pca_predict_r*), and the diagonal missing (pca_predict_nodiag_r*:
    imputed iteratively from the same top r PCs, diag_imputed).
    centred: the GRM C (blue-white-red) instead of A (files pca_predict_A_*)"""
    import benchmark as B
    import dw_grm as W
    ds, T = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    A = B.grm(G)[1] if centred else W.grm_scaled_uncentred(G)[0]
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    if centred:  # one scale for every matrix
        hi = np.percentile(np.abs(A[off]), 99)
        lo, cm = -hi, plt.get_cmap("RdBu_r").copy()
    else:
        lo, hi = A[off].min(), np.percentile(A[off], 99)
        cm = plt.get_cmap("Reds").copy()
    cm.set_bad("#d0d0d0")

    def mat(ax, M, title, nodiag=False):
        M = M.copy()
        if nodiag:
            np.fill_diagonal(M, np.nan)
        ax.imshow(M, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        if nodiag:
            for i in range(N):
                ax.text(i, i, "?", ha="center", va="center", fontsize=6, color=INK)
        for i, p in enumerate(pop):
            ax.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False,
                                       lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=INK)

    for nodiag in (False, True):
        for r in rs:
            M = diag_imputed(A, r) if nodiag else A
            w, V = np.linalg.eigh(M)
            w, V = w[::-1], V[:, ::-1]
            if centred:
                V = align_signs(V, T[o], np.arange(N))
            else:
                V = V * np.sign(V.sum(0) + 1e-12)  # PC1 (the mean) positive
                V[:, 1:] = align_signs(V[:, 1:], T[o], np.arange(N))
            P = (V[:, :r] * w[:r]) @ V[:, :r].T
            fig = plt.figure(figsize=(12, 4.2))
            ax0 = fig.add_axes([0.02, 0.10, 0.26, 0.74])
            ax1 = fig.add_axes([0.34, 0.10, 0.26, 0.74])
            mat(ax0, A, ("GRM $C$" if centred else "uncentered GRM $A$") + (": diagonal missing" if nodiag else ""), nodiag)
            mat(ax1, P, f"predicted from the top {r} PCs")
            for k in range(4):
                a, b = 2 * k, 2 * k + 1
                ax = fig.add_axes([0.67 + (k % 2) * 0.17, 0.50 - (k // 2) * 0.43, 0.13, 0.34])
                if a < r:  # PC a used (PC b too, unless r is odd)
                    scatter(ax, V, pop, np.zeros(N, bool), "", a=a, b=b)
                else:  # not used yet: gray
                    U = V / np.abs(V).max(0)
                    ax.scatter(U[:, a], U[:, b], s=34, color="#d9d9d9", edgecolor="white", linewidth=0.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(f"PC{a + 1}", color="#bbbbbb")
                    ax.set_ylabel(f"PC{b + 1}", color="#bbbbbb")
                    for s_ in ax.spines.values():
                        s_.set_edgecolor("#d9d9d9")
                ax.xaxis.labelpad = 1
                ax.yaxis.labelpad = 1
            hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8)
                  for p in POPS]
            fig.legend(hs, POPS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07),
                       fontsize=10)
            out = os.path.join(outdir, f"pca_predict_{'' if centred else 'A_'}{'nodiag_' if nodiag else ''}r{r}.pdf")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            r2 = 1 - ((A - P)[off] ** 2).sum() / ((A[off] - A[off].mean()) ** 2).sum()
            print(out, "N", N, "off-diag R2", round(r2, 3))


def pca_resid(bfile, out, n=5):
    """what the top two PCs leave unexplained (GRM minus its rank-2 part): the
    diagonal, largest in YRI, so PC3 spreads YRI instead of finding MXL"""
    import benchmark as B
    ds, T = F.dataset(bfile, n=n, scen="none")
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    o = np.argsort([POPS.index(p) for p in pop], kind="stable")
    G, pop = G[o], pop[o]
    T = orient_truth(T[o], pop)
    _, C = B.grm(G)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    V = align_signs(V, T, np.arange(len(G)))
    R = C - (V[:, :2] * w[:2]) @ V[:, :2].T
    N = len(C)
    fig = plt.figure(figsize=(12, 4.0))
    ax0 = fig.add_axes([0.03, 0.12, 0.22, 0.74])
    ax1 = fig.add_axes([0.37, 0.14, 0.27, 0.72])
    ax2 = fig.add_axes([0.75, 0.12, 0.24, 0.74])
    vmax = np.abs(C).max()  # the GRM's own scale
    ax0.imshow(R, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for i, p in enumerate(pop):
        ax0.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("GRM minus top 2 PCs", color=INK)
    x = np.arange(N)
    ax1.bar(x, np.diag(R), color=[PCOL[p] for p in pop], width=0.8)
    ax1.set_xticks([])
    ax1.set_xlabel("individuals (by population)", color=MUTED)
    ax1.set_ylabel("left-over diagonal", color=MUTED)
    ax1.set_title("what is left: mostly the diagonal", color=INK)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    scatter(ax2, V[:, :3], pop, np.zeros(N, bool), "so PC3 spreads YRI", a=1, b=2)
    F.arrow(fig, ax1, ax2, "next\neigenvector")
    p = fig.patches[-1] if fig.patches else fig.artists[-1]
    (x0, y), (x1, _) = p._posA_posB
    p.set_positions((x0, y), (x1 - 0.025, y))  # keep the arrow off the PC3 label
    for t in fig.texts:
        t.set_fontsize(10)
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    fig.legend(hs, POPS, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    d = np.diag(R)
    print(out, "resid diag share", round((d ** 2).sum() / (R ** 2).sum(), 3),
          "YRI share of PC3", round((V[pop == "YRI", 2] ** 2).sum(), 3))


def diag_imputed(A, r=4, iters=1000):
    """the uncentred GRM with its diagonal imputed by a rank-r fit (as ls_fig)"""
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    d = np.array([A[i, off[i]].mean() for i in range(N)])
    At = A.copy()
    for _ in range(iters):
        np.fill_diagonal(At, d)
        w, V = np.linalg.eigh(At)
        L = (V[:, -r:] * w[-r:]) @ V[:, -r:].T
        if np.max(np.abs(np.diag(L) - d)) < 1e-10:
            break
        d = np.diag(L).copy()
    np.fill_diagonal(At, d)
    return At


def pca_resid_rel(bfile, out, n=10, scen="many", fixed=False):
    """what the top two PCs leave unexplained with relatives (the first hook):
    off the diagonal the related pairs stand out, and PC3 follows them.
    fixed: the same on the uncentred GRM with its diagonal imputed (the
    small-N fix), with the mean dropped: PC3+1 still follows the relatives"""
    import benchmark as B
    ds, T = F.dataset(bfile, n=n, scen=scen)
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    N = len(G)
    rel = np.arange(N) >= ds["n0"]
    T = orient_truth(T, pop)
    if fixed:
        f = G.mean(0) / 2
        ok = (f > 0) & (f < 1)
        X = G[:, ok] / np.sqrt(2 * f[ok] * (1 - f[ok]))
        C = diag_imputed(X @ X.T / ok.sum())
    else:
        _, C = B.grm(G)
    m = int(fixed)  # the uncentred GRM's PC1 is the mean
    king = B.king_robust(G)
    o = F.order_rows(pop, ds["n0"], ds["Kped"])
    G, pop, rel, C, king, T = G[o], pop[o], rel[o], C[np.ix_(o, o)], king[np.ix_(o, o)], T[o]
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    V = np.hstack([V[:, :m], align_signs(V[:, m:], T, np.where(~rel)[0])])
    R = C - (V[:, :2 + m] * w[:2 + m]) @ V[:, :2 + m].T
    off = ~np.eye(N, dtype=bool)
    relp = (king > B.TAU) & off
    fig = plt.figure(figsize=(12, 4.0))
    ax0 = fig.add_axes([0.03, 0.12, 0.22, 0.74])
    ax1 = fig.add_axes([0.36, 0.16, 0.28, 0.70])
    ax2 = fig.add_axes([0.75, 0.12, 0.24, 0.74])
    shown = R.copy()
    np.fill_diagonal(shown, np.nan)
    vmax = np.median(R[relp])  # the MZ pair saturates; the other pairs stay visible
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d0d0d0")
    ax0.imshow(shown, cmap=cm, vmin=-vmax, vmax=vmax, interpolation="nearest")
    for i, p in enumerate(pop):
        ax0.add_patch(plt.Rectangle((-0.06 * N - 0.5, i - 0.5), 0.04 * N, 1, color=PCOL[p], clip_on=False, lw=0))
        if rel[i]:
            ax0.plot(-0.04 * N - 0.5, i, marker="*", ms=7, color=PCOL[p], mec=INK, mew=0.4, clip_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("GRM, diagonal imputed,\nminus mean and top 2 PCs" if fixed else "GRM minus top 2 PCs\n(diagonal hidden)",
                  color=INK)
    iu = np.triu_indices(N, 1)
    vals, isrel = R[iu], relp[iu]
    bins = np.linspace(vals.min(), vals.max(), 50)
    ax1.hist(vals[~isrel], bins=bins, color="#BBBBBB", label="all other pairs")
    ax1.set_yscale("log")
    ax1.scatter(vals[isrel], np.full(isrel.sum(), 1.6), marker="v", s=70, color="#D55E00", zorder=3,
                label="related pairs")
    ax1.set_ylim(0.7, 5000)
    ax1.set_xlabel("left-over covariance of a pair", color=MUTED)
    ax1.set_ylabel("pairs", color=MUTED)
    ax1.set_title("what is left: the related pairs stand out", color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    scatter(ax2, V[:, m:3 + m], pop, rel, "so PC3+1 follows the relatives" if fixed else "so PC3 follows the relatives",
            a=1, b=2)
    if fixed:
        ax2.set_xlabel("PC2+1", color=MUTED)
        ax2.set_ylabel("PC3+1", color=MUTED)
    F.arrow(fig, ax1, ax2, "next\neigenvector")
    p = fig.patches[-1] if fig.patches else fig.artists[-1]
    (x0, y), (x1, _) = p._posA_posB
    p.set_positions((x0, y), (x1 - 0.025, y))  # keep the arrow off the PC3 label
    for t in fig.texts:
        t.set_fontsize(10)
    legend(fig, True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    inv = relp.any(1)
    print(out, "related pairs", relp.sum() // 2, "share of resid SS: diag", round((np.diag(R) ** 2).sum() / (R ** 2).sum(), 3),
          "related", round((R[relp] ** 2).sum() / (R ** 2).sum(), 3), "PC3 weight on pair members",
          round((V[inv, 2 + m] ** 2).sum(), 3), "of", inv.sum())


if __name__ == "__main__":
    bfile, outdir = sys.argv[1], sys.argv[2]
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["hook", "grm"]
    os.makedirs(outdir, exist_ok=True)
    if "hook" in which:
        hook(bfile, 10, "many", os.path.join(outdir, "hook_relatives.pdf"),
             "standard PCA, 10 per population + 8 relatives")
        hook(bfile, 5, "none", os.path.join(outdir, "hook_smalln.pdf"), "standard PCA, 5 per population, no relatives")
    if "truth" in which:
        truth_fig(bfile, os.path.join(outdir, "truth.pdf"))
    if "ls" in which:
        ls_fig(bfile, os.path.join(outdir, "smalln_ls.pdf"))
    if "cs" in which:
        cs_fig(bfile, os.path.join(outdir, "cs_pcs.pdf"))
    if "grm" in which:
        grm_diag(bfile, os.path.join(outdir, "smalln_grm.pdf"))
    if "intro" in which:
        pca_intro(bfile, os.path.join(outdir, "pca_intro.pdf"))
    if "predict" in which:
        pca_predict(bfile, outdir)
    if "predict3" in which:  # r = 3: the number of ancestry axes in C
        pca_predict(bfile, outdir, rs=(3,))
    if "predict_A" in which:
        pca_predict(bfile, outdir, centred=False)
    if "resid" in which:
        pca_resid(bfile, os.path.join(outdir, "pca_resid.pdf"))
    if "resid_rel" in which:
        pca_resid_rel(bfile, os.path.join(outdir, "pca_resid_rel.pdf"))
    if "resid_rel_fixed" in which:
        pca_resid_rel(bfile, os.path.join(outdir, "pca_resid_rel_fixed.pdf"), fixed=True)
