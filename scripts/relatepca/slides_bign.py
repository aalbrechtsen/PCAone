#!/usr/bin/env python3
"""Slide figures: standard PCA as N grows, real genotypes (admixTjeck2).
The same 8 relatives as the first hook (scenario "many") with 5 .. 60
unrelated individuals per population. Truth: PCA of all 366 admixTjeck2
individuals, every sample projected onto it. One figure per PC pair
(1/2, 2/3, 3/4) with one panel per N (raw PCs, signs matched to the truth),
and the family share eta^2 of PC1-6 at every N.

usage: slides_bign.py <admixTjeck2 bfile> <outdir> [--ns 5,10,20,40,60]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import highn_test as H  # noqa: E402
import illustrate as I  # noqa: E402
import methods_figs as F  # noqa: E402
import slides_hook as S  # noqa: E402
from methods_figs import INK, MUTED, PCOL, POPS  # noqa: E402
from sim import read_bed  # noqa: E402

K = 8


def full_truth(Gall, cols, G, k=4):
    """PCA of all individuals of the panel, the sample projected onto it"""
    Gr = Gall[:, cols].astype(float)
    f = Gr.mean(0) / 2
    ok = (f > 0) & (f < 1)
    sd = np.sqrt(2 * f[ok] * (1 - f[ok]))
    Xr = (Gr[:, ok] - 2 * f[ok]) / sd
    _, v = I.top_eig(Xr @ Xr.T / ok.sum(), k)
    T = ((G[:, ok] - 2 * f[ok]) / sd) @ (Xr.T @ v)
    return T / np.linalg.norm(T, axis=0)


def run_n(bfile, Gall, n):
    ds, _ = F.dataset(bfile, n=n, scen="many")
    G = ds["G"].astype(float)
    N = len(G)
    pop = F.family_pop(ds)
    rel = np.arange(N) >= ds["n0"]
    T = S.orient_truth(full_truth(Gall, ds["cols"], G), pop)
    X = I.standardize(G)
    X = X[:, np.isfinite(X).all(0)]
    U = I.top_eig(X @ X.T / X.shape[1], K)[1]
    base = np.where(~rel)[0]
    U = S.align_signs(U, T, base)
    r2, err = H.score(U, T[:, :3], base, np.where(rel)[0], 3)
    r2_12 = H.score(U, T[:, :2], base, np.where(rel)[0], 2)[0]  # the PC1/2 plot: true axes 1-2 on PC1-2 only
    # families: each relative with the individuals it is related to (KING > tau)
    king = B.king_robust(G)
    famid = np.zeros(N, int)
    for f, r in enumerate(np.where(rel)[0], 1):
        mem = np.where(king[r] > B.TAU)[0]
        old = famid[mem][famid[mem] > 0]
        f = old.min() if len(old) else f
        famid[mem] = f
        famid[r] = f
    eta = [H.family_eta2(U[:, j], famid) for j in range(K)]
    return dict(n=n, N=N, U=U, pop=pop, rel=rel, r2=r2, r2_12=r2_12, err=err, eta=eta, famid=famid)


def figure(res, a, b, out):
    fig, axs = plt.subplots(1, len(res), figsize=(2.7 * len(res), 3.3))
    for ax, d in zip(axs, res):
        s = 30 if d["N"] <= 60 else 14 if d["N"] <= 120 else 8
        r2 = d["r2_12"] if (a, b) == (0, 1) else d["r2"]
        S.scatter(ax, d["U"], d["pop"], d["rel"], f"{d['n']} per population ($R^2$ = {r2.min():.2f})", a=a, b=b)
        for c in ax.collections:
            if c.get_paths() and c.get_sizes()[0] < 100:
                c.set_sizes([s])
        ax.title.set_fontsize(11)
        ax.set_xlabel(f"PC{a + 1} (family share = {d['eta'][a]:.2f})", color=MUTED, fontsize=9)
        ax.set_ylabel(f"PC{b + 1} (family share = {d['eta'][b]:.2f})", color=MUTED, fontsize=9)
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    hs.append(plt.Line2D([], [], marker="*", ls="", color="#bbbbbb", markeredgecolor=INK, markersize=12))
    fig.legend(hs, POPS + ["relative"], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06),
               fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_rows(res, pairs, out):
    """one row per PC pair, one column per N; the N and R2 only above the top row"""
    fig, axs = plt.subplots(len(pairs), len(res), figsize=(2.7 * len(res), 2.9 * len(pairs) + 0.4))
    for row, (a, b) in zip(axs, pairs):
        for ax, d in zip(row, res):
            s = 30 if d["N"] <= 60 else 14 if d["N"] <= 120 else 8
            title = f"{d['n']} per population ($R^2$ = {d['r2'].min():.2f})" if (a, b) == pairs[0] else ""
            S.scatter(ax, d["U"], d["pop"], d["rel"], title, a=a, b=b)
            for c in ax.collections:
                if c.get_paths() and c.get_sizes()[0] < 100:
                    c.set_sizes([s])
            ax.title.set_fontsize(11)
            ax.set_xlabel(f"PC{a + 1} (family share = {d['eta'][a]:.2f})", color=MUTED, fontsize=9)
            ax.set_ylabel(f"PC{b + 1} (family share = {d['eta'][b]:.2f})", color=MUTED, fontsize=9)
    hs = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=8) for p in POPS]
    hs.append(plt.Line2D([], [], marker="*", ls="", color="#bbbbbb", markeredgecolor=INK, markersize=12))
    fig.legend(hs, POPS + ["relative"], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03),
               fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def eta_figure(res, out, kk=3):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    blues = ["#DEEBF7", "#9ECAE1", "#4292C6", "#2171B5", "#08306B"]
    w = 0.8 / len(res)
    x = np.arange(1, K + 1)
    for i, d in enumerate(res):
        ax.bar(x - 0.4 + w * (i + 0.5), d["eta"], w, color=blues[i % len(blues)], label=f"{d['n']} per pop.")
    ax.axvspan(0.5, kk + 0.5, color="#EEEEEE", zorder=0)
    ax.text((kk + 1) / 2, 1.03, "ancestry axes", ha="center", fontsize=9, color=MUTED)
    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xlabel("PC")
    ax.set_ylabel(r"family share of the PC ($\eta^2$)")
    ax.legend(frameon=False, fontsize=9, ncol=len(res), loc="lower center", bbox_to_anchor=(0.5, 1.07))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bfile")
    ap.add_argument("outdir")
    ap.add_argument("--ns", default="5,10,20,40,60")
    a = ap.parse_args()
    _, _, Gall = read_bed(a.bfile)
    res = [run_n(a.bfile, Gall, n) for n in map(int, a.ns.split(","))]
    for d in res:
        print(f"n={d['n']} N={d['N']} R2 {np.round(d['r2'], 3)} R2 PC1-2 {np.round(d['r2_12'], 3)} rel err {d['err']:.2f} "
              f"eta2 PC1-6 {np.round(d['eta'], 2)}")
    os.makedirs(a.outdir, exist_ok=True)
    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        figure(res, x, y, os.path.join(a.outdir, f"bign_pc{x + 1}{y + 1}.pdf"))
    figure_rows(res, [(1, 2), (3, 4)], os.path.join(a.outdir, "bign_pc23_45.pdf"))
    eta_figure(res, os.path.join(a.outdir, "bign_eta.pdf"))


if __name__ == "__main__":
    main()
