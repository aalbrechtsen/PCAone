#!/usr/bin/env python3
"""Graphical abstracts of the three sample-size regimes of PCAone --robust
(small, large and very large N) for the methods document.

Real data in every data panel:
  small N      the admixTjeck2 example of methods_figs.py (N = 48), with the
               final relatedness from PCAone --robust aarobust-kin --impute-diag
  large N      the simulated N = 2000 data set (hn_N2000_rel30_r0): CS
               spectrum and the final relatedness from PCAone --robust
               detect-white; the matrices are the example's (N = 48 is
               readable, N = 2000 is not)
  very large N the simulated N = 20,000 data set (hn_N20000_rel30_r0): count
               sketch correlations against the plink2 KING truth
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import methods_figs as F  # noqa: E402
from methods_figs import INK, MUTED, PCOL, POPS  # noqa: E402
from relfind import CLASSES, read_kin0  # noqa: E402
from relfind_bits import sketch  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

REL = {"MZ": (0.0, 1.0), "parent-offspring": (1.0, 0.0), "full sibs": (0.5, 0.25), "2nd degree": (0.5, 0.0)}  # (k1, k2)
RCOL = {"MZ": "#CC79A7", "parent-offspring": "#0072B2", "full sibs": "#E69F00", "2nd degree": "#009E73",
        "3rd degree": "#56B4E9", "random pairs": "#999999"}


def grid2(fig, top=0.80, bottom=0.12, hgap=0.13, left=0.03, right=0.97, wgap=0.095, widths=(1, 1, 1)):
    """2 rows x 3 panels in reading order"""
    h = (top - bottom - hgap) / 2
    w = (right - left - wgap * 2) / sum(widths)
    axes = []
    for r in range(2):
        y = top - h - r * (h + hgap)
        x = left
        for wi in widths:
            axes.append(fig.add_axes([x, y, wi * w, h]))
            x += wi * w + wgap
    return axes


def step(ax, n, title):
    ax.set_title(f"{n}  {title}", color=INK, loc="left", fontsize=9)


def read_relpairs(fn):
    rows = [l.split() for l in open(fn)][1:]
    return [(r[0], r[1], *map(float, r[2:])) for r in rows]


def header(fig, title, sub, col):
    fig.add_artist(FancyBboxPatch((0.01, 0.915), 0.98, 0.07, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc=col, ec="none"))
    fig.text(0.02, 0.95, title, fontsize=12, weight="bold", color=INK, va="center")
    fig.text(0.98, 0.95, sub, fontsize=8, color=INK, va="center", ha="right")


def footer(fig, text, y=0.015):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.3)


def ibd_panel(ax, k1, k2, cls, title):
    """k1 against k2 of the related pairs (k0 = 1 - k1 - k2: distance to the
    diagonal); expected values as open circles"""
    ax.plot([0, 1], [1, 0], color="#dddddd", lw=0.8, zorder=0)
    for name, (e1, e2) in REL.items():
        m = cls == name
        if m.any():
            ax.scatter(k1[m], k2[m], s=12, color=RCOL[name], edgecolor="white", linewidth=0.3, zorder=2,
                       label=f"{name} ({m.sum()})")
        ax.scatter([e1], [e2], s=70, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("$k_1$", color=MUTED, labelpad=1)
    ax.set_ylabel("$k_2$", color=MUTED, labelpad=1)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.legend(fontsize=5.8, frameon=False, loc="upper right", handletextpad=0.2, borderaxespad=0.2)
    ax.set_title(title, color=INK)


def classify(kin, k0):
    return np.where(kin > 0.354, "MZ", np.where(kin > 0.177, np.where(k0 < 0.1, "parent-offspring", "full sibs"),
                                                 "2nd degree"))


def text_panel(ax, title, lines):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01,rounding_size=0.04",
                                transform=ax.transAxes, fc="#f7f7f7", ec="#cccccc", lw=0.7))
    ax.text(0.5, 0.93, title, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, weight="bold",
            color=INK)
    ax.text(0.08, 0.78, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=INK,
            linespacing=1.45)


# ------------------------------------------------------------------ small N
def fig_small(c, o, pop, rel, rp, out):
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    fig = plt.figure(figsize=(10, 7.4))
    header(fig, "Small N  (N $\\leq$ 1000):  aarobust-kin, dense engine",
           "--robust (auto) [--impute-diag]", "#e8f3e8")
    ax = grid2(fig)
    vmax = np.percentile(np.abs(c["C"][~np.eye(len(o), dtype=bool)]), 99)
    F.heat(ax[0], P(c["C"]), "", vmax=vmax, mask_diag=True, pop=pop)
    step(ax[0], 1, "one pass: GRM $C$\n    diagonal unobserved (grey)")
    Kc = np.where(c["cand"], c["king"], 0)
    np.fill_diagonal(Kc, 0)
    F.heat(ax[1], P(Kc), "", sparse=True, pop=pop)
    step(ax[1], 2, "KING-robust > 0.04:\n    the only pairs allowed in $S$")
    F.heat(ax[2], P(c["Lc"]), "", vmax=vmax, pop=pop)
    step(ax[2], 3, "PCP $C = L + S$:\n    structure $L$, diagonal imputed")
    Sd = P(c["Sc"]).copy()
    np.fill_diagonal(Sd, 0)
    F.heat(ax[3], Sd, "", sparse=True, pop=pop)
    step(ax[3], 4, "$S$: related pairs\n    ($\\hat\\phi > \\tau$, candidates only)")
    F.pcs(ax[4], c["U_aar"][o], pop, rel, "", 1, 2)
    step(ax[4], 5, "PCs of $L$ (--impute-diag)\n    relatives as stars")
    k0 = np.array([r[5] for r in rp])
    k1 = np.array([r[6] for r in rp])
    k2 = np.array([r[7] for r in rp])
    kin = np.array([r[4] for r in rp])
    ibd_panel(ax[5], k1, k2, classify(kin, k0), "")
    step(ax[5], 6, "final relatedness (1 pass)\n    IAF from the PCs; o = expected")
    fig.canvas.draw()
    F.arrow(fig, ax[0], ax[1])
    F.arrow(fig, ax[1], ax[2])
    F.arrow(fig, ax[3], ax[4])
    F.arrow(fig, ax[4], ax[5])
    F.legend_row(fig, 0.065)
    footer(fig, "All N $\\times$ N in memory after one pass; each PCP iteration needs a full eigendecomposition "
           "(N = 500: 7.5 s; N = 1000: 33 s).\nNo HWE assumption and no $K$.  At small N the observed diagonal "
           "reflects heterozygosity, not structure: use --impute-diag.\nExample: admixTjeck2, N = 48 with 8 "
           "relatives.", y=0.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ large N
def cs_spectrum(bfile, relpairs, nev=14):
    fam, _, G = read_bed(bfile)
    g = G.astype(float)
    N, M = g.shape
    D = (g * (2 - g)).mean(1)
    H = g @ g.T / M
    H[np.diag_indices(N)] -= D
    import illustrate as I
    edge = I.cs_noise_edge(g)
    ix = {r[1]: i for i, r in enumerate(fam)}
    S = np.zeros((N, N))
    for a, b, king, kd, *_ in relpairs:
        i, j = ix[a], ix[b]
        S[i, j] = S[j, i] = 2 * kd * np.sqrt(D[i] * D[j])
    ev_h = np.linalg.eigvalsh(H)[::-1][:nev]
    ev_hs = np.linalg.eigvalsh(H - S)[::-1][:nev]
    return ev_h, ev_hs, edge


def fig_large(c, o, pop, rel, spec, rp2, out):
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    N = len(o)
    fig = plt.figure(figsize=(10, 7.4))
    header(fig, "Large N  (N $\\leq$ 20,000):  dwg (detection + whitening, shown on the CS scale)",
           "--robust (auto); operator engine", "#e6eef8")
    ax = grid2(fig)
    Kc = np.where(c["cand"], c["king"], 0)
    np.fill_diagonal(Kc, 0)
    F.heat(ax[0], P(Kc), "", sparse=True, pop=pop)
    step(ax[0], 1, "pass 1 + KING over all pairs\n    (popcounts): candidates > 0.04")
    ev_h, ev_hs, edge = spec
    x = np.arange(1, len(ev_h) + 1)
    ax[1].bar(x - 0.2, ev_h, width=0.4, color="#bbbbbb", label="$H$ (CS matrix)")
    ax[1].bar(x + 0.2, ev_hs, width=0.4, color="#0072B2", label="$H-S$")
    ax[1].axhline(edge, color="#D55E00", lw=1, ls="--")
    ax[1].text(len(x) + 0.5, edge * 1.08, "noise edge $2\\sigma\\sqrt{N}$", color="#D55E00", fontsize=7,
               va="bottom", ha="right")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x[::2])
    ax[1].set_xlabel("eigenvalue", color=MUTED, labelpad=1)
    ax[1].tick_params(labelsize=7, colors=MUTED)
    ax[1].legend(fontsize=7, frameon=False, loc="upper right")
    r = int((ev_hs > edge).sum())
    step(ax[1], 2, f"rank from the noise edge\n    N = 2000: rank {r}")
    Sd = P(c["S"]).copy()
    np.fill_diagonal(Sd, 0)
    F.heat(ax[2], Sd, "", sparse=True, pop=pop)
    step(ax[2], 3, "fit $H \\approx L + S$: pairs with\n    $\\hat\\phi > \\tau$ among the candidates")
    Sig = F.sigma(c["noise"], c["pairs"], N)
    F.heat(ax[3], P(Sig), "", sparse=True, pop=pop, diag_dots=True)
    step(ax[3], 4, "$\\Sigma = v^{1/2}(I+2\\Psi)v^{1/2}$\n    block diagonal over families")
    F.pcs(ax[4], c["U_dw"][o], pop, rel, "", 1, 2)
    step(ax[4], 5, "PCs of $\\Sigma^{-1/2}G$,\n    mapped back by $\\Sigma^{1/2}$")
    k0 = np.array([r[5] for r in rp2])
    k1 = np.array([r[6] for r in rp2])
    k2 = np.array([r[7] for r in rp2])
    kin = np.array([r[4] for r in rp2])
    ibd_panel(ax[5], k1, k2, classify(kin, k0), "")
    step(ax[5], 6, "final relatedness (1 pass)\n    N = 2000, 538 pairs")
    fig.canvas.draw()
    F.arrow(fig, ax[0], ax[1])
    F.arrow(fig, ax[1], ax[2])
    F.arrow(fig, ax[3], ax[4])
    F.arrow(fig, ax[4], ax[5])
    F.legend_row(fig, 0.065)
    footer(fig, "KING over all pairs: $N^2M/64$ popcounts (6.5 s at N = 20,000).  Fit: each iteration one pass "
           "$G(G'V)$, nothing N $\\times$ N.\ndwg at N = 20,000: 26.5 s in-core (standard PCAone 24 s).  Panels 1, 3-5: "
           "the N = 48 example (readable); 2 and 6: simulated N = 2000.", y=0.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- very large N
def sketch_corr(bfile, kin0, s=2048, nrand=200000, seed=0):
    fam, _, G = read_bed(bfile)
    G = G.astype(np.int8)
    N = len(G)
    Y = sketch(G, s, 1, kind="count")
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    truth = read_kin0(kin0, [r[1] for r in fam])
    out = {}
    for name, lo, hi in CLASSES:
        P = np.array([p for p, k in truth.items() if lo < k <= hi])
        if len(P):
            out[name] = np.einsum("ij,ij->i", Y[P[:, 0]], Y[P[:, 1]])
    rng = np.random.default_rng(seed)
    i, j = rng.integers(N, size=nrand), rng.integers(N, size=nrand)
    m = (i != j) & np.array([(min(a, b), max(a, b)) not in truth for a, b in zip(i, j)])
    out["random pairs"] = np.einsum("ij,ij->i", Y[i[m]], Y[j[m]])
    return out, N


def fig_verylarge(sk, Nsk, timing, out):
    fig = plt.figure(figsize=(10, 7.4))
    header(fig, "Very large N  (N > 20,000):  sketch search, then dwg",
           "--robust (auto); --king-search auto", "#f6ecf4")
    ax = grid2(fig, widths=(1, 1.25, 0.9))
    # 1. count sketch schematic
    a = ax[0]
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis("off")
    rng = np.random.default_rng(3)
    Gs = rng.integers(0, 3, (10, 16))
    cols = rng.integers(0, 4, 16)
    ccol = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    cm = matplotlib.colors.ListedColormap(["#f4f4f4", "#9ecae1", "#08519c"])
    a.imshow(Gs, cmap=cm, vmin=-0.5, vmax=2.5, extent=(0.0, 0.55, 0.12, 0.72), aspect="auto",
             interpolation="nearest")
    for j in range(16):
        a.add_patch(plt.Rectangle((0.55 * j / 16, 0.74), 0.55 / 16, 0.05, color=ccol[cols[j]], lw=0))
        a.text(0.55 * (j + 0.5) / 16, 0.81, "+" if rng.random() < 0.5 else "$-$", fontsize=6, ha="center",
               color=MUTED)
    Ys = rng.normal(size=(10, 4))
    a.imshow(Ys, cmap="RdBu_r", extent=(0.72, 0.98, 0.12, 0.72), aspect="auto", interpolation="nearest")
    for q in range(4):
        a.add_patch(plt.Rectangle((0.72 + 0.26 * q / 4, 0.74), 0.26 / 4, 0.05, color=ccol[q], lw=0))
    a.annotate("", xy=(0.71, 0.42), xytext=(0.57, 0.42), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    a.text(0.275, 0.03, "genotypes (N $\\times$ M)", ha="center", fontsize=7, color=MUTED)
    a.text(0.85, 0.03, "sketch $Y$ (N $\\times$ s)", ha="center", fontsize=7, color=MUTED)
    step(a, 1, "count sketch in pass 1: SNP $j$\n    to column $h(j)$ with sign $\\pm$")
    # 2. sketch correlations
    a = ax[1]
    bins = np.linspace(-0.12, 1.02, 115)
    order = ["random pairs", "3rd degree", "2nd degree", "1st degree", "MZ/dup (>0.354)"]
    lab = {"MZ/dup (>0.354)": "MZ", "1st degree": "1st degree", "2nd degree": "2nd degree",
           "3rd degree": "3rd degree", "random pairs": "random pairs"}
    colr = {"MZ/dup (>0.354)": RCOL["MZ"], "1st degree": RCOL["parent-offspring"], "2nd degree": RCOL["2nd degree"],
            "3rd degree": RCOL["3rd degree"], "random pairs": RCOL["random pairs"]}
    for name in order:
        if name in sk:
            a.hist(sk[name], bins=bins, color=colr[name], alpha=0.85, label=f"{lab[name]} ({len(sk[name]):,})",
                   density=True, histtype="stepfilled", lw=0)
    sd = sk["random pairs"].std()
    a.axvline(np.sqrt(2 * np.log(Nsk)) * sd, color=INK, lw=0.7, ls=":")
    a.set_yscale("log")
    a.set_ylim(5e-3, 300)
    a.tick_params(labelsize=7, colors=MUTED)
    a.set_xlabel("correlation of sketch rows", color=MUTED, labelpad=1)
    a.legend(fontsize=6.5, frameon=False, loc="upper right")
    step(a, 2, f"relatives stand out  (N = {Nsk:,}, s = 2048)\n    unrelated: sd {sd:.3f} $\\approx 1/\\sqrt{{s}}$")
    # 3. neighbours
    text_panel(ax[2], "", [
        "top m = 5 per person:",
        "blocked float products,",
        "cost $N^2 s$, not $N^2 M$",
        "",
        "all m are relatives?",
        "  search 2m, 4m, ...",
        "",
        "exact KING-robust on",
        "~3.5 candidates / person"])
    step(ax[2], 3, "nearest neighbours")
    # 4. recall
    a = ax[3]
    names = ["MZ", "1st", "2nd", "3rd"]
    rec = [1.0, 1.0, 1.0, 0.988]
    a.bar(range(4), rec, color=[RCOL["MZ"], RCOL["parent-offspring"], RCOL["2nd degree"], RCOL["3rd degree"]],
          width=0.6)
    for x, v in enumerate(rec):
        a.text(x, v + 0.01, f"{100 * v:.1f}%" if v < 1 else "100%", ha="center", va="bottom", fontsize=7, color=INK)
    a.set_xticks(range(4))
    a.set_xticklabels(names, fontsize=7.5)
    a.set_ylim(0, 1.15)
    a.tick_params(labelsize=7, colors=MUTED)
    step(a, 4, "recall vs all-pairs KING\n    (N = 20,000; same pairs at 100k)")
    # 5. timing
    a = ax[4]
    labs = ["KING\nall pairs", "sketch\nsearch"]
    vals = [timing["king_all"], timing["sketch"]]
    a.bar([0, 1], vals, color=["#bbbbbb", "#CC79A7"], width=0.6)
    for x, v in zip([0, 1], vals):
        a.text(x, v, f"{v:.0f} s", ha="center", va="bottom", fontsize=7.5, color=INK)
    a.set_xticks([0, 1])
    a.set_xticklabels(labs, fontsize=7.5)
    a.tick_params(labelsize=7, colors=MUTED)
    a.set_ylim(0, max(vals) * 1.2)
    a.set_ylabel("seconds", color=MUTED, fontsize=7)
    step(a, 5, "finding the relatives\n    N = 100,000, 16 threads")
    # 6. rest
    text_panel(ax[5], "", [
        "dwg fit, residual-sketch",
        "evalAdmix + $k_0$ screen,",
        "whitening (operator engine)",
        "",
        "final relatedness:",
        "$\\phi$, $k_0$, $k_1$, $k_2$ (1 pass)",
        "",
        f"N = 100k: dwg {timing['dwg']:.0f} s",
        f"(detect-white {timing['total_sketch']:.0f} s;",
        f" standard PCAone {timing['standard']:.0f} s)"])
    step(ax[5], 6, "then as for large N")
    fig.canvas.draw()
    F.arrow(fig, ax[0], ax[1])
    F.arrow(fig, ax[1], ax[2])
    F.arrow(fig, ax[3], ax[4])
    F.arrow(fig, ax[4], ax[5])
    footer(fig, "Sketch: N $\\times$ M (columns and signs from a hash of the SNP index; in-core = out-of-core).  "
           "Search: $N^2 s$, independent of M.\nRelatives: correlation about $2\\phi$; unrelated: 0 $\\pm 1/\\sqrt{s}$ "
           "(dotted: expected maximum over N).  1-bit signs and LSH bucketing do not work.", y=0.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True, help="admixTjeck2")
    ap.add_argument("--n2k", required=True, help="simulated N = 2000 PLINK prefix")
    ap.add_argument("--n20k", required=True, help="simulated N = 20,000 PLINK prefix (with _king.kin0)")
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--timing", required=True,
                    help="king_all,sketch,total_all,total_sketch,standard,dwg (seconds, N = 100k)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    # small N example and its final relatedness from PCAone
    ds, T = F.dataset(a.bfile)
    c = F.compute(ds)
    n0, N = ds["n0"], len(ds["G"])
    pop = F.family_pop(ds)
    rel = np.arange(N) >= n0
    o = F.order_rows(pop, n0, ds["Kped"])
    ex = os.path.join(a.work, "example")
    write_bed(ex, [[f"r{i}", f"r{i}", "0", "0", "0", "-9"] for i in range(N)],
              [f"1\ts{j}\t0\t{j + 1}\tA\tC" for j in range(ds["G"].shape[1])], ds["G"].astype(np.int8))
    subprocess.run(f"{a.pcaone} -b {ex} -k 3 -n 8 --robust aarobust-kin --impute-diag -o {ex}_aar", shell=True,
                   check=True, capture_output=True)
    rp = read_relpairs(ex + "_aar.relpairs")
    fig_small(c, o, pop[o], rel[o], rp, os.path.join(a.out, "regime_small.pdf"))
    # large N
    p2 = os.path.join(a.work, "n2k")
    subprocess.run(f"{a.pcaone} -b {a.n2k} -k 4 -n 16 --robust detect-white -o {p2}", shell=True, check=True,
                   capture_output=True)
    rp2 = read_relpairs(p2 + ".relpairs")
    spec = cs_spectrum(a.n2k, rp2)
    fig_large(c, o, pop[o], rel[o], spec, rp2, os.path.join(a.out, "regime_large.pdf"))
    # very large N
    sk, Nsk = sketch_corr(a.n20k, a.n20k + "_king.kin0")
    t = dict(zip(["king_all", "sketch", "total_all", "total_sketch", "standard", "dwg"], map(float, a.timing.split(","))))
    fig_verylarge(sk, Nsk, t, os.path.join(a.out, "regime_verylarge.pdf"))
    with open(os.path.join(a.out, "regimes.txt"), "w") as f:
        f.write(f"example pairs {len(rp)}; N=2000 pairs {len(rp2)}; spectrum H {np.round(spec[0], 3)}; "
                f"H-S {np.round(spec[1], 3)}; edge {spec[2]:.3f}\n")
        for k, v in sk.items():
            f.write(f"sketch {k}: n={len(v)} mean={v.mean():.3f} sd={v.std():.3f} min={v.min():.3f} max={v.max():.3f}\n")


if __name__ == "__main__":
    main()
