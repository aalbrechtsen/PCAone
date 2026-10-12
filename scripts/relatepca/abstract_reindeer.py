#!/usr/bin/env python3
"""Graphical abstract of the reindeer report. A: the approach, a toy GRM split
into structure L, related pairs S and per-animal noise D, and what each method
does with the parts. B: related pairs called per method (KING: 2nd degree or
closer), both data sets, all SNPs and LD-pruned. C: dwg weight 1/sqrt(v_i) of
each animal by site (new data, all SNPs). D: F per animal from the standard
top 6 PCs and from the PCP smoother (no number of PCs), both data sets.

usage: abstract_reindeer.py <reindeer dir> <out.pdf>
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slides_reindeer import read_fam  # noqa: E402

plt.rcParams.update({"font.size": 11})
INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"
C_STD, C_AAR, C_DWG, C_KING, C_HAP = "#0072B2", "#D55E00", "#009E73", "#777777", "#56B4E9"


def n_pairs(path):
    return len(list(open(path))) - 1


def toy():
    """12 animals in 3 sites: structure, one sib pair and one MZ pair, noise
    larger in the third (more diverse) site"""
    rng = np.random.default_rng(3)
    Z = np.kron(np.eye(3), np.ones((4, 1)))
    L = Z @ np.array([[0.5, 0.1, 0.0], [0.1, 0.4, 0.15], [0.0, 0.15, 0.3]]) @ Z.T
    L -= L.mean()
    S = np.zeros((12, 12))
    S[1, 2] = S[2, 1] = 0.45
    S[8, 9] = S[9, 8] = 0.9
    v = np.r_[np.full(8, 0.6), np.full(4, 1.4)] * rng.uniform(0.9, 1.1, 12)
    return L, S, np.diag(v)


def panel_a(fig, gs):
    L, S, D = toy()
    mats = [(L + S + D, "GRM"), (L, "structure $L$\n(low rank)"), (S, "related pairs $S$\n(sparse)"),
            (D, "noise $D$\n(one $v_i$ per animal)")]
    sub = gs.subgridspec(1, 7, width_ratios=[1, 0.18, 1, 0.18, 1, 0.18, 1], wspace=0.05)
    vmax = np.abs(L + S + D).max()
    for k, (M, t) in enumerate(mats):
        a = fig.add_subplot(sub[0, 2 * k])
        a.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=11)
        if k < 3:
            s = fig.add_subplot(sub[0, 2 * k + 1])
            s.axis("off")
            s.text(0.5, 0.5, "=" if k == 0 else "+", ha="center", va="center", fontsize=22, color=INK)
    return sub


def main():
    d, out = sys.argv[1], sys.argv[2]
    fig = plt.figure(figsize=(14, 13.2))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.85, 0.72, 1.25, 0.7], hspace=0.36, wspace=0.32)
    panel_a(fig, gs[0, :])
    fig.text(0.07, 0.975, "A  The GRM = structure + related pairs + per-animal noise; the methods differ in what they do with each part",
             fontsize=13, fontweight="bold", color=INK)
    # what the methods do with the parts: the three choices
    ta = fig.add_subplot(gs[1, :])
    ta.axis("off")
    head = ["", "GRM diagonal $D$\n(own heterozygosity)", "related pairs $S$", "weight of each animal", "PCs from"]
    rows = [(C_STD, "standard PCA", "observed", "kept", "equal", "GRM"),
            (C_AAR, "aarobust-kin+diag", "PCP imputation\n(soft threshold, no k)", "PCP imputation",
             "equal", "fitted structure $L$"),
            (C_DWG, "dwg", "PC imputation\n(rank-limited fit, rank\nfrom the noise edge)",
             "PC imputation,\nthen whitened out", "whitening: $1/\\sqrt{v_i}$\n(heterozygosity\ndifferences)",
             "whitened genotypes")]
    xs = [0.0, 0.17, 0.39, 0.58, 0.79]
    for x, h in zip(xs, head):
        ta.text(x, 0.97, h, transform=ta.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="top")
    ta.plot([0, 1], [0.74, 0.74], transform=ta.transAxes, color=MUTED, lw=0.8)
    for r, (c, name, *cells) in enumerate(rows):
        y = 0.64 - r * 0.22
        ta.add_patch(plt.Rectangle((0.0, y - 0.025), 0.008, 0.05, transform=ta.transAxes, color=c))
        ta.text(0.014, y, name, transform=ta.transAxes, fontsize=11, fontweight="bold", color=INK, va="center")
        for x, t in zip(xs[1:], cells):
            ta.text(x, y, t, transform=ta.transAxes, fontsize=9.5, color=INK, va="center", linespacing=1.05)
    ta.text(0.0, -0.02, "All robust methods find related pairs with the same rule, $\\phi_{ij} = S_{ij}/\\sqrt{v_i v_j}"
            " > 2^{-3.5}$, among candidates from KING and evalAdmix; PCP must use the uncentred GRM.",
            transform=ta.transAxes, fontsize=9.5, color=MUTED, va="top")
    # B: pairs
    ab = fig.add_subplot(gs[2, 0])
    meth = [("KING", None, C_KING), ("aar.", "aar_all.relpairs", C_AAR),
            ("dwg", "dwg_all.relpairs", C_DWG)]
    king = {("new", "full"): 2, ("new", "ld"): 2, ("old", "full"): 20, ("old", "ld"): 116}
    xs = []
    for g, ds in enumerate(("new", "old")):
        for k, (lab, f, c) in enumerate(meth):
            for s, sub in enumerate(("full", "ld")):
                x = g * 4.0 + k * 1.15 + s * 0.48
                n = king[(ds, sub)] if f is None else n_pairs(os.path.join(d, ds, sub, f))
                ab.bar(x, max(n, 0.6), width=0.44, color=c if sub == "full" else "white", ec=c, lw=1.6,
                       hatch=None if sub == "full" else "///")
                ab.text(x, max(n, 0.6) * 1.12, str(n), ha="center", va="bottom", fontsize=8.5, color=INK)
            xs.append((g * 4.0 + k * 1.15 + 0.24, lab))
    ab.set_yscale("log")
    ab.set_ylim(0.5, 600)
    ab.set_xticks([x for x, _ in xs], [l for _, l in xs], fontsize=8.5)
    for g, t in enumerate(("new data (96)", "course data (101)")):
        ab.text(g * 4.0 + 1.4, 380, t, ha="center", fontsize=10.5, color=INK)
    from matplotlib.patches import Patch
    ab.legend(handles=[Patch(fc=MUTED, ec=MUTED, label="all SNPs"), Patch(fc="white", ec=MUTED, hatch="///", label="LD-pruned")],
              frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ab.set_ylabel("related pairs called")
    ab.set_title("B  Related pairs", loc="left", fontsize=13, fontweight="bold")
    # C: dwg weights by site
    ac = fig.add_subplot(gs[2, 1])
    ids, pops = read_fam(os.path.join(d, "new", "rd_full.fam"))
    pops = np.array(pops)
    w = 1 / np.sqrt(np.loadtxt(os.path.join(d, "new", "full", "dwg_all.noise")))
    w /= np.median(w)
    order = sorted(set(pops), key=lambda p: np.median(w[pops == p]))
    for k, p in enumerate(order):
        y = w[pops == p]
        ac.scatter(y, k + np.random.default_rng(k).uniform(-0.18, 0.18, len(y)), s=22, color=C_DWG,
                   edgecolors="white", lw=0.4, zorder=3)
        ac.vlines(np.median(y), k - 0.35, k + 0.35, color=INK, lw=1.6, zorder=4)
    ac.axvline(1, color=MUTED, ls="--", lw=0.9)
    ac.set_xscale("log", base=2)
    ac.set_xticks([0.5, 1, 2], ["1/2", "1", "2"])
    ac.set_yticks(range(len(order)), [p.replace("_", " ").replace("Kangerlussuaq Sisimiut", "Kanger.-Sisimiut")
                                      for p in order], fontsize=9)
    ac.set_xlabel("dwg weight $1/\\sqrt{v_i}$ / median (new data)")
    ac.set_title("C  Very different noise per site", loc="left", fontsize=13, fontweight="bold")
    ac.grid(axis="x", color=GRID, lw=0.5)
    # D: inbreeding
    ad = fig.add_subplot(gs[2, 2])

    def read_F(ds, name):
        return np.array([float(l.split()[-1]) for l in list(open(os.path.join(d, ds, name)))[1:]])

    cols = [("new", "std6inb.inbred", "new\nstd. 6 PCs", C_STD), ("new", "smooth_pcp.inbred", "new\nPCP", C_DWG),
            ("old", "std6inb.inbred", "course\nstd. 6 PCs", C_STD), ("old", "smooth_pcp.inbred", "course\nPCP", C_DWG)]
    for k, (ds, f, lab, c) in enumerate(cols):
        F = read_F(ds, f)
        ad.scatter(k + np.random.default_rng(k).uniform(-0.22, 0.22, len(F)), F, s=18, color=c, edgecolors="white",
                   lw=0.4, zorder=3)
        ad.hlines(np.median(F), k - 0.32, k + 0.32, color=INK, lw=1.6, zorder=4)
    ad.axhline(0, color=MUTED, ls="--", lw=0.9)
    ad.set_xticks(range(4), [c[2] for c in cols], fontsize=9)
    ad.set_ylabel("inbreeding F")
    ad.set_title("D  Inbreeding F: PCs vs PCP", loc="left", fontsize=13, fontweight="bold")
    ad.grid(axis="y", color=GRID, lw=0.5)
    for a in (ab, ac, ad):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    # E: animals per site and animals in a related pair (any method, all SNPs)
    ae = fig.add_subplot(gs[3, :])
    norm = lambda x: x.replace("/", "_").replace("-", "_")
    data = {}
    for ds in ("new", "old"):
        site_of = {l.split()[1]: norm(l.split()[0]) for l in open(os.path.join(d, ds, "rd_full.fam"))}
        rel = set()
        for l in list(open(os.path.join(d, ds, "kt.kin0" if ds == "new" else "kt_full.kin0")))[1:]:
            x = l.split()
            if float(x[-1]) > 2 ** -3.5:
                rel |= {x[1], x[3]}
        for f in ("aar_all.relpairs", "dwg_all.relpairs"):
            for l in list(open(os.path.join(d, ds, "full", f)))[1:]:
                rel |= {w.replace("_reftest", "") for w in l.split()[:2]}
        data[ds] = (site_of, rel)
    sites = sorted(set(data["new"][0].values()) | set(data["old"][0].values()))
    short = {"Akia_Maniitsoq": "Akia", "Kangerlussuaq_Sisimiut": "Kanger.-Sisimiut", "South_Greenland": "South Greenland"}
    for g, (ds, lab, c) in enumerate((("new", "new data (96)", C_STD), ("old", "course data (101)", C_DWG))):
        site_of, rel = data[ds]
        tot = np.array([sum(v == st for v in site_of.values()) for st in sites])
        nrel = np.array([sum(v == st and k in rel for k, v in site_of.items()) for st in sites])
        x = np.arange(len(sites)) + (g - 0.5) * 0.4
        ae.bar(x, tot, width=0.38, color="white", ec=c, lw=1.4, label=f"{lab}: all animals")
        ae.bar(x, nrel, width=0.38, color=c, ec=c, lw=1.4, label=f"{lab}: in a related pair")
        for xi, t_, r_ in zip(x, tot, nrel):
            ae.text(xi, t_ + 0.3, f"{r_}/{t_}", ha="center", va="bottom", fontsize=8, color=INK)
    ae.set_xticks(range(len(sites)), [short.get(st, st.replace("_", " ")) for st in sites], fontsize=9.5, rotation=20,
                  ha="right", rotation_mode="anchor")
    ae.set_ylabel("animals")
    ae.set_ylim(0, 31)
    ae.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    ae.grid(axis="y", color=GRID, lw=0.5)
    for sp in ("top", "right"):
        ae.spines[sp].set_visible(False)
    ae.set_title("E  Animals per site, and how many are in a related pair (KING 2nd degree or closer, aarobust-kin or dwg; all SNPs)",
                 loc="left", fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
