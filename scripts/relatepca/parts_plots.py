#!/usr/bin/env python3
"""Figures for the scenario reports: key combinations against n (individuals
per population). Lines with distinct markers (identity never by colour alone),
categorical colours in fixed order (validated palette), recessive grid.

usage: parts_plots.py OUTDIR (reads results/parts_*.tsv)"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "/kellyData/home/albrecht/codex/relatePCA/results/"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MRK = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def lines(ax, d, series, y, ylabel, title, pct=False, ylim=None):
    """ylim: zoom; values outside are drawn at the edge as open markers with
    their true value written next to them (truncated, not hidden)"""
    for i, (lab, combo) in enumerate(series):
        x = d[d.combo == combo].groupby("n")[y].mean()
        if pct:
            x = 100 * x
        v = x.values.copy()
        out = np.zeros(len(v), bool)
        if ylim is not None:
            out = (v < ylim[0]) | (v > ylim[1])
            v = np.clip(v, *ylim)
        ax.plot(x.index, v, color=COL[i], lw=2, label=lab, marker=MRK[i], ms=6, clip_on=False,
                markevery=[j for j in range(len(v)) if not out[j]])
        for j in np.where(out)[0]:
            ax.plot(x.index[j], v[j], marker=MRK[i], ms=7, mfc="white", mec=COL[i], mew=1.6, clip_on=False)
            ax.annotate(f"{x.values[j]:.0f}" if pct else f"{x.values[j]:.2f}", (x.index[j], v[j]),
                        textcoords="offset points", xytext=(6, 4 if v[j] == ylim[0] else -11), fontsize=7,
                        color=MUTED)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20, 40] if d.n.max() >= 40 else [5, 10, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("individuals per population (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def legend(fig, axes, ncol=None):
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=ncol or len(lab), frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))


def s1(out):
    d = pd.read_csv(RES + "parts_S1.tsv", sep="\t")
    d = d[d.k == 3].copy()
    d["wrong"] = d.minR2 < 0.95
    ser = [("cGRM · none (standard PCA)", "standard"),
           ("uncGRM · topR_r_edge · KING+fam · phiR_r_edge · whiteC", "topr/unc/kf/0/v/whitec"),
           ("raw · topR_r_edge · KING+fam · phiR_D_r_edge · white", "topr/raw/kf/0/D/white"),
           ("raw · topR_r_edge · KING+fam · phiR_r_edge · L", "topr/raw/kf/0/v/L"),
           ("cGRM · PCP · KING · phiS · L (aarobust-kin --impute-diag)", "pcp/cgrm/k/0/v/L")]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    lines(ax[0], d[d.M == 0], ser, "minR2", "mean min R²", "all 54k SNPs", ylim=(0.9, 1.0))
    lines(ax[1], d[d.M == 10000], ser, "minR2", "mean min R²", "10,000 SNPs", ylim=(0.8, 1.0))
    lines(ax[2], d[d.M == 10000], ser, "wrong", "data sets wrong (%)", "10,000 SNPs: min R² < 0.95", pct=True,
          ylim=(0, 40))
    legend(fig, ax, 2)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(out, "S1_vs_n.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "S1_vs_n.png"), dpi=130, bbox_inches="tight")


def s1s(out):
    d = pd.concat([pd.read_csv(RES + "parts_S1s.tsv", sep="\t"), pd.read_csv(RES + "parts_S1s_attrib.tsv", sep="\t")])
    d = d[(d.k == 3) & (d.M == 0)].drop_duplicates(["n", "scen", "rep", "combo"]).copy()
    d["wrong"] = d.minR2 < 0.95
    d["small"] = d.scen.str.split(":").str[1]
    ser = [("cGRM · none (standard PCA)", "standard"),
           ("uncGRM · topR_r_edge · KING+fam · phiR_r_edge · whiteC (dwg-like)", "topr/unc/kf/0/v/whitec"),
           ("raw · topR_r_edge · KING · phiR_D_r_edge · white (detect-white)", "topr/raw/k/0/D/white"),
           ("uncGRM · topR_r_edge · KING · phiR_r_edge · whiteC", "topr/unc/k/0/v/whitec"),
           ("cGRM · PCP · KING · phiS · L (aarobust-kin --impute-diag)", "pcp/cgrm/k/0/v/L")]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    lines(ax[0], d[d.small == "YRI2"], ser, "minR2", "mean min R²", "only 2 YRI (others n)", ylim=(0.7, 1.0))
    lines(ax[1], d[d.small == "MXL2"], ser, "minR2", "mean min R²", "only 2 MXL (others n)", ylim=(0.5, 1.0))
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=1, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.3, 1, 1))
    fig.savefig(os.path.join(out, "S1s_vs_n.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "S1s_vs_n.png"), dpi=130, bbox_inches="tight")


def s2(out):
    d = pd.concat([pd.read_csv(RES + "parts_S2.tsv", sep="\t"), pd.read_csv(RES + "parts_S2_kE.tsv", sep="\t")])
    if "panel" in d:
        d = d[(d.panel.isna()) | (d.panel == "admix")]
    d = d[d.k == 3].copy()
    d["wrong"] = d.minR2 < 0.95
    xanc = {"halfsibx", "grandx", "avuncx", "xanc", "childx"}
    ser = [("cGRM · none (standard PCA)", "standard"),
           ("raw · topR_r_edge · KING · phiR_D_r_edge · white (detect-white)", "topr/raw/k/0/D/white"),
           ("cGRM · PCP · KING · phiS · L (aarobust-kin --impute-diag)", "pcp/cgrm/k/0/v/L"),
           ("uncGRM · topR_r_edge · KING+fam+EA_r_edge(k0_mom) · phiR_r_edge · whiteC (dwg)", "topr/unc/kfe/1/v/whitec"),
           ("uncGRM · PCP · KING+EA_r_edge(k0_mom) · phiS · whiteC", "pcp/unc/kE/1/v/whitec")]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    lines(ax[0], d[~d.scen.str.contains(":")], ser, "wrong", "data sets wrong (%)", "relatives (all groups)", pct=True,
          ylim=(0, 10))
    lines(ax[1], d[d.scen.str.contains(":")], ser, "wrong", "data sets wrong (%)", "relatives + 2 YRI", pct=True,
          ylim=(0, 70))
    lines(ax[2], d[d.scen.isin(xanc) & (d.combo != "standard")], ser, "recall", "recall",
          "relatives of different ancestry")
    lines(ax[3], d[~d.scen.str.contains(":") & (d.combo != "standard")], ser, "err", "relatives' error",
          "placement of relatives")
    ax[2].set_ylim(0.5, 1.02)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.36, top=0.9, wspace=0.32)
    fig.savefig(os.path.join(out, "S2_vs_n.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "S2_vs_n.png"), dpi=130, bbox_inches="tight")


def s34(out):
    """standard / dwg / detect-white against N (simulated, 30% in families):
    N = 2,000 from the Python prototypes (mean of 2 reps), 5,000 / 20,000 /
    100,000 from PCAone; truth: reference panel up to 5,000, PCA of the
    unrelated individuals at 20,000 and 100,000"""
    import ast
    p = pd.read_csv(RES + "parts_S3_N2000.tsv", sep="\t")
    p = p[(p.src == "hn") & (p.rel == 0.3)]
    name = {"standard": "standard", "topr/unc/kfe/1/v/whitec": "dwg", "topr/raw/k/0/D/white": "detect-white"}
    rows = []
    for c, m in name.items():
        x = p[p.combo == c]
        rows.append(dict(N=2000, method=m, err=x[x.k == 4].err.mean(), fam=x[x.k == 20].fam_axes.mean(), sec=np.nan))
    c3 = [ast.literal_eval(ln) for ln in open(RES + "parts_S3_cpp.log") if ln.startswith("{")]
    c4 = pd.read_csv(RES + "parts_S4.tsv", sep="\t")
    for r in c3:
        if r["set"] == "hn:5000:0.3:0" and r["method"] in name.values():
            rows.append(dict(N=5000, method=r["method"], k=r["k"], err=r["err"], fam=r["fam_axes"], sec=r["sec"]))
    for _, r in c4[(c4.ooc == 0) & c4.bfile.isin(["hn_N20000_rel30_r0", "big_N100000"])].iterrows():
        rows.append(dict(N=r.N, method=r.method, k=r.k, err=r.err, fam=r.fam_axes, sec=r.sec))
    d = pd.DataFrame(rows)
    agg = lambda v, kk: d[(d.k.isna()) | (d.k == kk)].groupby(["method", "N"])[v].mean()  # noqa: E731
    err, fam, sec = agg("err", 4), agg("fam", 20), agg("sec", 4)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for i, m in enumerate(["standard", "dwg", "detect-white"]):
        for a, s, lab in [(ax[0], fam, "family axes among PCs 5-20"), (ax[1], err, "relatives' error"),
                          (ax[2], sec / sec["standard"], "time / standard PCA (in-core)")]:
            x = s[m].dropna()
            a.plot(x.index, x.values, color=COL[i], marker=MRK[i], lw=2, ms=6, label=m, clip_on=False)
            a.set_ylabel(lab)
    for a, t in zip(ax, ["family axes (k = 20)", "placement of relatives", "speed (rough)"]):
        a.set_xscale("log")
        a.set_xticks([2000, 5000, 20000, 100000])
        a.set_xticklabels(["2k", "5k", "20k", "100k"])
        a.set_xlabel("N (30% in families)")
        a.set_title(t, fontsize=9, color=INK)
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
    ax[2].set_yscale("log")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.27, top=0.9, wspace=0.32)
    fig.savefig(os.path.join(out, "S34_vs_N.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "S34_vs_N.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    out = sys.argv[1]
    s1(out)
    s1s(out)
    s2(out)
    s34(out)