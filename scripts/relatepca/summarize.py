#!/usr/bin/env python3
"""Tables and figures from benchmark.py results.

usage: summarize.py <results dir> <output dir>
reads main.tsv, large.tsv, wrongk.tsv, timing.tsv (whichever exist)
"""
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# plotted methods: baseline grey, oracle dashed, then the categorical palette
# in fixed order (reference palette, light mode)
STYLE = {
    "standard": ("#8a8984", "-", "standard PCA"),
    "white_gls_king": ("#2a78d6", "-", "whitening + GLS freq., KING"),
    "white_cs_king": ("#eb6834", "-", "CS whitening, KING"),
    "lrkin_cs_kc": ("#1baf7a", "-", "fixed rank + kinship, CS, KING screen"),
    "hyb_cs_kc": ("#eda100", "-", "detect (CS, screen) + CS whitening"),
    "aarobust": ("#e87ba4", "-", "AArobust (default $\\lambda$)"),
    "aarobust_kin": ("#008300", "-", "AArobust, kinship threshold + screen"),
    "white_gls_ped": ("#52514e", "--", "whitening + GLS freq., pedigree (oracle)"),
}
LAB = {"standard": "standard", "white_king": "whit., KING", "white_gls_king": "whit.+GLS, KING",
       "white_gls_ped": "whit.+GLS, ped.$^\\dagger$", "white_cs_king": "CS whit., KING",
       "lrkin_cs": "FR+kin CS", "lrkin_cs_kc": "FR+kin CS, screen", "lrkin_cs_autoK": "FR+kin CS, auto $K$",
       "hyb_gls": "detect+GLS whit.", "hyb_gls_kc": "detect+GLS whit., screen", "hyb_cs_kc": "detect+CS whit., screen",
       "auto_robust": "robust start+evalAdmix+CS whit.", "aarobust": "AArobust ($\\lambda$)", "aarobust_kin": "AArobust, kin.+screen", "pcpkin_cs": "PCP+kin CS"}
ALL = ["standard", "white_king", "white_gls_king", "white_gls_ped", "white_cs_king", "lrkin_cs", "lrkin_cs_t3",
       "lrkin_cs_autoK", "lrkin_cs_kc", "hyb_gls", "hyb_gls_t3", "hyb_gls_autoK", "hyb_gls_kc", "hyb_cs_kc",
       "aarobust", "aarobust_kin", "pcpkin_cs"]
KEY = ["standard", "white_king", "white_gls_king", "white_gls_ped", "white_cs_king", "lrkin_cs_kc",
       "lrkin_cs_autoK", "hyb_gls_kc", "hyb_cs_kc", "auto_robust", "aarobust", "aarobust_kin", "pcpkin_cs"]
SCEN_MAIN = ["none", "mz2", "nuclear2", "childx", "cousins", "many"]
SCEN_ORDER = ["none", "mz1", "mz2", "po", "nuclear2", "sibs2", "halfsib", "avunc", "grand", "cousins",
              "child_mxl", "childx", "combined", "many"]


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, sep="\t")
    extra = p.replace(".tsv", "_ar.tsv")  # separately run methods, merged in
    if os.path.exists(extra):
        d = pd.concat([d, pd.read_csv(extra, sep="\t")], ignore_index=True)
    if "method" not in d:
        return d
    skipped = d[d.method == "SKIP"]
    if len(skipped):
        print(f"{name}: {len(skipped)} skipped runs, e.g. {skipped.error.iloc[0]}")
    return d[d.method != "SKIP"].copy()


def small_multiples(d, value, xcol, fname, ylabel, ylim=None, logx=True, scen=SCEN_ORDER):
    scen = [s for s in scen if s in set(d.scen)]
    nc = 5
    nr = int(np.ceil(len(scen) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, scen):
        ds = d[d.scen == s]
        for m, (c, ls, lab) in STYLE.items():
            g = ds[ds.method == m].groupby(xcol)[value].mean()
            if len(g):
                ax.plot(g.index, g.values, ls, color=c, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(s, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        if logx:
            ax.set_xscale("log")
            xs = sorted(set(d[xcol]))
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" if x < 1000 else f"{x / 1000:g}k" for x in xs])
            ax.minorticks_off()
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(scen):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(xcol, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    h, l_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def table(d, value, index, columns, fname, fmt="{:.3f}", methods=KEY, caption_rows=None):
    t = d.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    cols = [m for m in methods if m in t.columns]
    t = t[cols]
    if index == "scen":
        t = t.loc[[s for s in SCEN_ORDER if s in t.index]]
    with open(os.path.join(OUT, fname), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n")
        f.write(" & " + " & ".join("\\rotatebox{75}{" + LAB.get(c, c.replace("_", " ")) + "}" for c in cols) + " \\\\\n\\midrule\n")
        for i, row in t.iterrows():
            vals = []
            best = None
            for c in cols:
                v = row[c]
                vals.append("" if pd.isna(v) else fmt.format(v))
            f.write(str(i).replace("_", "\\_") + " & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return t


main = load("main.tsv")
if main is not None:
    full = main[main.M > 20000]
    small_multiples(full, "minR2", "n", "main_minR2_vs_n.pdf", "min $R^2$ (mean of 10)", ylim=(0, 1.02),
                    scen=SCEN_MAIN)
    small_multiples(full, "err", "n", "main_err_vs_n.pdf", "error of relatives", scen=SCEN_MAIN[1:])
    small_multiples(full, "minR2", "n", "all_minR2_vs_n.pdf", "min $R^2$ (mean of 10)", ylim=(0, 1.02))
    small_multiples(full, "err", "n", "all_err_vs_n.pdf", "error of relatives")
    n10 = main[main.n == 10].copy()
    n10["Mk"] = n10.groupby(["M"]).M.transform(lambda x: x)  # actual M after filter
    n10["Mbin"] = np.select([n10.M < 3000, n10.M < 15000], [2000, 10000], 54000)
    small_multiples(n10, "minR2", "Mbin", "main_minR2_vs_M.pdf", "min $R^2$ (n = 10)", ylim=(0, 1.02),
                    scen=SCEN_MAIN)
    t1 = table(full[full.n == 10], "minR2", "scen", "method", "tab_minR2_n10.tex")
    t2 = table(full[full.n == 10], "err", "scen", "method", "tab_err_n10.tex", fmt="{:.2f}")
    t3 = table(full[full.n == 5], "minR2", "scen", "method", "tab_minR2_n5.tex")
    t4 = table(full[full.n == 5], "err", "scen", "method", "tab_err_n5.tex", fmt="{:.2f}")
    m2 = main[(main.n == 10) & (main.M < 3000)]
    table(m2, "minR2", "scen", "method", "tab_minR2_n10_M2k.tex")
    table(m2, "err", "scen", "method", "tab_err_n10_M2k.tex", fmt="{:.2f}")
    # overall: mean over scenarios with relatives, by n and method
    withrel = full[full.scen != "none"]
    ov = withrel.groupby(["n", "method"])[["minR2", "err"]].mean().unstack("method")
    print("\nmean over scenarios with relatives (M = all SNPs)\n", ov.round(3).to_string())
    # worst case over scenarios (mean over reps), per n
    worst = withrel.groupby(["n", "scen", "method"]).minR2.mean().groupby(["n", "method"]).min().unstack()
    print("\nworst scenario mean min R2\n", worst.round(3).to_string())
    # detection
    det = main[main.tp.notna()].copy()
    det["miss"] = det.n_true - det.tp
    dd = det.groupby(["Mbin" if "Mbin" in det else "M", "method"])[["tp", "fp", "miss"]].sum() if False else None
    det["Mbin"] = np.select([det.M < 3000, det.M < 15000], [2000, 10000], 54000)
    detsum = det.groupby(["Mbin", "n", "method"])[["n_true", "tp", "fp"]].sum().unstack("method")
    print("\ndetection (sums over runs)\n", detsum.to_string())
    detsum.to_csv(os.path.join(OUT, "detection.tsv"), sep="\t")
    # false positives in 'none' and 'cousins'
    fp = det[det.scen.isin(["none", "cousins"])].groupby(["Mbin", "scen", "method"]).fp.mean().unstack("method")
    print("\nmean false positives per run, none / cousins\n", fp.round(2).to_string())
    fp.to_csv(os.path.join(OUT, "false_positives.tsv"), sep="\t")
    # time
    tt = main.groupby(["n", "method"]).sec.median().unstack("method")
    print("\nmedian seconds\n", tt.round(3).to_string())
    # auto K
    ak = main[main.method == "lrkin_cs_autoK"].copy()
    ak["Mbin"] = np.select([ak.M < 3000, ak.M < 15000], [2000, 10000], 54000)
    print("\nautomatic K (lrkin_cs_autoK): share of runs per chosen K\n",
          ak.groupby(["Mbin", "n"]).K.value_counts(normalize=True).unstack().round(2).to_string())

large = load("large.tsv")
if large is not None:
    print("\nlarge (3 populations, k=2):")
    print(large.pivot_table(index=["n", "scen"], columns="method", values="minR2").round(3)[
        [m for m in ALL if m in set(large.method)]].to_string())
    print(large.pivot_table(index=["n", "scen"], columns="method", values="err").round(2)[
        [m for m in ALL if m in set(large.method)]].to_string())
    table(large[large.n == 85], "minR2", "scen", "method", "tab_large_minR2_n85.tex")
    table(large[large.n == 85], "err", "scen", "method", "tab_large_err_n85.tex", fmt="{:.2f}")

wk = load("wrongk.tsv")
if wk is not None:
    for kk in sorted(set(wk.k)):
        w = wk[wk.k == kk]
        print(f"\nwrong K: k = {kk} PCs given (true K-1 = 3); min R2 of the top {min(kk, 3)} true PCs")
        print(w.pivot_table(index="scen", columns="method", values="minR2").round(3)[
            [m for m in ALL if m in set(w.method)]].to_string())
        table(w, "minR2", "scen", "method", f"tab_wrongk{kk}_minR2.tex")
        table(w, "err", "scen", "method", f"tab_wrongk{kk}_err.tex", fmt="{:.2f}")

tm = load("timing.tsv")
if tm is not None:
    g = tm.groupby(["N", "method"]).sec.median().unstack("method")
    g = g[[m for m in ALL if m in g.columns]]
    g.to_csv(os.path.join(OUT, "timing_all.tsv"), sep="\t")
    g = g[[m for m in KEY if m in g.columns]]
    print("\ntiming: median seconds (single thread), M = all SNPs\n", g.round(3).to_string())
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, (c, ls, lab) in STYLE.items():
        if m in g:
            ax.plot(g.index, g[m], ls, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (individuals)")
    ax.set_ylabel("seconds (single thread)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "timing.pdf"))
    with open(os.path.join(OUT, "tab_timing.tex"), "w") as f:
        cols = list(g.columns)
        f.write("\\begin{tabular}{r" + "r" * len(cols) + "}\n\\toprule\n$N$ & " +
                " & ".join("\\rotatebox{75}{" + LAB.get(c, c.replace("_", " ")) + "}" for c in cols) + " \\\\\n\\midrule\n")
        for n_, row in g.iterrows():
            f.write(f"{n_} & " + " & ".join(f"{v:.2f}" for v in row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def overview(d, val, fname, fmt, agg="mean", scale=1.0, rows_filter=None):
    t = d.pivot_table(index=["Mb", "n"], columns="method", values=val, aggfunc=agg)
    t = t[[m for m in KEY if m in t.columns]] * scale
    with open(os.path.join(OUT, fname), "w") as f:
        f.write("\\begin{tabular}{rr" + "r" * len(t.columns) + "}\n\\toprule\n$M$ & $n$ & " +
                " & ".join("\\rotatebox{75}{" + LAB.get(c, c) + "}" for c in t.columns) + " \\\\\n\\midrule\n")
        prev = None
        for (mb, n_), row in t.iterrows():
            if prev is not None and mb != prev:
                f.write("\\midrule\n")
            f.write(("" if mb == prev else f"{mb // 1000}k") + f" & {n_} & " +
                    " & ".join(fmt.format(v) for v in row) + " \\\\\n")
            prev = mb
        f.write("\\bottomrule\n\\end{tabular}\n")


if main is not None:
    main["Mb"] = np.select([main.M < 3000, main.M < 15000], [2000, 10000], 54000)
    w = main[main.scen.isin(SCEN_MAIN[1:])].assign(fail=lambda x: x.minR2 < 0.95)
    overview(w, "fail", "tab_ov_fail.tex", "{:.0f}", scale=100)
    overview(w, "minR2", "tab_ov_minR2.tex", "{:.3f}")
    overview(w, "err", "tab_ov_err.tex", "{:.3f}")
    wa = main[main.scen != "none"].assign(fail=lambda x: x.minR2 < 0.95)
    overview(wa, "fail", "tab_ov_fail_all.tex", "{:.0f}", scale=100)
    overview(wa, "err", "tab_ov_err_all.tex", "{:.3f}")
    # per scenario, 54k SNPs, n = 10: appendix
    for nn in [5, 10]:
        sub_ = main[(main.M > 20000) & (main.n == nn)]
        table(sub_, "minR2", "scen", "method", f"tab_scen_minR2_n{nn}.tex")
        table(sub_, "err", "scen", "method", f"tab_scen_err_n{nn}.tex", fmt="{:.2f}")
    z = main[main.scen == "none"].assign(fail=lambda x: x.minR2 < 0.95)
    overview(z, "fail", "tab_ov_fail_none.tex", "{:.0f}", scale=100)
    wb = main[main.scen != "none"].assign(fail=lambda x: x.minR2_base < 0.95)
    overview(wb, "fail", "tab_ov_fail_basetruth.tex", "{:.0f}", scale=100)

aa = load("aa_lambda.tsv")
if aa is not None:
    aa["Mb"] = np.where(aa.M > 20000, "54k", "10k")
    aa["fail"] = aa.minR2 < 0.95
    aa["v"] = aa.val.round(4)
    t3 = round(2 ** -3.5, 4)
    sel = aa[((aa.kind == "mult") & aa.val.isin([1, 1.5])) | (aa.kind.isin(["kin", "kin_screen", "fr_grm", "fr_cs_screen"]) & (aa.v == t3))].copy()
    sel["col"] = np.where(sel.kind == "mult", "lam" + sel.val.astype(str), sel.kind)
    order = [("lam1.0", "PCP, $\\lambda=1/\\sqrt N$"), ("lam1.5", "PCP, $\\lambda=1.5/\\sqrt N$"),
             ("kin", "PCP, kinship thr."), ("kin_screen", "PCP, kinship thr. + screen"),
             ("fr_grm", "no norm, fixed rank, GRM"), ("fr_cs_screen", "no norm, fixed rank, CS + screen")]
    for scen_sel, fname in [("rel", "tab_aa_fail.tex"), ("none", "tab_aa_fail_none.tex")]:
        x = sel[sel.scen != "none"] if scen_sel == "rel" else sel[sel.scen == "none"]
        t = x.pivot_table(index=["Mb", "n"], columns="col", values="fail") * 100
        e = x.pivot_table(index=["Mb", "n"], columns="col", values="err")
        with open(os.path.join(OUT, fname), "w") as f:
            f.write("\\begin{tabular}{rr" + "r" * len(order) + "}\n\\toprule\n$M$ & $n$ & " +
                    " & ".join("\\rotatebox{75}{" + l + "}" for _, l in order) + " \\\\\n\\midrule\n")
            prev = None
            for (mb, n_), row in t.iterrows():
                if prev is not None and mb != prev:
                    f.write("\\midrule\n")
                cells = [f"{row[c]:.0f}" + ("" if scen_sel == "none" else f" ({e.loc[(mb, n_), c]:.3f})")
                         for c, _ in order]
                f.write(("" if mb == prev else mb) + f" & {n_} & " + " & ".join(cells) + " \\\\\n")
                prev = mb
            f.write("\\bottomrule\n\\end{tabular}\n")
