#!/usr/bin/env python3
"""LaTeX tables of the PCP-pi report (report/pcp_pi/) from the benchmark output
(pcp_pi_bench.py) and the reindeer runs (reindeer_pcp_pi.py). Cells: bias
(RMSE), or as stated in each table.

usage: pcp_pi_tables.py <main prefix> <giant prefix> <reindeer dir> <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd

from pcp_pi_summary import degree

NAMES = {0: "MZ", 1: "1st", 2: "2nd", 3: "3rd"}
FM = [("pca3", "PC $k{=}3$"), ("pca6", "PC 6"), ("pca10", "PC 10"), ("pcp-none", "PCP"),
      ("pcp-self", "PCP self"), ("dwg-none", "dwg"), ("dwg-self", "dwg self")]
RM = [("KING", "KING"), ("pca3", "PC 3 fam."), ("pca3-own", "PC 3 own"), ("pca10", "PC 10 fam."),
      ("PCAone-dwg", "\\texttt{dwg}"), ("pcp-self", "PCP self"), ("pcp-family", "PCP fam."),
      ("pcp-own", "PCP own"), ("dwg-own", "dwg own")]


def cell(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if not len(e):
        return "--"
    b = f"{e.mean():+.3f}".replace("-", "$-$")
    return f"{b} ({np.sqrt((e ** 2).mean()):.3f})"


def table(rows, head, cols, fmt="ll"):
    out = ["\\begin{tabular}{@{}" + fmt + "r" * len(cols) + "@{}}", "\\toprule",
           " & ".join(head + [c for _, c in cols]) + " \\\\", "\\midrule"]
    last = None
    for r in rows:
        if last is not None and r[0] != last:
            out.append("\\addlinespace")
        last = r[0]
        out.append(" & ".join(str(x) for x in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def write(outdir, name, s):
    with open(os.path.join(outdir, name + ".tex"), "w") as f:
        f.write(s + "\n")


def f_table(F, scen, groups, cols=FM):
    rows = []
    for n in sorted(F.n.unique()):
        for g in groups:
            d = F[(F.scen == scen) & (F.n == n) & (F.group == g)]
            if len(d):
                rows.append([n, g] + [cell(d[m] - d.F_true) for m, _ in cols])
    return table(rows, ["$n$", "group"], cols)


def r_table(R, scen, cols=RM):
    R = R[(R.scen == scen) & (R.kind == "true")].copy()
    R["deg"] = degree(R.kin_true)
    rows = []
    for n in sorted(R.n.unique()):
        for k in (0, 1, 2, 3):
            d = R[(R.n == n) & (R.deg == k)]
            if len(d):
                rows.append([n, f"{NAMES[k]} ({len(d)})"] + [cell(d[m] - d.kin_true) for m, _ in cols])
    return table(rows, ["$n$", "degree (pairs)"], cols)


def unrel_table(R, scens, cols=RM):
    rows = []
    for s in scens:
        for n in sorted(R.n.unique()):
            d = R[(R.scen == s) & (R.n == n) & (R.kind == "unrel")]
            if len(d):
                rows.append([s, n] + [f"{d[m].mean():.3f} / {100 * (d[m] > 2 ** -4.5).mean():.0f}\\%"
                                      if d[m].notna().mean() > 0.9 else "--" for m, _ in cols])
    return table(rows, ["design", "$n$"], cols)


def degree_table(R, scens, cols=RM):
    rows = []
    for s in scens:
        for n in sorted(R.n.unique()):
            d = R[(R.scen == s) & (R.n == n)]
            if not len(d):
                continue
            t = d[(d.kind == "true") & (degree(d.kin_true) < 4)]
            u = d[d.kind == "unrel"]
            r = [s, n]
            for m, _ in cols:
                if t[m].notna().all():
                    r.append(f"{100 * (degree(t[m]) == degree(t.kin_true)).mean():.0f} / "
                             f"{100 * (degree(u[m]) == 4).mean():.0f}")
                else:
                    r.append("--")
            rows.append(r)
    return table(rows, ["design", "$n$"], cols)


def ibd_table(R, scen, cols):
    R = R[(R.scen == scen) & (R.kind == "true")]
    R = R[degree(R.kin_true) < 4]
    rows = []
    for n in sorted(R.n.unique()):
        d = R[R.n == n]
        rows.append([n, len(d)] + [f"{np.sqrt(((d[m + '_k0'] - d.k0_true) ** 2).mean()):.3f} / "
                                   f"{np.sqrt(((d[m + '_k2'] - d.k2_true) ** 2).mean()):.3f}" for m, _ in cols])
    return table(rows, ["$n$", "pairs"], cols)


def main():
    main_p, giant_p, rdir, outdir = sys.argv[1:5]
    os.makedirs(outdir, exist_ok=True)
    F = pd.read_csv(main_p + "_F.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    R = pd.read_csv(main_p + "_R.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    P = pd.read_csv(main_p + "_P.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"]) if os.path.exists(main_p + "_P.tsv") else None
    write(outdir, "f_rel", f_table(F, "rel", ["founder", "relative", "inbred"]))
    write(outdir, "f_small", f_table(F, "small", ["founder", "relative", "inbred"]))
    write(outdir, "f_het", f_table(F, "het", ["excess het"]))
    write(outdir, "f_null", f_table(F, "null", ["founder"]))
    write(outdir, "r_rel", r_table(R, "rel"))
    write(outdir, "r_small", r_table(R, "small"))
    write(outdir, "r_unrel", unrel_table(R, ["null", "rel", "small"]))
    write(outdir, "r_degree", degree_table(R, ["rel", "small"]))
    ibd_cols = [("pca3", "PC 3 fam."), ("pca3-own", "PC 3 own"), ("pca10", "PC 10 fam."), ("pcp-self", "PCP self"),
                ("pcp-family", "PCP fam."), ("pcp-own", "PCP own"), ("dwg-own", "dwg own")]
    write(outdir, "r_ibd", ibd_table(R, "rel", ibd_cols))
    # PCs
    rows = []
    for (s, n), d in (P.groupby(["scen", "n"]) if P is not None else []):
        g = d.groupby("method").minR2.mean()
        rows.append([s, n] + [f"{g[m]:.2f}" for m in ("std", "aad", "dwg")])
    write(outdir, "p_main", table(rows, ["design", "$n$"], [(0, "standard"), (0, "\\texttt{aarobust-kin+diag}"),
                                                             (0, "\\texttt{dwg} ($k{=}3$)")]))
    # giant families
    G = pd.read_csv(giant_p + "_F.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    GR = pd.read_csv(giant_p + "_R.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    GP = pd.read_csv(giant_p + "_P.tsv", sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
    rows = []
    for (s, n), d in GP.groupby(["scen", "n"], sort=False):
        g = d.groupby("method")
        rows.append([s, n] + [f"{g.minR2.mean()[m]:.2f}" for m in ("std", "aad", "dwg")] +
                    [f"{g.err_family.mean()[m]:.1f}" for m in ("std", "aad", "dwg")])
    write(outdir, "g_pcs", table(rows, ["design", "$n$"], [(0, "std"), (0, "aad"), (0, "dwg"), (0, "std"),
                                                           (0, "aad"), (0, "dwg")]))
    gcols = [("pca3", "PC $k{=}3$"), ("pca10", "PC 10"), ("pcp-none", "PCP"), ("pcp-self", "PCP self"),
             ("dwg-self", "dwg self")]
    rows = []
    for (s, n), d in G[G.group == "family"].groupby(["scen", "n"], sort=False):
        rows.append([s, n] + [cell(d[m] - d.F_true) for m, _ in gcols])
    write(outdir, "g_f", table(rows, ["design", "$n$"], gcols))
    grcols = [("KING", "KING"), ("pca3", "PC 3 fam."), ("pca3-own", "PC 3 own"), ("PCAone-dwg", "\\texttt{dwg}"),
              ("pcp-self", "PCP self"), ("pcp-own", "PCP own"), ("pcpT-own", "PCP own, true"), ("dwg-own", "dwg own")]
    rows = []
    for (s, n), d in GR[GR.kind == "true"].groupby(["scen", "n"], sort=False):
        t = d[degree(d.kin_true) < 4]
        rows.append([s, n] + [cell(t[m] - t.kin_true) for m, _ in grcols])
    write(outdir, "g_r", table(rows, ["design", "$n$"], grcols))
    write(outdir, "g_degree", degree_table(GR, list(GR.scen.unique()), grcols))
    # reindeer
    for tag in ("new_ld", "old_full", "old_ld"):
        f = os.path.join(rdir, f"reindeer_{tag}_R.tsv")
        if not os.path.exists(f):
            continue
        X = pd.read_csv(f, sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
        cols = ["KING", "R0", "pca4", "pca10", "pcp-self", "pcp-own", "dwg-self", "dwg-own"]
        X = X.sort_values("KING", ascending=False)
        rows = []
        for _, r in X.iterrows():
            site = str(r.site).replace("_", " ").replace("Kangerlussuaq Sisimiut", "Kanger.-Sis.").replace(
                "Kangerlussuaq-Sisimiut", "Kanger.-Sis.").replace("Qeqertarsuatsiaat", "Qeqert.")
            rows.append([r.pair.replace("--", "--"), site] +
                        [f"{r[c]:.3f}".replace("-", "$-$") for c in cols])
        write(outdir, f"rd_{tag}", table(rows, ["pair", "site"], [(0, c.replace("pca", "PC ").replace(
            "pcp-", "PCP ").replace("dwg-", "dwg ").replace("R0", "$R_0$")) for c in cols]))
        Fx = pd.read_csv(os.path.join(rdir, f"reindeer_{tag}_F.tsv"), sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])
        fcols = ["pca4", "pca6", "pca10", "pcp-self", "dwg-self"]
        g = Fx.groupby("FID")[fcols].mean()
        nn = Fx.groupby("FID").size()
        rows = [[s.replace("_", " "), nn[s]] + [f"{g.loc[s, c]:+.3f}".replace("-", "$-$") for c in fcols]
                for s in g.index]
        write(outdir, f"rdF_{tag}", table(rows, ["site", "$n$"], [(0, c.replace("pca", "PC ").replace(
            "pcp-", "PCP ").replace("dwg-", "dwg ")) for c in fcols]))


if __name__ == "__main__":
    main()
