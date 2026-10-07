#!/usr/bin/env python3
"""Slide figures: dwg step by step (three layouts, A/B/C). The data are the
methods-figure data set (10 per population + the "many" families) plus one
cross-ancestry family (avuncx: CHB aunt -- CHB/YRI nephew), which KING misses
and the evalAdmix round finds. The fit is dwg_loc_ea, re-run here with every
step recorded (candidates by source, family axes, S, L, Sigma, PCs).

usage: slides_dwg_flow.py <bfile> <outdir> [explore]
"""
import os
import sys
import zlib

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
from methods_figs import INK, MUTED, PCOL, POPS  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
SPEC = B.SCENARIOS["many"] + ",avuncx:CHB:YRI"
KCOL, ECOL, FCOL = "#D55E00", "#7B3294", "#009E73"  # candidates from KING, evalAdmix, family axes


def dataset(bfile, n=10, rep=0):
    fam, bim, Gall = F.read_bed(bfile)
    lab = np.array([r[1] for r in fam])
    seed = zlib.crc32(f"{','.join(POPS)}|{n}|0|{SPEC}|{rep}".encode())
    return B.make_dataset(Gall, lab, bim, POPS, n, 0, SPEC, seed)


def fit_loc_rec(A, tau, cand, edge, rmax, c, rec, iters=500):
    """DL.fit_loc (kin_check=True) with the family axes recorded"""
    n = len(A)
    eye = np.eye(n, dtype=bool)
    cand = cand.copy()
    S = np.zeros_like(A)
    r = 1
    while True:
        L_old = None
        for _ in range(iters):
            w_, V = I.top_eig(A - S, r)
            L = (V * w_) @ V.T
            R = A - L
            v = np.maximum(np.diag(R), 1e-6)
            sel = I.kin_select(R, v, tau) & cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        if r >= min(rmax, n - 1):
            break
        ev, V = I.top_eig(A - S, r + 1)
        if ev[r] <= edge:
            break
        fam, top = DL.family_axis(V[:, r], A - L, np.maximum(np.diag(A - L), 1e-6), c, True)
        if fam:
            new = [(i, j) for a, i in enumerate(top) for j in top[a + 1:] if not cand[i, j]]
            rec["family"].append(dict(r=r, u=V[:, r].copy(), top=list(top), new=new,
                                      scree=I.top_eig(A - S, 12)[0], edge=edge))
            if not new:
                break
            for i, j in new:
                cand[i, j] = cand[j, i] = True
            continue
        rec["axes_ok"].append(V[:, r].copy())
        r += 1
    v = np.maximum(np.diag(A - L), 1e-6)
    phi = I.kin_scale(A - L, v)
    iu = np.triu_indices(n, 1)
    pairs = [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]
    return L, S, r, pairs, cand


def dwg_rec(G, k=3, c=4, rounds=5, k0_max=0.8):
    """DL.dwg_loc_ea(G, k, KING > 0.04, c, kin_check=True, k0_max) recorded"""
    N = len(G)
    off = ~np.eye(N, dtype=bool)
    king = B.king_robust(G)
    ck = (king > B.KING_SCREEN) & off
    A, f, w, _ = W.grm_scaled_uncentred(G)
    edge = W.noise_edge(f, w, N)

    def one(cand):
        rec = dict(family=[], axes_ok=[])
        L, S, r, pairs, cand2 = fit_loc_rec(A, B.TAU, cand, edge, k + 1, c, rec)
        ev, V = I.top_eig(L, r)
        keep = []
        for j in range(1, r):
            Rj = A - L + ev[j] * np.outer(V[:, j], V[:, j])
            fam, _ = DL.family_axis(V[:, j], Rj, np.maximum(np.diag(Rj), 1e-6), c, True)
            if not fam:
                keep.append(j)
        axes = V[:, keep] if keep else V[:, 1:2]
        return dict(L=L, S=S, r=r, pairs=pairs, cand=cand2, axes=axes, rec=rec)

    out = one(ck)
    first = out
    key = {(i, j) for i, j, _ in out["pairs"]}
    ce = np.zeros_like(ck)
    for _ in range(rounds):
        ea = I.evaladmix_kin(G, out["axes"], intercept=True)
        c0 = (ea > B.KING_SCREEN) & off & ~ck
        cand_e = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(c0, 1)))]
        k0 = DL.k0_moment(G, out["axes"], cand_e)
        ce = np.zeros_like(ck)
        for (i, j), v_ in k0.items():
            if v_ < k0_max:
                ce[i, j] = ce[j, i] = True
        out = one(ck | ce)
        new = {(i, j) for i, j, _ in out["pairs"]}
        if new == key:
            break
        key = new
    v = np.maximum(np.diag(A) - np.diag(out["L"]), 1e-6)
    # Sigma, whitened + centred matrix and PCs, as I.whitened_noise(..., centre=True)
    d = np.sqrt(v)
    Rm = np.eye(N)
    for i, j, phi in out["pairs"]:
        Rm[i, j] = Rm[j, i] = 2 * phi
    Sig = d[:, None] * Rm * d[None, :]
    ws, vs = np.linalg.eigh(Sig)
    ws = np.maximum(ws, 0.1 * d.min() ** 2)
    Wm, Wh = (vs / np.sqrt(ws)) @ vs.T, (vs * np.sqrt(ws)) @ vs.T
    u = Wm @ np.ones(N)
    u /= np.linalg.norm(u)
    P = np.eye(N) - np.outer(u, u)
    Mw = P @ Wm @ A @ Wm @ P
    _, U = I.top_eig(Mw, k)
    fam_cand = np.zeros_like(ck)
    for fr in first["rec"]["family"] + out["rec"]["family"]:
        for i, j in fr["new"]:
            fam_cand[i, j] = fam_cand[j, i] = True
    return dict(A=A, king=king, ck=ck, ce=ce, cf=fam_cand & ~ck & ~ce, first=first, final=out, Sig=Sig, Mw=Mw,
                U=Wh @ U, v=v)


def explore(bfile, k=3):
    for rep in range(4):
        ds = dataset(bfile, rep=rep)
        G = ds["G"].astype(float)
        pop = F.family_pop(ds)
        res = dwg_rec(G, k=k)
        K = ds["Kped"]
        fin = res["final"]
        pairs = {(i, j) for i, j, _ in fin["pairs"]}
        true = {(i, j) for i, j in zip(*np.where(np.triu(K, 1) > B.TAU))}
        print("rep", rep, "N", len(G), "r", fin["r"], "KING cand", int(res["ck"].sum() // 2), "EA cand",
              int(res["ce"].sum() // 2), "fam cand", int(res["cf"].sum() // 2),
              "family axes", [(f_["r"], len(f_["top"]), len(f_["new"])) for f_ in
                              res["first"]["rec"]["family"] + fin["rec"]["family"]],
              "found", len(pairs & true), "/", len(true), "false", len(pairs - true))
        for i, j in zip(*np.where(np.triu(res["ce"], 1))):
            print("   EA cand", pop[i], pop[j], "Kped", round(K[i, j], 3), "king", round(res["king"][i, j], 3),
                  "in S", (i, j) in pairs)


# ------------------------------------------------------------------ drawing


def prep(bfile, k=6, rep=0):
    ds = dataset(bfile, rep=rep)
    G = ds["G"].astype(float)
    pop = F.family_pop(ds)
    N = len(G)
    rel = np.arange(N) >= ds["n0"]
    res = dwg_rec(G, k=k)
    o = F.order_rows_spread(pop, ds["Kped"])
    P = lambda M: M[np.ix_(o, o)]  # noqa: E731
    fin = res["final"]
    fr = (res["first"]["rec"]["family"] + fin["rec"]["family"])[0]
    ok = res["first"]["rec"]["axes_ok"][-1]
    ea = I.evaladmix_kin(G, res["first"]["axes"], intercept=True)
    d = dict(scree=fr["scree"], edge=fr["edge"], r_fam=fr["r"], king=P(res["king"]), ck=P(res["ck"]), ce=P(res["ce"]), cf=P(res["cf"]), S=P(fin["S"]), L=P(fin["L"]),
             Mw=P(res["Mw"]), U=res["U"][o], pop=pop[o], rel=rel[o], u_fam=fr["u"][o], u_ok=ok[o], ea=P(ea),
             N=N, k=k)
    d["cand"] = d["ck"] | d["ce"] | d["cf"]
    return d


def boxes(ax, mask, col, pad=0.0, lw=1.3):
    for i, j in zip(*np.where(mask)):
        ax.add_patch(plt.Rectangle((j - 0.5 - pad, i - 0.5 - pad), 1 + 2 * pad, 1 + 2 * pad, fill=False, ec=col,
                                   lw=lw))


def kin_panel(ax, K, title, d, which=("k", "f", "e")):
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("white")
    M = K.copy()
    np.fill_diagonal(M, np.nan)
    ax.imshow(M, cmap=cm, vmin=-0.25, vmax=0.25, interpolation="nearest")
    if "k" in which:
        boxes(ax, d["ck"], KCOL)
    if "f" in which:
        boxes(ax, d["cf"] | (d["cand"] & ~d["ck"] & ("e" not in which)), FCOL, pad=0.0)
    if "e" in which:
        boxes(ax, d["ce"], ECOL, pad=0.6, lw=1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK)


def s_panel(ax, d, title="$S$: diagonal + candidates\nwith $\\phi_{\\mathrm{res}} > \\tau$"):
    F.heat(ax, d["S"], title, sparse=True, diag_dots=True)
    off = ~np.eye(d["N"], dtype=bool)
    out = d["cand"] & (d["S"] == 0) & off
    boxes(ax, out, MUTED, pad=0.4, lw=1.0)


def l_panel(ax, d, title="$L$: rank-$r$ fit of $A - S$\n(diagonal, pairs imputed)"):
    F.heat(ax, d["L"], title, positive=True)


def axis_panel(ax, d):
    n = d["N"]
    x = np.arange(n)
    for u, y0, col, lab in [(d["u_ok"], 1.2, MUTED, "structure axis"), (d["u_fam"], 0.0, FCOL, "family axis")]:
        u = u / np.abs(u).max() * 0.38
        ax.vlines(x, y0, y0 + u, color=col, lw=1.4)
        ax.axhline(y0, color="#cccccc", lw=0.6)
        ne = 1 / np.sum((u / np.linalg.norm(u)) ** 4)
        ax.text(n * 0.02, y0 + 0.45, f"{lab}: $n_{{\\mathrm{{eff}}}}$ = {ne:.1f}", fontsize=8, color=INK)
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.5, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.set_title("next axis of $A - S$:\nfamily if $n_{\\mathrm{eff}} < 4$", color=INK)


def wc_panel(ax, d):
    F.heat(ax, d["Mw"], "whitened + centered\n$P\\,\\Sigma^{-1/2}A\\Sigma^{-1/2}P$, $\\Sigma = S$")


def pc_panel(ax, d, a=0, b=1):
    F.pcs(ax, d["U"], d["pop"], d["rel"], "PCs ($\\times\\Sigma^{1/2}$)", a, b)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)


def mid(ax, side):
    b = ax.get_position()
    return {"l": (b.x0, (b.y0 + b.y1) / 2), "r": (b.x1, (b.y0 + b.y1) / 2), "t": ((b.x0 + b.x1) / 2, b.y1),
            "b": ((b.x0 + b.x1) / 2, b.y0)}[side]


def arr(fig, p0, p1, text="", rad=0.0, dx=0.0, dy=0.03, col=MUTED, fs=8, ha="center", va="bottom"):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                                   color=col, lw=1.0, connectionstyle=f"arc3,rad={rad}"))
    if text:
        fig.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, text, ha=ha, va=va, fontsize=fs, color=INK,
                 transform=fig.transFigure, linespacing=1.1)


def loop(fig, a1, a2, top="impute $L$", bot="$\\phi_{\\mathrm{res}} > \\tau$"):
    b1, b2 = a1.get_position(), a2.get_position()
    x0, x1 = b1.x1 + 0.006, b2.x0 - 0.006
    ym = (b1.y0 + b1.y1) / 2
    h = (b1.y1 - b1.y0) * 0.17
    arr(fig, (x0, ym + h), (x1, ym + h), rad=-0.5)
    arr(fig, (x1, ym - h), (x0, ym - h), rad=-0.5)
    fig.text((x0 + x1) / 2, ym + h * 0.55, top, ha="center", va="center", fontsize=8, color=INK)
    fig.text((x0 + x1) / 2, ym, "iterate", ha="center", va="center", fontsize=7.5, color=MUTED)
    fig.text((x0 + x1) / 2, ym - h * 0.55, bot, ha="center", va="center", fontsize=8, color=INK)


def key_row(fig, y=0.0, ea=True, fam=True):
    hs = [plt.Line2D([], [], marker="s", ls="", mfc="none", mec=KCOL, markersize=8, mew=1.4)]
    lab = ["KING candidate"]
    if fam:
        hs.append(plt.Line2D([], [], marker="s", ls="", mfc="none", mec=FCOL, markersize=8, mew=1.4))
        lab.append("family-axis candidate")
    if ea:
        hs.append(plt.Line2D([], [], marker="s", ls="", mfc="none", mec=ECOL, markersize=8, mew=1.4))
        lab.append("evalAdmix candidate")
    hs.append(plt.Line2D([], [], marker="s", ls="", mfc="none", mec=MUTED, markersize=8, mew=1.0))
    lab.append("candidate not in $S$")
    hs += [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], markeredgecolor="white", markersize=6) for p in POPS]
    hs.append(plt.Line2D([], [], marker="*", ls="", color="#bbbbbb", markeredgecolor=INK, markersize=9))
    lab += POPS + ["relative"]
    fig.legend(hs, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, y), fontsize=7.5,
               handletextpad=0.3, columnspacing=0.9)


def fig_a(d, out):
    """two rows: detect (KING -> S <-> L -> axis check, evalAdmix loop), then whiten + center -> PCs"""
    fig = plt.figure(figsize=(10, 5.6))
    w, h = 0.16, 0.16 * 10 / 5.6
    y1, y2 = 0.56, 0.11
    aK = fig.add_axes([0.03, y1, w, h])
    aS = fig.add_axes([0.30, y1, w, h])
    aL = fig.add_axes([0.55, y1, w, h])
    aX = fig.add_axes([0.80, y1, 0.18, h])
    aW = fig.add_axes([0.03, y2, w, h])
    aP = fig.add_axes([0.30, y2, w * 1.05, h])
    aE = fig.add_axes([0.55, y2, w, h])
    kin_panel(aK, d["king"], "$\\phi_{\\mathrm{KING}} > 0.04$:\ncandidates", d)
    s_panel(aS, d)
    l_panel(aL, d)
    axis_panel(aX, d)
    kin_panel(aE, d["ea"], "$\\phi_{\\mathrm{EA}}$ from the structure\naxes of $L$", d, which=("e",))
    wc_panel(aW, d)
    pc_panel(aP, d)
    for a in (aK, aS, aL, aX, aE, aW, aP):
        a.title.set_fontsize(9)
    arr(fig, mid(aK, "r"), mid(aS, "l"), "candidates")
    loop(fig, aS, aL)
    arr(fig, mid(aL, "r"), mid(aX, "l"), "")
    xt = aX.get_position()
    st = aS.get_position()
    arr(fig, (xt.x0 + 0.03, xt.y1 + 0.075), (st.x1 - 0.02, st.y1 + 0.075), rad=0.12,
        text="family axis: pairs of its top people become candidates; refit (else $r + 1$)", dy=0.035,
        col=FCOL)
    lb, eb, sb = aL.get_position(), aE.get_position(), aS.get_position()
    xl = (lb.x0 + lb.x1) / 2 + 0.03
    arr(fig, (xl, lb.y0), (xl, eb.y1 + 0.075), "evalAdmix\nround", dx=0.008, dy=-0.03, ha="left", va="center")
    arr(fig, (eb.x0 + 0.01, eb.y1 + 0.02), (sb.x1 - 0.02, sb.y0),
        "$\\phi_{\\mathrm{EA}} > 0.04$, $k_{0} < 0.8$:\nnew candidates", dx=-0.005, dy=-0.005, ha="right",
        va="center", col=ECOL)
    wb = aW.get_position()
    arr(fig, (sb.x0 + 0.02, sb.y0), ((wb.x0 + wb.x1) / 2 + 0.02, wb.y1 + 0.075), "$\\Sigma = S$", dx=0.01,
        ha="left", va="center", dy=0.0)
    pb = aP.get_position()
    arr(fig, mid(aW, "r"), (pb.x0 - 0.03, (pb.y0 + pb.y1) / 2), "eigen-\nvectors")
    key_row(fig, y=-0.04)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_b(d, out):
    """one row: candidates (all sources) -> S <-> L -> axis check -> PCs, loops back as arrows"""
    fig = plt.figure(figsize=(10, 4.2))
    w = 0.145
    h = w * 10 / 4.2
    y = 0.22
    aK = fig.add_axes([0.02, y, w, h])
    aS = fig.add_axes([0.235, y, w, h])
    aL = fig.add_axes([0.45, y, w, h])
    aX = fig.add_axes([0.645, y, 0.155, h])
    aP = fig.add_axes([0.855, y, w * 0.98, h])
    kin_panel(aK, d["king"], "$\\phi_{\\mathrm{KING}}$ with\nall candidates", d)
    s_panel(aS, d)
    l_panel(aL, d)
    axis_panel(aX, d)
    pc_panel(aP, d)
    aP.set_title("PCs: whiten ($\\Sigma = S$),\ncenter, $\\times\\Sigma^{1/2}$", color=INK)
    for a in (aK, aS, aL, aX, aP):
        a.title.set_fontsize(9)
    arr(fig, mid(aK, "r"), mid(aS, "l"), "candi-\ndates")
    loop(fig, aS, aL)
    arr(fig, mid(aL, "r"), mid(aX, "l"))
    pb = aP.get_position()
    arr(fig, mid(aX, "r"), (pb.x0 - 0.025, (pb.y0 + pb.y1) / 2))
    kt, xt = aK.get_position(), aX.get_position()
    arr(fig, ((xt.x0 + xt.x1) / 2, xt.y1 + 0.13), ((kt.x0 + kt.x1) / 2, kt.y1 + 0.13), rad=0.1, col=FCOL,
        text="family axis ($n_{\\mathrm{eff}} < 4$): its pairs become candidates", dy=0.035)
    lt = aL.get_position()
    arr(fig, ((lt.x0 + lt.x1) / 2, lt.y0 - 0.01), ((kt.x0 + kt.x1) / 2, kt.y0 - 0.01), rad=-0.12, col=ECOL,
        text="evalAdmix round: $\\phi_{\\mathrm{EA}} > 0.04$ from the structure axes, $k_0 < 0.8$",
        dy=-0.085, va="top")
    key_row(fig, y=-0.1)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_c(d, out1, out2):
    """two steps: (1) KING candidates -> S <-> L -> family axis; (2) evalAdmix candidates -> S -> whiten +
    center -> PCs"""
    fig = plt.figure(figsize=(10, 4.0))
    w = 0.2
    h = w * 10 / 4.0
    y = 0.2
    aK = fig.add_axes([0.02, y, w, h])
    aS = fig.add_axes([0.28, y, w, h])
    aL = fig.add_axes([0.54, y, w, h])
    aX = fig.add_axes([0.78, y, 0.2, h])
    kin_panel(aK, d["king"], "$\\phi_{\\mathrm{KING}}$: candidates", d, which=("k", "f"))
    s_panel(aS, d)
    l_panel(aL, d)
    axis_panel(aX, d)
    arr(fig, mid(aK, "r"), mid(aS, "l"), "candi-\ndates")
    loop(fig, aS, aL)
    arr(fig, mid(aL, "r"), mid(aX, "l"))
    kt, xt = aK.get_position(), aX.get_position()
    arr(fig, ((xt.x0 + xt.x1) / 2, xt.y1 + 0.13), ((kt.x0 + kt.x1) / 2, kt.y1 + 0.13), rad=0.08, col=FCOL,
        text="family axis ($n_{\\mathrm{eff}} < 4$): pairs of its top people become candidates; refit (else $r + 1$)",
        dy=0.03)
    key_row(fig, y=-0.08, ea=False)
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    fig = plt.figure(figsize=(10, 4.0))
    aE = fig.add_axes([0.02, y, w, h])
    aS = fig.add_axes([0.28, y, w, h])
    aW = fig.add_axes([0.54, y, w, h])
    aP = fig.add_axes([0.79, y, w, h])
    kin_panel(aE, d["ea"], "$\\phi_{\\mathrm{EA}}$ from the\nstructure axes of $L$", d, which=("e",))
    s_panel(aS, d, "$S$ refitted with\nKING + family + evalAdmix")
    wc_panel(aW, d)
    pc_panel(aP, d)
    arr(fig, mid(aE, "r"), mid(aS, "l"), "$\\phi_{\\mathrm{EA}} > 0.04$\n$k_0 < 0.8$", col=ECOL)
    arr(fig, mid(aS, "r"), mid(aW, "l"), "$\\Sigma = S$")
    pb = aP.get_position()
    arr(fig, mid(aW, "r"), (pb.x0 - 0.03, (pb.y0 + pb.y1) / 2), "eigen-\nvectors")
    key_row(fig, y=-0.08)
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)


def fig_rank(d, out):
    """the noise-family rule: eigenvalues of A - S (at the step where the family axis appears) against the
    noise edge, coloured ancestry / family / noise, and the loadings of an ancestry and the family axis"""
    fig = plt.figure(figsize=(10, 3.0))
    a1 = fig.add_axes([0.06, 0.16, 0.36, 0.72])
    a2 = fig.add_axes([0.52, 0.16, 0.46, 0.72])
    ev, edge, rf = d["scree"], d["edge"], d["r_fam"]
    x = np.arange(1, len(ev) + 1)
    col = [MUTED if i == 0 else ("#0072B2" if i < rf else (FCOL if i == rf else "#BBBBBB")) for i in range(len(ev))]
    a1.bar(x, ev, color=col, width=0.7)
    a1.axhline(edge, color="#D55E00", ls="--", lw=1.2)
    a1.text(len(ev) + 0.4, edge * 1.12, "noise edge $2\\sigma\\sqrt{N}$", ha="right", va="bottom", fontsize=9,
            color="#D55E00")
    a1.set_yscale("log")
    a1.set_xticks(x)
    a1.set_xlabel("axis", color=MUTED)
    a1.set_ylabel("eigenvalue of $A - S$", color=MUTED)
    for sp in ["top", "right"]:
        a1.spines[sp].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in (MUTED, "#0072B2", FCOL, "#BBBBBB")]
    a1.legend(hs, ["mean", "ancestry", "family", "noise"], frameon=False, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.0, 0.92))
    axis_panel(a2, d)
    a2.set_title("")
    a2.texts[0].set_text(a2.texts[0].get_text().replace("structure axis", f"ancestry axis (axis {rf})"))
    a2.texts[1].set_text(a2.texts[1].get_text().replace("family axis", f"family axis (axis {rf + 1})"))
    a2.set_xlabel("individuals", color=MUTED)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    bfile, outdir = sys.argv[1], sys.argv[2]
    d = prep(bfile)
    fig_rank(d, os.path.join(outdir, "dwg_rank.pdf"))
    if len(sys.argv) > 3 and sys.argv[3] == "rank":
        return
    fig_a(d, os.path.join(outdir, "dwg_flow_a.pdf"))
    fig_b(d, os.path.join(outdir, "dwg_flow_b.pdf"))
    fig_c(d, os.path.join(outdir, "dwg_flow_c1.pdf"), os.path.join(outdir, "dwg_flow_c2.pdf"))
    print("candidates KING", int(d["ck"].sum() // 2), "family", int(d["cf"].sum() // 2), "evalAdmix",
          int(d["ce"].sum() // 2), "pairs in S", int(((d["S"] != 0) & ~np.eye(d["N"], dtype=bool)).sum() // 2))


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "explore":
        explore(sys.argv[1], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    else:
        main()
