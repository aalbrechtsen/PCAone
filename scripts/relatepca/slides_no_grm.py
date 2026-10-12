#!/usr/bin/env python3
"""Slide figure: how dwg avoids the N x N GRM at large N. (1) the dense route
forms A = G G'/M (N x N); (2) the operator route only ever computes
A Q = G (G' Q) / M, one SNP block at a time; (3) what is stored instead: the
structure V (N x r), the related pairs S (sparse), the noise v (N) and the
block-diagonal whitening Sigma^-1/2 (one small block per family).

usage: slides_no_grm.py <out.pdf>
"""
import sys

import matplotlib
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, MUTED = "#222222", "#666666"
GREY, LIGHT = "#bdbdbd", "#ececec"
ACC = "#0072B2"  # the operator's vectors (N x r)
REL = "#6a51a3"  # related pairs (the purple of the relatedness degrees)


def box(a, x, y, w, h, fc, ec=INK, lw=1.0, **kw):
    a.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))


def label(a, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    a.text(x, y, s, **kw)


def setup(a, title):
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title(title, loc="left", color=INK, fontsize=13.5)


def dense(a):
    setup(a, "1  dense: form the GRM")
    # G: N x M (wide)
    box(a, 0.02, 0.46, 0.40, 0.30, LIGHT)
    label(a, 0.22, 0.61, "$G$")
    label(a, 0.22, 0.42, "$N \\times M$", fontsize=11, color=MUTED)
    a.annotate("", xy=(0.52, 0.61), xytext=(0.44, 0.61), arrowprops=dict(arrowstyle="-|>", color=MUTED))
    # A: N x N
    box(a, 0.54, 0.31, 0.44, 0.44, GREY)
    label(a, 0.76, 0.80, "$A = GG^\\top\\!/M$", fontsize=12)
    label(a, 0.76, 0.27, "$N \\times N$", fontsize=11, color=MUTED)
    a.plot([0.54, 0.98], [0.31, 0.75], color=MUTED, lw=1.5)
    a.plot([0.54, 0.98], [0.75, 0.31], color=MUTED, lw=1.5)
    label(a, 0.5, 0.12, "$N = 500{,}000$: $N^2$ doubles = 2 TB\ncost $N^2 M$", fontsize=11.5, color=MUTED)


def operator(a):
    setup(a, "2  operator: only $A Q$")
    y0, h = 0.30, 0.46
    # G with one SNP block highlighted
    box(a, 0.00, y0, 0.28, h, LIGHT)
    box(a, 0.12, y0, 0.05, h, GREY, lw=1.2)
    label(a, 0.145, y0 + h + 0.04, "block $b$", fontsize=10.5, color=MUTED)
    label(a, 0.06, y0 + h / 2, "$G$")
    a.annotate("", xy=(0.37, y0 + h / 2), xytext=(0.30, y0 + h / 2), arrowprops=dict(arrowstyle="-|>", color=MUTED))
    # G_b (N x b) times the small G_b' Q (b x r)
    box(a, 0.39, y0, 0.05, h, GREY, lw=1.2)
    label(a, 0.415, y0 - 0.05, "$G_b$", fontsize=12)
    label(a, 0.47, y0 + h / 2, "$\\cdot$", fontsize=16)
    box(a, 0.50, y0 + h / 2 - 0.025, 0.06, 0.05, ACC, alpha=0.85)
    label(a, 0.53, y0 + h / 2 - 0.08, "$G_b^\\top Q$", fontsize=11)
    label(a, 0.53, y0 + h / 2 + 0.08, "$b \\times r$", fontsize=10, color=MUTED)
    label(a, 0.63, y0 + h / 2, "$\\to$", fontsize=16)
    # accumulated A Q: N x r
    box(a, 0.69, y0, 0.06, h, ACC, alpha=0.85)
    label(a, 0.72, y0 - 0.05, "$AQ$", fontsize=12)
    label(a, 0.88, y0 + h / 2, "add up\nover all\nblocks", fontsize=11, color=MUTED)
    label(a, 0.5, 0.12, "$Q$: $N \\times r$ ($r \\leq k+1$), 24 MB at 500,000\n"
                        "one pass over the SNPs; cost $N M r$", fontsize=11.5, color=MUTED)


def stored(a):
    setup(a, "3  stored instead")
    # A ~ V Lambda V' + 2 v^1/2 Psi v^1/2 + diag(v)
    y0, h = 0.36, 0.40
    box(a, 0.00, y0, 0.06, h, ACC, alpha=0.85)
    label(a, 0.03, y0 - 0.05, "$V$", fontsize=12)
    label(a, 0.03, y0 - 0.12, "structure", fontsize=10, color=MUTED)
    label(a, 0.10, y0 + h / 2, "$+$", fontsize=15)
    # S: sparse, a few related pairs
    box(a, 0.14, y0, h, h, "white")
    pairs = [(0.08, 0.15), (0.22, 0.30), (0.45, 0.52), (0.60, 0.88), (0.74, 0.80)]
    for i, j in pairs:
        for p, q in ((i, j), (j, i)):
            a.plot(0.14 + p * h, y0 + h - q * h, "s", ms=4.5, color=REL)
    label(a, 0.14 + h / 2, y0 - 0.05, "$S$", fontsize=12)
    label(a, 0.14 + h / 2, y0 - 0.12, "related pairs", fontsize=10, color=MUTED)
    label(a, 0.58, y0 + h / 2, "$+$", fontsize=15)
    box(a, 0.62, y0, 0.03, h, LIGHT)
    label(a, 0.635, y0 - 0.05, "$v$", fontsize=12)
    label(a, 0.635, y0 - 0.12, "noise", fontsize=10, color=MUTED)
    # whitening: block diagonal, one block per family
    x0 = 0.76
    w = 0.24
    yb = y0 + (h - w) / 2
    box(a, x0, yb, w, w, "white")
    off = 0.0
    for s_ in (0.035, 0.055, 0.025, 0.06, 0.04, 0.045):
        if off + s_ > w:
            break
        box(a, x0 + off, yb + w - off - s_, s_, s_, REL, ec=REL, alpha=0.7, lw=0.5)
        off += s_
    label(a, x0 + w / 2, y0 - 0.05, "$\\Sigma^{-1/2}$", fontsize=12)
    label(a, x0 + w / 2, y0 - 0.12, "per family", fontsize=10, color=MUTED)
    label(a, 0.5, 0.12, "memory $\\propto N$ (plus the pairs), never $N^2$", fontsize=11.5, color=MUTED)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
    dense(ax[0])
    operator(ax[1])
    stored(ax[2])
    fig.tight_layout(w_pad=1.0)
    fig.savefig(sys.argv[1], bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
