#!/usr/bin/env python3
"""Slide figure + table: scaling of standard PCAone and dwg (current defaults:
candidates from all pairs up to N = 5000, the sketch above), and dwg forced to
all pairs at N = 10,000 and 20,000. Simulated data of the scaling test
(data/scale/sim_N*, M = 20,000) and data/relfind/big_N100000; recall against
plink2 KING >= tau (the *_king.kin0 files).

usage: slides_scaling.py <times.tsv> <run dir> <out.pdf>
  times.tsv: name, seconds, return code, pairs (scale_new.sh output)
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 12})
INK, MUTED = "#222222", "#666666"
TAU = 2 ** -3.5
D = "/kellyData/home/albrecht/codex/relatePCA/data"


def kin0(f):
    out = set()
    for l in list(open(f))[1:]:
        t = l.split()
        if float(t[-1]) >= TAU:
            out.add(tuple(sorted((t[1], t[3]))))
    return out


def relpairs(f):
    return {tuple(sorted(l.split()[:2])) for l in list(open(f))[1:]}


def main():
    tfile, run, out = sys.argv[1:4]
    T = {}
    for l in open(tfile):
        t = l.rstrip("\n").split("\t")
        if len(t) == 4 and t[2] == "0":
            T[t[0]] = float(t[1])
    Ns = [1000, 2000, 5000, 10000, 20000, 100000]
    print("N\tstd_s\tdwg_s\tdwg_all_s\ttruth\tdwg_recall\tdwg_extra\tall_recall")
    rows = []
    for N in Ns:
        kf = f"{D}/scale/sim_N{N}_king.kin0" if N < 100000 else f"{D}/relfind/big_N100000_king.kin0"
        tr = kin0(kf)
        rec, extra, rec_all = np.nan, np.nan, np.nan
        if os.path.exists(f"{run}/dwg_N{N}.relpairs"):
            P = relpairs(f"{run}/dwg_N{N}.relpairs")
            rec, extra = len(P & tr) / len(tr), len(P - tr)
        if os.path.exists(f"{run}/dwgall_N{N}.relpairs"):
            Pa = relpairs(f"{run}/dwgall_N{N}.relpairs")
            rec_all = len(Pa & tr) / len(tr)
        r = (N, T.get(f"std_N{N}", np.nan), T.get(f"dwg_N{N}", np.nan), T.get(f"dwgall_N{N}", np.nan), len(tr), rec,
             extra, rec_all)
        rows.append(r)
        print("\t".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r))
    R = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(R[:, 0], R[:, 1], "-o", color="#666666", lw=1.8, ms=5, label="standard PCAone")
    ax.plot(R[:, 0], R[:, 2], "-o", color="#0072B2", lw=1.8, ms=5, label="dwg (default)")
    ax.axvline(5000, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(5500, ax.get_ylim()[1] * 0.8, "all pairs | sketch", color=MUTED, fontsize=10, ha="center", va="top")
    ax.set_xlabel("$N$ (individuals)", color=MUTED)
    ax.set_ylabel("run time (s)", color=MUTED)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
