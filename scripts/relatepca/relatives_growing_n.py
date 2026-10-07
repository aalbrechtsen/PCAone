#!/usr/bin/env python3
"""Accuracy with relatives as N grows, real genotypes (admixTjeck2): the
same setting as the growing-N slides (scenario "many": 8 relatives, 5 .. 60
unrelated individuals per population), several replicates, for standard PCA
and the relatives methods (four_methods_bench.run_methods).

Truth: PCA of all 366 admixTjeck2 individuals, the sample projected onto it
(slides_bign.full_truth). The method's top 3 PCs are mapped onto the 3 true
ancestry axes by a linear fit on the unrelated individuals. Per axis:
  R2 unrelated  1 - SSE / SST over the unrelated individuals
  R2 relatives  1 - SSE over the relatives / (n_rel * variance of the axis
                among the unrelated): the relatives' placement on the scale
                of the ancestry spread
and the minimum over the 3 axes.

usage: relatives_growing_n.py <out.tsv> [reps] [workers]
       relatives_growing_n.py --plot <in.tsv> <out.pdf>
"""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import four_methods_bench as FM  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import slides_bign as SB  # noqa: E402

NS = (5, 10, 20, 40, 60)
POPS = "CEU,CHB,MXL,YRI".split(",")
BFILE = "/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2"


def one(job):
    n, rep = job
    seed = zlib.crc32(f"{','.join(POPS)}|{n}|0|many|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, POPS, n, 0, B.SCENARIOS["many"], seed)
    except ValueError:
        return []
    G = ds["G"].astype(float)
    N = len(G)
    T = SB.full_truth(Gb.G_ALL, ds["cols"], G)[:, :3]
    base, rel = np.arange(ds["n0"]), np.arange(ds["n0"], N)
    rows = []
    for m, (U, _) in FM.run_methods(G).items():
        Z = np.c_[U[:, :3], np.ones(N)]
        coef, *_ = np.linalg.lstsq(Z[base], T[base], rcond=None)
        P = Z @ coef
        var = T[base].var(0)
        r2u = 1 - ((T[base] - P[base]) ** 2).mean(0) / var
        r2r = 1 - ((T[rel] - P[rel]) ** 2).mean(0) / var
        rows.append((n, rep, N, m, r2u.min(), r2r.min()))
    return rows


def plot(tsv, out):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from methods_figs import INK
    d = pd.read_csv(tsv, sep="\t")
    meth = [("standard", "standard PCA", "#D55E00", "s"), ("aarobust_kin", "aarobust-kin", "#E69F00", "^"),
            ("detect_white", "detect-white", "#009E73", "D"), ("dwg", "dwg", "#0072B2", "o")]
    # 1 - R2 on a log axis (up = better), labelled in R2: 0.99 and 0.999 differ visibly
    gap = lambda v: np.clip(1 - np.asarray(v), 3e-4, 1)  # noqa: E731
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, col, title in [(axs[0], "r2_unrel", "unrelated individuals"), (axs[1], "r2_rel", "the 8 relatives")]:
        for m, name, c, mk in meth:
            x = d[d.method == m].groupby("n")[col]
            med = x.median()
            ax.plot(med.index, gap(med.values), color=c, lw=2.2, marker=mk, ms=6, label=name, zorder=3)
            ax.fill_between(med.index, gap(x.quantile(0.1).values), gap(x.quantile(0.9).values), color=c,
                            alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_yscale("log")
        ax.set_ylim(1.15, 3e-4)
        ax.set_yticks([1, 0.1, 0.01, 0.001])
        ax.set_yticklabels(["$\\leq$0", "0.9", "0.99", "0.999"])
        ax.minorticks_off()
        ax.set_xlabel("unrelated individuals per population")
        ax.set_title(title, color=INK)
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("min $R^2$ (3 ancestry axes)")
    axs[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(sys.argv[2], sys.argv[3])
        sys.exit()
    import pandas as pd
    out = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    jobs = [(n, r) for n in NS for r in range(reps)]
    with ProcessPoolExecutor(workers, initializer=Gb.init, initargs=(BFILE,)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "rep", "N", "method", "r2_unrel", "r2_rel"])
    d.to_csv(out, sep="\t", index=False)
    for v in ("r2_unrel", "r2_rel"):
        print(v)
        print(d.pivot_table(index="method", columns="n", values=v).round(3).to_string())
        print((d.assign(f=d[v] < 0.95).pivot_table(index="method", columns="n", values="f")).round(2).to_string())
