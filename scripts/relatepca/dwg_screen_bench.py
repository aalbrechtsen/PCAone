#!/usr/bin/env python3
"""dwg_v with different candidate screens: KING-robust (default), none (the
GRM kinship alone decides), and a two-stage evalAdmix screen (dwg_v without a
screen -> PCs -> evalAdmix kinship > 0.04 -> dwg_v again)."""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402

SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2", "err1_mz2"]


def one(job):
    n, scen, rep, k = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(float)
    N = len(G)
    rel = np.arange(ds["n0"], N)
    off = ~np.eye(N, dtype=bool)
    king = B.king_robust(G)
    cand_king = (king > B.KING_SCREEN) & off
    out = {}
    out["king_screen"], _, _ = W.dwg(G, k, B.TAU, cand_king, "v")
    U0, _, _ = W.dwg(G, k, B.TAU, off, "v")
    out["no_screen"] = U0
    kin = I.evaladmix_kin(G, U0, intercept=True)
    out["evaladmix_screen"], _, _ = W.dwg(G, k, B.TAU, (kin > B.KING_SCREEN) & off, "v")
    rows = []
    for m, U in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        rows.append((n, scen, rep, k, m, r2.min(), err))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    jobs = [(n, s, r, k) for k in (3, 10) for n in (5, 10, 20, 40) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(40, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    order = ["king_screen", "no_screen", "evaladmix_screen"]
    for k in (3, 10):
        x = d[d.k == k]
        t = x.pivot_table(index="method", columns="n", values="minR2").reindex(order).round(3)
        fl = x.assign(fl=x.minR2 < 0.95).pivot_table(index="method", columns="n", values="fl").reindex(order).round(3)
        e = x.groupby("method")["err"].mean().reindex(order).round(3)
        print(f"== k = {k}")
        print(pd.concat({"mean min R2": t, "failures": fl}, axis=1))
        print("relatives' error:", e.to_dict())
    x = d[d.k == 3]
    print("failures by scenario (k=3):")
    print(x.assign(fl=x.minR2 < 0.95).pivot_table(index="method", columns="scen", values="fl").reindex(order).round(2).to_string())
