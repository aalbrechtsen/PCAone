#!/usr/bin/env python3
"""detect-white-GRM (dw_grm.py) against detect-white (CS) and aarobust-kin:
small-N grid against the reference truth, and family axes at N = 2000."""
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


def methods(G, k, tau, cand):
    out = {}
    U, r, _ = W.dwg(G, k, tau, cand, "D")
    out["dwg_D"] = U
    U, r2, _ = W.dwg(G, k, tau, cand, "v")
    out["dwg_v"] = U
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L, S, _, rc = I.lr_kin_fd_auto(H, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
    pp = I.lr_kin_pairs(H, D, L, S)
    AM = G @ G.T / G.shape[1]
    out["detect_white"] = I.whitened_noise(AM, np.diag(AM) - np.diag(L), pp, k)
    return out, {"dwg_D": r, "dwg_v": r2, "detect_white": rc}


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
    rel = np.arange(ds["n0"], len(G))
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    out, ranks = methods(G, k, B.TAU, cand)
    _, C = B.grm(G)
    Lp, _, _ = I.pcp_kin(C, B.TAU, True, cand=cand)
    out["aarobust_kin"] = I.top_eig(Lp, k)[1]
    rows = []
    for m, U in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        rows.append((n, scen, rep, k, m, r2.min(), err, ranks.get(m, np.nan)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    jobs = [(n, s, r, k) for k in (3, 10) for n in (5, 10, 20, 40) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(40, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err", "rank"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    order = ["aarobust_kin", "detect_white", "dwg_D", "dwg_v"]
    for k in (3, 10):
        x = d[d.k == k]
        t = x.pivot_table(index="method", columns="n", values="minR2").reindex(order).round(3)
        fl = x.assign(fl=x.minR2 < 0.95).pivot_table(index="method", columns="n", values="fl").reindex(order).round(3)
        e = x.groupby("method")["err"].mean().reindex(order).round(3)
        rk = x.groupby("method")["rank"].mean().reindex(order).round(2)
        print(f"== k = {k}")
        print(pd.concat({"mean min R2": t, "failures": fl}, axis=1))
        print("relatives' error:", e.to_dict(), " mean rank:", rk.to_dict())
    x = d[d.k == 3]
    print("failures by scenario (k=3):")
    print(x.assign(fl=x.minR2 < 0.95).pivot_table(index="method", columns="scen", values="fl").reindex(order).round(2))
