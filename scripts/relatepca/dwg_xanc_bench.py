#!/usr/bin/env python3
"""Candidate screens for dwg_v on relatives of different ancestry:
KING-robust, evalAdmix kinship from the robust PCs (dwg_v with KING first),
the union of both, and no screen. PC accuracy (reference truth) and recall of
the true close pairs (pedigree kinship >= tau) among the detected pairs."""
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

SCEN = ["xanc", "halfsibx", "grandx", "avuncx", "childx", "many", "none"]


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
    true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(ds["Kped"] >= B.TAU, 1)))}
    king = B.king_robust(G)
    ck = (king > B.KING_SCREEN) & off
    out = {}
    U1, _, p1 = W.dwg(G, k, B.TAU, ck, "v")
    out["king"] = (U1, p1)
    ea = I.evaladmix_kin(G, U1, intercept=True)
    ce = (ea > B.KING_SCREEN) & off
    out["evaladmix_robust"] = W.dwg(G, k, B.TAU, ce, "v")[::2]
    out["union"] = W.dwg(G, k, B.TAU, ck | ce, "v")[::2]
    out["none"] = W.dwg(G, k, B.TAU, off, "v")[::2]
    rows = []
    ktrue = [king[i, j] for i, j in true]
    eatrue = [ea[i, j] for i, j in true]
    for m, (U, pairs) in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        det = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        rec = len(det & true) / len(true) if true else np.nan
        fp = len(det - true)
        rows.append((n, scen, rep, k, m, r2.min(), err, rec, fp, len(true),
                     min(ktrue) if ktrue else np.nan, min(eatrue) if eatrue else np.nan))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    jobs = [(n, s, r, k) for k in (3,) for n in (5, 10, 20) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(40, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err", "recall", "false_pairs",
                                    "n_true", "king_min_true", "ea_min_true"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    order = ["king", "evaladmix_robust", "union", "none"]
    print("mean min R2 / failures / recall of true pairs / false pairs, by n (k=3):")
    for stat, f in [("minR2", "mean"), ("fail", "mean"), ("recall", "mean"), ("false_pairs", "mean")]:
        x = d.assign(fail=d.minR2 < 0.95)
        print(f"-- {stat}"); print(x.pivot_table(index="method", columns="n", values=stat, aggfunc=f).reindex(order).round(3))
    print("-- recall by scenario"); print(d.pivot_table(index="method", columns="scen", values="recall").reindex(order).round(3).to_string())
    print("-- relatives' error by scenario"); print(d.pivot_table(index="method", columns="scen", values="err").reindex(order).round(3).to_string())
    t = d[d.method == "king"].groupby("scen")[["king_min_true", "ea_min_true"]].min().round(3)
    print("-- lowest KING / evalAdmix kinship of a true close pair, by scenario"); print(t.to_string())
