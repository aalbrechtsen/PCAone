#!/usr/bin/env python3
"""evalAdmix candidate screen for dwg_v: how many PCs go into the evalAdmix
kinship? all k, the true K-1, or r-1 with r the noise-edge rank of the fit
(no k needed); from the robust PCs (dwg_v with the KING screen) or from an
unscreened fit. Accuracy (reference truth), recall of the true close pairs,
false pairs."""
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

SCEN = ["xanc", "grandx", "avuncx", "none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2",
        "err1_mz2"]
KTRUE = 4


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
    U1, r1, p1 = W.dwg(G, k, B.TAU, ck, "v")
    out["king"] = (U1, p1)
    U0, r0, _ = W.dwg(G, k, B.TAU, off, "v")
    for src, U, r in [("robust", U1, r1), ("unscreened", U0, r0)]:
        for npc_name, npc in [("allk", k), ("K-1", KTRUE - 1), ("r-1", max(r - 1, 1))]:
            ea = I.evaladmix_kin(G, U[:, :npc], intercept=True)
            Ux, _, px = W.dwg(G, k, B.TAU, (ea > B.KING_SCREEN) & off, "v")
            out[f"ea_{src}_{npc_name}"] = (Ux, px)
    rows = []
    for m, (U, pairs) in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        det = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        rec = len(det & true) / len(true) if true else np.nan
        rows.append((n, scen, rep, k, m, r2.min(), err, rec, len(det - true), r1))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    jobs = [(n, s, r, k) for k in (3, 10) for n in (5, 10, 20) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(44, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err", "recall", "false_pairs", "rank"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    order = ["king"] + [f"ea_{s}_{p}" for s in ("robust", "unscreened") for p in ("allk", "K-1", "r-1")]
    d["fail"] = d.minR2 < 0.95
    for k in (3, 10):
        x = d[d.k == k]
        t = pd.concat({"fail": x.pivot_table(index="method", columns="n", values="fail").reindex(order),
                       "recall": x.pivot_table(index="method", columns="n", values="recall").reindex(order),
                       "false": x.pivot_table(index="method", columns="n", values="false_pairs").reindex(order)},
                      axis=1).round(3)
        print(f"== k = {k}  (fit rank r: mean {x[x.method == 'king']['rank'].mean():.2f})")
        print(t.to_string())
        print("relatives' error:", x.groupby("method")["err"].mean().reindex(order).round(3).to_dict())
