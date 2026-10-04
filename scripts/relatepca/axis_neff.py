#!/usr/bin/env python3
"""Effective number of individuals 1/sum(u^4) of each axis of the dwg_v fit
(k = 10): ancestry axes (the first K-1 = 3) against the extra axes the noise
edge added (rank > 4)."""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402

SCEN = ["xanc", "grandx", "avuncx", "cousins", "none", "many", "mz2", "nuclear2", "childx"]


def one(job):
    n, scen, rep = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    G = ds["G"].astype(float)
    N = len(G)
    king = B.king_robust(G)
    ck = (king > B.KING_SCREEN) & ~np.eye(N, dtype=bool)
    A, f, w, Gs = W.grm_scaled_uncentred(G)
    L, S, r, _ = W.fit(A, B.TAU, ck, W.noise_edge(f, w, N), 11, None)
    ev, U = I.top_eig(L, r)
    rows = []
    for j in range(1, r):  # skip the mean component
        u = U[:, j] / np.linalg.norm(U[:, j])
        rows.append((n, scen, rep, r, j, "ancestry" if j <= 3 else "extra", 1 / np.sum(u ** 4)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    jobs = [(n, s, r) for n in (5, 10, 20) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(44, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "rank", "axis", "type", "neff"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    print(d.groupby(["n", "type"])["neff"].describe()[["count", "min", "25%", "50%", "max"]].round(1).to_string())
