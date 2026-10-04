#!/usr/bin/env python3
"""dwg_gl: rounds of IAF from the robust PCs, with and without leave-one-out."""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dwg_gl as DG  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
from gl_sim import simulate_gl  # noqa: E402

SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "xanc", "grandx"]
VARIANTS = {"rounds3": dict(rounds=3), "deshrink1": dict(rounds=1, deshrink=True)}
_OLD = {"rounds1": dict(rounds=1), "deshrink1": dict(rounds=1, deshrink=True),
            "deshrink3": dict(rounds=3, deshrink=True)}


def one(job):
    n, scen, rep, depth = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, int(os.environ.get("GL_M", "10000")), B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(np.int8)
    rel = np.arange(ds["n0"], len(G))
    true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(ds["Kped"] >= B.TAU, 1)))}
    GL, _ = simulate_gl(G, depth, seed=seed % 100000)
    rows = []
    for v, kw in VARIANTS.items():
        R = DG.dwg_gl(GL, 3, **kw)
        r2, err = B.score(R["U"], T, ds["n0"], rel, 3)
        det = {(min(i, j), max(i, j)) for i, j, _ in R["pairs"]}
        rows.append((n, scen, rep, depth, v, r2.min(), err, len(det & true) / len(true) if true else np.nan,
                     len(det - true)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    jobs = [(n, s, r, d) for d in (2, 4, 8) for n in (5, 10) for s in SCEN for r in range(2)]
    with ProcessPoolExecutor(int(os.environ.get("GL_WORKERS", "16")), initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "depth", "variant", "minR2", "err", "recall", "false_pairs"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    d["fail"] = d.minR2 < 0.95
    for stat in ("minR2", "fail", "recall", "false_pairs", "err"):
        print(f"-- {stat}")
        print(d.pivot_table(index="variant", columns=["depth", "n"], values=stat).round(3).to_string())
