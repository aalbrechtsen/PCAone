#!/usr/bin/env python3
"""evalAdmix screen without knowing K: iterate
  fit (KING screen) -> axes of the fit, localised ones (n_eff = 1/sum u^4 < c)
  dropped -> evalAdmix kinship from the remaining axes -> refit with
  KING | evalAdmix > 0.04 as candidates -> repeat until the pairs are stable.
Compared with the KING screen and the evalAdmix screen with the true K-1."""
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

SCEN = ["xanc", "grandx", "avuncx", "halfsibx", "none", "mz2", "nuclear2", "childx", "cousins", "many",
        "inbredpop_mz2", "err1_mz2"]


def dwg_fit(G, k, cand):
    """dwg_v returning the whitened PCs, the fit's structure axes and pairs"""
    A, f, w, Gs = W.grm_scaled_uncentred(G)
    N = len(A)
    L, S, r, pairs = W.fit(A, B.TAU, cand, W.noise_edge(f, w, N), k + 1, None)
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    U = I.whitened_noise(A, v, pairs, k)
    _, V = I.top_eig(L, r)
    return U, V[:, 1:r], pairs, r


def ea_iter(G, k, ck, c, rounds=5):
    off = ~np.eye(len(G), dtype=bool)
    U, axes, pairs, r = dwg_fit(G, k, ck)
    key = None
    for _ in range(rounds):
        neff = 1 / np.sum((axes / np.linalg.norm(axes, axis=0)) ** 4, axis=0) if axes.shape[1] else np.array([])
        keep = axes[:, neff >= c] if axes.shape[1] else axes
        if keep.shape[1] == 0:
            keep = axes[:, :1] if axes.shape[1] else U[:, :1]
        ea = I.evaladmix_kin(G, keep, intercept=True)
        U, axes, pairs, r = dwg_fit(G, k, ck | ((ea > B.KING_SCREEN) & off))
        new = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        if new == key:
            break
        key = new
    return U, pairs, r


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
    ck = (B.king_robust(G) > B.KING_SCREEN) & off
    out = {}
    U1, axes, p1, r1 = dwg_fit(G, k, ck)
    out["king"] = (U1, p1, r1)
    ea = I.evaladmix_kin(G, U1[:, :3], intercept=True)
    out["ea_trueK"] = dwg_fit(G, k, ck | ((ea > B.KING_SCREEN) & off))
    out["ea_trueK"] = (out["ea_trueK"][0], out["ea_trueK"][2], out["ea_trueK"][3])
    for c in (3, 4):
        out[f"ea_iter_neff{c}"] = ea_iter(G, k, ck, c)
    rows = []
    for m, (U, pairs, r) in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        det = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        rec = len(det & true) / len(true) if true else np.nan
        rows.append((n, scen, rep, k, m, r2.min(), err, rec, len(det - true), r))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    jobs = [(n, s, r, k) for k in (3, 10) for n in (5, 10, 20, 40) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(44, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err", "recall", "false_pairs", "rank"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    d["fail"] = d.minR2 < 0.95
    order = ["king", "ea_trueK", "ea_iter_neff3", "ea_iter_neff4"]
    for k in (3, 10):
        x = d[d.k == k]
        t = pd.concat({"fail": x.pivot_table(index="method", columns="n", values="fail").reindex(order),
                       "recall": x.pivot_table(index="method", columns="n", values="recall").reindex(order),
                       "false": x.pivot_table(index="method", columns="n", values="false_pairs").reindex(order),
                       "rank": x.pivot_table(index="method", columns="n", values="rank").reindex(order)}, axis=1).round(3)
        print(f"== k = {k}")
        print(t.to_string())
        print("relatives' error:", x.groupby("method")["err"].mean().reindex(order).round(3).to_dict())
