#!/usr/bin/env python3
"""Is the GRM failure a centring or a scaling problem? Fixed rank + kinship
fit with the diagonal free on (a) the centred, scaled GRM (rank k) and (b) the
scaled but uncentred Gram G W^-1 G'/M, W = diag(2f(1-f)) (rank k+1, the mean
component dropped), against the CS fit (frkin) and PCP. Small-N scenarios,
reference truth as in benchmark.py."""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402

SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2", "err1_mz2"]


def fit_fd(M_, rank, tau, noise, cand, iters=500):
    """fixed rank (stepped 1..rank) + kinship, diagonal free; kinship scaled by
    a fixed per-individual noise"""
    n = len(M_)
    eye = np.eye(n, dtype=bool)
    S = np.zeros_like(M_)
    for r in range(1, rank + 1):
        Lo = None
        for _ in range(iters):
            w, v = I.top_eig(M_ - S, r)
            L = (v * w) @ v.T
            R = M_ - L
            sel = I.kin_select(R, noise, tau) & cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            if Lo is not None and np.max(np.abs(L - Lo)) < 1e-10:
                break
            Lo = L
    return L


def one(job):
    n, scen, rep = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(float)
    N, M = G.shape
    rel = np.arange(ds["n0"], N)
    k, tau = 3, B.TAU
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    f = G.mean(0) / 2
    ok = (f > 0) & (f < 1)
    w = 2 * f[ok] * (1 - f[ok])
    Gs = G[:, ok]
    # (a) centred + scaled GRM, rank k, kinship scaled by its diagonal
    X = (Gs - 2 * f[ok]) / np.sqrt(w)
    C = X @ X.T / ok.sum()
    La = fit_fd(C, k, tau, np.diag(C), cand)
    # (b) scaled, uncentred: A_s = G W^-1 G'/M; noise D_s = mean g(2-g)/w
    As = (Gs / np.sqrt(w)) @ (Gs / np.sqrt(w)).T / ok.sum()
    Ds = (Gs * (2 - Gs) / w).mean(1)
    Lb = fit_fd(As - np.diag(Ds), k + 1, tau, Ds, cand)  # CS-like: H_s = A_s - D_s
    # (c) CS (unscaled, uncentred) = frkin
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    Lc = fit_fd(H, k + 1, tau, D, cand)
    Lp, _, _ = I.pcp_kin(C, tau, True, cand=cand)
    res = {"centred_scaled_GRM": I.top_eig(La, k)[1], "uncentred_scaled": I.cs_pcs(Lb, k)[1],
           "CS_unscaled (frkin)": I.cs_pcs(Lc, k)[1], "PCP (aarobust-kin)": I.top_eig(Lp, k)[1]}
    return [(n, scen, rep, m, B.score(U, T, ds["n0"], rel, 3)[0].min()) for m, U in res.items()]


if __name__ == "__main__":
    jobs = [(n, s, r) for n in (5, 10, 20) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(40, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    import pandas as pd
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "method", "minR2"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    print(pd.concat({"mean min R2": d.pivot_table(index="method", columns="n", values="minR2").round(3),
                     "failures": d.assign(fl=d.minR2 < 0.95).pivot_table(index="method", columns="n",
                                                                             values="fl").round(2)}, axis=1))
