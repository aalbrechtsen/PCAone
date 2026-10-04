#!/usr/bin/env python3
"""AArobust (PCP of the GRM, diagonal unobserved) as a function of lambda,
on the benchmark.py grid. lambda = mult / sqrt(N), plus fixed lambdas that do
not scale with N."""
import argparse
import os
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed  # noqa: E402

MULTS = [0.5, 1, 1.5, 2, 3, 4, 6]
FIXED = [0.1, 0.2, 0.3, 0.5]
TAUS = [0.06, 2 ** -3.5, 0.12, 0.18]  # kinship thresholds replacing lambda


def one(job):
    pops, n, M, scen, rep = job
    fam, bim, Gall = B.DATA
    lab = np.array([r[1] for r in fam])
    seed = zlib.crc32(f"{pops}|{n}|{M}|{scen}|{rep}".encode())  # same data as benchmark.py
    try:
        ds = B.make_dataset(Gall, lab, bim, pops.split(","), n, M, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    k = len(pops.split(",")) - 1
    G, n0 = ds["G"], ds["n0"]
    N = len(G)
    rel = np.arange(n0, N)
    T = B.truth_positions(ds, k, cs=False)
    Tcs = B.truth_positions(ds, k, cs=True)
    T = Tcs = B.truth_reference(ds, B.DATA[2], k)  # score everything against the reference panel
    _, C = B.grm(G)
    rows = []
    cand = B.king_robust(G) > B.KING_SCREEN
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    for kind, vals in [("mult", MULTS), ("fixed", FIXED), ("kin", TAUS), ("kin_screen", TAUS),
                       ("fr_grm", TAUS), ("fr_grm_screen", TAUS), ("fr_cs_screen", TAUS)]:
        for v in vals:
            lam = v / np.sqrt(N) if kind == "mult" else v
            t0 = time.perf_counter()
            if kind.startswith("fr_"):
                # no norm, no lambda: rank-K fit + kinship threshold on the covariance
                cd = cand if kind.endswith("screen") else None
                if "cs" in kind:
                    L, S, it = I.lr_kin(H, k + 1, v, False, D, cand=cd)
                    U = I.cs_pcs(L, k)[1]
                    Tm = Tcs
                else:
                    L, S, it = I.lr_kin(C, k, v, True, cand=cd)
                    U = I.top_eig(L, k)[1]
                    Tm = T
                sec = time.perf_counter() - t0
                r2, err = B.score(U, Tm, n0, rel, k)
                rows.append(dict(n=n, M=G.shape[1], scen=scen, rep=rep, N=N, kind=kind, val=v, lam=lam,
                                 minR2=float(r2.min()), err=err, iters=it, sec=sec))
                continue
            if kind.startswith("kin"):
                L, S, it = I.pcp_kin(C, v, True, cand=cand if kind == "kin_screen" else None)
            else:
                L, S, it = I.rpca(C, lam, free_diag=True)
            U = I.top_eig(L, k)[1]
            sec = time.perf_counter() - t0
            r2, err = B.score(U, T, n0, rel, k)
            rows.append(dict(n=n, M=G.shape[1], scen=scen, rep=rep, N=N, kind=kind, val=v, lam=lam,
                             minR2=float(r2.min()), err=err, iters=it, sec=sec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    jobs = [("CEU,CHB,MXL,YRI", n, M, s, r) for n in [5, 10, 20, 40] for M in [0, 10000]
            for s in B.SCENARIOS for r in range(a.reps)]
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=B.init, initargs=(a.bfile,)) as ex:
        for rr in ex.map(one, jobs, chunksize=2):
            rows += rr
    cols = list(rows[0])
    with open(a.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(len(jobs), "runs ->", a.out)


if __name__ == "__main__":
    main()
