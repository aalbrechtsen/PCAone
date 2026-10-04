#!/usr/bin/env python3
"""Stage 2 scaling test for PCAone --robust (dense vs operator engine).

Simulated data (the real panel has only 374 individuals): Balding-Nichols
populations (K = 4, F_ST = 0.05) plus an admixed group, unlinked SNPs, and
relatives (MZ copies, parent-offspring trios, full sibs) making up ~5% of the
sample. Candidate pairs for the operator engine come from plink2
--make-king-table. Reports wall time (PCAone -n THREADS), passes, and the
agreement between engines where the dense one is run.
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import write_bed  # noqa: E402


def simulate(N, M, rng, K=4, fst=0.05):
    p = rng.uniform(0.05, 0.95, M)
    a, b = p * (1 - fst) / fst, (1 - p) * (1 - fst) / fst
    P = rng.beta(a, b, (K, M)).clip(0.01, 0.99)
    n_rel = int(0.05 * N)
    n0 = N - n_rel
    # 80% from single populations, 20% admixed between populations 0 and 1
    pop = rng.integers(0, K, n0)
    q = np.zeros((n0, K))
    q[np.arange(n0), pop] = 1
    adm = rng.random(n0) < 0.2
    x = rng.random(adm.sum())
    q[adm] = 0
    q[adm, 0], q[adm, 1] = x, 1 - x
    pi = q @ P
    G = (rng.random((n0, M)) < pi).astype(np.int8) + (rng.random((n0, M)) < pi).astype(np.int8)
    rel = []
    while len(rel) < n_rel:
        t = rng.integers(3)
        a_, b_ = rng.choice(n0, 2, replace=False)
        if t == 0:  # MZ copy
            rel.append(G[a_].copy())
        else:       # child of a_ and b_ (one allele from each, unlinked)
            ga = (rng.random(M) < G[a_] / 2).astype(np.int8)
            gb = (rng.random(M) < G[b_] / 2).astype(np.int8)
            rel.append(ga + gb)
            if t == 2 and len(rel) < n_rel:  # a sib
                ga = (rng.random(M) < G[a_] / 2).astype(np.int8)
                gb = (rng.random(M) < G[b_] / 2).astype(np.int8)
                rel.append(ga + gb)
    return np.vstack([G, np.array(rel[:n_rel])])


def run(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed: {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return time.time() - t0, r.stdout + r.stderr


def subspace_cos(U, V):
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    return float(np.linalg.svd(Qu.T @ Qv, compute_uv=False).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--work", required=True)
    ap.add_argument("--Ns", default="1000,2000,5000,10000,20000")
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--dense-max", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    rng = np.random.default_rng(7)
    print("N\tengine\tmode\tseconds\tpasses\tpairs\tcos_vs_dense")
    for N in [int(x) for x in a.Ns.split(",")]:
        pre = os.path.join(a.work, f"sim_N{N}")
        if not os.path.exists(pre + ".bed"):
            G = simulate(N, a.M, rng)
            G = G[:, G.min(0) != G.max(0)]
            bim = [f"1\tsnp{i}\t0\t{i + 1}\tA\tC" for i in range(G.shape[1])]
            write_bed(pre, [["F", f"I{i}", "0", "0", "0", "-9"] for i in range(N)], bim, G)
        tk, _ = run(f"{a.plink2} --bfile {pre} --make-king-table --king-table-filter 0.04 "
                    f"--threads {a.threads} --out {pre}_king")
        print(f"{N}\tplink2-KING\t-\t{tk:.1f}\t-\t-\t-", flush=True)
        ts, _ = run(f"{a.pcaone} -b {pre} -k 3 -n {a.threads} -v 1 -o {pre}_std")
        print(f"{N}\tstandard (-d 2)\t-\t{ts:.1f}\t-\t-\t-", flush=True)
        Ud = {}
        for mode in ["detect-white", "frkin"]:
            for eng in ["dense", "operator"]:
                if eng == "dense" and N > a.dense_max:
                    continue
                extra = f"--kinship {pre}_king.kin0" if eng == "operator" else ""
                t, log = run(f"{a.pcaone} -b {pre} -k 3 -n {a.threads} -v 1 --robust {mode} "
                             f"--robust-engine {eng} {extra} -o {pre}_{mode}_{eng}")
                U = np.loadtxt(f"{pre}_{mode}_{eng}.eigvecs", ndmin=2)
                npairs = len(open(f"{pre}_{mode}_{eng}.relpairs").read().strip().split("\n")) - 1
                m = re.search(r"(\d+)\s*passes over the genotypes in total", log)
                passes = m.group(1) if m else "1"
                c = ""
                if eng == "dense":
                    Ud[mode] = U
                elif mode in Ud:
                    c = f"{subspace_cos(Ud[mode], U):.6f}"
                print(f"{N}\t{eng}\t{mode}\t{t:.1f}\t{passes}\t{npairs}\t{c}", flush=True)


if __name__ == "__main__":
    main()
