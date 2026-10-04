#!/usr/bin/env python3
"""Stage 1 validation: PCAone --robust (C++) against the Python prototypes.

For benchmark datasets, each --robust mode is run on a PLINK file of the data
and compared with the prototype it ports:
  detect-white    benchmark.py hyb_cs_fdA (detection rank from the noise edge, at most k + 1)
  cswhite         benchmark.py white_cs_king
  frkin           benchmark.py lrkin_cs_fd
  aarobust-kin    benchmark.py aarobust_kin
Required: the same detected related pairs and |cos| per PC > 0.9999 (subspace
cosine reported too), in-core and out-of-core (-m).
"""
import argparse
import os
import subprocess
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

MODES = {"detect-white": "hyb_cs_fdA", "cswhite": "white_cs_king", "frkin": "lrkin_cs_fd",
         "aarobust-kin --impute-diag": "aarobust_kin"}


def cos_per_pc(U, V):
    c = np.abs(np.sum(U * V, 0)) / (np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0))
    return float(c.min())


def subspace_cos(U, V):
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    return float(np.linalg.svd(Qu.T @ Qv, compute_uv=False).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ks", default="3", help="comma-separated -k values")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    fam, bim, Gall = read_bed(a.bfile)
    lab = np.array([r[1] for r in fam])
    pops = "CEU,CHB,MXL,YRI"
    worst = {m: [1.0, 1.0, True] for m in MODES}
    n_ok = 0
    for k, n in [(int(k_), n_) for k_ in a.ks.split(",") for n_ in [5, 10, 20, 40]]:
        for scen in ["none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2", "err1_mz2"]:
            seed = zlib.crc32(f"{pops}|{n}|0|{scen}|0".encode())
            try:
                ds = B.make_dataset(Gall, lab, bim, pops.split(","), n, 0, B.SCENARIOS[scen], seed)
            except ValueError:
                continue
            G = ds["G"]
            ids = [f"I{i}" for i in range(len(G))]
            pre = os.path.join(a.work, f"{scen}_n{n}_k{k}")
            write_bed(pre, [["F", x, "0", "0", "0", "-9"] for x in ids], [bim[i] for i in ds["cols"]],
                      G.astype(np.int8))
            iu = np.triu_indices(len(G), 1)
            ref = B.run_methods(G, k, B.TAU, [], only=set(MODES.values()))
            for mode, pm in MODES.items():
                Upy, _, info = ref[pm]
                py_pairs = {(min(i, j), max(i, j)) for i, j, _ in info.get("pairs", [])}
                for tag, extra in [("in-core", ""), ("ooc", "-m 0.002")]:
                    t0 = time.time()
                    r = subprocess.run(f"{a.pcaone} -b {pre} -k {k} -n 4 -v 0 {extra} --robust {mode} "
                                       f"-o {pre}_{mode.split()[0]}_{tag}", shell=True, capture_output=True, text=True)
                    if r.returncode:
                        sys.exit(f"PCAone failed: {mode} {tag}\n{r.stdout}\n{r.stderr}")
                    sec = time.time() - t0
                    U = np.loadtxt(f"{pre}_{mode.split()[0]}_{tag}.eigvecs", ndmin=2)
                    lines = open(f"{pre}_{mode.split()[0]}_{tag}.relpairs").read().strip().split("\n")[1:]
                    c_pairs = {tuple(sorted((int(l.split()[0][1:]), int(l.split()[1][1:])))) for l in lines if l}
                    same = c_pairs == py_pairs if "pairs" in info else True
                    cp, cs = cos_per_pc(Upy, U), subspace_cos(Upy, U)
                    worst[mode][0] = min(worst[mode][0], cp)
                    worst[mode][1] = min(worst[mode][1], cs)
                    worst[mode][2] &= same
                    flag = "" if (cp > 0.9999 and same) else "   <-- CHECK"
                    print(f"k={k:2d} n={n:2d} {scen:9s} {mode:27s} {tag:8s} pairs {len(c_pairs)} same={same} "
                          f"|cos| per PC {cp:.6f} subspace {cs:.6f} ({sec:.2f} s){flag}", flush=True)
                    n_ok += cp > 0.9999 and same
    print("\nworst over all datasets: mode, min |cos| per PC, min subspace cos, pairs always identical")
    for m, (cp, cs, same) in worst.items():
        print(f"  {m:15s} {cp:.6f} {cs:.6f} {same}")


if __name__ == "__main__":
    main()
