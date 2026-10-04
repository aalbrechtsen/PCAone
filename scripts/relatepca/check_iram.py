#!/usr/bin/env python3
"""Stage 2 validation: PCAone --robust-engine operator (matrix-free) against the
dense engine, in-core and out-of-core. Candidates for the iram engine come from
plink2 --make-king-table (--kinship), which is the same KING-robust estimator
the dense engine computes internally."""
import argparse
import os
import subprocess
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402


def run(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return time.time() - t0


def load(pre):
    U = np.loadtxt(pre + ".eigvecs", ndmin=2)
    lines = open(pre + ".relpairs").read().strip().split("\n")[1:]
    pairs = {tuple(sorted(l.split()[:2])) for l in lines if l}
    return U, pairs


def subspace_cos(U, V):
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    return float(np.linalg.svd(Qu.T @ Qv, compute_uv=False).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    fam, bim, Gall = read_bed(a.bfile)
    lab = np.array([r[1] for r in fam])
    pops = "CEU,CHB,MXL,YRI"
    worst = {}
    for n in [5, 10, 40]:
        for scen in ["none", "mz2", "childx", "many", "inbredpop_mz2"]:
            seed = zlib.crc32(f"{pops}|{n}|0|{scen}|0".encode())
            try:
                ds = B.make_dataset(Gall, lab, bim, pops.split(","), n, 0, B.SCENARIOS[scen], seed)
            except ValueError:
                continue
            G = ds["G"]
            ids = [f"I{i}" for i in range(len(G))]
            pre = os.path.join(a.work, f"{scen}_n{n}")
            write_bed(pre, [["F", x, "0", "0", "0", "-9"] for x in ids], [bim[i] for i in ds["cols"]],
                      G.astype(np.int8))
            run(f"{a.plink2} --bfile {pre} --make-king-table --king-table-filter {B.KING_SCREEN} "
                f"--threads 4 --out {pre}_king")
            for mode in ["detect-white", "frkin", "cswhite"]:
                base = f"{a.pcaone} -b {pre} -k 3 -n 4 -v 0 --robust {mode}"
                kin = f"--kinship {pre}_king.kin0"
                td = run(f"{base} --robust-engine dense -o {pre}_{mode}_dense")
                Ud, pd_ = load(f"{pre}_{mode}_dense")
                for tag, extra in [("op", ""), ("op-ooc", "-m 0.002")]:
                    ti = run(f"{base} --robust-engine operator {kin} {extra} -o {pre}_{mode}_{tag}")
                    Ui, pi_ = load(f"{pre}_{mode}_{tag}")
                    cs = subspace_cos(Ud, Ui)
                    c1 = float(np.min(np.abs(np.sum(Ud * Ui, 0)) /
                                      (np.linalg.norm(Ud, axis=0) * np.linalg.norm(Ui, axis=0))))
                    same = pd_ == pi_
                    key = (mode, tag)
                    w = worst.setdefault(key, [1.0, 1.0, True])
                    w[0], w[1], w[2] = min(w[0], c1), min(w[1], cs), w[2] and same
                    flag = "" if (c1 > 0.9999 and same) else "   <-- CHECK"
                    print(f"n={n:2d} {scen:14s} {mode:13s} {tag:8s} pairs {len(pi_)} same={same} "
                          f"|cos| per PC {c1:.6f} subspace {cs:.6f}  dense {td:.2f}s {tag} {ti:.2f}s{flag}",
                          flush=True)
    print("\nworst: mode, engine, min |cos| per PC, min subspace cos, pairs always identical")
    for (m, t), (c1, cs, same) in worst.items():
        print(f"  {m:13s} {t:8s} {c1:.6f} {cs:.6f} {same}")


if __name__ == "__main__":
    main()
