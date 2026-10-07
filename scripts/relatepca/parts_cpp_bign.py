#!/usr/bin/env python3
"""S4 (and N = 20,000 for S3): PCAone (C++) modes on existing large PLINK data
sets with family ids in the .fam (FID = family number, 'U<i>' = unrelated).

Truth: no reference panel is stored, so the truth is the standard PCA of the
unrelated individuals of the data set itself (14,000-70,000 people: precise),
with every individual projected onto it (allele frequencies and loadings of
the unrelated). Scores: min R2 over the top kk axes (unrelated individuals),
relatives' error, family axes among PCs kk+1..20 (eta^2 > 0.5), number of
pairs, time and peak memory.

usage: parts_cpp_bign.py --out FILE --bfiles a,b --kk 4 --methods standard,dwg,detect-white
          [--threads 40] [--ooc-methods dwg]"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import highn_test as HN  # noqa: E402
from sim import read_bed  # noqa: E402

PCAONE = "/kellyData/home/albrecht/codex/relatePCA/PCAone/PCAone"
MODES = {"standard": "", "dwg": "--robust dwg", "detect-white": "--robust detect-white",
         "cswhite": "--robust cswhite"}


def truth_unrelated(G, famid, kk, rng=np.random.default_rng(1)):
    """PCA of the unrelated individuals (randomized SVD), all projected"""
    U0 = np.where(famid == 0)[0]
    f = G[U0].mean(0, dtype=np.float64) / 2
    ok = (f > 0.01) & (f < 0.99)
    sd = np.sqrt(2 * f[ok] * (1 - f[ok])).astype(np.float32)
    mu = (2 * f[ok]).astype(np.float32)
    X = (G[U0][:, ok].astype(np.float32) - mu) / sd
    Q = rng.standard_normal((X.shape[1], kk + 10)).astype(np.float32)
    Y = X @ Q
    for _ in range(4):  # power iterations
        Y, _ = np.linalg.qr(Y)
        Y = X @ (X.T @ Y)
    Y, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    V = Vt[:kk].T
    del X
    T = np.zeros((len(G), kk), np.float32)
    for s in range(0, len(G), 5000):
        T[s:s + 5000] = ((G[s:s + 5000][:, ok].astype(np.float32) - mu) / sd) @ V
    return T / np.linalg.norm(T, axis=0)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bfiles", required=True)
    ap.add_argument("--kk", type=int, default=4)
    ap.add_argument("--methods", default="standard,dwg,detect-white")
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--ooc-methods", default="")
    ap.add_argument("--work", default="/kellyData/home/albrecht/codex/relatePCA/data/parts")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    rows = []
    for bf in a.bfiles.split(","):
        t0 = time.time()
        fam, _, G = read_bed(bf)
        famid = np.array([0 if r[0].startswith("U") else int(r[0]) for r in fam])
        T = truth_unrelated(G, famid, a.kk)
        del G
        print(f"# {bf}: N={len(famid)}, truth in {time.time() - t0:.0f} s", flush=True)
        base, rel = np.where(famid == 0)[0], np.where(famid > 0)[0]
        runs = [(m, k, False) for m in a.methods.split(",") for k in (a.kk, 20)]
        runs += [(m, a.kk, True) for m in a.ooc_methods.split(",") if m]
        for m, k, ooc in runs:
            out = os.path.join(a.work, f"{os.path.basename(bf)}_{m}_k{k}{'_m' if ooc else ''}")
            cmd = f"{PCAONE} -b {bf} -k {k} -n {a.threads} -v 1 {MODES[m]} {'-m 2' if ooc else ''} -o {out}"
            t1 = time.time()
            r = subprocess.run(f"/usr/bin/time -v {cmd}", shell=True, capture_output=True, text=True)
            sec = time.time() - t1
            mem = np.nan
            for ln in r.stderr.split("\n"):
                if "Maximum resident set size" in ln:
                    mem = int(ln.split()[-1]) / 1e6
            row = dict(bfile=os.path.basename(bf), N=len(famid), method=m, k=k, ooc=int(ooc), sec=round(sec, 1),
                       mem_gb=round(mem, 2))
            if r.returncode:
                row["error"] = (r.stdout + r.stderr)[-300:].replace("\n", " ")
            else:
                U = np.loadtxt(out + ".eigvecs", ndmin=2)
                r2, err = HN.score(U, T, base, rel, a.kk)
                eta = [HN.family_eta2(U[:, j], famid) for j in range(a.kk, U.shape[1])]
                row.update(minR2=round(float(r2.min()), 4), err=round(float(err), 4),
                           fam_axes=int(sum(e > 0.5 for e in eta)))
                if os.path.exists(out + ".relpairs"):
                    row["pairs"] = sum(1 for _ in open(out + ".relpairs")) - 1
            print(row, flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
