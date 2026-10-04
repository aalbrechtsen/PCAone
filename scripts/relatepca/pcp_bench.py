#!/usr/bin/env python3
"""aarobust-kin --impute-diag: PCP (default) against the soft-impute variant at
the noise edge with partial IRAM eigensolves (--pcp-edge --pcp-iram), scored
against the reference truth (benchmark.py) on the small-N scenarios."""
import argparse
import os
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

VARIANTS = {"pcp": "", "soft_iram": "--pcp-edge --pcp-iram"}
SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2", "err1_mz2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ns", default="5,10,20")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    fam, bim, Gall = read_bed(a.bfile)
    lab = np.array([r[1] for r in fam])
    pops = "CEU,CHB,MXL,YRI".split(",")
    jobs = [(n, s, r) for n in map(int, a.ns.split(",")) for s in SCEN for r in range(a.reps)]

    def one(job):
        n, scen, rep = job
        seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
        try:
            ds = B.make_dataset(Gall, lab, bim, pops, n, 0, B.SCENARIOS[scen], seed)
        except ValueError:
            return []
        T = B.truth_reference(ds, Gall, a.k)
        G = ds["G"]
        pre = os.path.join(a.work, f"{scen}_n{n}_r{rep}")
        write_bed(pre, [["F", f"I{i}", "0", "0", "0", "-9"] for i in range(len(G))], [bim[i] for i in ds["cols"]],
                  G.astype(np.int8))
        rel = np.arange(ds["n0"], len(G))
        rows, U = [], {}
        for v, extra in VARIANTS.items():
            subprocess.run(f"{a.pcaone} -b {pre} -k {a.k} -n 2 -v 0 --robust aarobust-kin --impute-diag {extra} "
                           f"-o {pre}_{v}", shell=True, check=True, capture_output=True)
            U[v] = np.loadtxt(f"{pre}_{v}.eigvecs", ndmin=2)
            r2, err = B.score(U[v], T, ds["n0"], rel, a.k)
            npairs = len(open(f"{pre}_{v}.relpairs").read().strip().split("\n")) - 1
            rows.append((n, scen, rep, v, r2.min(), err, npairs))
        c = np.abs((U["pcp"] * U["soft_iram"]).sum(0) / np.linalg.norm(U["pcp"], axis=0) /
                   np.linalg.norm(U["soft_iram"], axis=0)).min()
        return [r + (c,) for r in rows]

    with ThreadPoolExecutor(24) as ex, open(a.out, "w") as fo:
        fo.write("n\tscen\trep\tvariant\tminR2\terr\tnpairs\tmin_cos_pcp_vs_soft\n")
        for rows in ex.map(one, jobs):
            for r in rows:
                fo.write("\t".join(str(x) for x in r) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
