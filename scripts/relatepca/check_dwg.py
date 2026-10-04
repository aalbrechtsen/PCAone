#!/usr/bin/env python3
"""PCAone --robust dwg against the Python prototype (dwg_loc.dwg_loc_ea with
the kinship check and the k0 confirmation): related pairs and |cos| per PC,
in-core and out-of-core."""
import os
import subprocess
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dwg_loc as DL  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
from sim import write_bed  # noqa: E402

pcaone, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
Gb.init("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2")
pops = "CEU,CHB,MXL,YRI".split(",")
worst_cos, all_same, n = 1.0, True, 0
for k in (3, 10):
    for nn in (5, 10, 20):
        for scen in ["none", "mz2", "childx", "cousins", "many", "xanc", "grandx", "avuncx", "inbredpop_mz2"]:
            for rep in (0, 1):
                seed = zlib.crc32(f"{','.join(pops)}|{nn}|0|{scen}|{rep}".encode())
                try:
                    ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, nn, 0, B.SCENARIOS[scen], seed)
                except ValueError:
                    continue
                G = ds["G"].astype(float)
                N = len(G)
                pre = os.path.join(work, f"{scen}_n{nn}_r{rep}")
                write_bed(pre, [["F", f"I{i}", "0", "0", "0", "-9"] for i in range(N)],
                          [Gb.BIM[i] for i in ds["cols"]], G.astype(np.int8))
                ck = (B.king_robust(G) > B.KING_SCREEN) & ~np.eye(N, dtype=bool)
                Upy, ppy, rpy = DL.dwg_loc_ea(G, k, ck, 4, kin_check=True, k0_max=0.8)
                py = {(min(i, j), max(i, j)) for i, j, _ in ppy}
                for tag, extra in [("in", ""), ("ooc", "-m 0.002")]:
                    out = f"{pre}_k{k}_{tag}"
                    r = subprocess.run(f"{pcaone} -b {pre} -k {k} -n 4 -v 0 {extra} --robust dwg -o {out}", shell=True,
                                       capture_output=True, text=True)
                    if r.returncode:
                        sys.exit(r.stdout + r.stderr)
                    U = np.loadtxt(out + ".eigvecs", ndmin=2)
                    cc = {tuple(sorted((int(l.split()[0][1:]), int(l.split()[1][1:]))))
                          for l in open(out + ".relpairs").read().strip().split("\n")[1:] if l}
                    cos = np.abs((Upy * U).sum(0) / np.linalg.norm(Upy, axis=0) / np.linalg.norm(U, axis=0))
                    same = cc == py
                    c3 = cos[:3].min()
                    worst_cos = min(worst_cos, c3)
                    all_same &= same
                    n += 1
                    flag = "" if same and c3 > 0.9999 else "   <-- CHECK"
                    print(f"k={k:2d} n={nn:2d} {scen:14s} r{rep} {tag:3s} pairs {len(cc):2d}/{len(py):2d} same={same} "
                          f"|cos| PC1-3 {c3:.6f} all {cos.min():.4f}{flag}", flush=True)
print(f"\n{n} runs; worst |cos| over PC1-3: {worst_cos:.6f}; pairs always identical: {all_same}")
