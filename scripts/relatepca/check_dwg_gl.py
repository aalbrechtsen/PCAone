#!/usr/bin/env python3
"""PCAone --robust with genotype likelihoods (-G) against the Python prototype
(dwg_gl.py, deshrunk, one round): related pairs and |cos| per PC, plus the
accuracy of both against the reference truth."""
import os
import subprocess
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dwg_gl as DG  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
from gl_sim import simulate_gl, write_beagle  # noqa: E402

pcaone, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
Gb.init("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2")
pops = "CEU,CHB,MXL,YRI".split(",")
for depth in (2, 4, 8):
    for n in (5, 10):
        for scen in ("none", "nuclear2", "many", "xanc"):
            seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|0".encode())
            ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 10000, B.SCENARIOS[scen], seed)
            T = B.truth_reference(ds, Gb.G_ALL, 3)
            G = ds["G"].astype(np.int8)
            N = len(G)
            rel = np.arange(ds["n0"], N)
            GL, _ = simulate_gl(G, depth, seed=seed % 100000)
            pre = os.path.join(work, f"{scen}_n{n}_d{depth}")
            write_beagle(pre + ".beagle.gz", GL, [Gb.BIM[i] for i in ds["cols"]])
            R = DG.dwg_gl(GL, 3, rounds=1, deshrink=True)
            py = {(min(i, j), max(i, j)) for i, j, _ in R["pairs"]}
            r = subprocess.run(f"{pcaone} -G {pre}.beagle.gz -k 3 -n 4 --robust -o {pre}_cpp", shell=True,
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stdout[-1500:] + r.stderr[-1500:])
            U = np.loadtxt(pre + "_cpp.eigvecs", ndmin=2)
            cc = {tuple(sorted((int(l.split()[0]), int(l.split()[1]))))
                  for l in open(pre + "_cpp.relpairs").read().strip().split("\n")[1:] if l}
            cos = np.abs((R["U"] * U).sum(0) / np.linalg.norm(R["U"], axis=0) / np.linalg.norm(U, axis=0))
            r2p = B.score(R["U"], T, ds["n0"], rel, 3)[0].min()
            r2c = B.score(U, T, ds["n0"], rel, 3)[0].min()
            print(f"depth {depth} n {n:2d} {scen:9s} pairs cpp {len(cc):2d} py {len(py):2d} same={cc == py} "
                  f"|cos| {cos.min():.4f}  minR2 cpp {r2c:.3f} py {r2p:.3f}", flush=True)
