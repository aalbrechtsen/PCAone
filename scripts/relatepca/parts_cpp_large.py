#!/usr/bin/env python3
"""S3/S4: PCAone (C++) modes on large data sets with known truth, for the
parts reports: accuracy (min R2, relatives' error, family axes at k = 20,
recall / false pairs where the pedigree is known) and rough run time.

Data (regenerated with the same seeds as before, written as PLINK files):
  hn   highn_test.simulate(N, rel, 20000, 3000, crc32("N|rel|rep")); truth =
       reference panel PCA (4 axes); no pair truth (recall '?')
  hap  hapnest_test.simulate on the HAPNEST pool (seed rep+1); truth =
       reference panel (5 axes); pedigree pairs

usage: parts_cpp_large.py --out FILE --sets hn:5000:0.3:0,hap:5000:0.3:0 \
          --methods standard,dwg,detect-white,cswhite [--threads 20] [--jobs 4]
          [--work DIR] [--extra-k 20] [--ooc]"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import highn_test as HN  # noqa: E402
import parts_bench_large as PL  # noqa: E402
from sim import write_bed  # noqa: E402

PCAONE = "/kellyData/home/albrecht/codex/relatePCA/PCAone/PCAone"
MODES = {"standard": "", "dwg": "--robust dwg", "detect-white": "--robust detect-white",
         "cswhite": "--robust cswhite", "aarobust-kin": "--robust aarobust-kin --impute-diag",
         "frkin": "--robust frkin"}


def prepare(spec, work):
    src, N, rel, rep = spec.split(":")
    N, rel, rep = int(N), float(rel), int(rep)
    pre = os.path.join(work, f"{src}_N{N}_rel{int(rel * 100)}_r{rep}")
    G, T, famid, truth = (PL.data_hn if src == "hn" else PL.data_hap)(N, rel, rep)
    if not os.path.exists(pre + ".bed"):
        bim = [f"1\tsnp{i}\t0\t{i + 1}\tA\tC" for i in range(G.shape[1])]
        write_bed(pre, [[str(f) if f else f"U{i}", f"I{i}", "0", "0", "0", "-9"] for i, f in enumerate(famid)],
                  bim, G.astype(np.int8))
    return dict(spec=spec, src=src, N=N, rel=rel, rep=rep, pre=pre, T=T, famid=famid, truth=truth)


def run_one(d, method, k, threads, ooc):
    out = f"{d['pre']}_{method}_k{k}{'_m' if ooc else ''}"
    cmd = f"{PCAONE} -b {d['pre']} -k {k} -n {threads} -v 1 {MODES[method]} {'-m 2' if ooc else ''} -o {out}"
    t0 = time.time()
    r = subprocess.run(f"/usr/bin/time -v {cmd}", shell=True, capture_output=True, text=True)
    sec = time.time() - t0
    mem = np.nan
    for ln in r.stderr.split("\n"):
        if "Maximum resident set size" in ln:
            mem = int(ln.split()[-1]) / 1e6  # GB
    row = dict(set=d["spec"], src=d["src"], N=d["N"], rel=d["rel"], rep=d["rep"], method=method, k=k,
               ooc=int(ooc), sec=round(sec, 1), mem_gb=round(mem, 2))
    if r.returncode:
        row["error"] = (r.stdout + r.stderr)[-300:].replace("\n", " ")
        return row
    U = np.loadtxt(out + ".eigvecs", ndmin=2)
    kk = d["T"].shape[1]
    base, rel = np.where(d["famid"] == 0)[0], np.where(d["famid"] > 0)[0]
    r2, err = HN.score(U, d["T"], base, rel, kk)
    eta = [HN.family_eta2(U[:, j], d["famid"]) for j in range(kk, U.shape[1])]
    row.update(minR2=round(float(r2.min()), 4), err=round(float(err), 4), fam_axes=int(sum(e > 0.5 for e in eta)))
    if os.path.exists(out + ".relpairs"):
        det = set()
        for ln in list(open(out + ".relpairs"))[1:]:
            t = ln.split()
            det.add(tuple(sorted((int(t[0][1:]), int(t[1][1:])))))
        row["pairs"] = len(det)
        if d["truth"] is not None:
            tp = {p for p, v in d["truth"].items() if v >= 2 ** -3.5}
            row["recall"] = round(len(tp & det) / max(len(tp), 1), 4)
            row["false_pairs"] = sum(1 for p in det if d["truth"].get(p, 0) < 2 ** -4.5)
    return row


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--methods", default="standard,dwg,detect-white,cswhite")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--work", default="/kellyData/home/albrecht/codex/relatePCA/data/parts")
    ap.add_argument("--extra-k", type=int, default=20)
    ap.add_argument("--ooc", action="store_true", help="also run each method out-of-core (-m 2)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if "hap" in a.sets:
        PL.hap_load()
    rows = []
    for spec in a.sets.split(","):
        t0 = time.time()
        d = prepare(spec, a.work)
        print(f"# {spec}: data in {time.time() - t0:.0f} s", flush=True)
        kk = d["T"].shape[1]
        jobs = [(m, k, False) for m in a.methods.split(",") for k in (kk, a.extra_k)]
        if a.ooc:
            jobs += [(m, kk, True) for m in a.methods.split(",")]
        with ThreadPoolExecutor(a.jobs) as ex:
            for row in ex.map(lambda j: run_one(d, j[0], j[1], a.threads, j[2]), jobs):
                print(row, flush=True)
                rows.append(row)
        pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
