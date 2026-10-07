#!/usr/bin/env python3
"""S3 (large N with relatives) for the parts benchmark: the combinations of
parts_bench.run_combo on large simulated data sets, in Python (dense).

Data
  hn   highn_test.simulate: K=5 Balding-Nichols populations (F_ST 0.05 ..
       0.002) + an admixed group, families (MZ, sibs, trios, nuclear,
       3-generation, first cousins), rel = 10% or 30% of the sample in
       families; truth = PCA of a 3000-person reference panel (4 axes).
       Same seeds as data/highn (zlib.crc32("N|rel|rep")).
  hap  hapnest_test.simulate: HAPNEST (6 ancestries, real LD) with families
       made by transmission, incl. relatives of different ancestry; truth =
       PCA of the 3600-person reference panel (5 axes); true pairs from the
       pedigree (kinship >= tau).
Scores: min R2 over the true axes (unrelated individuals), relatives' error,
family axes (k = 20: PCs beyond the true axes with family eta^2 > 0.5),
recall and false pairs (hap: true pedigree pairs; false = kinship < 2^-4.5),
seconds.

usage: parts_bench_large.py --out FILE [--Ns 2000] [--workers 20] (set
OMP_NUM_THREADS for the threads per job)"""
import argparse
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import hapnest_test as HT  # noqa: E402
import highn_test as HN  # noqa: E402
import illustrate as I  # noqa: E402
import parts_bench as P  # noqa: E402

HAPDIR = "/kellyData/home/albrecht/codex/relatePCA/data/hapnest/"
HAPPOOL = {}
TAU3 = 2 ** -4.5

COMBOS = ["standard", "cswhite", "topr/raw/k/0/D/white", "topr/unc/kfe/1/v/whitec", "topr/unc/ke/1/v/whitec",
          "topr/unc/k/0/v/whitec", "topr/unc/ke/1/v/L", "pcp/cgrm/k/0/v/L", "pcp/unc/kE/1/v/whitec",
          "pcp/cgrm/kE/1/v/L"]


def hap_load():
    from sim import read_bed
    fam, bim, G = read_bed(HAPDIR + "hapnest_pool")
    info = {ln.split()[0]: ln.split()[1:] for ln in list(open(HAPDIR + "individuals.tsv"))[1:]}
    ids = [r[1] for r in fam]
    HAPPOOL.update(G=G.astype(np.int8), bim=bim, anc=np.array([info[x][0] for x in ids]),
                   st=np.array([info[x][1] for x in ids]))


def data_hn(N, rel, rep):
    G, famid, ftype, R = HN.simulate(N, rel, 20000, 3000, zlib.crc32(f"{N}|{rel}|{rep}".encode()))
    keep = G.min(0) != G.max(0)
    G, R = G[:, keep].astype(float), R[:, keep].astype(float)
    return G, HN.reference_truth(R, G, 4), famid, None


def data_hap(N, rel, rep):
    H = HAPPOOL
    pool, ref = np.where(H["st"] == "pool")[0], np.where(H["st"] == "ref")[0]
    Gs, famid, ftype, truth, _ = HT.simulate(H["G"][pool], H["anc"][pool], H["bim"], N, rel, rep + 1)
    keep = Gs.min(0) != Gs.max(0)
    Gs = Gs[:, keep].astype(float)
    T, _ = HT.reference_truth(H["G"][ref][:, keep].astype(float), Gs, 5)
    return Gs, T, famid, truth


def one(job):
    src, N, rel, rep, k, combos = job
    G, T, famid, truth = (data_hn if src == "hn" else data_hap)(N, rel, rep)
    kk = T.shape[1]
    base, relidx = np.where(famid == 0)[0], np.where(famid > 0)[0]
    t = time.time()
    king = B.king_robust(G)
    ck = (king > B.KING_SCREEN) & ~np.eye(len(G), dtype=bool)
    t_king = time.time() - t
    tp = {p for p, v in truth.items() if v >= B.TAU} if truth is not None else None
    rows = []
    for c in combos:
        t = time.time()
        if c == "standard":
            U, det = P.standard(G, k), set()
        elif c == "cswhite":  # CS (HWE) diagonal, KING >= tau decides, whitening with v = D
            pp = B.pairs_above(np.where(ck, king, 0), B.TAU)
            U, det = I.whitened_cs(G, pp, k), {(i, j) for i, j, _ in pp}
        elif c.startswith("dec:"):
            U, det, _ = P.run_dec(G, c, k, ck, king)
        else:
            U, det, _ = P.run_combo(G, c, k, ck)
        sec = time.time() - t + (0 if c == "standard" else t_king)
        r2, err = HN.score(U, T, base, relidx, kk)
        eta = [HN.family_eta2(U[:, j], famid) for j in range(kk, U.shape[1])]
        rec = fp = np.nan
        if tp is not None and c != "standard":
            rec = len(tp & det) / max(len(tp), 1)
            fp = sum(1 for p in det if truth.get(p, 0) < TAU3)
        rows.append((src, N, rel, rep, k, c, float(r2.min()), float(err), int(sum(e > 0.5 for e in eta)),
                     rec, fp, len(det), round(sec, 2)))
        print(rows[-1], flush=True)
    return rows


COLS = ["src", "N", "rel", "rep", "k", "combo", "minR2", "err", "fam_axes", "recall", "false_pairs", "pairs", "sec"]

if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ns", default="2000")
    ap.add_argument("--srcs", default="hn,hap")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--combos", default=",".join(COMBOS))
    a = ap.parse_args()
    combos = a.combos.split(",")
    jobs = []
    for N in map(int, a.Ns.split(",")):
        for src in a.srcs.split(","):
            for rel in (0.1, 0.3):
                for rep in range(2):
                    for k in ((4 if src == "hn" else 5), 20):
                        jobs.append((src, N, rel, rep, k, combos))
    if "hap" in a.srcs:
        hap_load()  # before the fork: shared
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rr in ex.map(one, jobs, chunksize=1) for r in rr]
    pd.DataFrame(rows, columns=COLS).to_csv(a.out, sep="\t", index=False)
    print(f"{len(jobs)} jobs in {time.time() - t0:.0f} s")
