#!/usr/bin/env python3
"""Sign-bit sketch vs float sketch for the relative finder (see relfind.py).

The float sketch Y = X Omega (s columns) is reduced to its signs, 1 bit per
column, so a pair comparison becomes a popcount of s/64 words. Each
individual's top-m neighbours (largest bit agreement) are the candidates;
recall against plink2 KING by class. Also reports the per-bit agreement of
true pairs, which decides whether bucketed (LSH) search can work.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import read_bed  # noqa: E402
from relfind import CLASSES, read_kin0  # noqa: E402


def sketch(G, s, seed, chunk=4000, kind="dense"):
    """dense: +-1 Omega (M x s); count: one +-1 per SNP in a random column
    (count sketch; cost N*M instead of N*M*s)"""
    f = G.mean(0) / 2
    sd = np.sqrt(2 * f * (1 - f))
    cols = np.where(sd > 0)[0]
    rng = np.random.default_rng(seed)
    Y = np.zeros((G.shape[0], s), dtype=np.float32)
    for c0 in range(0, len(cols), chunk):
        c = cols[c0:c0 + chunk]
        if kind == "dense":
            Om = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(len(c), s))
        else:
            Om = np.zeros((len(c), s), dtype=np.float32)
            Om[np.arange(len(c)), rng.integers(s, size=len(c))] = rng.choice([-1.0, 1.0], size=len(c))
        Y += ((G[:, c] - 2 * f[c]) / sd[c]).astype(np.float32) @ Om
    return Y


def topm(score_block, N, m):
    """score_block(r0, r1) -> (r1-r0) x N similarities; returns top-m neighbour indices"""
    nbr = np.empty((N, m), dtype=np.int64)
    for r0 in range(0, N, 500):
        r1 = min(N, r0 + 500)
        S = score_block(r0, r1).astype(np.float32)
        S[np.arange(r1 - r0), np.arange(r0, r1)] = -np.inf
        nbr[r0:r1] = np.argpartition(-S, m, axis=1)[:, :m]
    return nbr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--kin0", required=True)
    ap.add_argument("--s", default="1024,2048,4096")
    ap.add_argument("--m", default="5,10")
    ap.add_argument("--sketch", default="dense", choices=["dense", "count"])
    ap.add_argument("--kinds", default="float,bits")
    ap.add_argument("--cache", default=None, help="npy file for the float sketch")
    a = ap.parse_args()

    fam, _, G = read_bed(a.bfile)
    G = G.astype(np.int8)
    N, M = G.shape
    truth = read_kin0(a.kin0, [r[1] for r in fam])
    tclass = {n: {p for p, k in truth.items() if lo < k <= hi} for n, lo, hi in CLASSES}
    ss = [int(x) for x in a.s.split(",")]
    ms = [int(x) for x in a.m.split(",")]
    if a.cache and os.path.exists(a.cache):
        Y = np.load(a.cache)
    else:
        t0 = time.time()
        Y = sketch(G, max(ss), 1, kind=a.sketch)
        print(f"# sketch s={max(ss)}: {time.time() - t0:.1f}s", flush=True)
        if a.cache:
            np.save(a.cache, Y)
    print(f"# N={N} M={M}")
    print("kind\ts\tm\tcand/ind\t" + "\t".join(f"recall {n}" for n, _, _ in CLASSES) + "\tsearch_s")
    for s in ss:
        R = Y[:, :s] / np.linalg.norm(Y[:, :s], axis=1, keepdims=True)
        B = np.packbits(Y[:, :s] > 0, axis=1).view(np.uint64)  # N x s/64
        # per-bit agreement of the true 2nd-degree pairs and of random pairs
        p2 = np.array(sorted(tclass["2nd degree"]))
        agree = lambda I, J: 1 - np.bitwise_count(B[I] ^ B[J]).sum(1) / s  # noqa: E731
        rng = np.random.default_rng(0)
        ri, rj = rng.integers(N, size=20000), rng.integers(N, size=20000)
        ok = ri != rj
        a2, ar = agree(p2[:, 0], p2[:, 1]), agree(ri[ok], rj[ok])
        print(f"# s={s}: bit agreement 2nd degree min {a2.min():.4f} median {np.median(a2):.4f}; "
              f"random pairs mean {ar.mean():.4f} sd {ar.std():.4f}", flush=True)
        for kind in a.kinds.split(","):
            t0 = time.time()
            if kind == "float":
                nbr = topm(lambda r0, r1: R[r0:r1] @ R.T, N, max(ms))
            else:
                nbr = topm(lambda r0, r1: -np.bitwise_count(B[r0:r1, None, :] ^ B[None]).sum(2, dtype=np.int32),
                           N, max(ms))
            dt = time.time() - t0
            # rank the m neighbours properly before cutting at smaller m
            for m in ms:
                if m < max(ms):
                    if kind == "float":
                        sc = np.einsum("ij,ikj->ik", R, R[nbr])
                    else:
                        sc = -np.bitwise_count(B[:, None, :] ^ B[nbr]).sum(2)
                    sub = np.take_along_axis(nbr, np.argsort(-sc, axis=1)[:, :m], 1)
                else:
                    sub = nbr
                cand = {(min(i, j), max(i, j)) for i in range(N) for j in sub[i]}
                rec = [len(tclass[n] & cand) / max(len(tclass[n]), 1) for n, _, _ in CLASSES]
                print(f"{kind}\t{s}\t{m}\t{len(cand) / N:.1f}\t" + "\t".join(f"{x:.4f}" for x in rec) +
                      f"\t{dt:.0f}", flush=True)


if __name__ == "__main__":
    main()
