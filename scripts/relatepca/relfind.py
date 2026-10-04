#!/usr/bin/env python3
"""Prototype: fast detection of close relatives without an all-pairs KING.

Residual sketch + nearest neighbours:
  1. X = standardised genotypes (N x M); U = top structure PCs (from a fast
     standard PCA, e.g. PCAone -k K-1).
  2. One pass: Y = X Omega / sqrt(s) for a random +-1 matrix Omega (M x s).
  3. Residual sketch R = (I - U U') Y: ancestry removed without forming the
     residual genotypes. Rows normalised to unit length.
  4. For close relatives the residual correlation is about 2 phi (0.25 for 2nd
     degree); for unrelated pairs it is 0 with sd about 1/sqrt(s). Each
     individual's top-m neighbours in R are the candidate pairs.
  5. Exact KING-robust only for the candidates.

Here the neighbour search is exact (blocked products, N^2 s) to measure the
separation; a fast approximate search replaces it later. Truth: plink2 KING
over all pairs (the .kin0 of the data).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import read_bed  # noqa: E402

CLASSES = [("MZ/dup (>0.354)", 0.354, 1.0), ("1st degree", 0.177, 0.354), ("2nd degree", 0.0884, 0.177),
           ("3rd degree", 0.0442, 0.0884)]


def read_kin0(fn, ids):
    ix = {x: i for i, x in enumerate(ids)}
    pairs = {}
    with open(fn) as f:
        head = f.readline().lstrip("#").split()
        a, b, k = head.index("IID1"), head.index("IID2"), head.index("KINSHIP")
        for line in f:
            t = line.split()
            i, j = ix[t[a]], ix[t[b]]
            pairs[(min(i, j), max(i, j))] = float(t[k])
    return pairs


def king_pairs(G, pairs):
    """KING-robust for the given pairs (G: N x M int8, 0/1/2)"""
    het = (G == 1)
    nh = het.sum(1).astype(float)
    out = np.empty(len(pairs))
    P = np.array(pairs)
    for s0 in range(0, len(P), 2000):
        I, J = P[s0:s0 + 2000, 0], P[s0:s0 + 2000, 1]
        gi, gj = G[I], G[J]
        hh = ((gi == 1) & (gj == 1)).sum(1)
        i0 = (((gi == 0) & (gj == 2)) | ((gi == 2) & (gj == 0))).sum(1)
        mn = np.maximum(np.minimum(nh[I], nh[J]), 1)
        out[s0:s0 + 2000] = (hh - 2 * i0) / (2 * mn) + 0.5 - (nh[I] + nh[J]) / (4 * mn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--pcs", required=True, help="PCAone .eigvecs with the structure PCs")
    ap.add_argument("--kin0", required=True, help="plink2 KING table (truth)")
    ap.add_argument("--s", default="128,256,512,1024")
    ap.add_argument("--m", default="5,10,20")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    t0 = time.time()
    fam, bim, G = read_bed(a.bfile)
    G = G.astype(np.int8)
    N, M = G.shape
    ids = [r[1] for r in fam]
    truth = read_kin0(a.kin0, ids)
    U = np.loadtxt(a.pcs, ndmin=2)
    U, _ = np.linalg.qr(U)
    f = G.mean(0) / 2
    sd = np.sqrt(2 * f * (1 - f))
    keep = sd > 0
    print(f"# N={N} M={M} structure PCs={U.shape[1]} truth pairs (KING>0.04)={len(truth)} "
          f"read {time.time() - t0:.1f}s", flush=True)
    rng = np.random.default_rng(a.seed)
    smax = max(int(x) for x in a.s.split(","))
    # one pass: Y = X Omega for the largest s (smaller s use the first columns)
    t0 = time.time()
    Om = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(int(keep.sum()), smax))
    Y = np.zeros((N, smax), dtype=np.float32)
    cols = np.where(keep)[0]
    for c0 in range(0, len(cols), 4000):
        c = cols[c0:c0 + 4000]
        X = ((G[:, c] - 2 * f[c]) / sd[c]).astype(np.float32)
        Y += X @ Om[c0:c0 + len(c)]
    t_sketch = time.time() - t0
    print(f"# sketch pass (s={smax}): {t_sketch:.1f}s", flush=True)

    tclass = {name: {p for p, k in truth.items() if lo < k <= hi} for name, lo, hi in CLASSES}
    print("s\tm\tcandidates\tcand/ind\t" + "\t".join(f"recall {n}" for n, _, _ in CLASSES) +
          "\tsearch_s\tking_s\tsep_2nd_min\tsep_unrel_max")
    for s in [int(x) for x in a.s.split(",")]:
        R = Y[:, :s].astype(np.float64)
        R = R - U @ (U.T @ R)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        R = R.astype(np.float32)
        mmax = max(int(x) for x in a.m.split(","))
        t0 = time.time()
        nbr = np.empty((N, mmax), dtype=np.int64)
        nsim = np.empty((N, mmax), dtype=np.float32)
        for r0 in range(0, N, 2000):
            Sb = R[r0:r0 + 2000] @ R.T
            Sb[np.arange(Sb.shape[0]), np.arange(r0, r0 + Sb.shape[0])] = -np.inf
            idx = np.argpartition(-Sb, mmax, axis=1)[:, :mmax]
            v = np.take_along_axis(Sb, idx, 1)
            o = np.argsort(-v, axis=1)
            nbr[r0:r0 + 2000] = np.take_along_axis(idx, o, 1)
            nsim[r0:r0 + 2000] = np.take_along_axis(v, o, 1)
        t_search = time.time() - t0
        # separation: sketch correlation of true 2nd-degree pairs vs the top
        # unrelated neighbour of each individual
        p2 = list(tclass["2nd degree"])
        c2 = np.array([R[i] @ R[j] for i, j in p2]) if p2 else np.array([np.nan])
        top_unrel = []
        for i in range(0, N, 50):
            for j, v in zip(nbr[i], nsim[i]):
                if (min(i, j), max(i, j)) not in truth:
                    top_unrel.append(v)
                    break
        for m in [int(x) for x in a.m.split(",")]:
            cand = {(min(i, j), max(i, j)) for i in range(N) for j in nbr[i, :m]}
            t0 = time.time()
            kin = king_pairs(G, sorted(cand))
            t_king = time.time() - t0
            rec = [len(tclass[n] & cand) / max(len(tclass[n]), 1) for n, _, _ in CLASSES]
            print(f"{s}\t{m}\t{len(cand)}\t{len(cand) / N:.1f}\t" + "\t".join(f"{x:.4f}" for x in rec) +
                  f"\t{t_search:.1f}\t{t_king:.1f}\t{np.nanmin(c2):.3f}\t{np.max(top_unrel):.3f}", flush=True)
    for n, _, _ in CLASSES:
        print(f"# {n}: {len(tclass[n])} true pairs")


if __name__ == "__main__":
    main()
