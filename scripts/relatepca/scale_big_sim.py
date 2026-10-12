#!/usr/bin/env python3
"""Large scaling data (the model of scale_test.simulate, generated in SNP
blocks so that N x M never sits in memory as floats): K = 4 populations,
F_ST 0.05 (Balding-Nichols), 20% admixed between populations 0 and 1, 5%
relatives (MZ copy, child of two founders, plus a full sib), unlinked SNPs.
The pedigree is drawn once; each SNP block draws its allele frequencies,
the founders' genotypes and the relatives' genotypes from their parents',
in parallel with independent seeds, and is appended to a SNP-major PLINK
bed. The true relationships (pedigree kinship of every related pair) are
written to <out>.truth.tsv.

usage: scale_big_sim.py <N> <M> <out prefix> [workers] [block]
"""
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

K, FST = 4, 0.05
G2CODE = np.array([3, 2, 0], dtype=np.uint8)  # 0, 1, 2 copies of A1 (as sim.write_bed)


def pedigree(N, rng):
    """founder ancestry q, and the relatives: (kind, a, b), kind mz/child/sib; parents are founders"""
    n_rel = int(0.05 * N)
    n0 = N - n_rel
    pop = rng.integers(0, K, n0)
    q = np.zeros((n0, K), np.float32)
    q[np.arange(n0), pop] = 1
    adm = rng.random(n0) < 0.2
    x = rng.random(adm.sum())
    q[adm] = 0
    q[adm, 0], q[adm, 1] = x, 1 - x
    ped = []
    while len(ped) < n_rel:
        t = rng.integers(3)
        a, b = rng.choice(n0, 2, replace=False)
        if t == 0:
            ped.append(("mz", a, -1))
        else:
            ped.append(("child", a, b))
            if t == 2 and len(ped) < n_rel:
                ped.append(("sib", a, b))
    return q, ped[:n_rel], n0


def truth(ped, n0):
    """pedigree kinship of all related pairs (founders unrelated, parents always founders):
    child = mean of its parents' kinship with y, MZ = copy of a"""
    Kin = {}

    def get(i, j):
        return 0.5 if i == j else Kin.get((min(i, j), max(i, j)), 0.0)

    by_parent = {}  # founder -> relatives descending from it (only those can be related to a new one)
    for r, (kind, a, b) in enumerate(ped):
        x = n0 + r
        cand = {a} | set(by_parent.get(a, [])) | ({b} | set(by_parent.get(b, [])) if b >= 0 else set())
        for y in cand:
            k = get(a, y) if kind == "mz" else 0.5 * (get(a, y) + get(b, y))
            if k > 0:
                Kin[(min(x, y), max(x, y))] = k
        by_parent.setdefault(a, []).append(x)
        if b >= 0:
            by_parent.setdefault(b, []).append(x)
    return Kin


def block(args):
    """encoded bed bytes of one SNP block for all N individuals"""
    seed, bsz, q, kinds, A, Bp = args
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, bsz)
    a, b = p * (1 - FST) / FST, (1 - p) * (1 - FST) / FST
    P = rng.beta(a, b, (K, bsz)).clip(0.01, 0.99).astype(np.float32)
    pi = q @ P
    G = (rng.random(pi.shape, dtype=np.float32) < pi).astype(np.int8)
    G += rng.random(pi.shape, dtype=np.float32) < pi
    del pi
    R = np.empty((len(kinds), bsz), np.int8)
    mz = kinds == 0
    R[mz] = G[A[mz]]
    ch = ~mz
    if ch.any():
        ga = (rng.random((ch.sum(), bsz), dtype=np.float32) < G[A[ch]] / 2).astype(np.int8)
        gb = rng.random((ch.sum(), bsz), dtype=np.float32) < G[Bp[ch]] / 2
        R[ch] = ga + gb
    G = np.vstack([G, R])
    n = G.shape[0]
    codes = G2CODE[G.T]  # bsz x n
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.ones((bsz, pad), np.uint8)], axis=1)
    codes = codes.reshape(bsz, -1, 4)
    by = codes[:, :, 0] | (codes[:, :, 1] << 2) | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)
    # monomorphic SNPs are possible but rare (p in 0.05..0.95); kept, PCAone drops them
    return by.astype(np.uint8).tobytes()


def main():
    N, M, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    bsz = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    seed = zlib.crc32(f"scale_big|{N}|{M}".encode())
    rng = np.random.default_rng(seed)
    q, ped, n0 = pedigree(N, rng)
    kinds = np.array([0 if k == "mz" else 1 for k, _, _ in ped], np.int8)
    A = np.array([a for _, a, _ in ped], np.int64)
    Bp = np.array([b for _, _, b in ped], np.int64)
    Kin = truth(ped, n0)
    with open(out + ".truth.tsv", "w") as f:
        f.write("ID1\tID2\tkinship\trelation\n")
        for (i, j), k in sorted(Kin.items()):
            f.write(f"I{i}\tI{j}\t{k}\t\n")
    with open(out + ".fam", "w") as f:
        f.writelines(f"F I{i} 0 0 0 -9\n" for i in range(N))
    with open(out + ".bim", "w") as f:
        f.writelines(f"1\tsnp{j}\t0\t{j + 1}\tA\tC\n" for j in range(M))
    nb = (M + bsz - 1) // bsz
    seeds = np.random.SeedSequence(seed).spawn(nb)
    jobs = [(seeds[i], min(bsz, M - i * bsz), q, kinds, A, Bp) for i in range(nb)]
    with open(out + ".bed", "wb") as f, ProcessPoolExecutor(workers) as ex:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for i, by in enumerate(ex.map(block, jobs, chunksize=1)):
            f.write(by)
            if (i + 1) % max(1, nb // 20) == 0:
                print(f"block {i + 1}/{nb}", flush=True)
    print(f"{out}: N={N} M={M} relatives={len(ped)} related pairs={len(Kin)} "
          f"(>= 2^-3.5: {sum(k >= 2 ** -3.5 for k in Kin.values())})", flush=True)


if __name__ == "__main__":
    main()
