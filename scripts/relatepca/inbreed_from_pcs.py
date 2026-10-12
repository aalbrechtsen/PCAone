#!/usr/bin/env python3
"""Per-sample inbreeding coefficient from any set of PCs, as PCAone --inbreed 2
computes it (src/FileUSV.cpp, src/InbredSamples.cpp): individual allele
frequencies pi = f + U U' (g/2 - f) (clipped to [1e-4, 1 - 1e-4]), expected
heterozygosity E = sum_s 2 pi (1 - pi), F = 1 - O / E. Here U may be the PCs
of a robust run (e.g. aarobust-kin --impute-diag), which PCAone --inbreed
cannot read (no singular values or loadings).

loo: leave-one-out allele frequencies. pi = H g/2 with the hat matrix
H = 11'/N + U U' (U orthogonal to 1), so each animal's own genotype enters its
pi with weight h_ii = 1/N + sum_k U_ik^2 (about (k+1)/N), which pulls pi towards
its own genotype and F down as k grows. The leave-one-out pi of a linear
smoother is (pi - h_ii g/2) / (1 - h_ii).

usage: inbreed_from_pcs.py <bfile> <eigvecs> <k> <out.tsv> [loo]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import read_bed  # noqa: E402


def main():
    bfile, vecs, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    loo = len(sys.argv) > 5 and sys.argv[5] == "loo"
    fam, _, G = read_bed(bfile)
    U = np.loadtxt(vecs)[:, :k]
    U = np.linalg.qr(U - U.mean(0))[0]  # orthonormal, orthogonal to the mean (PCAone PCs are)
    N = G.shape[0]
    h = 1 / N + (U ** 2).sum(1)  # own weight in pi
    O, E = np.zeros(N), np.zeros(N)
    for a in range(0, G.shape[1], 50000):
        g = G[:, a:a + 50000].astype(np.float64)
        f = g.mean(0) / 2
        pi = f + U @ (U.T @ (g / 2 - f))
        if loo:
            pi = (pi - h[:, None] * g / 2) / (1 - h[:, None])
        pi = np.clip(pi, 1e-4, 1 - 1e-4)
        O += (g == 1).sum(1)
        E += (2 * pi * (1 - pi)).sum(1)
    F = 1 - O / E
    with open(out, "w") as o:
        o.write("#FID\tIID\tN_SITES\tO_HET\tE_HET\tF\n")
        for r, a, b, c in zip(fam, O, E, F):
            o.write(f"{r[0]}\t{r[1]}\t{G.shape[1]}\t{int(a)}\t{b:.2f}\t{c:.6f}\n")


if __name__ == "__main__":
    main()
