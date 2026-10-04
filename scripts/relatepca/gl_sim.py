#!/usr/bin/env python3
"""Genotype likelihoods from genotypes by simulating reads: depth per
individual and site ~ Poisson(d_i) with d_i ~ Gamma(shape 4) around the mean
depth; each read carries one allele of the genotype, flipped with
probability err. GL(g) = prod_reads P(read | g), g = copies of the second
allele (beagle order AA, AB, BB with B the counted allele), normalised.
Optionally written as a gzipped beagle file for PCAone -G."""
import gzip

import numpy as np


def simulate_gl(G, depth, err=0.01, seed=1, shape=4.0):
    """G: N x M genotypes 0/1/2 (copies of the counted allele). Returns
    GL (N x M x 3, normalised) and the individual mean depths."""
    rng = np.random.default_rng(seed)
    N, M = G.shape
    d_ind = rng.gamma(shape, depth / shape, N)
    D = rng.poisson(d_ind[:, None], size=(N, M))
    p_alt = np.where(G == 2, 1 - err, np.where(G == 1, 0.5, err))
    alt = rng.binomial(D, p_alt)
    ref = D - alt
    lg = np.empty((N, M, 3))
    for g in range(3):
        pa = (g / 2) * (1 - err) + (1 - g / 2) * err
        lg[:, :, g] = alt * np.log(pa) + ref * np.log(1 - pa)
    lg -= lg.max(2, keepdims=True)
    GL = np.exp(lg)
    GL /= GL.sum(2, keepdims=True)
    return GL, d_ind


def write_beagle(fn, GL, bim):
    N, M, _ = GL.shape
    with gzip.open(fn, "wt") as f:
        f.write("marker\tallele1\tallele2\t" + "\t".join(f"Ind{i}\tInd{i}\tInd{i}" for i in range(N)) + "\n")
        for s in range(M):
            t = bim[s].split()
            row = GL[:, s, :].reshape(-1)
            f.write(f"{t[0]}_{t[3]}\t0\t1\t" + "\t".join(f"{x:.6g}" for x in row) + "\n")
