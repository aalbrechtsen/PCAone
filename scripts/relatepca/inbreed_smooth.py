#!/usr/bin/env python3
"""Per-sample inbreeding coefficient without choosing the number of PCs: the
individual allele frequencies are the best linear prediction of each
individual's ancestry from everyone's genotypes, using a robust split of the
GRM into structure L and the rest S (each individual's noise on the diagonal
and the related pairs):

  pi_s = f_s + W (g_s / 2 - f_s),   W = L (L + S)^-1

Strong ancestry directions get weight ~1, noise directions ~0, and relatives
do not feed each other's pi (their shared part is in S). F = 1 - O / E with
E = sum_s 2 pi (1 - pi), as PCAone --inbreed 2.

  pcp  L, S from PCP with the kinship rule and the diagonal unobserved
       (illustrate.pcp_kin, the aarobust-kin idea) on the centred GRM. PCP
       ends with L + S = GRM exactly (L carries the off-diagonal noise too),
       so W is computed in the space whitened by S: weights max(0, 1 - 1/lam)
       for the eigenvalues lam of S^-1/2 GRM S^-1/2 (= L GRM^-1 without the
       clipping).
  dwg  L (rank from the noise edge), S from the dwg fit (dw_grm.fit) on the
       uncentred scaled GRM; L centred (the mean dropped), W = L (L + S)^-1.
  k<n> the top n PCs of the centred GRM (projection, as PCAone --inbreed 2)

usage: inbreed_smooth.py <bfile> pcp|dwg|k<n> <out.tsv> [loo] [cand=<relpairs>]
  loo: leave-one-out pi, (pi - W_ii g/2) / (1 - W_ii)
  cand: only these pairs (a PCAone .relpairs: IDs in the first two columns)
        may enter S; without it every pair may (then within-site drift of small
        sites is taken for relatedness)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as DW  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed  # noqa: E402


def centred_grm(G, block=50000):
    N, M = G.shape
    C, m = np.zeros((N, N)), 0
    for a in range(0, M, block):
        g = G[:, a:a + block].astype(np.float64)
        f = g.mean(0) / 2
        ok = (f > 0) & (f < 1)
        X = (g[:, ok] - 2 * f[ok]) / np.sqrt(2 * f[ok] * (1 - f[ok]))
        C += X @ X.T
        m += ok.sum()
    return C / m


def scaled_uncentred(G, block=50000):
    N, M = G.shape
    A, m = np.zeros((N, N)), 0
    fs, ws = [], []
    for a in range(0, M, block):
        g = G[:, a:a + block].astype(np.float64)
        f = g.mean(0) / 2
        ok = (f > 0) & (f < 1)
        w = 2 * f[ok] * (1 - f[ok])
        X = g[:, ok] / np.sqrt(w)
        A += X @ X.T
        m += ok.sum()
        fs.append(f[ok])
        ws.append(w)
    return A / m, np.concatenate(fs), np.concatenate(ws)


def smoother_pcp(G, tau=B.TAU, cand=None):
    C = centred_grm(G)
    L, S, _ = I.pcp_kin(C, tau, free_diag=True, cand=cand)
    S = (S + S.T) / 2
    # S^-1/2 (S: noise diagonal + pairs; floor its eigenvalues)
    e, Q = np.linalg.eigh(S)
    e = np.maximum(e, 1e-3 * np.median(np.diag(S)))
    Sh, Sih = (Q * np.sqrt(e)) @ Q.T, (Q / np.sqrt(e)) @ Q.T
    lam, V = np.linalg.eigh(Sih @ C @ Sih)
    wts = np.maximum(0, 1 - 1 / np.maximum(lam, 1e-12))
    W = Sh @ (V * wts) @ V.T @ Sih
    npairs = int((np.abs(np.triu(S, 1)) > 1e-12).sum())
    # parts kept for the figures (not printed)
    return W, dict(pairs=npairs, eff_rank=float(wts.sum()), parts=dict(C=C, L=L, S=S, lam=lam, wts=wts))


def smoother_dwg(G, tau=B.TAU, rmax=40):
    A, f, w = scaled_uncentred(G)
    N = len(A)
    cand = np.ones((N, N), bool)
    L, S, r, pairs = DW.fit(A, tau, cand, DW.noise_edge(f, w, N), rmax, None)
    S = (S + S.T) / 2
    # the mean is in L (uncentred fit): centre L and act on the deviations from f
    P = np.eye(N) - 1 / N
    Lc = P @ L @ P
    W = Lc @ np.linalg.inv(Lc + S)
    return W, dict(pairs=len(pairs), rank=r)


def smoother_k(G, k):
    C = centred_grm(G)
    U = I.top_eig(C, k)[1]
    return U @ U.T, dict(k=k)


def inbreeding(G, W, loo=False, raw=False, block=50000):
    """raw: pi = W g/2 (W from the uncentred GRM); else pi = f + W (g/2 - f)"""
    N = G.shape[0]
    h = np.diag(W).copy()
    O, E = np.zeros(N), np.zeros(N)
    for a in range(0, G.shape[1], block):
        g = G[:, a:a + block].astype(np.float64)
        f = g.mean(0) / 2
        x = g / 2 if raw else g / 2 - f
        p = W @ x
        if loo:
            p = (p - h[:, None] * x) / (1 - h[:, None])
        pi = np.clip(p if raw else f + p, 1e-4, 1 - 1e-4)
        O += (g == 1).sum(1)
        E += (2 * pi * (1 - pi)).sum(1)
    return 1 - O / E, h


def smoother(G, method, cand=None):
    if method == "pcp":
        return smoother_pcp(G, cand=cand)
    if method == "dwg":
        return smoother_dwg(G)
    return smoother_k(G, int(method[1:]))


def main():
    bfile, method, out = sys.argv[1], sys.argv[2], sys.argv[3]
    loo = "loo" in sys.argv[4:]
    fam, _, G = read_bed(bfile)
    cand = None
    for a in sys.argv[4:]:
        if a.startswith("cand="):
            ids = {r[1]: i for i, r in enumerate(fam)}
            cand = np.zeros((len(fam), len(fam)), bool)
            for l in list(open(a[5:]))[1:]:
                x, y = l.split()[:2]
                cand[ids[x], ids[y]] = cand[ids[y], ids[x]] = True
    W, info = smoother(G, method, cand)
    F, h = inbreeding(G, W, loo, info.get("raw", False))
    print(method, {a: b for a, b in info.items() if a != "parts"}, "own weight W_ii: median", round(float(np.median(h)), 3), "max", round(float(h.max()), 3))
    with open(out, "w") as o:
        o.write("#FID\tIID\tW_II\tF\n")
        for r, a, b in zip(fam, h, F):
            o.write(f"{r[0]}\t{r[1]}\t{a:.4f}\t{b:.6f}\n")


if __name__ == "__main__":
    main()
