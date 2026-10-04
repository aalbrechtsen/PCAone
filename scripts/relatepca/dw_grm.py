#!/usr/bin/env python3
"""detect-white on the GRM scale (no CS matrix). The matrix is the uncentred,
SNP-standardised Gram A_s = G W^-1 G'/M, W = diag(2f(1-f)), restored from the
centred GRM if the data arrive centred (A_s = C + u1' + 1u' + mu'mu). Fit with
the diagonal free, KING screen, kinship rule, rank from the noise edge (at most
k + 1), then whitening Sigma = v^1/2 (I + 2 Psi) v^1/2 with v = A_ii - L_ii and
the mean component dropped.

  dwg_D  kinship scaled by heterozygosity on the GRM scale, D_s = mean g(2-g)/w
  dwg_v  kinship scaled by the fitted noise v = A_ii - L_ii (no heterozygosity,
         no HWE assumption within individuals)
"""
import numpy as np

import illustrate as I


def grm_scaled_uncentred(G):
    f = G.mean(0) / 2
    ok = (f > 0) & (f < 1)
    w = 2 * f[ok] * (1 - f[ok])
    Gt = G[:, ok] / np.sqrt(w)
    return Gt @ Gt.T / ok.sum(), f[ok], w, G[:, ok]


def noise_edge(f, w, N):
    eg2 = 2 * f * (1 - f) + 4 * f ** 2
    M = len(f)
    return 2 * np.sqrt(np.sum((eg2 ** 2 - 16 * f ** 4) / w ** 2) / M ** 2) * np.sqrt(N)


def fit(A, tau, cand, edge, rmax, scale, iters=500):
    """scale: 'D' (fixed vector, given) or None (fitted noise v = A_ii - L_ii)"""
    n = len(A)
    eye = np.eye(n, dtype=bool)
    S = np.zeros_like(A)
    r = 1
    while True:
        L_old = None
        for _ in range(iters):
            w_, V = I.top_eig(A - S, r)
            L = (V * w_) @ V.T
            R = A - L
            v = np.maximum(np.diag(R), 1e-6) if scale is None else scale
            sel = I.kin_select(R, v, tau) & cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        if r >= min(rmax, n - 1) or I.top_eig(A - S, r + 1)[0][r] <= edge:
            break
        r += 1
    v = np.maximum(np.diag(A - L), 1e-6) if scale is None else scale
    phi = I.kin_scale(A - L, v)
    iu = np.triu_indices(n, 1)
    pairs = [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]
    return L, S, r, pairs


def dwg(G, k, tau, cand, kin_scale="v"):
    A, f, w, Gs = grm_scaled_uncentred(G)
    N = len(A)
    Ds = (Gs * (2 - Gs) / w).mean(1)
    L, S, r, pairs = fit(A, tau, cand, noise_edge(f, w, N), k + 1, Ds if kin_scale == "D" else None)
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    return I.whitened_noise(A, v, pairs, k), r, pairs
