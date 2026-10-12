#!/usr/bin/env python3
"""Individual allele frequencies (pi) for inbreeding F and relatedness (kinship,
IBD k0/k1/k2), with and without a number of PCs k. One module for the PCP-pi
benchmark (report/inbreeding/PLAN_pcp_pi.md).

pi estimators, each a linear smoother pi = f + W (g/2 - f) (f: sample
frequency) or, for the PCs, the regression of g/2 on [U, 1]:

  pca k     top-k PCs of a PCAone run (U), with an intercept, as PCAone's
            final relatedness (src/Robust.cpp final_relatedness): the IAF are
            the least-squares fit of g/2 on [U, 1]
  pcp       W = Lt C^-1, from the PCP split of the centred GRM C (inbreed_smooth
            smoother_pcp: Lt = S^1/2 V max(0, lam - 1) V' S^1/2, so W = Lt C^-1)
  dwg       W = Lc (Lc + S)^-1 from the dwg fit (inbreed_smooth smoother_dwg)

Leave-out versions (for pair i, j with family F, connected through the pairs
the method calls):
  pca:      the regression refitted without the rows of F (as PCAone)
  pcp, dwg: the exact best linear prediction without F,
            W_{i,-F} = Lt_{i,-F} C_{-F,-F}^-1, f from the others
  loo:      F = {i} (for inbreeding)

IBD: the likelihood of PCAone (P(g_i, g_j | IBD 0, 1, 2) with pi_i, pi_j; IBD1 and
IBD2 with the pair mean) maximised on the simplex. The log-likelihood is
concave, so the maximum is the interior stationary point if there is one,
else on an edge (each edge a 1-D concave problem).
"""
import os
import subprocess

import numpy as np


# ---------------------------------------------------------------- IBD
def ibd_lik(ga, gb, pa, pb):
    """M x 3: P(ga, gb | IBD = 0, 1, 2), as PCAone final_relatedness"""
    pa = np.clip(pa, 1e-3, 1 - 1e-3)
    pb = np.clip(pb, 1e-3, 1 - 1e-3)

    def hwe(g, p):
        return np.where(g == 0, (1 - p) ** 2, np.where(g == 1, 2 * p * (1 - p), p * p))

    def other(g, shared, p):
        d = g - shared
        return np.where(d == 0, 1 - p, np.where(d == 1, p, 0.0))

    ps = 0.5 * (pa + pb)
    L = np.empty((len(ga), 3))
    L[:, 0] = hwe(ga, pa) * hwe(gb, pb)
    L[:, 1] = (1 - ps) * other(ga, 0, pa) * other(gb, 0, pb) + ps * other(ga, 1, pa) * other(gb, 1, pb)
    L[:, 2] = np.where(ga == gb, hwe(ga, ps), 0.0)
    return L


def _ll(L, k):
    t = L @ k
    if np.any(t <= 0):
        return -np.inf
    return np.log(t).sum()


def _edge(L, a, b):
    """max over k = t e_a + (1 - t) e_b, t in [0, 1] (concave in t)"""
    d = L[:, a] - L[:, b]

    def dl(t):
        tt = L[:, b] + t * d
        return np.sum(d / np.maximum(tt, 1e-300))

    def d2(t):
        tt = L[:, b] + t * d
        return -np.sum((d / np.maximum(tt, 1e-300)) ** 2)

    if dl(0.0) <= 0:
        t = 0.0
    elif dl(1.0) >= 0:
        t = 1.0
    else:
        lo, hi, t = 0.0, 1.0, 0.5
        for _ in range(60):
            g = dl(t)
            lo, hi = (t, hi) if g > 0 else (lo, t)
            tn = t - g / d2(t)
            if not (lo < tn < hi):
                tn = 0.5 * (lo + hi)
            if abs(tn - t) < 1e-10:
                t = tn
                break
            t = tn
    k = np.zeros(3)
    k[a], k[b] = t, 1 - t
    return k, _ll(L, k)


def ml_ibd(L):
    """(k0, k1, k2) maximising sum log(L k) on the simplex"""
    # interior: damped Newton in (k0, k2), k1 = 1 - k0 - k2
    x = np.array([1 / 3, 1 / 3])
    D = np.stack([L[:, 0] - L[:, 1], L[:, 2] - L[:, 1]], 1)
    inside = True
    k = np.array([x[0], 1 - x.sum(), x[1]])
    ll = _ll(L, k)
    for _ in range(100):
        t = L @ k
        w = D / t[:, None]
        g = w.sum(0)
        H = -(w.T @ w)
        step = -np.linalg.solve(H - 1e-12 * np.eye(2), g)
        s = 1.0
        while s > 1e-10:
            xn = x + s * step
            kn = np.array([xn[0], 1 - xn.sum(), xn[1]])
            if kn.min() >= 0:
                lln = _ll(L, kn)
                if lln >= ll:
                    break
            s *= 0.5
        if s <= 1e-10 or kn.min() < 1e-9:
            inside = kn.min() >= 1e-9 and s > 1e-10
            if not inside:
                break
        dx = np.abs(xn - x).sum()
        x, k, ll = xn, kn, lln
        if dx < 1e-10:
            break
    best = (k, ll) if inside and k.min() > 1e-9 else (None, -np.inf)
    # stationary inside only if the gradient vanishes there
    if best[0] is not None:
        t = L @ k
        if np.abs((D / t[:, None]).sum(0)).max() > 1e-6 * len(L):
            best = (None, -np.inf)
        else:
            return k  # concave: the interior stationary point is the maximum
    for a, b in ((0, 1), (0, 2), (1, 2)):
        kk, l2 = _edge(L, a, b)
        if l2 > best[1]:
            best = (kk, l2)
    return best[0]


# ---------------------------------------------------------------- pi
def families(pairs, N):
    """connected components of the pairs: list of member arrays, and the
    component of each individual"""
    root = list(range(N))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for i, j in pairs:
        root[find(i)] = find(j)
    comp = np.array([find(i) for i in range(N)])
    return comp


def pi_pca(G, U, rows, out=()):
    """IAF of the rows from the regression of g/2 on [U, 1], fitted without
    the individuals in out (PCAone: the pair's family). G: genotypes (int) or
    G / 2 (float)"""
    N = G.shape[0]
    V = np.hstack([U, np.ones((N, 1))])
    keep = np.setdiff1d(np.arange(N), np.asarray(out, int))
    Wr = np.zeros((len(rows), N))
    Wr[:, keep] = V[rows] @ np.linalg.pinv(V[keep])
    return Wr @ (G / 2.0 if G.dtype.kind in "iu" else G)


def smoother_lc(Lt, C):
    """W = Lt C^-1 (all rows)"""
    return Lt @ np.linalg.inv(C)


def pi_lin(Gh, Lt, C, rows, out=(), colsum=None):
    """best linear prediction of the rows' pi from the others' genotypes,
    without the individuals in out: pi_i = f + Lt_{i,K} C_{K,K}^-1 (g_K/2 - f),
    K = all but out (f: frequency over K). Gh = G / 2 (float), colsum its
    column sums (optional, for speed)"""
    N = Gh.shape[0]
    out = np.asarray(out, int)
    K = np.setdiff1d(np.arange(N), out)
    if colsum is None:
        colsum = Gh.sum(0)
    f = (colsum - Gh[out].sum(0)) / len(K) if len(out) else colsum / N
    Wr = np.zeros((len(rows), N))
    Wr[:, K] = Lt[np.ix_(rows, K)] @ np.linalg.inv(C[np.ix_(K, K)])
    return f + Wr @ Gh - Wr.sum(1)[:, None] * f


def pcp_parts(G, cand=None, tau=None):
    """(Lt, Cm): the PCP structure part (whitened by S, eigenvalues above 1
    kept) and the model covariance Lt + S, so that W = Lt Cm^-1 is the
    smoother of inbreed_smooth"""
    import inbreed_smooth as SM
    W, info = SM.smoother_pcp(G, cand=cand) if tau is None else SM.smoother_pcp(G, tau, cand)
    P = info["parts"]
    C, S, lam = P["C"], P["S"], P["lam"]
    e, Q = np.linalg.eigh(S)
    e = np.maximum(e, 1e-3 * np.median(np.diag(S)))
    Sh, Sih = (Q * np.sqrt(e)) @ Q.T, (Q / np.sqrt(e)) @ Q.T
    lam2, V = np.linalg.eigh(Sih @ C @ Sih)
    Lt = Sh @ (V * np.maximum(lam2 - 1, 0)) @ V.T @ Sh
    # model covariance Lt + S (S floored): W = Lt (Lt + S)^-1 has the weights
    # max(0, 1 - 1/lam); the centred GRM itself is singular (centring)
    Cm = Sh @ (V * (np.maximum(lam2 - 1, 0) + 1)) @ V.T @ Sh
    return Lt, Cm, info


def dwg_parts(G, cand=None):
    """(Lc, Lc + S) of the dwg fit (inbreed_smooth smoother_dwg); cand: the
    pairs that may enter S (default all)"""
    import dw_grm as DW
    import inbreed_smooth as SM
    A, f, w = SM.scaled_uncentred(G)
    N = len(A)
    if cand is None:
        cand = np.ones((N, N), bool)
    cand = cand | np.eye(N, dtype=bool)  # the diagonal (noise) is always free
    L, S, r, pairs = DW.fit(A, 2 ** -3.5, cand, DW.noise_edge(f, w, N), 40, None)
    S = (S + S.T) / 2
    # floor S's eigenvalues as pcp_parts (an MZ pair makes S singular)
    e, V = np.linalg.eigh(S)
    S = (V * np.maximum(e, 1e-3 * np.median(np.diag(S)))) @ V.T
    P = np.eye(N) - 1 / N
    Lc = P @ L @ P
    return Lc, Lc + S, dict(rank=r, pairs=len(pairs))


# ---------------------------------------------------------------- F and R
def inbreeding_from_pi(G, PI):
    """F = 1 - O/E per row, E = sum 2 pi (1 - pi) (PCAone --inbreed 2)"""
    PI = np.clip(PI, 1e-4, 1 - 1e-4)
    O = (G == 1).sum(1)
    E = (2 * PI * (1 - PI)).sum(1)
    return 1 - O / E


_C = []


def _cfit():
    """the C version of ibd_lik + ml_ibd (ibd_fit.c), compiled on first use"""
    if not _C:
        import ctypes
        d = os.path.dirname(os.path.abspath(__file__))
        so = os.path.join(d, "ibd_fit.so")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(os.path.join(d, "ibd_fit.c")):
            subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", so, os.path.join(d, "ibd_fit.c"), "-lm"],
                           check=True)
        lib = ctypes.CDLL(so)
        P = ctypes.POINTER
        lib.ibd_fit.argtypes = [P(ctypes.c_int8), P(ctypes.c_int8), P(ctypes.c_double), P(ctypes.c_double),
                                ctypes.c_long, P(ctypes.c_double)]
        _C.append((lib, ctypes))
    return _C[0]


def pair_rel(G, i, j, pa, pb, python=False):
    """kinship k1/4 + k2/2 and (k0, k1, k2) of a pair from its pi"""
    if python:
        k = ml_ibd(ibd_lik(G[i], G[j], pa, pb))
    else:
        lib, ct = _cfit()
        ga = np.ascontiguousarray(G[i], np.int8)
        gb = np.ascontiguousarray(G[j], np.int8)
        pa = np.ascontiguousarray(pa, np.float64)
        pb = np.ascontiguousarray(pb, np.float64)
        k = np.zeros(3)
        P = ct.POINTER
        lib.ibd_fit(ga.ctypes.data_as(P(ct.c_int8)), gb.ctypes.data_as(P(ct.c_int8)),
                    pa.ctypes.data_as(P(ct.c_double)), pb.ctypes.data_as(P(ct.c_double)), len(ga),
                    k.ctypes.data_as(P(ct.c_double)))
    return 0.25 * k[1] + 0.5 * k[2], k
