#!/usr/bin/env python3
"""Illustrate how close relatives distort a small-sample PCA, and test two fixes.

Data: n unrelated real individuals per population from a PLINK file. Relatives
are then created *from individuals in that sample* by copying genotypes (MZ
twins / duplicates) or alleles (children: one recombinant gamete from each
parent, random phase, 1 cM/Mb). Hence the PCA "as it should look" is known:
the PCA of the unrelated individuals alone, with every relative at its
expected position (an MZ copy on its source, a child at the parental mean).

Methods, all eigendecompositions of the N x N GRM C = X X' / M of the full
sample (X standardised with the sample allele frequencies), top k = K-1 PCs:
  standard   C as is
  zhou       matrix substitution (Zhou, Marron & Wright 2018, Biometrics):
             entries of related pairs replaced by the median off-diagonal entry
  masked_lr  entries of the *known* related pairs and the diagonal are treated
             as missing and imputed from the rank-k fit, iterated to convergence
  cs         Chen & Storey (2015) PCA as in van Waaij et al. (2023):
             H = G'G/M - D, D = diag(mean G(2-G)), uncentred and unscaled;
             eigenvectors 2..k+1 (the first is the mean)
  cs_masked_lr  masked_lr on H instead of the GRM: H is low rank in
             expectation including its diagonal, so only the entries of
             related pairs are imputed (rank k+1 fit)
  Robust PCA = principal component pursuit (PCP, Candes et al. 2011), M = L + S
  with L low rank and S sparse, by inexact ALM; it needs no list of related
  pairs. lambda = 1/sqrt(dim) unless --lam-mult is given:
  rpca_data  PCP of the standardised data matrix X (N x M)
  rsvd_grm   PCP of the GRM, diagonal as any other entry ("robust SVD")
  aarobust   PCP of the GRM with the diagonal unobserved (absorbed by S without
             penalty): AArobust (Albrechtsen, slides PCAforRelatedInd.pdf)
  rpca_cs    PCP of the Chen & Storey matrix H; PCs 2..k+1 of L
  Robust PCA with a kinship threshold tau (--tau) in place of lambda: an entry
  is sparse when its structure-adjusted kinship R_ij / (2 sqrt(v_i v_j)) > tau,
  R = M - L, v = noise variance (GRM: residual diagonal; CS: D):
  pcpkin_grm PCP (ALM) of the GRM, diagonal unobserved
  pcpkin_cs  PCP (ALM) of the double-centred Chen & Storey matrix J H J
  lrkin_grm  fixed rank k, raised stepwise (AltProj), GRM, diagonal unobserved
  lrkin_cs   fixed rank k+1, raised stepwise, Chen & Storey matrix
  New variants:
  heteropca_grm  HeteroPCA (Zhang, Cai & Wu 2022): GRM with its diagonal
             iteratively replaced by the diagonal of the rank-k fit
  heteropca_raw  HeteroPCA of the raw Gram G G'/M (rank k+1, first PC dropped):
             a model-free alternative to the CS diagonal correction
  lrkin_raw  fixed rank + kinship on the raw Gram with the diagonal free
             (HeteroPCA + kinship threshold; lrkin_grm is the same on the GRM)
  white_cs_ped / white_cs_king  whitening on the CS scale: raw genotypes
             whitened by Sigma = D^1/2 (I + 2 Psi) D^1/2 (D = heterozygosity),
             top k+1 eigenvectors, first dropped, mapped back by Sigma^1/2
  wt_ped / wt_king  weighted PCA, w_i = 1 / (1 + sum_j 2 phi_ij), scores
             mapped back by w^-1/2
  lrkin_cs_ea  lrkin_cs with the bias-corrected evalAdmix kinship
             (b - c)/2 of the current fit in the threshold
  white_auto standard PCA -> evalAdmix kinship -> GRM whitening, 3 rounds
  Versions of the shortlisted methods:
  hyb_cs_white   lrkin_cs detects the related pairs; their structure-adjusted
             kinship from the residual is used for CS whitening (no KING)
  auto_robust    robust start (lrkin_cs) -> evalAdmix kinship -> CS whitening,
             3 rounds
  white_gls_ped / white_gls_king  GRM whitening with GLS allele frequencies
             f = 1'S^-1 g / 1'S^-1 1, S = I + 2 Psi
  hyb_gls        lrkin_cs detects the pairs and their kinship; GRM whitening
             with GLS allele frequencies
  lrkin_cs_w     lrkin_cs, final eigenvectors of the completed matrix with
             effective-number weights w = 1/(1 + sum 2 phi)
  lrkin_cs_autoK lrkin_cs with the rank raised only while the next eigenvalue
             exceeds the noise edge 2 sigma sqrt(N) (automatic K)
  white_ped  kinship-whitened PCA, PCAone --kinship with the pedigree kinship
  white_king kinship-whitened PCA, PCAone --kinship with plink2 KING-robust
             estimates (pairs >= 2^-3.5)
Related pairs for zhou/masked_lr are taken from the known pedigree. The cs
methods are scored against the Chen & Storey PCA of the unrelated individuals.

Writes <out>/fig_*.pdf, <out>/table.tex and <out>/results.json.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import Dropper, read_bed, write_bed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"CEU": "#1b9e77", "CHB": "#d95f02", "MXL": "#7570b3", "YRI": "#66a61e"}
REL_COL = "#e7298a"


def standardize(G):
    f = G.mean(axis=0) / 2
    sd = np.sqrt(2 * f * (1 - f))
    return (G - 2 * f) / sd


def top_eig(C, k):
    w, v = np.linalg.eigh(C)
    return w[::-1][:k], v[:, ::-1][:, :k]


def zhou_ms(C, pairs):
    C = C.copy()
    off = C[~np.eye(len(C), dtype=bool)]
    med = np.median(off)
    for i, j in pairs:
        C[i, j] = C[j, i] = med
    return C


def cs_matrix(G):
    """Chen & Storey: G'G/M minus the binomial variance on the diagonal"""
    H = G @ G.T / G.shape[1]
    H[np.diag_indices_from(H)] -= (G * (2 - G)).mean(axis=1)
    return H


def cs_pcs(H, k):
    """top k structure PCs of a Chen & Storey matrix: drop the mean component"""
    w, v = top_eig(H, k + 1)
    return w[1:], v[:, 1:]


def masked_lr(C, pairs, k, iters=1000, tol=1e-10, mask_diag=True):
    C = C.copy()
    n = len(C)
    mask = np.eye(n, dtype=bool) if mask_diag else np.zeros((n, n), dtype=bool)
    for i, j in pairs:
        mask[i, j] = mask[j, i] = True
    off = C[~np.eye(n, dtype=bool)]
    C[mask & ~np.eye(n, dtype=bool)] = np.median(off)  # start from "unrelated"
    for it in range(iters):
        w, v = top_eig(C, k)
        L = (v * w) @ v.T
        d = np.max(np.abs(C[mask] - L[mask]))
        C[mask] = L[mask]
        if d < tol:
            break
    return C, it + 1


def rpca(M, lam=None, tol=1e-7, maxit=1000, free_diag=False):
    """principal component pursuit, M = L + S, by inexact ALM (Lin, Chen & Ma 2010).

    free_diag: the diagonal is unobserved (PCP with missing entries): it is not
    used to fit L and goes into S without an L1 penalty. This is AArobust.
    A symmetric M is kept symmetric by thresholding its eigenvalues."""
    sym = M.shape[0] == M.shape[1] and np.allclose(M, M.T)
    lam = 1 / np.sqrt(max(M.shape)) if lam is None else lam
    free = np.eye(len(M), dtype=bool) if free_diag else np.zeros(M.shape, dtype=bool)
    Mo = np.where(free, 0, M)
    norm2 = np.linalg.norm(Mo, 2)
    Y = Mo / max(norm2, np.abs(Mo).max() / lam)
    mu, rho = 1.25 / norm2, 1.5
    mu_bar = mu * 1e7
    S = np.zeros_like(M)
    nM = np.linalg.norm(Mo)
    for it in range(maxit):
        A = M - S + Y / mu
        if sym:
            w, v = np.linalg.eigh((A + A.T) / 2)
            w = np.sign(w) * np.maximum(np.abs(w) - 1 / mu, 0)
            L = (v * w) @ v.T
        else:
            U, d, Vt = np.linalg.svd(A, full_matrices=False)
            L = (U * np.maximum(d - 1 / mu, 0)) @ Vt
        B = M - L + Y / mu
        S = np.sign(B) * np.maximum(np.abs(B) - lam / mu, 0)
        S[free] = (M - L)[free]  # unobserved: absorbed exactly, no penalty
        Z = M - L - S
        Y += mu * Z
        mu = min(mu * rho, mu_bar)
        if np.linalg.norm(Z) / nM < tol:
            break
    return L, S, it + 1


def kin_scale(R, noise):
    """structure-adjusted kinship of every pair from a residual R = M - L:
    phi_ij = R_ij / (2 sqrt(v_i v_j)), v = per-individual noise variance on the
    scale of M (GRM: residual diagonal; Chen & Storey: D)"""
    v = np.maximum(noise, 1e-12)
    return R / (2 * np.sqrt(np.outer(v, v)))


def kin_select(R, noise, tau):
    """off-diagonal entries whose kinship exceeds tau"""
    sel = kin_scale(R, noise) > tau
    np.fill_diagonal(sel, False)
    return sel


def pcp_kin(M, tau, free_diag, D=None, tol=1e-7, maxit=1000, cand=None):
    """PCP (inexact ALM) where the sparse step is a kinship threshold instead of
    an l1 shrinkage: S_ij = residual if its kinship > tau, else 0 (no lambda).

    free_diag (GRM): the diagonal is unobserved and absorbed by S; the noise
    variance is the residual diagonal. Otherwise (Chen & Storey) the diagonal
    stays in L and the noise variance is D."""
    n = len(M)
    eye = np.eye(n, dtype=bool)
    Mo = np.where(eye, 0, M) if free_diag else M
    norm2 = np.linalg.norm(Mo, 2)
    Y = Mo / max(norm2, np.abs(Mo).max() * np.sqrt(n))
    mu, rho = 1.25 / norm2, 1.5
    mu_bar = mu * 1e7
    S = np.zeros_like(M)
    nM = np.linalg.norm(Mo)
    for it in range(maxit):
        A = M - S + Y / mu
        w, v = np.linalg.eigh((A + A.T) / 2)
        w = np.sign(w) * np.maximum(np.abs(w) - 1 / mu, 0)
        L = (v * w) @ v.T
        B = M - L + Y / mu
        noise = np.diag(M - L) if free_diag else D
        sel = kin_select(B, noise, tau)
        if cand is not None:  # only screened candidate pairs may enter S
            sel &= cand
        S = np.where(sel, B, 0)
        if free_diag:
            S[eye] = (M - L)[eye]
        Z = M - L - S
        Y += mu * Z
        mu = min(mu * rho, mu_bar)
        if np.linalg.norm(Z) / nM < tol:
            break
    return L, S, it + 1


def lr_kin(M, rank, tau, free_diag, D=None, iters=500, cand=None):
    """fixed-rank alternating projections with a kinship threshold (AltProj
    style, Netrapalli et al. 2014): the rank is raised one step at a time, and
    at each rank L = rank-r fit of M - S and S = residuals of pairs whose
    kinship > tau (plus the diagonal if free_diag) are alternated to
    convergence. Raising the rank stepwise lets a family be detected before the
    rank is large enough for it to be absorbed by L."""
    n = len(M)
    eye = np.eye(n, dtype=bool)
    S = np.where(eye, M, 0) if free_diag else np.zeros_like(M)
    total = 0
    for r in range(1, rank + 1):
        L_old = None
        for it in range(iters):
            w, v = top_eig(M - S, r)
            L = (v * w) @ v.T
            R = M - L
            noise = np.diag(R) if free_diag else D
            sel = kin_select(R, noise, tau)
            if cand is not None:  # only screened candidate pairs may enter S
                sel &= cand
            S = np.where(sel, R, 0)
            if free_diag:
                S[eye] = R[eye]
            total += 1
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
    return L, S, total


def lr_kin_fd(H, rank, tau, D, cand=None, iters=500):
    """fixed rank + kinship on the CS matrix with the diagonal free: the
    diagonal is imputed from the rank-r fit (as in AArobust / HeteroPCA)
    instead of trusting D = mean g(2-g), which assumes HWE within individuals.
    Off-diagonal entries are scaled to kinship with D as before."""
    n = len(H)
    eye = np.eye(n, dtype=bool)
    S = np.zeros_like(H)
    total = 0
    for r in range(1, rank + 1):
        L_old = None
        for it in range(iters):
            w, v = top_eig(H - S, r)
            L = (v * w) @ v.T
            R = H - L
            sel = kin_select(R, D, tau)
            if cand is not None:
                sel &= cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            total += 1
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
    return L, S, total


def lr_kin_fd_auto(H, tau, D, edge, cand=None, rmax=50, iters=500):
    """lr_kin_fd with the rank chosen from the data instead of -k: the rank is
    raised while the next eigenvalue of H - S (after the fit at the current
    rank) exceeds the noise edge. Returns L, S, total iterations and the rank."""
    n = len(H)
    eye = np.eye(n, dtype=bool)
    S = np.zeros_like(H)
    total, r = 0, 1
    while True:
        L_old = None
        for it in range(iters):
            w, v = top_eig(H - S, r)
            L = (v * w) @ v.T
            R = H - L
            sel = kin_select(R, D, tau)
            if cand is not None:
                sel &= cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            total += 1
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        if r >= min(rmax, n - 1) or top_eig(H - S, r + 1)[0][r] <= edge:
            break
        r += 1
    return L, S, total, r


def whitened_noise(AM, noise, pairs_phi, k, floor=0.1):
    """whitening of the raw Gram AM = G G'/M with a given per-individual noise
    variance: Sigma = v^1/2 (I + 2 Psi) v^1/2 (CS whitening uses v = D)"""
    N = len(AM)
    d = np.sqrt(np.maximum(noise, 1e-12))
    R_ = np.eye(N)
    for i, j, phi in pairs_phi:
        R_[i, j] = R_[j, i] = 2 * phi
    Sig = d[:, None] * R_ * d[None, :]
    w, v = np.linalg.eigh(Sig)
    w = np.maximum(w, floor * d.min() ** 2)
    Wm, Wh = (v / np.sqrt(w)) @ v.T, (v * np.sqrt(w)) @ v.T
    _, U = top_eig(Wm @ AM @ Wm, k + 1)
    return (Wh @ U)[:, 1:]


def double_centre(M):
    J = np.eye(len(M)) - 1.0 / len(M)
    return J @ M @ J


def detected(S, pairs):
    """true and false positive pairs among the nonzero off-diagonal S"""
    n = len(S)
    iu = np.triu_indices(n, 1)
    found = {(int(i), int(j)) for i, j, x in zip(iu[0], iu[1], S[iu]) if abs(x) > 1e-12}
    return len(found & set(pairs)), len(found - set(pairs))


def sparse_hits(S, pairs, n_true):
    """how many of the largest off-diagonal |S| entries are true related pairs"""
    n = len(S)
    iu = np.triu_indices(n, 1)
    order = np.argsort(-np.abs(S[iu]))
    top = {(int(iu[0][o]), int(iu[1][o])) for o in order[:n_true]}
    nz = int((np.abs(S[iu]) > 1e-8).sum())
    return len(top & set(pairs)), nz


def align(S, T, rows):
    """flip signs of S's columns to agree with T on rows (display only)"""
    sg = np.sign(np.sum(S[rows] * T[rows], axis=0))
    sg[sg == 0] = 1
    return S * sg


def scores_vs_truth(S, T, base, rel):
    """R2 of each truth PC regressed on all k method PCs (+1) over the unrelated
    individuals, and the error of the relatives after the same linear map,
    relative to the RMS spread of the unrelated individuals."""
    Z = np.c_[S, np.ones(len(S))]
    B, *_ = np.linalg.lstsq(Z[base], T[base], rcond=None)
    P = Z @ B
    r2 = 1 - ((T[base] - P[base]) ** 2).sum(0) / ((T[base] - T[base].mean(0)) ** 2).sum(0)
    spread = np.sqrt(((T[base] - T[base].mean(0)) ** 2).sum(1).mean())
    err = np.sqrt(((T[rel] - P[rel]) ** 2).sum(1).mean()) / spread if len(rel) else np.nan
    return r2, err


def make_relatives(kind, G, labels, rng, drop):
    """relatives built from individuals already in the sample.

    kind = mz_<n>          n identical copies of the first CEU individual
           child_<pop>_<n> n children of two <pop> individuals in the sample
    Returns genotypes (r x M), the parents of each relative (indices into the
    sample; an MZ copy has its source as both) and a description."""
    pops = np.array(labels)
    t = kind.split("_")
    newG, parents = [], []
    if t[0] == "mz":
        n = int(t[1])
        src = int(np.where(pops == "CEU")[0][0])
        for _ in range(n):
            newG.append(G[src].copy())
            parents.append((src, src))
        desc = f"{n} MZ cop{'y' if n == 1 else 'ies'} of a CEU"
    elif t[0] == "child":
        pop, n = t[1], int(t[2])
        a, b = (int(x) for x in np.where(pops == pop)[0][1:3])
        ha, hb = drop.phase(G[a]), drop.phase(G[b])
        for _ in range(n):
            newG.append((drop.gamete(ha) + drop.gamete(hb)).astype(G.dtype))
            parents.append((a, b))
        desc = f"{n} child{'' if n == 1 else 'ren'} of two {pop}"
    else:
        sys.exit("unknown kind " + kind)
    return np.array(newG), parents, desc


def related_pairs(n0, parents):
    """related pairs: each relative with its parent(s) / MZ source, and the
    relatives with each other (full sibs or MZ copies of the same source)"""
    pairs = set()
    for i, (a, b) in enumerate(parents):
        x = n0 + i
        pairs.update({(a, x), (b, x)})
        pairs.update((n0 + j, x) for j in range(i))
    return sorted(pairs)


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed: {cmd}\n{r.stdout}\n{r.stderr}")


def whitened_numpy(X, pairs_phi, k, floor=0.1):
    """reference implementation of PCAone --kinship (dense, for checking)"""
    N, M = X.shape
    Sig = np.eye(N)
    for i, j, phi in pairs_phi:
        Sig[i, j] = Sig[j, i] = 2 * phi
    w, v = np.linalg.eigh(Sig)
    w = np.maximum(w, floor)
    Xw = (v / np.sqrt(w)) @ v.T @ X
    _, U = top_eig(Xw @ Xw.T / M, k)
    return (v * np.sqrt(w)) @ v.T @ U


def cov2cor(A):
    s_ = np.sqrt(np.maximum(np.diag(A), 1e-300))
    return A / np.outer(s_, s_)


def evaladmix_kin(G, U, intercept):
    """evalAdmix kinship (b - c)/2 given scores U (N x r) of a PCA of G
    (individuals in rows, 0/1/2): b = correlation of residuals G(I-P),
    c = the correlation the fit induces, cov2cor((I-P) D (I-P))."""
    N, M = G.shape
    V = np.c_[U, np.ones(N)] if intercept else U
    P = V @ np.linalg.pinv(V.T @ V) @ V.T
    IP = np.eye(N) - P
    gbar = G.mean(axis=1)
    A = G @ G.T
    b = cov2cor(IP @ (A - M * np.outer(gbar, gbar)) @ IP)
    c = cov2cor(IP @ np.diag((G * (2 - G)).mean(axis=1)) @ IP)
    kin = (b - c) / 2
    np.fill_diagonal(kin, 0)
    return kin


def lr_kin_ea(H, G, rank, tau, iters=500):
    """lrkin_cs with the threshold applied to the evalAdmix kinship of the
    current rank-r fit (projection on its eigenvectors, as PCA 1 in evalPCA.R)"""
    S = np.zeros_like(H)
    total = 0
    for r in range(1, rank + 1):
        L_old = None
        for it in range(iters):
            w, v = top_eig(H - S, r)
            L = (v * w) @ v.T
            sel = evaladmix_kin(G, v, intercept=False) > tau
            np.fill_diagonal(sel, False)
            S = np.where(sel, H - L, 0)
            total += 1
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
    return L, S, total


def whitened_cs(G, pairs_phi, k, floor=0.1):
    """whitening on the CS scale: Sigma = D^1/2 (I + 2 Psi) D^1/2"""
    N, M = G.shape
    d = np.sqrt((G * (2 - G)).mean(axis=1))
    R_ = np.eye(N)
    for i, j, phi in pairs_phi:
        R_[i, j] = R_[j, i] = 2 * phi
    Sig = d[:, None] * R_ * d[None, :]
    w, v = np.linalg.eigh(Sig)
    w = np.maximum(w, floor * d.min() ** 2)
    Xw = (v / np.sqrt(w)) @ v.T @ G
    _, U = top_eig(Xw @ Xw.T / M, k + 1)
    return ((v * np.sqrt(w)) @ v.T @ U)[:, 1:]


def weighted_pca(X, pairs_phi, k):
    """weighted PCA: w_i = 1 / (1 + sum_j 2 phi_ij)"""
    N, M = X.shape
    eff = np.ones(N)
    for i, j, phi in pairs_phi:
        eff[i] += 2 * phi
        eff[j] += 2 * phi
    sw = 1 / np.sqrt(eff)
    Xw = X * sw[:, None]
    _, U = top_eig(Xw @ Xw.T / M, k)
    return U / sw[:, None]


def white_auto(G, X, k, tau, rounds=3):
    """standard PCA -> evalAdmix kinship -> GRM whitening, iterated"""
    M = X.shape[1]
    _, U = top_eig(X @ X.T / M, k)
    for _ in range(rounds):
        kin = evaladmix_kin(G, U, intercept=True)
        iu = np.triu_indices(len(G), 1)
        pp = [(int(i), int(j), float(kin[i, j])) for i, j in zip(*iu) if kin[i, j] > tau]
        U = whitened_numpy(X, pp, k)
    return U, pp


def lr_kin_pairs(H, D, L, S):
    """related pairs and structure-adjusted kinship from a lrkin fit on CS"""
    phi = kin_scale(H - L, D)
    iu = np.triu_indices(len(H), 1)
    return [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]


def hyb_cs_white(G, H, D, k, tau):
    L, S, _ = lr_kin(H, k + 1, tau, False, D)
    pp = lr_kin_pairs(H, D, L, S)
    return whitened_cs(G, pp, k), pp


def auto_robust(G, H, D, k, tau, rounds=3):
    L, _, _ = lr_kin(H, k + 1, tau, False, D)
    U = cs_pcs(L, k)[1]
    for _ in range(rounds):
        kin = evaladmix_kin(G, U, intercept=True)
        iu = np.triu_indices(len(G), 1)
        pp = [(int(i), int(j), float(kin[i, j])) for i, j in zip(*iu) if kin[i, j] > tau]
        U = whitened_cs(G, pp, k)
    return U, pp


def whitened_gls(G, pairs_phi, k, floor=0.1):
    """GRM whitening with GLS allele frequencies"""
    N, M = G.shape
    Sig = np.eye(N)
    for i, j, phi in pairs_phi:
        Sig[i, j] = Sig[j, i] = 2 * phi
    w, v = np.linalg.eigh(Sig)
    w = np.maximum(w, floor)
    Sinv1 = (v / w) @ v.T @ np.ones(N)
    f = (Sinv1 @ G) / Sinv1.sum() / 2
    f = np.clip(f, 1e-6, 1 - 1e-6)
    X = (G - 2 * f) / np.sqrt(2 * f * (1 - f))
    Xw = (v / np.sqrt(w)) @ v.T @ X
    _, U = top_eig(Xw @ Xw.T / M, k)
    return (v * np.sqrt(w)) @ v.T @ U


def lrkin_cs_weighted(H, D, k, tau):
    L, S, _ = lr_kin(H, k + 1, tau, False, D)
    pp = lr_kin_pairs(H, D, L, S)
    eff = np.ones(len(H))
    for i, j, phi in pp:
        eff[i] += 2 * phi
        eff[j] += 2 * phi
    sw = 1 / np.sqrt(eff)
    Hc = H - S  # relatives' entries replaced by the low-rank fit
    _, U = cs_pcs(sw[:, None] * Hc * sw[None, :], k)
    return U / sw[:, None]


def cs_noise_edge(G):
    """2 sigma sqrt(N): spectral edge of the noise in the CS matrix, with
    sigma^2 = Var(H_ij) for an unrelated pair under the pooled frequencies"""
    N, M = G.shape
    f = G.mean(axis=0) / 2
    eg2 = 2 * f * (1 - f) + 4 * f ** 2
    sigma2 = np.sum(eg2 ** 2 - 16 * f ** 4) / M ** 2
    return 2 * np.sqrt(sigma2) * np.sqrt(N)


def lrkin_cs_autoK(G, H, D, tau, rmax=10, iters=500):
    """lrkin_cs where the rank is raised only while the next eigenvalue of
    H - S exceeds the noise edge; returns structure PCs and the chosen K"""
    n = len(H)
    edge = cs_noise_edge(G)
    S = np.zeros_like(H)
    r = 1
    while True:
        L_old = None
        for it in range(iters):
            w, v = top_eig(H - S, r)
            L = (v * w) @ v.T
            S = np.where(kin_select(H - L, D, tau), H - L, 0)
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        nxt = top_eig(H - S, r + 1)[0][r]
        if nxt <= edge or r >= rmax:
            break
        r += 1
    return v[:, 1:], r, edge


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pops", default="CEU,CHB,MXL,YRI")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, default=3, help="number of PCs, K-1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--tau", type=float, default=2 ** -3.5, help="kinship threshold for the *kin methods")
    ap.add_argument("--tau-sweep", default="0.03,0.05,0.07,0.09,0.12,0.2,0.3")
    ap.add_argument("--lam-mult", type=float, default=1.0, help="PCP lambda = lam_mult / sqrt(dim)")
    ap.add_argument("--lam-sweep", default="0.5,0.75,1,1.5,2,3,4", help="lambda multipliers for the sweep table")
    ap.add_argument("--scenarios", default="mz_1,mz_2,child_CEU_1,child_MXL_1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    fam, bim, Gall = read_bed(a.bfile)
    pops = a.pops.split(",")
    lab = np.array([r[1] for r in fam])
    idx = np.concatenate([rng.choice(np.where(lab == p)[0], a.n, replace=False) for p in pops])
    G0 = Gall[idx].astype(float)
    labels = list(lab[idx])
    poly = G0.std(0) > 0
    G0, bim = G0[:, poly], [l for l, k in zip(bim, poly) if k]
    # independent truth: reference panel of all individuals of these populations
    # that are not in the sample, everyone in the data projected onto its PCA
    ref_idx = np.setdiff1d(np.where(np.isin(lab, pops))[0], idx)
    Gref0 = Gall[ref_idx][:, poly].astype(float)
    fr0 = Gref0.mean(0) / 2
    okr = (fr0 > 0) & (fr0 < 1)
    sdr = np.sqrt(2 * fr0[okr] * (1 - fr0[okr]))
    Xr0 = (Gref0[:, okr] - 2 * fr0[okr]) / sdr
    Vr0 = Xr0.T @ top_eig(Xr0 @ Xr0.T / okr.sum(), a.k)[1]

    def ref_truth(Gm):
        T_ = ((Gm[:, okr] - 2 * fr0[okr]) / sdr) @ Vr0
        return T_ / np.linalg.norm(T_, axis=0)  # unit-length columns, as eigenvectors
    n0, k = len(G0), a.k
    drop = Dropper(bim, rng)

    # the PCA as it should look: unrelated individuals only
    X0 = standardize(G0)
    w0, T0 = top_eig(X0 @ X0.T / X0.shape[1], k)
    print(f"{n0} unrelated individuals, {G0.shape[1]} sites; truth eigenvalues {np.round(w0, 2)}")
    wcs0, Tcs0 = cs_pcs(cs_matrix(G0), k)
    r2_truths, _ = scores_vs_truth(Tcs0, T0, np.arange(n0), [])
    print(f"CS truth eigenvalues {np.round(wcs0, 3)}; R2 of standard truth PCs on CS truth PCs {np.round(r2_truths, 4)}")

    results = {"n_unrel": n0, "sites": int(G0.shape[1]), "truth_eig": w0.tolist(), "truth_cs_eig": wcs0.tolist(),
               "r2_std_truth_on_cs_truth": r2_truths.tolist(), "scen": {}}
    methods = ["standard", "zhou", "masked_lr", "white_ped", "white_king", "cs", "cs_masked_lr",
               "rpca_data", "rsvd_grm", "aarobust", "rpca_cs"]
    methods += ["pcpkin_grm", "pcpkin_cs", "lrkin_grm", "lrkin_cs"]
    new_methods = ["heteropca_grm", "heteropca_raw", "lrkin_raw", "white_cs_ped", "white_cs_king",
                   "wt_ped", "wt_king", "lrkin_cs_ea", "white_auto"]
    methods += new_methods
    v2_methods = ["hyb_cs_white", "auto_robust", "white_gls_ped", "white_gls_king", "hyb_gls",
                  "lrkin_cs_w", "lrkin_cs_autoK"]
    methods += v2_methods
    v3_methods = ["aarobust_kin", "lrkin_cs_kc", "hyb_cs_kc"]
    methods += v3_methods
    uses_cs_truth = {"cs", "cs_masked_lr", "rpca_cs", "lrkin_cs", "pcpkin_cs",
                     "heteropca_raw", "lrkin_raw", "white_cs_ped", "white_cs_king", "lrkin_cs_ea",
                     "hyb_cs_white", "auto_robust", "lrkin_cs_w", "lrkin_cs_autoK", "lrkin_cs_kc", "hyb_cs_kc"}
    ids0 = [fam[i][0] for i in idx]
    panels, sweep_inputs, kin_inputs = {}, {}, {}
    for scen in a.scenarios.split(","):
        R, parents, desc = make_relatives(scen, G0, labels, rng, drop)
        G = np.vstack([G0, R])
        N = len(G)
        rel = np.arange(n0, N)
        base = np.arange(n0)
        # expected position of each relative in the truth PCA
        T = np.vstack([T0, np.array([(T0[p] + T0[q]) / 2 for p, q in parents])])
        Tcs = np.vstack([Tcs0, np.array([(Tcs0[p] + Tcs0[q]) / 2 for p, q in parents])])
        pairs = related_pairs(n0, parents)
        X = standardize(G)
        X = X[:, np.isfinite(X).all(0)]
        C = X @ X.T / X.shape[1]
        w_std, _ = top_eig(C, 8)
        out = {"desc": desc, "N": N, "pairs": len(pairs), "eig_standard": w_std.tolist(), "m": {}}
        S = {}
        _, S["standard"] = top_eig(C, k)
        _, S["zhou"] = top_eig(zhou_ms(C, pairs), k)
        Ca, nit = masked_lr(C, pairs, k)
        _, S["masked_lr"] = top_eig(Ca, k)
        out["masked_lr_iters"] = nit
        H = cs_matrix(G)
        _, S["cs"] = cs_pcs(H, k)
        Hr, nit = masked_lr(H, pairs, k + 1, mask_diag=False)
        _, S["cs_masked_lr"] = cs_pcs(Hr, k)
        out["cs_masked_lr_iters"] = nit

        # standard robust PCA (PCP): data matrix, GRM, GRM with free diagonal, CS matrix
        lm = a.lam_mult
        Ld, _, it_d = rpca(X, lm / np.sqrt(max(X.shape)))
        S["rpca_data"] = np.linalg.svd(Ld, full_matrices=False)[0][:, :k]
        Lg, Sg, it_g = rpca(C, lm / np.sqrt(N))
        _, S["rsvd_grm"] = top_eig(Lg, k)
        La, Sa, it_a = rpca(C, lm / np.sqrt(N), free_diag=True)
        _, S["aarobust"] = top_eig(La, k)
        Lc, Sc, it_c = rpca(H, lm / np.sqrt(N))
        _, S["rpca_cs"] = cs_pcs(Lc, k)
        # kinship-thresholded robust PCA
        Dh = (G * (2 - G)).mean(axis=1)
        kin_runs = {"pcpkin_grm": pcp_kin(C, a.tau, True),
                    "pcpkin_cs": pcp_kin(double_centre(H), a.tau, False, Dh),
                    "lrkin_grm": lr_kin(C, k, a.tau, True),
                    "lrkin_cs": lr_kin(H, k + 1, a.tau, False, Dh)}
        out["kin"] = {}
        for name, (L_, S_, it_) in kin_runs.items():
            S[name] = cs_pcs(L_, k)[1] if name == "lrkin_cs" else top_eig(L_, k)[1]
            tp, fp = detected(S_, pairs)
            out["kin"][name] = {"iters": it_, "tp": tp, "fp": fp, "n_pairs": len(pairs)}
        kin_inputs[scen] = (C, H, Dh, pairs)
        out["rpca"] = {}
        for name, L_, S_, it_ in [("data", Ld, None, it_d), ("rsvd_grm", Lg, Sg, it_g), ("aarobust", La, Sa, it_a),
                                  ("cs", Lc, Sc, it_c)]:
            rk = int((np.linalg.svd(L_, compute_uv=False) > 1e-6 * np.linalg.norm(L_, 2)).sum())
            info = {"iters": it_, "rank_L": rk}
            if S_ is not None:
                hits, nz = sparse_hits(S_, pairs, len(pairs))
                info.update({"top_sparse_are_relatives": hits, "n_pairs": len(pairs), "nonzero_offdiag_S": nz,
                             "diag_S_mean": float(np.diag(S_).mean())})
            out["rpca"][name] = info

        # kinship-whitened PCA through the PCAone binary
        d = os.path.join(a.out, "work", scen)
        os.makedirs(d, exist_ok=True)
        ids = ids0 + [f"REL{i + 1}" for i in range(len(R))]
        write_bed(f"{d}/test", [["F", x, "0", "0", "0", "-9"] for x in ids], bim, G.astype(np.int8))
        phi = 0.5 if scen.startswith("mz") else 0.25
        with open(f"{d}/pedigree.kin0", "w") as fk:
            fk.write("ID1\tID2\tKINSHIP\n")
            fk.writelines(f"{ids[i]}\t{ids[j]}\t{phi}\n" for i, j in pairs)
        pcaone = f"{a.pcaone} -b {d}/test -k {k} -d 3 -n 4 -v 0"
        run(f"{pcaone} --kinship {d}/pedigree.kin0 -o {d}/white_ped")
        S["white_ped"] = np.loadtxt(f"{d}/white_ped.eigvecs", ndmin=2)
        run(f"{a.plink2} --bfile {d}/test --make-king-table --king-table-filter 0.04 --threads 4 --out {d}/king")
        run(f"{pcaone} --kinship {d}/king.kin0 -o {d}/white_king")
        S["white_king"] = np.loadtxt(f"{d}/white_king.eigvecs", ndmin=2)
        king = [l.split() for l in open(f"{d}/king.kin0").read().strip().split("\n")[1:]]
        out["king"] = [(l[1], l[3], float(l[-1])) for l in king]
        ref = whitened_numpy(X, [(i, j, phi) for i, j in pairs], k)
        cos = np.abs(np.sum(ref * S["white_ped"], 0)) / (np.linalg.norm(ref, axis=0) *
                                                         np.linalg.norm(S["white_ped"], axis=0))
        out["check_white_vs_numpy"] = float(np.max(np.abs(1 - cos)))

        # new variants
        pos = {x: i for i, x in enumerate(ids)}
        ped_phi = [(i, j, phi) for i, j in pairs]
        king_phi = [(pos[x], pos[y], f) for x, y, f in out["king"] if f >= a.tau]
        Graw = G @ G.T / G.shape[1]
        _, S["heteropca_grm"] = top_eig(masked_lr(C, [], k)[0], k)
        S["heteropca_raw"] = cs_pcs(masked_lr(Graw, [], k + 1)[0], k)[1]
        Lr, Sr, _ = lr_kin(Graw, k + 1, a.tau, True)
        S["lrkin_raw"] = cs_pcs(Lr, k)[1]
        S["white_cs_ped"] = whitened_cs(G, ped_phi, k)
        S["white_cs_king"] = whitened_cs(G, king_phi, k)
        S["wt_ped"] = weighted_pca(X, ped_phi, k)
        S["wt_king"] = weighted_pca(X, king_phi, k)
        Le, Se, _ = lr_kin_ea(H, G, k + 1, a.tau)
        S["lrkin_cs_ea"] = cs_pcs(Le, k)[1]
        S["white_auto"], auto_pairs = white_auto(G, X, k, a.tau)
        S["hyb_cs_white"], hyb_pairs = hyb_cs_white(G, H, Dh, k, a.tau)
        S["auto_robust"], ar_pairs = auto_robust(G, H, Dh, k, a.tau)
        S["white_gls_ped"] = whitened_gls(G, ped_phi, k)
        S["white_gls_king"] = whitened_gls(G, king_phi, k)
        S["hyb_gls"] = whitened_gls(G, hyb_pairs, k)
        S["lrkin_cs_w"] = lrkin_cs_weighted(H, Dh, k, a.tau)
        Uak, Kauto, edge = lrkin_cs_autoK(G, H, Dh, a.tau)
        S["lrkin_cs_autoK"] = Uak
        out["autoK"] = {"K": Kauto, "edge": edge, "eig_H": top_eig(H, 8)[0].tolist()}
        out["hyb_phi"] = [(i, j, round(f, 3)) for i, j, f in hyb_pairs]
        out["auto_robust_phi"] = [(i, j, round(f, 3)) for i, j, f in ar_pairs]
        out["kin"]["lrkin_raw"] = dict(zip(("tp", "fp"), detected(Sr, pairs)))
        out["kin"]["lrkin_cs_ea"] = dict(zip(("tp", "fp"), detected(Se, pairs)))
        ap_ = {(min(i, j), max(i, j)) for i, j, _ in auto_pairs}
        out["kin"]["white_auto"] = {"tp": len(ap_ & set(pairs)), "fp": len(ap_ - set(pairs)),
                                    "phi": [round(f, 3) for _, _, f in auto_pairs]}
        # KING-screened variants (candidates: plink2 KING > 0.04, the --king-table-filter)
        cand = np.zeros((N, N), dtype=bool)
        for x, y, f_ in out["king"]:
            cand[pos[x], pos[y]] = cand[pos[y], pos[x]] = True
        _, S["aarobust_kin"] = top_eig(pcp_kin(C, a.tau, True, cand=cand)[0], k)
        Lk, Sk, _ = lr_kin(H, k + 1, a.tau, False, Dh, cand=cand)
        S["lrkin_cs_kc"] = cs_pcs(Lk, k)[1]
        S["hyb_cs_kc"] = whitened_cs(G, lr_kin_pairs(H, Dh, Lk, Sk), k)
        Tref = ref_truth(G)
        for m in methods:
            Tm = Tcs if m in uses_cs_truth else T
            S[m] = align(S[m], Tm, base)
            r2, err = scores_vs_truth(S[m], Tm, base, rel)
            r2r, errr = scores_vs_truth(S[m], Tref, base, rel)
            out["m"][m] = {"r2": r2.tolist(), "err_rel": err, "r2_ref": r2r.tolist(), "err_ref": errr}
        results["scen"][scen] = out
        S["truth"], S["truth_cs"] = T, Tcs
        sweep_inputs[scen] = (C, H, T, Tcs, base, rel, pairs)
        panels[scen] = (S, N)
        print(f"{scen}: {desc}; N={N}; standard eigenvalues {np.round(w_std[:6], 2)}; "
              f"PCAone vs numpy whitening |1-cos| = {out['check_white_vs_numpy']:.1e}; KING pairs {out['king']}")
        print("   robust PCA:", out["rpca"])
        print("   detection:", out["kin"])
        for m in methods:
            print(f"   {m:11s} R2 per PC {np.round(out['m'][m]['r2'], 3)}  relatives err {out['m'][m]['err_rel']:.3f}")

    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    # figures: one per pair of PCs, rows = scenarios, cols = truth + methods
    titles = {"truth": "unrelated only (truth)", "standard": "standard PCA",
              "zhou": "Zhou matrix substitution", "masked_lr": "masked low-rank imputation",
              "white_ped": "whitened, pedigree kinship", "white_king": "whitened, KING kinship",
              "truth_cs": "CS, unrelated only (truth)", "cs": "Chen & Storey (CS)",
              "cs_masked_lr": "masked low-rank on CS matrix", "rpca_data": "robust PCA of data matrix",
              "rsvd_grm": "robust SVD of GRM", "aarobust": "AArobust (GRM, free diagonal)",
              "rpca_cs": "robust PCA of CS matrix",
              "pcpkin_grm": "PCP+kinship, GRM", "pcpkin_cs": "PCP+kinship, centred CS",
              "lrkin_grm": "fixed rank+kinship, GRM", "lrkin_cs": "fixed rank+kinship, CS",
              "heteropca_grm": "HeteroPCA, GRM", "heteropca_raw": "HeteroPCA, raw Gram",
              "lrkin_raw": "HeteroPCA+kinship, raw Gram", "white_cs_ped": "CS whitening, pedigree",
              "white_cs_king": "CS whitening, KING", "wt_ped": "weighted, pedigree", "wt_king": "weighted, KING",
              "lrkin_cs_ea": "fixed rank+evalAdmix kin, CS", "white_auto": "whitening, evalAdmix (auto)",
              "hyb_cs_white": "detect (lrkin CS) + CS whitening", "auto_robust": "robust start + evalAdmix + CS whit.",
              "white_gls_ped": "whitening, GLS freq., pedigree", "white_gls_king": "whitening, GLS freq., KING",
              "hyb_gls": "detect (lrkin CS) + whitening GLS", "lrkin_cs_w": "lrkin CS, weighted final fit",
              "aarobust_kin": "AArobust, kinship thr. + screen", "lrkin_cs_kc": "fixed rank+kinship CS, screen",
              "hyb_cs_kc": "detect (screen) + CS whitening", "lrkin_cs_autoK": "lrkin CS, automatic K"}
    figsets = {"": ["truth", "standard", "zhou", "masked_lr", "white_ped", "white_king"],
               "_cs": ["truth_cs", "cs", "masked_lr", "cs_masked_lr"],
               "_rpca": ["truth", "rpca_data", "rsvd_grm", "aarobust", "truth_cs", "rpca_cs"],
               "_kin": ["truth", "pcpkin_grm", "pcpkin_cs", "lrkin_grm", "truth_cs", "lrkin_cs"],
               "_v2": ["truth", "white_king", "white_gls_king", "hyb_gls", "truth_cs", "white_cs_king",
                       "hyb_cs_white", "auto_robust", "lrkin_cs", "lrkin_cs_w"],
               "_new": ["truth", "heteropca_grm", "wt_king", "white_auto", "truth_cs", "heteropca_raw",
                        "lrkin_raw", "white_cs_king", "lrkin_cs_ea"]}
    for tag, cols in figsets.items():
        for (p1, p2) in [(0, 1), (1, 2)]:
            scens = list(panels)
            fig, axes = plt.subplots(len(scens), len(cols), figsize=(3.75 * len(cols), 3.6 * len(scens)),
                                     squeeze=False)
            for r, scen in enumerate(scens):
                S, N = panels[scen]
                for c, name in enumerate(cols):
                    Y = S[name]
                    ax = axes[r, c]
                    for p in pops:
                        ii = [i for i in range(n0) if labels[i] == p]
                        ax.scatter(Y[ii, p1], Y[ii, p2], s=22, c=COL[p], label=p, edgecolors="none")
                    ax.scatter(Y[n0:N, p1], Y[n0:N, p2], s=40, c=REL_COL, marker="^", label="relatives",
                               edgecolors="k", linewidths=0.4)
                    if r == 0:
                        ax.set_title(titles[name], fontsize=11)
                    ax.set_xlabel(f"PC{p1 + 1}")
                    ax.set_ylabel(f"PC{p2 + 1}")
                    ax.tick_params(labelsize=7)
                    if c == 0:
                        ax.text(-0.32, 0.5, results["scen"][scen]["desc"], transform=ax.transAxes,
                                rotation=90, va="center", ha="center", fontsize=9)
            axes[0, 0].legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f"fig{tag}_pc{p1 + 1}{p2 + 1}.pdf"))
            plt.close(fig)

    # lambda sweep for the PCP variants on the N x N matrices
    variants = [("rsvd_grm", False, False), ("aarobust", False, True), ("rpca_cs", True, False)]
    mults = [float(x) for x in a.lam_sweep.split(",")]
    sweep = {}
    for scen, (C, H, T, Tcs, base, rel, pairs) in sweep_inputs.items():
        for name, use_cs, fd in variants:
            for mlt in mults:
                N = len(C)
                L_, S_, _ = rpca(H if use_cs else C, mlt / np.sqrt(N), free_diag=fd)
                P = cs_pcs(L_, k)[1] if use_cs else top_eig(L_, k)[1]
                r2, err = scores_vs_truth(P, Tcs if use_cs else T, base, rel)
                sweep[(scen, name, mlt)] = (r2[2], err, sparse_hits(S_, pairs, len(pairs))[0], len(pairs))
    results["lambda_sweep"] = [dict(scen=s_, method=m_, mult=l_, r2_pc3=v[0], err_rel=v[1], hits=v[2], pairs=v[3])
                               for (s_, m_, l_), v in sweep.items()]
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "table_lambda.tex"), "w") as f:
        f.write("\\begin{tabular}{ll" + "c" * len(mults) + "}\n\\toprule\n")
        f.write(" & & \\multicolumn{%d}{c}{$\\lambda \\times \\sqrt{N}$} \\\\\n" % len(mults))
        f.write("relatives & method & " + " & ".join(f"{m:g}" for m in mults) + " \\\\\n\\midrule\n")
        for si, scen in enumerate(sweep_inputs):
            for vi, (name, _, _) in enumerate(variants):
                cells = []
                for mlt in mults:
                    r2, err, h, npair = sweep[(scen, name, mlt)]
                    cell = f"{r2:.2f}/{err:.2f}"
                    cells.append("\\textbf{" + cell + "}" if (r2 >= 0.99 and err <= 0.1) else cell)
                lab_ = "\\texttt{" + scen.replace("_", "\\_") + "}" if vi == 0 else ""
                f.write(f"{lab_} & {name.replace('_', ' ')} & " + " & ".join(cells) + " \\\\\n")
            if si < len(sweep_inputs) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # tau sweep for the kinship-thresholded variants
    taus = [float(x) for x in a.tau_sweep.split(",")]
    kvars = ["pcpkin_grm", "pcpkin_cs", "lrkin_grm", "lrkin_cs"]
    tsweep = {}
    for scen, (C, H, Dh, pairs) in kin_inputs.items():
        S_, N_ = panels[scen]
        base, rel = np.arange(n0), np.arange(n0, N_)
        for t in taus:
            runs = {"pcpkin_grm": lambda: pcp_kin(C, t, True),
                    "pcpkin_cs": lambda: pcp_kin(double_centre(H), t, False, Dh),
                    "lrkin_grm": lambda: lr_kin(C, k, t, True),
                    "lrkin_cs": lambda: lr_kin(H, k + 1, t, False, Dh)}
            for name in kvars:
                L_, Sp, _ = runs[name]()
                P = cs_pcs(L_, k)[1] if name == "lrkin_cs" else top_eig(L_, k)[1]
                Tm = S_["truth_cs"] if name in uses_cs_truth else S_["truth"]
                r2, err = scores_vs_truth(P, Tm, base, rel)
                tp, fp = detected(Sp, pairs)
                tsweep[(scen, name, t)] = (r2[2], err, tp, fp, len(pairs))
    results["tau_sweep"] = [dict(scen=s_, method=m_, tau=t_, r2_pc3=v[0], err_rel=v[1], tp=v[2], fp=v[3], pairs=v[4])
                            for (s_, m_, t_), v in tsweep.items()]
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "table_tau.tex"), "w") as f:
        f.write("\\begin{tabular}{ll" + "c" * len(taus) + "}\n\\toprule\n")
        f.write(" & & \\multicolumn{%d}{c}{kinship threshold $\\tau$} \\\\\n" % len(taus))
        f.write("relatives & method & " + " & ".join(f"{t:g}" for t in taus) + " \\\\\n\\midrule\n")
        for si, scen in enumerate(kin_inputs):
            for vi, name in enumerate(kvars):
                cells = []
                for t in taus:
                    r2, err, tp, fp, npair = tsweep[(scen, name, t)]
                    cell = f"{r2:.2f}/{err:.2f}" + ("" if (tp == npair and fp == 0) else f" ({tp},{fp})")
                    ok = r2 >= 0.99 and err <= 0.1 and tp == npair and fp == 0
                    cells.append("\\textbf{" + cell + "}" if ok else cell)
                lab_ = "\\texttt{" + scen.replace("_", "\\_") + "}" if vi == 0 else ""
                f.write(f"{lab_} & {name.replace('_', ' ')} & " + " & ".join(cells) + " \\\\\n")
            if si < len(kin_inputs) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # base-sample truth vs reference-panel truth
    with open(os.path.join(a.out, "table_reftruth.tex"), "w") as f:
        rows = ["standard", "zhou", "masked_lr", "rsvd_grm", "aarobust", "aarobust_kin", "white_king", "white_gls_king",
                "white_gls_ped", "white_cs_king", "hyb_gls", "auto_robust", "lrkin_cs", "lrkin_cs_kc", "hyb_cs_kc"]
        scens = list(results["scen"])
        f.write("\\begin{tabular}{l" + "cc" * len(scens) + "}\n\\toprule\n & " +
                " & ".join("\\multicolumn{2}{c}{\\texttt{" + sc.replace("_", "\\_") + "}}" for sc in scens) +
                " \\\\\n method & " + " & ".join(["base", "ref."] * len(scens)) + " \\\\\n\\midrule\n")
        for m in rows:
            cells = []
            for sc in scens:
                mm = results["scen"][sc]["m"][m]
                cells += [f"{min(mm['r2']):.3f}", f"{min(mm['r2_ref']):.3f}"]
            f.write(m.replace("_", " ") + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n")
        for m in rows:
            cells = []
            for sc in scens:
                mm = results["scen"][sc]["m"][m]
                cells += [f"{mm['err_rel']:.2f}", f"{mm['err_ref']:.2f}"]
            f.write(m.replace("_", " ") + " (error) & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # LaTeX table of the versions of the shortlisted methods
    with open(os.path.join(a.out, "table_v2.tex"), "w") as f:
        rows = ["white_ped", "white_king", "white_cs_ped", "white_cs_king", "lrkin_cs"] + v2_methods
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("relatives & method & $R^2_{\\mathrm{PC1}}$ & $R^2_{\\mathrm{PC2}}$ & $R^2_{\\mathrm{PC3}}$ "
                "& error rel. \\\\\n\\midrule\n")
        for si, (scen, out) in enumerate(results["scen"].items()):
            for i, m in enumerate(rows):
                r2 = out["m"][m]["r2"]
                lab_ = "\\texttt{" + scen.replace("_", "\\_") + "}" if i == 0 else ""
                star = "$^*$" if m in uses_cs_truth else ""
                extra = f" (K={out['autoK']['K']})" if m == "lrkin_cs_autoK" else ""
                f.write(f"{lab_} & {m.replace('_', ' ')}{star}{extra} & " + " & ".join(f"{x:.3f}" for x in r2)
                        + f" & {out['m'][m]['err_rel']:.2f} \\\\\n")
                if i == 4:
                    f.write("\\cmidrule{2-6}\n")
            if si < len(results["scen"]) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # LaTeX table of the new variants, with three references
    with open(os.path.join(a.out, "table_new.tex"), "w") as f:
        rows = ["standard", "white_ped", "lrkin_cs"] + new_methods
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("relatives & method & $R^2_{\\mathrm{PC1}}$ & $R^2_{\\mathrm{PC2}}$ & $R^2_{\\mathrm{PC3}}$ "
                "& error rel. \\\\\n\\midrule\n")
        for si, (scen, out) in enumerate(results["scen"].items()):
            for i, m in enumerate(rows):
                r2 = out["m"][m]["r2"]
                lab_ = "\\texttt{" + scen.replace("_", "\\_") + "}" if i == 0 else ""
                star = "$^*$" if m in uses_cs_truth else ""
                f.write(f"{lab_} & {m.replace('_', ' ')}{star} & " + " & ".join(f"{x:.3f}" for x in r2)
                        + f" & {out['m'][m]['err_rel']:.2f} \\\\\n")
                if i == 2:
                    f.write("\\cmidrule{2-6}\n")
            if si < len(results["scen"]) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # LaTeX table
    with open(os.path.join(a.out, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("relatives & method & $R^2_{\\mathrm{PC1}}$ & $R^2_{\\mathrm{PC2}}$ & $R^2_{\\mathrm{PC3}}$ "
                "& error rel. \\\\\n\\midrule\n")
        for scen, out in results["scen"].items():
            for i, m in enumerate(methods):
                r2 = out["m"][m]["r2"]
                f.write(("\\texttt{" + scen.replace("_", "\\_") + "}" if i == 0 else "") + f" & {m.replace('_', ' ')}{'$^*$' if m in uses_cs_truth else ''} & "
                        + " & ".join(f"{x:.3f}" for x in r2) + f" & {out['m'][m]['err_rel']:.2f} \\\\\n")
            f.write("\\midrule\n" if scen != list(results["scen"])[-1] else "")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
