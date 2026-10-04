#!/usr/bin/env python3
"""Prototype of dwg for genotype likelihoods (dense engine).

1. Initial individual allele frequencies (IAF) by a PCAngsd-style EM:
   posterior means E under a prior from the IAF, PCA of E (diagonal
   corrected), IAF from the top components, iterated.
2. Posteriors with the IAF prior: E[g], Var(g | data) per individual and site.
3. dwg on the posterior means (diagonal free). Posterior means shrink towards
   the prior, so the kinship of relatives is attenuated by sqrt(rho_i rho_j),
   rho_i = 1 - sum Var(g|data) / sum 2 pi (1 - pi) (genotype reliability):
   the kinship rule uses phi / sqrt(rho_i rho_j); the whitening uses the
   observed (attenuated) covariance. Noise edge from the posterior means.
4. Candidates: KING from expected counts (posterior probabilities),
   corrected by sqrt(rho_i rho_j), and the evalAdmix screen on the posterior
   means; evalAdmix-only candidates confirmed by k0 < 0.8 from the genotype
   likelihoods themselves (IBD likelihood summed over genotypes).
5. IAF from the robust PCs, posteriors and dwg again (rounds).
"""
import numpy as np

import benchmark as B
import illustrate as I
from dwg_loc import TAU3, family_axis

K0_MAX = 0.8
NEFF = 4


def em_freq(GL, it=100, tol=1e-6):
    """allele frequencies by EM from 0.25, as PCAone (emMAF_with_GL: RMS change < tol)"""
    GL = GL.astype(np.float64)
    f = np.full(GL.shape[1], 0.25)
    for _ in range(it):
        pr = np.stack([(1 - f) ** 2, 2 * f * (1 - f), f ** 2], 1)
        post = GL * pr[None]
        post /= post.sum(2, keepdims=True)
        fn = (post[:, :, 1] + 2 * post[:, :, 2]).mean(0) / 2
        d = np.sqrt(np.mean((fn - f) ** 2))
        f = fn
        if d < tol:
            break
    return np.clip(f, 1e-4, 1 - 1e-4)


def posteriors(GL, Pi):
    """Pi: N x M IAF (or M allele frequencies); posterior probabilities N x M x 3"""
    Pi = np.asarray(Pi, dtype=GL.dtype)
    if Pi.ndim == 1:
        Pi = Pi[None, :]
    post = GL.copy()
    post[:, :, 0] *= (1 - Pi) ** 2
    post[:, :, 1] *= 2 * Pi * (1 - Pi)
    post[:, :, 2] *= Pi ** 2
    post /= post.sum(2, keepdims=True)
    return post


def iaf_from_axes(E, U):
    """individual allele frequencies by regression of the posterior means on [U, 1]"""
    V = np.c_[U, np.ones(len(E))]
    return np.clip(0.5 * V @ np.linalg.lstsq(V, E, rcond=None)[0], 1e-3, 1 - 1e-3)


def iaf_from_axes_loo(E, U):
    """IAF by regression on [U, 1] with each individual left out of its own
    fit: at small N the own genotypes would otherwise enter the prior"""
    N = len(E)
    V = np.c_[U, np.ones(N)]
    VtV, VtE = V.T @ V, V.T @ E
    Pi = np.empty_like(E)
    for i in range(N):
        vi = V[i]
        b = np.linalg.solve(VtV - np.outer(vi, vi) + 1e-9 * np.eye(len(vi)), VtE - np.outer(vi, E[i]))
        Pi[i] = 0.5 * vi @ b
    return np.clip(Pi, 1e-3, 1 - 1e-3)


def pcangsd_iaf(GL, f, r, it=10):
    """PCAngsd-style: E -> SVD of standardised E -> IAF from r components"""
    N, M = GL.shape[:2]
    Pi = np.tile(f, (N, 1))
    sd = np.sqrt(2 * f * (1 - f))
    for _ in range(it):
        post = posteriors(GL, Pi)
        E = post[:, :, 1] + 2 * post[:, :, 2]
        X = (E - 2 * f) / sd
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Pi_new = np.clip((U[:, :r] * S[:r] @ Vt[:r] * sd + 2 * f) / 2, 1e-3, 1 - 1e-3)
        if np.max(np.abs(Pi_new - Pi)) < 1e-4:
            Pi = Pi_new
            break
        Pi = Pi_new
    return Pi


def king_expected(post):
    p0, p1, p2 = post[:, :, 0], post[:, :, 1], post[:, :, 2]
    hh = p1 @ p1.T
    i0 = p0 @ p2.T + p2 @ p0.T
    nh = p1.sum(1)
    mn = np.maximum(np.minimum.outer(nh, nh), 1e-9)
    K = (hh - 2 * i0) / (2 * mn) + 0.5 - (nh[:, None] + nh[None, :]) / (4 * mn)
    np.fill_diagonal(K, 0)
    return K


def gl_k0(GL, Pi_i, Pi_j, i, j, iters=30):
    """ML k0 (EM over the three IBD states) from genotype likelihoods with
    IAF pi_i, pi_j (allele-level likelihood of pcaone-ibd summed over genotypes)"""
    gi, gj = GL[i], GL[j]  # M x 3
    pa, pb = Pi_i, Pi_j
    ps = 0.5 * (pa + pb)
    hw = lambda p: np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2], 1)  # noqa: E731
    l0 = (gi * hw(pa)).sum(1) * (gj * hw(pb)).sum(1)
    # IBD1: shared allele S ~ ps; non-shared alleles ~ pa, pb
    def nonshared(g, p):  # P(genotype g | shared allele S) for S = 0, 1
        return np.stack([np.stack([1 - p, p, np.zeros_like(p)], 1), np.stack([np.zeros_like(p), 1 - p, p], 1)], 0)
    na, nb = nonshared(None, pa), nonshared(None, pb)  # 2 x M x 3
    l1 = (1 - ps) * (gi * na[0]).sum(1) * (gj * nb[0]).sum(1) + ps * (gi * na[1]).sum(1) * (gj * nb[1]).sum(1)
    l2 = (gi * gj * hw(ps)).sum(1)
    L = np.stack([l0, l1, l2], 1)
    k = np.array([1 / 3, 1 / 3, 1 / 3])
    for _ in range(iters):
        w = L * k
        t = w.sum(1, keepdims=True)
        kn = (w / t).mean(0)
        if np.abs(kn - k).sum() < 1e-4:  # the screen only needs k0 < 0.8
            k = kn
            break
        k = kn
    return k


def fit(A, tau, cand, edge, rmax, rho, iters=500):
    """dwg fit (family axes, kinship check) with the kinship scale v*rho"""
    n = len(A)
    eye = np.eye(n, dtype=bool)
    cand = cand.copy()
    S = np.zeros_like(A)
    r = 1
    while True:
        L_old = None
        for _ in range(iters):
            w_, V = I.top_eig(A - S, r)
            L = (V * w_) @ V.T
            R = A - L
            v = np.maximum(np.diag(R), 1e-9) * rho
            sel = I.kin_select(R, v, tau) & cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        if r >= min(rmax, n - 1):
            break
        ev, V = I.top_eig(A - S, r + 1)
        if ev[r] <= edge:
            break
        fam, top = family_axis(V[:, r], A - L, np.maximum(np.diag(A - L), 1e-9) * rho, NEFF, True)
        if fam:
            new = [(i, j) for a, i in enumerate(top) for j in top[a + 1:] if not cand[i, j]]
            if not new:
                break
            for i, j in new:
                cand[i, j] = cand[j, i] = True
            continue
        r += 1
    vr = np.maximum(np.diag(A - L), 1e-9)
    phi_obs = I.kin_scale(A - L, vr)  # attenuated (for the whitening)
    iu = np.triu_indices(n, 1)
    pairs = [(int(i), int(j), float(phi_obs[i, j])) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]
    ev, V = I.top_eig(L, r)
    keep = []
    for j in range(1, r):
        Rj = A - L + ev[j] * np.outer(V[:, j], V[:, j])
        f_, _ = family_axis(V[:, j], Rj, np.maximum(np.diag(Rj), 1e-9) * rho, NEFF, True)
        if not f_:
            keep.append(j)
    axes = V[:, keep] if keep else V[:, 1:2]
    return L, S, r, pairs, axes, vr


def dwg_on_posteriors(GL, Pi, f, k, tau, cand_extra=None, deshrink=False, post=None):
    if post is None:
        post = posteriors(GL, Pi)
    E = post[:, :, 1] + 2 * post[:, :, 2]
    var = post[:, :, 1] + 4 * post[:, :, 2] - E ** 2
    N, M = E.shape
    w = 2 * f * (1 - f)
    rho = np.clip(1 - var.sum(1) / (2 * Pi * (1 - Pi)).sum(1), 0.02, 1.0)
    if deshrink:
        # undo the shrinkage towards the prior: x = 2 pi + (E - 2 pi) / rho_i has
        # approximately unbiased off-diagonal cross-products; the extra noise
        # goes to the (free) diagonal
        E = 2 * Pi + (E - 2 * Pi) / rho[:, None]
    Es = E / np.sqrt(w)
    A = Es @ Es.T / M
    m1, m2 = E.mean(0), (E ** 2).mean(0)
    edge = 2 * np.sqrt(np.sum(np.maximum(m2 ** 2 - m1 ** 4, 0) / w ** 2) / M ** 2) * np.sqrt(N)
    Kexp = king_expected(post) / np.sqrt(np.outer(rho, rho))
    off = ~np.eye(N, dtype=bool)
    ck = (Kexp > B.KING_SCREEN) & off
    if cand_extra is not None:
        ck |= cand_extra
    L, S, r, pairs, axes, vr = fit(A, tau, ck, edge, k + 1, rho)
    return dict(post=post, E=E, A=A, rho=rho, edge=edge, ck=ck, L=L, S=S, r=r, pairs=pairs, axes=axes, v=vr)


def dwg_gl(GL, k, tau=B.TAU, rounds=3, screen_rounds=3, loo=False, deshrink=False):
    """round 1 with the HWE prior (sample allele frequencies); later rounds
    with IAF from the robust axes of the previous round (no dependence on k)"""
    N, M = GL.shape[:2]
    GL = GL.astype(np.float64)
    f = em_freq(GL)
    Pi = np.tile(f, (N, 1))
    for _ in range(rounds):
        R = dwg_on_posteriors(GL, Pi, f, k, tau, deshrink=deshrink)
        key = {(i, j) for i, j, _ in R["pairs"]}
        extra = np.zeros((N, N), dtype=bool)
        for _ in range(screen_rounds):
            # evalAdmix screen on the posterior means, confirmed by GL k0
            ea = I.evaladmix_kin(R["E"], R["axes"], intercept=True) / np.sqrt(np.outer(R["rho"], R["rho"]))
            ce = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(ea > B.KING_SCREEN, 1)))
                  if not R["ck"][i, j] and not extra[i, j]]
            Pi_ax = iaf_from_axes(R["E"], R["axes"])
            add = [(i, j) for i, j in ce if gl_k0(GL, Pi_ax[i], Pi_ax[j], i, j)[0] < K0_MAX]
            if not add:
                break
            for i, j in add:
                extra[i, j] = extra[j, i] = True
            R = dwg_on_posteriors(GL, Pi, f, k, tau, extra, deshrink=deshrink, post=R["post"])
            nk = {(i, j) for i, j, _ in R["pairs"]}
            if nk == key:
                break
            key = nk
        U = I.whitened_noise(R["A"], R["v"], R["pairs"], k)
        Ur = U[:, :max(R["r"] - 1, 1)]  # IAF from the robust PCs (leave-one-out)
        Pi = iaf_from_axes_loo(R["E"], Ur) if loo else iaf_from_axes(R["E"], Ur)
    R["U"] = U
    R["Pi"] = Pi
    return R
