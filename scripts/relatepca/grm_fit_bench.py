#!/usr/bin/env python3
"""Fixed-rank + kinship fits on the GRM (instead of the CS matrix or PCP),
scored against the reference truth on the small-N scenarios of benchmark.py.

  grm_fit    rank-r fit of the GRM with the diagonal free and the kinship rule
             (KING-screened candidates); rank raised while the next eigenvalue
             of C - S exceeds the GRM noise edge 2 sqrt(N/M), at most k;
             PCs = top k eigenvectors of L
  grm_white  the same detection, then whitening of the GRM with
             Sigma = v^1/2 (I + 2 Psi) v^1/2, v = C_ii - L_ii
  grm_iter   grm_white, then repeat: structure re-estimated in the span of the
             whitened PCs (diagonal imputed), pairs re-detected from the
             residual, re-whitened, until the pairs no longer change

All three need only the top r eigenvectors per step (IRAM-friendly).
References: standard PCA, aarobust-kin --impute-diag (PCP), detect-white.
"""
import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed  # noqa: E402

SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2", "err1_mz2"]
METHODS = ["standard", "aarobust_kin", "detect_white", "grm_fit", "grm_white", "grm_iter", "grm2_fit", "grm2_white",
           "grm2_iter"]
G_ALL = LAB = BIM = None


def grm_fit(C, tau, cand, edge, rmax, iters=500):
    """fixed rank + kinship on the GRM, diagonal free, rank from the noise edge"""
    n = len(C)
    eye = np.eye(n, dtype=bool)
    S = np.zeros_like(C)
    r = 1
    while True:
        L_old = None
        for _ in range(iters):
            w, v = I.top_eig(C - S, r)
            L = (v * w) @ v.T
            R = C - L
            sel = I.kin_select(R, np.diag(R), tau) & cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        if r >= min(rmax, n - 1) or I.top_eig(C - S, r + 1)[0][r] <= edge:
            break
        r += 1
    return L, S, r


def grm_fit2(C, tau, cand, rmax, iters=500):
    """grm_fit with (a) the kinship scaled by the observed GRM diagonal (fixed,
    like D for the CS matrix) and (b) the noise edge estimated from the data:
    2 sd(off-diagonal residual outside S) sqrt(N), re-estimated at every rank"""
    n = len(C)
    eye = np.eye(n, dtype=bool)
    vC = np.diag(C).copy()
    S = np.zeros_like(C)
    r = 1
    while True:
        L_old = None
        for _ in range(iters):
            w, v = I.top_eig(C - S, r)
            L = (v * w) @ v.T
            R = C - L
            sel = I.kin_select(R, vC, tau) & cand
            S = np.where(sel, R, 0)
            S[eye] = R[eye]
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        off = ~eye & ~sel
        edge = 2 * R[off].std() * np.sqrt(n)
        if r >= min(rmax, n - 1) or I.top_eig(C - S, r + 1)[0][r] <= edge:
            break
        r += 1
    return L, S, r


def pairs_of(R, v, sel):
    phi = I.kin_scale(R, v)
    iu = np.triu_indices(len(R), 1)
    return [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if sel[i, j]]


def whiten_grm(C, v, pairs, k, floor=0.1):
    """top k eigenvectors of Sigma^-1/2 C Sigma^-1/2 mapped back by Sigma^1/2
    (the GRM is centred: no mean component to drop)"""
    N = len(C)
    d = np.sqrt(np.maximum(v, 1e-12))
    R_ = np.eye(N)
    for i, j, phi in pairs:
        R_[i, j] = R_[j, i] = 2 * phi
    Sig = d[:, None] * R_ * d[None, :]
    w, Q = np.linalg.eigh(Sig)
    w = np.maximum(w, floor * d.min() ** 2)
    Wm, Wh = (Q / np.sqrt(w)) @ Q.T, (Q * np.sqrt(w)) @ Q.T
    _, U = I.top_eig(Wm @ C @ Wm, k)
    return Wh @ U


def structure_in_span(C, U, Ldiag, iters=50):
    """L = P C~ P with P the projector on span(U) and the diagonal of C~ imputed
    from L itself (iterated)"""
    Q, _ = np.linalg.qr(U)
    Ct = C.copy()
    for _ in range(iters):
        np.fill_diagonal(Ct, Ldiag)
        L = Q @ (Q.T @ Ct @ Q) @ Q.T
        if np.max(np.abs(np.diag(L) - Ldiag)) < 1e-10:
            break
        Ldiag = np.diag(L).copy()
    return L


def run(G, k, tau):
    out = {}
    _, C = B.grm(G)
    N, M = G.shape
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    out["standard"] = I.top_eig(C, k)[1]
    Lp, _, _ = I.pcp_kin(C, tau, True, cand=cand)
    out["aarobust_kin"] = I.top_eig(Lp, k)[1]
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L, S, _, _ = I.lr_kin_fd_auto(H, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
    pp = I.lr_kin_pairs(H, D, L, S)
    AM = G @ G.T / M
    out["detect_white"] = I.whitened_noise(AM, np.diag(AM) - np.diag(L), pp, k)
    # GRM variants
    edge = 1.1 * 2 * np.sqrt(N / M) * np.mean(np.diag(C))
    L, S, r = grm_fit(C, tau, cand, edge, rmax=k)
    out["grm_fit"] = I.top_eig(L, k)[1]
    R = C - L
    v = np.diag(R).copy()
    sel = (np.abs(S) > 1e-12) & ~np.eye(N, dtype=bool)
    pairs = pairs_of(R, v, sel)
    U = whiten_grm(C, v, pairs, k)
    out["grm_white"] = U
    key = set((i, j) for i, j, _ in pairs)
    Ldiag = np.diag(L).copy()
    for _ in range(10):
        Ls = structure_in_span(C, U, Ldiag)
        R = C - Ls
        v = np.maximum(np.diag(R), 1e-6)
        sel = I.kin_select(R, v, tau) & cand
        pairs = pairs_of(R, v, sel)
        new = set((i, j) for i, j, _ in pairs)
        U = whiten_grm(C, v, pairs, k)
        Ldiag = np.diag(Ls).copy()
        if new == key:
            break
        key = new
    out["grm_iter"] = U
    # corrected: fixed kinship scale (observed diagonal), empirical noise edge
    L, S, r2 = grm_fit2(C, tau, cand, rmax=k)
    out["grm2_fit"] = I.top_eig(L, k)[1]
    R = C - L
    vC = np.diag(C).copy()
    v = np.maximum(np.diag(R), 1e-6)
    sel = (np.abs(S) > 1e-12) & ~np.eye(N, dtype=bool)
    pairs = pairs_of(R, vC, sel)
    U = whiten_grm(C, v, pairs, k)
    out["grm2_white"] = U
    key = set((i, j) for i, j, _ in pairs)
    Ldiag = np.diag(L).copy()
    for _ in range(10):
        Ls = structure_in_span(C, U, Ldiag)
        R = C - Ls
        v = np.maximum(np.diag(R), 1e-6)
        sel = I.kin_select(R, vC, tau) & cand
        pairs = pairs_of(R, vC, sel)
        new = set((i, j) for i, j, _ in pairs)
        U = whiten_grm(C, v, pairs, k)
        Ldiag = np.diag(Ls).copy()
        if new == key:
            break
        key = new
    out["grm2_iter"] = U
    return out, r2


def init(bfile):
    global G_ALL, LAB, BIM
    fam, BIM, G_ALL = read_bed(bfile)
    LAB = np.array([r[1] for r in fam])


def one(job):
    n, scen, rep, k = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(G_ALL, LAB, BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, G_ALL, 3)
    G = ds["G"].astype(float)
    rel = np.arange(ds["n0"], len(G))
    res, r = run(G, k, B.TAU)
    rows = []
    for m in METHODS:
        r2, err = B.score(res[m], T, ds["n0"], rel, 3)
        rows.append((n, scen, rep, k, m, r2.min(), err, r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--ns", default="5,10,20,40")
    ap.add_argument("--ks", default="3,10")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(n, s, r, k) for k in map(int, a.ks.split(",")) for n in map(int, a.ns.split(",")) for s in SCEN
            for r in range(a.reps)]
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.bfile,)) as ex, open(a.out, "w") as fo:
        fo.write("n\tscen\trep\tk\tmethod\tminR2\terr\tgrm_rank\n")
        for rows in ex.map(one, jobs):
            for row in rows:
                fo.write("\t".join(str(x) for x in row) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
