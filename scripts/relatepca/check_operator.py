#!/usr/bin/env python3
"""Stage 0: do the shortlisted methods work through an IRAM-style operator?

For benchmark datasets, compare the dense eigen-solves used by the prototypes
(numpy eigh on an N x N matrix) with an implicitly restarted Lanczos/Arnoldi
solver (scipy eigsh = ARPACK, the same algorithm family as PCAone's Spectra
SymEigsSolver) that only sees matrix-free products with the genotypes:

  CS            y = G (G'x)/M - D x - S x           (raw genotypes)
  centred CS    y = Xc (Xc'x)/M - J D J x - J S J x  (SNP-centred genotypes, as
                PCAone's out-of-core reader returns them); top k = structure PCs
  CS whitening  y = W G (G' W x)/M,  W = Sigma^-1/2 block diagonal

and run the full fixed-rank + kinship loop (rank stepping, KING screen) with
the operator solver in place of eigh. Also check PCAone -d 0 (in-core, -m)
against numpy standard PCA.
"""
import argparse
import os
import subprocess
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402


def op_eig(matvec, n, r, v0=None):
    """top r eigenpairs (largest algebraic) of a symmetric operator by ARPACK"""
    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    w, v = eigsh(A, k=r, which="LA", tol=1e-12, ncv=min(n, max(2 * r + 1, 20)), v0=v0)
    o = np.argsort(w)[::-1]
    return w[o], v[:, o]


def cosmin(U, V):
    """smallest |cos| between matching columns"""
    c = np.abs(np.sum(U * V, 0)) / (np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0))
    return float(c.min())


def subspace_cos(U, V):
    """cosine of the largest principal angle between column spaces"""
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    return float(np.linalg.svd(Qu.T @ Qv, compute_uv=False).min())


def lr_kin_operator(G, D, rank, tau, cand, centred=False, iters=500):
    """fixed rank + kinship with every eigen-solve done by ARPACK on the
    matrix-free operator; S is kept as a sparse set of entries"""
    N, M = G.shape
    if centred:
        Gc = G - G.mean(0)
        J = lambda x: x - x.mean()  # noqa: E731
        base = lambda x: Gc @ (Gc.T @ x) / M - J(D * J(x))  # noqa: E731
    else:
        base = lambda x: G @ (G.T @ x) / M - D * x  # noqa: E731
    S = np.zeros((N, N))
    # H_ij for every pair, needed only for candidates; dense here for checking
    Hfull = np.array([base(e) for e in np.eye(N)]).T
    v0 = None
    for r in range(1, rank + 1):
        L_old = None
        for _ in range(iters):
            if centred:
                mv = lambda x: base(x) - J(S @ J(x))  # noqa: E731
            else:
                mv = lambda x: base(x) - S @ x  # noqa: E731
            w, v = op_eig(mv, N, r)
            L = (v * w) @ v.T
            # residual of the observed matrix
            Rm = Hfull - L
            sel = I.kin_select(Rm, D, tau) & cand
            S = np.where(sel, Rm, 0)
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-9:
                break
            L_old = L
    return L, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    fam, bim, Gall = read_bed(a.bfile)
    lab = np.array([r[1] for r in fam])
    pops = "CEU,CHB,MXL,YRI"
    k = 3
    rows = []
    for n in [5, 10, 40]:
        for scen in ["none", "mz2", "childx"]:
            seed = zlib.crc32(f"{pops}|{n}|0|{scen}|0".encode())
            ds = B.make_dataset(Gall, lab, bim, pops.split(","), n, 0, B.SCENARIOS[scen], seed)
            G = ds["G"]
            N, M = G.shape
            D = (G * (2 - G)).mean(1)
            H = I.cs_matrix(G)
            cand = B.king_robust(G) > B.KING_SCREEN
            res = {"n": n, "scen": scen, "N": N}

            # 1. CS matrix, top k+1, operator vs dense
            wd, vd = I.top_eig(H, k + 1)
            wo, vo = op_eig(lambda x: G @ (G.T @ x) / M - D * x, N, k + 1)
            res["CS eig"] = cosmin(vd, vo)

            # 2. centred CS from centred genotypes vs double-centred dense
            Gc = G - G.mean(0)
            J = lambda x: x - x.mean()  # noqa: E731
            wcd, vcd = I.top_eig(I.double_centre(H), k)
            wco, vco = op_eig(lambda x: Gc @ (Gc.T @ x) / M - J(D * J(x)), N, k)
            res["centred CS eig"] = cosmin(vcd, vco)
            # and: are the centred structure PCs the same space as CS PCs 2..k+1?
            res["centred vs CS PCs"] = subspace_cos(vcd, vd[:, 1:])

            # 3. full fixed rank + kinship loop, dense vs operator (raw)
            Ld, Sd, _ = I.lr_kin(H, k + 1, B.TAU, False, D, cand=cand)
            Lo, So = lr_kin_operator(G, D, k + 1, B.TAU, cand)
            pd_ = {(i, j) for i, j in zip(*np.nonzero(np.triu(np.abs(Sd) > 1e-12, 1)))}
            po_ = {(i, j) for i, j in zip(*np.nonzero(np.triu(np.abs(So) > 1e-12, 1)))}
            res["frkin pairs same"] = pd_ == po_
            res["frkin PCs"] = subspace_cos(I.cs_pcs(Ld, k)[1], I.cs_pcs(Lo, k)[1])
            # centred variant of the loop (rank k), compared with the raw dense one
            Lc, Sc = lr_kin_operator(G, D, k, B.TAU, cand, centred=True)
            pc_ = {(i, j) for i, j in zip(*np.nonzero(np.triu(np.abs(Sc) > 1e-12, 1)))}
            res["frkin centred pairs same"] = pd_ == pc_
            res["frkin centred PCs"] = subspace_cos(I.cs_pcs(Ld, k)[1], I.top_eig(Lc, k)[1])

            # 4. CS whitening with the detected kinship: dense vs operator
            pp = I.lr_kin_pairs(H, D, Ld, Sd)
            Ud = I.whitened_cs(G, pp, k)
            d = np.sqrt(D)
            R_ = np.eye(N)
            for i, j, phi in pp:
                R_[i, j] = R_[j, i] = 2 * phi
            Sig = d[:, None] * R_ * d[None, :]
            w_, v_ = np.linalg.eigh(Sig)
            w_ = np.maximum(w_, 0.1 * d.min() ** 2)
            Wm, Wh = (v_ / np.sqrt(w_)) @ v_.T, (v_ * np.sqrt(w_)) @ v_.T
            _, vw = op_eig(lambda x: Wm @ (G @ (G.T @ (Wm @ x))) / M, N, k + 1)
            res["CS whitening"] = subspace_cos(Ud, (Wh @ vw)[:, 1:])
            res["pairs"] = len(pd_)
            rows.append(res)
            print(res, flush=True)

    # 5. PCAone -d 0 in-core and out-of-core vs numpy standard PCA
    seed = zlib.crc32(f"{pops}|10|0|mz2|0".encode())
    ds = B.make_dataset(Gall, lab, bim, pops.split(","), 10, 0, B.SCENARIOS["mz2"], seed)
    G = ds["G"]
    ids = [f"I{i}" for i in range(len(G))]
    pre = os.path.join(a.work, "chk")
    write_bed(pre, [["F", x, "0", "0", "0", "-9"] for x in ids], [bim[i] for i in ds["cols"]], G.astype(np.int8))
    _, C = B.grm(G)
    Ustd = I.top_eig(C, k)[1]
    for tag, extra in [("in-core", ""), ("out-of-core", "-m 0.002")]:
        subprocess.run(f"{a.pcaone} -b {pre} -k {k} -d 0 -n 4 -v 0 {extra} -o {pre}_{tag}", shell=True,
                       check=True, capture_output=True)
        U = np.loadtxt(f"{pre}_{tag}.eigvecs", ndmin=2)
        print(f"PCAone -d 0 {tag}: min |cos| with numpy standard PCA = {cosmin(Ustd, U):.8f}")


if __name__ == "__main__":
    main()
