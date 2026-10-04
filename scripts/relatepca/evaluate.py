#!/usr/bin/env python3
"""Compare PCA methods on data with close relatives (see sim.py).

Truth: a PCA of the reference panel (ref.bed, the unused individuals of the same
populations); every test individual is projected onto its top k axes.

For each method the top k PCs (k = K-1) are aligned to the truth by a
similarity Procrustes fit (rotation/reflection, one scale, translation) on the
*unrelated set* only, and the error is reported separately for the unrelated set
and for the remaining relatives, relative to the spread of the truth:

    err = sqrt( sum ||a_i - fit_i||^2 / sum ||a_i - mean(a)||^2 )

Methods
  std         standard PCA of everyone (PCAone -d 3)
  unrel_proj  PCA of the unrelated set, relatives projected onto its loadings
              (PC-AiR with oracle unrelated set)
  white_true  PCAone --kinship with the pedigree kinship
  white_king  PCAone --kinship with plink2 KING-robust kinship (>= 2^-3.5)
  lr_impute   GRM with the entries of KING-close pairs imputed from a rank-k fit
              (low rank + sparse, iterated; the "AArobust" idea)
"""
import argparse
import os
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import read_bed  # noqa: E402

KIN_MIN = 2 ** -3.5


def standardize(G, f=None):
    G = G.astype(float)
    if f is None:
        f = G.mean(axis=0) / 2
    sd = np.sqrt(2 * f * (1 - f))
    ok = sd > 1e-9
    return (G[:, ok] - 2 * f[ok]) / sd[ok], f, ok


def top_eig(C, k):
    w, v = np.linalg.eigh(C)
    return w[::-1][:k], v[:, ::-1][:, :k]


def procrustes_err(X, A, fit_idx):
    """align X to A on rows fit_idx; return aligned X"""
    Xc, Ac = X[fit_idx] - X[fit_idx].mean(0), A[fit_idx] - A[fit_idx].mean(0)
    U, s, Vt = np.linalg.svd(Xc.T @ Ac)
    R = U @ Vt
    scale = s.sum() / (Xc ** 2).sum()
    return (X - X[fit_idx].mean(0)) @ R * scale + A[fit_idx].mean(0)


def rel_err(A, Y, rows, centre):
    if len(rows) == 0:
        return np.nan
    return np.sqrt(((A[rows] - Y[rows]) ** 2).sum() / ((A[rows] - centre) ** 2).sum())


def run(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit(f"failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return r


def read_pairs(fn, ids, kmin):
    pos = {x: i for i, x in enumerate(ids)}
    lines = open(fn).read().split("\n")
    head = [h.lstrip("#").upper() for h in lines[0].split()]
    c1 = head.index("IID1") if "IID1" in head else head.index("ID1")
    c2 = head.index("IID2") if "IID2" in head else head.index("ID2")
    ck = head.index("KINSHIP")
    out = []
    for l in lines[1:]:
        t = l.split()
        if len(t) > ck and float(t[ck]) >= kmin:
            out.append((pos[t[c1]], pos[t[c2]], float(t[ck])))
    return out


def lr_impute(C, pairs, k, iters=100):
    """impute the GRM entries of close pairs from its rank-k approximation"""
    C = C.copy()
    if not pairs:
        return C
    I = np.array([p[0] for p in pairs] + [p[1] for p in pairs])
    J = np.array([p[1] for p in pairs] + [p[0] for p in pairs])
    for _ in range(iters):
        w, v = top_eig(C, k)
        L = (v * w) @ v.T
        new = L[I, J]
        if np.max(np.abs(C[I, J] - new)) < 1e-8:
            break
        C[I, J] = new
    return C


def one_rep(args):
    a, scen, fams, rep = args
    d = os.path.join(a.work, f"{scen}_r{rep}")
    seed = 1000 * rep + zlib.crc32(scen.encode()) % 1000
    run(f"{sys.executable} {HERE}/sim.py --bfile {a.bfile} --out {d} --pops {a.pops} "
        f"--n-unrel {a.n_unrel} --families '{fams}' --seed {seed}")
    k = a.k
    fam, _, G = read_bed(os.path.join(d, "test"))
    ids = [r[1] for r in fam]
    info = [l.split("\t") for l in open(os.path.join(d, "info.tsv")).read().strip().split("\n")[1:]]
    unrel = np.array([int(r[4]) for r in info]) == 1
    is_base = np.array([r[2] == "UNREL" for r in info])
    rel_rows = np.where(~is_base)[0]
    N = len(ids)

    # truth: reference PCA, test projected
    _, _, Gr = read_bed(os.path.join(d, "ref"))
    Xr, fr, ok = standardize(Gr)
    _, _, Vt = np.linalg.svd(Xr, full_matrices=False)
    Xt, _, _ = standardize(G[:, ok], fr[ok])
    A = Xt @ Vt[:k].T

    X, f, _ = standardize(G)
    M = X.shape[1]
    C = X @ X.T / M
    res = {}

    # std (PCAone)
    run(f"{a.pcaone} -b {d}/test -k {k} -d 3 -n 1 -v 0 -o {d}/std")
    res["std"] = np.loadtxt(f"{d}/std.eigvecs", ndmin=2)

    # oracle unrelated set + projection of the rest
    Xu = X[unrel]
    w, v = top_eig(Xu @ Xu.T / M, k)
    V = Xu.T @ v / np.sqrt(w * M)
    S = np.zeros((N, k))
    S[unrel] = v
    S[~unrel] = X[~unrel] @ V / np.sqrt(w * M)
    res["unrel_proj"] = S

    # whitened, truth and KING kinship
    run(f"{a.pcaone} -b {d}/test -k {k} -d 3 -n 1 -v 0 --kinship {d}/truth.kin0 -o {d}/wtrue")
    res["white_true"] = np.loadtxt(f"{d}/wtrue.eigvecs", ndmin=2)
    run(f"{a.plink2} --bfile {d}/test --make-king-table --king-table-filter {KIN_MIN / 2} --out {d}/king --threads 1")
    run(f"{a.pcaone} -b {d}/test -k {k} -d 3 -n 1 -v 0 --kinship {d}/king.kin0 -o {d}/wking")
    res["white_king"] = np.loadtxt(f"{d}/wking.eigvecs", ndmin=2)

    # low rank + sparse imputation of KING-close entries
    pairs = read_pairs(f"{d}/king.kin0", ids, KIN_MIN)
    _, v = top_eig(lr_impute(C, pairs, k), k)
    res["lr_impute"] = v

    # numpy cross-check of the C++ whitening against the pedigree kinship
    Sig = np.eye(N)
    for i, j, phi in read_pairs(f"{d}/truth.kin0", ids, KIN_MIN):
        Sig[i, j] = Sig[j, i] = 2 * phi
    ew, ev = np.linalg.eigh(Sig)
    ew = np.maximum(ew, 0.1)
    Lh, Linvh = (ev * np.sqrt(ew)) @ ev.T, (ev / np.sqrt(ew)) @ ev.T
    Xw = Linvh @ X
    _, vw = top_eig(Xw @ Xw.T / M, k)
    ref_w = Lh @ vw
    chk = np.abs(np.abs(np.sum(ref_w * res["white_true"], 0)) /
                 (np.linalg.norm(ref_w, axis=0) * np.linalg.norm(res["white_true"], axis=0)) - 1).max()

    rows = []
    centre_u = A[unrel].mean(0)
    for m, Xm in res.items():
        Y = procrustes_err(Xm[:, :k], A, np.where(unrel)[0])
        rows.append(dict(scen=scen, rep=rep, method=m, N=N,
                         err_unrel=rel_err(A, Y, np.where(is_base)[0], centre_u),
                         err_rel=rel_err(A, Y, rel_rows, centre_u),
                         err_all=rel_err(A, Y, np.arange(N), centre_u)))
    if not a.keep:
        run(f"rm -rf {d}")
    return rows, chk


SCENARIOS = {
    "none": "",
    "dup2_CEU": "dup:CEU:2",
    "dup3_CEU": "dup:CEU:3",
    "sibs4_CEU": "sibs:CEU:4",
    "nuclear3_MXL": "nuclear:MXL:3",
    "halfsibs4_YRI": "halfsibs:YRI:4",
    "deg2_mix": "avuncular:CEU,grand:CHB,avuncular:MXL,grand:YRI",
    "extended3_CEU": "extended:CEU:3",
    "many": "sibs:CEU:4,nuclear:CHB:2,sibs:MXL:3,dup:YRI:2",
    "admixed_sibs": "sibs:CEUxYRI:4",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--pcaone", required=True)
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True, help="results tsv")
    ap.add_argument("--pops", default="CEU,CHB,MXL,YRI")
    ap.add_argument("--k", type=int, default=3, help="number of PCs, K-1")
    ap.add_argument("--n-unrel", type=int, default=10)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    jobs = [(a, s, SCENARIOS[s], r) for s in a.scenarios.split(",") for r in range(a.reps)]
    allrows, chks = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for rows, chk in ex.map(one_rep, jobs):
            allrows += rows
            chks.append(chk)
    cols = ["scen", "rep", "method", "N", "err_unrel", "err_rel", "err_all"]
    with open(a.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in allrows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"max |1 - |cos|| between C++ and numpy whitened PCs: {max(chks):.2e}")
    # summary: mean error per scenario x method
    meths = list(dict.fromkeys(r["method"] for r in allrows))
    for col in ["err_unrel", "err_rel"]:
        print(f"\nmean {col} (relative to truth spread), k = {a.k}, {a.reps} reps")
        print("scenario".ljust(16) + "".join(m.rjust(12) for m in meths))
        for s in dict.fromkeys(r["scen"] for r in allrows):
            vals = [np.nanmean([r[col] for r in allrows if r["scen"] == s and r["method"] == m]) for m in meths]
            print(s.ljust(16) + "".join(f"{v:12.4f}" for v in vals))


if __name__ == "__main__":
    main()
