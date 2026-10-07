#!/usr/bin/env python3
"""The four small-N methods side by side (prototypes, which PCAone
reproduces: check_pcaone.py, check_dwg.py), plus standard PCA as reference,
on admixTjeck2 with the benchmark.py datasets:

  unc_fit       uncentred, SNP-standardised GRM; rank-r fit with the diagonal
                (and related pairs) free, rank from the noise edge (<= k+1);
                PCs of L, mean dropped
  aarobust_kin  PCP of the centred, scaled GRM with the kinship threshold,
                diagonal free (--impute-diag); PCs of L
  detect_white  Chen & Storey matrix, rank-r fit with the diagonal free,
                whitening with the observed noise
  dwg           uncentred GRM fit + family axes + evalAdmix/k0 screen,
                whitening, centred final step (as PCAone)

Scenarios: HWE within individuals (inbreeding, genotype errors) and
relatives (incl. relatives of different ancestry). Per run: min R2 of the 3
ancestry axes against the reference truth, relatives' error, recall of the
true pairs (pedigree kinship >= tau) and false pairs.

usage: four_methods_bench.py <out.tsv> [workers]
"""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import dwg_loc as DL  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402

HWE = ["inbred1", "inbredcous", "inbredpop", "err1", "inbredpop_mz2", "err1_mz2"]
REL = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "xanc", "grandx", "avuncx", "halfsibx"]
K = 3


def pairs_from_S(S):
    N = len(S)
    iu = np.triu_indices(N, 1)
    return {(int(i), int(j)) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12}


def one(job):
    group, n, scen, rep = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(float)
    N = len(G)
    rel = np.arange(ds["n0"], N)
    true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(ds["Kped"] >= B.TAU, 1)))}
    out = run_methods(G)
    rows = []
    for m, (Um, det) in out.items():
        r2, err = B.score(Um, T, ds["n0"], rel, 3)
        rec = len(det & true) / len(true) if true else np.nan
        rows.append((group, n, scen, rep, m, r2.min(), err, rec, len(det - true), len(true)))
    return rows


def run_methods(G):
    """{method: (top K ancestry PCs, detected pairs)} for a genotype matrix"""
    N = len(G)
    off = ~np.eye(N, dtype=bool)
    cand = (B.king_robust(G) > B.KING_SCREEN) & off
    out = {}
    # standard PCA
    X = I.standardize(G)
    X = X[:, np.isfinite(X).all(0)]
    out["standard"] = (I.top_eig(X @ X.T / X.shape[1], K)[1], set())
    # uncentred GRM fit
    A, f, w, _ = W.grm_scaled_uncentred(G)
    L, S, r, pp, _ = DL.fit_loc(A, B.TAU, cand, W.noise_edge(f, w, N), K + 1, 4, kin_check=True)
    out["unc_fit"] = (I.cs_pcs(L, K)[1], {(min(i, j), max(i, j)) for i, j, _ in pp})
    # aarobust-kin
    _, C = B.grm(G)
    Lp, Sp, _ = I.pcp_kin(C, B.TAU, True, cand=cand)
    out["aarobust_kin"] = (I.top_eig(Lp, K)[1], pairs_from_S(Sp))
    # detect-white
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L2, S2, _, _ = I.lr_kin_fd_auto(H, B.TAU, D, I.cs_noise_edge(G), cand, rmax=K + 1)
    p2 = I.lr_kin_pairs(H, D, L2, S2)
    AM = G @ G.T / G.shape[1]
    out["detect_white"] = (I.whitened_noise(AM, np.diag(AM) - np.diag(L2), p2, K),
                           {(min(i, j), max(i, j)) for i, j, _ in p2})
    # dwg
    U, pd_, _ = DL.dwg_loc_ea(G, K, cand, 4, kin_check=True, k0_max=0.8)
    out["dwg"] = (U, {(min(i, j), max(i, j)) for i, j, _ in pd_})
    return out


if __name__ == "__main__":
    import pandas as pd
    out = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    jobs = [("hwe", n, s, r) for n in (5, 10, 20, 40) for s in HWE for r in range(10)]
    jobs += [("rel", n, s, r) for n in (5, 10, 20, 40) for s in REL for r in range(5)]
    with ProcessPoolExecutor(workers, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["group", "n", "scen", "rep", "method", "minR2", "err", "recall", "false_pairs",
                                    "n_true"])
    d.to_csv(out, sep="\t", index=False)
    d["fail"] = d.minR2 < 0.95
    order = ["standard", "unc_fit", "aarobust_kin", "detect_white", "dwg"]
    for g in ("hwe", "rel"):
        x = d[d.group == g]
        print(f"== {g}: runs wrong by scenario")
        print(x.pivot_table(index="method", columns="scen", values="fail").reindex(order).round(3).to_string())
        print(f"== {g}: by n (wrong / recall / false pairs / relatives' error)")
        for v in ("fail", "recall", "false_pairs", "err"):
            print(v)
            print(x.pivot_table(index="method", columns="n", values=v).reindex(order).round(3).to_string())
