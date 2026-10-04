#!/usr/bin/env python3
"""Undo the centring of the GRM and fit it: from the centred, scaled GRM C,
the allele frequencies f and the per-individual cross term u = X_c mu (one
number per individual), restore the uncentred scaled Gram

    A_s = C + u 1' + 1 u' + (mu'mu) 1 1'     (X_c = G~ - 1 mu', G~ = G W^-1/2)

then fit it with the diagonal free and the kinship rule (rank from the noise
edge, at most k + 1, mean component dropped). Compared with aarobust-kin
(--impute-diag) and detect-white on the small-N grid against the reference
truth (benchmark.py)."""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402

SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "inbredpop_mz2", "err1_mz2"]


def one(job):
    n, scen, rep, k = job
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(float)
    N, _ = G.shape
    rel = np.arange(ds["n0"], N)
    tau = B.TAU
    king = B.king_robust(G)
    cand = king > B.KING_SCREEN
    np.fill_diagonal(cand, False)
    f = G.mean(0) / 2
    ok = (f > 0) & (f < 1)
    M = int(ok.sum())
    sw = np.sqrt(2 * f[ok] * (1 - f[ok]))
    Gt = G[:, ok] / sw                      # scaled, uncentred
    mu = 2 * f[ok] / sw                     # column means of Gt
    Xc = Gt - mu                            # what centred data gives
    # --- inputs available from centred data: C, f, u
    C = Xc @ Xc.T / M
    u = Xc @ mu / M
    As = C + u[:, None] + u[None, :] + (mu @ mu) / M
    restore_err = np.abs(As - Gt @ Gt.T / M).max()
    # CS-type matrix on the restored scale: diagonal noise D_s = mean g(2-g)/w
    Ds = (G[:, ok] * (2 - G[:, ok]) / sw ** 2).mean(1)
    Hs = As - np.diag(Ds)
    fo = f[ok]
    eg2 = 2 * fo * (1 - fo) + 4 * fo ** 2
    edge = 2 * np.sqrt(np.sum((eg2 ** 2 - 16 * fo ** 4) / sw ** 4) / M ** 2) * np.sqrt(N)
    L, S, _, r = I.lr_kin_fd_auto(Hs, tau, Ds, edge, cand, rmax=k + 1)
    pp = I.lr_kin_pairs(Hs, Ds, L, S)
    res = {"restored_fit": I.cs_pcs(L, k)[1],
           "restored_white": I.whitened_noise(As, np.diag(As) - np.diag(L), pp, k)}
    # PCs of the restored Gram with the related-pair entries imputed from L;
    # diagonal imputed (L), CS-corrected (A - D) or observed; mean dropped
    off = (np.abs(S) > 1e-12) & ~np.eye(N, dtype=bool)
    for name, diag in [("imp_diagL", np.diag(L)), ("imp_diagCS", np.diag(Hs)), ("imp_diagobs", np.diag(As))]:
        Mi = np.where(off, L, Hs)
        np.fill_diagonal(Mi, diag)
        res[name] = I.cs_pcs(Mi, k)[1]
    # references
    _, Cg = B.grm(G)
    Lp, _, _ = I.pcp_kin(Cg, tau, True, cand=cand)
    res["aarobust_kin"] = I.top_eig(Lp, k)[1]
    H = I.cs_matrix(G)
    D = (G * (2 - G)).mean(1)
    L2, S2, _, _ = I.lr_kin_fd_auto(H, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
    pp2 = I.lr_kin_pairs(H, D, L2, S2)
    AM = G @ G.T / G.shape[1]
    res["detect_white"] = I.whitened_noise(AM, np.diag(AM) - np.diag(L2), pp2, k)
    res["standard"] = I.top_eig(Cg, k)[1]
    rows = []
    for m, U in res.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        rows.append((n, scen, rep, k, m, r2.min(), err, r, restore_err))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    jobs = [(n, s, r, k) for k in (3, 10) for n in (5, 10, 20, 40) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(40, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err", "rank", "restore_err"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    print("max |restored - direct uncentred Gram|:", d.restore_err.max())
    order = ["standard", "aarobust_kin", "detect_white", "restored_fit", "restored_white", "imp_diagL", "imp_diagCS",
             "imp_diagobs"]
    for k in (3, 10):
        x = d[d.k == k]
        t = x.pivot_table(index="method", columns="n", values="minR2").reindex(order).round(3)
        fl = x.assign(fl=x.minR2 < 0.95).pivot_table(index="method", columns="n", values="fl").reindex(order).round(3)
        e = x.groupby("method")["err"].mean().reindex(order).round(3)
        print(f"== k = {k}")
        print(pd.concat({"mean min R2": t, "failures": fl}, axis=1))
        print("relatives' error:", e.to_dict())
        print("restored rank by n:", x[x.method == "restored_fit"].groupby("n")["rank"].mean().round(2).to_dict())
