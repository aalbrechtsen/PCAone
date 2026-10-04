#!/usr/bin/env python3
"""dwg_v with a localisation rule in the rank stepping: an eigenvector above
the noise edge that rests on fewer than c individuals (n_eff = 1/sum u^4) is
not taken as structure; the pairs among its top individuals become candidates
and the fit is repeated at the same rank; if no new pair can be added (e.g.
first cousins, below tau) the rank is not raised. Optionally followed by the
k-free evalAdmix screen iteration (dwg_ea_iter.py)."""
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
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402
from dwg_ea_iter import dwg_fit, ea_iter  # noqa: E402

SCEN = ["xanc", "grandx", "avuncx", "halfsibx", "none", "mz2", "nuclear2", "childx", "cousins", "many",
        "inbredpop_mz2", "err1_mz2"]


TAU3 = 2 ** -4.5


def family_axis(u, R, v, c, kin_check):
    """is the axis u a family axis? n_eff < c and (if kin_check) at least one
    pair among its top individuals has structure-adjusted kinship > 2^-4.5
    (R must be the residual without this axis)"""
    u = u / np.linalg.norm(u)
    neff = 1 / np.sum(u ** 4)
    top = np.argsort(-np.abs(u))[:max(2, int(np.ceil(neff)))]
    if neff >= c:
        return False, top
    if not kin_check:
        return True, top
    phi = I.kin_scale(R, v)[np.ix_(top, top)]
    return phi[np.triu_indices(len(top), 1)].max() > TAU3, top


def fit_loc(A, tau, cand, edge, rmax, c, iters=500, kin_check=False):
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
            v = np.maximum(np.diag(R), 1e-6)
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
        fam, top = family_axis(V[:, r], A - L, np.maximum(np.diag(A - L), 1e-6), c, kin_check)
        if fam:
            if not kin_check:
                top = np.argsort(-np.abs(V[:, r]))[:max(2, len(top) + 1)]
            new = [(i, j) for a, i in enumerate(top) for j in top[a + 1:] if not cand[i, j]]
            if not new:
                break  # localised and nothing to add: not structure, stop here
            for i, j in new:
                cand[i, j] = cand[j, i] = True
            continue
        r += 1
    v = np.maximum(np.diag(A - L), 1e-6)
    phi = I.kin_scale(A - L, v)
    iu = np.triu_indices(n, 1)
    pairs = [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]
    return L, S, r, pairs, cand


def dwg_loc(G, k, cand, c, kin_check=False):
    A, f, w, Gs = W.grm_scaled_uncentred(G)
    N = len(A)
    L, S, r, pairs, cand2 = fit_loc(A, B.TAU, cand, W.noise_edge(f, w, N), k + 1, c, kin_check=kin_check)
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    U = I.whitened_noise(A, v, pairs, k)
    ev, V = I.top_eig(L, r)
    # structure axes for the evalAdmix screen: drop family axes (kinship from
    # the residual with that axis put back)
    keep = []
    for j in range(1, r):
        Rj = A - L + ev[j] * np.outer(V[:, j], V[:, j])
        fam, _ = family_axis(V[:, j], Rj, np.maximum(np.diag(Rj), 1e-6), c, kin_check)
        if not fam:
            keep.append(j)
    return U, V[:, keep] if keep else V[:, 1:2], pairs, r


def k0_moment(G, axes, pairs):
    """moment estimate of k0 per pair: observed / expected opposite
    homozygotes, individual allele frequencies from [axes, 1] with the pair
    left out of the regression"""
    N = len(G)
    V = np.c_[axes, np.ones(N)]
    VtV = V.T @ V
    R = V.T @ G  # (m+1) x M
    out = {}
    for i, j in pairs:
        A_ = VtV - np.outer(V[i], V[i]) - np.outer(V[j], V[j])
        Ai = np.linalg.pinv(A_)
        Rl = R - np.outer(V[i], G[i]) - np.outer(V[j], G[j])
        pi = np.clip(0.5 * (V[i] @ Ai @ Rl), 1e-3, 1 - 1e-3)
        pj = np.clip(0.5 * (V[j] @ Ai @ Rl), 1e-3, 1 - 1e-3)
        obs = np.sum((G[i] == 0) & (G[j] == 2)) + np.sum((G[i] == 2) & (G[j] == 0))
        exp = np.sum(pi ** 2 * (1 - pj) ** 2 + (1 - pi) ** 2 * pj ** 2)
        out[(i, j)] = obs / max(exp, 1e-12)
    return out


def dwg_loc_ea(G, k, ck, c, rounds=5, kin_check=False, k0_max=None):
    off = ~np.eye(len(G), dtype=bool)
    U, axes, pairs, r = dwg_loc(G, k, ck, c, kin_check)
    key = {(min(i, j), max(i, j)) for i, j, _ in pairs}  # stop when a refit does not change the pairs
    for _ in range(rounds):
        ea = I.evaladmix_kin(G, axes, intercept=True)
        ce = (ea > B.KING_SCREEN) & off & ~ck
        if k0_max is not None:
            # evalAdmix-only candidates must also look IBD-related (k0 < k0_max)
            cand_e = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(ce, 1)))]
            k0 = k0_moment(G, axes, cand_e)
            ce = np.zeros_like(ce)
            for (i, j), v_ in k0.items():
                if v_ < k0_max:
                    ce[i, j] = ce[j, i] = True
        U, axes, pairs, r = dwg_loc(G, k, ck | ce, c, kin_check)
        new = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        if new == key:
            break
        key = new
    return U, pairs, r


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
    N = len(G)
    rel = np.arange(ds["n0"], N)
    off = ~np.eye(N, dtype=bool)
    true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(ds["Kped"] >= B.TAU, 1)))}
    ck = (B.king_robust(G) > B.KING_SCREEN) & off
    out = {}
    U1, _, p1, r1 = dwg_fit(G, k, ck)
    out["king"] = (U1, p1, r1)
    ea = I.evaladmix_kin(G, U1[:, :3], intercept=True)
    U2, _, p2, r2_ = dwg_fit(G, k, ck | ((ea > B.KING_SCREEN) & off))
    out["ea_trueK"] = (U2, p2, r2_)
    out["ea_iter"] = ea_iter(G, k, ck, 3)
    out["loc4_ea_iter"] = dwg_loc_ea(G, k, ck, 4)
    out["lockin4_ea_iter"] = dwg_loc_ea(G, k, ck, 4, kin_check=True)
    out["lockin4_ea_k0"] = dwg_loc_ea(G, k, ck, 4, kin_check=True, k0_max=0.8)
    out["lockin4_ea_k07"] = dwg_loc_ea(G, k, ck, 4, kin_check=True, k0_max=0.7)
    rows = []
    for m, (U, pairs, r) in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        det = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        rec = len(det & true) / len(true) if true else np.nan
        rows.append((n, scen, rep, k, m, r2.min(), err, rec, len(det - true), r))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 240)
    jobs = [(n, s, r, k) for k in (3, 10) for n in (5, 10, 20, 40) for s in SCEN for r in range(5)]
    with ProcessPoolExecutor(44, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "k", "method", "minR2", "err", "recall", "false_pairs", "rank"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    d["fail"] = d.minR2 < 0.95
    order = ["king", "ea_trueK", "lockin4_ea_k0", "lockin4_ea_k07"]
    for k in (3, 10):
        x = d[d.k == k]
        t = pd.concat({"fail": x.pivot_table(index="method", columns="n", values="fail").reindex(order),
                       "recall": x.pivot_table(index="method", columns="n", values="recall").reindex(order),
                       "false": x.pivot_table(index="method", columns="n", values="false_pairs").reindex(order),
                       "rank": x.pivot_table(index="method", columns="n", values="rank").reindex(order)}, axis=1).round(3)
        print(f"== k = {k}")
        print(t.to_string())
        print("relatives' error:", x.groupby("method")["err"].mean().reindex(order).round(3).to_dict())
    x = d[d.k == 10]
    print("== k = 10 failures by scenario")
    print(x.pivot_table(index="method", columns="scen", values="fail").reindex(order).round(2).to_string())
