#!/usr/bin/env python3
"""Haploid input against inbreeding and relatives: one allele drawn at random
from each genotype (a heterozygote gives 0 or 1 with probability 1/2), so the
input is 0/1 whatever the individual's inbreeding (Bernoulli(pi) for any F).
Same simulated designs, truth and scores as inbreed_pca_bench.py.

Changes for haploid data:
  - the GRM is scaled by f(1-f) (PCAone --haploid; ploidy 1);
  - a related pair's GRM entry is phi (diploid: 2 phi), so the kinship rule is
    phi = R_ij / sqrt(v_i v_j) > tau, i.e. the diploid rule with tau / 2;
  - the noise edge uses the haploid moments (E h^2 = f; Var(h_i h_j) = f^2 - f^4);
  - KING-robust needs heterozygous genotypes, so PCAone's candidate screen
    finds nothing, and it does not exist for pseudo-haploid data. The Python fit
    takes its candidates from a haploid evalAdmix screen iterated with the fit
    (hap_ea_iter; no genotypes needed), and for comparison from KING-robust on
    the diploid genotypes (> 0.04; only possible in a simulation), from every
    pair, or, as a best case, the true pairs.
  - haploid evalAdmix: residuals of the 0/1 alleles after the fit's structure
    axes; the induced correlation uses each individual's noise variance from
    the fitted allele frequencies, mean pi(1 - pi) (h(1 - h) is always 0 for
    haploid data), and the residual correlation of relatives is phi itself
    (diploid: 2 phi), so kinship = b - c (diploid evalAdmix: (b - c) / 2).

Methods: PCAone --haploid: standard, --impute-diag, --robust aarobust-kin,
--robust dwg; Python haploid dwg fit (rank from the haploid noise edge, at most
k + 1): PCs of its structure part (not whitened) and whitened (_w).

usage: inbreed_pca_hap.py <workdir> <out.tsv> [workers] [scenarios]
"""
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as DW  # noqa: E402
import illustrate as I  # noqa: E402
import inbreed_smooth_sim as SIM  # noqa: E402
from inbreed_pca_bench import K, SCEN, scores, true_pairs  # noqa: E402
from inbreeding_bench import PCAONE  # noqa: E402
from sim import write_bed  # noqa: E402

POP_MIN = 5  # population rule: complete cliques of at least 5 called individuals
PCAONE_METHODS = {"standard": [], "aarobust-diag": ["--impute-diag"], "aarobust-kin": ["--robust", "aarobust-kin"],
                  "dwg": ["--robust", "dwg"]}


def haploid(X, rng):
    return np.where(X == 1, rng.integers(0, 2, X.shape), X // 2).astype(np.int8)


def grm_hap(H):
    """uncentred GRM scaled by f(1-f), the allele frequencies and f(1-f)"""
    f = H.mean(0)
    ok = (f > 0) & (f < 1)
    w = f[ok] * (1 - f[ok])
    Z = H[:, ok] / np.sqrt(w)
    return Z @ Z.T / ok.sum(), f[ok], w


def noise_edge_hap(f, w, N):
    """as dw_grm.noise_edge with haploid moments: Var(h_i h_j) = f^2 - f^4"""
    M = len(f)
    return 2 * np.sqrt(np.sum((f ** 2 - f ** 4) / w ** 2) / M ** 2) * np.sqrt(N)


def evaladmix_kin_hap(H, U):
    """haploid evalAdmix kinship given structure axes U (N x r)"""
    N, M = H.shape
    V = np.c_[U, np.ones(N)]
    P = V @ np.linalg.pinv(V.T @ V) @ V.T
    IP = np.eye(N) - P
    hbar = H.mean(axis=1)
    b = I.cov2cor(IP @ (H @ H.T - M * np.outer(hbar, hbar)) @ IP)
    pi = np.clip(P @ H, 1e-3, 1 - 1e-3)
    c = I.cov2cor(IP @ np.diag((pi * (1 - pi)).mean(axis=1)) @ IP)
    kin = b - c
    np.fill_diagonal(kin, 0)
    return kin


def population_cliques(called, N, m, phi=None, margin=B.TAU / 2, comp_max=10):
    """groups of individuals in which every pair is called (complete cliques of
    the called pairs, connected components) with at least m members: taken for
    a population rather than a family (a family with both parents is never
    complete: the parents are not related). Returns their pairs, except pairs
    whose kinship phi[(i, j)] exceeds the group's median by more than margin
    (relatives within the population, e.g. an MZ pair). Also connected groups of
    at least comp_max individuals, complete or not (e.g. two sites whose shared
    ancestry the fit is missing: a block of between-site pairs)."""
    nb = [set() for _ in range(N)]
    for i, j in called:
        nb[i].add(j)
        nb[j].add(i)
    seen, out = set(), set()
    for s0 in range(N):
        if s0 in seen or not nb[s0]:
            continue
        comp, todo = {s0}, [s0]
        while todo:
            x = todo.pop()
            for y in nb[x] - comp:
                comp.add(y)
                todo.append(y)
        seen |= comp
        complete = all(len(nb[x] & comp) == len(comp) - 1 for x in comp)
        if (len(comp) >= m and complete) or (comp_max and len(comp) >= comp_max):
            pp = {(min(a, b), max(a, b)) for a in comp for b in nb[a] & comp}
            if phi:
                med = np.median([phi[q] for q in pp])
                pp = {q for q in pp if phi[q] <= med + margin}
            out |= pp
    return out


def hap_ea_iter(H, c=3, rounds=8, start="all", k=K, full=False, pop_min=None, union=False, rmax=None):
    """candidates from the haploid data only: a first fit with every pair as a
    candidate (phi_S rule; start="none": no candidates), then evalAdmix on the
    fit's axes that are not localised (n_eff = 1/sum u^4 >= c), pairs above 0.04
    as candidates, refit; repeat until the pairs are stable.
    pop_min: population rule, complete cliques of at least pop_min called
    individuals are taken for populations; their pairs are excluded from then on.
    union: the candidates of the next round are the pairs called by the fit plus
    the evalAdmix pairs (as PCAone's KING | evalAdmix), not evalAdmix alone"""
    N = len(H)
    off = ~np.eye(N, dtype=bool)
    cand = off.copy() if start == "all" else np.zeros((N, N), bool)
    banned = np.zeros((N, N), bool)
    key = None
    for _ in range(rounds):
        U, Uw, called, r, axes, extra = hap_fit(H, cand & ~banned, axes=True, k=k, rmax=rmax)
        if pop_min:
            pop = population_cliques(called, N, pop_min,
                                     {(min(i, j), max(i, j)): ph for i, j, ph in extra["pairs"]})
            if pop:
                for i, j in pop:
                    banned[i, j] = banned[j, i] = True
                key = None
                continue  # refit without them before the evalAdmix screen
        if called == key:
            break
        key = called
        if axes.shape[1]:
            neff = 1 / np.sum((axes / np.linalg.norm(axes, axis=0)) ** 4, axis=0)
            keep = axes[:, neff >= c] if (neff >= c).any() else axes[:, :1]
        else:
            keep = U[:, :1]
        ea = (evaladmix_kin_hap(H.astype(float), keep) > B.KING_SCREEN) & off
        if union:  # as PCAone: the fit's own pairs stay candidates, evalAdmix adds
            cand = ea.copy()
            for i, j in called:
                cand[i, j] = cand[j, i] = True
        else:
            cand = ea
        cand &= ~banned
    if full:  # also the noise v, the pairs with kinship and the final candidates
        return U, Uw, called, r, dict(extra, cand=cand)
    return U, Uw, called, r


def hap_fit(H, cand, axes=False, k=K, rmax=None):
    """rmax: highest rank of the fit (default k + 1, as dwg)"""
    A, f, w = grm_hap(H.astype(float))
    N = len(A)
    L, S, r, pairs = DW.fit(A, B.TAU / 2, cand, noise_edge_hap(f, w, N), rmax or k + 1, None)
    P = np.eye(N) - 1 / N
    U = np.linalg.eigh(P @ L @ P)[1][:, ::-1][:, :k]
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    # whitening: the fit reports R / (2 sqrt(v v)) (diploid scale) and the noise
    # correlation used is twice that, R / sqrt(v v): the haploid correlation
    Uw = I.whitened_noise(A, v, pairs, k, centre=True)
    out = (U, Uw, {(min(i, j), max(i, j)) for i, j, _ in pairs}, r)
    if axes:
        Lc = P @ L @ P
        ev, Vv = np.linalg.eigh(Lc)
        out += (Vv[:, ::-1][:, :max(r - 1, 0)],  # the structure axes (mean dropped)
                dict(v=v, pairs=pairs))
    return out


def hap_aarobust(H, cand, k=K, centred=False):
    """haploid aarobust-kin+diag: PCP with the kinship rule on the haploid scale
    (tau / 2) and the diagonal unobserved; on the centred haploid GRM or the
    uncentred one (PCs from the centred structure part)"""
    H = H.astype(float)
    N = len(H)
    P = np.eye(N) - 1 / N
    if centred:
        f = H.mean(0)
        X = (H - f) / np.sqrt(f * (1 - f))
        M = X @ X.T / H.shape[1]
    else:
        M = grm_hap(H)[0]
    L, S, _ = I.pcp_kin(M, B.TAU / 2, free_diag=True, cand=cand)
    U = np.linalg.eigh(P @ L @ P)[1][:, ::-1][:, :k]
    called = {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(np.abs(S) > 1e-12, 1)))}
    return U, called


def one(job):
    scen, n, seed, workdir = job
    X, pops, group, ex = SIM.build(seed, n, scen, full=True)
    Gall, lab = SIM._DATA["G"], SIM._DATA["lab"]
    used = ex["idx"][ex["idx"] >= 0]
    ref = np.setdiff1d(np.flatnonzero(np.isin(lab, SIM.POPS)), used)
    T = B.truth_reference(dict(G=ex["clean"].astype(float), ref_idx=ref, cols=ex["cols"]), Gall, K)
    H = haploid(X, np.random.default_rng(seed + 7919))
    keep = H.min(0) != H.max(0)
    H, T_ = H[:, keep], T
    n0 = 4 * n
    tp = true_pairs(group, pops, n0) if scen not in ("inb", "none") else set()
    rows = []

    def add(m, U, called):
        r2, mev, g = scores(U, T_, pops, group)
        rows.append(dict(scen=scen, n=n, seed=seed, method=m, minR2=r2, mev=mev, pairs=len(called),
                         true_found=len(called & tp), **{f"err_{x.replace(' ', '_')}": g[x] for x in g}))

    d = os.path.join(workdir, f"hap_{scen}_n{n}_s{seed}")
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, "h")
    write_bed(pre, [[pops[i], f"I{i}", "0", "0", "0", "-9"] for i in range(len(H))],
              [f"1\ts{j}\t0\t{j + 1}\tA\tC" for j in range(H.shape[1])], (2 * H).astype(np.int8))
    for m, extra in PCAONE_METHODS.items():
        out = os.path.join(d, m)
        r = subprocess.run([PCAONE, "-b", pre, "--haploid", "-k", str(K), "-n", "1", "-o", out] + extra,
                           capture_output=True, text=True)
        if r.returncode != 0:
            continue
        called = set()
        if os.path.exists(out + ".relpairs"):
            for l in list(open(out + ".relpairs"))[1:]:
                x, y = (int(w[1:]) for w in l.split()[:2])
                called.add((min(x, y), max(x, y)))
        add("hap " + m, np.loadtxt(out + ".eigvecs")[:, :K], called)
    shutil.rmtree(d)
    N = len(H)
    U, Uw, called, _ = hap_fit(H, np.ones((N, N), bool))
    add("hap dwg-fit, all pairs", U, called)
    add("hap dwg-fit, all pairs, whitened", Uw, called)
    U, Uw, called, _, ex1 = hap_ea_iter(H, full=True)
    add("hap dwg-fit, haploid evalAdmix", U, called)
    add("hap dwg-fit, haploid evalAdmix, whitened", Uw, called)
    U, Uw, called, _, ex2 = hap_ea_iter(H, full=True, pop_min=POP_MIN)
    add("hap dwg-fit, haploid evalAdmix + population rule", U, called)
    add("hap dwg-fit, haploid evalAdmix + population rule, whitened", Uw, called)
    U, Uw, called, _, ex3 = hap_ea_iter(H, full=True, pop_min=POP_MIN, union=True)
    add("hap dwg-fit, union screen + population rule", U, called)
    add("hap dwg-fit, union screen + population rule, whitened", Uw, called)
    for nm, cnd in (("screen", ex1["cand"]), ("screen + population rule", ex2["cand"]),
                    ("union screen + population rule", ex3["cand"])):
        for cen in (True, False):
            Ua, ca = hap_aarobust(H, cnd, centred=cen)
            add(f"hap aarobust, {'centred' if cen else 'uncentred'}, {nm}", Ua, ca)
    tc = np.zeros((N, N), bool)
    for i, j in tp:
        tc[i, j] = tc[j, i] = True
    for cen in (True, False):
        Ua, ca = hap_aarobust(H, tc, centred=cen)
        add(f"hap aarobust, {'centred' if cen else 'uncentred'}, true pairs", Ua, ca)
    kin = B.king_robust(X)
    U, Uw, called, _ = hap_fit(H, kin > 0.04)
    add("hap dwg-fit, diploid KING", U, called)
    add("hap dwg-fit, diploid KING, whitened", Uw, called)
    cand = np.zeros((N, N), bool)
    for i, j in tp:
        cand[i, j] = cand[j, i] = True
    U, Uw, called, _ = hap_fit(H, cand)
    add("hap dwg-fit, true pairs", U, called)
    add("hap dwg-fit, true pairs, whitened", Uw, called)
    return rows


def main():
    workdir, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    scen = sys.argv[4].split(",") if len(sys.argv) > 4 else SCEN
    jobs = [(s, n, seed, workdir) for s in scen for n in (5, 10) for seed in range(100, 110)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(out, sep="\t", index=False)
    for v in ("minR2", "err_relative", "err_inbred", "err_excess_het", "err_other", "true_found", "pairs"):
        print("==", v)
        print(d.pivot_table(index="method", columns=["scen", "n"], values=v).round(2).to_string())


if __name__ == "__main__":
    main()
