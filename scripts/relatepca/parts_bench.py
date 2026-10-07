#!/usr/bin/env python3
"""Small-N benchmark of the --robust methods as combinations of parts
(report/method_parts.md), built from the existing prototypes:

  diag    topr  fit of the top r PCs, diagonal free, rank from the noise edge
                (<= k+1)                                    dw_grm.fit / dwg_loc.fit_loc
          pcp   nuclear norm with the kinship rule           illustrate.pcp_kin
  matrix  raw   raw Gram G G'/M (= the CS matrix with the diagonal free)
          unc   uncentred, SNP-scaled GRM                   dw_grm.grm_scaled_uncentred
          cgrm  centred GRM C (pcp only; top r fails centred)
  cand    k     KING-robust > 0.04
          kf    k + family-axis rule (topr only)            dwg_loc.family_axis
          kfe   kf + evalAdmix from the structure axes, iterated (dwg_loc_ea)
          ke    k + evalAdmix (pcp: axes = top k PCs of L)
  k0      0/1   evalAdmix-only candidates need k0 < 0.8     dwg_loc.k0_moment
  scale   v     kinship scaled by the fitted noise v = A_ii - L_ii
          D     scaled by heterozygosity on the matrix scale
  final   L     PCs of L (mean PC dropped if uncentred)
          white whitening, mean PC dropped                  illustrate.whitened_noise
          whitec whitening, centred (as dwg)

A combination is written diag/matrix/cand/k0/scale/final, e.g. dwg =
topr/unc/kfe/1/v/whitec. "standard" is standard PCA.

usage: parts_bench.py --design check|S1|S2 --out FILE [--workers 80]
"""
import argparse
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as W  # noqa: E402
import dwg_loc as DL  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402

# panels: bed file, populations, number of true ancestry axes
PANELS = {"admix": ("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",
                    "CEU,CHB,MXL,YRI".split(","), 3),
          "world": ("/kellyData/home/albrecht/codex/relatePCA/data/world/world100k",
                    "CEU,CHB,YRI,ASW".split(","), 2)}  # ASW lies on the CEU-YRI cline
BFILE, POPS, KTRUE = PANELS["admix"]


def init(panel):
    global BFILE, POPS, KTRUE
    BFILE, POPS, KTRUE = PANELS[panel]
    Gb.init(BFILE)
FAM_C = 4  # family axis: n_eff < 4


def build_matrix(G, mat):
    """matrix, its noise edge, heterozygosity on its scale, centred?"""
    N, M = G.shape
    if mat == "raw":
        return G @ G.T / M, I.cs_noise_edge(G), (G * (2 - G)).mean(1), False
    if mat == "unc":
        A, f, w, Gs = W.grm_scaled_uncentred(G)
        return A, W.noise_edge(f, w, N), (Gs * (2 - Gs) / w).mean(1), False
    if mat == "cgrm":
        _, C = B.grm(G)
        return C, 1.1 * 2 * np.sqrt(N / M) * np.diag(C).mean(), None, True
    raise ValueError(mat)


def fit_topr(A, tau, cand, edge, rmax, scale, fam, force=None):
    """top-r fit with the diagonal free (dwg_loc.fit_loc; scale None = fitted
    v, else a fixed vector; fam False = no family-axis rule, as dw_grm.fit)"""
    n = len(A)
    eye = np.eye(n, dtype=bool)
    cand = cand.copy()
    S = np.zeros_like(A)
    r = 1
    while True:
        L_old = None
        for _ in range(500):
            w_, V = I.top_eig(A - S, r)
            L = (V * w_) @ V.T
            R = A - L
            v = np.maximum(np.diag(R), 1e-6) if scale is None else scale
            sel = I.kin_select(R, v, tau) & cand
            if force is not None:  # pairs always in S (imputed whatever their kinship)
                sel |= force
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
        if fam:
            v = np.maximum(np.diag(A - L), 1e-6) if scale is None else scale
            isfam, top = DL.family_axis(V[:, r], A - L, v, FAM_C, True)
            if isfam:
                new = [(i, j) for a, i in enumerate(top) for j in top[a + 1:] if not cand[i, j]]
                if not new:
                    break
                for i, j in new:
                    cand[i, j] = cand[j, i] = True
                continue
        r += 1
    return L, S, r


def pairs_of(A, L, S, v):
    phi = I.kin_scale(A - L, v)
    iu = np.triu_indices(len(A), 1)
    return [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]


def run_combo(G, combo, k, ck):
    """(top k PCs, detected pairs, rank) of one combination"""
    diag, mat, cand, k0, scale, final = combo.split("/")
    A, edge, D, centred = build_matrix(G, mat)
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    fixed = D if scale == "D" else None

    def fit(cm):
        if diag == "topr":
            L, S, r = fit_topr(A, B.TAU, cm, edge, k + 1, fixed, "f" in cand)
        else:
            L, S, _ = I.pcp_kin(A, B.TAU, True, D=fixed, cand=cm)
            r = None
        v = np.maximum(np.diag(A - L), 1e-6) if fixed is None else fixed
        return L, S, r, pairs_of(A, L, S, v)

    def topr_axes(A_, L, r):
        """as dwg_loc.dwg_loc: eigenvectors 2..r of L without family axes"""
        ev, V = I.top_eig(L, r)
        keep = []
        for j in range(1, r):
            Rj = A_ - L + ev[j] * np.outer(V[:, j], V[:, j])
            isfam, _ = DL.family_axis(V[:, j], Rj, np.maximum(np.diag(Rj), 1e-6), FAM_C, True)
            if not isfam:
                keep.append(j)
        return V[:, keep] if keep else V[:, 1:2]

    def axes_of(L, r):
        """structure axes for the evalAdmix screen"""
        if diag == "topr":
            return topr_axes(A, L, r)
        if "E" in cand:  # PCP: screen axes from a top-r fit of the uncentered GRM (family axes dropped)
            Au, eu, _, _ = build_matrix(G, "unc")
            Lu, _, ru = fit_topr(Au, B.TAU, ck, eu, k + 1, None, False)
            return topr_axes(Au, Lu, ru)
        nax = k
        _, V = I.top_eig(L, nax + (0 if centred else 1))
        return V if centred else V[:, 1:]

    L, S, r, pairs = fit(ck)
    if "e" in cand or "E" in cand:
        key = {(i, j) for i, j, _ in pairs}
        for _ in range(5):
            axes = axes_of(L, r)
            ea = I.evaladmix_kin(G, axes, intercept=True)
            ce = (ea > B.KING_SCREEN) & off & ~ck
            if k0 == "1":
                cl = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(ce, 1)))]
                ce = np.zeros_like(ce)
                for (i, j), x in DL.k0_moment(G, axes, cl).items():
                    if x < 0.8:
                        ce[i, j] = ce[j, i] = True
            L, S, r, pairs = fit(ck | ce)
            new = {(i, j) for i, j, _ in pairs}
            if new == key:
                break
            key = new
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    if final == "L":
        U = I.top_eig(L, k)[1] if centred else I.cs_pcs(L, k)[1]
    else:
        U = I.whitened_noise(A, v, pairs, k, centre=final == "whitec")
    return U, {(i, j) for i, j, _ in pairs}, r


def k0_ml(G, axes, pairs, iters=300, eps=0.01):
    """maximum likelihood k0 per pair (EM over k0, k1, k2), with the same
    individual allele frequencies as k0_moment (axes, pair left out). Per SNP:
    IBD0 both HWE(pi), HWE(pj); IBD1 one shared allele drawn with the mean of
    pi and pj, the other alleles from pi and pj; IBD2 g_i = g_j ~ HWE(mean).
    eps of the IBD0 likelihood is mixed into IBD1/IBD2 (genotype errors)."""
    N = len(G)
    V = np.c_[axes, np.ones(N)]
    VtV = V.T @ V
    R = V.T @ G

    def hwe(g, p):
        return np.where(g == 0, (1 - p) ** 2, np.where(g == 1, 2 * p * (1 - p), p ** 2))

    def a1(x, p):  # one allele: x copies (0/1) of the counted allele
        return np.where(x == 1, p, np.where(x == 0, 1 - p, 0.0))

    out = {}
    for i, j in pairs:
        A_ = VtV - np.outer(V[i], V[i]) - np.outer(V[j], V[j])
        Ai = np.linalg.pinv(A_)
        Rl = R - np.outer(V[i], G[i]) - np.outer(V[j], G[j])
        p = np.clip(0.5 * (V[i] @ Ai @ Rl), 1e-3, 1 - 1e-3)
        q = np.clip(0.5 * (V[j] @ Ai @ Rl), 1e-3, 1 - 1e-3)
        s = (p + q) / 2
        gi, gj = G[i], G[j]
        P0 = hwe(gi, p) * hwe(gj, q)
        P1 = s * a1(gi - 1, p) * a1(gj - 1, q) + (1 - s) * a1(gi, p) * a1(gj, q)
        P2 = np.where(gi == gj, hwe(gi, s), 0.0)
        Lk = np.c_[P0, (1 - eps) * P1 + eps * P0, (1 - eps) * P2 + eps * P0]
        kk = np.array([0.5, 0.3, 0.2])
        for _ in range(iters):
            w = Lk * kk
            w /= w.sum(1, keepdims=True)
            kn = w.mean(0)
            if np.abs(kn - kk).max() < 1e-6:
                kk = kn
                break
            kk = kn
        out[(i, j)] = kk[0]
    return out


def run_dec(G, combo, k, ck, king):
    """candidates and decision as separate parts:
    dec:imputation/matrix/candidates/decision/k0/final
      candidates  k = KING > 0.04; e = evalAdmix > 0.04; ke = both; all = all pairs
      decision    R = residual kinship phi_R > tau (the fit's own rule);
                  EA = evalAdmix kinship > tau; KING = KING >= tau;
                  KEA = KING >= tau (fixed before the first imputation) plus
                  evalAdmix > tau for the rest (candidates: all pairs)
                  FR = pairs with KING > FORCE (0.06) always imputed (forced into
                  S); the fit decides among KING / evalAdmix candidates
                  FKEA = KEA with KING > FORCE masked from the start; forced
                  pairs count as related only if KING >= tau
                  K0 = k0 < 0.8 alone decides among the candidates (ke: KING
                  > 0.04 or evalAdmix > 0.04)
      k0          1: a pair decided by evalAdmix (or proposed by it) also needs
                  k0_mom < 0.8; 2: the same with k0_ML (maximum likelihood)
      candidates "all" with decision R: every pair may enter S (the fit decides)
    When the decision is EA or KING the imputation runs with a fixed mask of
    the decided pairs (fit with tau = -inf on the mask), alternating with the
    decision until the mask is stable (<= 8 rounds). The structure axes for
    evalAdmix: a top-r fit of the uncentered GRM with the current mask, family
    axes dropped (as kE). Sigma in the whitening uses the deciding kinship."""
    imp, mat, cand, dec, k0, final = combo.split(":")[1].split("/")
    A, edge, D, centred = build_matrix(G, mat)
    Au, eu, _, _ = build_matrix(G, "unc")
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    NEG = -1e9

    def impute(mask, tau=NEG, force=None):
        if imp == "topr":
            L, S, r = fit_topr(A, tau, mask, edge, k + 1, None, False, force)
        else:
            L, S, _ = I.pcp_kin(A, tau, True, cand=mask, force=force)
        return L, S

    def axes(mask):
        Lu, _, ru = fit_topr(Au, NEG, mask, eu, k + 1, None, False)
        ev, V = I.top_eig(Lu, ru)
        keep = []
        for j in range(1, ru):
            Rj = Au - Lu + ev[j] * np.outer(V[:, j], V[:, j])
            isfam, _ = DL.family_axis(V[:, j], Rj, np.maximum(np.diag(Rj), 1e-6), FAM_C, True)
            if not isfam:
                keep.append(j)
        return V[:, keep] if keep else V[:, 1:2]

    def k0_ok(ax, M_):
        cl = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(M_, 1)))]
        out = np.zeros_like(M_)
        est = k0_ml if k0 == "2" else DL.k0_moment
        for (i, j), x in est(G, ax, cl).items():
            if x < 0.8:
                out[i, j] = out[j, i] = True
        return out

    report = None  # forced variants: masked pairs count as related only if their kinship > tau
    if dec == "FR":  # KING > FORCE always imputed; KING / evalAdmix candidates, the fit decides the rest
        force = (king > FORCE) & off
        kc = ck | force
        L, S = impute(kc, B.TAU, force)
        Sm = np.abs(S) > 1e-12
        for _ in range(8):
            ax = axes(Sm & off)
            ce = (I.evaladmix_kin(G, ax, True) > B.KING_SCREEN) & off
            if k0 in ("1", "2"):
                ce = k0_ok(ax, ce & ~kc)
            L, S = impute(kc | ce, B.TAU, force)
            Sm2 = np.abs(S) > 1e-12
            if (Sm2 == Sm).all():
                break
            Sm = Sm2
        R = A - L
        phi = I.kin_scale(R, np.maximum(np.diag(R), 1e-6))
        mask = (np.abs(S) > 1e-12) & off
        report = mask & (~force | (phi > B.TAU))
        dec = "done"
    kc = ck if cand in ("k", "ke") else off.copy() if cand == "all" else np.zeros_like(ck)
    if dec == "KING":
        mask = (king >= B.TAU) & ck
        L, S = impute(mask)
        phi = king
    elif dec == "R":  # candidates (KING and/or evalAdmix), the fit decides
        mask = kc.copy()
        L, S = impute(np.zeros_like(ck), B.TAU) if cand == "e" else impute(kc, B.TAU)
        Sm = np.abs(S) > 1e-12
        for _ in range(8):
            if "e" not in cand:
                break
            ax = axes(Sm & off)
            ce = (I.evaladmix_kin(G, ax, True) > B.KING_SCREEN) & off
            if k0 in ("1", "2"):
                ce = k0_ok(ax, ce & ~kc)
            new = kc | ce
            L, S = impute(new, B.TAU)
            Sm2 = np.abs(S) > 1e-12
            if (Sm2 == Sm).all():
                break
            Sm = Sm2
        R = A - L
        phi = I.kin_scale(R, np.maximum(np.diag(R), 1e-6))
        mask = (np.abs(S) > 1e-12) & off
    elif dec == "done":
        pass
    elif dec in ("KEA", "FKEA"):  # KING >= tau (FKEA: > FORCE) masked from the start; evalAdmix > tau adds the rest
        kmask = ((king >= B.TAU) if dec == "KEA" else (king > FORCE)) & off
        mask = kmask.copy()
        for _ in range(8):
            ax = axes(mask)
            ea = I.evaladmix_kin(G, ax, True)
            add = (ea > B.TAU) & off & ~kmask
            if k0 in ("1", "2"):
                add = k0_ok(ax, add)
            new = kmask | add
            if (new == mask).all():
                break
            mask = new
        L, S = impute(mask)
        phi = np.where(kmask, king, ea)
        if dec == "FKEA":
            report = mask & (~kmask | (king >= B.TAU))
    elif dec == "K0":  # k0 < 0.8 alone decides among KING / evalAdmix candidates
        mask = np.zeros_like(ck)
        for _ in range(8):
            ax = axes(mask)
            ea = I.evaladmix_kin(G, ax, True)
            new = k0_ok(ax, (ck | (ea > B.KING_SCREEN)) & off)
            if (new == mask).all():
                break
            mask = new
        L, S = impute(mask)
        phi = np.maximum(ea, 0)
    else:  # EA decides
        mask = np.zeros_like(ck)
        for _ in range(8):
            ax = axes(mask)
            ea = I.evaladmix_kin(G, ax, True)
            new = (ea > B.TAU) & off
            if cand == "k":
                new &= ck
            elif cand == "ke":
                new &= ck | (ea > B.KING_SCREEN)
            if k0 in ("1", "2"):
                new = k0_ok(ax, new)
            if (new == mask).all():
                break
            mask = new
        L, S = impute(mask)
        phi = ea
    iu = np.triu_indices(N, 1)
    pairs = [(int(i), int(j), float(phi[i, j])) for i, j in zip(*iu) if mask[i, j]]
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    if final == "L":
        U = I.top_eig(L, k)[1] if centred else I.cs_pcs(L, k)[1]
    else:
        U = I.whitened_noise(A, v, pairs, k, centre=final == "whitec")
    if report is not None:
        return U, {(i, j) for i, j, _ in pairs if report[i, j]}, None
    return U, {(i, j) for i, j, _ in pairs}, None


def standard(G, k):
    X = I.standardize(G)
    X = X[:, np.isfinite(X).all(0)]
    return I.top_eig(X @ X.T / X.shape[1], k)[1]


def dataset(n, M, scen, rep):
    seed = zlib.crc32(f"{','.join(POPS)}|{n}|{M}|{scen}|{rep}".encode())
    if "ASW" in POPS:  # 1000 Genomes panel: scenarios of world_test.py (incl. ASW families)
        from world_test import SCEN as WSCEN
        spec = WSCEN.get(scen, B.SCENARIOS.get(scen))
    else:
        spec = B.SCENARIOS[scen]
    return B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, POPS, n, M, spec, seed)


def one(job):
    design, n, M, scen, rep, k, combos = job
    small = None
    if ":" in scen:  # e.g. none:MXL2 = population MXL cut to 2 individuals
        scen, sp = scen.split(":")
        small = (sp[:-1], int(sp[-1]))
    try:
        ds = dataset(n, M, scen, rep)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, KTRUE)
    G = ds["G"].astype(float)
    Kp = ds["Kped"]
    n0 = ds["n0"]
    if small:  # keep the first small[1] base individuals of that population
        bp = np.array(ds["base_pop"])
        keep_b = [i for i in range(n0) if bp[i] != small[0] or (bp[:i] == small[0]).sum() < small[1]]
        keep = np.r_[keep_b, np.arange(n0, len(G))].astype(int)
        G, T, Kp = G[keep], T[keep], Kp[np.ix_(keep, keep)]
        n0 = len(keep_b)
        scen = f"{scen}:{small[0]}{small[1]}"
    N = len(G)
    rel = np.arange(n0, N)
    true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(Kp >= B.TAU, 1)))}
    off = ~np.eye(N, dtype=bool)
    t = time.time()
    king = B.king_robust(G)
    ck = (king > B.KING_SCREEN) & off
    t_king = time.time() - t
    rows = []
    for c in combos:
        t = time.time()
        if c == "standard":
            U, det, r = standard(G, k), set(), None
        elif c.startswith("dec:"):
            U, det, r = run_dec(G, c, k, ck, king)
        else:
            U, det, r = run_combo(G, c, k, ck)
        sec = time.time() - t + (0 if c == "standard" else t_king)
        r2, err = B.score(U, T, n0, rel, KTRUE)
        rec = len(det & true) / len(true) if true else np.nan
        rows.append((design, n, M, scen, rep, k, N, c, r2.min(), err, rec, len(det - true), len(true), r, sec,
                     ",".join(f"{x:.4f}" for x in r2)))
    return rows


# ---- S1k: the number of structure axes unknown / hard to infer -------------
# bn:      6 populations (star tree, Balding-Nichols), F_ST from strong to
#          below the noise edge; disc = n per population, cont = 6n
#          individuals with Dirichlet(0.3) ancestry (continuous structure).
#          Truth = top 5 PCs of the standardized expected genotypes 2Pi; an
#          axis is detectable when its signal eigenvalue exceeds the BBP
#          threshold sqrt(N/M) of the standardized GRM.
# hapnest: 6 ancestries (3 admixed) with realistic LD, n per ancestry from
#          the pool; truth = PCA of the 3600-person reference panel (5 axes).
BN_FST = [0.08, 0.05, 0.03, 0.015, 0.008, 0.004]
HAP = {}


def sim_bn(n, M, mode, rep):
    rng = np.random.default_rng(zlib.crc32(f"bn|{n}|{M}|{mode}|{rep}".encode()))
    K = len(BN_FST)
    pa = rng.uniform(0.05, 0.95, M)
    P = np.array([rng.beta(pa * (1 - F) / F, (1 - pa) * (1 - F) / F) for F in BN_FST])  # K x M
    if mode == "disc":
        Q = np.repeat(np.eye(K), n, axis=0)
    elif mode == "small":  # population 2 (F_ST 0.05) with only 2 individuals
        Q = np.repeat(np.eye(K), [n, 2] + [n] * (K - 2), axis=0)
    else:
        Q = rng.dirichlet(np.full(K, 0.3), K * n)
    Pi = np.clip(Q @ P, 1e-4, 1 - 1e-4)
    G = rng.binomial(2, Pi).astype(float)
    f = G.mean(0) / 2
    ok = (f > 0) & (f < 1)
    G, Pi = G[:, ok], Pi[:, ok]
    fp = Pi.mean(0)
    Xbar = (2 * Pi - 2 * fp) / np.sqrt(2 * fp * (1 - fp))
    U, sv, _ = np.linalg.svd(Xbar, full_matrices=False)
    N, Mk = G.shape
    theta = sv[:K - 1] ** 2 / Mk  # signal eigenvalues on the GRM scale
    return G, U[:, :K - 1], theta / np.sqrt(N / Mk)


def hap_init():
    """HAPNEST pool + reference PCA, loaded once in the parent (shared by fork)"""
    from sim import read_bed
    base = "/kellyData/home/albrecht/codex/relatePCA/data/hapnest/"
    fam, _, Gall = read_bed(base + "hapnest_pool")
    info = {}
    for ln in open(base + "individuals.tsv").read().split("\n")[1:]:
        if ln.strip():
            iid, anc, st = ln.split("\t")
            info[iid] = (anc, st)
    ids = [r[1] for r in fam]
    anc = np.array([info[i][0] for i in ids])
    st = np.array([info[i][1] for i in ids])
    R = Gall[st == "ref"].astype(float)
    f = R.mean(0) / 2
    keep = (f > 0.01) & (f < 0.99)
    sd = np.sqrt(2 * f[keep] * (1 - f[keep]))
    _, _, Vt = np.linalg.svd((R[:, keep] - 2 * f[keep]) / sd, full_matrices=False)
    HAP.update(G=Gall, anc=anc, pool=(st == "pool"), f=f, keep=keep, sd=sd, V=Vt[:5].T)


def hap_data(n, rep, mode="hap"):
    rng = np.random.default_rng(zlib.crc32(f"{mode}|{n}|{rep}".encode()))
    # hapsmall: EAS with only 2 individuals
    idx = np.concatenate([rng.choice(np.where(HAP["pool"] & (HAP["anc"] == a))[0],
                                     2 if (mode == "hapsmall" and a == "eas") else n, replace=False)
                          for a in ["eur", "eas", "afr", "amr", "csa", "mid"]])
    G = HAP["G"][idx].astype(float)
    k_ = HAP["keep"]
    T = ((G[:, k_] - 2 * HAP["f"][k_]) / HAP["sd"]) @ HAP["V"]
    f = G.mean(0) / 2
    ok = (f > 0) & (f < 1)
    return G[:, ok], T / np.linalg.norm(T, axis=0), np.full(5, np.nan)


def one_k(job):
    design, src, n, M, mode, rep, k, combos = job
    if src == "bn":
        G, T, det_ratio = sim_bn(n, M, mode, rep)
    else:
        G, T, det_ratio = hap_data(n, rep, mode)
    N = len(G)
    kk = min(k, T.shape[1])
    ck = (B.king_robust(G) > B.KING_SCREEN) & ~np.eye(N, dtype=bool)
    rows = []
    for c in combos:
        t = time.time()
        if c == "standard":
            U, det, r = standard(G, k), set(), None
        else:
            U, det, r = run_combo(G, c, k, ck)
        sec = time.time() - t
        r2, _ = B.score(U, T, N, np.array([], int), kk)
        rows.append((design, src, mode, n, G.shape[1], rep, k, N, c, ",".join(f"{x:.4f}" for x in r2),
                     ",".join(f"{x:.2f}" for x in det_ratio[:kk]), len(det), r, sec))
    return rows


COLS_K = ["design", "src", "mode", "n", "M", "rep", "k", "N", "combo", "r2_axes", "signal_ratio", "pairs", "rank",
          "sec"]
S1K_COMBOS = ["standard", "topr/unc/kf/0/v/whitec", "topr/unc/k/0/v/whitec", "topr/raw/kf/0/D/white",
              "topr/unc/kf/0/v/L",
              "pcp/cgrm/k/0/v/L", "pcp/unc/k/0/v/whitec", "pcp/raw/k/0/v/white"]


COLS = ["design", "n", "M", "scen", "rep", "k", "N", "combo", "minR2", "err", "recall", "false_pairs",
        "n_true", "rank", "sec", "r2_axes"]

# four_methods_bench.py methods as combinations (check design)
REF = {"unc_fit": "topr/unc/kf/0/v/L", "detect_white": "topr/raw/k/0/D/white",
       "dwg": "topr/unc/kfe/1/v/whitec", "aarobust_kin": "pcp/cgrm/k/0/v/L"}

S1_SCEN = ["none", "inbred1", "inbredcous", "inbredpop", "err1"]
S1_COMBOS = (["standard"]
             + [f"topr/{m}/kf/0/{s}/{f}" for m in ("raw", "unc") for s in ("v", "D") for f in ("L", "white", "whitec")]
             + ["topr/unc/k/0/v/whitec"]
             + [f"pcp/{m}/k/0/v/{f}" for m in ("raw", "unc") for f in ("L", "white", "whitec")]
             + ["pcp/cgrm/k/0/v/L"])


S2_SCEN = (["none", "mz1", "mz2", "po", "nuclear2", "sibs2", "halfsib", "avunc", "grand", "cousins", "child_mxl",
            "childx", "combined", "many", "halfsibx", "grandx", "avuncx", "xanc", "inbredpop_mz2", "err1_mz2"]
           + ["mz2:YRI2", "nuclear2:YRI2", "cousins:YRI2"])
S2_WSCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "xanc", "grandx", "avuncx", "asw", "aswx"]
S2_COMBOS = ["standard", "topr/raw/k/0/D/white",
             "pcp/cgrm/k/0/v/L", "pcp/cgrm/ke/0/v/L", "pcp/cgrm/ke/1/v/L", "pcp/unc/k/0/v/whitec",
             "pcp/unc/ke/1/v/whitec",
             "topr/unc/k/0/v/whitec", "topr/unc/ke/1/v/whitec", "topr/unc/kf/0/v/whitec",
             "topr/unc/kfe/0/v/whitec", "topr/unc/kfe/1/v/whitec", "topr/unc/kfe/1/D/whitec"]


S2D_COMBOS = ["standard"] + [f"dec:{i}/unc/{c}/{d}/{q}/whitec" for i in ("pcp", "topr")
                              for c, d, q in [("k", "R", "0"), ("ke", "R", "1"), ("e", "R", "1"), ("k", "EA", "0"),
                                              ("all", "EA", "0"), ("all", "EA", "1"), ("k", "KING", "0")]]
S2D_KEA = [f"dec:{i}/unc/all/KEA/{q}/whitec" for i in ("pcp", "topr") for q in ("0", "1")]
# options from the slides not yet tested: all pairs with the fit deciding, k0_ML in place of
# k0_mom, k0 alone as the decision (D9 again as the reference on the same data sets)
FORCE = 0.06  # KING above this: the pair is always imputed (FR, FKEA)
S2D_FORCE = [f"dec:{i}/unc/{c}/{d}/1/whitec" for i in ("pcp", "topr") for c, d in [("ke", "FR"), ("all", "FKEA")]]
S2D_NEW = [f"dec:{i}/unc/{c}/{d}/{q}/whitec" for i in ("pcp", "topr")
           for c, d, q in [("all", "R", "0"), ("all", "KEA", "1"), ("all", "KEA", "2"), ("ke", "R", "2"),
                           ("ke", "K0", "1"), ("ke", "K0", "2")]]


def jobs_for(design, kt=3):
    if design == "check":
        return [("check", n, 0, s, r, 3, list(REF.values())) for n in (5, 10) for s in ("none", "mz2", "xanc", "inbredpop_mz2")
                for r in range(2)]
    if design == "S1k":
        jobs = [("S1k", "bn", n, 20000, mode, r, k, S1K_COMBOS) for mode in ("disc", "cont", "small") for n in (5, 10, 20)
                for k in (2, 5, 15) for r in range(10)]
        return jobs + [("S1k", "hap", n, 0, mode, r, k, S1K_COMBOS) for mode in ("hap", "hapsmall")
                       for n in (5, 10, 20) for k in (2, 5, 15) for r in range(10)]
    if design == "S2k":
        scen = S2_WSCEN if kt == 2 else S2_SCEN
        return [("S2k", n, 0, s, r, k, S2D_KEA) for n in (5, 10, 20, 40) for s in scen for k in (kt, 10)
                for r in range(5)]
    if design == "S2f":
        scen = S2_WSCEN if kt == 2 else S2_SCEN
        return [("S2f", n, 0, s, r, k, S2D_FORCE) for n in (5, 10, 20, 40) for s in scen for k in (kt, 10)
                for r in range(5)]
    if design == "S2n":
        scen = S2_WSCEN if kt == 2 else S2_SCEN
        return [("S2n", n, 0, s, r, k, S2D_NEW) for n in (5, 10, 20, 40) for s in scen for k in (kt, 10)
                for r in range(5)]
    if design == "S2d":
        scen = S2_WSCEN if kt == 2 else S2_SCEN
        return [("S2d", n, 0, s, r, k, S2D_COMBOS) for n in (5, 10, 20, 40) for s in scen for k in (kt, 10)
                for r in range(5)]
    if design == "S2":
        scen = S2_WSCEN if kt == 2 else S2_SCEN
        return [("S2", n, 0, s, r, k, S2_COMBOS) for n in (5, 10, 20, 40) for s in scen for k in (kt, 10)
                for r in range(5)]
    if design == "S1s":
        return [("S1s", n, M, f"{s}:{p}2", r, k, S1K_COMBOS) for n in (5, 10, 20) for M in (0, 10000)
                for s in ("none", "err1") for p in ("MXL", "YRI") for k in (kt, 10) for r in range(10)]
    if design == "S1":
        return [("S1", n, M, s, r, k, S1_COMBOS) for n in (5, 10, 20) for M in (0, 10000) for s in S1_SCEN
                for k in (kt, 10) for r in range(10)]
    raise ValueError(design)


def check():
    """the harness against four_methods_bench.run_methods on the same data"""
    import four_methods_bench as F
    worst = {}
    for (_, n, M, scen, rep, k, _) in jobs_for("check"):
        ds = dataset(n, M, scen, rep)
        G = ds["G"].astype(float)
        N = len(G)
        ck = (B.king_robust(G) > B.KING_SCREEN) & ~np.eye(N, dtype=bool)
        ref = F.run_methods(G)
        for name, c in REF.items():
            U, det, _ = run_combo(G, c, k, ck)
            Ur, detr = ref[name]
            a = U / np.linalg.norm(U, axis=0)
            b = Ur / np.linalg.norm(Ur, axis=0)
            cos = np.abs((a * b).sum(0)).min()
            w = worst.setdefault(name, [1.0, True])
            w[0] = min(w[0], cos)
            w[1] &= det == detr
            print(f"n={n} {scen:14s} rep={rep} {name:13s} |cos| {cos:.6f} pairs same {det == detr}", flush=True)
    print("worst:", worst)


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--panel", default="admix", choices=list(PANELS))
    a = ap.parse_args()
    if a.design == "check":
        init(a.panel)
        check()
        sys.exit()
    jobs = jobs_for(a.design, PANELS[a.panel][2])
    t0 = time.time()
    if a.design == "S1k":
        hap_init()  # before the fork: shared with the workers
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for rr in ex.map(one_k, jobs, chunksize=1) for r in rr]
        pd.DataFrame(rows, columns=COLS_K).to_csv(a.out, sep="\t", index=False)
    else:
        with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.panel,)) as ex:
            rows = [r for rr in ex.map(one, jobs, chunksize=1) for r in rr]
        pd.DataFrame(rows, columns=COLS).to_csv(a.out, sep="\t", index=False)
    print(f"{len(jobs)} data sets, {len(rows)} rows in {time.time() - t0:.0f} s")
