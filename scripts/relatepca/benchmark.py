#!/usr/bin/env python3
"""Extensive test of PCA methods robust to close relatives (small N).

Each run draws n unrelated individuals per population (the "base" sample),
optionally subsamples M SNPs, and adds relatives made by copying genotypes or
alleles (random phase, recombinant gametes at 1 cM/Mb). Founders of a family
are either base individuals or "external" individuals of the same
population that are not in the data.

Truth: the PCA of the base sample. External founders are placed in it with an
augmented PCA (base + external founders) mapped onto the base PCA by a linear
fit on the base individuals; every relative's expected position is the mean of
its parents' positions (an MZ copy: its source). CS-type methods are scored
against the same construction with the Chen & Storey PCA.

Scores (per method): R2 of each true PC regressed on the method's PCs (+1) over
the base individuals, min over PCs, and the error of the relatives (RMS
distance to the expected position after the same linear map, relative to the
spread of the base individuals). Wall time per method (single thread), and
detection (true/false related pairs) where a method detects relatives.

Family specs (comma separated) TYPE:POP[:count]:
  mz       count identical copies of a base individual
  child    count children of two base individuals (PO, and sibs if count > 1)
  childx   one child of base individuals of two populations, POP = A x B
  sibs     count full sibs, external parents
  halfsib  two half sibs: one shared external parent
  avunc    uncle/aunt and niece/nephew (2nd degree), external founders
  grand    grandchild of a base individual (2nd degree)
  cousins  two first cousins (3rd degree, phi = 1/16), external founders
  inbred   count inbred individuals, each a child of two full sibs (F = 1/4),
           external founders, no relatives in the data
  inbredcous  one child of two first cousins (F = 1/16)
  err      err:RATE adds genotype errors at RATE to every genotype (a
           different genotype drawn at random)
"""
import argparse
import os
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import illustrate as I  # noqa: E402
from sim import Dropper, read_bed  # noqa: E402

TAU = 2 ** -3.5
KING_SCREEN = 0.04  # liberal KING candidate screen for the *_kc variants
TAU3 = 2 ** -4.5  # 3rd/4th degree boundary, for the *_t3 variants

SCENARIOS = {
    "none": "",
    "mz1": "mz:CEU:1",
    "mz2": "mz:CEU:2",
    "po": "child:CEU:1",
    "nuclear2": "child:CEU:2",
    "sibs2": "sibs:CEU:2",
    "halfsib": "halfsib:CEU",
    "avunc": "avunc:CEU",
    "grand": "grand:CEU",
    "cousins": "cousins:CEU",
    "child_mxl": "child:MXL:1",
    "childx": "childx:CEUxYRI",
    "combined": "mz:CEU:2,child:MXL:1",
    "many": "mz:YRI:1,child:CEU:1,sibs:CHB:2,halfsib:MXL,avunc:CEU",
}
# HWE violations within individuals (stage 1b)
SCENARIOS_HWE = {
    "inbred1": "inbred:CEU:1",
    "inbredcous": "inbredcous:CEU",
    "inbredpop": "inbred:CEU:5",
    "err1": "err:0.01",
    "inbredpop_mz2": "inbred:CEU:5,mz:CEU:2",
    "err1_mz2": "mz:CEU:2,err:0.01",
}
SCENARIOS.update(SCENARIOS_HWE)
# relatives of different ancestry (KING-robust is pulled down for them)
SCENARIOS_XANC = {
    "xanc": "halfsibx:CEU:CHB:YRI,grandx:YRI:CEU,avuncx:CHB:YRI,childx:CEUxYRI",
    "halfsibx": "halfsibx:CEU:CHB:YRI",
    "grandx": "grandx:YRI:CEU",
    "avuncx": "avuncx:CHB:YRI",
}
SCENARIOS.update(SCENARIOS_XANC)

# ------------------------------------------------------------------ data


def king_robust(G):
    """plink2 --make-king (KING-robust between-family) kinship matrix"""
    het = (G == 1).astype(float)
    a0 = (G == 0).astype(float)
    a2 = (G == 2).astype(float)
    HH = het @ het.T
    I0 = a0 @ a2.T + a2 @ a0.T
    nh = het.sum(1)
    mn = np.minimum(nh[:, None], nh[None, :])
    K = (HH - 2 * I0) / (2 * mn) + 0.5 - (nh[:, None] + nh[None, :]) / (4 * mn)
    np.fill_diagonal(K, 0)
    return K


def pairs_above(K, tau):
    iu = np.triu_indices(len(K), 1)
    sel = K[iu] >= tau
    return [(int(i), int(j), float(K[i, j])) for i, j in zip(iu[0][sel], iu[1][sel])]


def make_dataset(Gall, lab, bim, pops, n, M, spec, seed):
    rng = np.random.default_rng(seed)
    pool = {p: list(rng.permutation(np.where(lab == p)[0])) for p in pops if p in set(lab)}
    for p in pops:
        if len(pool[p]) < n:
            raise ValueError(f"only {len(pool[p])} {p} individuals")
    base_idx = [pool[p].pop() for p in pops for _ in range(n)]
    base_pop = [p for p in pops for _ in range(n)]
    # SNPs: polymorphic in the base sample, then subsample
    Gb = Gall[base_idx].astype(np.int8)
    poly = np.where(Gb.min(0) != Gb.max(0))[0]
    if M and M < len(poly):
        poly = np.sort(rng.choice(poly, M, replace=False))
    bim_s = [bim[i] for i in poly]
    Gs = lambda i: Gall[i, poly].astype(np.int8)  # noqa: E731
    drop = Dropper(bim_s, rng)

    # individuals: dict id -> (kind, info)
    nodes = {}
    for b, (gi, p) in enumerate(zip(base_idx, base_pop)):
        nodes[f"B{b}"] = ("base", gi, p)
    unused = {p: [f"B{b}" for b in range(len(base_idx)) if base_pop[b] == p] for p in pops}
    for p in unused:
        rng.shuffle(unused[p])
    ext, data_rel, parents = [], [], {}
    err_rate = 0.0
    counter = [0]

    def new(prefix):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def base_of(p):
        if p not in unused or not unused[p]:
            raise ValueError(f"not enough base {p} individuals for the families")
        return unused[p].pop()

    def external(p):
        if p not in pool or not pool[p]:
            raise ValueError(f"no external {p} individuals left")
        x = new("E")
        nodes[x] = ("ext", pool[p].pop(), p)
        ext.append(x)
        return x

    def child(f, m, in_data):
        x = new("C")
        nodes[x] = ("child", (f, m), None)
        parents[x] = (f, m)
        if in_data:
            data_rel.append(x)
        return x

    for fs in [s for s in spec.split(",") if s]:
        t = fs.split(":")
        kind, P = t[0], t[1]
        c = int(t[2]) if len(t) > 2 and t[2].isdigit() else 1
        if kind == "mz":
            src = base_of(P)
            for _ in range(c):
                x = new("D")
                nodes[x] = ("copy", src, None)
                parents[x] = (src, src)
                data_rel.append(x)
        elif kind == "child":
            f, m = base_of(P), base_of(P)
            for _ in range(c):
                child(f, m, True)
        elif kind == "childx":
            pa, pb = P.split("x")
            child(base_of(pa), base_of(pb), True)
        elif kind == "sibs":
            f, m = external(P), external(P)
            for _ in range(c):
                child(f, m, True)
        elif kind == "halfsib":
            e0 = external(P)
            child(e0, external(P), True)
            child(e0, external(P), True)
        elif kind == "avunc":
            f, m = external(P), external(P)
            child(f, m, True)
            b = child(f, m, False)
            child(b, external(P), True)
        elif kind == "grand":
            c1 = child(base_of(P), external(P), False)
            child(c1, external(P), True)
        elif kind == "halfsibx":  # father from P, mothers from two other populations
            pa, pb, pc = t[1], t[2], t[3]
            e0 = external(pa)
            child(e0, external(pb), True)
            child(e0, external(pc), True)
        elif kind == "grandx":  # grandparent (in data) from P, grandchild 1/4 P
            pa, pb = t[1], t[2]
            c1 = child(base_of(pa), external(pb), False)
            child(c1, external(pb), True)
        elif kind == "avuncx":  # aunt from P (in data), nephew half P, half Q
            pa, pb = t[1], t[2]
            f, m = external(pa), external(pa)
            child(f, m, True)
            b = child(f, m, False)
            child(b, external(pb), True)
        elif kind == "inbred":
            for _ in range(c):
                f, m = external(P), external(P)
                s1, s2 = child(f, m, False), child(f, m, False)
                child(s1, s2, True)
        elif kind == "inbredcous":
            f, m = external(P), external(P)
            s1, s2 = child(f, m, False), child(f, m, False)
            c1, c2 = child(s1, external(P), False), child(s2, external(P), False)
            child(c1, c2, True)
        elif kind == "err":
            err_rate = float(P)
        elif kind == "cousins":
            f, m = external(P), external(P)
            s1, s2 = child(f, m, False), child(f, m, False)
            child(s1, external(P), True)
            child(s2, external(P), True)
        else:
            raise ValueError(kind)

    # genetics
    hap = {}

    def haps(x):
        if x in hap:
            return hap[x]
        kind, info, _ = nodes[x]
        if kind in ("base", "ext"):
            hap[x] = drop.phase(Gs(info))
        elif kind == "copy":
            hap[x] = haps(info)
        else:
            f, m = info
            hap[x] = np.stack([drop.gamete(haps(f)), drop.gamete(haps(m))])
        return hap[x]

    rows = [f"B{b}" for b in range(len(base_idx))] + data_rel
    G = np.array([Gs(nodes[x][1]) if nodes[x][0] == "base" else haps(x).sum(0) for x in rows], dtype=float)
    Gext = np.array([Gs(nodes[x][1]) for x in ext], dtype=float) if ext else np.zeros((0, G.shape[1]))
    if err_rate > 0:  # genotype errors: replace by one of the two other genotypes
        hit = rng.random(G.shape) < err_rate
        G[hit] = (G[hit] + rng.integers(1, 3, hit.sum())) % 3
    keep = G.std(0) > 0
    G, Gext = G[:, keep], Gext[:, keep]
    cols = poly[keep]
    ref_idx = [i for p in pops for i in pool[p]]  # every individual not used in the data

    # pedigree kinship among all nodes (topological order = insertion order)
    order = list(nodes)
    ix = {x: i for i, x in enumerate(order)}
    Kp = np.zeros((len(order), len(order)))
    for i, x in enumerate(order):
        kind, info, _ = nodes[x]
        if kind in ("base", "ext"):
            Kp[i, i] = 0.5
        elif kind == "copy":
            s = ix[info]
            Kp[i, :i] = Kp[:i, i] = Kp[s, :i]
            Kp[i, i] = Kp[i, s] = Kp[s, i] = Kp[s, s]
        else:
            f, m = ix[info[0]], ix[info[1]]
            Kp[i, i] = 0.5 + 0.5 * Kp[f, m]
            Kp[i, :i] = Kp[:i, i] = 0.5 * (Kp[f, :i] + Kp[m, :i])
    ri = [ix[x] for x in rows]
    Kdata = Kp[np.ix_(ri, ri)]
    np.fill_diagonal(Kdata, 0)
    return dict(G=G, Gext=Gext, rows=rows, ext=ext, nodes=nodes, parents=parents, n0=len(base_idx),
                base_pop=base_pop, Kped=Kdata, cols=cols, ref_idx=ref_idx)


def truth_reference(ds, Gall, k):
    """population truth: PCA of a reference panel of all individuals not used
    in the data (same populations), every individual of the data projected onto
    it with the panel's allele frequencies"""
    Gr = Gall[np.ix_(ds["ref_idx"], ds["cols"])].astype(float)
    f = Gr.mean(0) / 2
    ok = (f > 0) & (f < 1)
    sd = np.sqrt(2 * f[ok] * (1 - f[ok]))
    Xr = (Gr[:, ok] - 2 * f[ok]) / sd
    _, v = I.top_eig(Xr @ Xr.T / ok.sum(), k)
    T = ((ds["G"][:, ok] - 2 * f[ok]) / sd) @ (Xr.T @ v)
    return T / np.linalg.norm(T, axis=0)  # unit-length columns, as eigenvectors: PCs weighted equally


def truth_positions(ds, k, cs):
    """truth PCs of the base sample and expected positions of the relatives"""
    G, n0 = ds["G"], ds["n0"]
    Gb = G[:n0]

    def pcs(Gm):
        if cs:
            return I.cs_pcs(I.cs_matrix(Gm), k)[1]
        X = I.standardize(Gm)
        X = X[:, np.isfinite(X).all(0)]
        return I.top_eig(X @ X.T / X.shape[1], k)[1]

    T0 = pcs(Gb)
    pos = {f"B{b}": T0[b] for b in range(n0)}
    if ds["ext"]:
        A = pcs(np.vstack([Gb, ds["Gext"]]))
        Z = np.c_[A, np.ones(len(A))]
        B, *_ = np.linalg.lstsq(Z[:n0], T0, rcond=None)
        for e, x in enumerate(ds["ext"]):
            pos[x] = Z[n0 + e] @ B

    def p(x):
        if x not in pos:
            f, m = ds["parents"][x]
            pos[x] = (p(f) + p(m)) / 2
        return pos[x]

    return np.array([p(x) for x in ds["rows"]])


# ------------------------------------------------------------------ methods


def grm(G):
    X = I.standardize(G)
    X = X[:, np.isfinite(X).all(0)]
    return X, X @ X.T / X.shape[1]


def lrkin_cs_autoK_full(G, H, D, tau, rmax=10, iters=500):
    edge = I.cs_noise_edge(G)
    S = np.zeros_like(H)
    r = 1
    while True:
        L_old = None
        for _ in range(iters):
            w, v = I.top_eig(H - S, r)
            L = (v * w) @ v.T
            S = np.where(I.kin_select(H - L, D, tau), H - L, 0)
            if L_old is not None and np.max(np.abs(L - L_old)) < 1e-10:
                break
            L_old = L
        if r >= rmax or I.top_eig(H - S, r + 1)[0][r] <= edge:
            break
        r += 1
    return L, S, r


def run_methods(G, k, tau, ped_pairs, only=None):
    """returns {method: (U, seconds, info)}"""
    out = {}

    def timed(name, f):
        t0 = time.perf_counter()
        U, info = f()
        out[name] = (U, time.perf_counter() - t0, info)

    def standard():
        _, C = grm(G)
        return I.top_eig(C, k)[1], {}

    def white_king():
        X, _ = grm(G)
        pp = pairs_above(king_robust(G), tau)
        return I.whitened_numpy(X, pp, k), {"pairs": pp}

    def white_gls_king():
        pp = pairs_above(king_robust(G), tau)
        return I.whitened_gls(G, pp, k), {"pairs": pp}

    def white_gls_ped():
        return I.whitened_gls(G, ped_pairs, k), {}

    def lrkin_cs():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, _ = I.lr_kin(H, k + 1, tau, False, D)
        return I.cs_pcs(L, k)[1], {"pairs": I.lr_kin_pairs(H, D, L, S)}

    def lrkin_cs_autoK():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, r = lrkin_cs_autoK_full(G, H, D, tau)
        return I.cs_pcs(L, r - 1)[1] if r > 1 else np.zeros((len(G), 1)), {
            "pairs": I.lr_kin_pairs(H, D, L, S), "K": r}

    def hyb_gls():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, _ = I.lr_kin(H, k + 1, tau, False, D)
        pp = I.lr_kin_pairs(H, D, L, S)
        return I.whitened_gls(G, pp, k), {"pairs": pp}

    def hyb_gls_autoK():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, r = lrkin_cs_autoK_full(G, H, D, tau)
        pp = I.lr_kin_pairs(H, D, L, S)
        return I.whitened_gls(G, pp, k), {"pairs": pp, "K": r}

    def lrkin_cs_t3():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, _ = I.lr_kin(H, k + 1, TAU3, False, D)
        return I.cs_pcs(L, k)[1], {"pairs": I.lr_kin_pairs(H, D, L, S)}

    def hyb_gls_t3():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, _ = I.lr_kin(H, k + 1, TAU3, False, D)
        pp = I.lr_kin_pairs(H, D, L, S)
        return I.whitened_gls(G, pp, k), {"pairs": pp}

    def lrkin_cs_kc():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        cand = king_robust(G) > KING_SCREEN
        L, S, _ = I.lr_kin(H, k + 1, tau, False, D, cand=cand)
        return I.cs_pcs(L, k)[1], {"pairs": I.lr_kin_pairs(H, D, L, S)}

    def hyb_gls_kc():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        cand = king_robust(G) > KING_SCREEN
        L, S, _ = I.lr_kin(H, k + 1, tau, False, D, cand=cand)
        pp = I.lr_kin_pairs(H, D, L, S)
        return I.whitened_gls(G, pp, k), {"pairs": pp}

    def aarobust_kin():
        # AArobust with the kinship threshold in place of lambda, KING screen
        _, C = grm(G)
        cand = king_robust(G) > KING_SCREEN
        L, S, _ = I.pcp_kin(C, tau, True, cand=cand)
        return I.top_eig(L, k)[1], {}

    def white_cs_king():
        pp = pairs_above(king_robust(G), tau)
        return I.whitened_cs(G, pp, k), {"pairs": pp}

    def hyb_cs_kc():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        cand = king_robust(G) > KING_SCREEN
        L, S, _ = I.lr_kin(H, k + 1, tau, False, D, cand=cand)
        pp = I.lr_kin_pairs(H, D, L, S)
        return I.whitened_cs(G, pp, k), {"pairs": pp}

    def auto_robust():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        U, pp = I.auto_robust(G, H, D, k, tau)
        return U, {"pairs": pp}

    def lrkin_cs_fd():
        # raw Gram with the diagonal free (= the CS matrix off the diagonal)
        AM = G @ G.T / G.shape[1]
        D = (G * (2 - G)).mean(1)
        cand = king_robust(G) > KING_SCREEN
        L, S, _ = I.lr_kin_fd(AM, k + 1, tau, D, cand)
        return I.cs_pcs(L, k)[1], {"pairs": I.lr_kin_pairs(AM, D, L, S)}

    def hyb_cs_fdA():
        # detect-white with the detection rank chosen by the noise edge, at most k + 1;
        # fit of the raw Gram with the diagonal free (the CS matrix is not needed)
        AM = G @ G.T / G.shape[1]
        D = (G * (2 - G)).mean(1)
        cand = king_robust(G) > KING_SCREEN
        L, S, _, r = I.lr_kin_fd_auto(AM, tau, D, I.cs_noise_edge(G), cand, rmax=k + 1)
        pp = I.lr_kin_pairs(AM, D, L, S)
        v = np.diag(AM) - np.diag(L)
        return I.whitened_noise(AM, v, pp, k), {"pairs": pp, "K": r}

    def hyb_cs_fd():
        # free-diagonal detection; whitening with the estimated noise variance
        # v_i = (G G'/M)_ii - L_ii (the observed diagonal minus the structure part)
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        cand = king_robust(G) > KING_SCREEN
        L, S, _ = I.lr_kin_fd(H, k + 1, tau, D, cand)
        pp = I.lr_kin_pairs(H, D, L, S)
        AM = G @ G.T / G.shape[1]
        v = np.diag(AM) - np.diag(L)
        return I.whitened_noise(AM, v, pp, k), {"pairs": pp}

    def aarobust():
        _, C = grm(G)
        return I.top_eig(I.rpca(C, free_diag=True)[0], k)[1], {}

    def pcpkin_cs():
        H = I.cs_matrix(G)
        D = (G * (2 - G)).mean(1)
        L, S, _ = I.pcp_kin(I.double_centre(H), tau, False, D)
        return I.top_eig(L, k)[1], {}

    for name, f in [("standard", standard), ("white_king", white_king), ("white_gls_king", white_gls_king),
                    ("white_gls_ped", white_gls_ped), ("lrkin_cs", lrkin_cs), ("lrkin_cs_autoK", lrkin_cs_autoK),
                    ("hyb_gls", hyb_gls), ("hyb_gls_autoK", hyb_gls_autoK), ("lrkin_cs_t3", lrkin_cs_t3),
                    ("hyb_gls_t3", hyb_gls_t3), ("lrkin_cs_kc", lrkin_cs_kc), ("hyb_gls_kc", hyb_gls_kc),
                    ("aarobust", aarobust), ("aarobust_kin", aarobust_kin), ("white_cs_king", white_cs_king),
                    ("hyb_cs_kc", hyb_cs_kc), ("auto_robust", auto_robust), ("lrkin_cs_fd", lrkin_cs_fd),
                    ("hyb_cs_fd", hyb_cs_fd), ("hyb_cs_fdA", hyb_cs_fdA),
                    ("pcpkin_cs", pcpkin_cs)]:
        if only is None or name in only:
            timed(name, f)
    return out


CS_TYPE = {"lrkin_cs", "lrkin_cs_autoK", "lrkin_cs_t3", "lrkin_cs_kc", "lrkin_cs_fd", "pcpkin_cs"}


def score(U, T, n0, rel, kk):
    """R2 of the top kk true PCs on the method's top kk PCs"""
    U = U[:, :kk]
    Z = np.c_[U, np.ones(len(U))]
    base = np.arange(n0)
    B, *_ = np.linalg.lstsq(Z[base], T[base, :kk], rcond=None)
    P = Z @ B
    Tb = T[base, :kk]
    r2 = 1 - ((Tb - P[base]) ** 2).sum(0) / ((Tb - Tb.mean(0)) ** 2).sum(0)
    spread = np.sqrt(((Tb - Tb.mean(0)) ** 2).sum(1).mean())
    err = np.sqrt(((T[rel, :kk] - P[rel]) ** 2).sum(1).mean()) / spread if len(rel) else np.nan
    return r2, err


def one_run(job):
    (bfile, pops, n, M, scen, rep, k_used, tau) = job
    fam, bim, Gall = DATA
    lab = np.array([r[1] for r in fam])
    seed = zlib.crc32(f"{pops}|{n}|{M}|{scen}|{rep}".encode())
    try:
        ds = make_dataset(Gall, lab, bim, pops.split(","), n, M, SCENARIOS[scen], seed)
    except ValueError as e:
        return [dict(pops=pops, n=n, M=M, scen=scen, rep=rep, k=k_used, method="SKIP", error=str(e))]
    k_true = len(pops.split(",")) - 1
    G, n0 = ds["G"], ds["n0"]
    N = len(G)
    rel = np.arange(n0, N)
    T = truth_positions(ds, k_true, cs=False)
    Tcs = truth_positions(ds, k_true, cs=True)
    Tref = truth_reference(ds, Gall, k_true)
    iu = np.triu_indices(N, 1)
    ped_pairs = [(int(i), int(j), float(ds["Kped"][i, j])) for i, j in zip(*iu) if ds["Kped"][i, j] >= tau]
    true_set = {(i, j) for i, j, _ in ped_pairs}
    res = run_methods(G, k_used, tau, ped_pairs, ONLY)
    rows = []
    kk = min(k_used, k_true)
    for m, (U, sec, info) in res.items():
        kk_ = kk if U.shape[1] >= kk else U.shape[1]
        # primary: reference-panel truth (same for every method, relatives at their realised position)
        r2, err = score(U, Tref, n0, rel, kk_)
        # secondary: the base-sample truth used before (standard or CS PCA of the base sample)
        r2b, errb = score(U, Tcs if m in CS_TYPE else T, n0, rel, kk_)
        row = dict(pops=pops, n=n, M=G.shape[1], scen=scen, rep=rep, k=k_used, N=N, method=m,
                   minR2=float(r2.min()), r2=";".join(f"{x:.4f}" for x in r2), err=err, sec=sec,
                   K=info.get("K", ""), minR2_base=float(r2b.min()), err_base=errb, n_ref=len(ds["ref_idx"]))
        if "pairs" in info:
            found = {(min(i, j), max(i, j)) for i, j, _ in info["pairs"]}
            row["tp"], row["fp"], row["n_true"] = len(found & true_set), len(found - true_set), len(true_set)
        rows.append(row)
    return rows


DATA = None
ONLY = None


def init(bfile, only=None):
    global DATA, ONLY
    DATA = read_bed(bfile)
    ONLY = only


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", choices=["main", "large", "wrongk", "timing", "quick", "hwe"], default="quick")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--methods", default="", help="comma-separated subset of methods (default: all)")
    a = ap.parse_args()
    only = set(a.methods.split(",")) if a.methods else None
    P4, P3 = "CEU,CHB,MXL,YRI", "CEU,CHB,YRI"
    jobs = []
    if a.design == "main":
        for n in [5, 10, 20, 40]:
            for M in [0, 10000, 2000]:
                for s in SCENARIOS:
                    for r in range(a.reps):
                        jobs.append((a.bfile, P4, n, M, s, r, 3, TAU))
    elif a.design == "large":
        for n in [60, 85]:
            for s in SCENARIOS:
                if s in ("child_mxl", "many", "combined"):
                    continue  # MXL not in the 3-population design
                for r in range(a.reps):
                    jobs.append((a.bfile, P3, n, 0, s, r, 2, TAU))
    elif a.design == "wrongk":
        for kk in [2, 4]:
            for s in SCENARIOS:
                for r in range(a.reps):
                    jobs.append((a.bfile, P4, 10, 0, s, r, kk, TAU))
    elif a.design == "hwe":
        for n in [5, 10, 20, 40]:
            for M in [0, 10000]:
                for s in SCENARIOS_HWE:
                    for r in range(a.reps):
                        jobs.append((a.bfile, P4, n, M, s, r, 3, TAU))
    elif a.design == "timing":
        for pops, n in [(P4, 10), (P4, 25), (P4, 50), (P3, 75), (P3, 98)]:
            for r in range(3):
                jobs.append((a.bfile, pops, n, 0, "combined" if "MXL" in pops else "mz2", r,
                             len(pops.split(",")) - 1, TAU))
    else:
        for s in SCENARIOS:
            jobs.append((a.bfile, P4, 10, 0, s, 0, 3, TAU))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.bfile, only)) as ex:
        for rr in ex.map(one_run, jobs, chunksize=1):
            rows += rr
    cols = ["pops", "n", "M", "scen", "rep", "k", "N", "method", "minR2", "r2", "err", "minR2_base", "err_base",
            "n_ref", "sec", "K", "tp", "fp", "n_true", "error"]
    with open(a.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"{len(jobs)} runs, {len(rows)} rows in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
