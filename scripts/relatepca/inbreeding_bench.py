#!/usr/bin/env python3
"""Inbreeding and excess heterozygosity within individuals: what they do to
PCA, and whether PCAone's aarobust-kin or dwg fix it. admixTjeck2 (CEU, CHB,
MXL, YRI; 54k SNPs), subsets of n per population and the full data.

Scenarios (the affected individuals are CEU; no relatives anywhere):
  none       no change
  inb_1/5/pop   1, 5 or all CEU individuals inbred: each heterozygous genotype
             becomes homozygous (either allele) with probability F = 0.5
             (genotype frequencies p^2 + Fpq, 2pq(1-F), q^2 + Fpq)
  het_1/5/pop   excess heterozygosity: each homozygous genotype becomes
             heterozygous with probability h = 0.2 (as from contamination)

Methods: PCAone (the binary) with -k 3: standard, --impute-diag
(aarobust-diag), --robust aarobust-kin (observed diagonal in the final PCs),
--robust aarobust-kin --impute-diag, --robust dwg. Truth: subsets, a reference panel (all individuals not in the
data) with the clean genotypes of the data projected (benchmark.truth_reference);
full data, standard PCA of the clean genotypes. Score: min R2 of the 3 true
ancestry axes on a method's top 3 PCs over all individuals (benchmark.score),
and the affected individuals' error (relative to the spread of all
individuals); MEV and minSSE as in the PCAone paper (src/Utils.cpp); the
affected individuals' error in units of the spread of their own population
(err_pop; err_pop_ref: the same for the unaffected CEU); the largest individual error over the median one (max_med); pairs
called related; dwg noise of the affected individuals relative to the others.

usage: inbreeding_bench.py <bfile> <workdir> <out.tsv> [workers]
"""
import os
import shutil
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

PCAONE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PCAone")
POPS = ["CEU", "CHB", "MXL", "YRI"]
SCEN = {"none": (None, 0), "inb_1": ("inb", 1), "inb_5": ("inb", 5), "inb_pop": ("inb", None),
        "het_1": ("het", 1), "het_5": ("het", 5), "het_pop": ("het", None)}
F_INB, H_HET, K = 0.5, 0.2, 3
METHODS = {"standard": [], "aarobust-diag": ["--impute-diag"], "aarobust-kin": ["--robust", "aarobust-kin"],
           "aarobust-kin+diag": ["--robust", "aarobust-kin", "--impute-diag"], "dwg": ["--robust", "dwg"]}
DATA = {}


def init(bfile):
    fam, bim, G = read_bed(bfile)
    DATA.update(G=G, lab=np.array([r[1] for r in fam]), bim=bim)


def transform(g, kind, rng):
    """one individual's genotypes: inbreeding (het -> hom) or excess heterozygosity (hom -> het)"""
    g = g.copy()
    if kind == "inb":
        hit = (g == 1) & (rng.random(g.shape) < F_INB)
        g[hit] = 2 * rng.integers(0, 2, hit.sum())
    else:
        hit = (g != 1) & (rng.random(g.shape) < H_HET)
        g[hit] = 1
    return g


def unit(X):
    X = X - X.mean(0)
    return X / np.linalg.norm(X, axis=0)


def extra_scores(U, T, aff, pop):
    """MEV and minSSE (PCAone, src/Utils.cpp; centred, unit-length columns),
    err_pop and max_med (positions: the truth regressed on the method's PCs,
    as benchmark.score)"""
    Uu, Tu = unit(U), unit(T)
    Qu, Qt = np.linalg.qr(Uu)[0], np.linalg.qr(Tu)[0]
    mev = np.mean([np.linalg.norm(Qu.T @ Qt[:, j]) for j in range(Qt.shape[1])])
    minsse = sum(min(((Uu[:, i] - s * Tu[:, j]) ** 2).sum() for j in range(Tu.shape[1]) for s in (1, -1))
                 for i in range(Uu.shape[1]))
    Z = np.c_[U, np.ones(len(U))]
    P = Z @ np.linalg.lstsq(Z, T, rcond=None)[0]
    e = np.sqrt(((P - T) ** 2).sum(1))
    max_med = e.max() / np.median(e)
    # own population: CEU (all affected individuals are CEU)
    same = np.flatnonzero(pop == "CEU")
    oth = np.setdiff1d(same, aff)
    ref = oth if len(oth) > 1 else same  # all affected: the spread of the clean truth
    spread = np.sqrt(((T[ref] - T[ref].mean(0)) ** 2).sum(1).mean())
    err_pop = np.sqrt((e[aff] ** 2).mean()) / spread if len(aff) else np.nan
    # the same for the unaffected CEU: the reference value
    err_pop_ref = np.sqrt((e[oth] ** 2).mean()) / spread if len(oth) else np.nan
    return mev, minsse, err_pop, err_pop_ref, max_med


def one(job):
    n, scen, rep, workdir = job[:4]
    keep_dir = len(job) > 4 and job[4]  # keep the run directory (example plots)
    rng = np.random.default_rng(zlib.crc32(f"{n}|{scen}|{rep}".encode()))
    G, lab = DATA["G"], DATA["lab"]
    if n:  # subset: n per population, the rest is the reference panel
        idx = np.concatenate([rng.choice(np.flatnonzero(lab == p), n, replace=False) for p in POPS])
    else:
        idx = np.flatnonzero(np.isin(lab, POPS))
    Gc = G[idx].astype(np.int8)
    poly = Gc.min(0) != Gc.max(0)
    Gc = Gc[:, poly]
    kind, cnt = SCEN[scen]
    ceu = np.flatnonzero(lab[idx] == "CEU")
    aff = np.array([], int) if kind is None else rng.choice(ceu, min(cnt or len(ceu), len(ceu)), replace=False)
    Gm = Gc.copy()
    for i in aff:
        Gm[i] = transform(Gc[i], kind, rng)
    keep = Gm.min(0) != Gm.max(0)
    Gm, Gc = Gm[:, keep], Gc[:, keep]
    cols = np.flatnonzero(poly)[keep]
    if n:
        ref = np.setdiff1d(np.flatnonzero(np.isin(lab, POPS)), idx)
        T = B.truth_reference(dict(G=Gc.astype(float), ref_idx=ref, cols=cols), G, K)
    else:
        X = I.standardize(Gc.astype(float))
        T = I.top_eig(X @ X.T / X.shape[1], K)[1]
    name = f"n{n}_{scen}_r{rep}"
    d = os.path.join(workdir, name)
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, "g")
    fam = [[f"I{i}", lab[j], "0", "0", "0", "-9"] for i, j in enumerate(idx)]
    write_bed(pre, fam, [DATA["bim"][c] for c in cols], Gm)
    rows = []
    affset = {f"I{i}" for i in aff}
    for m, extra in METHODS.items():
        out = os.path.join(d, m)
        r = subprocess.run([PCAONE, "-b", pre, "-k", str(K), "-n", "2", "-o", out] + extra,
                           capture_output=True, text=True)
        if r.returncode != 0:
            rows.append((n, scen, rep, m) + (np.nan,) * 10)
            continue
        U = np.loadtxt(out + ".eigvecs")[:, :K]
        r2, err = B.score(U, T, len(idx), aff, K)
        npair = naff = 0
        if os.path.exists(out + ".relpairs"):
            for l in list(open(out + ".relpairs"))[1:]:
                a, b = l.split()[:2]
                npair += 1
                naff += a in affset or b in affset
        noise = np.nan
        if os.path.exists(out + ".noise") and len(aff):
            v = np.loadtxt(out + ".noise")
            oth = np.setdiff1d(np.arange(len(v)), aff)
            noise = np.median(v[aff]) / np.median(v[oth])
        rows.append((n, scen, rep, m, r2.min(), err, npair, naff, noise) + extra_scores(U, T, aff, lab[idx]))
    if keep_dir:
        np.savetxt(os.path.join(d, "affected.txt"), aff, fmt="%d")
        np.savetxt(os.path.join(d, "truth.txt"), T)
    else:
        shutil.rmtree(d)
    return rows


def main():
    bfile, workdir, out = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    jobs = [(n, s, r, workdir) for n in (5, 10, 20, 40) for s in SCEN for r in range(10)]
    jobs += [(0, s, r, workdir) for s in SCEN for r in range(3)]
    with ProcessPoolExecutor(workers, initializer=init, initargs=(bfile,)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    import pandas as pd
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "method", "minR2", "err_aff", "pairs", "pairs_aff",
                                    "noise_aff", "mev", "minsse", "err_pop", "err_pop_ref", "max_med"])
    d.to_csv(out, sep="\t", index=False)
    for v in ("minR2", "err_aff", "pairs", "noise_aff", "mev", "minsse", "err_pop", "err_pop_ref", "max_med"):
        print("==", v)
        print(d.pivot_table(index=["scen", "method"], columns="n", values=v).round(3).to_string())


if __name__ == "__main__":
    main()
