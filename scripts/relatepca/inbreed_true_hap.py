#!/usr/bin/env python3
"""True haploid vs pseudo-haploid input. The designs of inbreed_smooth_sim.build
(admixTjeck2, n per population, 3 full sibs of two CEU, MZ copy of a CHB, inbred
YRI, excess-het MXL, child of two sibs), but the relatives are made from
haplotypes with recombination (1 cM / Mb, Poisson crossovers on the bim
positions), so relatives share IBD segments. The real founders are unphased:
their two haplotypes are a random phasing of their genotypes. Every founder
haplotype carries a label, copied with it, so the realised IBD of any two
haplotypes is the fraction of SNPs with the same label.

Two inputs from the same diploid individuals:
  pseudo  one allele drawn at random at every SNP (as inbreed_pca_hap.py)
  true    one whole haplotype of each individual (chosen at random): a single
          genome copy, as a haploid organism, a male X or one phased haplotype

Realised kinship of a pair: pseudo, the mean IBD of the 4 haplotype pairs (the
expectation of the random draws); true, the IBD of the two chosen haplotypes.
"realised pairs" are those with realised kinship > tau; the pedigree pairs are
those of inbreed_pca_bench.true_pairs. Methods as in inbreed_pca_hap.py: PCAone
--haploid (standard, aarobust-diag, dwg), the haploid dwg fit with the union
screen and the population rule (unweighted and whitened), haploid aarobust
(uncentred PCP with the same candidates), and the haploid dwg fit with the
realised pairs as candidates (best case).

usage: inbreed_true_hap.py <workdir> <out.tsv> [workers] [scenarios]
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
import inbreed_pca_hap as HP  # noqa: E402
import inbreed_smooth_sim as SIM  # noqa: E402
from inbreed_pca_bench import K, scores, true_pairs  # noqa: E402
from inbreeding_bench import PCAONE  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

SCEN = ["none", "rel", "both", "relinb"]
_POS = {}


def positions():
    if not _POS:
        bim = list(open(SIM.BFILE + ".bim"))
        chrom = np.array([int(l.split()[0]) for l in bim])
        pos = np.array([int(l.split()[3]) for l in bim]) / 1e8  # Morgan, 1 cM / Mb
        _POS.update(chrom=chrom, pos=pos)
    return _POS["chrom"], _POS["pos"]


def gamete(h, lab, rng):
    """one gamete of an individual with haplotypes h (2 x M) and labels lab"""
    chrom, pos = positions()
    src = np.empty(h.shape[1], np.int64)
    for c in np.unique(chrom):
        k = np.flatnonzero(chrom == c)
        p = pos[k]
        x = np.sort(rng.uniform(p[0], p[-1], rng.poisson(p[-1] - p[0])))
        src[k] = (rng.integers(0, 2) + np.searchsorted(x, p)) % 2
    j = np.arange(h.shape[1])
    return h[src, j], lab[src, j]


def build(seed, n, scen):
    """haplotypes (N x 2 x M), labels, pops, group, dict(clean, idx, cols)"""
    if not SIM._DATA:
        fam, _, G = read_bed(SIM.BFILE)
        SIM._DATA.update(G=G, lab=np.array([r[1] for r in fam]))
    G, lab = SIM._DATA["G"], SIM._DATA["lab"]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(lab == p), n, replace=False) for p in SIM.POPS])
    X = G[idx].astype(np.int8)
    N0, M = X.shape
    h1 = np.where(X == 1, rng.integers(0, 2, X.shape), X // 2)  # random phase
    hap = [np.stack([a, g - a]).astype(np.int8) for a, g in zip(h1, X)]
    labs = [np.stack([np.full(M, 2 * i), np.full(M, 2 * i + 1)]) for i in range(N0)]
    pops, rows = list(lab[idx]), list(idx)

    def child(a, b):
        ga, la = gamete(hap[a], labs[a], rng)
        gb, lb = gamete(hap[b], labs[b], rng)
        hap.append(np.stack([ga, gb]))
        labs.append(np.stack([la, lb]))

    ceu = [i for i, p in enumerate(pops) if p == "CEU"]
    a, b = ceu[0], ceu[1]
    rel = scen in ("rel", "both", "relinb", "rel_inbchild")
    if rel:
        for _ in range(3):
            child(a, b)
        c = pops.index("CHB")
        hap.append(hap[c].copy())
        labs.append(labs[c].copy())
        pops += ["CEU"] * 3 + ["CHB"]
        rows += [-1] * 4
        if scen in ("relinb", "rel_inbchild"):
            child(N0, N0 + 1)
            pops += ["CEU"]
            rows += [-1]
    N = len(hap)
    group = ["other"] * N
    if rel:
        for i in [a, b, pops.index("CHB")] + list(range(N0, N)):
            group[i] = "relative"
    clean = np.array([h.sum(0) for h in hap], np.int8)
    if scen in ("relinb", "rel_inbchild"):
        group[N - 1] = "inbred"
    het = []
    if scen == "both":  # inbred YRI (het -> hom, 0.5) and excess-het MXL (hom -> het, 0.2)
        iy, im = pops.index("YRI"), pops.index("MXL")
        h = hap[iy]
        hit = (h[0] != h[1]) & (rng.random(M) < 0.5)
        al = rng.integers(0, 2, M)
        h[0, hit] = h[1, hit] = al[hit]
        group[iy] = "inbred"
        het = [im]
    if scen == "relinb":
        het = [N0 + 3]
    for im in het:
        h = hap[im]
        hit = (h[0] == h[1]) & (rng.random(M) < 0.2)
        h[1, hit] = 1 - h[1, hit]
        group[im] = "excess het"
    H2 = np.stack(hap)  # N x 2 x M
    L2 = np.stack(labs)
    X = H2.sum(1)
    keep = X.min(0) != X.max(0)
    return (H2[:, :, keep], L2[:, :, keep], np.array(pops), np.array(group),
            dict(clean=clean[:, keep], idx=np.array(rows), cols=np.flatnonzero(keep)))


def realised(L2, pick=None):
    """realised kinship matrix: pick=None, mean IBD of the 4 haplotype pairs;
    else the IBD of the chosen haplotypes pick[i]"""
    N = len(L2)
    K_ = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if pick is None:
                v = np.mean([(L2[i, x] == L2[j, y]).mean() for x in (0, 1) for y in (0, 1)])
            else:
                v = (L2[i, pick[i]] == L2[j, pick[j]]).mean()
            K_[i, j] = K_[j, i] = v
    return K_


def one(job):
    scen, n, seed, workdir = job
    H2, L2, pops, group, ex = build(seed, n, scen)
    Gall, lab = SIM._DATA["G"], SIM._DATA["lab"]
    used = ex["idx"][ex["idx"] >= 0]
    ref = np.setdiff1d(np.flatnonzero(np.isin(lab, SIM.POPS)), used)
    T = B.truth_reference(dict(G=ex["clean"].astype(float), ref_idx=ref, cols=ex["cols"]), Gall, K)
    N = len(H2)
    n0 = 4 * n
    tp = true_pairs(group, pops, n0) if scen != "none" else set()
    rng = np.random.default_rng(seed + 7919)
    rows = []
    for mode in ("pseudo", "true"):
        if mode == "pseudo":
            H = HP.haploid(H2.sum(1), rng)
            Kr = realised(L2)
        else:
            pick = rng.integers(0, 2, N)
            H = H2[np.arange(N), pick]
            Kr = realised(L2, pick)
        H = H[:, H.min(0) != H.max(0)]
        iu = np.triu_indices(N, 1)
        rp = {(int(i), int(j)) for i, j in zip(*iu) if Kr[i, j] > B.TAU}
        kped = [Kr[i, j] for i, j in sorted(tp)]

        def add(m, U, called):
            r2, mev, g = scores(U, T, pops, group)
            rows.append(dict(scen=scen, n=n, seed=seed, mode=mode, method=m, minR2=r2, mev=mev,
                             pairs=len(called), ped_pairs=len(tp), ped_found=len(called & tp),
                             real_pairs=len(rp), real_found=len(called & rp),
                             ped_kin_mean=np.mean(kped) if kped else np.nan,
                             ped_below=int(np.sum(np.array(kped) <= B.TAU)) if kped else 0,
                             **{f"err_{x.replace(' ', '_')}": g[x] for x in g}))

        d = os.path.join(workdir, f"th_{mode}_{scen}_n{n}_s{seed}")
        os.makedirs(d, exist_ok=True)
        pre = os.path.join(d, "h")
        write_bed(pre, [[pops[i], f"I{i}", "0", "0", "0", "-9"] for i in range(N)],
                  [f"1\ts{j}\t0\t{j + 1}\tA\tC" for j in range(H.shape[1])], (2 * H).astype(np.int8))
        for m, extra in (("standard", []), ("aarobust-diag", ["--impute-diag"]), ("dwg", ["--robust", "dwg"])):
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
            add("PCAone " + m, np.loadtxt(out + ".eigvecs")[:, :K], called)
        shutil.rmtree(d)
        U, Uw, called, _, exs = HP.hap_ea_iter(H, full=True, pop_min=HP.POP_MIN, union=True)
        add("dwg-fit, screen", U, called)
        add("dwg-fit, screen, whitened", Uw, called)
        Ua, ca = HP.hap_aarobust(H, exs["cand"], centred=False)
        add("aarobust, screen", Ua, ca)
        cand = np.zeros((N, N), bool)
        for i, j in rp:
            cand[i, j] = cand[j, i] = True
        U, Uw, called, _ = HP.hap_fit(H, cand)
        add("dwg-fit, realised pairs", U, called)
        add("dwg-fit, realised pairs, whitened", Uw, called)
        Ua, ca = HP.hap_aarobust(H, cand, centred=False)
        add("aarobust, realised pairs", Ua, ca)
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
    for v in ("minR2", "err_relative", "err_other", "pairs", "ped_found", "real_pairs", "real_found",
              "ped_kin_mean", "ped_below"):
        print("==", v)
        print(d.pivot_table(index=["mode", "method"], columns=["scen", "n"], values=v).round(2).to_string())


if __name__ == "__main__":
    main()
