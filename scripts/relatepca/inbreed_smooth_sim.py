#!/usr/bin/env python3
"""Check of inbreed_smooth.py on admixTjeck2: n per population, plus 3 full sibs
(children of two CEU in the sample), an MZ copy of a CHB, one YRI inbred
(F = 0.5) and one MXL with excess heterozygosity (hom -> het, 0.2). Prints F
from the top 3/6/10 PCs and the PCP smoother (each with and without
leave-one-out); 'dwg' adds the dwg smoother.

usage: inbreed_smooth_sim.py <seed> <n per population> [dwg]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inbreed_smooth as IS  # noqa: E402
from sim import read_bed  # noqa: E402

BFILE = "/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2"
POPS = ["CEU", "CHB", "MXL", "YRI"]
_DATA = {}


def build(seed, n, scen="both", full=False):
    """genotypes, labels and group of each individual (other / relative /
    inbred / excess het; an inbred or excess-het relative is in the latter).

    scen  both     3 full sibs of two CEU and an MZ copy of a CHB (relatives),
                   one YRI inbred (F = 0.5), one MXL with excess heterozygosity
          rel      only the relatives
          inb      only the inbred YRI and the excess-het MXL
          relinb   the relatives, plus an inbred relative: a child of two of the
                   sibs (F = 0.25, related to the whole family), and the MZ copy
                   with excess heterozygosity
          controls for relinb (same family size and 15 pairs):
          rel_sib4     the relatives plus a 4th full sib (not inbred)
          rel_sib4het  the same, the MZ copy with excess heterozygosity
          rel_inbchild the relatives plus the child of two sibs, MZ copy normal
    full: also return dict(clean=genotypes before the changes, idx=rows of the
    data in admixTjeck2 (-1 for the added relatives), cols=SNP columns kept)"""
    if not _DATA:
        fam, _, G = read_bed(BFILE)
        _DATA.update(G=G, lab=np.array([r[1] for r in fam]))
    G, lab = _DATA["G"], _DATA["lab"]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(lab == p), n, replace=False) for p in POPS])
    X = G[idx].astype(np.int8)
    pops = list(lab[idx])
    rows = list(idx)

    def gam(g):  # one random allele per SNP (unlinked)
        return np.where(g == 1, rng.integers(0, 2, g.shape), g // 2)

    ceu = [i for i, p in enumerate(pops) if p == "CEU"]
    a, b = ceu[0], ceu[1]
    rel = scen in ("both", "rel", "relinb", "rel_sib4", "rel_sib4het", "rel_inbchild")
    if rel:
        kids = [gam(X[a]) + gam(X[b]) for _ in range(3)]  # 3 full sibs, children of a, b
        mz = X[pops.index("CHB")].copy()  # MZ copy of a CHB
        X = np.vstack([X] + kids + [mz]).astype(np.int8)
        pops += ["CEU"] * 3 + ["CHB"]
        rows += [-1] * 4
        if scen in ("relinb", "rel_inbchild"):  # child of two sibs: inbred (F = 0.25) by descent
            X = np.vstack([X, gam(kids[0]) + gam(kids[1])]).astype(np.int8)
            pops += ["CEU"]
            rows += [-1]
        elif scen in ("rel_sib4", "rel_sib4het"):  # a 4th full sib (not inbred)
            X = np.vstack([X, gam(X[a]) + gam(X[b])]).astype(np.int8)
            pops += ["CEU"]
            rows += [-1]
    group = ["other"] * len(X)
    if rel:
        for i in [a, b, pops.index("CHB")] + list(range(len(idx), len(X))):
            group[i] = "relative"
    clean = X.copy()
    if scen in ("both", "inb"):
        iy, im = pops.index("YRI"), pops.index("MXL")
    else:
        iy = im = None
    if scen in ("rel_sib4", "rel_sib4het"):
        group[len(X) - 1] = "relative"
    if scen in ("relinb", "rel_inbchild"):
        group[len(X) - 1] = "inbred"  # the child of the sibs
    if scen in ("relinb", "rel_sib4het"):
        im = len(idx) + 3  # the MZ copy: excess heterozygosity
        g = X[im].copy()
        hit = (g != 1) & (rng.random(g.shape) < 0.2)
        g[hit] = 1
        X[im] = g
        group[im] = "excess het"
    if iy is not None:
        g = X[iy].copy()
        hit = (g == 1) & (rng.random(g.shape) < 0.5)
        g[hit] = 2 * rng.integers(0, 2, hit.sum())
        X[iy] = g
        g = X[im].copy()
        hit = (g != 1) & (rng.random(g.shape) < 0.2)
        g[hit] = 1
        X[im] = g
        group[iy], group[im] = "inbred", "excess het"
    keep = X.min(0) != X.max(0)
    out = (X[:, keep], np.array(pops), np.array(group))
    if full:
        return out + (dict(clean=clean[:, keep], idx=np.array(rows), cols=np.flatnonzero(keep)),)
    return out


def main():
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    X, pops, group = build(seed, n)
    methods = ["k3", "k6", "k10", "pcp"] + (["dwg"] if "dwg" in sys.argv[3:] else [])
    res = {}
    for m in methods:
        W, info = IS.smoother(X, m)
        for loo in (False, True):
            res[m + ("_loo" if loo else "")] = IS.inbreeding(X, W, loo, info.get("raw", False))[0]
        print(m, {a: b for a, b in info.items() if a != "parts"})
    print(f"{'':16s}" + "".join(f"{k:>9s}" for k in res))
    for i in np.flatnonzero(group != "other"):
        print(f"{pops[i] + ' ' + group[i]:16s}" + "".join(f"{res[k][i]:+9.3f}" for k in res))
    o = group == "other"
    print(f"{'others mean':16s}" + "".join(f"{res[k][o].mean():+9.3f}" for k in res))
    print(f"{'others sd':16s}" + "".join(f"{res[k][o].std():9.3f}" for k in res))


if __name__ == "__main__":
    main()
