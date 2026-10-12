#!/usr/bin/env python3
"""Experimental: haploid dwg and haploid aarobust on a reindeer data set. One
allele drawn at random from each genotype (seed fixed), monomorphic sites
dropped; the methods of inbreed_pca_hap.py:

  hapdwg  haploid dwg fit (kinship rule phi = R / sqrt(v v) > tau, haploid
          noise edge, rank at most k + 1) with the haploid candidate screen
          (inbreed_pca_hap.hap_ea_iter: the fit's pairs | haploid evalAdmix,
          population rule for complete cliques of >= 5 animals); PCs after whitening (hapdwg_all.eigvecs) and of the
          structure part (hapfit_all.eigvecs); the fitted noise v of each animal
          (hapdwg_all.noise; weight 1 / sqrt(v)) and the pairs (hapdwg_all.pairs)
  hapaar  PCP with the kinship rule on the haploid scale (tau / 2) and the
          diagonal unobserved, on the uncentred haploid GRM (centring hurts:
          inbreeding report), with the screen's candidates; PCs of its centred
          structure part (hapaar_all.eigvecs, pairs hapaar_all.pairs): the
          haploid aarobust-kin+diag

usage: reindeer_hap.py <run dir: new/full, old/ld, ...> [k]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import illustrate as I  # noqa: E402
import inbreed_pca_hap as HP  # noqa: E402
from sim import read_bed  # noqa: E402


def centred_grm_hap(H):
    f = H.mean(0)
    X = (H - f) / np.sqrt(f * (1 - f))
    return X @ X.T / H.shape[1]


def write_pairs(path, names, pairs):
    with open(path, "w") as o:
        o.write("ID1\tID2\tKINSHIP\n")
        for i, j, phi in pairs:
            o.write(f"{names[i]}\t{names[j]}\t{phi:.4f}\n")


def main():
    d = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    bfile = os.path.join(d, "..", "rd_full" if os.path.basename(os.path.normpath(d)) == "full" else "rd_ld")
    fam, _, G = read_bed(bfile)
    names = [r[1] for r in fam]
    H = HP.haploid(G, np.random.default_rng(20261008))
    H = H[:, H.min(0) != H.max(0)].astype(np.float64)
    U, Uw, called, r, ex = HP.hap_ea_iter(H, k=k, full=True, pop_min=HP.POP_MIN, union=True)
    np.savetxt(os.path.join(d, "hapdwg_all.eigvecs"), Uw, fmt="%.8f")
    np.savetxt(os.path.join(d, "hapfit_all.eigvecs"), U, fmt="%.8f")
    np.savetxt(os.path.join(d, "hapdwg_all.noise"), ex["v"], fmt="%.6f")
    # the fit reports R / (2 sqrt(v v)); haploid kinship is twice that
    write_pairs(os.path.join(d, "hapdwg_all.pairs"), names, [(i, j, 2 * phi) for i, j, phi in ex["pairs"]])
    A = HP.grm_hap(H)[0]
    L, S, _ = I.pcp_kin(A, B.TAU / 2, free_diag=True, cand=ex["cand"])
    Pc = np.eye(len(A)) - 1 / len(A)
    Ua = np.linalg.eigh(Pc @ L @ Pc)[1][:, ::-1][:, :k]
    np.savetxt(os.path.join(d, "hapaar_all.eigvecs"), Ua, fmt="%.8f")
    v = np.diag(A - L)
    iu = np.triu_indices(len(A), 1)
    pa = [(int(i), int(j), float(S[i, j] / np.sqrt(v[i] * v[j]))) for i, j in zip(*iu) if abs(S[i, j]) > 1e-12]
    write_pairs(os.path.join(d, "hapaar_all.pairs"), names, pa)
    print(d, "SNPs", H.shape[1], "hapdwg rank", r, "pairs", len(called), "| hapaar pairs", len(pa),
          "| candidates", int(ex["cand"].sum() // 2))


if __name__ == "__main__":
    main()
