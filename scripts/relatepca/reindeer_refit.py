#!/usr/bin/env python3
"""Experimental: aarobust-kin+refit on a reindeer data set. The pairs called
by aarobust-kin --impute-diag (<run dir>/aad_all.relpairs), then a rank-limited
fit of the uncentred scaled GRM with the diagonal and these pairs left out
(dw_grm.fit, rank from the noise edge, at most k + 1, as dwg). Writes the top k
PCs of its centred structure part (refit_all.eigvecs, not whitened) and the
PCs after dwg's whitening by the fitted noise and these pairs
(refitw_all.eigvecs), and the pairs it keeps (refit_all.pairs).

usage: reindeer_refit.py <run dir: new/full, old/ld, ...> [k]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dw_grm as DW  # noqa: E402
import illustrate as I  # noqa: E402
import inbreed_smooth as IS  # noqa: E402
from sim import read_bed  # noqa: E402


def main():
    d = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    bfile = os.path.join(d, "..", "rd_full" if os.path.basename(os.path.normpath(d)) == "full" else "rd_ld")
    fam, _, G = read_bed(bfile)
    ids = {r[1]: i for i, r in enumerate(fam)}
    N = len(fam)
    cand = np.zeros((N, N), bool)
    for l in list(open(os.path.join(d, "aad_all.relpairs")))[1:]:
        x, y = l.split()[:2]
        cand[ids[x], ids[y]] = cand[ids[y], ids[x]] = True
    A, f, w = IS.scaled_uncentred(G)
    L, S, r, pairs = DW.fit(A, B.TAU, cand, DW.noise_edge(f, w, N), k + 1, None)
    P = np.eye(N) - 1 / N
    U = np.linalg.eigh(P @ L @ P)[1][:, ::-1][:, :k]
    np.savetxt(os.path.join(d, "refit_all.eigvecs"), U, fmt="%.8f")
    v = np.maximum(np.diag(A) - np.diag(L), 1e-6)
    Uw = I.whitened_noise(A, v, pairs, k, centre=True)
    np.savetxt(os.path.join(d, "refitw_all.eigvecs"), Uw, fmt="%.8f")
    names = [r_[1] for r_ in fam]
    with open(os.path.join(d, "refit_all.pairs"), "w") as o:
        o.write("ID1\tID2\tKINSHIP\n")
        for i, j, phi in pairs:
            o.write(f"{names[i]}\t{names[j]}\t{phi:.4f}\n")
    print(d, "rank", r, "pairs", len(pairs), "of", int(cand.sum() // 2), "candidates")


if __name__ == "__main__":
    main()
