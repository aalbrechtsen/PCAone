#!/usr/bin/env python3
"""Failure example on admixTjeck2 for the parts reports: a small population
(2 YRI among 5 CEU, CHB, MXL), no relatives. PCs of the truth and of four
combinations (standard PCA, dwg-like with the family-axis rule,
detect-white without it, PCP = aarobust-kin --impute-diag), raw PC1-PC2 and PC2-PC3.

usage: parts_fail_fig.py OUT.pdf [n rep M small]"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import parts_bench as P  # noqa: E402

PCOL = {"CEU": "#0072B2", "CHB": "#E69F00", "MXL": "#009E73", "YRI": "#CC79A7"}
METHODS = [("standard", "standard PCA"),
           ("topr/unc/kf/0/v/whitec", "top r, uncentered GRM,\nfamily rule, whitened (dwg-like)"),
           ("topr/raw/k/0/D/white", "top r, raw Gram, no family\nrule, whitened (detect-white)"),
           ("pcp/cgrm/k/0/v/L", "PCP, GRM, PCs of L\n(aarobust-kin --impute-diag)")]


def main(out, n=5, rep=5, M=0, small="YRI2"):
    P.init("admix")
    ds = P.dataset(n, M, "none", rep)
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(float)
    bp = np.array(ds["base_pop"])
    sp, ns = small[:-1], int(small[-1])
    keep = np.array([i for i in range(len(G)) if bp[i] != sp or (bp[:i] == sp).sum() < ns])
    G, T, pop = G[keep], T[keep], bp[keep]
    N = len(G)
    ck = (B.king_robust(G) > B.KING_SCREEN) & ~np.eye(N, dtype=bool)
    res = [("truth", T, None)]
    for c, lab in METHODS:
        U = P.standard(G, 3) if c == "standard" else P.run_combo(G, c, 3, ck)[0]
        r2, _ = B.score(U, T, N, np.array([], int), 3)
        res.append((lab, U, r2.min()))
    fig, ax = plt.subplots(2, len(res), figsize=(2.6 * len(res), 5.0))
    for j, (lab, U, m) in enumerate(res):
        for i, (a, b) in enumerate([(0, 1), (1, 2)]):
            x = ax[i, j]
            for p in PCOL:
                s = pop == p
                x.scatter(U[s, a], U[s, b], s=22 if p != sp else 40, color=PCOL[p], label=p,
                          edgecolor="k" if p == sp else "none", lw=0.6)
            x.set_xticks([])
            x.set_yticks([])
            x.set_xlabel(f"PC{a + 1}", fontsize=8)
            x.set_ylabel(f"PC{b + 1}", fontsize=8)
        title = "truth (reference panel)" if m is None else f"{lab}\nmin R² = {m:.3f}"
        ax[0, j].set_title(title, fontsize=8, color="#b00" if (m is not None and m < 0.95) else "k")
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"admixTjeck2: {n} CEU, CHB, MXL and only {ns} {sp}, no relatives, "
                 f"{'all' if M == 0 else M} SNPs", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:4]), *(a[4:5]))
