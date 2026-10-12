#!/usr/bin/env python3
"""PCP-pi on the reindeer (PLAN_pcp_pi.md step 6): F of every animal and the
kinship / IBD of the related pairs (Table 1 of the reindeer report: every pair
called by aarobust-kin, aarobust-kin+diag or dwg, or with KING > 2^-3.5), from
  pca<k>     top-k PCs of aarobust-kin+diag (aad_all), family left out (= PCAone for k = 10)
  pcp-*      PCP smoother (candidates: the pairs of aar/aad/dwg), leave-out none/self/pair/family
  dwg-*      dwg smoother, same candidates
and KING, R0 (plink2 counts).

usage: reindeer_pcp_pi.py <dir> <full|ld> <out prefix>
writes <out>_F.tsv and <out>_R.tsv; prints both
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import pcp_pi as Q  # noqa: E402
from pcp_pi_bench import lin_F, lin_pair  # noqa: E402
from reindeer_pairs_table import r0_table  # noqa: E402
from sim import read_bed  # noqa: E402

KS = (4, 6, 10)


def main():
    d, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fam, _, G = read_bed(os.path.join(d, "rd_ld" if mode == "ld" else "rd_full"))
    G = G.astype(np.int8)
    N = len(fam)
    ids = [r[1] for r in fam]
    pos = {x: i for i, x in enumerate(ids)}

    def pairs(run):
        rows = list(open(os.path.join(d, mode, run + ".relpairs")))
        return {tuple(sorted((pos[l.split()[0]], pos[l.split()[1]]))) for l in rows[1:]}

    called = {m: pairs(m + "_all") for m in ("aar", "aad", "dwg")}
    Kk = B.king_robust(G)
    iu = np.triu_indices(N, 1)
    ev = sorted(set().union(*called.values()) | {(int(i), int(j)) for i, j in zip(*iu) if Kk[i, j] > B.TAU})
    cand = np.zeros((N, N), bool)
    for i, j in set().union(*called.values()):
        cand[i, j] = cand[j, i] = True
    Gf = G.astype(float)
    Lp, Cp, _ = Q.pcp_parts(Gf, cand=cand)
    Ld, Cd, info = Q.dwg_parts(Gf, cand=cand)
    U = np.loadtxt(os.path.join(d, mode, "aad_all.eigvecs"))
    Gh = G / 2.0
    cs = Gh.sum(0)
    # F
    Fr = dict(FID=[r[0] for r in fam], IID=ids)
    for k in KS:
        V = np.hstack([U[:, :k], np.ones((N, 1))])
        H = V @ np.linalg.pinv(V)
        P = H @ Gh
        h = np.diag(H)[:, None]
        Fr[f"pca{k}"] = Q.inbreeding_from_pi(G, P)
        Fr[f"pca{k}-self"] = Q.inbreeding_from_pi(G, (P - h * Gh) / (1 - h))
    for nm, (Lt, C) in (("pcp", (Lp, Cp)), ("dwg", (Ld, Cd))):
        Fr[f"{nm}-none"] = lin_F(G, Lt, C, False)
        Fr[f"{nm}-self"] = lin_F(G, Lt, C, True)
    F = pd.DataFrame(Fr)
    # R
    r0 = r0_table(os.path.join(d, f"kt_{mode}_all.kin0"))
    comp_aad = Q.families(list(called["aad"]), N)
    comp_dwg = Q.families(list(called["dwg"]), N)
    rows = []
    for i, j in ev:
        row = dict(pair=f"{ids[i]}--{ids[j]}".replace("_reftest", ""),
                   site=fam[i][0] if fam[i][0] == fam[j][0] else f"{fam[i][0]}/{fam[j][0]}",
                   KING=Kk[i, j], R0=r0[tuple(sorted((ids[i], ids[j])))])
        for k in KS:
            Pk = Q.pi_pca(Gh, U[:, :k], [i, j], out=np.flatnonzero((comp_aad == comp_aad[i]) | (comp_aad == comp_aad[j])))
            row[f"pca{k}"], kk = Q.pair_rel(G, i, j, Pk[0], Pk[1])
            row[f"pca{k}_k2"] = kk[2]
        for nm, (Lt, C), comp in (("pcp", (Lp, Cp), comp_aad), ("dwg", (Ld, Cd), comp_dwg)):
            for how in ("none", "self", "pair", "family", "own"):
                pa, pb = lin_pair(Gh, cs, Lt, C, i, j, how, comp)
                row[f"{nm}-{how}"], kk = Q.pair_rel(G, i, j, pa, pb)
                row[f"{nm}-{how}_k2"] = kk[2]
        rows.append(row)
    R = pd.DataFrame(rows)
    F.to_csv(out + "_F.tsv", sep="\t", index=False)
    R.to_csv(out + "_R.tsv", sep="\t", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    cols = ["pair", "site", "KING", "R0"] + [c for c in R.columns if c.startswith(("pca", "pcp", "dwg"))
                                              and not c.endswith("_k2")]
    print(R[cols].round(3).to_string(index=False))
    print(F.groupby("FID")[[c for c in F.columns if c not in ("FID", "IID")]].mean().round(3).to_string())
    print("dwg fit", info)


if __name__ == "__main__":
    main()
