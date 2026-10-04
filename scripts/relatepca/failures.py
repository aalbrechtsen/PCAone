#!/usr/bin/env python3
"""One scenario that shows how each non-shortlisted method fails.

The 40 unrelated individuals of illustrate.py (10 each of CEU, CHB, MXL, YRI,
seed 1) plus two families made from them: two MZ copies of a CEU individual
and one child of two MXL. The first breaks the methods that cannot stop a
family from taking a PC; the second (an admixed family) breaks those that
shrink or mis-place relatives. One panel per method, PC2 vs PC3 after an
orthogonal Procrustes rotation onto the truth (display only), with
the smallest R2 over the K-1 true PCs and the relatives' error in the
title (red: min R2 < 0.97 or error > 0.15; scored against a reference panel of the unused individuals).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import illustrate as I  # noqa: E402
from sim import Dropper, read_bed, write_bed  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--out", required=True, help="output prefix (pdf + tsv)")
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--tau", type=float, default=2 ** -3.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    k = 3
    pops = ["CEU", "CHB", "MXL", "YRI"]

    fam, bim, Gall = read_bed(a.bfile)
    lab = np.array([r[1] for r in fam])
    idx = np.concatenate([rng.choice(np.where(lab == p)[0], 10, replace=False) for p in pops])
    G0 = Gall[idx].astype(float)
    labels = list(lab[idx])
    poly = G0.std(0) > 0
    G0, bim = G0[:, poly], [l for l, q in zip(bim, poly) if q]
    n0 = len(G0)
    drop = Dropper(bim, rng)

    # two families, built from the sample
    parents, pairs_phi, Rs = [], [], []
    for kind, phi in [("mz_2", 0.5), ("child_MXL_1", 0.25)]:
        R, par, _ = I.make_relatives(kind, G0, labels, rng, drop)
        off = n0 + sum(len(r) for r in Rs)
        fam_pairs = I.related_pairs(n0, par)  # indices assuming the relatives start at n0
        shift = lambda x: x if x < n0 else x - n0 + off  # noqa: E731
        pairs_phi += [(shift(i), shift(j), phi) for i, j in fam_pairs]
        parents += par
        Rs.append(R)
    G = np.vstack([G0] + Rs)
    N = len(G)
    base, rel = np.arange(n0), np.arange(n0, N)
    pairs = sorted((min(i, j), max(i, j)) for i, j, _ in pairs_phi)
    print(f"N = {N}; related pairs (pedigree): {pairs_phi}")

    # truths
    X0 = I.standardize(G0)
    _, T0 = I.top_eig(X0 @ X0.T / X0.shape[1], k)
    _, Tc0 = I.cs_pcs(I.cs_matrix(G0), k)
    T = np.vstack([T0, [(T0[p] + T0[q]) / 2 for p, q in parents]])
    Tc = np.vstack([Tc0, [(Tc0[p] + Tc0[q]) / 2 for p, q in parents]])

    # independent truth: reference panel of all unused individuals of these
    # populations; every individual of the data projected onto its PCA
    ref_idx = np.setdiff1d(np.where(np.isin(lab, pops))[0], idx)
    Gr = Gall[ref_idx][:, poly].astype(float)
    fr = Gr.mean(0) / 2
    okr = (fr > 0) & (fr < 1)
    sdr = np.sqrt(2 * fr[okr] * (1 - fr[okr]))
    Xr = (Gr[:, okr] - 2 * fr[okr]) / sdr
    Tref = ((G[:, okr] - 2 * fr[okr]) / sdr) @ (Xr.T @ I.top_eig(Xr @ Xr.T / okr.sum(), k)[1])
    Tref /= np.linalg.norm(Tref, axis=0)  # unit-length columns, as eigenvectors

    X = I.standardize(G)
    M = X.shape[1]
    C = X @ X.T / M
    H = I.cs_matrix(G)
    Dh = (G * (2 - G)).mean(axis=1)
    Graw = G @ G.T / M

    # KING kinship from the same data
    d = a.out + "_work"
    os.makedirs(d, exist_ok=True)
    ids = [f"I{i}" for i in range(N)]
    write_bed(f"{d}/test", [["F", x, "0", "0", "0", "-9"] for x in ids], bim, G.astype(np.int8))
    subprocess.run(f"{a.plink2} --bfile {d}/test --make-king-table --king-table-filter 0.04 --threads 4 "
                   f"--out {d}/king", shell=True, check=True, capture_output=True)
    king = [l.split() for l in open(f"{d}/king.kin0").read().strip().split("\n")[1:]]
    king_phi = [(int(l[1][1:]), int(l[3][1:]), float(l[-1])) for l in king if float(l[-1]) >= a.tau]
    cand = np.zeros((N, N), dtype=bool)  # KING screen: plink2 pairs > 0.04
    for l in king:
        i_, j_ = int(l[1][1:]), int(l[3][1:])
        cand[i_, j_] = cand[j_, i_] = True
    print("KING pairs >= tau:", king_phi)

    lam2 = 2 / np.sqrt(N)
    S = {}
    _, S["standard PCA"] = I.top_eig(C, k)
    _, S["Zhou substitution"] = I.top_eig(I.zhou_ms(C, pairs), k)
    _, S["masked low-rank, GRM"] = I.top_eig(I.masked_lr(C, pairs, k)[0], k)
    S["robust PCA of data matrix"] = np.linalg.svd(I.rpca(X)[0], full_matrices=False)[0][:, :k]
    _, S["robust SVD, default $\\lambda$"] = I.top_eig(I.rpca(C)[0], k)
    _, S["robust SVD, $\\lambda=2/\\sqrt{N}$"] = I.top_eig(I.rpca(C, lam2)[0], k)
    _, S["AArobust (GRM, free diag.)"] = I.top_eig(I.rpca(C, free_diag=True)[0], k)
    _, S["AArobust, kinship thr. + screen"] = I.top_eig(I.pcp_kin(C, a.tau, True, cand=cand)[0], k)
    _, S["HeteroPCA, GRM"] = I.top_eig(I.masked_lr(C, [], k)[0], k)
    S["HeteroPCA+kinship, raw Gram"] = I.cs_pcs(I.lr_kin(Graw, k + 1, a.tau, True)[0], k)[1]
    S["Chen & Storey (CS)"] = I.cs_pcs(H, k)[1]
    S["weighted PCA, KING"] = I.weighted_pca(X, king_phi, k)
    S["auto whitening (evalAdmix)"] = I.white_auto(G, X, k, a.tau)[0]
    # shortlisted, for contrast
    S["SHORTLIST: whitening, KING"] = I.whitened_numpy(X, king_phi, k)
    S["SHORTLIST: CS whitening, KING"] = I.whitened_cs(G, king_phi, k)
    S["SHORTLIST: fixed rank+kinship, CS"] = I.cs_pcs(I.lr_kin(H, k + 1, a.tau, False, Dh)[0], k)[1]
    # versions of the shortlisted methods
    v2 = {}
    v2["whitening, GLS freq., pedigree"] = I.whitened_gls(G, pairs_phi, k)
    v2["whitening, GLS freq., KING"] = I.whitened_gls(G, king_phi, k)
    U_h, hyb_pairs = I.hyb_cs_white(G, H, Dh, k, a.tau)
    v2["detect (lrkin CS) + CS whitening"] = U_h
    v2["detect (lrkin CS) + whitening GLS"] = I.whitened_gls(G, hyb_pairs, k)
    v2["robust start + evalAdmix + CS whit."] = I.auto_robust(G, H, Dh, k, a.tau)[0]
    v2["lrkin CS, weighted final fit"] = I.lrkin_cs_weighted(H, Dh, k, a.tau)
    U_a, K_a, _ = I.lrkin_cs_autoK(G, H, Dh, a.tau)
    v2[f"lrkin CS, automatic K (K={K_a})"] = U_a
    Lk, Sk, _ = I.lr_kin(H, k + 1, a.tau, False, Dh, cand=cand)
    v2["fixed rank+kinship CS, screen"] = I.cs_pcs(Lk, k)[1]
    v2["detect (screen) + CS whitening"] = I.whitened_cs(G, I.lr_kin_pairs(H, Dh, Lk, Sk), k)
    v2["CS whitening, KING"] = I.whitened_cs(G, king_phi, k)
    v2["AArobust, kinship thr. + screen"] = S["AArobust, kinship thr. + screen"]
    print("lrkin CS kinship of detected pairs:", [(i, j, round(f, 3)) for i, j, f in hyb_pairs])
    cs_based = {"detect (lrkin CS) + CS whitening", "robust start + evalAdmix + CS whit.",
                "lrkin CS, weighted final fit", f"lrkin CS, automatic K (K={K_a})","HeteroPCA+kinship, raw Gram", "Chen & Storey (CS)", "SHORTLIST: CS whitening, KING",
                "SHORTLIST: fixed rank+kinship, CS"}

    def score_and_plot(methods_dict, fname, ncol=4):
        panels = [("truth: reference panel", Tref, None)]
        rows = []
        for m, Y in methods_dict.items():
            Y = I.align(Y, Tref, base)
            r2, err = I.scores_vs_truth(Y, Tref, base, rel)
            # display only: orthogonal Procrustes (rotation, one scale, shift) of all
            # k PCs onto the truth, fitted on the unrelated individuals
            Yc, Tb = Y - Y[base].mean(0), Tref[base] - Tref[base].mean(0)
            U_, d_, Vt_ = np.linalg.svd(Yc[base].T @ Tb)
            Yd = Yc @ (U_ @ Vt_) * (d_.sum() / (Yc[base] ** 2).sum()) + Tref[base].mean(0)
            panels.append((m, Yd, (r2.min(), err)))
            rows.append((m, r2, err))
            print(f"{m:40s} R2 {np.round(r2, 3)}  relatives err {err:.2f}")

        col = I.COL
        mz = np.arange(n0, n0 + 2)
        src = int(np.where(np.array(labels) == "CEU")[0][0])
        child = np.array([N - 1])
        mxl_par = [p for p in parents[-1]]
        nc = ncol
        nr = int(np.ceil(len(panels) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.9 * nr), squeeze=False)
        for ax, (name, Y, sc) in zip(axes.flat, panels):
            for p in pops:
                ii = [i for i in range(n0) if labels[i] == p]
                ax.scatter(Y[ii, 1], Y[ii, 2], s=20, c=col[p], label=p, edgecolors="none")
            ax.scatter(Y[[src], 1], Y[[src], 2], s=60, facecolors="none", edgecolors="k", linewidths=1,
                       label="MZ source")
            ax.scatter(Y[mz, 1], Y[mz, 2], s=45, c="#e7298a", marker="^", edgecolors="k", linewidths=0.4,
                       label="MZ copies")
            ax.scatter(Y[mxl_par, 1], Y[mxl_par, 2], s=60, facecolors="none", edgecolors="#7570b3",
                       linewidths=1.2, label="MXL parents")
            ax.scatter(Y[child, 1], Y[child, 2], s=55, c="#e6ab02", marker="*", edgecolors="k", linewidths=0.4,
                       label="MXL child")
            title = name if sc is None else f"{name}\nmin $R^2$={sc[0]:.3f}, error={sc[1]:.2f}"
            bad = sc is not None and (sc[0] < 0.97 or sc[1] > 0.15)
            ax.set_title(title, fontsize=9.5, color="#b2182b" if bad else ("#1a7f37" if sc else "k"))
            ax.set_xlabel("PC2", fontsize=8)
            ax.set_ylabel("PC3", fontsize=8)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[len(panels):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=6.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(fname)

        return rows

    rows = score_and_plot(S, a.out + ".pdf")
    v2_all = {"SHORTLIST: whitening, KING": S["SHORTLIST: whitening, KING"],
              "SHORTLIST: CS whitening, KING": S["SHORTLIST: CS whitening, KING"],
              "SHORTLIST: fixed rank+kinship, CS": S["SHORTLIST: fixed rank+kinship, CS"]}
    v2_all.update(v2)
    rows += score_and_plot(v2_all, a.out + "_v2.pdf")
    # compact overview: the reference failure, the robust variants and the shortlist
    ov = {"standard PCA": S["standard PCA"],
          "AArobust (default $\\lambda$)": S["AArobust (GRM, free diag.)"],
          "AArobust, kinship thr. + screen": S["AArobust, kinship thr. + screen"],
          "GRM whitening, KING": S["SHORTLIST: whitening, KING"],
          "CS whitening, KING": v2["CS whitening, KING"],
          "fixed rank + kinship, CS, screen": v2["fixed rank+kinship CS, screen"],
          "detect (screen) + CS whitening": v2["detect (screen) + CS whitening"],
          "robust start + evalAdmix + CS whit.": v2["robust start + evalAdmix + CS whit."]}
    score_and_plot(ov, a.out + "_overview.pdf", ncol=3)
    with open(a.out + ".tsv", "w") as f:
        f.write("method\tR2_PC1\tR2_PC2\tR2_PC3\terr_rel\n")
        for m, r2, err in rows:
            f.write(f"{m}\t" + "\t".join(f"{x:.4f}" for x in r2) + f"\t{err:.4f}\n")


if __name__ == "__main__":
    main()
