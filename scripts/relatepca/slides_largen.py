#!/usr/bin/env python3
"""Slide figures for the large-N symptom: the highn_test.py dataset (N=2000,
30% in families, rep 0), standard PCAone vs --robust, both with -k 20.

  largen_placement.pdf    relatives' true position -> PCA position (arrows),
                          on the two truth axes where standard PCA errs most
  largen_familyaxes.pdf   PC5 vs PC6 (the first PCs beyond the 4 ancestry
                          axes) with the largest families coloured, and the
                          family share eta^2 of every PC

usage: slides_largen.py <PCAone> <workdir> <outdir>
"""
import os
import subprocess
import sys
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import highn_test as H  # noqa: E402
from sim import write_bed  # noqa: E402

N, REL, REP, M, NREF, K = 2000, 0.3, 0, 20000, 3000, 20
GREY, REDC, BLUE = "#BBBBBB", "#D55E00", "#0072B2"
FAMC = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def data(pcaone, work):
    os.makedirs(work, exist_ok=True)
    pre = os.path.join(work, "largen")
    seed = zlib.crc32(f"{N}|{REL}|{REP}".encode())
    G, famid, ftype, R = H.simulate(N, REL, M, NREF, seed)
    keep = G.min(0) != G.max(0)
    G, R = G[:, keep], R[:, keep]
    T = H.reference_truth(R, G, len(H.FST) - 1)
    bim = [f"1\tsnp{i}\t0\t{i + 1}\tA\tC" for i in range(G.shape[1])]
    write_bed(pre, [[str(famid[i]) if famid[i] else f"U{i}", f"I{i}", "0", "0", "0", "-9"] for i in range(N)],
              bim, G)
    U = {}
    for name, extra in [("standard", ""), ("robust", "--robust")]:
        out = f"{pre}_{name}"
        if not os.path.exists(out + ".eigvecs"):
            subprocess.run(f"{pcaone} -b {pre} -k {K} -n 16 -v 1 {extra} -o {out}", shell=True, check=True,
                           capture_output=True)
        U[name] = np.loadtxt(out + ".eigvecs", ndmin=2)
    return U, T, famid, ftype


def mapped(U, T, base, kk):
    """method PCs mapped onto the truth by least squares on the unrelated"""
    Z = np.c_[U[:, :kk], np.ones(len(U))]
    B, *_ = np.linalg.lstsq(Z[base], T[base, :kk], rcond=None)
    return Z @ B


def placement(U, T, famid, out):
    kk = T.shape[1]
    base, rel = np.where(famid == 0)[0], np.where(famid > 0)[0]
    P = {m: mapped(U[m], T, base, kk) for m in U}
    e = ((T[rel] - P["standard"][rel]) ** 2).mean(0)
    a, b = np.sort(np.argsort(e)[-2:])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, m, title in zip(axs, ["standard", "robust"], ["standard PCA", "robust (dwg)"]):
        _, err = H.score(U[m], T, base, rel, kk)
        ax.scatter(T[base, a], T[base, b], s=4, c=GREY, lw=0)
        for i in rel:
            ax.annotate("", (P[m][i, a], P[m][i, b]), (T[i, a], T[i, b]),
                        arrowprops=dict(arrowstyle="-|>", color=REDC, lw=0.6, mutation_scale=6, alpha=0.7))
        ax.scatter(T[rel, a], T[rel, b], s=5, c="k", lw=0, zorder=3)
        ax.set_title(f"{title}: relatives' error {err:.2f}", fontsize=11)
        ax.set_xlabel(f"true PC{a + 1}")
        ax.set_ylabel(f"true PC{b + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    lim = [(min(x.get_xlim()[0] for x in axs), max(x.get_xlim()[1] for x in axs)),
           (min(x.get_ylim()[0] for x in axs), max(x.get_ylim()[1] for x in axs))]
    for ax in axs:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
    fig.text(0.5, 0.01, "black: a relative's true position; arrow: where PCA puts it.  grey: unrelated",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return a, b


def familyaxes(U, famid, out, kk=4):
    eta = {m: np.array([H.family_eta2(U[m][:, j], famid) for j in range(K)]) for m in U}
    # the families that dominate standard PC5 / PC6
    fams = [f for f in np.unique(famid[famid > 0])]
    load = {f: np.abs(U["standard"][famid == f, kk:kk + 2]).sum() for f in fams}
    top = sorted(fams, key=lambda f: -load[f])[:len(FAMC)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, m, title in zip(axs[:2], ["standard", "robust"], ["standard PCA", "robust (dwg)"]):
        u = U[m]
        ax.scatter(u[:, kk], u[:, kk + 1], s=4, c=GREY, lw=0)
        for c, f in zip(FAMC, top):
            s = famid == f
            ax.scatter(u[s, kk], u[s, kk + 1], s=22, c=c, lw=0.4, edgecolor="k", zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC{kk + 1}")
        ax.set_ylabel(f"PC{kk + 2}")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    lim = max(np.abs(U[m][:, kk:kk + 2]).max() for m in U) * 1.05
    for ax in axs[:2]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax = axs[2]
    x = np.arange(1, K + 1)
    ax.bar(x - 0.2, eta["standard"], 0.4, color=REDC, label="standard PCA")
    ax.bar(x + 0.2, eta["robust"], 0.4, color=BLUE, label="robust (dwg)")
    rng = np.random.default_rng(1)
    chance = np.mean([H.family_eta2(rng.standard_normal(len(famid)), famid) for _ in range(200)])
    ax.axhline(chance, color="#555555", lw=0.8, ls=":", zorder=4)
    ax.text(K + 0.4, chance + 0.02, "chance", ha="right", fontsize=8, color="#555555", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1))
    ax.axvspan(0.5, kk + 0.5, color="#EEEEEE", zorder=0)
    ax.text((kk + 1) / 2, 1.02, "ancestry", ha="center", fontsize=9, color="#555555")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0.5, K + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("PC")
    ax.set_ylabel(r"family share of the PC ($\eta^2$)")
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.33, 0.01, "coloured: the 6 families that load most on standard PC5-6; grey: everyone else",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print("eta2 chance level", round(chance, 3))
    return eta


def main():
    pcaone, work, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    U, T, famid, ftype = data(pcaone, work)
    a, b = placement(U, T, famid, os.path.join(outdir, "largen_placement.pdf"))
    eta = familyaxes(U, famid, os.path.join(outdir, "largen_familyaxes.pdf"))
    base, rel = np.where(famid == 0)[0], np.where(famid > 0)[0]
    for m in U:
        r2, err = H.score(U[m], T, base, rel, T.shape[1])
        print(m, "minR2", round(r2.min(), 3), "err", round(err, 3), "family axes", int((eta[m][4:] > 0.5).sum()),
              "max eta extra", round(eta[m][4:].max(), 2))
    print("placement axes", a + 1, b + 1, "N rel", len(rel))


if __name__ == "__main__":
    main()
