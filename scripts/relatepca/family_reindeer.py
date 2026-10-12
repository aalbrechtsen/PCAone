#!/usr/bin/env python3
"""Family share eta^2 of PC1-10 for a reindeer data set: standard PCAone,
aarobust-kin and dwg, all SNPs (full/) next to LD-pruned (ld/). The families
are the connected groups of one fixed set of related pairs (a PCAone
.relpairs file), the same for all methods; unrelated animals form one group
(eta^2 as in highn_test.family_eta2). Dashed: eta^2 expected with the same
family sizes on random animals of the same sites (labels permuted within
each site, standard PCAone).

usage: family_reindeer.py <dir> <pairs.relpairs> <out.pdf>
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from highn_test import family_eta2 as eta2  # noqa: E402
from slides_reindeer import fam_of, read_fam  # noqa: E402

plt.rcParams.update({"font.size": 13})
INK, MUTED, GRID = "#222222", "#666666", "#bdbdbd"
# (run, label, colour, marker, line style, sideways offset): offsets and styles keep
# lines that coincide visible
RUNS = (("std_all", "standard PCAone", "#0072B2", "o", "-", -0.2), ("aar_all", "aarobust-kin", "#D55E00", "s", "--", -0.07),
        ("aad_all", "aarobust-kin+diag", "#CC79A7", "D", "-.", 0.07), ("dwg_all", "dwg", "#009E73", "^", ":", 0.2))


def families(ids, pairfile):
    """family number per animal (0: in no pair): connected groups of the pairs"""
    up = {i: i for i in ids}

    def root(i):
        while up[i] != i:
            i = up[i]
        return i

    for l in list(open(pairfile))[1:]:
        a, b = l.split()[:2]
        up[root(a)] = root(b)
    paired = {w for l in list(open(pairfile))[1:] for w in l.split()[:2]}
    roots = sorted({root(i) for i in paired})
    return np.array([roots.index(root(i)) + 1 if i in paired else 0 for i in ids])


def main():
    d, pairfile, out = sys.argv[1:4]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    rng = np.random.default_rng(1)
    for a, sub, title in zip(ax, ("full", "ld"), ("all SNPs", "LD-pruned")):
        ids, pops = read_fam(fam_of(os.path.join(d, sub), "std_all"))
        pops = np.array(pops)
        famid = families(ids, pairfile)
        x = np.arange(1, 11)
        for run, label, col, m, ls, dx in RUNS:
            V = np.loadtxt(os.path.join(d, sub, run + ".eigvecs"))[:, :10]
            e = np.array([eta2(V[:, j], famid) for j in range(10)])
            a.plot(x + dx, e, marker=m, ls=ls, color=col, lw=2, ms=7, mec="white", mew=0.8, label=label)
            print(sub, run, np.round(e, 2))
        V = np.loadtxt(os.path.join(d, sub, "std_all.eigvecs"))[:, :10]
        def shuffle():
            f = famid.copy()
            for p in np.unique(pops):
                k = np.flatnonzero(pops == p)
                f[k] = rng.permutation(f[k])
            return f

        perms = [shuffle() for _ in range(200)]
        e0 = np.array([np.mean([eta2(V[:, j], f) for f in perms]) for j in range(10)])
        print(sub, "random families within sites", np.round(e0, 2))
        # a grey bar per PC, drawn on top: visible also where it equals a method
        a.plot(x, e0, ls="", marker="_", ms=16, mew=2.2, color=MUTED, zorder=5, label="random families\nwithin sites")
        a.set_xticks(x)
        a.set_ylim(0, 1)
        a.set_xlabel("PC")
        a.set_title(title, color=INK)
        a.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        print(sub, "families", famid.max(), "animals in families", (famid > 0).sum())
    ax[0].set_ylabel("family share $\\eta^2$")
    h, lab = ax[1].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(0.995, 0.5), frameon=False)
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
