#!/usr/bin/env python3
"""LaTeX rows: every pair called related by any method on one SNP set of a
reindeer data set, with each method's kinship (- if not called). KING-robust
(computed here on the same SNP set; called if > 2^-3.5), aarobust-kin
(aar_all.relpairs), aarobust-kin+diag (aad_all), dwg (dwg_all), refit
(refit_all.pairs), and the IBD probabilities k0/k1/k2 that
aarobust-kin and dwg estimate for the pairs they call. R0 = IBS0 / HETHET (plink2 --make-king-table on the
same SNPs, <dir>/kt_<full|ld>_all.kin0). Pairs are
sorted by the number of methods that call them.

usage: reindeer_pairs_table.py <dir> <full|ld>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
from sim import read_bed  # noqa: E402

METHODS = [("aar_all.relpairs", "aarobust-kin"), ("aad_all.relpairs", "+diag"), ("dwg_all.relpairs", "dwg"),
           ("refit_all.pairs", "refit")]


def read_pairs(path):
    rows = list(open(path))
    head = rows[0].split()
    c = head.index("KINSHIP")
    return {tuple(sorted(l.split()[:2])): float(l.split()[c]) for l in rows[1:]}


def read_ibd(path):
    rows = list(open(path))
    h = rows[0].split()
    c = [h.index(x) for x in ("K0", "K1", "K2")]
    return {tuple(sorted(l.split()[:2])): [float(l.split()[j]) for j in c] for l in rows[1:]}


def ibd_cell(v):
    return "/".join(f"{x:.2f}".lstrip("0") if x >= 0 else f"{x:.2f}" for x in v) if v else "--"


def r0_table(path):
    """R0 = IBS0 / HETHET of every pair from a plink2 --make-king-table file"""
    out = {}
    for l in list(open(path))[1:]:
        x = l.split()
        out[tuple(sorted((x[1], x[3])))] = float(x[6]) / max(float(x[5]), 1e-12)
    return out


def king(prefix):
    fam, _, G = read_bed(prefix)
    return [r[1] for r in fam], [r[0] for r in fam], B.king_robust(G)


def main():
    d, mode = sys.argv[1], sys.argv[2]
    ids, sites, Kl = king(os.path.join(d, "rd_ld" if mode == "ld" else "rd_full"))
    Ka = Kl if mode == "full" else king(os.path.join(d, "rd_full"))[2]
    pos = {x: i for i, x in enumerate(ids)}
    calls = {m: read_pairs(os.path.join(d, mode, f)) for f, m in METHODS}
    r0 = r0_table(os.path.join(d, f"kt_{mode}_all.kin0"))
    ibd = [read_ibd(os.path.join(d, mode, f)) for f in ("aar_all.relpairs", "dwg_all.relpairs")]
    allp = set().union(*calls.values())
    iu = np.triu_indices(len(ids), 1)
    allp |= {tuple(sorted((ids[i], ids[j]))) for i, j in zip(*iu) if Kl[i, j] > B.TAU}
    rows = []
    for p in allp:
        i, j = pos[p[0]], pos[p[1]]
        vals = [calls[m].get(p) for _, m in METHODS]
        n = sum(v is not None for v in vals) + (Kl[i, j] > B.TAU)
        rows.append((-n, -Ka[i, j], p, i, j, vals))
    for _, _, p, i, j, vals in sorted(rows):
        p = tuple(x.replace("_reftest", "") for x in p)
        site = sites[i] if sites[i] == sites[j] else f"{sites[i]} / {sites[j]}"
        site = site.replace("_", " ").replace("Kangerlussuaq Sisimiut", "Kanger.-Sisimiut")
        kl = f"\\textbf{{{Kl[i, j]:.3f}}}" if Kl[i, j] > B.TAU else f"{Kl[i, j]:.3f}"
        kl = kl.replace("-", "$-$")
        ka = f"{Ka[i, j]:.3f}".replace("-", "$-$")
        cells = " & ".join(f"{v:.3f}" if v is not None else "--" for v in vals)
        q = tuple(sorted((ids[i], ids[j])))
        cells += " & " + " & ".join(ibd_cell(b.get(q)) for b in ibd)
        cells = f"{r0[q]:.3f} & " + cells
        if mode == "full":
            print(f"{p[0]} -- {p[1]} & {site} & {kl} & {cells} \\\\")
        else:
            print(f"{p[0]} -- {p[1]} & {site} & {kl} & {cells} \\\\")
    print("% pairs", len(rows), {m: len(c) for m, c in calls.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
