#!/usr/bin/env python3
"""Small-N test on a second real panel (1000 Genomes CEU, CHB, YRI and the
admixed ASW; world.merged.pruned thinned to 100k SNPs): relatives made by
transmission with recombination from real genotypes (benchmark.make_dataset),
including relatives of different ancestry and ASW families. PCAone methods
scored against the reference truth (the unused individuals)."""
import os
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
from sim import read_bed, write_bed  # noqa: E402

SCEN = {"none": "", "mz2": "mz:CEU:2", "nuclear2": "child:CEU:2", "childx": "childx:CEUxYRI",
        "cousins": "cousins:CEU", "xanc": B.SCENARIOS["xanc"], "grandx": "grandx:YRI:CEU",
        "avuncx": "avuncx:CHB:YRI", "asw": "sibs:ASW:2,child:ASW:1,grandx:ASW:CHB",
        "aswx": "halfsibx:ASW:CEU:CHB,avuncx:ASW:CHB"}
METHODS = {"standard": "", "aarobust-kin": "--robust --impute-diag", "detect-white": "--robust detect-white",
           "dwg": "--robust"}


def main(bfile, pcaone, work, out):
    os.makedirs(work, exist_ok=True)
    fam, bim, Gall = read_bed(bfile)
    lab = np.array([r[1] for r in fam])
    pops = ["CEU", "CHB", "YRI", "ASW"]
    kk = 2  # ancestry axes: CEU-YRI and CHB (ASW lies on the CEU-YRI cline)
    jobs = [(n, s, r, k) for n in (5, 10, 20) for s in SCEN for r in range(3) for k in (kk, 10)]

    def one(job):
        n, scen, rep, k = job
        seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
        try:
            ds = B.make_dataset(Gall, lab, bim, pops, n, 0, SCEN[scen], seed)
        except ValueError:
            return []
        T = B.truth_reference(ds, Gall, kk)
        G = ds["G"]
        N = len(G)
        pre = os.path.join(work, f"{scen}_n{n}_r{rep}_k{k}")
        write_bed(pre, [["F", f"I{i}", "0", "0", "0", "-9"] for i in range(N)], [bim[i] for i in ds["cols"]],
                  G.astype(np.int8))
        rel = np.arange(ds["n0"], N)
        true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(ds["Kped"] >= B.TAU, 1)))}
        rows = []
        for m, extra in METHODS.items():
            o = f"{pre}_{m}"
            r = subprocess.run(f"{pcaone} -b {pre} -k {k} -n 2 -v 0 {extra} -o {o}", shell=True, capture_output=True,
                               text=True)
            if r.returncode:
                return [("ERR", m, r.stdout[-300:])]
            U = np.loadtxt(o + ".eigvecs", ndmin=2)
            r2, err = B.score(U, T, ds["n0"], rel, kk)
            rec = fp = np.nan
            if m != "standard" and os.path.exists(o + ".relpairs"):
                det = {tuple(sorted((int(l.split()[0][1:]), int(l.split()[1][1:]))))
                       for l in open(o + ".relpairs").read().strip().split("\n")[1:] if l}
                rec = len(det & true) / len(true) if true else np.nan
                fp = len(det - true)
            rows.append((n, scen, rep, k, m, r2.min(), err, rec, fp))
        return rows

    import pandas as pd
    with ThreadPoolExecutor(24) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    bad = [r for r in rows if r[0] == "ERR"]
    if bad:
        print(bad[:3])
    d = pd.DataFrame([r for r in rows if r[0] != "ERR"],
                     columns=["n", "scen", "rep", "k", "method", "minR2", "err", "recall", "false_pairs"])
    d.to_csv(out, sep="\t", index=False)
    d["fail"] = d.minR2 < 0.95
    pd.set_option("display.width", 220)
    order = list(METHODS)
    for k in (kk, 10):
        x = d[d.k == k]
        print(f"== k = {k}")
        print(pd.concat({"mean min R2": x.pivot_table(index="method", columns="n", values="minR2").reindex(order),
                         "fail": x.pivot_table(index="method", columns="n", values="fail").reindex(order),
                         "recall": x.pivot_table(index="method", columns="n", values="recall").reindex(order),
                         "false": x.pivot_table(index="method", columns="n", values="false_pairs").reindex(order)},
                        axis=1).round(3).to_string())
        print("relatives' error:", x.groupby("method")["err"].mean().reindex(order).round(3).to_dict())
    print("== recall by scenario (k = 2)")
    print(d[d.k == kk].pivot_table(index="method", columns="scen", values="recall").reindex(order).round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
