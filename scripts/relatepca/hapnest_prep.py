#!/usr/bin/env python3
"""Extract a HAPNEST subset (synthetic biobank genotypes, chromosomes 12-22) for
the real-LD test of --robust: founders for the sample and a disjoint
reference panel per ancestry, SNPs common to all ancestry files, merged and
LD-pruned with plink2."""
import os
import subprocess
import sys

import numpy as np

SRC = "/kellyData/home/jonas/data/hapnest"
ANC = ["eur", "eas", "afr", "amr", "csa", "mid"]
N_POOL = {"eur": 1500, "eas": 1500, "afr": 1500, "amr": 800, "csa": 800, "mid": 400}  # sample + family founders
N_REF = 600  # reference panel per ancestry
CHR = range(12, 23)


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        sys.exit(cmd + "\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main(out):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(2026)
    pops = []
    for a in ANC:
        ids = [l.split()[1] for l in open(f"{SRC}/synthetic_v1_chr-22.{a}.fam")]
        pick = rng.choice(len(ids), N_POOL[a] + N_REF, replace=False)
        pool, ref = [ids[i] for i in pick[:N_POOL[a]]], [ids[i] for i in pick[N_POOL[a]:]]
        with open(f"{tmp}/keep_{a}.txt", "w") as f:
            f.writelines(f"{x}\t{x}\n" for x in pool + ref)
        pops += [(x, a, "pool") for x in pool] + [(x, a, "ref") for x in ref]
    with open(f"{out}/individuals.tsv", "w") as f:
        f.write("IID\tancestry\tset\n")
        f.writelines(f"{x}\t{a}\t{s}\n" for x, a, s in pops)
    merge = []
    for c in CHR:
        common = None
        for a in ANC:
            ids = {l.split()[1] for l in open(f"{SRC}/synthetic_v1_chr-{c}.{a}.bim")}
            common = ids if common is None else common & ids
        with open(f"{tmp}/snps_{c}.txt", "w") as f:
            f.writelines(x + "\n" for x in sorted(common))
        for a in ANC:
            o = f"{tmp}/c{c}_{a}"
            run(f"plink2 --bfile {SRC}/synthetic_v1_chr-{c}.{a} --keep {tmp}/keep_{a}.txt --extract {tmp}/snps_{c}.txt "
                f"--make-bed --threads 16 --out {o}")
            merge.append(o)
        print(f"chr{c}: {len(common)} common SNPs", flush=True)
    # plink2 cannot merge different samples yet: plink 1.9 --merge-list
    with open(f"{tmp}/rest.txt", "w") as f:
        f.writelines(m + "\n" for m in merge[1:])
    run(f"plink --bfile {merge[0]} --merge-list {tmp}/rest.txt --make-bed --threads 16 --out {tmp}/all")
    # LD pruning on the merged founders (MAF >= 0.05 over all of them)
    run(f"plink2 --bfile {tmp}/all --maf 0.05 --indep-pairwise 500kb 0.1 --threads 16 --out {tmp}/prune")
    run(f"plink2 --bfile {tmp}/all --extract {tmp}/prune.prune.in --make-bed --threads 16 --out {out}/hapnest_pool")
    n = sum(1 for _ in open(f"{out}/hapnest_pool.bim"))
    print(f"merged and pruned: {n} SNPs, {len(pops)} individuals -> {out}/hapnest_pool", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
