#!/usr/bin/env python3
"""Simulate related individuals on top of real genotypes, for testing --kinship.

Relatives are made by copying alleles from real individuals: each founder is
randomly phased, and every child receives one recombinant gamete from each
parent (crossovers Poisson along each chromosome at 1 cM/Mb), so relatives
share genuine IBD segments rather than independent per-site draws.

Output (in --out):
  test.{bed,bim,fam}   unrelated base individuals + the simulated relatives
  ref.{bed,bim,fam}    the remaining individuals of the same populations; the
                       ground-truth axes are a PCA of this panel
  truth.kin0           pedigree kinship for test pairs with phi > 0 (ID1 ID2 KINSHIP)
  info.tsv             IID, POP, FAM, ROLE, UNREL (1 = in a maximal unrelated set),
                       expected ancestry fractions Q_<pop>

Family types (--families, comma separated, TYPE:POP[:N]):
  dup         one individual and N-1 identical copies (MZ twins / duplicates)
  sibs        N full sibs, parents not in the data
  nuclear     both parents + N children
  halfsibs    N half sibs sharing one parent, nobody else in the data
  avuncular   an uncle/aunt and a niece/nephew (2nd degree)
  grand       grandparent and grandchild (2nd degree)
  extended    grandparents, N children and their own children (spouses absent)
POP may be two populations joined by 'x' (e.g. CEUxYRI) for admixed parents.
"""
import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------- PLINK I/O
CODE2G = np.array([2, -1, 1, 0], dtype=np.int8)  # bed 2-bit code -> copies of A1


def read_bed(prefix):
    fam = [l.split() for l in open(prefix + ".fam")]
    bim = [l.rstrip("\n") for l in open(prefix + ".bim")]
    n, m = len(fam), len(bim)
    nb = (n + 3) // 4
    raw = np.fromfile(prefix + ".bed", dtype=np.uint8)
    assert raw[0] == 0x6C and raw[1] == 0x1B and raw[2] == 0x01, "not a SNP-major bed"
    raw = raw[3:].reshape(m, nb)
    codes = np.stack([(raw >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(m, nb * 4)[:, :n]
    G = CODE2G[codes].T  # n x m
    if (G < 0).any():
        sys.exit("missing genotypes are not supported by the simulator")
    return fam, bim, G


def write_bed(prefix, fam, bim, G):
    n, m = G.shape
    with open(prefix + ".fam", "w") as f:
        f.writelines(" ".join(r) + "\n" for r in fam)
    with open(prefix + ".bim", "w") as f:
        f.writelines(l + "\n" for l in bim)
    g2code = np.array([3, 2, 0], dtype=np.uint8)  # 0,1,2 copies of A1
    codes = g2code[G.T]  # m x n
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.ones((m, pad), dtype=np.uint8)], axis=1)
    codes = codes.reshape(m, -1, 4)
    byte = codes[:, :, 0] | (codes[:, :, 1] << 2) | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        f.write(byte.astype(np.uint8).tobytes())


# ---------------------------------------------------------- gene dropping
class Dropper:
    """Haplotypes are n_hap x m int8 arrays of A1 alleles (0/1)."""

    def __init__(self, bim, rng):
        self.rng = rng
        chrom = np.array([l.split()[0] for l in bim])
        pos = np.array([int(l.split()[3]) for l in bim])
        self.chroms = []
        for c in dict.fromkeys(chrom):  # keep order
            idx = np.where(chrom == c)[0]
            self.chroms.append((idx, pos[idx]))

    def phase(self, g):
        """random phase of one genotype vector (0/1/2 copies of A1)"""
        h1 = (g == 2).astype(np.int8)
        h2 = h1.copy()
        het = g == 1
        coin = self.rng.random(het.sum()) < 0.5
        h1[het] = coin
        h2[het] = ~coin
        return np.stack([h1, h2])

    def gamete(self, haps, labs=None):
        """one gamete; with labs (founder-haplotype labels, same shape as haps)
        also the labels of the gamete, from the same crossovers (no extra draws)"""
        out = np.empty(haps.shape[1], dtype=np.int8)
        lab = None if labs is None else np.empty(labs.shape[1], dtype=labs.dtype)
        for idx, pos in self.chroms:
            lo, hi = pos[0], pos[-1]
            nco = self.rng.poisson((hi - lo) / 1e8)  # 1 cM per Mb
            cuts = np.sort(self.rng.uniform(lo, hi, nco))
            which = (np.searchsorted(cuts, pos) + self.rng.integers(2)) % 2
            out[idx] = np.where(which == 0, haps[0, idx], haps[1, idx])
            if labs is not None:
                lab[idx] = np.where(which == 0, labs[0, idx], labs[1, idx])
        return out if labs is None else (out, lab)


# ---------------------------------------------------------- pedigrees
def build_family(kind, pops, n, fid):
    """Return (members, in_data) where members = list of (id, father, mother, pop-or-None).

    Founders carry a population label (drawn from that pool); others have parents.
    """
    pa, pb = (pops + [pops[0]])[:2] if len(pops) == 1 else pops
    M = []

    def founder(tag, pop):
        M.append((f"{fid}_{tag}", None, None, pop))
        return f"{fid}_{tag}"

    def child(tag, f, m):
        M.append((f"{fid}_{tag}", f, m, None))
        return f"{fid}_{tag}"

    data = []
    if kind == "dup":
        f = founder("O", pa)
        data = [f] + [f"{fid}_D{i+1}" for i in range(n - 1)]
        M += [(x, "=", f, None) for x in data[1:]]
    elif kind in ("sibs", "nuclear"):
        f, m = founder("F", pa), founder("M", pb)
        kids = [child(f"C{i+1}", f, m) for i in range(n)]
        data = kids + ([f, m] if kind == "nuclear" else [])
    elif kind == "halfsibs":
        f = founder("F", pa)
        for i in range(n):
            m = founder(f"M{i+1}", pb)
            data.append(child(f"C{i+1}", f, m))
    elif kind == "avuncular":
        gf, gm = founder("GF", pa), founder("GM", pb)
        a, b = child("A", gf, gm), child("B", gf, gm)
        s = founder("S", pb)
        data = [a, child("N", b, s)]
    elif kind == "grand":
        gf, gm = founder("GF", pa), founder("GM", pb)
        p = child("P", gf, gm)
        s = founder("S", pb)
        data = [gf, child("G", p, s)]
    elif kind == "extended":
        gf, gm = founder("GF", pa), founder("GM", pb)
        data = [gf, gm]
        for i in range(n):
            c = child(f"C{i+1}", gf, gm)
            s = founder(f"S{i+1}", pa if i % 2 else pb)
            data += [c] + [child(f"C{i+1}K{j+1}", c, s) for j in range(2)]
    else:
        sys.exit(f"unknown family type {kind}")
    return M, data


def kinship(members):
    """pedigree kinship over the members of one family (topologically ordered)"""
    ids = [m[0] for m in members]
    pos = {x: i for i, x in enumerate(ids)}
    K = np.zeros((len(ids), len(ids)))
    for i, (x, f, m, _) in enumerate(members):
        if f == "=":  # identical copy of m
            K[i, :i] = K[:i, i] = K[pos[m], :i]
            K[i, i] = K[pos[m], pos[m]]
            K[i, pos[m]] = K[pos[m], i] = K[pos[m], pos[m]]
            continue
        K[i, i] = 0.5 if f is None else 0.5 + 0.5 * K[pos[f], pos[m]]
        for j in range(i):
            K[i, j] = K[j, i] = 0.0 if f is None else 0.5 * (K[pos[f], j] + K[pos[m], j])
    return ids, K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pops", default="CEU,CHB,YRI")
    ap.add_argument("--n-unrel", type=int, default=10, help="unrelated individuals per population")
    ap.add_argument("--families", default="", help="e.g. sibs:CEU:4,nuclear:YRI:3")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    fam, bim, G = read_bed(a.bfile)
    pops = a.pops.split(",")
    label = np.array([r[1] for r in fam])
    pool = {p: list(rng.permutation(np.where(label == p)[0])) for p in pops}

    def draw(p):
        if not pool[p]:
            sys.exit(f"ran out of {p} individuals")
        return pool[p].pop()

    drop = Dropper(bim, rng)
    out_fam, out_G, info, pairs = [], [], [], []
    hap = {}

    # unrelated base individuals: real genotypes as they are
    for p in pops:
        for _ in range(a.n_unrel):
            i = draw(p)
            out_fam.append(["UNREL", fam[i][0], "0", "0", "0", "-9"])
            out_G.append(G[i])
            info.append(dict(IID=fam[i][0], POP=p, FAM="UNREL", ROLE="unrelated", UNREL=1, Q={p: 1.0}))

    specs = [s for s in a.families.split(",") if s]
    for fi, spec in enumerate(specs):
        parts = spec.split(":")
        kind, fpops = parts[0], parts[1].split("x")
        n = int(parts[2]) if len(parts) > 2 else 2
        fid = f"FAM{fi+1}"
        members, data = build_family(kind, fpops, n, fid)
        q = {}
        for x, f, m, p in members:
            if f == "=":
                hap[x], q[x] = hap[m], q[m]
            elif f is None:
                i = draw(p)
                hap[x] = drop.phase(G[i])
                q[x] = {p: 1.0}
            else:
                hap[x] = np.stack([drop.gamete(hap[f]), drop.gamete(hap[m])])
                q[x] = {k: 0.5 * (q[f].get(k, 0) + q[m].get(k, 0)) for k in set(q[f]) | set(q[m])}
        ids, K = kinship(members)
        par = {x: ((None, None) if f == "=" else (f, m)) for x, f, m, _ in members}
        for k, x in enumerate(data):
            f, m = par[x]
            fam_f = f if f in data else "0"
            fam_m = m if m in data else "0"
            out_fam.append([fid, x, fam_f, fam_m, "0", "-9"])
            out_G.append(hap[x].sum(axis=0).astype(np.int8))
            info.append(dict(IID=x, POP="x".join(fpops), FAM=fid, ROLE=kind, UNREL=int(k == 0), Q=q[x]))
        pos = {x: i for i, x in enumerate(ids)}
        for u in range(len(data)):
            for v in range(u + 1, len(data)):
                phi = K[pos[data[u]], pos[data[v]]]
                if phi > 0:
                    pairs.append((data[u], data[v], phi))

    os.makedirs(a.out, exist_ok=True)
    out_G = np.array(out_G, dtype=np.int8)
    # drop sites monomorphic in the test set: they carry no information and
    # every method would have to filter them identically
    poly = (out_G.min(axis=0) != out_G.max(axis=0))
    bim_t = [l for l, k in zip(bim, poly) if k]
    write_bed(os.path.join(a.out, "test"), out_fam, bim_t, out_G[:, poly])
    ref = [i for p in pops for i in pool[p]]
    write_bed(os.path.join(a.out, "ref"), [fam[i] for i in ref], bim_t, G[np.ix_(ref, np.where(poly)[0])])
    with open(os.path.join(a.out, "truth.kin0"), "w") as f:
        f.write("ID1\tID2\tKINSHIP\n")
        f.writelines(f"{x}\t{y}\t{phi:.6f}\n" for x, y, phi in pairs)
    with open(os.path.join(a.out, "info.tsv"), "w") as f:
        f.write("IID\tPOP\tFAM\tROLE\tUNREL\t" + "\t".join(f"Q_{p}" for p in pops) + "\n")
        for r in info:
            f.write(f"{r['IID']}\t{r['POP']}\t{r['FAM']}\t{r['ROLE']}\t{r['UNREL']}\t"
                    + "\t".join(f"{r['Q'].get(p, 0):.4f}" for p in pops) + "\n")
    with open(os.path.join(a.out, "sim.json"), "w") as f:
        json.dump(vars(a), f)
    print(f"{a.out}: {len(out_fam)} test individuals ({sum(r['UNREL'] for r in info)} in the unrelated set), "
          f"{len(ref)} reference, {poly.sum()} sites, {len(pairs)} related pairs")


if __name__ == "__main__":
    main()
