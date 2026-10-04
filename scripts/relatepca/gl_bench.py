#!/usr/bin/env python3
"""dwg with genotype likelihoods (dwg_gl.py) on the small-N benchmark:
simulated reads at mean depth 1, 2, 4, 8x (gl_sim.py), against dwg on the
true genotypes and standard PCAngsd-style PCA of the posterior means."""
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402
import dwg_gl as DG  # noqa: E402
import dwg_loc as DL  # noqa: E402
import grm_fit_bench as Gb  # noqa: E402
import illustrate as I  # noqa: E402
from gl_sim import simulate_gl  # noqa: E402

SCEN = ["none", "mz2", "nuclear2", "childx", "cousins", "many", "xanc", "grandx"]


def one(job):
    n, scen, rep, depth = job
    k = 3
    pops = "CEU,CHB,MXL,YRI".split(",")
    seed = zlib.crc32(f"{','.join(pops)}|{n}|0|{scen}|{rep}".encode())
    try:
        ds = B.make_dataset(Gb.G_ALL, Gb.LAB, Gb.BIM, pops, n, 0, B.SCENARIOS[scen], seed)
    except ValueError:
        return []
    T = B.truth_reference(ds, Gb.G_ALL, 3)
    G = ds["G"].astype(np.int8)
    N = len(G)
    rel = np.arange(ds["n0"], N)
    true = {(int(i), int(j)) for i, j in zip(*np.where(np.triu(ds["Kped"] >= B.TAU, 1)))}
    out = {}
    if depth == 1:  # the genotype reference once per dataset
        ck = (B.king_robust(G.astype(float)) > B.KING_SCREEN) & ~np.eye(N, dtype=bool)
        U, pp, _ = DL.dwg_loc_ea(G.astype(float), k, ck, 4, kin_check=True, k0_max=0.8)
        out["dwg_genotypes"] = (U, pp)
    GL, _ = simulate_gl(G, depth, seed=seed % 100000)
    R = DG.dwg_gl(GL, k)
    out["dwg_gl"] = (R["U"], R["pairs"])
    f = DG.em_freq(GL)
    Pi = DG.pcangsd_iaf(GL, f, 3)
    post = DG.posteriors(GL, Pi)
    E = post[:, :, 1] + 2 * post[:, :, 2]
    X = (E - 2 * f) / np.sqrt(2 * f * (1 - f))
    C = X @ X.T / X.shape[1]
    np.fill_diagonal(C, ((post * ((np.arange(3)[None, None] - 2 * f[None, :, None]) ** 2)).sum(2) /
                         (2 * f * (1 - f))).mean(1))  # PCAngsd diagonal
    out["pcangsd_standard"] = (I.top_eig(C, k)[1], [])
    rows = []
    for m, (U, pairs) in out.items():
        r2, err = B.score(U, T, ds["n0"], rel, 3)
        det = {(min(i, j), max(i, j)) for i, j, *_ in pairs}
        rec = len(det & true) / len(true) if (true and m != "pcangsd_standard") else np.nan
        fp = len(det - true) if m != "pcangsd_standard" else np.nan
        rows.append((n, scen, rep, depth if m != "dwg_genotypes" else 0, m, r2.min(), err, rec, fp,
                     float(R["rho"].mean())))
    return rows


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    jobs = [(n, s, r, d) for d in (1, 2, 4, 8) for n in (5, 10, 20) for s in SCEN for r in range(2)]
    with ProcessPoolExecutor(44, initializer=Gb.init,
                             initargs=("/kellyData/home/albrecht/codex/admix/ngsadmix/data/admixTjeck/admixTjeck2",)) as ex:
        rows = [r for rr in ex.map(one, jobs) for r in rr]
    d = pd.DataFrame(rows, columns=["n", "scen", "rep", "depth", "method", "minR2", "err", "recall", "false_pairs",
                                    "rho"])
    d.to_csv(sys.argv[1], sep="\t", index=False)
    d["fail"] = d.minR2 < 0.95
    for stat in ("minR2", "fail", "recall", "false_pairs", "err"):
        print(f"-- {stat} (rows: method, depth; columns: n)")
        print(d.pivot_table(index=["method", "depth"], columns="n", values=stat).round(3).to_string())
    print("mean reliability rho by depth:", d[d.method == "dwg_gl"].groupby("depth")["rho"].mean().round(2).to_dict())
