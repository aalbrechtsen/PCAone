#!/usr/bin/env python3
"""Table for the M = 500,000 scaling slide: run times (stopped at 1 h: return
code 124) of standard PCAone and dwg in RAM / out-of-core, and the relatives
found against the true relationships written by scale_big_sim.py
(<dir>/N<N>.truth.tsv, pedigree kinship >= tau).

usage: slides_scaling_big.py <run dir>
"""
import os
import sys

TAU = 2 ** -3.5
MEM = {"": "in RAM", "_ooc": "out-of-core"}


def times(d):
    T = {}
    for l in open(os.path.join(d, "times.tsv")):
        n, s, rc, _ = l.rstrip("\n").split("\t")
        if int(rc) != 143:  # 143: killed by hand, keep the earlier run
            T[n] = (float(s), int(rc))
    return T


def pairs(f):
    return {tuple(sorted(l.split()[:2])) for l in list(open(f))[1:]}


def fmt(t):
    if t is None:
        return "--"
    s, rc = t
    if rc == 124:
        return "stopped ($>$1\\,h)"
    if rc != 0:
        return f"failed ({rc})"
    return f"{s:.0f}\\,s" if s >= 10 else f"{s:.1f}\\,s"


def th(n):
    """thousands separator for LaTeX from 10,000 on"""
    return f"{n:,}".replace(",", "{,}") if n >= 10000 else str(n)


def main():
    d = sys.argv[1]
    T = times(d)
    for N, mem in [(1000, ""), (2000, ""), (5000, ""), (20000, ""), (100000, "_ooc"), (500000, "_ooc")]:
        tf = os.path.join(d, f"N{N}.truth.tsv")
        if not os.path.exists(tf):
            continue
        tr = {tuple(sorted(l.split()[:2])) for l in list(open(tf))[1:] if float(l.split()[2]) >= TAU}
        std, dwg, dall = T.get(f"std_N{N}{mem}"), T.get(f"dwg_N{N}{mem}"), T.get(f"dwgall_N{N}{mem}")
        found = "--"
        rp = os.path.join(d, f"dwg_N{N}{mem}.relpairs")
        if dwg and dwg[1] == 0 and os.path.exists(rp):
            P = pairs(rp)
            found = f"{100 * len(P & tr) / len(tr):.0f}\\%" + (f" (+{len(P - tr)})" if P - tr else "")
        # dwg default: all pairs up to N = 5000, the sketch above
        a_col = fmt(dwg) if N <= 5000 else fmt(dall)
        s_col = "--" if N <= 5000 else fmt(dwg)
        print(f"{th(N)} & {MEM[mem]} & {fmt(std)} & {a_col} & {s_col} & {th(len(tr))} & "
              f"{found} \\\\")
        rpa = os.path.join(d, f"dwgall_N{N}{mem}.relpairs")
        if os.path.exists(rpa):
            Pa = pairs(rpa)
            print(f"% dwg all pairs N={N}: {len(Pa & tr)}/{len(tr)} true, {len(Pa - tr)} extra")
    sims = {k: v for k, v in T.items() if k.startswith("sim_")}
    print("% simulation seconds:", {k: round(v[0]) for k, v in sims.items()})


if __name__ == "__main__":
    main()
