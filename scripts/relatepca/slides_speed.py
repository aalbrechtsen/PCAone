#!/usr/bin/env python3
"""Slide figure: run time against N for standard PCAone, aarobust-kin,
detect-white and dwg (PCAone, 16 threads). Points are measured
(results/scale_summary.tsv, robust_speed_dense_cpp.txt and the methods
report: dwg in the operator engine at N <= 2000 and N = 100,000,
detect-white with the sketch search and standard PCAone at N = 100,000);
dashed lines extrapolate:
  aarobust-kin   full eigendecomposition per iteration: ~ N^3
  standard       ~ N
  detect-white,  linear part + the sketch neighbor search ~ N^2 s, which is
  dwg            ~50 s at N = 100,000 and ~1.5 h at N = 10^6 (methods report)

usage: slides_speed.py <out.pdf>
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 11})
INK, MUTED = "#222222", "#666666"
SKETCH = 5400 / 1e12  # seconds per N^2: 1.5 h at N = 10^6

MEAS = {
    "standard PCA": ([500, 1000, 2000, 5000, 10000, 20000, 100000], [0.4, 0.6, 1.3, 4.6, 6.5, 24.4, 148]),
    "aarobust-kin": ([500, 1000, 2000], [4.7, 30.7, 248]),
    "detect-white": ([500, 1000, 2000, 5000, 10000, 20000, 100000], [3.8, 3.9, 8.1, 16.4, 32.2, 79.0, 118]),
    "dwg": ([500, 1000, 2000, 100000], [0.7, 1.1, 1.8, 226]),
}
COL = {"standard PCA": "#666666", "aarobust-kin": "#D55E00", "detect-white": "#E69F00", "dwg": "#0072B2"}


def model(name, N):
    n0, t0 = MEAS[name][0][-1], MEAS[name][1][-1]
    if name == "aarobust-kin":
        return t0 * (N / n0) ** 3
    if name == "standard PCA":
        return t0 * N / n0
    lin = (t0 - SKETCH * n0 ** 2) / n0
    return lin * N + SKETCH * N ** 2


def main():
    out = sys.argv[1]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    Nx = np.logspace(np.log10(500), 6, 200)
    for name in MEAS:
        n, t = np.array(MEAS[name][0], float), np.array(MEAS[name][1])
        c = COL[name]
        ax.plot(n, t, "o", color=c, ms=5)
        for a in range(len(n) - 1):  # solid between neighbouring measurements, dotted across a wide gap
            ax.plot(n[a:a + 2], t[a:a + 2], "-" if n[a + 1] / n[a] <= 10 else ":", color=c, lw=1.8)
        ext = Nx[Nx >= n[-1]]
        ax.plot(ext, model(name, ext), "--", color=c, lw=1.6)
        if name == "aarobust-kin":
            xl = 9000
            ax.text(xl, model(name, xl) * 1.6, name, color=c, fontsize=11, ha="right", va="bottom")
        else:
            dy = {"dwg": 1.6, "detect-white": 0.62, "standard PCA": 0.7}.get(name, 1.0)
            lab = "standard PCA\n(super fast PCAone)" if name == "standard PCA" else name
            ax.text(1.08e6, model(name, 1e6) * dy, lab, color=c, fontsize=11, ha="left", va="center",
                    linespacing=1.0)
    for sec, lab in [(60, "1 min"), (3600, "1 hour"), (86400, "1 day")]:
        ax.axhline(sec, color="#dddddd", lw=0.8, zorder=0)
        ax.text(560, sec * 1.12, lab, color=MUTED, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(450, 1e6)
    ax.set_ylim(0.2, 3e5)
    ax.set_xlabel("$N$ (individuals)", color=MUTED)
    ax.set_ylabel("run time (s)", color=MUTED)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.plot([], [], "-o", color=MUTED, label="measured")
    ax.plot([], [], "--", color=MUTED, label="extrapolated")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    for name in MEAS:
        print(name, {N: round(float(model(name, N))) for N in (1e4, 1e5, 1e6)})


if __name__ == "__main__":
    main()
