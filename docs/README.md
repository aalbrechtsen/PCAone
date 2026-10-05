# PCAone docs

**New to this work? Start with [HANDOVER.md](HANDOVER.md)** — current state, the
ideas that matter, what is and is not established, and what to do next.

## Relatedness

| | |
|---|---|
| [robust/slides/](robust/slides/relatedness_pca.pdf) | Slides (Beamer, ~30 min): relatedness in PCA, the small-N problem, `dwg`, kinship for admixed individuals, evidence |
| [robust/](robust/README.md) | `--robust`: PCA robust to close relatives — the methods (`dwg`, `detect-white`, `aarobust-kin`, …), graphical abstracts, the three sample-size regimes, final relatedness (k0, k1, k2) and the evidence; also as [PDF](robust/pcaone_robust_methods.pdf) |
| [evaladmix.md](evaladmix.md) | `--evaladmix`: correlation of residuals from the top PCs, i.e. kinship. Accuracy against PC-Relate, RelateAdmix and the other evalAdmix routes |
| [pcaone-ibd.md](pcaone-ibd.md) | `pcaone-ibd`: IBD sharing probabilities `(k0,k1,k2)`, which separate parent–offspring from full sibs |
| [reproducing-the-benchmark.md](reproducing-the-benchmark.md) | two commands that regenerate every published number |
| [relatedness/](relatedness/) | the longer written analysis, PDF + LaTeX source |

## Bug fixes

| | |
|---|---|
| [read-usv-fix.md](read-usv-fix.md) | `read_usv()` transposed any multi-column matrix. Affected ancestry-adjusted LD: 99.96% of R² values changed at K=2 |
| [ld-robustness-fixes.md](ld-robustness-fixes.md) | NaN R² from zero-variance variants; segfault on a missing `.mbim` |
| [hwe-lrt-fix.md](hwe-lrt-fix.md) | `--inbreed` HWE likelihood ratio test was not on a common scale |

## Quick start

```bash
make                                                       # PCAone and pcaone-ibd

PCAone     -b plink -k <K-1> --evaladmix --maf 0.05 -o pcs # kinship
pcaone-ibd -b plink -P pcs --min-kin 0.05 -o rel           # k0, k1, k2
```

Use `K-1` PCs. That single choice matters more than anything else documented
here — one too many costs a factor of seven to ten in RMSE.
