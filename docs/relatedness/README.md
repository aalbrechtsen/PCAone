# Written analysis

The reasoning behind the relatedness code in this repository. The `docs/*.md`
files are the user-facing reference; these are the arguments and the evidence.
LaTeX source sits beside each PDF — rebuild with `pdflatex <file>.tex` run
twice, no bibliography step.

| file | contents |
|---|---|
| `kinship_three.pdf` | **PC-Relate, PCAngsd and evalAdmix are one estimator**, instantiated three times. The four ways to run evalAdmix; why an overall RMSE is the wrong statistic; how PC-Relate derives `(k0,k1,k2)`; full methods and commands |
| `ibd_options.pdf` | **The `(k0,k1,k2)` design space.** Moments vs maximum likelihood, why pooling `pi` fails catastrophically, the leave-one-out correction and its cost, and why this became a second program rather than a PCAone mode |
| `relatedness_admixed.pdf` | **Wider survey**: relatedness estimation for admixed samples across REAP, RelateAdmix, PC-Relate, SEEKIN, NGSremix, PCAngsd, EMU and PCAone |
| `evalAdmix_pcaone.R` | The original R prototype. Superseded by `--evaladmix`, kept as an independent implementation to check against |

The `.tex` files deliberately avoid `titlesec` and `enumitem` so they build on a
minimal TeX Live install. Keep it that way.
