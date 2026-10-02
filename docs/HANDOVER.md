# Handover: relatedness estimation in PCAone

State of the relatedness work as of **2026-10-02**, written so that someone —
human or agent — can pick it up without the preceding conversation.

Everything here was validated on one dataset. Read [What is not
established](#what-is-not-established) before extending any of it.

## What exists

| | what it is | docs |
|---|---|---|
| `PCAone --evaladmix` | correlation of residuals from the top PCs; kinship `phi` | [evaladmix.md](evaladmix.md) |
| `pcaone-ibd` | IBD sharing probabilities `(k0,k1,k2)` for related pairs | [pcaone-ibd.md](pcaone-ibd.md) |
| `scripts/benchmark/` | regenerates every published number | [reproducing-the-benchmark.md](reproducing-the-benchmark.md) |
| three bug fixes | `read_usv`, NaN R², `.mbim` segfault | [read-usv-fix.md](read-usv-fix.md), [ld-robustness-fixes.md](ld-robustness-fixes.md) |
| `docs/relatedness/` | the longer written analysis (PDF + LaTeX source) | below |

```bash
make                                                      # builds PCAone and pcaone-ibd
PCAone     -b plink -k <K-1> --evaladmix --maf 0.05 -o pcs
pcaone-ibd -b plink -P pcs --min-kin 0.05 -o rel
```

### `docs/relatedness/`

Written analysis, with LaTeX source alongside each PDF so it can be edited and
rebuilt (`pdflatex <file>.tex`, twice, no bibliography). These are the reasoning
behind the code; the `docs/*.md` files are the user-facing reference.

| file | contents |
|---|---|
| `kinship_three.pdf` | PC-Relate vs PCAngsd vs evalAdmix: they are one estimator; the four ways to run evalAdmix; the RMSE-decomposition argument; how PC-Relate derives `(k0,k1,k2)`; methods and commands |
| `ibd_options.pdf` | the `(k0,k1,k2)` design space — moments vs ML, why pooling `pi` fails, leave-one-out, why a second program |
| `relatedness_admixed.pdf` | wider survey of relatedness methods for admixed samples (REAP, RelateAdmix, PC-Relate, SEEKIN, NGSremix, PCAngsd, EMU, PCAone) |
| `evalAdmix_pcaone.R` | the original R prototype, superseded by `--evaladmix` but useful as an independent check |

The `.tex` files were written to build with a minimal TeX Live — no `titlesec`,
no `enumitem`. Keep it that way or the next person needs `texlive-latex-extra`.

## The three ideas that make it work

Anyone changing this code should understand these, because each is load-bearing
and none is obvious.

**1. The kinship statistic needs no residual matrix.** With complete data,
`cor(G(I-P))` reduces to three streaming accumulations — the `N x N` Gram matrix,
per-sample mean genotype, per-sample heterozygosity:

```
Rtilde'Rtilde = (I-P) [ G'G - M gbar gbar' ] (I-P)
```

verified to 1e-15. This is why `--evaladmix` is `O(N^2)` memory regardless of
site count and why it never forms `pi`. **It breaks with missing data**, where
each pair sees a different site set.

**2. The IBD likelihood must be written at the allele level.** Each individual's
*own* frequency governs its *own* non-shared allele; only the IBD-shared allele
takes a pooled `pi_ij`. Pooling to one frequency per site — the obvious
implementation — reports a parent–offspring pair as **83% unrelated**. The
reason is that relatives in admixed data often do not share ancestry: in the
benchmark, half-sib pairs differ by 0.43 in admixture proportion. If you
refactor the likelihood, re-run the per-class table; this failure is invisible
in aggregate statistics.

**3. Restricting to related pairs is what makes the likelihood affordable.**
Related pairs are `O(N)`, not `O(N^2)`. Without `--min-kin`, `(k0,k1,k2)` would
need six simultaneous `N x N` accumulators — 43 GB at N=30,000. With it, only
the kinship matrix is `N x N`.

## Reading the numbers

Two traps, both of which caught earlier drafts of this work:

- **An RMSE over all pairs is a measurement of unrelated pairs.** They are 99.24%
  of them. Ranking methods by it puts RelateAdmix first, when on the 60 pairs
  that actually have relatedness to estimate it is last. Always split related
  from unrelated.
- **Kinship cannot separate parent–offspring from full sibs.** Both have
  `phi = 1/4`. Any claim to distinguish them must come from `k0`.

## What is not established

- **One dataset.** N=126, K=2, complete genotypes, discrete well-separated
  source populations, PLINK input, one pedigree. Nothing here is tested on
  continuous ancestry, misspecified `K`, missing data, or `N` large enough for
  `O(N^2)` memory to bite.
- **Half sibs stay attenuated**, `k0 = 0.530` against 0.5, after leave-one-out.
  The pair is out of the allele-frequency regression but still inside the PCA
  that produced the scores. Removing it there too means recomputing the
  decomposition per pair.
- **`pi_ij = (pi_i + pi_j)/2` is a choice, not a derivation.** Under an admixture
  model the IBD allele comes from one ancestral population, which `pi` alone
  cannot identify.
- **The number of PCs matters far more than any of this.** One PC too many costs
  a factor of seven to ten in RMSE — two orders of magnitude more than the gap
  between methods. PC-Relate is *not* more robust to this despite needing no
  explicit `K`.
- **`--maf` with out-of-core is rejected by PCAone itself**, unrelated to this
  work.
- **`--inbreed` paths are untouched and unverified by anything here.** The
  benchmark cannot exercise them (they need `-P`).

## If you are picking this up

Start by reproducing the benchmark — it takes about ten minutes and confirms the
toolchain:

```bash
bash scripts/benchmark/run_all.sh  ~/bench  ./PCAone
Rscript scripts/benchmark/benchmark.R ~/bench
```

The RMSE table should match [evaladmix.md](evaladmix.md) to the digits shown,
and the three cross-checks at the end should all read 0. If they do not, fix
that before trusting anything else.

### Known next steps, roughly in order of value

1. **Validate at scale.** Everything rests on N=126. A cohort of 10k–50k would
   test the `O(N^2)` memory claim, whether related pairs really are `O(N)`, and
   whether the small-sample effects shrink as expected.
2. **Missing data.** Both tools assume complete genotypes. `--evaladmix` needs
   pairwise site counts alongside the Gram matrix; `pcaone-ibd` needs per-pair
   site masks. This is the biggest gap for real data.
3. **PGEN/BGEN input for `pcaone-ibd`**, which currently reads PLINK `.bed` only.
4. **The half-sib residue**, if it survives at larger N. Would need the pair
   removed from the PCA, not just from the frequency regression.
5. **Add `pcaone-ibd` to the benchmark script**, so the `(k0,k1,k2)` per-class
   table is regenerated rather than hand-checked.
6. **`read_usv` callers.** The transposition bug is fixed, but anyone who ran
   `-D`/`-R`/`--ld-r2`/`--clump` with `-P` and `-k > 1` before it has wrong
   results and should regenerate them.

### Where to be careful

- `src/EvalAdmix.cpp` out-of-core branch: the block reader always returns
  *centred* genotypes regardless of `params.center`, and estimates `F` on the
  fly. `F` and `centered_geno_lookup` are allocated there for that reason, and
  heterozygosity is recovered as `x = G_centred + f_s`. The in-core path runs
  with `center = false` and needs none of this. Changing either path means
  re-checking both.
- `--evaladmix` writes `.mbim`; `pcaone-ibd` depends on it to apply the same
  site filter. That is the only coupling between the two binaries — do not
  remove it without giving `pcaone-ibd` another way to learn the filter.
- PCAone codes genotypes as `{0, 0.5, 1}`, i.e. `g/2`. It cancels inside
  `--evaladmix` (both `b` and `c` become correlations) but not elsewhere.
- evalAdmix-family output is `2*phi`, not `phi`. `--evaladmix` already halves it;
  the external `evalAdmix` program does not.
