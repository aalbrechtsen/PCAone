# `pcaone-ibd`: IBD sharing probabilities (k0, k1, k2)

Kinship alone cannot identify a relationship. **Parent–offspring and full sibs
both have `phi = 1/4`**, so `--evaladmix` reports 0.2496 and 0.2475 for them —
indistinguishable. Only the IBD sharing probabilities separate them: a parent and
child share exactly one allele IBD everywhere, `(k0,k1,k2) = (0,1,0)`, while full
sibs average `(1/4, 1/2, 1/4)`.

`pcaone-ibd` estimates those from PCAone's output plus the genotypes. It is a
separate binary and shares no state with PCAone — everything it needs is in the
published `.eigvecs`, `.kinship` and `.mbim`.

## Running it

```bash
# 1. PCA + kinship. K=2 here, so K-1 = 1 PC.
PCAone -b smallPlink -k 1 --evaladmix --maf 0.05 -o pcs

# 2. IBD probabilities for the related pairs
pcaone-ibd -b smallPlink -P pcs --min-kin 0.05 -o rel
```

```
samples 126, sites in .bim 104290, sites used by the PCA 102185
using 1 PC(s) + intercept, leave-one-out allele frequencies
pairs with kinship > 0.05: 50 of 7875
individuals involved: 100
wrote 50 pairs to rel.ibd
```

Build with `make ibd` (or `make`, which now builds both).

| option | |
|---|---|
| `-b <prefix>` | PLINK `.bed/.bim/.fam` prefix |
| `-P <prefix>` | PCAone output prefix (`.eigvecs`, `.kinship`, `.mbim`) |
| `-o <prefix>` | output prefix (default `pcaone-ibd`) |
| `--min-kin <f>` | only estimate pairs above this kinship (default 0.05) |
| `-k <int>` | PCs to use (default: all in `.eigvecs`; use `K-1`) |
| `-n <int>` | threads (default 4) |
| `--no-loo` | disable the leave-one-out allele frequencies (see below) |

### Output

`rel.ibd`, one row per surviving pair — a sparse list, not a matrix, because `k`
is meaningless for the 99.4% of pairs that are unrelated:

```
ID1     ID2     kin       k0        k1        k2        nsnp
MIX7    MIX8    0.500000  0.000000  0.000000  1.000000  102185
MIX27   MIX28   0.249608  0.000000  0.998365  0.001635  102185
MIX47   MIX48   0.247549  0.252646  0.504112  0.243242  102185
```

## Accuracy

On the benchmark data (126 individuals, 102,185 sites, K=2, known pedigree; see
[reproducing-the-benchmark.md](reproducing-the-benchmark.md)):

| relationship | n | k0 | k1 | k2 | expected |
|---|---|---|---|---|---|
| duplicate | 10 | 0.000 | 0.000 | 1.000 | 0, 0, 1 |
| parent–offspring | 10 | 0.000 | 0.998 | 0.002 | 0, 1, 0 |
| full sib | 10 | 0.253 | 0.504 | 0.243 | .25, .5, .25 |
| half sib | 10 | 0.530 | 0.466 | 0.004 | .5, .5, 0 |
| first cousin | 10 | 0.742 | 0.255 | 0.003 | .75, .25, 0 |

**The point of the exercise.** Parent–offspring and full sibs have
indistinguishable kinship — 0.2496 and 0.2475 — but `k0` separates them
completely: 0.000 (range 0.000–0.000) against 0.253 (range 0.245–0.265). No
overlap.

Estimates are always inside the simplex, by construction.

Runtime: **0.83 s** for 50 pairs over 102,185 sites on 16 threads.

## How it works

Individual allele frequencies are recovered from the PC scores and the
genotypes — this is why no PCAone internals are needed:

```
pi_is = 0.5 * [ V (V'V)^-1 V' G ]_is ,    V = [PC_1..PC_k, 1]
```

`(k0,k1,k2)` are then fitted per pair by EM over the three IBD states, with the
likelihood written at the **allele** level rather than the genotype level:

```
IBD=0:  g_i ~ HWE(pi_i),  g_j ~ HWE(pi_j),  independent
IBD=1:  shared allele S ~ pi_ij ;  i's other ~ pi_i ;  j's other ~ pi_j
IBD=2:  g_i = g_j ~ HWE(pi_ij)
```

with `pi_ij = (pi_i + pi_j)/2`.

**Keeping `pi_i` and `pi_j` apart is essential, not a refinement.** Relatives in
admixed data frequently do not share ancestry — in this dataset half-sib pairs
differ by 0.43 in admixture proportion, first cousins by 0.33, because that is
the case these methods exist to handle. Pooling to a single frequency per site,
which is the obvious first implementation, reports a **parent–offspring pair as
83% unrelated** (`k0 = 0.831`). The failure is invisible until tried.

### Why only related pairs

`k` is meaningless for unrelated pairs, but the restriction also does real work:
it makes a per-pair likelihood affordable, because **related pairs are O(N), not
O(N²)**. Without it this would need six simultaneous `N x N` accumulators.

At `phi > 0.05` on this data, 50 of 7875 pairs survive — every first cousin and
closer, with no unrelated pair leaking through. Second cousins (true
`phi = 0.0156`) fall below, which is deliberate: their `k2` is noise either way.

| threshold | pairs kept | related kept | unrelated leaked |
|---|---|---|---|
| `phi > 0.02` | 53 | 53 | 0 |
| `phi > 0.05` | 50 | 50 | 0 |
| `phi > 0.08` | 40 | 40 | 0 |

### Leave-one-out allele frequencies (default)

`pi_i` is fitted using data that includes individual `i` — and `j` — so it has
partly absorbed the relatedness being measured, biasing `k` toward "unrelated".
Since `beta_s = (V'V)^-1 V'g_s`, dropping the pair's two rows from both sides is
a rank-2 downdate, so `pi` can be re-formed without the pair at the cost of one
`(k+1)x(k+1)` solve per pair. Same idea as evalAdmix's leave-one-out refit of
`F`, transposed to the PC regression.

| | `--no-loo` | default | expected |
|---|---|---|---|
| full sib, k0 | 0.264 | **0.253** | 0.25 |
| half sib, k0 | 0.565 | **0.530** | 0.5 |
| first cousin, k0 | 0.794 | **0.742** | 0.75 |
| **RMSE**, 50 pairs × 3 params | 0.03049 | **0.01737** | 43% lower |

Cost, over three runs: 0.83–0.86 s with it, 0.83–0.86 s without (6.03 s vs
6.10 s single-threaded) — within run-to-run noise. The downdate adds four
multiply–adds per site in place of two, which disappears beside reading the
`.bed` and running the EM. Hence the default.

## Limitations

- **Half sibs remain attenuated**, `k0 = 0.530` against a truth of 0.5. The pair
  is now out of the allele-frequency regression but still inside the PCA that
  produced the scores; removing it there too would mean recomputing the
  decomposition per pair.
- **`pi_ij = (pi_i + pi_j)/2` is a choice, not a derivation.** Under an admixture
  model the IBD allele comes from one ancestral population, which `pi` alone
  cannot identify. It works well here and is untested on more extreme ancestry
  differences.
- **PLINK `.bed` input only**, and **complete genotypes assumed** — missing calls
  are treated as 0 with a warning.
- `k2` is only estimable where there is IBD-2 to see. For everything but full
  sibs and duplicates the truth is ~0 and the estimate is mostly noise.
- Validated on one dataset: N=126, K=2, one pedigree. In particular, "related
  pairs are O(N)" is a property of real cohorts, not of this dataset.
