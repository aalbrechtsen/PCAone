# Out-of-core BED winSVD without a permuted copy (`--perm-mem`)

*Proposal and benchmark for upstream PCAone (based on main 5387585).*

## Summary

Out-of-core winSVD on a BED currently writes a shuffled copy
(`<out>.perm.bed`) before the PCA: one extra read and write of the whole
genotype file, and disk space for a second copy. This branch adds an opt-in
alternative, `--perm-mem <GiB>`, that reads the permuted order **straight from
the input BED**:

* **Order.** The `-w` bands are *interleaved*: band b holds SNPs b, b+W,
  b+2W, … in source order, dealt round-robin and fitted to the uniform read
  blocks so that each band is exactly one bucket of `bandFactor × blocksize`
  SNPs (the same invariance the random-band copy relies on).
* **Reads.** A *window* of k consecutive bands is read at once. Interleaved
  bands b … b+k−1 are neighbours in the BED, so a window is read as contiguous
  pieces of k SNPs, issued as many concurrent `pread()`s in file order. RAM is
  traded directly for read length (and seeks).
* **Overlap.** The next window is read in a background thread while the
  current one is decoded and multiplied, so the I/O is mostly hidden behind
  the computation. If the BED fits in the budget it is read once and kept;
  later passes read nothing.
* No permutation step, no extra disk space, no change to the default
  (`--perm-mem 0` keeps the random-band copy).

Results (details below), on simulated data with LD, structure and inversions:

| | accuracy vs exact | time |
|---|---|---|
| D1 10k × 1M (2.5 GB), cold, 4 GB cap | same as random bands (within seed spread) | 246 s → **206 s** |
| D2 100k × 500k (12.5 GB), cold | subspace overlap with upstream 0.99981 | 1950 s → **1116 s** |

## Why interleaved bands, not random ones, for logical reads

With random bands a window of k bands is a random k/W subset of the SNPs,
so reads stay ~1–2 SNPs long unless k ≈ W. With interleaved bands the window
reads contiguous k-SNP pieces.

Interleaving is not less accurate. On D1, against the exact PCs (`--svd 3`):

| order | epochs | max rel. eigenvalue error | worst PC \|cor\| | subspace overlap |
|---|---|---|---|---|
| random bands, seed 112 | 8 | 1.2e-4 | 0.99966 | 0.99979 |
| random bands, seed 1 | 8 | 2.0e-4 | 0.99935 | 0.99970 |
| random bands, seed 2 | 8 | 1.5e-4 | 0.99931 | 0.99973 |
| interleaved (`--perm-mem`) | 8 | 1.7e-4 | 0.99923 | 0.99966 |

What *does* matter is that every band gets an even share of every local
region. We first tried "approximately permuted" orders, runs of c consecutive
SNPs dealt to bands (which would give long reads without any RAM). On D1,
against the exact PCs (eigenvalues 243, 232 for the populations; 14.1, 11.9,
10.5 for the three inversions; 2.61–2.44, nearly equal, for PCs 6–10):

| run length c | epochs | PC1–5 \|cor\| | worst of PC6–10 | subspace (10 PCs) |
|---|---|---|---|---|
| 1 (old interleave, earlier code) | 7 | 1.0000 | 0.9992 | 0.99959 |
| 4 | 7 | 1.0000 | 0.9985 | 0.99928 |
| 16 | 12 | 1.0000 | 0.54 | 0.9898 |
| 64 | 21 (maxp, no convergence) | 1.0000 | 0.44 | 0.9310 |
| no permutation | 21 (maxp, no convergence) | 1.0000 | 0.57 | 0.9220 |

The structure PCs are found in every case, even without a permutation. What
long runs cost is convergence: more epochs (each a full pass over the BED),
or none within `--maxp`, and poorly determined PCs near the noise level. With
runs of c SNPs a local region is shared out in multiples of c (an LD-heavy
3000-SNP region at c = 16: 32 or 48 SNPs per band), so the bands disagree more
and the window updates settle slowly. So the reads must come from the exact
interleave; the window is what makes them long.

## Where the time goes

Benchmarks on a spinning RAID (PERC H730, ext4, ~200–270 MB/s sequential),
files evicted from page cache (`posix_fadvise DONTNEED`) and the process run
in a cgroup with `MemoryMax` so the page cache cannot hold the BED between
passes. Raw replays of one winSVD pass, without the PCA (`iobench`):

**Read patterns, D2 (12.5 GB), one full pass**

| pattern | MB/s |
|---|---|
| sequential (reading the permuted copy) | 167–250 (disk shared over NFS) |
| interleave read SNP by SNP, 32 threads | 39 |
| window k = 16 (3.1 GB) | 160 |
| window k = 32 (6.3 GB) | 245 |

* Many concurrent reads in file order are essential: one thread reaches
  9 MB/s on single SNPs, 32 threads 39 MB/s; shuffled submission order is 3×
  worse than sorted.
* `mmap` (with or without `MADV_WILLNEED`) was never better than concurrent
  `pread`; disabling readahead made no difference.

**Large N, D3 (500k × 200k, 25 GB): read length grows with N**

| `--perm-mem` | window | contiguous read | MB/s | s/pass |
|---|---|---|---|---|
| (sequential copy) | – | – | 272 | 92 |
| 4 | 5 bands | 0.6 MB | 152 | 164 |
| 8 | 11 bands | 1.3 MB | 181 | 138 |
| 16 | 22 bands | 2.7 MB | 201 | 124 |
| ≥ 25 (e.g. 50) | whole BED | sequential | 221 | 113, **once per run** |

Raw window reads are somewhat slower than one sequential stream, but the
copy costs an extra read *and* write first, its passes are read in the
foreground, and with the background read the window I/O overlaps the
computation (D1: 20.5 GB read with 5 s of waiting in total).

## PCA timings

`-k 10 -n 20`, cold page cache, cgroup memory cap; upstream = main 5387585.

**D1 (10k × 1M, 2.5 GB), `-m 1`**

| run | permute s | read+decode s | wall s | epochs |
|---|---|---|---|---|
| upstream (random copy), 4 GB cap | 23 | 58 | 246 | 8 |
| `--perm-mem 1` (13-band windows), 4 GB cap | 0 | 47 (I/O wait 5) | 206 | 8 |
| `--perm-mem 4` (whole BED once), 6 GB cap | 0 | 54 (I/O wait 15) | 222 | 8 |

**D2 (100k × 500k, 12.5 GB), `-m 4`**

| run | permute s | read+decode s | wall s | epochs | max RSS |
|---|---|---|---|---|---|
| upstream (random copy), 7 GB cap | 91 | 863 | 1950 | 9 | 3.5 GB |
| `--perm-mem 4` (10-band windows, 244 KB reads), 11 GB cap | 0 | 218 (I/O wait 18 s, 100 GB read) | 1116 | 8 | 7.1 GB |
| `--perm-mem 16` (whole BED once), 20 GB cap | 0 | 266 (I/O wait 60 s, 12.5 GB read) | 1136 | 8 | 15.1 GB |

The two `--perm-mem` runs give byte-identical PCs; against upstream's run the
max relative eigenvalue error is 2.0e-4, worst PC |cor| 0.99926 and subspace
overlap 0.99981 (no exact reference at this size).

**Where the 834 s on D2 come from**

| | upstream | `--perm-mem 4` | saved |
|---|---|---|---|
| writing the permuted copy | 91 s | 0 | 91 s (11 %) |
| read+decode, all passes | 863 s (≈ 96 s/pass) | 218 s (≈ 27 s/pass) | 645 s (77 %) |
| compute per pass | ≈ 111 s | ≈ 112 s | – |
| passes | 9 | 8 | ≈ 1 pass (likely chance; on D1 both took 8) |

So most of the gain is the **background read**, not skipping the copy: the
copy is read in the foreground every pass (read, compute, read, …), while
`--perm-mem` reads the next window during the computation. Giving the
existing sequential path the same background read would recover much of
that too, and is worth doing independently; what only the logical order
gives is no permutation step, no second BED on disk, and with a large budget
a single read of the BED for the whole run.

## Less RAM, randomisation, adaptive windows (`--perm-chunk`, `--perm-rotate`, `--perm-adapt`)

The contiguous read length is c × k SNPs: k bands per window (RAM) times c
neighbouring SNPs dealt to a band at a time.

* `--perm-chunk c` deals chunks of c neighbouring SNPs to the bands: the same
  read length with c times less RAM. (c ≥ 16 hurts convergence, see above.)
* `--perm-rotate` rotates the bands by a random offset (`--seed`) in every
  stretch of W chunks. Bands stay exactly balanced; a window's piece of a
  stretch splits at most once, so reads stay long.
* `--perm-adapt` starts with windows of ~256 KiB reads and doubles them after
  a pass that spent more than 5 % of its time stalled on I/O it could have
  read ahead; `--perm-mem` becomes a cap rather than a target.

**D1 (10k × 1M), cold, 4 GB cap; vs exact PCs**

| run | window RAM | epochs | wall s | I/O wait s | subspace | worst PC \|cor\| |
|---|---|---|---|---|---|---|
| c = 1, `--perm-mem 1` | 0.95 GB | 8 | 209 | 4.5 | 0.99971 | 0.99949 |
| c = 2, `--perm-mem 0.5` | 0.45 GB | 9 | 239 | 8.0 | 0.99980 | 0.99950 |
| c = 4, `--perm-mem 0.25` | 0.21 GB | 9 | 222 | 11.5 | 0.99970 | 0.99933 |
| c = 1 + rotate, seed 112 | 0.95 GB | 8 | 183 | 4.6 | 0.99977 | 0.99957 |
| c = 1 + rotate, seed 7 | 0.95 GB | 8 | 203 | 4.7 | 0.99972 | 0.99947 |
| c = 4 + rotate, `--perm-mem 0.25` | 0.21 GB | 10 | 237 | 8.7 | 0.99965 | 0.99931 |

**D2 (100k × 500k), cold; vs upstream's run**

| run | window RAM | max RSS | epochs | wall s | I/O wait s | subspace |
|---|---|---|---|---|---|---|
| c = 1, `--perm-mem 4` | 3.6 GB | 7.1 GB | 8 | 1116 | 18 | 0.99981 |
| **c = 4, `--perm-mem 1`** | **0.73 GB** | **4.2 GB** | 8 | **1110** | 4 | 0.99978 |
| c = 1 + rotate, `--perm-mem 4` | 3.6 GB | 7.1 GB | 9 | 1246 | 14 | 0.99971 |
| `--perm-adapt`, cap 16 GB (stayed at 11 bands) | ≈ 4 GB | 7.5 GB | 8 | 1118 | 20 | 0.99982 |
| c = 1, `--perm-mem 16` (whole BED) | 12.5 GB | 15.1 GB | 8 | 1136 | 60 | — |

* Accuracy is the same for every option (the spread is that of random seeds).
* **Chunks of 4 cut the window RAM 4–5× at the same speed on D2**; on D1 they
  cost one extra epoch (+6 %). c = 2–4 is a good choice when RAM is tight.
* Rotation is neutral: epochs moved by ±1 either way, as with random bands.
* `--perm-adapt` keeps the window as small as the computation allows: with a
  16 GB cap it used ~4 GB and ran as fast as reading the whole BED once.
  (A first version counted the unavoidable first read of each pass as a stall
  and doubled the window every pass; fixed.)

## Correctness

* `tests/test_bed_logical_perm.py` (in `make test_bed_permutation`): for
  windows from the whole BED down to one band, plain and with `--emu`, and for
  `--perm-chunk`, `--perm-rotate` (same `--seed` → same order) and
  `--perm-adapt` (byte-identical PCs to fixed windows), the
  eigenvalues, eigenvectors and loadings equal those of a physically reordered
  copy in the same order run with `-S`; `.mbim` is in input order; no
  `.perm.*` is written; the input BED/BIM/FAM are byte-identical after a
  `-v 1` run (whose cleanup used to delete `params.filein.*`, which with this
  option would be the input — the cleanup now requires `filein == <out>.perm`).
* The existing `tests/test_bed_permutation.py` still passes.
* On D1 and D2, results are identical for every `--perm-mem`.

## Memory

`--perm-mem` is in addition to `-m`: two windows of k/W of the BED while the
next one loads (or the whole BED once), plus 4 bytes per SNP for the order.
For large N, a few GB already give multi-MB reads; 50 GB holds a 500k × 1M
BED in about five windows, or a BED up to 50 GB whole.

## Other observations (not part of this change)

* `H += X * Gb` in the out-of-core winSVD still uses Eigen's built-in GEMM,
  which only splits the result's columns (≈ (k + oversamples)/4 threads: 5 for
  `-k 10`). Splitting the rows of X over threads by hand, as `mul_Xt_Y` does
  for the other product, made a 10000 × 7813 block 2.6× faster in our fork.
* Possible follow-ups: make `--perm-mem` the default (auto-sized from free
  RAM, falling back to the copy below a minimum window), and give the
  sequential and BGEN paths the same background read.

## Reproducing

`scripts/perm-mem-bench/` on this branch: the simulator (`simgeno.cpp`), the
I/O replay without the PCA (`iobench.cpp`), cache eviction and cgroup-capped
run scripts, and the accuracy comparison (`compare.R`). See its README.
