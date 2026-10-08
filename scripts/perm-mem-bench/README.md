# Benchmark tools for `--perm-mem`

Used for `docs/winsvd-perm-mem.md`.

| file | what |
|---|---|
| `simgeno.cpp` | simulate PLINK data with LD blocks, 3 populations + admixed, inversion-like regions (`g++ -O3 -march=native -fopenmp`) |
| `iobench.cpp` | replay the reads of one winSVD pass (sequential, interleaved per block, windows of k bands, mmap, prefetch) without the PCA |
| `evict.c` | drop a file from the page cache (`posix_fadvise DONTNEED`, no root) |
| `run1.sh` | evict, run PCAone in a cgroup with `MemoryMax` (caps the page cache too), `/usr/bin/time -v` |
| `iorun.sh` | the same for `iobench` |
| `compare.R` | eigenvalue error, per-PC correlation and subspace overlap against a reference run |

Datasets:

```sh
simgeno -o D1 -n 10000  -m 1000000 --seed 11   # 2.5 GB
simgeno -o D2 -n 100000 -m 500000  --seed 22   # 12.5 GB
simgeno -o D3 -n 500000 -m 200000  --seed 33   # 25 GB
```

Example (cold, 4 GB cap):

```sh
PCAONE=./PCAone ./run1.sh out/up 4 "D1.bed" -b D1 -k 10 -m 1 -n 20
PCAONE=./PCAone ./run1.sh out/br 4 "D1.bed" -b D1 -k 10 -m 1 -n 20 --perm-mem 1
./iorun.sh io.tsv 5 D2 window --block 3907 --W 64 --c 1 --kb 16 --threads 32
```
