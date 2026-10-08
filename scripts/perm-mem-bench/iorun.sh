#!/bin/bash
# iorun.sh RESULTS.tsv CAP_GB bed_prefix iobench-args...
# Evict the .bed from page cache, then run iobench in a cgroup with MemoryMax=CAP
# (caps page cache too) and append its result line to RESULTS.tsv.
set -u
res=$1; cap=$2; pre=$3; shift 3
T=$(dirname "$(readlink -f "$0")")
"$T/evict" "$pre.bed"
line=$(systemd-run --user --scope --quiet -p MemoryMax=${cap}G -p MemorySwapMax=0 "$T/iobench" "$pre" "$@" 2>&1 | tail -1)
echo -e "$(basename $pre)\t$line" | tee -a "$res"
