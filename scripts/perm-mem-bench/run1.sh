#!/bin/bash
# run1.sh OUTPREFIX CAP_GB|none "files to evict" PCAone-args...
# Runs one PCAone job (optionally inside a cgroup with MemoryMax=CAP, which also
# caps the page cache), after evicting the given files from page cache.
set -u
out=$1; cap=$2; ev=$3; shift 3
T=$(dirname "$(readlink -f "$0")")
PCAONE=${PCAONE:-PCAone}
for f in $ev; do [ -e "$f" ] && "$T/evict" "$f"; done
sync
cmd=(/usr/bin/time -v -o "$out.time" "$PCAONE" "$@" -o "$out")
if [ "$cap" != none ]; then
  cmd=(systemd-run --user --scope --quiet -p MemoryMax=${cap}G -p MemorySwapMax=0 "${cmd[@]}")
fi
echo "${cmd[*]}" > "$out.cmd"
"${cmd[@]}" > "$out.stdout" 2>&1
echo "exit $?" >> "$out.cmd"
