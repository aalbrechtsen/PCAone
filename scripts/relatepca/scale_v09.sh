#!/bin/bash
# The M = 500,000 scaling runs again with PCAone v0.9.0 (2-bit calls) and the
# packed dwg: standard PCAone and dwg in RAM for every N, out-of-core (-m 16)
# from N = 20,000; each run stopped after 1 h (return code 124).
# Data: scale_big_sim.py (<big>/N<N>.bed, .truth.tsv), linked into <out>.
# times.tsv: name, seconds, return code, pairs, peak RAM (kB)
# usage: scale_v09.sh <big dir> <out dir>
BIG=$1; OUT=$2; mkdir -p $OUT; cd $OUT
P=/kellyData/home/albrecht/codex/relatePCA/PCAone/PCAone
for N in 1000 2000 5000 20000 100000 500000; do
  for e in bed bim fam truth.tsv; do [ -e N$N.$e ] || ln -s $BIG/N$N.$e N$N.$e; done
done
run() {
  local name=$1; shift
  [ -s $name.done ] && return
  /usr/bin/time -f "%e %M" -o $name.time timeout -s KILL 3600 "$@" -o $name > $name.out 2>&1
  local rc=$?; [ $rc -eq 137 ] && rc=124
  read s m < <(tail -1 $name.time)
  local np=-1; [ -s $name.relpairs ] && np=$(($(wc -l < $name.relpairs) - 1))
  printf "%s\t%s\t%s\t%s\t%s\n" $name $s $rc $np $m >> times.tsv
  echo done > $name.done
}
for N in 1000 2000 5000 20000 100000 500000; do
  run std_N$N $P -b N$N -k 5 -n 64
  run dwg_N$N $P -b N$N -k 5 -n 64 --robust
  [ $N -eq 20000 ] && run dwgall_N$N $P -b N$N -k 5 -n 64 --robust --king-search all --ea-search all
  if [ $N -ge 20000 ]; then
    run std_N${N}_ooc $P -b N$N -k 5 -n 64 -m 16
    run dwg_N${N}_ooc $P -b N$N -k 5 -n 64 --robust --robust-engine operator -m 16
  fi
done
