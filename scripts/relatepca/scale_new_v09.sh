#!/bin/bash
# The M = 20,000 scaling runs (slides_scaling.py, solid lines) again with
# PCAone v0.9.0 and the packed dwg: standard PCAone and dwg (default), dwg with
# all pairs at N = 10,000 and 20,000; 16 threads, in RAM.
# times.tsv: name, seconds, return code, pairs, peak RAM (kB)
# usage: scale_new_v09.sh <out dir>
OUT=$1; mkdir -p $OUT; cd $OUT
P=/kellyData/home/albrecht/codex/relatePCA/PCAone/PCAone
D=/kellyData/home/albrecht/codex/relatePCA/data
for N in 1000 2000 5000 10000 20000; do
  for e in bed bim fam; do [ -e sim_N$N.$e ] || cp $D/scale/sim_N$N.$e .; done
done
for e in bed bim fam; do [ -e sim_N100000.$e ] || cp $D/relfind/big_N100000.$e sim_N100000.$e; done
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
for N in 1000 2000 5000 10000 20000 100000; do
  run std_N$N $P -b sim_N$N -k 5 -n 16
  run dwg_N$N $P -b sim_N$N -k 5 -n 16 --robust
  [ $N -eq 10000 -o $N -eq 20000 ] && run dwgall_N$N $P -b sim_N$N -k 5 -n 16 --robust --king-search all --ea-search all
done
