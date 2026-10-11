#!/bin/bash
# round 24: N fast timelines (TL_ONLY op kinds) of a build -> build/tlo_<build>_<tag>_run<i>.json; then r24/tlo.py on them
# usage: r24/tlo.sh BUILD KINDS(comma) [N=3] [TAG=x]
b=$1; k=$2; n=${3:-3}; tag=${4:-x}
cd ~/git/app/.scratch/npu-tapnext || exit 1
busy=$(docker ps --format '{{.Names}}' | grep -v 'othersvc2\|othersvc')
[ -n "$busy" ] && { echo "NPU/docker busy: $busy"; exit 1; }
for i in $(seq 1 $n); do
  TL_ONLY=$k TAILN=1 r16/nrun.sh tlo_$b python3 step8.py timeline --build $b --depth 12 > /dev/null
  grep -q "rc=0" build/tlo_$b.out || { tail -5 build/tlo_$b.out; exit 1; }
  grep "WARNING\|polls" build/tlo_$b.out | head -3
  cp build/tl_${b}_d12.json build/tlo_${b}_${tag}_run$i.json
done
python3 r24/tlo.py build/tlo_${b}_${tag}_run*.json
