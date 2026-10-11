#!/bin/bash
# round 24: DETAIL timelines of one build for a list of ops (one dispatch each); usage: r24/tld.sh BUILD OP [OP ..]
b=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
busy=$(docker ps --format '{{.Names}}' | grep -v 'othersvc2\|othersvc')
[ -n "$busy" ] && { echo "NPU/docker busy: $busy"; exit 1; }
for op in "$@"; do
  DETAIL=$op TAILN=1 r16/nrun.sh tld_${b}_$op python3 step8.py timeline --build $b --depth 12 > /dev/null
  cp build/tl_${b}_d12_$op.json build/tld_${b}_$op.json 2>/dev/null
done
