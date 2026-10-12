#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 24: DETAIL timelines of one build for a list of ops (one dispatch each); usage: r24/tld.sh BUILD OP [OP ..]
b=$1; shift
cd "$SHM_SCRATCH/npu-tapnext" || exit 1
busy=$(docker ps --format '{{.Names}}' | grep -v "${IGNORE_PROCS:-^$}")
[ -n "$busy" ] && { echo "NPU/docker busy: $busy"; exit 1; }
for op in "$@"; do
  DETAIL=$op TAILN=1 r16/nrun.sh tld_${b}_$op python3 step8.py timeline --build $b --depth 12 > /dev/null
  cp build/tl_${b}_d12_$op.json build/tld_${b}_$op.json 2>/dev/null
done
