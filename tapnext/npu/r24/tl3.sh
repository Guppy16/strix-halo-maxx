#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 24: N timelines of one build (step8.py timeline), saved as build/tl_<build>_d<depth>_run<i>.json, then r24/tlcmp.py on them.
# usage: r24/tl3.sh BUILD [N=3] [DEPTH=12]   (refuses to run while another NPU container is up)
b=$1; n=${2:-3}; d=${3:-12}
cd "$SHM_SCRATCH/npu-tapnext" || exit 1
busy=$(docker ps --format '{{.Names}}' | grep -v "${IGNORE_PROCS:-^$}")
[ -n "$busy" ] && { echo "NPU/docker busy: $busy"; exit 1; }
echo "BOX $(uptime | sed 's/.*load/load/')"
for i in $(seq 1 $n); do
  TAILN=3 r16/nrun.sh tl_$b python3 step8.py timeline --build $b --depth $d > /dev/null
  grep -q "rc=0" build/tl_$b.out || { tail -5 build/tl_$b.out; exit 1; }
  grep WARNING build/tl_$b.out | head -3
  cp build/tl_${b}_d$d.json build/tl_${b}_d${d}_run$i.json
done
python3 r24/tlcmp.py build/tl_${b}_d${d}_run*.json
