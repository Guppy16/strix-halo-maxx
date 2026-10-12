#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# round 23: one A/B timing window: note it in RESULTS_FROM_MAIN, record box state, run r18/ab.py, close the window.
# usage: ab.sh TAG ROUNDS A B [B ...]
tag=$1; rounds=$2; shift 2
a=$1; shift
f=$SHM_REPO/tapnext/research/RESULTS_FROM_MAIN.md
busy=$(docker ps --format '{{.Names}}' | grep -v "${IGNORE_PROCS:-^$}")
[ -n "$busy" ] && { echo "NPU/docker busy: $busy - not timing"; exit 1; }
echo "- WINDOW OPEN $(date +%H:%M) (main A/B $tag: $a vs $*)" >> $f
echo "BOX $(uptime | sed 's/.*load/load/') gpu_busy $(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | tr '\n' ' ')"
cd "$SHM_SCRATCH/npu-tapnext"
TMO=3000 r16/nrun.sh ab_$tag python3 r18/ab.py --a $a --b "$@" --rounds $rounds | grep "^AB\|^SPREAD\|^DELTA\|rc="
echo "BOX-END $(uptime | sed 's/.*load/load/')"
echo "- WINDOW CLOSED $(date +%H:%M)" >> $f
