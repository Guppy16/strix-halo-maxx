#!/bin/bash
# Runs one GPU benchmark with GTT checks: start only below 70% of the pool, kill it above 85%.
# Logs GTT and GPU busy every 2 s to <log>.gpu so contended runs are visible.
# usage: gpu_safe.sh <log> <command...>
log=$1; shift
dev=$(dirname $(ls /sys/class/drm/card*/device/mem_info_gtt_total | head -1))
total=$(cat $dev/mem_info_gtt_total)
used=$(cat $dev/mem_info_gtt_used)
if [ $((used * 100 / total)) -ge 70 ]; then echo "GTT $((used * 100 / total))% >= 70%: not starting" | tee $log; exit 2; fi
cd "${SHM_SCRATCH:-$(dirname "$0")/../../../.scratch}/moondream-opt"
"$@" > $log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
    used=$(cat $dev/mem_info_gtt_used)
    echo "$(date +%T) gtt $((used * 100 / total))% busy $(cat $dev/gpu_busy_percent)%" >> $log.gpu
    if [ $((used * 100 / total)) -ge 85 ]; then
        kill $pid; echo "KILLED: GTT $((used * 100 / total))% >= 85%" | tee -a $log; exit 3
    fi
    sleep 2
done
wait $pid
