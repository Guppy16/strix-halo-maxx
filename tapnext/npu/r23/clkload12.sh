#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 23 (main): research's r23/clkload.sh capped at 12 load threads (main's worker rule) and 0 / 6 / 12 threads per kind; loads
# niced, stopped by their own PIDs (plain kill). Logs package power (amdgpu hwmon) and Tctl with the core clock (r7/clk.py).
set -u
HW=$(grep -l amdgpu /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
KT=$(grep -l k10temp /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
DK="docker run --rm --name clkload23 --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron"
sample() { echo "power_uW $(cat $HW/power1_input 2>/dev/null || cat $HW/power1_average) tctl_mC $(cat $KT/temp1_input)"; }
for kind in spin vec; do
  for n in 0 6 12; do
    pids=()
    for i in $(seq 1 $n); do
      if [ $kind = spin ]; then nice -n 10 yes > /dev/null & else
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -c "
import numpy as np, time
a = np.random.rand(512, 512).astype(np.float32)
t = time.time()
while time.time() - t < 90: a = a @ a; a /= np.abs(a).max() + 1" & fi
      pids+=($!)
    done
    sleep 20
    echo "== $kind $n threads: $(sample)"
    $DK bash -c "cd /work/npu-tapnext && python3 r7/clk.py run --out build/clk" 2>&1 | grep -iE "fit on median" | tail -1
    echo "   after: $(sample)"
    [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null
    sleep 10
  done
done
