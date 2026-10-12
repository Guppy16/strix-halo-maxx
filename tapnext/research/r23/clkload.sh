#!/bin/bash
# pass 18 CLKLOAD (needs the NPU and a quiet box): is the NPU clock drop under CPU
# load (1.81 -> 1.405 GHz, round 21) a SoC power budget (STAPM / PPT shared by CPU and NPU) or a DPM policy?
# For each load kind x thread count: start the load, wait 20 s (power settles), run r7/clk.py (core clock from a register loop),
# log amdgpu hwmon power (the APU package / PPT figure the SMU reports) and k10temp Tctl, stop the load.
#   pre-registered: power budget -> the clock tracks package power: the vector load (numpy matmul, AVX-512) drops it at fewer
#   threads than the scalar spin (`yes`); DPM policy -> it tracks thread count (or does not move) whatever the load's power.
# usage (from $SHM_SCRATCH, nothing else running):  npu-research/r23/clkload.sh > npu-research/r23/clkload.out
# needs build/clk (python3 r7/clk.py build --out build/clk, npu-iron, CPU only) - run it first.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
set -u
HW=$(grep -l amdgpu /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
KT=$(grep -l k10temp /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
DK="docker run --rm --name clkload --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron"
sample() { echo "power_uW $(cat $HW/power1_input 2>/dev/null || cat $HW/power1_average) tctl_mC $(cat $KT/temp1_input)"; }
spin() { for i in $(seq "$1"); do yes > /dev/null & done; }
vec() { for i in $(seq "$1"); do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -c "
import numpy as np, time
a = np.random.rand(512, 512).astype(np.float32)
t = time.time()
while time.time() - t < 120: a = a @ a; a /= np.abs(a).max() + 1" & done; }
for kind in spin vec; do
  for n in 0 4 8 12 16 20; do
    [ "$n" -gt 0 ] && $kind "$n"
    sleep 20
    echo "== $kind $n threads: $(sample)"
    $DK bash -c "cd /work/npu-tapnext && python3 r7/clk.py run --out build/clk" 2>&1 | grep -iE "ghz|clock|slope" | tail -2
    echo "   after: $(sample)"
    pkill -P $$ yes 2>/dev/null; pkill -P $$ python3 2>/dev/null; wait 2>/dev/null
    sleep 10
  done
done
