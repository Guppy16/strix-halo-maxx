#!/bin/bash
# Pass 20 (CLKLOAD2: needs the NPU and a quiet box): which part of "12 numpy threads" drops the NPU to 1.41 GHz,
# when 12 spin threads at the same ~99 W do not. 12 threads per kind, niced, stopped by PID; logs package power, Tctl, the mean
# CPU MHz and the NPU core clock (r7/clk.py median fit). Based on r23/clkload12.sh.
#   vec     the round-23 load again (512 x 512 fp32 matmul: OpenBLAS's AVX-512 kernel, L2 / L3 resident)       -> expect 1.41
#   vec_l1  64 x 64 matmul (AVX-512, L1 resident: vector current without cache / fabric traffic)
#   avx2    the vec load with OPENBLAS_CORETYPE=Haswell (256-bit FMA: half the vector width)
#   mem     np.copyto on 64 MB arrays (DRAM bandwidth, almost no FLOPs)
# Reading: mem drops it, vec_l1 not -> DRAM / fabric (the NPU shares the data fabric or its DPM); vec_l1 drops it, mem not -> CPU
# vector current (an EDC / rail limit the SMU resolves by lowering the NPU); avx2 between them -> width-dependent current.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
set -u
HW=$(grep -l amdgpu /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
KT=$(grep -l k10temp /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
DK="docker run --rm --name clkload25 --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron"
mhz() { awk '/cpu MHz/ {s += $4; n++} END {printf "%.0f", s / n}' /proc/cpuinfo; }
sample() { echo "power_uW $(cat $HW/power1_input 2>/dev/null || cat $HW/power1_average) tctl_mC $(cat $KT/temp1_input) cpu_MHz $(mhz)"; }
load() {
  case $1 in
    vec) OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 exec nice -n 10 python3 -c "
import numpy as np, time
a = np.random.rand(512, 512).astype(np.float32); t = time.time()
while time.time() - t < 90: a = a @ a; a /= np.abs(a).max() + 1" ;;
    vec_l1) OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 exec nice -n 10 python3 -c "
import numpy as np, time
a = np.random.rand(64, 64).astype(np.float32); t = time.time()
while time.time() - t < 90:
    for _ in range(1000): a = a @ a; a *= 0.01" ;;
    avx2) OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 exec nice -n 10 python3 -c "
import numpy as np, time
a = np.random.rand(512, 512).astype(np.float32); t = time.time()
while time.time() - t < 90: a = a @ a; a /= np.abs(a).max() + 1" ;;
    mem) exec nice -n 10 python3 -c "
import numpy as np, time
a = np.ones(16 << 20, np.float32); b = np.empty_like(a); t = time.time()
while time.time() - t < 90: np.copyto(b, a); np.copyto(a, b)" ;;
  esac
}
busy=$(docker ps --format '{{.Names}}' | grep -v -E "${IGNORE_PROCS:-^\$}")   # IGNORE_PROCS: regex of unrelated container names
[ -n "$busy" ] && { echo "NPU/docker busy: $busy"; exit 1; }
echo "== idle: $(sample)"
for kind in ${KINDS:-idle vec vec_l1 avx2 mem}; do
  pids=()
  if [ $kind != idle ]; then for i in $(seq 1 12); do load $kind > /dev/null & pids+=($!); done; sleep 20; fi
  echo "== $kind 12 threads: $(sample)"
  $DK bash -c "cd /work/npu-tapnext && python3 r7/clk.py run --out build/clk" 2>&1 | grep -iE "fit on median" | tail -1
  echo "   after: $(sample)"
  [ ${#pids[@]} -gt 0 ] && { kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null; sleep 10; }
done
