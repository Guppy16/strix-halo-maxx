#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# usage (host): r8.sh LOG run8-args...  runs run8.py in npu-iron with the NPU; output -> build/LOG.out
log=$1; shift
docker run --rm --name nt4-run --device /dev/accel/accel0 --ulimit memlock=-1:-1 -e DBG -e ATTN_KEYS -e M_REAL -e FIRST \
  -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron bash -c "export PYTHONPATH=/opt/xilinx/xrt/python:/work/npu-tapnext; cd /work/npu-tapnext && timeout 600 python3 run8.py $* > build/$log.out 2>&1; echo rc=\$? >> build/$log.out; chmod a+rw build/$log.out"
cat build/$log.out | tail -40
