#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# round 16: run any python command in npu-iron-torch with the NPU; usage (host): r16/nrun.sh LOG python3 args...
# output -> build/LOG.out (rc= last line); TMO seconds (default 900)
log=$1; shift
docker run --rm --name nt16-$log -e NONKS --device /dev/accel/accel0 --ulimit memlock=-1:-1 -e OPENBLAS_NUM_THREADS -e THREADS -e DETAIL -e TL_ONLY \
  -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO -v "${TAPNEXT_SRC:?path to the TAPNext torch model.py}":/tapnext/model.py:ro -v "${TAPNEXT_WEIGHTS:?path to tapnextpp_256.safetensors}":/tapnext/tapnextpp_256.safetensors:ro npu-iron-torch bash -c "cd /work/npu-tapnext && timeout ${TMO:-900} $* > build/$log.out 2>&1; echo rc=\$? >> build/$log.out; chmod a+rw build/$log.out"
tail -${TAILN:-40} build/$log.out
