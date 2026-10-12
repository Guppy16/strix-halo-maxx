#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# usage (host): s8.sh LOG step8-args...   runs step8.py in npu-iron-torch with the NPU; output -> build/LOG.out
log=$1; shift
docker run --rm --name nt4-s8-$log --device /dev/accel/accel0 --ulimit memlock=-1:-1 -e THREADS -e EPI_OVERRIDE -e DETAIL \
  -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO -v "${TAPNEXT_SRC:?path to the TAPNext torch model.py}":/tapnext/model.py:ro -v "${TAPNEXT_WEIGHTS:?path to tapnextpp_256.safetensors}":/tapnext/tapnextpp_256.safetensors:ro npu-iron-torch bash -c "cd /work/npu-tapnext && timeout ${TMO:-900} python3 step8.py $* > build/$log.out 2>&1; echo rc=\$? >> build/$log.out; chmod a+rw build/$log.out"
tail -${TAILN:-40} build/$log.out
