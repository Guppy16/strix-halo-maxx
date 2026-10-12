#!/bin/bash
# usage (host, from the repo root): tapnext/crop-study/run.sh NAME crop_study.py-args...   CPU only, nice 19, 8 threads; log in out/NAME.log
name=$1; shift
D=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$(dirname "$D")")
SCRATCH=${SHM_SCRATCH:-$ROOT/../.scratch}
mkdir -p $D/out
docker run --rm --name crop-$name -e OMP_NUM_THREADS=8 -v $SCRATCH:/work \
  -e TAPNEXT_SRC -e TAPNEXT_WEIGHTS -v $ROOT:$ROOT \
  npu-iron-torch nice -n 19 python3 $D/crop_study.py "$@" > $D/out/$name.log 2>&1
echo "rc=$?" >> $D/out/$name.log
chmod -R a+rwX $D/out $D/*.json 2>/dev/null
