#!/bin/bash
# usage (host, from the repo root): tapnext/crop-study/run.sh NAME crop_study.py-args...   CPU only, nice 19, 8 threads; log in out/NAME.log
name=$1; shift
D=$(cd "$(dirname "$0")" && pwd)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
mkdir -p $D/out
docker run --rm --name crop-$name -e OMP_NUM_THREADS=8 -v $SHM_SCRATCH:/work \
  -e TAPNEXT_SRC -e TAPNEXT_WEIGHTS -v $SHM_REPO:$SHM_REPO \
  npu-iron-torch nice -n 19 python3 $D/crop_study.py "$@" > $D/out/$name.log 2>&1
echo "rc=$?" >> $D/out/$name.log
chmod -R a+rwX $D/out $D/*.json 2>/dev/null
