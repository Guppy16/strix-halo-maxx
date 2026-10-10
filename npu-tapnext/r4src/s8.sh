#!/bin/bash
# usage (host): s8.sh LOG step8-args...   runs step8.py in npu-iron-torch with the NPU; output -> build/LOG.out
log=$1; shift
docker run --rm --name nt4-s8-$log --device /dev/accel/accel0 --ulimit memlock=-1:-1 -e THREADS -e EPI_OVERRIDE -e DETAIL \
  -v $HOME/git/app/.scratch:/work -v $HOME/git/app:/repo:ro npu-iron-torch bash -c "cd /work/npu-tapnext && timeout ${TMO:-900} python3 step8.py $* > build/$log.out 2>&1; echo rc=\$? >> build/$log.out; chmod a+rw build/$log.out"
tail -${TAILN:-40} build/$log.out
