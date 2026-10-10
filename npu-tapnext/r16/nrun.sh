#!/bin/bash
# round 16: run any python command in npu-iron-torch with the NPU; usage (host): r16/nrun.sh LOG python3 args...
# output -> build/LOG.out (rc= last line); TMO seconds (default 900)
log=$1; shift
docker run --rm --name nt16-$log --device /dev/accel/accel0 --ulimit memlock=-1:-1 -e OPENBLAS_NUM_THREADS -e THREADS \
  -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx -v $HOME/git/app:/repo:ro npu-iron-torch bash -c "cd /work/npu-tapnext && timeout ${TMO:-900} $* > build/$log.out 2>&1; echo rc=\$? >> build/$log.out; chmod a+rw build/$log.out"
tail -${TAILN:-40} build/$log.out
