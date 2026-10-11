#!/bin/bash
# usage (host): bb.sh NAME [design8 args...]   regenerates build/NAME.sched from plan8 and builds design8 (CPU only)
name=$1; shift
docker run --rm --name nt4-build-$name -e AIECC_EXTRA -e EW_OPT -e EW_DEFS -e A_BANKS -e A_HALF -e P23 -e A_RT -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron bash -c "cd /work/npu-tapnext && python3 run8.py sched $name && DESIGN=design8 bash b.sh $name --sched /work/npu-tapnext/build/$name.sched $*"
tail -2 build/$name.log
