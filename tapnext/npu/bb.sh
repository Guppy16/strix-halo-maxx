#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# usage (host): bb.sh NAME [design8 args...]   regenerates build/NAME.sched from plan8 and builds design8 (CPU only)
name=$1; shift
docker run --rm --name nt4-build-$name -e AIECC_EXTRA -e EW_OPT -e EW_DEFS -e A_BANKS -e A_HALF -e P23 -e P60G -e A_RT -e P43 -e ASM_FIT -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron bash -c "cd /work/npu-tapnext && python3 run8.py sched $name && DESIGN=design8 bash b.sh $name --sched /work/npu-tapnext/build/$name.sched $* && python3 r11/stackaudit.py build/$name.prj"
tail -2 build/$name.log
