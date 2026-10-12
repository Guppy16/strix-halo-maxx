#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# usage (host, npu-tapnext): r7/t4var.sh TAG KDEFS   builds t4-TAG with design8 kdefs (comma-separated) and times it
# (20 iters after 10 warm-up) with the run8 accuracy checks -> build/r7_t4_TAG.out
cd "$SHM_SCRATCH/npu-tapnext"
tag=$1; kd=$2; base=${BASE:-t4}
EW_OPT=-O2 ./bb.sh $base-$tag --lookahead 2 --kdefs=$kd >/dev/null 2>&1
tail -n 1 build/$base-$tag.log | grep -q "rc=0" || { echo "BUILD FAILED $tag"; tail -n 5 build/$base-$tag.log; exit 1; }
./r8.sh r7_${base}_$tag run $base-$tag --iters 20 --warmup 10 >/dev/null
grep -E "TIME|attn O|RUN FAILED|rc=" build/r7_${base}_$tag.out
