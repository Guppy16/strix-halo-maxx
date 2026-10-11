#!/bin/bash
# round 23: r22/try22.sh (full12r22z) + the round-23 kept changes: MSPLIT-P (down_b / mlp3_b run their last tile as two row passes,
# rb 0-5 then 6-17, so norm2 / norm0 start on rows 0-5; MSPLIT= without), which needs the E7 kernel with the row-block count in r1
# (r23/mm_rot_e7r68t), acc_hi at 0x71A00 (the pass-1 record table) and --l2-depth 6 (two memtile BDs for the new A send tasks).
# usage: r23/try23.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V] [--args ..]   (NOREG / NOTIME as try21)
name=$1; shift
cd ~/git/app/.scratch/npu-tapnext || exit 1
R23=()
MS=${MSPLIT-down_b,mlp3_b}
if [ -n "$MS" ]; then
  R23+=(--env MSPLIT=$MS --k-add -DMSPLIT --args "--l2-depth 6")
  export E7_V=${E7_V:-e7r68t} KS_ACC_HI=${KS_ACC_HI:-0x71A00}
fi
exec r22/try22.sh "$name" "${R23[@]}" "$@"
