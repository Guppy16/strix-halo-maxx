#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 23: r22/try22.sh (full12r22z) + the round-23 kept changes:
#   KTL (research r24/mm_e7k_r70b: one modulo loop per GEMM tile) as r23/mm_rot_k70p (the record period's lda.2d wrap an argument);
#   MSPLIT-P (down_b / mlp3_b run their last tile as two row passes, rb 0-5 then 6-17, so norm2 / norm0 start on rows 0-5;
#   MSPLIT= without, then the plain KTL kernel k70b), with acc_hi at 0x71A00;
#   YXSSM (the SSM starts during yx, z panels into half 1; YXSSM=0 without) + YXR (yx steps y / x interleaved; YXR=0 without)
#   + OUTPRE / ZCH (out issued before the SSM head fills; OUTPRE=0 without both) + OPH1 (out_proj A in half 1; its last
#   tile split too: MSPLIT includes out_proj);
#   --l2-depth 3 (L2=; five memtile BDs for the new tasks; 3 / 5 / 6 / 8 all within 0.02).
# E7_V=e7r68t: MSPLIT on the round-22 per-k-step kernel instead of KTL.
# usage: r23/try23.sh NAME [--sweep] [--ew-add ..] [--k-add ..] [--env K=V] [--args ..]   (NOREG / NOTIME as try21)
name=$1; shift
cd "$SHM_SCRATCH/npu-tapnext" || exit 1
R23=()
MS=${MSPLIT-down_b,mlp3_b,out_proj}
if [ -n "$MS" ]; then
  R23+=(--env MSPLIT=$MS --k-add -DMSPLIT)
  export E7_V=${E7_V:-k70p} KS_ACC_HI=${KS_ACC_HI:-0x71A00}
else
  export E7_V=${E7_V:-k70b} KS_ACC_HI=${KS_ACC_HI:-0x719C0}
fi
L2=${L2:-3}; [ "${YXSSM:-1}" = 1 ] && R23+=(--env YXSSM=1 --env YXR=${YXR:-1} --env ZCH=${OUTPRE:-1} --env OUTPRE=${OUTPRE:-1} --env OPH1=${OPH1:-1}); R23+=(--args "--l2-depth $L2")
exec r22/try22.sh "$name" "${R23[@]}" "$@"
