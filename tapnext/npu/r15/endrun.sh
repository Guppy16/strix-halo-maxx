#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# round 15: depth variants of a build suffix, the sweep and DAVIS at depth 12 / 10 (host side; sequential)
S=$1
cd "$SHM_SCRATCH/npu-tapnext"
for d in 10 8 6; do
  P60G=1 P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 EW_DEFS="${EWD:--DSSM_LUT -DSSM_EXP3 -DSIGK -DSSX -DCPY64 -DCONVK -DRECK -DNRMK}" AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full$d$S --lookahead 2 --l2-depth 8 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU,-DATTKF,-DATTKE,-DATTKB,-DKFOLD,-DZASM,-DATTV,-DVTK,-DRESCK 2>&1 | tail -1
done
TMO=1500 TAILN=12 ./s8.sh ${S}_sweep sweep --build full12$S --iters 60 --warmup 20
TMO=3000 TAILN=6 ./s8.sh ${S}_davis12 davis --build full12$S --depth 12 --videos 10
TMO=3000 TAILN=6 ./s8.sh ${S}_davis10 davis --build full12$S --depth 10 --videos 10
echo ENDRUN_DONE
