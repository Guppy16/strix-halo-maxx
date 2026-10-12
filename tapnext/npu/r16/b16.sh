#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# round 16 (host): build NAME = full12r15m's configuration + extra EW_DEFS ($EWX) / kdefs ($KX, comma list) / design8 args
# usage: [D=depth] [EWX="-DSQK"] [KX=",-DX"] r16/b16.sh NAME [design8 args...]
name=$1; shift
cd "$SHM_SCRATCH/npu-tapnext"
P60G=1 P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 \
EW_DEFS="-DSSM_LUT -DSSM_EXP3 -DSIGK -DSSX -DCPY64 -DCONVK -DRECK -DNRMK -DZRING ${EWX}" AIECC_EXTRA=--reclaim-runtime-bds \
./bb.sh $name --lookahead 2 --l2-depth 8 \
  --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU,-DATTKF,-DATTKE,-DATTKB,-DKFOLD,-DZASM,-DATTV,-DVTK,-DRESCK${KX} "$@" 2>&1 | tail -${TAILN:-2}
docker run --rm --name nt16-sz-$name -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron bash -c "B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin; cd /work/npu-tapnext/build/$name.prj && for c in 0_2 0_5; do echo -n \"core \$c text: \"; \$B/llvm-size elfs_main_core_\$c/elfs_main_core_\$c.elf | tail -1 | awk '{print \$1}'; done" 2>/dev/null
