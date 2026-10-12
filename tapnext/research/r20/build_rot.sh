#!/bin/bash
# pass 15: fit + link + lint + tcall + replay on the full12r18m placement for ROT variants solved by r20/gen_rot.py
#   r20/build_rot.sh g1a g0a      (from $SHM_SCRATCH/npu-research)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHM_SCRATCH=$(cd "${SHM_SCRATCH:-$REPO/../.scratch}" && pwd)   # scratch / working dir (docker mounts it as /work)
set -e
cd "$(dirname "$0")/.."
P="c_obj=0x78900,slots=0x77b80:0x78000:0x78480,rec=0x71804,b=0x7d100"
DK="docker run --rm --network none -v $SHM_SCRATCH:/work -v $REPO:$REPO npu-iron"
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
mkdir -p r20/build
for v in "$@"; do
  $DK bash -c "cd /work/npu-research && python3 r20/gen_rot.py fit r20/rot_$v.json --out r20/mm_rot_$v.s | head -2 && \
    $B/clang --target=aie2p-none-unknown-elf -c r20/mm_rot_$v.s -o r20/build/mm_rot_$v.o && \
    $B/ld.lld r20/build/mm_rot_$v.o -e mm_rot -o r20/build/mm_rot_$v.elf && \
    $B/llvm-objdump -d -z --no-show-raw-insn r20/build/mm_rot_$v.elf > r20/mm_rot_$v.dis && \
    $B/llvm-nm -S r20/build/mm_rot_$v.o | grep \" T mm_rot\" && \
    python3 ../../common/tools/asmlint.py r20/mm_rot_$v.s --zolstats | grep -v \"^  \" ; \
    rm -f r20/.fit_mm_rot.* ; chmod -R a+rw r20"
  python3 ../../common/tools/tcall.py "r20/mm_rot_$v.dis" --func mm_rot | head -2
  for a in 0x71980 0x71940 0x719c0; do
    echo -n "  acc_hi=$a: "; python3 r20/gen_rot.py check "r20/mm_rot_$v.s" --json "r20/rot_$v.json" --place "$P,acc_hi=$a"
  done
done
