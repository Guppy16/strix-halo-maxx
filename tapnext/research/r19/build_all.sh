#!/bin/bash
# analysis pass 14: solve (optional) + fit + link + lint + tcall + replay on the r17d / r18h-ROT placements, per ROT variant.
#   r19/build_all.sh [--solve] g2f g1f g0f        (from $SHM_SCRATCH/npu-research)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
set -e
cd "$(dirname "$0")/.."
SOLVE=0
[ "$1" = "--solve" ] && { SOLVE=1; shift; }
P="c_obj=0x78900,slots=0x77b80:0x78000:0x78480,rec=0x71800,b=0x7d100"
PLACES=(--place "acc_hi=0x718c0" --place "$P,acc_hi=0x71900" --place "$P,acc_hi=0x71980" --place "$P,acc_hi=0x719c0" --place "$P,acc_hi=0x71a00")
UV="nice -n 10 $HOME/.local/bin/uv run --no-project --with ortools --with numpy python"
DK="docker run --rm --network none -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron"
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
for v in "$@"; do
  g=${v:1:1}
  free=$([ "${v:2:1}" = f ] && echo --free || true)
  [ $SOLVE = 1 ] && $UV r19/gen_rot.py solve --g "$g" $free --time 600 --json "r19/rot_$v.json" "${PLACES[@]}" | head -1
  $DK bash -c "cd /work/npu-research && python3 r19/gen_rot.py fit r19/rot_$v.json --out r19/mm_rot_$v.s | head -2 && \
    $B/clang --target=aie2p-none-unknown-elf -c r19/mm_rot_$v.s -o r19/build/mm_rot_$v.o && \
    $B/ld.lld r19/build/mm_rot_$v.o -e mm_rot -o r19/build/mm_rot_$v.elf && \
    $B/llvm-objdump -d -z --no-show-raw-insn r19/build/mm_rot_$v.elf > r19/mm_rot_$v.dis && \
    $B/llvm-nm -S r19/build/mm_rot_$v.o | grep ' T mm_rot' && \
    python3 ../../common/tools/asmlint.py r19/mm_rot_$v.s --zolstats | grep -v '^  ' && \
    rm -f r19/.fit_mm_rot.* && chmod a+rw r19/mm_rot_$v.* r19/rot_$v.json"
  python3 ../../common/tools/tcall.py "r19/mm_rot_$v.dis" --func mm_rot | head -2
  for pl in "acc_hi=0x718c0" "$P,acc_hi=0x71900" "$P,acc_hi=0x71980" "$P,acc_hi=0x719c0" "$P,acc_hi=0x71a00"; do
    echo -n "  $pl: "; python3 r19/gen_rot.py check "r19/mm_rot_$v.s" --json "r19/rot_$v.json" --place "$pl"
  done
done
