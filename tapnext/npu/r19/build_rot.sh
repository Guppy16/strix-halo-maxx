#!/bin/bash
SHM_REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel); SHM_SCRATCH=${SHM_SCRATCH:-$SHM_REPO/../.scratch}
# round 19: research's r19/build_all.sh on main's r18/gen_rot.py (ROT_ACQ_GAP / ROT_REL_GAP, the probed g0 wrap): solve + fit + link +
# lint + tcall + replay per variant; output r19/rot_<v>.json, r19/mm_rot_<v>.s / .h / .dis.
#   r19/build_rot.sh [--solve] g1a g0a     (v = g<g><a = acq gap 6 + rel gap 8, --free>)
set -e
cd "$SHM_SCRATCH/npu-tapnext"
SOLVE=0
[ "$1" = "--solve" ] && { SOLVE=1; shift; }
P="c_obj=0x78900,slots=0x77b80:0x78000:0x78480,rec=0x71800,b=0x7d100"
PL=("$P,acc_hi=0x71980" "$P,acc_hi=0x719c0" "$P,acc_hi=0x71a00" "$P,acc_hi=0x71940")
PLACES=(); for p in "${PL[@]}"; do PLACES+=(--place "$p"); done
UV="nice -n 10 $HOME/.local/bin/uv run --no-project --with ortools --with numpy python"
DK="docker run --rm --network none -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron"
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
mkdir -p build/r19
for v in "$@"; do
  g=${v:1:1}
  [ $SOLVE = 1 ] && ROT_ACQ_GAP=${ROT_ACQ_GAP:-6} ROT_REL_GAP=${ROT_REL_GAP:-8} $UV r18/gen_rot.py solve --g "$g" --free --time ${TIME:-900} --workers 12 --json "r19/rot_$v.json" "${PLACES[@]}" | head -3
  $DK bash -c "cd /work/npu-tapnext && nice -n 10 python3 r18/gen_rot.py fit r19/rot_$v.json --out r19/mm_rot_$v.s | head -2 && \
    $B/clang --target=aie2p-none-unknown-elf -c r19/mm_rot_$v.s -o build/r19/mm_rot_$v.o && \
    $B/ld.lld build/r19/mm_rot_$v.o -e mm_rot -o build/r19/mm_rot_$v.elf && \
    $B/llvm-objdump -d -z --no-show-raw-insn build/r19/mm_rot_$v.elf > r19/mm_rot_$v.dis && \
    $B/llvm-nm -S build/r19/mm_rot_$v.o | grep ' T mm_rot' && \
    python3 ../../common/tools/asmlint.py r19/mm_rot_$v.s --zolstats | grep -v '^  ' ; \
    rm -f r19/.fit_mm_rot.* ; chmod a+rw r19/mm_rot_$v.* r19/rot_$v.json build/r19/*"
  python3 ../../common/tools/tcall.py "r19/mm_rot_$v.dis" --func mm_rot | head -2
  for pl in "${PL[@]}"; do echo -n "  $pl: "; python3 r18/gen_rot.py check "r19/mm_rot_$v.s" --json "r19/rot_$v.json" --place "$pl" | tail -1; done
done
