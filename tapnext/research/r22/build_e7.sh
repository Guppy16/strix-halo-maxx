#!/bin/bash
# pass 17: fit (prologue alignment) + assemble + link + asmlint + whole-call replay (pads 0 / 1, both B buffers, acc_hi range)
# for E7'' schedules solved by r22/gen_e7.py:   r22/build_e7.sh r64_x12 [r66_x8 ...]   (json r22/e7_<v>.json)
set -e
cd "$(dirname "$0")/.."
DK="docker run --rm --network none -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron"
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
mkdir -p r22/build
for v in "$@"; do
  $DK bash -c "cd /work/npu-research && python3 r22/gen_e7.py fit r22/e7_$v.json --out r22/mm_e7_$v.s && \
    $B/clang --target=aie2p-none-unknown-elf -c r22/mm_e7_$v.s -o r22/build/mm_e7_$v.o && \
    $B/ld.lld r22/build/mm_e7_$v.o -e mm_e7 -o r22/build/mm_e7_$v.elf && \
    $B/llvm-objdump -d -z --no-show-raw-insn r22/build/mm_e7_$v.elf > r22/mm_e7_$v.dis && \
    $B/llvm-nm -S r22/build/mm_e7_$v.o | grep \" T mm_e7\" ; \
    python3 ../../common/tools/asmlint.py r22/mm_e7_$v.s --zolstats | grep -v \"^  \" ; \
    rm -f r22/.fit_mm_e7.* ; chmod -R a+rw r22"
  python3 ../../common/tools/tcall.py "r22/mm_e7_$v.dis" --func mm_e7 | head -2 || true
  for a in 0x71900 0x71940 0x71980; do
    echo -n "  acc_hi=$a: "; python3 r22/gen_e7.py check "r22/mm_e7_$v.s" --json "r22/e7_$v.json" --place "acc_hi=$a" | tail -1
  done
done
