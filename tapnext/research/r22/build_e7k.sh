#!/bin/bash
# pass 17: KTL kernels from r22/gen_e7k.py: fit + assemble + link + asmlint + tcall + multi-k-step replay (2..5 k-step tiles)
#   r22/build_e7k.sh r70      (json r22/e7k_<v>_n16.json -> r22/mm_e7k_<v>.s / .h / .dis)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
set -e
cd "$(dirname "$0")/.."
DK="docker run --rm --network none -v $SHM_SCRATCH:/work -v $SHM_REPO:$SHM_REPO npu-iron"
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
mkdir -p r22/build
for v in "$@"; do
  J=$(ls r22/e7k_${v}_n*.json | head -1)
  $DK bash -c "cd /work/npu-research && python3 r22/gen_e7k.py fit $J --out r22/mm_e7k_$v.s && \
    $B/clang --target=aie2p-none-unknown-elf -c r22/mm_e7k_$v.s -o r22/build/mm_e7k_$v.o && \
    $B/ld.lld r22/build/mm_e7k_$v.o -e mm_e7k -o r22/build/mm_e7k_$v.elf && \
    $B/llvm-objdump -d -z --no-show-raw-insn r22/build/mm_e7k_$v.elf > r22/mm_e7k_$v.dis && \
    $B/llvm-nm -S r22/build/mm_e7k_$v.o | grep \" T mm_e7k\" ; \
    python3 ../../common/tools/asmlint.py r22/mm_e7k_$v.s --zolstats | grep -v \"^  \" ; \
    rm -f r22/.fit_mm_e7k.* ; chmod -R a+rw r22"
  python3 ../../common/tools/tcall.py "r22/mm_e7k_$v.dis" --func mm_e7k | head -2 || true
  for n in 2 3 5 13; do
    echo -n "  nks=$n: "; python3 r22/gen_e7k.py check "r22/mm_e7k_$v.s" --json "$J" --nks $n | tail -1
  done
done
