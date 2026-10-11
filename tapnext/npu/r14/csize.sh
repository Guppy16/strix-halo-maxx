#!/bin/bash
# round 14 (in npu-iron): mm8.cc -> object with the deployed GEMM-core flags (full12r13f) + extra -D flags; per-function
# text sizes. usage: r14/csize.sh OUT.o [extra flags]   (the generated asm headers in build/ must exist)
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
out=$1; shift
cd /work/npu-tapnext
$B/clang++ -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf \
  -I/opt/ironenv/lib/python3.13/site-packages/llvm-aie/include/aie2p-none-unknown-elf -I/work/npu-moondream/mlir-aie/aie_kernels \
  -I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include -I/work/npu-tapnext -DDIM_M=144 -DDIM_N=64 -DDIM_K=64 -DRHO=9 \
  -DBLKU=1 -DATTN_KEYS=1072 -DATTN_SCALE=0.125f -Os -O2 -DMM_ASM -DMM_KSTEP -DMM_PIECES -DMM_KS -DGELU_MS -DGUP -DCPY32 -DPAN \
  -DGELU_FLAT -DBGELU -DATTKF -DATTKE -DATTKB  "$@" -c mm8.cc -o $out 2>&1 | grep -i "error" | head
$B/llvm-nm --size-sort -S $out | grep -i " t " | python3 -c 'import sys; [print("%6d %s" % (int(p[1], 16), p[3])) for p in (l.split() for l in sys.stdin)]' | sort -n | tail -${N:-30}
$B/llvm-size $out | tail -1
