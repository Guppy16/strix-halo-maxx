#!/bin/bash
# usage (in npu-iron): r10/mmasm.sh OUT.s [extra -D flags]   mm8.cc -> asm with the deployed GEMM-core flags (r9s)
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
out=$1; shift
cd /work/npu-tapnext && python3 r9/gen_ks.py build/mm_ks.s > /dev/null
$B/clang++ -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I/opt/ironenv/lib/python3.13/site-packages/llvm-aie/include/aie2p-none-unknown-elf -I/work/npu-moondream/mlir-aie/aie_kernels -I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include -I/work/npu-tapnext -DDIM_M=144 -DDIM_N=64 -DDIM_K=64 -DRHO=9 -DBLKU=1 -DATTN_KEYS=1072 -DATTN_SCALE=0.125f -Os -O2 -DMM_ASM -DMM_KSTEP -DMM_PIECES -DMM_KS "$@" -S r10/gx.cc -o $out 2>&1 | grep -i "error" | head
