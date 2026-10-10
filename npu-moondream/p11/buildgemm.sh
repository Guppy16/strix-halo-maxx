#!/bin/bash
# C++ wrapper (-DASM_BLK, events on) compiled normally + gemm_blk.s assembled, merged with ld.lld -r -> p11/build/mm_asm.o
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin; INC=/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include
cd /work/p11/build || exit 1
F="-O2 -DNDEBUG -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I$INC -I/work/mlir-aie/aie_kernels -DDIM_M=96 -DDIM_N=96 -DDIM_K=64 -DRHO=6 -DNO_PREFETCH -DBLK12 -DBLKU=2 -DASM_BLK"
$P/clang++ $F -c /work/p11/mm_f32acc.cc -o mm_cpp.o 2>mm.err || { cat mm.err; exit 1; }
$P/clang --target=aie2p-none-unknown-elf -c /work/p11/gemm_blk.s -o gemm_blk.o 2>>mm.err || { grep error mm.err | head; exit 1; }
$P/ld.lld -r mm_cpp.o gemm_blk.o -o mm_asm.o 2>>mm.err || { cat mm.err; exit 1; }
$P/llvm-nm mm_asm.o | grep -i " T "
chmod a+rw mm_cpp.o gemm_blk.o mm_asm.o; echo BUILT mm_asm.o
