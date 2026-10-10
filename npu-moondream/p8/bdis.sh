#!/bin/bash
# usage: bdis.sh NAME -D... : compile the kernel alone and disassemble mm_acc (inside npu-iron)
name=$1; shift
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
INC=/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include
cd /work/p8/build
$P/clang++ -O2 -DNDEBUG -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I$INC -I/work/mlir-aie/aie_kernels -DDIM_M=96 -DDIM_N=96 -DDIM_K=64 "$@" -c /work/p8/mm_f32acc.cc -o $name.o 2> $name.cerr || { cat $name.cerr; exit 1; }
$P/llvm-objdump -d --no-show-raw-insn $name.o > $name.dis
awk "/<mm_acc>:/,/<acc_store>|section .text.acc_store/" $name.dis > $name.mm.dis
chmod a+rw $name.*
