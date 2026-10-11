#!/bin/bash
# usage (in npu-iron, /work = .scratch): r9/xs.sh SRC OUT.s [flags]   compile to asm like the EW kernels (loop-count experiments)
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
src=$1; out=$2; shift 2
$B/clang++ -O2 -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf \
  -I/opt/ironenv/lib/python3.13/site-packages/llvm-aie/include/aie2p-none-unknown-elf -I/work/npu-moondream/mlir-aie/aie_kernels \
  -I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include "$@" -S $src -o $out 2>&1 | grep -E "error" | head -5
