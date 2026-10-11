#!/bin/bash
# round 19 (in npu-iron): mm8.cc -> object with a build's kdefs (from build/BUILD.json) + extra -D flags; per-function text and the
# -z disassembly of one function. usage: r19/kcc.sh BUILD OUT.o FUNC [extra flags]  (build/*.h must be that build's headers)
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
bld=$1; out=$2; fn=$3; shift 3
cd /work/npu-tapnext
kd=$(python3 -c "import json; print(' '.join(json.load(open('build/$bld.json'))['kdefs'].split(',')))")
$B/clang++ -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf \
  -I/opt/ironenv/lib/python3.13/site-packages/llvm-aie/include/aie2p-none-unknown-elf -I/work/npu-moondream/mlir-aie/aie_kernels \
  -I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include -I/work/npu-tapnext -DDIM_M=144 -DDIM_N=64 -DDIM_K=64 -DRHO=9 \
  -DBLKU=1 -DATTN_KEYS=1072 -DATTN_SCALE=0.125f $kd "$@" -c mm8.cc -o $out 2>&1 | grep -i "error" | head
$B/llvm-nm --size-sort -S $out | grep -i " t " | python3 -c 'import sys; [print("%6d %s" % (int(p[1], 16), p[3])) for p in (l.split() for l in sys.stdin)]' | sort -n | grep -i "$fn"
$B/llvm-size $out | tail -1
$B/llvm-objdump -d -z --no-show-raw-insn --disassemble-symbols=$fn $out > ${out%.o}.dis
wc -l ${out%.o}.dis
