#!/bin/bash
# C++ (events on, -DASM_SCORES) compiled normally + scores.s assembled separately, merged with ld.lld -r -> p10/build/attn_hs_ev.o
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin; INC=/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include
cd /work/p10/build || exit 1
F="-O2 -DNDEBUG -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I$INC -I/work/mlir-aie/aie_kernels -DATTN_KEYS=729 -DATTN_SCALE=0.11785113019775793f -DASM_SCORES"
$P/clang++ $F -c /work/p10/attn15.cc -o attn_cpp_ev.o 2>merge.err || { cat merge.err; exit 1; }
$P/clang --target=aie2p-none-unknown-elf -c /work/p10/scores.s -o scores.o 2>>merge.err || { cat merge.err; exit 1; }
$P/ld.lld -r attn_cpp_ev.o scores.o -o attn_hs_ev.o 2>>merge.err || { cat merge.err; exit 1; }
$P/llvm-nm attn_hs_ev.o | grep -i "scores_asm\|chunk_k\|chunk_v"
echo "events: $($P/llvm-objdump -d attn_hs_ev.o | grep -c event)"
chmod a+rw attn_cpp_ev.o scores.o attn_hs_ev.o; echo BUILT
