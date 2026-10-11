#!/bin/bash
# C++ (-DASM_SCORES $EXTRA) -> .s -> fixed dialect (asmfix.py), append the hand-scheduled scores.s, assemble -> p10/build/$OUT.o
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin; INC=/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include
cd /work/p10/build || exit 1
OUT=${OUT:-attn_asm2}; EXTRA=${EXTRA--DNO_EVENTS}
F="-O2 -DNDEBUG -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I$INC -I/work/mlir-aie/aie_kernels -DATTN_KEYS=729 -DATTN_SCALE=0.11785113019775793f $EXTRA -DASM_SCORES"
$P/clang++ $F -S /work/p10/attn15.cc -o ${OUT}_cpp.s 2>${OUT}.err || { cat ${OUT}.err; exit 1; }
python3 /work/p10/asmfix.py ${OUT}_cpp.s ${OUT}_fixed.s
cat ${OUT}_fixed.s /work/p10/scores.s > ${OUT}.s
$P/clang --target=aie2p-none-unknown-elf -c ${OUT}.s -o ${OUT}.o 2>${OUT}_as.err || { echo ASSEMBLE FAILED; grep error ${OUT}_as.err | head; exit 1; }
echo "events in object: $($P/llvm-objdump -d ${OUT}.o | grep -c 'event')"
$P/llvm-nm ${OUT}.o | grep -i "scores\|chunk_k"
chmod a+rw ${OUT}* 2>/dev/null; echo BUILT ${OUT}.o
