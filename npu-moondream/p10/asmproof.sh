#!/bin/bash
# toolchain proof: C++ -> .s -> (asmfix) -> .o via the integrated assembler must equal C++ -> .o
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin; INC=/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include
mkdir -p /work/p10/build && cd /work/p10/build || exit 1
SRC=$1; TAG=$2; shift 2
F="-O2 -DNDEBUG -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I$INC -I/work/mlir-aie/aie_kernels $*"
$P/clang++ $F -c $SRC -o ${TAG}_direct.o 2>${TAG}_direct.err || { cat ${TAG}_direct.err; exit 1; }
$P/clang++ $F -S $SRC -o ${TAG}.s 2>${TAG}_s.err || { cat ${TAG}_s.err; exit 1; }
python3 /work/p10/asmfix.py ${TAG}.s ${TAG}_fixed.s
$P/clang --target=aie2p-none-unknown-elf -c ${TAG}_fixed.s -o ${TAG}_asm.o 2>${TAG}_asm.err || { echo "ASSEMBLE FAILED:"; grep -v "incorrect bundle" ${TAG}_asm.err | grep error | head -20; exit 1; }
ls -la ${TAG}_direct.o ${TAG}_asm.o
$P/llvm-objdump -d --no-show-raw-insn ${TAG}_direct.o > ${TAG}_direct.dis; $P/llvm-objdump -d --no-show-raw-insn ${TAG}_asm.o > ${TAG}_asm.dis
echo "disassembly diff lines: $(diff ${TAG}_direct.dis ${TAG}_asm.dis | grep -c '^[<>]')"
for s in $($P/llvm-readelf -S ${TAG}_direct.o | grep -oE '\.(text|data|bss|rodata)[A-Za-z_0-9.]*' | sort -u); do
  $P/llvm-objcopy -O binary --only-section=$s ${TAG}_direct.o d.bin 2>/dev/null; $P/llvm-objcopy -O binary --only-section=$s ${TAG}_asm.o a.bin 2>/dev/null
  if cmp -s d.bin a.bin; then echo "same  $s $(wc -c < d.bin) B"; else echo "DIFF  $s $(wc -c < d.bin) vs $(wc -c < a.bin) B"; fi
done
echo "asm lines $(wc -l < ${TAG}.s)"
chmod a+rw ${TAG}* d.bin a.bin 2>/dev/null
