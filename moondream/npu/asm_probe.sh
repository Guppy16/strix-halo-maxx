#!/bin/bash
# Can Peano's clang assemble AIE2P assembly (integrated assembler), without llvm-mc?
P=$PEANO_INSTALL_DIR
ls $P/bin | tr '\n' ' '; echo
cd /tmp
printf 'int add(int a, int b) { return a + b; }\n' > t.c
$P/bin/clang --target=aie2p-none-unknown-elf -O2 -S t.c -o t.s && echo "--- emitted asm:" && grep -vE '^\s*(\.|//|$)' t.s | head -12
$P/bin/clang --target=aie2p-none-unknown-elf -c t.s -o t.o && echo "ASSEMBLE OK" && $P/bin/llvm-objdump -d t.o | tail -6
