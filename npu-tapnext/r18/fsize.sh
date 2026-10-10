#!/bin/bash
# round 18 (in npu-iron): per-function text of a linked core elf. usage: r18/fsize.sh ELF
B=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
$B/llvm-size -A $1 | grep -E "^\.text|^\.data|^\.bss"
$B/llvm-nm --size-sort -S $1 | grep -i " t " | python3 -c "import sys; [print(\"%6d %s\" % (int(p[1], 16), p[3])) for p in (l.split() for l in sys.stdin)]" | sort -n | tail -${N:-60}
