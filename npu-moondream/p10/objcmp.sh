#!/bin/bash
# compare two AIE objects: whole file, section contents (objdump -s), disassembly
P=/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin
cd /work/p10/build
A=$1; B=$2
cmp $A $B && echo WHOLE-FILE-IDENTICAL
$P/llvm-objdump -s $A | grep -v "file format" > sd.hex; $P/llvm-objdump -s $B | grep -v "file format" > sa.hex
echo "section-contents diff lines: $(diff sd.hex sa.hex | grep -c '^[<>]')"
diff sd.hex sa.hex | head -8
$P/llvm-objdump -d --no-show-raw-insn $A | grep -v "file format" > dd.dis; $P/llvm-objdump -d --no-show-raw-insn $B | grep -v "file format" > da.dis
echo "disassembly diff lines: $(diff dd.dis da.dis | grep -c '^[<>]')"; diff dd.dis da.dis | head -6
chmod a+rw sd.hex sa.hex dd.dis da.dis
