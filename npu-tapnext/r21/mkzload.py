"""Round 21 ZLOAD: derive mm_rot_<v>z.s from a deployed ROT kernel: every kp-0 MAC (A operand ex0, the job's first MAC on its freshly
loaded accumulator) takes its config from r6 instead of r3; r6 = the 4th argument (r0 of a void function), copied before the preamble
overwrites r0. mm8.cc -DZLOAD passes 781 (780 | zero_acc, aie2p_vmult.h bit 0) for a tile's first k-step and 780 otherwise, so the
first k-step ignores whatever the pieces hold and acc_zero / acc_zero_hi have nothing to do.
usage: python3 r21/mkzload.py r19/mm_rot_g0a.s r21/mm_rot_g0z.s"""
import re
import sys

src = open(sys.argv[1]).read()
n0 = len(re.findall(r"vmac\.f dm\d, dm\d, ex0, ex\d+, r3", src))
assert n0 > 0 and not re.search(r"\br6\b", src)
out = re.sub(r"(vmac\.f dm\d, dm\d, ex0, ex\d+), r3", r"\1, r6", src)
assert not re.search(r"vmac\.f dm\d, dm\d, ex0, ex\d+, r3", out)
out = out.replace("mm_rot:\n\tmovs p3, p2\n", "mm_rot:\n\tmov r6, r0\n\tmovs p3, p2\n", 1)
assert "mov r6, r0" in out
open(sys.argv[2], "w").write(out)
print(f"mkzload: {n0} kp-0 MACs -> r6")
