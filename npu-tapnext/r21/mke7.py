"""Round 21 E7'': take research's generated mm_e7 kernel (npu-research r22/gen_e7.py fit -> mm_e7_<v>.s / .h) into design8's ROT slot:
symbol / labels / arrays renamed to mm_rot's (design8 ROT_V=<v> copies r21/mm_rot_<v>.s / .h; mm8.cc -DE7 calls it with the X buffer).
usage: python3 r21/mke7.py ../npu-research/r22/mm_e7_r68try.s r21/mm_rot_e7r68.s"""
import re
import sys
from pathlib import Path

s_in, s_out = Path(sys.argv[1]), Path(sys.argv[2])
s = s_in.read_text()
assert "mov r6, r0" in s, "not a ZLOAD-ready kernel"
s = re.sub(r"\bmm_e7\b", "mm_rot", s).replace(".Lmm_e7_", ".Lmm_rot_").replace("mm_e7_start", "mm_rot_start").replace("mm_e7_last", "mm_rot_last")
s_out.write_text(s)
h = s_in.with_suffix(".h").read_text()
h = h.replace("MM_E7_NREC", "MM_ROT_NREC").replace("mm_e7_order", "mm_rot_order").replace("mm_e7_border", "mm_rot_border")
s_out.with_suffix(".h").write_text(h)
print(f"mke7: {s_out} / .h")
