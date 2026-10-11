"""Round 22 KTL: take research's one-loop-per-tile kernel (npu-research r22/gen_e7k.py -> mm_e7k_<v>.s / .h) into design8's ROT slot
(ROT_V=<v>: design8 copies r22/mm_rot_<v>.s / .h). The symbol stays mm_e7k (mm8 -DE7 calls it from mm_ktile when the header defines
MM_E7K_NPER); the header gains mm_rot_border (pack8.rot_order reads the B order under that name).
usage: python3 r22/mkktl.py ../npu-research/r22/mm_e7k_r70.s r22/mm_rot_k70.s"""
import re
import sys
from pathlib import Path

s_in, s_out = Path(sys.argv[1]), Path(sys.argv[2])
s = s_in.read_text()
assert "mov r6, r0" in s and "mm_e7k:" in s, "not a KTL kernel"
s_out.write_text(s)
h = s_in.with_suffix(".h").read_text()
b = re.search(r"mm_e7k_border\[64\] = \{([^}]*)\}", h).group(1)
h += f"// main round 22 (r22/mkktl.py): the B order under the ROT name for pack8\nstatic constexpr int8_t mm_rot_border[64] = {{{b}}};\n"
s_out.with_suffix(".h").write_text(h)
print(f"mkktl: {s_out} / .h")
