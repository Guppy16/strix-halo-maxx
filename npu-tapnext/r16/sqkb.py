"""Round 16: one-core bench of sqk (r16/gen_sqk.py) vs ssm9::sqrt_lut_n (r16/sqkb.cc), every bf16 0x0000-0x7FFF.
    python3 r16/sqkb.py build --s build/r16/sqk.s --out build/r16/sqkb ; python3 r16/sqkb.py run --out build/r16/sqkb"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
import apb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "run"])
ap.add_argument("--out", required=True)
ap.add_argument("--s")
ap.add_argument("--cc", default=str(HERE / "sqkb.cc"))
o = ap.parse_args()
if o.cmd == "build":
    o.asm = o.s
    apb.build(o)
else:
    d = apb.Dev(o.out)
    y = d.call([0, 0, 1])  # one block first: a hang shows up alone
    print(f"block 0: sqk {y[0]} cycles, C++ {y[2]}; bad {y[1]}, guard changed {y[6]}", flush=True)
    tot = 0
    for b0 in range(0, 32768, 512 * 16):
        y = d.call([0, b0, 16])
        tot += int(y[1])
        print(f"patterns {b0:#06x}-{b0 + 512 * 16 - 1:#06x}: sqk {y[0]} cycles, C++ {y[2]}; bad {y[1]}"
              + (f", first {y[3]:#06x}: {y[4] & 0xFFFF:#06x} vs {y[5] & 0xFFFF:#06x}" if y[1] else "") + f"; guard changed {y[6]}", flush=True)
    print("TOTAL bad", tot)
