"""Round 15: one-core bench of ssq (r16/gen_ssq.py) vs ssm9::loga_exp_a2 (r15/ssqb.cc).
    python3 r16/ssqb.py build --s build/r15/ssx.s --out build/r16/ssqb ; python3 r16/ssqb.py run --out build/r16/ssqb"""
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
o = ap.parse_args()
if o.cmd == "build":
    o.asm, o.cc = o.s, str(HERE / "ssqb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed, q in ((0, 4), (1, 16), (2, 40), (3, 120), (4, 400)):  # sp max 1, 4, 10, 30, 100 (-60 clamp from ~7.5 on)
        y = d.call([0, seed, q])
        print(f"seed {seed} sp max {q / 4}: ssq {y[0]} cycles, C++ {y[3]}; bad a {y[1]}, bad mult {y[2]}"
              + (f", first {y[4]}: {y[5] & 0xFFFF:#06x} vs {y[6] & 0xFFFF:#06x}" if y[1] or y[2] else "") + f"; guard changed {y[7]}")
