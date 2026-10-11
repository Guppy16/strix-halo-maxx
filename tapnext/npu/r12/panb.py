"""Round 12: one-core bench of pan (r12/gen_pan.py) vs mm8.cc's C++ panel pushes (r12/panb.cc).
    python3 r12/panb.py build --s build/r12/pan.s --out build/r12/panb ; python3 r12/panb.py run --out build/r12/panb"""
import argparse
import shutil
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
    shutil.copy(Path(o.s).with_suffix(".h"), HERE.parent / "build" / "r12" / "pan_bench.h")
    o.asm, o.cc = o.s, str(HERE / "panb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for jump, group in ((0, 6), (1152, 3), (1152, 2)):
        for seed in (0, 1):
            y = d.call([0, jump, group, seed])
            print(f"jump {jump} group {group} seed {seed}: pan {y[0]} cycles ({y[0] / 6:.0f} per row block), C++ {y[3]} "
                  f"({y[3] / 6:.0f}); {'OK' if y[1] == 0 else f'BAD {y[1]} bytes, first {y[2]}'}")
