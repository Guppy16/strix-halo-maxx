"""Round 15: one-core bench of convk (r15/convb.cc).  python3 r15/convb.py build --s build/r15/convk.s --out build/r15/convb | run --out ..."""
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
    o.asm, o.cc = o.s, str(HERE / "convb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed, xq, wq in ((0, 4, 4), (1, 40, 2), (2, 400, 8), (3, 2, 400), (4, 4000, 4000)):
        y = d.call([0, seed, xq, wq])
        print(f"seed {seed} x {xq / 4} w {wq / 4}: convk {y[0]} cycles, C++ {y[2]}; bad {y[1]}"
              + (f", first {y[3]}: {y[4] & 0xFFFF:#06x} vs {y[5] & 0xFFFF:#06x}" if y[1] else "") + f"; guard changed {y[6]}")
