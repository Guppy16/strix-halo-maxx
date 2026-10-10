"""Round 15: one-core bench of resc (r15/rescb.cc).  python3 r15/rescb.py build --s build/r15/resc.s --out build/r15/rescb | run --out ..."""
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
    o.asm, o.cc = o.s, str(HERE / "rescb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed, sc in ((0, 1), (1, 100), (2, 10000), (3, 3)):
        y = d.call([0, seed, sc])
        print(f"seed {seed} scale {sc}: resc {y[0]} cycles, C++ {y[2]}; bad {y[1]}" + (f", first {y[3]}" if y[1] else "") + f"; guard changed {y[4]}")
