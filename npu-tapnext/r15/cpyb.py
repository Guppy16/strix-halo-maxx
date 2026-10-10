"""Round 15: one-core bench of cpy64 (r15/cpyb.cc).  python3 r15/cpyb.py build|run --out build/r15/cpyb [--s build/r15/cpy64.s]"""
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
    o.asm, o.cc = o.s, str(HERE / "cpyb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for n in (16, 32, 16):
        y = d.call([0, n])
        print(f"cpy64 {n} vectors: {y[0]} cycles, bad {y[1]}, guard changed {y[2]}")
