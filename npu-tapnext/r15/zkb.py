"""Round 15: one-core bench of zk (r15/zkb.cc).  build --s build/r15/zk.s --out build/r15/zkb | run --out ..."""
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
    o.asm, o.cc = o.s, str(HERE / "zkb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed in (0, 1, 2):
        y = d.call([0, seed])
        print(f"seed {seed}: zk {y[0]} cycles, C++ {y[2]}; bad panel bytes {y[1]}; bytes written past 576: {y[3]}")
