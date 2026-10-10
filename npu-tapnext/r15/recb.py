"""Round 15: one-core bench of reck (r15/recb.cc).  python3 r15/recb.py build --s build/r15/reck4.s --out build/r15/recb | run --out ..."""
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
    o.asm, o.cc = o.s, str(HERE / "recb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed, first in ((0, 0), (1, 1), (2, 0), (3, 1)):
        y = d.call([0, seed, first])
        print(f"seed {seed} first {first}: reck {y[0]} cycles, C++ {y[3]}; bad hb {y[1]}, bad h {y[2]}"
              + (f", first bad {y[4]}" if y[1] or y[2] else "") + f"; guard changed {y[5]}")
