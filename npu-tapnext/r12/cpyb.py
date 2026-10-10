"""Round 12: one-core bench of cpy32 (r12/gen_copy.py) via r11/apb.py's harness and r12/cpyb.cc.
    python3 r12/cpyb.py build --s build/r12/cpy32.s --out build/r12/cpyb ; python3 r12/cpyb.py run --out build/r12/cpyb"""
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
    shutil.copy(Path(o.s).with_suffix(".h"), HERE.parent / "build" / "r12" / "cpy32_bench.h")
    o.asm, o.cc = o.s, str(HERE / "cpyb.cc")
    apb.build(o)
else:
    y = apb.Dev(o.out).call([0])
    print(f"cpy32: {y[0]} cycles for 162 units; wrong bytes {y[1]} (first {y[2]}: got {y[3]:#x} want {y[4]:#x})")
