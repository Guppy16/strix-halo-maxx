"""Round 12: one-core bench of gup (r12/gen_gup.py) against today's C++ gated path (r12/gupb.cc).

    python3 r12/gupb.py build --s build/r12/gup.s --out build/r12/gupb      (npu-iron)
    python3 r12/gupb.py run --out build/r12/gupb                            (npu-iron, NPU)
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
import apb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--s", help="gup asm (its .h next to it gives GUP_U)")
    ap.add_argument("--seeds", default="0,1,2")
    o = ap.parse_args()
    if o.cmd == "build":
        shutil.copy(Path(o.s).with_suffix(".h"), HERE.parent / "build" / "r12" / "gup_bench.h")
        o.asm, o.cc = o.s, str(HERE / "gupb.cc")
        apb.build(o)
        return
    d = apb.Dev(o.out)
    for stride in (320, 576):
        for seed in (int(s) for s in o.seeds.split(",")):
            y = d.call([0, stride, seed])
            nb = 6 * 4  # blocks
            ok = "OK" if y[1] == 0 else f"BAD {y[1]} bytes, first {y[2]}: got {y[4]:#04x} want {y[5]:#04x}"
            print(f"stride {stride} seed {seed}: gup {y[0]} cycles ({y[0] / nb:.1f} per 64-lane block), C++ {y[3]} "
                  f"({y[3] / nb:.1f})  {ok}", flush=True)


if __name__ == "__main__":
    main()
