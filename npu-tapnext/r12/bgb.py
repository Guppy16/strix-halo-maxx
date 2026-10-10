"""Round 12: one-core bench of bgelu (r12/gen_bgelu.py) vs bias_pcs + gelu_lut_n (r12/bgb.cc).
    python3 r12/bgb.py build --s build/r12/bg.s --out build/r12/bgb ; python3 r12/bgb.py run --out build/r12/bgb"""
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
    shutil.copy(Path(o.s).with_suffix(".h"), HERE.parent / "build" / "r12" / "bgelu_bench.h")
    o.asm, o.cc = o.s, str(HERE / "bgb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for erf in (0, 1):
        for seed, scale in ((0, 0), (1, 0), (2, 8), (3, 40), (4, 2)):
            y = d.call([0, erf, seed, scale])
            print(f"erf {erf} seed {seed} scale {scale * 0.25 if scale else 6.0}: bgelu {y[0]} cycles ({y[0] / 32:.1f} per 32 lanes), "
                  f"C++ {y[3]} ({y[3] / 32:.1f}); {'OK' if y[1] == 0 else f'BAD {y[1]}, first {y[2]}: {y[4]:#06x} vs {y[5]:#06x}'}")
