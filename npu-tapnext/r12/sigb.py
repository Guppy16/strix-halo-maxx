"""Round 12: one-core bench of sigk (r12/gen_sig.py) vs ssm9::sigmoid_lut_n (r12/sigb.cc).
    python3 r12/sigb.py build --s build/r12/sig.s --out build/r12/sigb ; python3 r12/sigb.py run --out build/r12/sigb"""
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
    shutil.copy(Path(o.s).with_suffix(".h"), HERE.parent / "build" / "r12" / "sig_bench.h")
    o.asm, o.cc = o.s, str(HERE / "sigb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed, q in ((0, 24), (1, 64), (2, 4), (3, 80)):
        y = d.call([0, seed, q])
        print(f"seed {seed} scale {q / 4}: sigk {y[0]} cycles, C++ {y[3]}; {'OK' if y[1] == 0 else f'BAD {y[1]}, first {y[2]}: {y[4]:#06x} vs {y[5]:#06x}'}"
              f"; guard before {y[6]} B changed, garbage vectors after {y[7]}")
        if y[1] and seed == 0:
            import numpy as np
            got = y[16:528].view(np.uint16).reshape(32, 32)
            ref = y[528:1040].view(np.uint16).reshape(32, 32)
            for k in range(32):  # which reference vector (if any) each output vector equals
                m = [j for j in range(32) if (got[k] == ref[j]).all()]
                print(f"  out vector {k:2d}: equals ref {m}, first lanes {[hex(v) for v in got[k][:3]]} want {[hex(v) for v in ref[k][:3]]}")
