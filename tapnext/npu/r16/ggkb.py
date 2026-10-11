"""Round 16: one-core bench of ggk (r16/gen_ggk.py) vs two gate_gemm calls (r16/ggkb.cc).
    python3 r16/ggkb.py build --s build/r16/ggk.s --out build/r16/ggkb ; python3 r16/ggkb.py run --out build/r16/ggkb"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
import apb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "run"])
ap.add_argument("--out", required=True)
ap.add_argument("--s")
ap.add_argument("--cc", default=str(HERE / "ggkb.cc"))
o = ap.parse_args()
if o.cmd == "build":
    o.asm = o.s
    apb.build(o)
else:
    d = apb.Dev(o.out)
    import os
    for kk in range(1, 9) if os.environ.get("KK") else ():
        y = d.call([0, 3, 128, kk, int(os.environ.get('ZB', '0'))])
        print(f"only A panel {kk - 1}: bad {y[1]}", flush=True)
        if os.environ.get("DUMPKK") == str(kk - 1):
            got, want = y[16:48].view(np.uint16), y[80:112].view(np.uint16)
            import ggkref
            w, a, tb = ggkref.inputs(3, 128)
            print("   got ", got[:16].tolist()); print("   want", want[:16].tolist())
    for seed, e0 in ((0, 120), (1, 124), (2, 116), (3, 128)):
        y = d.call([0, seed, e0])
        print(f"seed {seed} exp {e0}: ggk {y[0]} cycles, C++ {y[2]}; bad {y[1]}"
              + (f", first {y[3]}: {y[4] & 0xFFFF:#06x} vs {y[5] & 0xFFFF:#06x}" if y[1] else "") + f"; guard changed {y[6]}", flush=True)
        if y[1]:
            print("bad per 32-lane half (gate, np, c, h):", list(y[200:232]))
        import ggkref
        w, a, tb = ggkref.inputs(seed, e0)
        got, want = y[16:48].view(np.uint16), y[80:112].view(np.uint16)
        for v in ("nt", "nn"):
            ref = ggkref.block(w, a, tb, variant=v)
            print(f"  ref {v}: = ggk {int((ref == got).sum())}/64, = C++ {int((ref == want).sum())}/64")
        print("  ggk vs hypotheses:", ggkref.hyp(w, a, tb, got), ggkref.hyp2(w, a, tb, got))
