"""Round 16: decode r16/popb.cc's dump: for each of 8 pops, which source bytes landed in the x part / e words."""
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
ap.add_argument("--probe", type=int, default=0)
o = ap.parse_args()
if o.cmd == "build":
    o.asm, o.cc = o.s, str(HERE / "popb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    y = d.call([o.probe]).view(np.uint8)[:576]
    src = (np.arange(3072) % 251).astype(np.uint8)
    for k in range(8 if o.probe < 4 or o.probe >= 6 else 5):
        x, e = y[64 * k:64 * k + 64], y[512 + 8 * k:512 + 8 * k + 8]
        off = 640 + 72 * (8 + k) if o.probe >= 6 else 72 * k if o.probe < 4 else (640 + 72 * (16 + 8 + k) if o.probe == 4 else 640 + 1152 + 72 * (8 + k))  # probe 4: B pops 8-12 of the second period
        want = src[off:off + 72].reshape(8, 9)
        ok_e = sorted(e.tolist()) == sorted(want[:, 0].tolist())
        ok_x = sorted(x.tolist()) == sorted(want[:, 1:].reshape(-1).tolist())
        print(f"pop {k}: e {e.tolist()} (want {want[:, 0].tolist()}) {'OK' if ok_e else 'BAD'}; x bytes {'OK' if ok_x else 'BAD'} "
              f"first x {x[:9].tolist()}")
