"""Round 19: build / run the B-wrap probes (r19/wrapgen.py, poplf0b.cc) on one core; per probe: which of the 128 pops are right."""
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
ap.add_argument("--probes", default="0,1,2,3,4")
o = ap.parse_args()
if o.cmd == "build":
    o.asm, o.cc = str(HERE / "poplf0.s"), str(HERE / "poplf0b.cc")
    apb.build(o)
    sys.exit()
src = (np.arange(4608) % 251).astype(np.int64)
want = []
for j in range(64):
    p = src[72 * j:72 * j + 72].reshape(8, 9)
    x, e = p[:, 1:].reshape(-1), p[:, 0]
    want.append((x.sum(), (x * x).sum(), e.sum(), (e * e).sum()))
d = apb.Dev(o.out)
for k in (int(c) for c in o.probes.split(",")):
    y = d.call([k])
    got = y[:512].reshape(128, 4)
    bad = [i for i in range(128) if tuple(got[i]) != tuple(want[i % 64])]
    print(f"probe{k}: {128 - len(bad)}/128 pops right; guards {y[512] & 0xFFFF:#x}; bad pops {bad[:24]}", flush=True)
