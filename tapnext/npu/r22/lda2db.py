"""Round 22 KTLP-lite: does scalar lda.2d wrap as the 2D rule says (dn 4, dj 4, m -16: offsets 0, 4, 8, 12, 16, 0, ...)?
r dest (probe0), p dest (probe1, via mov), m dest (probe2, via mov). usage: python3 r22/lda2db.py build|run --out build/r22/lda2d"""
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
o = ap.parse_args()
if o.cmd == "build":
    o.asm, o.cc = str(HERE / "lda2d.s"), str(HERE / "lda2db.cc")
    apb.build(o)
    sys.exit()
a = np.array([(i * 7 + 1) % 256 for i in range(256)], np.uint8).view(np.int32)
want = [int(a[(4 * (i % 5)) // 4]) for i in range(15)]
d = apb.Dev(o.out)
for k in range(5):
    y = d.call([k])[:15]
    want = [int(a[(4 * (i % 5)) // 4]) for i in range(15)] if k < 3 else [int(a[i]) for i in range(15)]
    print(f"probe{k}: {'OK' if list(y) == want else 'BAD'} raw {[hex(int(v) & 0xffffffff) for v in y[:6]]} offsets {[int(np.where(a == v)[0][0]) * 4 if v in a else None for v in y]}", flush=True)
