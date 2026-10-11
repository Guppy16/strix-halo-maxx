"""pass 19 MULLAT bench driver (main's r22/lda2db.py pattern on r11/apb.py).
usage (npu-iron): python3 r24/mullatb.py build|run --out /work/npu-research/r24/build/mullat"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "npu" / "r11"))
import apb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "run"])
ap.add_argument("--out", required=True)
o = ap.parse_args()
if o.cmd == "build":
    o.asm, o.cc = str(HERE / "mullat.s"), str(HERE / "mullatb.cc")
    apb.build(o)
    sys.exit()
names = ["ZOL mul d1", "ZOL mul d2", "ZOL mul d3", "straight mul d1", "straight mul d2", "straight mul d3", "ZOL add d1", "ZOL add d2"]
d = apb.Dev(o.out)
for k in range(8):
    y = [int(v) for v in d.call([k])[:3]]
    print(f"probe{k} {names[k]:16s}: {y}  (old 1, new {35 if 'mul' in names[k] else 6})", flush=True)
