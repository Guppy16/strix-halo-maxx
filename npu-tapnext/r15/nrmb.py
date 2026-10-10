"""Round 15: one-core bench of nrm_rms / nrm_ln (r15/nrmb.cc).  build --s A.s,B.s --out ... | run --out ..."""
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
    parts = o.s.split(",")
    merged = Path(o.out + "_in.s")
    merged.write_text("".join(Path(p).read_text() for p in parts))
    o.asm, o.cc = str(merged), str(HERE / "nrmb.cc")
    apb.build(o)
else:
    d = apb.Dev(o.out)
    for seed, ln, q in ((0, 0, 0), (1, 0, 5), (2, 1, 0), (3, 1, 11), (4, 0, 11), (5, 1, 3)):
        y = d.call([0, seed, ln, q])
        print(f"seed {seed} {'ln ' if ln else 'rms'} q {q:2d}: asm {y[0]} cycles, C++ {y[3]}; bad panel bytes {y[1]}, bad z {y[2]}"
              + (f", first {y[4]}" if y[1] or y[2] else "") + f"; bytes written past 576: {y[5]}; bad X' {y[6]}")
