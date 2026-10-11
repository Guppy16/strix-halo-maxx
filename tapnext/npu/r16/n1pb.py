"""Round 16: one-core bench of n1p (r16/gen_n1p.py) vs norm_rows_k's pass-1 loop (r16/n1pb.cc), unit offsets that wrap.
    python3 r16/n1pb.py build --s build/r16/n1p.s --out build/r16/n1pb ; python3 r16/n1pb.py run --out build/r16/n1pb"""
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
    o.asm, o.cc = o.s, str(HERE / "n1pb.cc")
    o.data_size = 24 * 1024  # the 15 KB ring + stage + reference
    apb.build(o)
else:
    d = apb.Dev(o.out)
    IN_B = 768 * 20
    tot = 0
    for n_in in (1, 2, 3):
        for off in (0, 128, IN_B - 1024, IN_B - 896, IN_B - 128, IN_B - 2048 - 384, IN_B - 3072, 5 * 768):
            y = d.call([0, off // 128, n_in, off])
            tot += int(y[1])
            print(f"n_in {n_in} off {off:5d}: n1p {y[0]} cycles (call alone {y[7]}), C++ {y[2]}; bad {y[1]}"
                  + (f", first {y[3]}: {y[4] & 0xFFFF:#06x} vs {y[5] & 0xFFFF:#06x}" if y[1] else "") + f"; guard {y[6]}", flush=True)
    print("TOTAL bad", tot)
