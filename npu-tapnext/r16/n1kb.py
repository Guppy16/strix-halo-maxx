"""Round 16: one-core bench of n1k_ln / n1k_rms (r16/gen_n1k.py) vs norm_rows_k's pass 1 + statistics (r16/n1kb.cc).
    python3 r16/n1kb.py build --s build/r16/n1k.s --out build/r16/n1kb ; python3 r16/n1kb.py run --out build/r16/n1kb"""
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
    o.asm, o.cc = o.s, str(HERE / "n1kb.cc")
    o.data_size = 44 * 1024
    apb.build(o)
else:
    d = apb.Dev(o.out)
    IN_B = 768 * 20
    tot = 0
    for ln in (1, 0):
        for n_in in (1, 2, 3):
            for off in (0, IN_B - 896, IN_B - 128, IN_B - 2048 - 384, 5 * 768):
                y = d.call([0, off // 128 + 7 * n_in, n_in, off, ln])
                tot += int(y[1]) + int(y[3])
                print(f"{'ln ' if ln else 'rms'} n_in {n_in} off {off:5d}: asm {y[0]} cycles, C++ {y[2]} (12 units); bad stage {y[1]}, acc {y[3]}"
                      + (f", first {y[4]}: {y[5] & 0xFFFFFFFF:#x} vs {y[6] & 0xFFFFFFFF:#x}" if y[1] or y[3] else "") + f"; guard {y[7]}", flush=True)
    print("TOTAL bad", tot)
