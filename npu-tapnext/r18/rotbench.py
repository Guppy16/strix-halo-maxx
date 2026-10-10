"""Round 18: npu-research r19/rotb.py (one-core ROT bench, bit-exact vs mm_ks) on a ROT variant of r18/ (mm_rot_<v>.s / .h).
usage: python3 r18/rotbench.py build|run --v g2a [--calls 1,4]   (npu-iron; run needs the NPU)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "npu-research" / "r19"))
import rotb  # noqa: E402

rotb.HERE = HERE
rotb.BD = HERE.parent / "build" / "r18" / "rotb"
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--v", default="g2a")
    ap.add_argument("--calls", default="1,4")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    o = ap.parse_args()
    dict(build=rotb.build, run=rotb.run)[o.cmd](o)
