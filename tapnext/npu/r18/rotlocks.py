"""Round 18: npu-research r19/rotb.py with -DROTB_LOCKS (lock values aF / aU after the calls), built into build/r18/rotl.
usage: python3 r18/rotlocks.py build | run   (npu-iron; run needs the NPU)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "research" / "r19"))
import rotb  # noqa: E402

rotb.FLAGS.append("-DROTB_LOCKS")
rotb.BD = HERE.parent / "build" / "r18" / "rotl"


class O:
    v = "g2f"
    calls = "1,2,4"
    timeout_ms = 3000


if sys.argv[1] == "build":
    rotb.build(O)
else:
    for var in ("ks", "rot"):
        for n in (1, 2, 4):
            y = rotb.call(str(rotb.BD / f"rotb_g2f_{var}"), n, 3000)
            print(f"{var} {n} call(s): aF {int(y[39])} aU {int(y[40])} (raw lock registers 0x1F040 / 0x1F050)", flush=True)
