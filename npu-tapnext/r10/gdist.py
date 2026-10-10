"""Round 10: per-tile distribution of the GEMM PROF records (r8/gprof.py's records): busy per k-step for each tile,
to separate op-start waits (inside the first k-step's A acquire) from the steady k-step rate.
usage: python3 r10/gdist.py NAME build/X.npy"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "r8"))
import gprof  # noqa: E402
from run8 import plan_of  # noqa: E402

name, path = sys.argv[1], sys.argv[2]
p, _ = plan_of(name)
act = np.load(path).view(np.uint32)
hits = np.nonzero(act == gprof.MAGIC)[0]
rngs = [(op["name"], gprof.ranges(op)) for op in p.ops if op["kind"] == "gemm"]
per = {}
for h in hits:
    b = int(h) * 4
    owner = None
    for nm, rr in rngs:
        if any(f <= b <= l for f, l in rr):
            owner = nm
    if owner is None:
        continue
    rec = [int(v) for v in act[h + 1:h + gprof.NREC]]
    if rec[1] == 0 or rec[1] > 10 ** 7:
        continue
    per.setdefault(owner, []).append(rec)
print(f"{'op':12s} {'n':>4s} | busy cycles per k-step: min p10 p50 p90 max | kwall/ks p50 | busy-sum / op (us)")
for nm, _ in rngs:
    if nm not in per:
        continue
    a = np.array(per[nm], float)
    ks = a[:, 1] / 9
    b = a[:, 0] / ks
    kw = a[:, 2] / ks
    q = np.percentile(b, [0, 10, 50, 90, 100])
    print(f"{nm:12s} {len(a):4d} | " + " ".join(f"{v:6.0f}" for v in q) + f" | {np.median(kw):6.0f} | {a[:, 0].sum() / 24 / 1810:8.1f}")
