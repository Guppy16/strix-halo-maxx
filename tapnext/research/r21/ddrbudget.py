"""Rev 16: DDR traffic per op per step (bytes read / written, from plan8's TAPs) against the op's measured share of the step
(timeline, blocks 1-11 mean) -> average read / write GB/s per op, and the phases that sit at the aggregate read cap.

  GEMM: A fills (shim MM2S 0, per column; none for reuse_a), B fills (weights, shims 0-2 MM2S 1: 3 rows x nt x (K/64+1) x 4,608 B),
        C drains (shim S2MM 0: the whole drain TAPs, junk tails of APAN objects included).
  attention: Q fills (A), K / V fills (B), O drains.   EW: params + input fills, drains.
usage: python3 ddrbudget.py BUILD_JSON tl1.json [tl2.json ...]   (host python; plan8 from npu-tapnext)"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.expanduser("~/git/app/.scratch/npu-tapnext"))
bj = json.load(open(sys.argv[1]))
os.environ.update(A_HALF="1" if bj.get("a_half") else "0", P23="1" if bj.get("p23") else "0", P60G="1" if bj.get("p60g") else "0",
                  XROT=str(bj.get("xrot", 0)), AL1="1" if bj.get("al1") else "0")
import plan8  # noqa: E402

p, _ = plan8.full_step(12)
CAP = (55.0, 59.0)  # GB/s aggregate read cap (DDRCAP, round 19)


def tb(t):
    if isinstance(t, list):
        return sum(tb(x) for x in t)
    if isinstance(t, tuple):
        return sum(tb(x) for x in t)
    s = t["sizes"]
    return s[0] * s[1] * s[2] * s[3]


share = defaultdict(list)
for f in sys.argv[2:]:
    last = 0.0
    for op in json.load(open(f))["ops"]:
        if op["last"] is not None:
            if op["name"].split("_")[0] != "b0":
                k = op["name"].split("_", 1)[1] if op["name"] != "final" else "final"
                share[k].append((op["last"] - last) * 1e3)
            last = op["last"]
rows = {}
for op in p.ops:
    k = op["name"].split("_", 1)[1] if op["name"] != "final" else "final"
    if op["name"].split("_")[0] not in ("b1",) and k != "final":
        continue
    if op["kind"] == "gemm":
        a = 0 if op.get("reuse_a") else sum(tb(x) for x in op["a"])
        b = 3 * op["nt"] * (op["K"] // 64 + 1) * plan8.OBJ
        w = sum(tb(x) for x in op["c"])
        rows[k] = (a, b, w, f"K{op['K']} nt{op['nt']} {op['epi']}")
    elif op["kind"] == "attn":
        a = sum(tb(x) for x in op["a"])
        b = len(op["passes"]) * sum(tb(t) for r in op["b"] for t in r)
        w = sum(tb(x) for pc in op["c"] for x in pc)
        rows[k] = (a, b, w, "attention")
    else:
        r = sum(tb(x) for c in op["fills"] for x in c)
        w = sum(tb(x) for x in op["c"])
        rows[k] = (r, 0, w, op["fn"] + (f" {op['mode']} n_in{op['n_in']}" if op["fn"] == "norm" else ""))
print("per block (block 1 ops); MB = 1e6 B; share = timeline us (blocks 1-11 mean); GB/s = MB / share")
print(f"{'op':9s} {'what':22s} {'A/in MB':>8s} {'B MB':>6s} {'out MB':>7s} {'share':>6s} {'rd GB/s':>8s} {'wr GB/s':>8s} {'r+w':>6s}")
tr = tw = tt = 0.0
for k, (a, b, w, what) in rows.items():
    s = np.mean(share[k]) if share[k] else float("nan")
    rd, wr = (a + b) / 1e6, w / 1e6
    tr += rd
    tw += wr
    tt += s if s == s else 0
    print(f"{k:9s} {what:22s} {a / 1e6:8.3f} {b / 1e6:6.3f} {wr:7.3f} {s:6.1f} {rd / s * 1e3:8.1f} {wr / s * 1e3:8.1f} {(rd + wr) / s * 1e3:6.1f}")
print(f"block: read {tr:.2f} MB, write {tw:.2f} MB in {tt:.0f} us -> mean {tr / tt * 1e3:.1f} GB/s read, {tw / tt * 1e3:.1f} write")
print(f"step (x12): read {12 * tr:.1f} MB, write {12 * tw:.1f} MB; floor at the read cap {CAP}: "
      f"{12 * tr / CAP[1]:.2f}..{12 * tr / CAP[0]:.2f} ms")
