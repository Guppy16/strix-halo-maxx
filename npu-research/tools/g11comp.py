"""Rev 5: compose the GEMM-only step (g11) from its PROF records and split it into steady k-steps, first-tile (A slot)
waits, row coupling, epilogues and tile handoffs. Checks the composition against the measured g11 time. No device.

usage: python3 g11comp.py [ACT.npy SCHED]   (default: g11sp.npy + g11-r9sp.sched; measured g11-r9s 29.47 ms)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profdec  # noqa: E402

GHZ = profdec.GHZ
act_p = Path(sys.argv[1]) if len(sys.argv) > 1 else profdec.B / "g11sp.npy"
sch_p = Path(sys.argv[2]) if len(sys.argv) > 2 else profdec.B / "g11-r9sp.sched"
act = np.load(act_p, mmap_mode="r").view(np.uint32).ravel()
ops = json.loads(sch_p.read_text())
gem = [(op["name"], op, profdec.ranges(op)) for op in ops if op["kind"] == "gemm"]
hits = np.nonzero(np.asarray(act) == profdec.MAGIC)[0]
per = defaultdict(list)
for h in hits:
    b = int(h) * 4
    own = None
    for name, op, rr in gem:
        for ci, f, l in rr:
            if f <= b <= l:
                own = (name, ci, b - f)
    if own is None:
        continue
    busy, calls, kwall, epi, gkb, ga, first, gmax, l2e, hand, zero = (int(v) for v in act[h + 1:h + 12])
    per[own[0]].append(dict(col=own[1], off=own[2], ks=calls // 9, busy=busy, kwall=kwall, epi=epi, gkb=gkb,
                            first=first, l2e=l2e, hand=hand, zero=zero))

rows = []
tot = defaultdict(float)
print(f"{'op':9s} {'nt':>3s} {'ks':>3s} | per core (us): {'steady ks':>9s} {'B gaps':>7s} {'1st wait':>8s} {'coupling':>8s}"
      f" {'epilogue':>8s} {'handoff':>7s} {'zero':>5s} {'total':>7s} | steady cyc/ks")
for kind in dict.fromkeys(n.rstrip("0123456789") for n, _, _ in gem):
    inst = [n for n, _, _ in gem if n.rstrip("0123456789") == kind and n in per]
    rs = [r for n in inst for r in per[n]]
    op = next(o for n, o, _ in gem if n == inst[0])
    nt = op["nt"]
    ks = rs[0]["ks"]
    cyc = np.array([r["busy"] / r["ks"] for r in rs])
    steady = float(np.percentile(cyc, 10))  # uncoupled, not-first tiles (yx / qkv: most tiles are coupled)
    # first tile of each core = records with the op's largest cyc/ks per (instance, col, row): take per core the max
    # excess; coupling = the rest of the excess above steady on the other tiles
    by_core = defaultdict(list)
    for n in inst:
        recs = sorted(per[n], key=lambda r: (r["col"], r["off"]))
        for r in recs:
            by_core[(n, r["col"])].append(r)
    first_ex, coup_ex = [], []
    for key, rr in by_core.items():
        ex = sorted(((r["busy"] - steady * r["ks"]) for r in rr), reverse=True)
        rows_g = 3
        first_ex.append(sum(ex[:rows_g]) / rows_g)  # the 3 cores' first tiles
        coup_ex.append(sum(max(e, 0) for e in ex[rows_g:]) / rows_g)
    m = {k: np.median([r[k] for r in rs]) for k in ("gkb", "epi", "hand", "zero")}
    parts = dict(steady=nt * ks * steady, gaps=nt * m["gkb"], first=np.median(first_ex), coup=np.median(coup_ex),
                 epi=nt * m["epi"], hand=nt * m["hand"], zero=nt * m["zero"])
    parts = {k: v / GHZ / 1e3 for k, v in parts.items()}
    t = sum(parts.values())
    for k, v in parts.items():
        tot[k] += v
    print(f"{kind:9s} {nt:3d} {ks:3d} | {parts['steady']:9.1f} {parts['gaps']:7.1f} {parts['first']:8.1f} {parts['coup']:8.1f}"
          f" {parts['epi']:8.1f} {parts['hand']:7.1f} {parts['zero']:5.1f} {t:7.1f} | {steady:.0f}")
blk = sum(tot.values())
print(f"{'block':9s}         | " + " ".join(f"{tot[k]:8.1f}" for k in ("steady", "gaps", "first", "coup", "epi", "hand", "zero"))
      + f" {blk:7.1f} us")
print(f"x12 = {12 * blk / 1e3:.2f} ms (g11-r9s measured 29.47 ms; the difference is launch + tail + C drains)")
print("shares: " + ", ".join(f"{k} {100 * v / blk:.1f}%" for k, v in tot.items()))
