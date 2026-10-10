"""Rev 5: per-op wall time of the GEMM-only step from its PROF records: per column the slowest of the 3 cores (the A
broadcast couples them), records split into the 3 rows by drain offset ([row][tile] order). No device.

usage: python3 g11wall.py [ACT.npy SCHED]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profdec  # noqa: E402

GHZ = profdec.GHZ


def load(act_p, sch_p):
    act = np.load(act_p, mmap_mode="r").view(np.uint32).ravel()
    ops = json.loads(Path(sch_p).read_text())
    gem = [(op["name"], op, profdec.ranges(op)) for op in ops if op["kind"] == "gemm"]
    hits = np.nonzero(np.asarray(act) == profdec.MAGIC)[0]
    per = defaultdict(list)
    for h in hits:
        b = int(h) * 4
        own = None
        for name, op, rr in gem:
            for k, (ci, f, l) in enumerate(rr):
                if f <= b <= l:
                    own = (name, ci, b - f, k)
        if own is None:
            continue
        busy, calls, kwall, epi, gkb, ga, first, gmax, l2e, hand, zero = (int(v) for v in act[h + 1:h + 12])
        per[own[0]].append(dict(col=own[1], off=(own[3], own[2]), ks=calls // 9, busy=busy, kwall=kwall, epi=epi,
                                gkb=gkb, first=first, hand=hand, zero=zero))
    return gem, per


def walls(gem, per):
    out = {}
    for name, op, _ in gem:
        if name not in per:
            continue
        nt = op["nt"]
        cols = defaultdict(list)
        for r in per[name]:
            cols[r["col"]].append(r)
        ws, steady = [], []
        for c, rr in cols.items():
            rr.sort(key=lambda r: r["off"])
            if len(rr) != 3 * nt:
                continue
            tot = [sum(r["busy"] + r["gkb"] + r["first"] + r["epi"] + r["hand"] + r["zero"] for r in rr[i * nt:(i + 1) * nt])
                   for i in range(3)]
            ws.append(max(tot) / GHZ / 1e3)
            steady += [r["busy"] / r["ks"] for r in rr]
        if ws:
            out[name] = (float(np.median(ws)), float(np.percentile(steady, 10)), op)
    return out


def main():
    act_p = Path(sys.argv[1]) if len(sys.argv) > 1 else profdec.B / "g11sp.npy"
    sch_p = Path(sys.argv[2]) if len(sys.argv) > 2 else profdec.B / "g11-r9sp.sched"
    gem, per = load(act_p, sch_p)
    w = walls(gem, per)
    kinds = defaultdict(list)
    for name, (wall, st, op) in w.items():
        kinds[name.rstrip("0123456789")].append((wall, st, op))
    tot = 0.0
    print(f"{'op':9s} {'n':>3s} {'wall us':>8s} {'steady':>7s} {'steady-only us':>14s} {'rest us':>8s}")
    for k, v in kinds.items():
        wall = float(np.median([x[0] for x in v]))
        st = float(np.median([x[1] for x in v]))
        op = v[0][2]
        ksu = op["nt"] * op["K"] // 64 * (st + 67) / GHZ / 1e3
        tot += wall
        print(f"{k:9s} {len(v):3d} {wall:8.1f} {st:7.0f} {ksu:14.1f} {wall - ksu:8.1f}")
    print(f"block (ops with complete records) {tot:.1f} us")


if __name__ == "__main__":
    main()
