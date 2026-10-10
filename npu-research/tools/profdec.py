"""Rev 5: decode the GEMM-core PROF records of a saved ACT image (npu-tapnext/build/g11*p.npy) without the device.

Same mapping as npu-tapnext/r7/prof.py (records found by magic, owner = the last op in plan order whose column
drains cover the record), but the plan comes from the build's .sched JSON, so no pyxrt / Step8 is needed.
Prints per op kind: k-step cycles (busy / k-steps, incl. the A waits inside mm_ks), the gap between k-steps
(B acquire + IRON code), the first-k-step wait, and the per-column / per-row spread of the k-step.

usage: python3 profdec.py [ACT.npy SCHED] (default: g11sp.npy + g11-r9sp.sched)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parents[2] / "npu-tapnext" / "build"
MAGIC = 0x464F5250
GHZ = 1.81
RHO = 9


def tap_ends(t):
    last = t["off"] + sum((n - 1) * st for n, st in zip(t["sizes"][:3], t["strides"][:3])) + t["sizes"][3] - 4
    return t["off"], last


def ranges(op):
    out = []
    for ci, c in enumerate(op["c"]):
        for t in c if isinstance(c, list) else [c]:
            out.append((ci, *tap_ends(t)))
    return out


def main():
    act_p = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "g11sp.npy"
    sch_p = Path(sys.argv[2]) if len(sys.argv) > 2 else B / "g11-r9sp.sched"
    act = np.load(act_p, mmap_mode="r").view(np.uint32).ravel()
    ops = json.loads(sch_p.read_text())
    gem = [(op["name"], op, ranges(op)) for op in ops if op["kind"] == "gemm"]
    hits = np.nonzero(np.asarray(act) == MAGIC)[0]
    per = defaultdict(list)
    for h in hits:
        b = int(h) * 4
        owner = None
        for name, op, rr in gem:
            for ci, f, l in rr:
                if f <= b <= l:
                    owner = (name, op, ci, b - f)
        if owner is None:
            continue
        w = [int(v) for v in act[h + 1:h + 12]]
        busy, calls, kwall, epi, gkb, ga, first, gmax, l2e, hand, zero = w
        if calls == 0 or calls > 100000:
            continue
        per[owner[0]].append(dict(col=owner[2], off=owner[3], busy=busy, calls=calls, kwall=kwall, epi=epi, gkb=gkb,
                                  first=first, gmax=gmax, hand=hand, zero=zero))
    print(f"{act_p.name}: {len(hits)} magic words, {sum(map(len, per.values()))} records, {len(per)} ops")
    kinds = defaultdict(list)
    meta = {}
    for name, op, _ in gem:
        if name in per:
            kind = name.rstrip("0123456789")
            kinds[kind] += per[name]
            meta[kind] = op
    print(f"{'op':9s} {'K':>5s} {'nt':>3s} {'recs':>5s} {'ks/tile':>7s} {'cyc/ks':>7s} {'p10':>6s} {'p90':>6s} {'gap/ks':>7s}"
          f" {'first us':>8s} {'B/cyc A':>7s} | per column median cyc/ks")
    for kind, rs in kinds.items():
        op = meta[kind]
        ks = np.array([r["calls"] / RHO for r in rs])
        cyc = np.array([r["busy"] for r in rs]) / ks
        gap = np.array([r["gkb"] for r in rs]) / np.maximum(ks - 1, 1)
        first = np.array([r["first"] for r in rs]) / GHZ / 1e3
        cols = defaultdict(list)
        for r, c in zip(rs, cyc):
            cols[r["col"]].append(c)
        colmed = " ".join(f"{np.median(cols[c]):5.0f}" for c in sorted(cols))
        print(f"{kind:9s} {op['K']:5d} {op['nt']:3d} {len(rs):5d} {np.median(ks):7.0f} {np.median(cyc):7.0f} "
              f"{np.percentile(cyc, 10):6.0f} {np.percentile(cyc, 90):6.0f} {np.median(gap):7.0f} {np.median(first):8.2f}"
              f" {10368 / np.median(cyc + gap):7.2f} | {colmed}")


if __name__ == "__main__":
    main()
