"""Rev 5: where the slow k-steps sit (per op instance, tile order by drain offset, mean vs median). No device."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profdec  # noqa: E402

act_p = Path(sys.argv[1]) if len(sys.argv) > 1 else profdec.B / "g11sp.npy"
sch_p = Path(sys.argv[2]) if len(sys.argv) > 2 else profdec.B / "g11-r9sp.sched"
import json  # noqa: E402

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
    busy, calls, kwall, epi, gkb, ga, first, gmax = (int(v) for v in act[h + 1:h + 9])
    per[own[0]].append((own[1], own[2], busy / (calls / 9), (kwall - busy) / (calls / 9), first, epi))
tot_mean, tot_med = defaultdict(list), defaultdict(list)
for name in [g[0] for g in gem]:
    if name not in per:
        continue
    rs = per[name]
    kind = name.rstrip("0123456789")
    cyc = np.array([r[2] for r in rs])
    tot_mean[kind].append(cyc.mean())
    tot_med[kind].append(np.median(cyc))
print("op kind: mean of per-instance mean cyc/ks | median | instances")
for k in tot_mean:
    print(f"  {k:9s} mean {np.mean(tot_mean[k]):7.0f}  median {np.median(tot_med[k]):7.0f}  n {len(tot_mean[k])}"
          f"  per-instance means: " + " ".join(f"{v:.0f}" for v in tot_mean[k]))
for name in sys.argv[3:] or ["down_a5", "out5", "yx5", "qkv5", "up_a5"]:
    rs = sorted([r for r in per.get(name, []) if r[0] == 0], key=lambda r: r[1])
    print(f"\n{name} column 0, records in drain-offset order: cyc/ks  (k-wall - busy)/ks  first-gap  epi")
    for r in rs:
        print(f"   off {r[1]:8d}  {r[2]:7.0f}  {r[3]:6.0f}  {r[4]:7d}  {r[5]:7d}")
