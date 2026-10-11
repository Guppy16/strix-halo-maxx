"""Pass 20: per-op share of the step's wall from step8 timelines.
advance(op) = how far op's last poll moves the frontier (max of every earlier op's last); the advances sum to the wall.
Timelines poll at ~19 us, so single values are coarse; averages over 12 blocks x runs are not.
usage: python3 advance.py tl1.json [tl2.json ...]"""

import json
import sys
from collections import defaultdict

import numpy as np

adv = defaultdict(list)
order = []
walls = []
for f in sys.argv[1:]:
    ops = json.load(open(f))["ops"]
    front = 0.0
    for op in ops:
        k = op["name"].split("_", 1)[1] if op["name"] != "final" else "final"
        if k not in order:
            order.append(k)
        if op["last"] is None:
            adv[k].append(0.0)
            continue
        a = max(0.0, op["last"] - front)
        adv[k].append(a * 1e3)
        front = max(front, op["last"])
    walls.append(front)
print(f"{len(walls)} runs, wall {np.mean(walls):.3f} ms (frontier)")
tot = 0.0
for k in order:
    a = np.array(adv[k])
    per_step = a.sum() / len(walls) / 1e3
    tot += per_step
    print(f"{k:10s} mean/blk {a.mean():7.1f} us   per step {per_step:6.3f} ms")
print(f"sum {tot:.3f}")
