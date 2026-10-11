"""Rev 16: per-op (kind within a block) gap / span medians from step8 timelines, plus per-run sums.
usage: python3 tlops21.py tl1.json [tl2.json ...]"""
import json
import sys
from collections import defaultdict

import numpy as np

runs = [json.load(open(f))["ops"] for f in sys.argv[1:]]
g = defaultdict(list)
s = defaultdict(list)
order = []
for r in runs:
    last = 0.0
    for op in r:
        k = op["name"].split("_", 1)[1] if op["name"] != "final" else "final"
        if k not in order:
            order.append(k)
        if op["first"] is not None:
            g[k].append((op["first"] - last) * 1e3)
            if op["last"] is not None:
                s[k].append((op["last"] - op["first"]) * 1e3)
        if op["last"] is not None:
            last = op["last"]
print(f"{len(runs)} runs; us; mean / median / n")
print("op         gap_mean  gap_med  span_mean    n")
for k in order:
    a, b = np.array(g[k]), np.array(s[k])
    print(f"{k:10s} {a.mean():9.1f} {np.median(a):8.1f} {b.mean():10.1f} {len(a):4d}")
