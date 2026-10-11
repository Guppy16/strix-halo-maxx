"""Rev 16: per-op gap distributions (timeline polls quantise at ~16.5 us), block 0 excluded; mean +- SE.
usage: python3 tlhist21.py tl1.json [...]"""
import json
import sys
from collections import defaultdict

import numpy as np

runs = [json.load(open(f)) for f in sys.argv[1:]]
g = defaultdict(list)
sp = defaultdict(list)
order = []
P = []
for d in runs:
    P.append(d["dispatch_ms"] / d["polls"] * 1e3)
    last = 0.0
    for op in d["ops"]:
        k = op["name"].split("_", 1)[1] if op["name"] != "final" else "final"
        b = op["name"].split("_")[0]
        if k not in order:
            order.append(k)
        if op["first"] is not None and b != "b0":
            g[k].append((op["first"] - last) * 1e3)
            if op["last"] is not None:
                sp[k].append((op["last"] - op["first"]) * 1e3)
        if op["last"] is not None:
            last = op["last"]
print(f"{len(runs)} runs; mean poll period {np.mean(P):.1f} us; blocks 1-11; gap histogram in units of 16.5 us")
print("op          gap_mean   se  span_mean  hist(0,1,2,3,4+)")
for k in order:
    a = np.array(g[k])
    if len(a) == 0:
        continue
    q = np.clip(np.round(a / 16.5).astype(int), 0, 4)
    h = np.bincount(q, minlength=5)
    print(f"{k:10s} {a.mean():9.1f} {a.std() / np.sqrt(len(a)):5.1f} {np.mean(sp[k]):9.1f}  {list(h)}")
