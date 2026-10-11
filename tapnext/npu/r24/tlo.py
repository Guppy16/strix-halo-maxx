"""Round 24: means over blocks 1..11 and runs of a TL_ONLY timeline (only some op kinds): for consecutive watched ops, first(op) -
last(prev) and each op's wall, in us. usage: tlo.py RUN1.json [RUN2.json ..]"""
import json
import sys
from collections import defaultdict

import numpy as np

acc = defaultdict(list)
for f in sys.argv[1:]:
    ops = [o for o in json.load(open(f))["ops"] if o["first"] is not None and o["last"] is not None]
    for a, b in zip(ops, ops[1:]):
        if a["name"].split("_")[0] != b["name"].split("_")[0] or a["name"].startswith("b0_"):
            continue
        ka, kb = a["name"].split("_", 1)[1], b["name"].split("_", 1)[1]
        acc[(ka, "wall")].append(1e3 * (a["last"] - a["first"]))
        acc[(ka + " -> " + kb, "last->first")].append(1e3 * (b["first"] - a["last"]))
        acc[(ka + " -> " + kb, "last->last")].append(1e3 * (b["last"] - a["last"]))
        if a.get("pass1") is not None:
            acc[(ka, "pass1->last")].append(1e3 * (a["last"] - a["pass1"]))
for (k, what), v in acc.items():
    print(f"{k:22s} {what:12s} {np.mean(v):7.1f} us  (sd {np.std(v):4.1f}, n {len(v)})")
