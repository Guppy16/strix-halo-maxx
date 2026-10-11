"""Per-op wall / start-gap means (blocks 1..depth-1) of step8 timeline JSONs; usage: tlcmp.py A.json [B.json]"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(f):
    d = json.load(open(f))
    acc = defaultdict(list)
    prev = None
    for o in d["ops"]:
        b, k = o["name"].split("_")[0], o["name"].split("_", 1)[-1]
        if b not in ("b0", "final"):
            acc[k].append((1e3 * (o["last"] - o["first"]), 1e3 * (o["first"] - prev) if prev is not None else 0, 1e3 * o["first"]))
        prev = o["last"]
    return d["dispatch_ms"], {k: np.array(v).mean(0) for k, v in acc.items()}


tabs = [load(f) for f in sys.argv[1:]]
print("dispatch", [round(t[0], 2) for t in tabs])
for k in tabs[0][1]:
    print(f"{k:10s}" + "".join(f"  wall {t[1][k][0]:7.1f} gap {t[1][k][1]:7.1f}" for t in tabs))
