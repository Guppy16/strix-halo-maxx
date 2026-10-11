"""Round 24: per-op wall / start-gap means over blocks 1..depth-1 of step8 timeline JSONs, averaged over runs.
usage: tlcmp.py A_run1.json A_run2.json ... [-- B_run1.json ...]   (one column per build; "-" separates builds)
gap = first word - the previous op's last word (negative: the op started before its predecessor ended);
p1 = a split op's pass-1 end before its last word (round 24 step8 marks)."""
import json
import sys
from collections import defaultdict

import numpy as np


def load(files):
    acc, disp = defaultdict(list), []
    for f in files:
        d = json.load(open(f))
        disp.append(d["dispatch_ms"])
        prev = None
        for o in d["ops"]:
            b, k = o["name"].split("_")[0], o["name"].split("_", 1)[-1]
            if o["first"] is None or o["last"] is None:
                prev = o["last"] if o["last"] is not None else prev
                continue
            if b not in ("b0", "final") and prev is not None:
                p1 = 1e3 * (o["last"] - o["pass1"]) if o.get("pass1") is not None else np.nan
                acc[k].append((1e3 * (o["last"] - o["first"]), 1e3 * (o["first"] - prev), p1))
            prev = o["last"]
    return np.mean(disp), {k: np.nanmean(np.array(v, float), 0) if len(v) else None for k, v in acc.items()}


args = sys.argv[1:]
sets, cur = [], []
for a in args:
    if a == "-":
        sets.append(cur)
        cur = []
    else:
        cur.append(a)
sets.append(cur)
tabs = [load(s) for s in sets if s]
print("dispatch", [round(t[0], 2) for t in tabs], "(us below)")
tot = [[0.0, 0.0] for _ in tabs]
for k in tabs[0][1]:
    line = f"{k:10s}"
    for j, t in enumerate(tabs):
        v = t[1].get(k)
        if v is None:
            line += "  " + " " * 34
            continue
        tot[j][0] += v[0]
        tot[j][1] += v[1]
        p1 = f" p1 -{v[2]:5.1f}" if not np.isnan(v[2]) else " " * 9
        line += f"  wall {v[0]:6.1f} gap {v[1]:6.1f}{p1}"
    print(line)
print("sum/block " + "".join(f"  wall {a:6.1f} gap {b:6.1f}" + " " * 9 for a, b in tot))
