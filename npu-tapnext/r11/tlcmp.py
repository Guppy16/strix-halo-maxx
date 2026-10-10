"""Round 11: compare two step8 timelines (build/tl_*.json) by op kind: summed wall per step (ms) and per block (us).
    python3 r11/tlcmp.py build/tl_A.json build/tl_B.json"""
import json
import re
import sys
from collections import defaultdict


def kinds(path):
    d = json.load(open(path))
    ops = d["ops"] if isinstance(d, dict) else d
    agg = defaultdict(float)
    prev = 0.0
    for o in sorted((o for o in ops if o["last"] is not None), key=lambda o: o["last"]):  # wall = last-to-last (ms), as RESULTS reports per block
        nm = re.sub(r"^b\d+_", "", o["name"])
        nm = re.sub(r"\d+$", "", nm)
        agg[nm] += (o["last"] - prev) * 1e3
        prev = o["last"]
    return agg, ops


a, oa = kinds(sys.argv[1])
b, ob = kinds(sys.argv[2])
print(f"{'op':14s} {'A ms':>8s} {'B ms':>8s} {'B-A us':>8s}")
for k in sorted(set(a) | set(b), key=lambda k: -a.get(k, 0)):
    print(f"{k:14s} {a.get(k, 0) / 1e3:8.2f} {b.get(k, 0) / 1e3:8.2f} {b.get(k, 0) - a.get(k, 0):8.0f}")
print(f"{'total':14s} {sum(a.values()) / 1e3:8.2f} {sum(b.values()) / 1e3:8.2f}")
