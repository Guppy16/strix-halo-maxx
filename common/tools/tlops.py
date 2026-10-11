"""Per-op wall (us, blocks 1-11 mean / median / min / max) across timelines (npu-research rev 8).

wall(op) = last(op) - last(previous op) on the host-polled timeline (step8.py timeline), as sim13.walls.
usage: python3 tlops.py TAG1 TAG2 ...   (TAG = build suffix, e.g. r11b r12e -> build/tl_full12TAG_d12.json)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parents[2] / "tapnext" / "npu" / "build"


def walls(tag):
    d = json.loads((B / f"tl_full12{tag}_d12.json").read_text())
    w, prev = defaultdict(list), None
    for o in d["ops"]:
        if o["last"] is None:
            continue
        blk, _, kind = o["name"].partition("_")
        if prev is not None and blk != "b0":
            w[kind].append((o["last"] - prev) * 1e3)
        prev = o["last"]
    return w, d["dispatch_ms"], d["polls"]


tags = sys.argv[1:]
data = {t: walls(t) for t in tags}
order = list(data[tags[0]][0])
print("op        " + "".join(f"{t:>16s}" for t in tags) + "   (mean / median us per block, blocks 1-11)")
for k in order:
    print(f"{k:9s} " + "".join(f"{np.mean(data[t][0][k]):8.1f}/{np.median(data[t][0][k]):6.1f} " for t in tags))
print(f"{'block':9s} " + "".join(f"{sum(np.mean(v) for v in data[t][0].values()):15.0f} " for t in tags))
print(f"{'dispatch':9s} " + "".join(f"{data[t][1]:15.2f} " for t in tags))
print(f"{'polls':9s} " + "".join(f"{data[t][2]:15d} " for t in tags))
for k in sys.argv and ["mlp3_a", "out_proj", "attn", "qkv", "mlp0", "norm3"]:
    print(f"\n{k} per block: " + " | ".join(f"{t}: " + " ".join(f"{x:.0f}" for x in data[t][0][k]) for t in tags))
