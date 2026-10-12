"""Round 20 debug: one frame on two builds at depth d, then where their ACT images differ (by plan region, by column / k-step).
usage (npu-iron-torch, NPU): python3 r20/actdiff.py BUILD_A BUILD_B DEPTH"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step8 import Step8, demo_queries, preprocess  # noqa: E402

frames = np.load("/work/tapnet/data/davis_reg.npy")[:1]
video = preprocess(frames)
acts = []
for b in sys.argv[1:3]:
    s = Step8(b, int(sys.argv[3]))
    s.reset()
    s.step(video[0], demo_queries())
    acts.append(s.read_all().copy())
    regions = s.info["regions"]
    del s
a, b = acts
d = np.nonzero(a != b)[0] * 4
print("differing words:", len(d))
for name, r in sorted(regions.items(), key=lambda kv: kv[1]["base"]):
    lo = r["base"]
    size = r.get("col", 0) * 8 or r.get("bytes", 0)
    k = d[(d >= lo) & (d < lo + max(size, 1))]
    if len(k):
        col = (k - lo) // r["col"] if r.get("col") else k * 0
        print(f"{name:6s} kind {r['kind']:5s} differing bytes {4 * len(k):9d}  columns {np.bincount(col, minlength=8)[:8]}")
        if r["kind"] == "apan":
            q = ((k - lo) % r["col"]) // r["kstride"]
            print("        per k-step chunk:", np.bincount(q, minlength=r["nq"]))
        if r["kind"] == "q16":
            q = ((k - lo) % r["col"]) // 28800
            print("        per k-step:", np.bincount(q, minlength=12)[:16])
