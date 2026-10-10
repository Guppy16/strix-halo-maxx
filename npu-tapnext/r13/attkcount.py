"""Round 13: read the fused K loop's slow-unit counters (a -DATTK_COUNT build writes 'ATTC', slow, units over the first words
of every head group's attention output): run N frames, then scan the O region for the records. Per core the last record is
cumulative over the run.
    python3 r13/attkcount.py --build full12r13cc [--frames 3]      (npu-iron-torch, NPU)
"""
import argparse
import numpy as np
import torch
import sys
sys.path.insert(0, "/work/npu-tapnext")
import step8 as S8  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--frames", type=int, default=3)
o = ap.parse_args()
frames = np.load("/work/tapnet/data/clip30.npy")[: o.frames]
video = S8.preprocess(frames)
s = S8.Step8(o.build, 12)
q = S8.demo_queries()
s.reset()
for it in range(o.frames):
    rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
    s.step(video[it], rel)
act = s.read_all()
reg = s.info["regions"]["O"] if "O" in s.info["regions"] else None
print("regions:", list(s.info["regions"].keys())[:12])
base, size = reg["base"] // 4, 8 * reg["col"] // 4
w = act[base:base + size]
idx = np.nonzero(w == 0x43545441)[0]
print(f"{len(idx)} records in O; per record (slow, units, %):")
recs = sorted(int(w[i + 1]) for i in idx if i + 1 < len(w))
units = 4 * 3 * 6 * 23 * 12 * o.frames  # per core, cumulative
print("  slow counts (ascending):", recs)
print(f"  units per core {units}: slow {100 * min(recs) / units:.1f} .. {100 * max(recs) / units:.1f} % (records seen: the last head group of each core drained)")
