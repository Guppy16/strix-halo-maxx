"""Round 21 E7 debug: block 0's yx output (depth 1, frame 0) of two builds, diffed by output column (cb = col % 64 // 8, tile =
col // 64) and by row block. usage (npu-iron-torch, NPU): python3 r21/e7dbg.py BUILD_REF BUILD_TEST"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pack8 as P  # noqa: E402
from step8 import COLS, KSTEP16, Step8, demo_queries, preprocess  # noqa: E402

frames = np.load("/work/tapnet/data/clip30.npy")[:1]
video = preprocess(frames)
q = demo_queries()
outs = []
for b in sys.argv[1:3]:
    s = Step8(b, 1)
    s.reset()
    s.step(video[0], torch.cat([q[..., :1], q[..., 1:]], -1))
    act = s.read_all()
    reg = s.info["areas"][0]["yx"]
    w = np.concatenate([act[(reg["base"] + c * reg["col"]) // 4:][: 24 * KSTEP16 // 4] for c in range(COLS)])
    outs.append(np.asarray(P.from_q16(w, ncols=24 * 64), np.float32)[:1072])
    del s
a, b = outs
d = np.abs(a - b)
print("differing values", (d > 0).mean())
cols = (d > 0).mean(0)
print("by tile (64 cols):", np.round([cols[64 * t:64 * t + 64].mean() for t in range(24)], 3))
print("by cb (8 cols within a tile):", np.round([cols.reshape(24, 8, 8)[:, cb].mean() for cb in range(8)], 3))
rows = (d > 0).mean(1)
print("by row block (8 rows) of the first 144:", np.round([rows[8 * r:8 * r + 8].mean() for r in range(18)], 3))
print("by 144-row column group:", np.round([rows[144 * c:144 * c + 144].mean() for c in range(8)], 3))
