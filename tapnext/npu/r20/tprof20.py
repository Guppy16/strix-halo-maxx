"""Round 20: dump every TPROF record (mm8 -DTPROF, part-2 location under CSPLIT) with its owner op and the core's tile counter.

Fields per record (cycles): first (k = 0 .. the k == 1 stamp: B / A waits + 1 k-step), ks (the other k-steps), nks, mid, epi,
hand (previous epilogue end -> mm_ktile entry), t0 (core timer at mm_ktile entry), n (the core's tile counter).
usage (npu-iron-torch, NPU): python3 r20/tprof20.py BUILD OUT.npz
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r8"))
import gprof  # noqa: E402

MAGIC = 0x46525054


def main():
    import pyxrt
    import torch
    from step8 import Step8, demo_queries, preprocess

    frames = np.load("/work/tapnet/data/davis_reg.npy")[:3]
    video = preprocess(frames)
    s = Step8(sys.argv[1], 12)
    q = demo_queries()
    s.reset()
    for t in range(3):
        s.step(video[t], torch.cat([q[..., :1] - t, q[..., 1:]], -1))
    s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    act = np.frombuffer(s.bo_a.map(), np.uint32).copy()
    hits = np.nonzero(act == MAGIC)[0]
    rngs = [(i, op["name"], gprof.ranges(op)) for i, op in enumerate(s.p.ops) if op["kind"] == "gemm"]
    rows, names = [], []
    for h in hits:
        b = int(h) * 4
        owner = None
        for i, name, rr in rngs:
            for c, (f, l) in enumerate(rr):
                if f <= b <= l:
                    owner = (i, name, c)
        if owner is None or h + 12 >= len(act):
            continue
        rec = act[h + 1:h + 11].astype(np.int64)
        if not os.environ.get("NONKS") and not (0 < rec[2] <= 64):
            continue
        rows.append([owner[0], owner[2], b] + list(rec))
        names.append(owner[1])
    a = np.array(rows, np.int64)
    np.savez(sys.argv[2], a=a, names=np.array(names), opnames=np.array([op["name"] for op in s.p.ops]),
             opkinds=np.array([op["kind"] for op in s.p.ops]))
    print(f"TPROF20 {sys.argv[1]}: {len(hits)} magic, {len(a)} records -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
