"""Round 18 KS-P38 (research rev 13 E3): read the -DKSP38 records (mm8.cc acc_epilogue, magic KS38) after 3 frames of step8.
usage (npu-iron-torch, NPU): python3 r18/ksp38.py BUILD"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import pyxrt
    import torch
    from step8 import Step8, demo_queries, preprocess

    frames = np.load("/work/tapnet/data/clip30.npy")[:3]
    video = preprocess(frames)
    s = Step8(sys.argv[1], 12)
    q = demo_queries()
    s.reset()
    for t in range(3):
        s.step(video[t], torch.cat([q[..., :1] - t, q[..., 1:]], -1))
    s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    act = np.frombuffer(s.bo_a.map(), np.uint32).copy()
    hits = np.nonzero(act == 0x3833534B)[0]
    r = np.array([act[h + 1:h + 6] for h in hits if h + 6 < len(act)], np.int64)
    r = r[(r[:, 4] > 0) & (r[:, 4] < 1000)]
    per = r[:, :4] / r[:, 4:5]
    print(f"KSP38: {len(r)} tile records, {r[:, 4].sum()} k-steps; per k-step (median / p10 / p90 over tiles):")
    for i, nm in enumerate(["counter0", "counter1", "counter2", "timer"]):
        print(f"  {nm:9s} {np.median(per[:, i]):8.1f} {np.percentile(per[:, i], 10):8.1f} {np.percentile(per[:, i], 90):8.1f}")


if __name__ == "__main__":
    main()
