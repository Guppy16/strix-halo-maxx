"""Round 20: attention boundary stamps (mm8 -DATPROF): per head group record in the O object (magic 'ATPF'): last GEMM epilogue
end on this core, first Q copy, attn_begin, first K chunk, attn_end entry, attn_end exit, ntiles. Block 11, 3 frames.
usage (npu-iron-torch, NPU): python3 r20/atprof.py BUILD"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
MAGIC = 0x46505441


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
    hits = np.nonzero(act == MAGIC)[0]
    r = np.array([act[h + 1:h + 8] for h in hits], np.int64)
    print(f"{len(hits)} records")
    G = 1810.0
    tle, tq, tb, tk, te, tx, nt = r.T
    print("per record (us): qcopy-wait = first qcopy - last epilogue end; begin-after-qcopy; first K - begin; K/V phase; end")
    order = np.argsort(tb)
    for name, v in (("first qcopy - last epi end", tq - tle), ("begin - first qcopy", tb - tq), ("first K - begin", tk - tb),
                    ("end entry - first K", te - tk), ("attn_end", tx - te)):
        print(f"  {name:28s} median {np.median(v) / G:8.2f}  min {v.min() / G:8.2f}  max {v.max() / G:8.2f}")
    # head groups in time order (cluster by begin time)
    tbs = np.sort(tb)
    groups = np.split(tbs, np.nonzero(np.diff(tbs) > 5000)[0] + 1)
    print("head-group begin times relative to the first (us):", [round((g.min() - tbs[0]) / G, 1) for g in groups])
    first = tb < tbs[0] + 5000
    print(f"first group: last-epi-end spread {(tle[first].max() - tle[first].min()) / G:.2f}; qcopy wait median {np.median((tq - tle)[first]) / G:.2f}, "
          f"first K - begin {np.median((tk - tb)[first]) / G:.2f}, group {np.median((tx - tq)[first]) / G:.2f} us")


if __name__ == "__main__":
    main()
