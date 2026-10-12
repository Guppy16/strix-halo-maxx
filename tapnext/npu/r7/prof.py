"""Per-tile GEMM-core cycle profile of a -DPROF build (round 7; npu-research P15/P16-lite).

The PROF kernels stamp the core timer (enabled from the runtime sequence) and write, over the first 20 bytes of each
output tile: magic 'PROF', cycles inside mm_acc, mm_acc calls, cycles from acc_zero to the epilogue (the k loop incl.
every wait), epilogue cycles. Outputs are garbage: timing only. Records are found by their magic in the ACT buffer
after a step and given to the last op (plan order) whose column drains cover them, so per-block shared regions show
the last block's tiles. Cycles at 1.81 GHz (RESULTS 16.1).

usage (npu-iron-torch, NPU): python3 r7/prof.py --build full12r7p [--frames 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyxrt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from step8 import Step8, demo_queries, preprocess, tap_ends  # noqa: E402

MAGIC = 0x464F5250
GHZ = 1.81


def ranges(op):
    """(first, last) byte ranges of every column drain of a gemm op"""
    out = []
    for c in op["c"]:
        for t in (c if isinstance(c, list) else [c]):
            out.append(tap_ends(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--frames", type=int, default=3)
    o = ap.parse_args()
    import torch

    frames = np.load("/work/tapnet/data/davis_reg.npy")[:o.frames]
    video = preprocess(frames)
    s = Step8(o.build, o.depth)
    q = demo_queries()
    s.reset()
    for t in range(o.frames):
        s.step(video[t], torch.cat([q[..., :1] - t, q[..., 1:]], -1))
    s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    act = np.frombuffer(s.bo_a.map(), np.uint32).copy()
    hits = np.nonzero(act == MAGIC)[0]
    gemms = [(i, op) for i, op in enumerate(s.p.ops) if op["kind"] == "gemm"]
    rngs = [(i, op["name"], ranges(op)) for i, op in gemms]
    per = {}
    for h in hits:
        b = int(h) * 4
        owner = None
        for i, name, rr in rngs:
            if any(f <= b <= l for f, l in rr):
                owner = name
        if owner is None or h + 4 >= len(act):
            continue
        busy, calls, kwall, epi = (int(v) for v in act[h + 1:h + 5])
        if calls == 0 or calls > 100000:
            continue
        per.setdefault(owner, []).append((busy, calls, kwall, epi))
    print(f"{len(hits)} records; ops with records: {len(per)}")
    print(f"{'op':14s} {'tiles':>5s} {'calls':>6s} {'cyc/call':>9s} {'busy%':>6s} {'k-wall us':>9s} {'epi us':>7s} {'tile us':>8s}"
          f" | k-wait us per tile: {'min':>6s} {'p25':>6s} {'med':>6s} {'p75':>6s} {'max':>6s}")
    for name in [n for _, n, _ in rngs if n in per]:
        a = np.array(per.pop(name), float)
        busy, calls, kwall, epi = a.mean(0)
        print(f"{name:14s} {len(a):5d} {calls:6.0f} {busy / calls:9.1f} {100 * busy / kwall:6.1f} {kwall / GHZ / 1e3:9.2f} "
              f"{epi / GHZ / 1e3:7.2f} {(kwall + epi) / GHZ / 1e3:8.2f} | " +
              " ".join(f"{v:6.2f}" for v in np.percentile((a[:, 2] - a[:, 0]) / GHZ / 1e3, [0, 25, 50, 75, 100])))


if __name__ == "__main__":
    main()
