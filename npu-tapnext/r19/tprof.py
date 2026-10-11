"""Round 19: per-tile GEMM stamps of a -DTPROF build (mm8.cc mm_ktile / acc_epilogue, magic 'TPRF') after 3 frames of step8.

Per tile: first = the first k-step incl. its B / A waits, ks = the other k-steps per k-step, mid = mm_ktile return -> epilogue,
epi = the epilogue, hand = previous epilogue end -> this tile's start (C acquire + loop code; a core's first tile of an op also
carries the previous op). Records land where each tile's C object drains: the last op (plan order) whose drains cover them.
Prints per op kind: tiles found, medians (cycles), and the excess over the steady k-step per tile.
usage (npu-iron-torch, NPU): python3 r19/tprof.py BUILD
"""
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
    ops = s.p.ops
    rngs = [(op["name"], gprof.ranges(op), op) for op in ops if op["kind"] == "gemm"]
    per = {}
    for h in hits:
        b = int(h) * 4
        owner = None
        for name, rr, op in rngs:
            if any(f <= b <= l for f, l in rr):
                owner = (name.split("_", 1)[1], op)
        if owner is None or h + 8 >= len(act):
            continue
        rec = act[h + 1:h + 8].astype(np.int64)
        if not (0 < rec[2] <= 64):
            continue
        per.setdefault(owner[0], (owner[1], []))[1].append(rec)
    print(f"TPROF {sys.argv[1]}: {len(hits)} records; cycles (median over tiles; p90 in brackets); 1.81 GHz")
    print(f"{'kind':10s} {'K':>5s} {'nt':>3s} {'tiles':>5s} {'first':>13s} {'ks/kstep':>9s} {'mid':>5s} {'epi':>13s} {'hand':>15s} "
          f"{'tile excess':>11s} {'op excess us':>12s}")
    tot = 0.0
    for kind, (op, rs) in sorted(per.items(), key=lambda kv: kv[0]):
        a = np.array(rs)
        nks = np.median(a[:, 2])
        ksk = np.median(a[:, 1] / np.maximum(a[:, 2] - 1, 1))
        med = np.median(a, 0)
        p90 = np.percentile(a, 90, 0)
        hand = a[:, 5]
        hmed = np.median(hand)
        # excess per tile over nks steady k-steps: (first - ksk) + mid + epi + hand (median hand: steady tiles)
        exc = (med[0] - ksk) + med[3] + med[4] + hmed
        opx = exc * op["nt"] / 1.81e3
        tot += 12 * opx
        print(f"{kind:10s} {op['K']:5d} {op['nt']:3d} {len(a):5d} {med[0]:6.0f} [{p90[0]:5.0f}] {ksk:9.1f} {med[3]:5.0f} "
              f"{med[4]:6.0f} [{p90[4]:5.0f}] {hmed:6.0f} [{np.percentile(hand, 90):6.0f}] {exc:11.0f} {opx:12.1f}")
    print(f"sum of op excess x 12 blocks: {tot / 1e3:.2f} ms (steady tiles; the op's first-tile waits come on top)")


if __name__ == "__main__":
    main()
