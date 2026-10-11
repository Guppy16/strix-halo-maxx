"""Per-op GEMM-core cycle profile of a -DPROF build, from a run8 test's ACT image (round 8; r7/prof.py for the step).

mm8.cc -DPROF writes per output tile: magic, cycles inside mm_acc, calls, acc_zero -> epilogue (k loop incl. waits),
epilogue cycles, the inter-call gaps at k-step starts (B + A acquire) and inside k-steps (A only), the tile's first
wait, the largest k-step-start gap and last call -> epilogue. Records go to the last op (plan order) whose drains
cover them. Cycles at 1.81 GHz.

usage: PROF_OUT=X.npy r8.sh LOG run NAME ... ; python3 r8/gprof.py NAME build/X.npy
       python3 r8/gprof.py --step BUILD   (in npu-iron-torch with the NPU: runs 3 frames of step8 first)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

MAGIC = 0x464F5250
GHZ = 1.81
NREC = 12


def tap_ends(t):
    """first and last byte a plan8 TAP touches"""
    last = t["off"] + sum((s - 1) * st for s, st in zip(t["sizes"][:3], t["strides"][:3])) + t["sizes"][3] - 1
    return t["off"], last


def ranges(op):
    out = []
    for c in op["c"]:
        for t in (c if isinstance(c, list) else [c]):
            out.append(tap_ends(t))
    return out


def table(ops, act):
    hits = np.nonzero(act == MAGIC)[0]
    rngs = [(op["name"], ranges(op)) for op in ops if op["kind"] == "gemm"]
    per = {}
    for h in hits:
        b = int(h) * 4
        owner = None
        for name, rr in rngs:
            if any(f <= b <= l for f, l in rr):
                owner = name
        if owner is None or h + NREC >= len(act):
            continue
        rec = [int(v) for v in act[h + 1:h + NREC]]
        if rec[1] == 0 or rec[1] > 100000:
            continue
        per.setdefault(owner, []).append(rec)
    us = lambda c: c / GHZ / 1e3
    print(f"{len(hits)} records; ops with records: {len(per)} (cycles at {GHZ} GHz; us = per tile, mean over tiles)")
    print(f"{'op':12s} {'tiles':>5s} {'calls':>5s} {'cyc/call':>8s} {'busy%':>6s} {'kwall':>7s} {'first':>6s} {'gKB/ks':>7s} "
          f"{'gA/call':>7s} {'gKBmax':>7s} {'tail':>5s} {'epi':>6s} {'hand':>6s} {'zero':>5s} | first wait us min/med/max")
    rows = {}
    for name in [n for n, _ in rngs if n in per]:
        a = np.array(per.pop(name), float)
        busy, calls, kwall, epi, gkb, ga, first, gmax, tail, hand, zero = a.mean(0)
        handm = np.median(a[:, 9])
        ks = calls / 9
        rows[name] = dict(tiles=len(a), calls=calls, cyc_call=busy / calls, busy=busy, kwall=kwall, epi=epi, gkb=gkb, ga=ga,
                          first=first, gmax=a[:, 7].max(), tail=tail, ks=ks)
        print(f"{name:12s} {len(a):5d} {calls:5.0f} {busy / calls:8.1f} {100 * busy / kwall:6.1f} {us(kwall):7.2f} {us(first):6.2f} "
              f"{gkb / max(ks - 1, 1):7.0f} {ga / max(calls - ks, 1):7.1f} {a[:, 7].max():7.0f} {tail:5.0f} {us(epi):6.2f} {us(handm):6.2f} {us(zero):5.2f} | "
              + " ".join(f"{us(v):6.2f}" for v in np.percentile(a[:, 6], [0, 50, 100])))
    return rows


def attn_table(act):
    """attention records (mm8.cc -DPROF attn_end): per head group, by pass (ntiles 2 = pass 0, 1 = pass 1)"""
    hits = np.nonzero(act == 0x50545441)[0]
    rec = np.array([act[h + 1:h + 8] for h in hits if h + 8 < len(act)], np.int64)
    if not len(rec):
        return
    us = lambda c: c / GHZ / 1e3
    print(f"attention: {len(rec)} head-group records (us, median; pre = Q copies + A waits before attn_begin)")
    print(f"{'pass':>4s} {'n':>5s} {'pre':>7s} {'begin':>7s} {'K':>7s} {'V':>7s} {'gaps':>7s} {'end':>7s} {'sum':>7s}")
    for nt in (2, 1):
        r = rec[rec[:, 6] == nt]
        if len(r):
            m = np.median(r[:, :6], 0)
            print(f"{2 - nt:4d} {len(r):5d} " + " ".join(f"{us(v):7.2f}" for v in m) + f" {us(m[1:].sum()):7.2f}")


def main():
    if sys.argv[1] == "--step":
        import pyxrt
        import torch
        from step8 import Step8, demo_queries, preprocess

        frames = np.load("/work/tapnet/data/clip30.npy")[:3]
        video = preprocess(frames)
        s = Step8(sys.argv[2], 12)
        q = demo_queries()
        s.reset()
        for t in range(3):
            s.step(video[t], torch.cat([q[..., :1] - t, q[..., 1:]], -1))
        s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        act = np.frombuffer(s.bo_a.map(), np.uint32).copy()
        table(s.p.ops, act)
        attn_table(act)
        return
    from run8 import plan_of  # noqa: F401

    p, _ = plan_of(sys.argv[1])
    table(p.ops, np.load(sys.argv[2]))


if __name__ == "__main__":
    main()
