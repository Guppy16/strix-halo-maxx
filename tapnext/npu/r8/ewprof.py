"""SSM phase profile of an EW_DEFS=-DPROF build (round 8): ew8.cc stamps the core timer between the phases of every
chunk and writes the running totals into each chunk's output pad (unused bytes, so the step stays correct).

usage (npu-iron-torch, NPU): python3 r8/ewprof.py BUILD [--frames 3]
       (any container, no NPU):  python3 r8/ewprof.py TEST --act build/X.npy   (a run8 test's ACT, PROF_OUT=X.npy)
Prints, per SSM op (block) and averaged over the 8 columns, the cycles per chunk of each phase (216 chunks per op:
12 heads x 18 row blocks of 8 rows) and the op total in us at 1.81 GHz.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from gprof import tap_ends  # noqa: E402

MAGIC = 0x504D5353
GHZ = 1.81
PHASES = ["in_wait", "conv", "gates_gemm", "sigmoid", "loga+exp", "sqrt", "recur", "z_panels", "out_wait", "out_copy",
          "par_wait", "par_copy", "cp_h_pan", "cp_x"]  # round 15: "copies" (phase 12) split out of out_copy (the rest: record + releases)
NPH = len(PHASES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--act", default=None)
    o = ap.parse_args()
    if o.act:
        from run8 import plan_of

        report(plan_of(o.build)[0].ops, np.load(o.act))
        return
    import pyxrt
    import torch
    from step8 import Step8, demo_queries, preprocess

    frames = np.load("/work/tapnet/data/davis_reg.npy")[:o.frames]
    video = preprocess(frames)
    s = Step8(o.build, 12)
    q = demo_queries()
    s.reset()
    for t in range(o.frames):
        s.step(video[t], torch.cat([q[..., :1] - t, q[..., 1:]], -1))
    s.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    act = np.frombuffer(s.bo_a.map(), np.uint32).copy()
    report(s.p.ops, act)


def report(ops, act):
    hits = np.nonzero(act == MAGIC)[0]
    ssm = [op for op in ops if op["kind"] == "ew" and op["fn"] == "ssm"]
    per = {}
    for h in hits:
        rec = act[h + 1:h + 2 + NPH].astype(np.int64)
        b = int(h) * 4
        for op in ssm:
            for c, t in enumerate(op["c"]):
                f, l = tap_ends(t)
                if f <= b <= l:
                    per.setdefault(op["name"], {}).setdefault(c, []).append(rec)
    print(f"{len(hits)} records, {len(per)} SSM ops; cycles per chunk (mean over columns), 1.81 GHz")
    print(f"{'op':10s} " + " ".join(f"{p:>10s}" for p in PHASES) + f" {'total us':>9s}")
    allr = []
    for name in [op["name"] for op in ssm if op["name"] in per]:
        cols = per[name]
        last = np.array([max(r, key=lambda v: v[0]) for r in cols.values()])  # the op's final totals per column
        n = last[:, 0].mean()
        ph = last[:, 1:1 + NPH].mean(0) / n
        allr.append(ph)
        print(f"{name:10s} " + " ".join(f"{v:10.0f}" for v in ph) + f" {last[:, 1:1 + NPH].sum(1).mean() / GHZ / 1e3:9.1f}"
              + f"  (chunks {n:.0f}, cols {len(last)})")
    # round 16: words 16-18 of a record = the norm n1k steps' cycles, their count, the in_need cycles around them (cumulative)
    last_all = [max(r, key=lambda v: v[0]) for name in per for r in per[name].values()]
    ext = act[hits[:, None] + np.arange(16, 20)[None, :]].astype(np.int64) if len(hits) else None
    if ext is not None and ext[:, 1].max() > 0:
        best = ext[np.argmax(ext[:, 1])]
        print(f"norm n1k steps: {best[1]} calls, {best[0] / max(best[1], 1):.1f} cycles per call, in_need {best[2] / max(best[1], 1):.1f} per unit, out_begin {best[3] / max(best[1], 1):.1f} per unit")
        ext2 = act[hits[:, None] + np.arange(20, 30)[None, :]].astype(np.int64)  # round 18 NIO2 (-DNIO2 builds)
        e = ext2[np.argmax(ext[:, 1])].reshape(2, 5)
        for md, nm in enumerate(("rms", "ln")):
            n2, n1 = max(e[md, 3], 1), max(e[md, 4], 1)
            print(f"NIO2 {nm}: pass-2 units {e[md, 3]}, nrm call {e[md, 0] / n2:.1f}, whole pass-2 unit {e[md, 1] / n2:.1f}; "
                  f"pass-1 units {e[md, 4]}, whole pass-1 unit {e[md, 2] / n1:.1f} cycles (stamps included)")
    m = np.mean(allr, 0)
    print(f"{'mean':10s} " + " ".join(f"{v:10.0f}" for v in m) + f" {m.sum() * 216 / GHZ / 1e3:9.1f}")
    print("share     " + " ".join(f"{100 * v / m.sum():9.1f}%" for v in m))


if __name__ == "__main__":
    main()
