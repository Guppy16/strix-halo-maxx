"""Round 25: the norm-unit stamps of an EW_DEFS=-DNIO2 build (ew8.cc round 18) after 3 frames, per EW column: mean cycles per unit
of the nrm call, the whole pass-2 unit, the whole pass-1 unit, by mode (RMS, LN). Records sit in the SSM chunk pads (cumulative
totals): the last record per column holds the run's totals. usage (npu-iron-torch, NPU): python3 r25/nio.py BUILD"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r8"))
from gprof import tap_ends  # noqa: E402

MAGIC = 0x504D5353


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
    ssm = [op for op in s.p.ops if op["kind"] == "ew" and op["fn"] == "ssm"]
    best = {}
    for h in hits:
        rec = act[h:h + 30].astype(np.int64)
        b = int(h) * 4
        for op in ssm:
            for c, t in enumerate(op["c"]):
                f, l = tap_ends(t)
                if f <= b <= l and rec[17] > best.get(c, (0, None))[0]:
                    best[c] = (rec[17], rec)
    print(f"{len(hits)} records; per column, cycles per unit (stamps included): RMS nrm / pass-2 unit / pass-1 unit (units) | LN same")
    tot = []
    for c, (_, r) in sorted(best.items()):
        g = r[20:30].reshape(2, 5)
        v = [g[m, i] / max(g[m, 3 if i < 2 else 4], 1) for m in range(2) for i in range(3)]
        tot.append(v)
        print(f"col {c}: RMS {v[0]:6.0f} {v[1]:6.0f} {v[2]:6.0f} ({g[0, 3]}, {g[0, 4]}) | LN {v[3]:6.0f} {v[4]:6.0f} {v[5]:6.0f} ({g[1, 3]}, {g[1, 4]})")
    t = np.mean(tot, 0)
    print(f"mean : RMS {t[0]:6.0f} {t[1]:6.0f} {t[2]:6.0f} | LN {t[3]:6.0f} {t[4]:6.0f} {t[5]:6.0f}")


if __name__ == "__main__":
    main()
