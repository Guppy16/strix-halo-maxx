"""Round 24 SSMIN: read the SSM input-wait records of an EW_DEFS=-DSSMIN build (ew8.cc) after 3 frames.
usage (npu-iron-torch, NPU): python3 r24/ssmin.py BUILD"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r8"))
from gprof import tap_ends  # noqa: E402

MAGIC = 0x494D5353


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
    per = {}
    for h in hits:
        rec = act[h + 1:h + 7].astype(np.int64)
        b = int(h) * 4
        for op in ssm:
            for c, t in enumerate(op["c"]):
                f, l = tap_ends(t)
                if f <= b <= l:
                    per.setdefault(op["name"], {}).setdefault(c, []).append(rec)
    print(f"{len(hits)} records; per SSM op, mean over columns, cycles per chunk (the op's last record per column minus its first)")
    print(f"{'op':8s} {'chunks':>6s} {'in_need':>8s} {'par_need':>8s} {'chunk':>8s}  per column in_need")
    tot = []
    for name in [op["name"] for op in ssm if op["name"] in per]:
        rows = []
        for c, rs in sorted(per[name].items()):
            rs = sorted(rs, key=lambda v: v[0])
            a, b = rs[0], rs[-1]
            n = b[0] - a[0]
            if n <= 0:
                continue
            rows.append((c, n, (b[1] - a[1]) / n, (b[2] - a[2]) / n, (b[3] - a[3]) / n))
        if not rows:
            continue
        r = np.array(rows)
        tot.append(r[:, 2:5].mean(0))
        print(f"{name:8s} {r[:, 1].mean():6.0f} {r[:, 2].mean():8.0f} {r[:, 3].mean():8.0f} {r[:, 4].mean():8.0f}  " + " ".join(f"{v:5.0f}" for v in r[:, 2]))
    t = np.mean(tot, 0)
    print(f"{'mean':8s} {'':6s} {t[0]:8.0f} {t[1]:8.0f} {t[2]:8.0f}")
    # round 24: word 6 = the core timer at the record (EW cores' timers): per op, each column's first / last chunk time (us, relative to
    # the op's earliest first chunk) -> the column skew at the SSM's start and end
    print("column skew (us at 1.81 GHz): first-chunk spread, last-chunk spread, and the last chunk per column relative to the earliest")
    for name in [op["name"] for op in ssm if op["name"] in per]:
        fs, ls = [], []
        for c, rs in sorted(per[name].items()):
            rs = sorted(rs, key=lambda v: v[0])
            fs.append(rs[0][5] & 0xFFFFFFFF)
            ls.append(rs[-1][5] & 0xFFFFFFFF)
        fs, ls = np.array(fs, np.int64), np.array(ls, np.int64)
        print(f"{name:8s} start spread {(fs.max() - fs.min()) / 1810:6.2f}  end spread {(ls.max() - ls.min()) / 1810:6.2f}  ends "
              + " ".join(f"{(v - ls.min()) / 1810:5.2f}" for v in ls) + "  starts " + " ".join(f"{(v - fs.min()) / 1810:5.2f}" for v in fs))


if __name__ == "__main__":
    main()
