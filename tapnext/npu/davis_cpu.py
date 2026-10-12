"""TAP-Vid DAVIS 'first' with the vendored model on the CPU (bf16), depth / support-point knobs, no GPU, no NPU.
Same loader, query sampling and metrics as npu_step.py davis, so the numbers are comparable to its npu_real row.

    python3 davis_cpu.py --depth 10 [--videos 10] [--support 8 --support-radius 8] [--threads 8]
Appends a row to davis_cpu.jsonl.
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

ROOT = "/work/.scratch/npu-tapnext"
tn = types.ModuleType("tn")
exec(open("/tapnext/model.py").read().replace("-> TAPNext:", '-> "TAPNext":'), tn.__dict__)


def preprocess(frames_u8, res=256):
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    return (t / 127.5 - 1.0).permute(0, 2, 3, 1).contiguous()


def load_dv():
    src = "\n".join(l for l in open("/work/.scratch/tapnet/davis.py").read().split("\n") if "import tnpp" not in l)
    dv = types.ModuleType("dv")
    dv.__dict__["__file__"] = "/work/.scratch/tapnet/davis.py"
    exec(src.split("def track_video")[0], dv.__dict__)
    dv.ROOT = __import__("pathlib").Path("/work/.scratch/tapnet")
    return dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--support", type=int, default=0, help="co-tracked grid points per query (discarded on output)")
    ap.add_argument("--support-radius", type=float, default=8.0)
    ap.add_argument("--slots", type=int, default=48, help="query slots (rows) per step; real + support queries must fit")
    ap.add_argument("--threads", type=int, default=8)
    o = ap.parse_args()
    torch.set_num_threads(o.threads)
    dv = load_dv()
    data = dv.load_davis(dv.ROOT / "data" / "tapvid_davis.zip")
    names = sorted(data.keys())[: o.videos]
    model = tn.TAPNext()
    tn.load_state(model, "/tapnext/tapnextpp_256.safetensors")
    model.to_half(torch.bfloat16).eval()
    model.blocks = model.blocks[: o.depth]  # upstream-style layer dropping: run the first `depth` blocks, skip the rest
    per_video, ms = [], []
    for name in names:
        d = data[name]
        frames = d["video"]
        T = frames.shape[0]
        pts = d["points"] * np.array([256.0, 256.0])
        qp, gt_tracks, gt_occ = dv.sample_queries_first(d["occluded"], pts)
        Q = qp.shape[1]
        qq = qp
        if o.support:
            qq = dv.add_support(torch.from_numpy(qp.astype(np.float32)), o.support, o.support_radius).numpy()  # [1, Q * (1 + support), 3], real queries first
        n = qq.shape[1]
        q = torch.from_numpy(qq.astype(np.float32))
        slots = max(o.slots, n)
        q = torch.cat([q, torch.full((1, slots - n, 3), 10000.0)], 1) if n < slots else q
        video = preprocess(frames)
        caches = None
        tr, vl = [], []
        t1 = time.perf_counter()
        with torch.no_grad():
            for t in range(T):
                rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
                tracks, logits, vis, caches = model.step(video[t], rel, caches)
                tr.append(tracks[:Q].clone())
                vl.append(vis[:Q].clone())
        ms.append((time.perf_counter() - t1) * 1e3 / T)
        tracks = torch.stack(tr).numpy()
        vis = torch.stack(vl).numpy()
        pred_tracks = tracks[..., ::-1].transpose(1, 0, 2)[None]
        pred_occ = ~(vis > 0).T[None]
        m = dv.compute_tapvid_metrics(qp, gt_occ, gt_tracks, pred_occ, pred_tracks)
        per_video.append({k: float(np.mean(v)) for k, v in m.items()})
        print(f"{name:24s} T={T:3d} Q={Q:3d} slots={slots} AJ={per_video[-1]['average_jaccard']*100:5.1f} "
              f"OA={per_video[-1]['occlusion_accuracy']*100:5.1f} PTS={per_video[-1]['average_pts_within_thresh']*100:5.1f} {ms[-1]:.0f} ms/f", flush=True)
    agg = {k: round(float(np.mean([v[k] for v in per_video])) * 100, 2) for k in per_video[0]}
    row = dict(mode="cpu_bf16", depth=o.depth, support=o.support, support_radius=o.support_radius, videos=len(names),
               ms_per_frame=round(float(np.median(ms)), 1), **agg)
    with open(f"{ROOT}/davis_cpu.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps(row))
    print("DAVIS_DONE")


if __name__ == "__main__":
    main()
