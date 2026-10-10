"""TAP-Vid DAVIS 'first' AJ for the VENDORED TAPNext++ (app/jobs/tapnext/model.py) with the NPU's GEMM numerics
emulated on the GPU: every trunk nn.Linear and the attention QK^T / PV run with bfp16ebs8 operands (blocks of 8
along K share an exponent, 8-bit mantissas, round-to-nearest-even) and fp32 accumulation, outputs rounded to bf16
(what the NPU kernels produce). Heads stay fp32 as in the live model.

usage: davis_npu.py --mode {bf16,bfp,bfp_attn} [--videos 10] [--tag name]
  bf16      the vendored model as it runs live (bf16 trunk)                 -> should match tnpp's 66.35 / 66.3
  bfp       + bfp16 operands on all trunk Linear layers (NPU GEMM program)
  bfp_attn  + bfp16 Q/K/V/P in the attention (NPU attention kernel v15 numerics)
"""
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "~/git/app/.scratch/tapnet")
import davis  # noqa: E402  (load_davis, sample_queries_first, compute_tapvid_metrics)
import tnpp  # noqa: E402  (preprocess)

spec = importlib.util.spec_from_file_location("tn", "~/git/app/agent/app/jobs/tapnext/model.py")
tn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tn)

BLOCK = 8


def bfp_quantize(x: torch.Tensor) -> torch.Tensor:
    """What the NPU reads of x (..., K), K % 8 == 0, as float32 (mlir-aie bfp.quantize, conv_even)."""
    b = x.float().reshape(*x.shape[:-1], -1, BLOCK)
    _, e = torch.frexp(b.abs().amax(-1, keepdim=True))
    scale = torch.ldexp(torch.ones_like(b[..., :1]), e - 7)
    m = torch.round(b / scale)
    scale = torch.where(m.amax(-1, keepdim=True) > 127, scale * 2, scale)
    m = torch.round(b / scale).clamp(-128, 127)
    return (m * scale).reshape(x.shape)


_linear = F.linear
_sdpa = F.scaled_dot_product_attention
MODE = "bf16"


def bfp_linear(x, w, b=None):
    if x.dtype != torch.bfloat16 or x.shape[-1] % BLOCK:  # heads (fp32) and anything odd stay exact
        return _linear(x, w, b)
    y = bfp_quantize(x) @ bfp_quantize(w).T  # fp32 accumulate
    if b is not None:
        y = y + b.float()
    return y.to(x.dtype)


def bfp_attention(q, k, v, **kw):
    """v15 numerics: s = bf16(bfp(q) bfp(k)^T) * scale -> bf16; p = bf16(exp(s - max)); P bfp; O fp32; o = bf16(O / sum)."""
    if q.dtype != torch.bfloat16:
        return _sdpa(q, k, v, **kw)
    scale = torch.tensor(q.shape[-1] ** -0.5, dtype=torch.bfloat16).float()
    s = (bfp_quantize(q) @ bfp_quantize(k).transpose(-1, -2)).to(torch.bfloat16).float()
    s = (s * scale).to(torch.bfloat16).float()
    e = torch.exp(s - s.amax(-1, keepdim=True)).to(torch.bfloat16).float()
    nk = e.shape[-1]; pad = (-nk) % BLOCK  # the NPU pads the keys to a multiple of 48 with zeros (P = 0 there)
    e = F.pad(e, (0, pad)); vq = F.pad(bfp_quantize(v), (0, 0, 0, pad))
    p = bfp_quantize(e)
    o = p @ vq
    return (o / p.sum(-1, keepdim=True)).to(torch.bfloat16)


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bf16")
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    MODE = args.mode
    if MODE in ("bfp", "bfp_attn"):
        F.linear = bfp_linear
    if MODE == "bfp_attn":
        F.scaled_dot_product_attention = bfp_attention
    dev = torch.device("cuda")
    model = tn.TAPNext()
    tn.load_state(model, "~/git/app/data/models/tapnextpp_256.safetensors")
    model.to(dev).to_half(torch.bfloat16).eval()
    data = davis.load_davis()
    names = sorted(data.keys())[: args.videos]
    per_video, ms = [], []
    for name in names:
        d = data[name]
        frames = d["video"]
        T = frames.shape[0]
        pts = d["points"] * np.array([256.0, 256.0])
        qp, gt_tracks, gt_occ = davis.sample_queries_first(d["occluded"], pts)
        q = torch.from_numpy(qp.astype(np.float32)).to(dev)  # [1,Q,3] (t,y,x)
        video = tnpp.preprocess(frames, 256, dev)
        caches = None
        tr, vl = [], []
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.no_grad():
            for t in range(T):
                rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
                tracks, logits, vis, caches = model.step(video[t], rel, caches)
                tr.append(tracks.clone()); vl.append(vis.clone())
        torch.cuda.synchronize(); ms.append((time.perf_counter() - t1) * 1e3 / T)
        tracks = torch.stack(tr).cpu().numpy(); vis = torch.stack(vl).cpu().numpy()  # [T,Q,2] (y,x), [T,Q]
        pred_tracks = tracks[..., ::-1].transpose(1, 0, 2)[None]
        pred_occ = ~(vis > 0).T[None]
        m = davis.compute_tapvid_metrics(qp, gt_occ, gt_tracks, pred_occ, pred_tracks)
        per_video.append({k: float(np.mean(v)) for k, v in m.items()})
        print(f"{name:24s} T={T:3d} Q={qp.shape[1]:3d} AJ={per_video[-1]['average_jaccard']*100:5.1f} OA={per_video[-1]['occlusion_accuracy']*100:5.1f} PTS={per_video[-1]['average_pts_within_thresh']*100:5.1f} {ms[-1]:.1f} ms/f", flush=True)
    agg = {k: round(float(np.mean([v[k] for v in per_video])) * 100, 2) for k in per_video[0]}
    row = dict(mode=MODE, tag=args.tag, videos=len(names), ms_per_frame=round(float(np.median(ms)), 1), **agg)
    with open("~/git/app/.scratch/npu-tapnext/davis_npu.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps(row))
    print("DAVIS_DONE")


if __name__ == "__main__":
    main()
