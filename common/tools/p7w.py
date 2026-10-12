"""P7 pre-check, weight side only (rev 6): how much does folding the norm gain g into the consumer weights (W' = diag(g) W,
then bfp16 along K) change the weight quantisation error, against today's bfp16(W) with g applied to the activations?

For a normalised input n with unit rms per feature, the weight-side output error is
    today:  n diag(g) (q(W) - W)          folded:  n (q(diag(g) W) - diag(g) W)
so with independent features the ratio of their Frobenius norms is the weight-side error ratio of the fold (1.0 = same).
Also prints the gain statistics per site (spread inside 8-blocks along K is what hurts the shared exponent).
The activation side (bfp16 of the un-normalised x vs of the normalised, gained y) needs real activations: P7a.

usage: python3 p7w.py   (numpy only; reads $TAPNEXT_WEIGHTS (default $SHM_SCRATCH/models/tapnextpp_256.safetensors) read-only)
"""

import json
import os
import struct
from pathlib import Path

import numpy as np

ST = Path(os.environ.get("TAPNEXT_WEIGHTS") or Path(os.environ.get("SHM_SCRATCH") or Path(__file__).resolve().parents[2] / ".scratch") / "models/tapnextpp_256.safetensors")


def tensors():
    f = ST.open("rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    mm = np.memmap(ST, dtype=np.uint8, mode="r")

    def get(k):
        e = h[k]
        a, b = e["data_offsets"]
        assert e["dtype"] == "F32"
        return np.frombuffer(mm[8 + n + a:8 + n + b], dtype=np.float32).reshape(e["shape"])

    return get


def bfp16(w):
    """bfp16ebs8 along the last axis (blocks of 8): shared exponent of the block max, int8 mantissas (RNE, |m| <= 127)."""
    s = w.shape
    b = w.reshape(*s[:-1], s[-1] // 8, 8).astype(np.float64)
    mx = np.abs(b).max(-1, keepdims=True)
    e = np.floor(np.log2(np.where(mx > 0, mx, 1.0)))
    step = 2.0 ** (e - 6)
    m = np.clip(np.round(b / step), -127, 127)
    return (m * step).reshape(s)


def site(name, g, W):
    """W: [N, K] (out, in) as the GEMM's B is packed ([n][k], blocks along k); g: [K]."""
    g = g.astype(np.float64)
    W = W.astype(np.float64)
    Wp = W * g[None, :]
    e_today = (bfp16(W) - W) * g[None, :]
    e_fold = bfp16(Wp) - Wp
    r = np.linalg.norm(e_fold) / np.linalg.norm(e_today)
    gb = np.abs(g).reshape(-1, 8)
    spread = gb.max(1) / np.maximum(gb.min(1), 1e-12)
    print(f"  {name:28s} |g| min {np.abs(g).min():7.4f} med {np.median(np.abs(g)):7.4f} max {np.abs(g).max():8.4f}"
          f"  8-block max/min p50 {np.median(spread):6.2f} p90 {np.percentile(spread, 90):7.2f} max {spread.max():9.1f}"
          f"  | weight-side error ratio fold/today {r:5.3f}")
    return r


def main():
    get = tensors()
    print("P7 weight-side pre-check (bfp16 along K, unit-rms normalised inputs):")
    worst = {}
    for b in range(12):
        p = f"blocks.{b}."
        print(f"block {b}")
        g0 = get(p + "ssm_block.temporal_pre_norm.scale") + 1.0
        Wyx = np.concatenate([get(p + "ssm_block.recurrent_block.linear_y.weight"), get(p + "ssm_block.recurrent_block.linear_x.weight")])
        worst.setdefault("norm0 -> yx", []).append(site("norm0 (RMS) -> yx", g0, Wyx))
        g1 = get(p + "ssm_block.channel_pre_norm.scale") + 1.0
        wu = get(p + "ssm_block.mlp_block.ffw_up.w")  # [2, 768 in, 3072 out]
        Wup = np.concatenate([wu[0].T, wu[1].T])  # [6144 out, 768 in]
        worst.setdefault("norm1 -> up", []).append(site("norm1 (RMS) -> up_a/up_b", g1, Wup))
        g2 = get(p + "vit_block.ln_1.weight")
        worst.setdefault("norm2 -> qkv", []).append(site("norm2 (LN) -> qkv", g2, get(p + "vit_block.self_attention.in_proj_weight")))
        g3 = get(p + "vit_block.ln_2.weight")
        worst.setdefault("norm3 -> mlp0", []).append(site("norm3 (LN) -> mlp0", g3, get(p + "vit_block.mlp.0.weight")))
    print("summary (weight-side error ratio fold/today over the 12 blocks):")
    for k, v in worst.items():
        print(f"  {k:14s} median {np.median(v):5.3f}  max {np.max(v):5.3f}")


if __name__ == "__main__":
    main()
