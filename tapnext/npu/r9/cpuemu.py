"""SSM numerics inside the CPU model (round 9): run the torch bf16 tracker with the RG-LRU's transcendentals replaced by
bit-level emulations (r9/ssmemu.py) and compare the tracks against the plain torch bf16 run, so a numeric change can be
judged end to end without a build. 48 query points on a grid at t = 0, the original 30-frame test clip (private, not included).

usage (npu-iron-torch): python3 r9/cpuemu.py [--frames 30] [--cands deployed,lut,...]
Prints per variant: |track - torch| in px (mean, p95, max over points x frames, and the last 10 frames), visibility flips.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ssmemu  # noqa: E402
import step8  # noqa: E402
from emu import F32, bf  # noqa: E402


def to_np(t):
    return t.detach().float().numpy()


def to_bf(a, like):
    return torch.from_numpy(np.ascontiguousarray(a, F32)).to(torch.bfloat16).reshape(like.shape)


def make_forward(sig, a_om, sqrt):
    def forward(self, x, h0):
        gate_x = to_bf(sig(bf(to_np(self.input_gate(x)))), x)
        gate_a = to_bf(sig(bf(to_np(self.a_gate(x)))), x)
        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a_np, om_np, _ = a_om(to_np(log_a))
        a = to_bf(a_np, x)
        gated_x = x * gate_x
        if h0 is None:
            return gated_x, gated_x.float()
        mult = to_bf(sqrt(om_np), x)
        normalized_x = gated_x * mult
        h = a.float() * h0 + normalized_x.float()
        return h.to(x.dtype), h
    return forward


def perturb_sig(rate, seed=0):
    """control: torch's exact sigmoid with `rate` of the outputs moved by one bf16 ulp up or down at random"""
    rng = np.random.default_rng(seed)

    def fn(z):
        s = bf(1 / (1 + np.exp(-np.asarray(z, np.float64))))
        b = ssmemu.bits16(s)
        flip = rng.random(s.shape) < rate
        step = np.where(rng.random(s.shape) < 0.5, -1, 1)
        return np.where(flip & (s > 0) & (s < 1), ssmemu.from16(b + step), s)
    return fn


def exact_sig(z):
    return bf(1 / (1 + np.exp(-np.asarray(z, np.float64))))


def exact_a_om(la):
    la = np.asarray(la, np.float64)
    return bf(np.exp(la)), bf(1 - bf(np.exp(2 * la))), np.exp(la)


VARIANTS = {
    "exact": (exact_sig, exact_a_om, ssmemu.sqrt_exact),  # fp64 math, RNE: torch to the last rounding
    "deployed": (ssmemu.sig_nk8, ssmemu.a_om_nk8, ssmemu.sqrt_nk8),
    "lut": (ssmemu.sig_table(-12, 8, -3), ssmemu.a_om_nk8, ssmemu.sqrt_exact),
    "lut+rr64": (ssmemu.sig_table(-12, 8, -3), ssmemu.a_om_rr(64), ssmemu.sqrt_exact),
    "flip5%": (perturb_sig(0.05), exact_a_om, ssmemu.sqrt_exact),
    "flip18%": (perturb_sig(0.18), exact_a_om, ssmemu.sqrt_exact),
}


def grid_queries(n=48):
    ys, xs = np.meshgrid(np.linspace(24, 232, 8), np.linspace(24, 232, 6), indexing="ij")
    q = np.stack([np.zeros(n), ys.ravel()[:n], xs.ravel()[:n]], 1)
    return torch.tensor(q[None], dtype=torch.float32)


def run(model, video, q, frames):
    caches, trs, vis = None, [], []
    with torch.no_grad():
        for t in range(frames):
            rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
            tr, _, v, caches = model.step(video[t], rel, caches)
            trs.append(tr.numpy().copy())
            vis.append(v.numpy().copy())
    return np.stack(trs), np.stack(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--cands", default="exact,deployed,lut,flip5%,flip18%")
    o = ap.parse_args()
    torch.set_num_threads(16)
    frames = np.load("/work/tapnet/data/davis_reg.npy")[: o.frames]
    video = step8.preprocess(frames)
    model = step8.load_model(o.depth)
    q = grid_queries()
    orig = step8.tn.RGLRU.forward
    ref_tr, ref_vis = run(model, video, q, o.frames)
    print(f"torch bf16 reference: {ref_tr.shape[1]} points x {o.frames} frames, depth {o.depth}")
    for name in o.cands.split(","):
        step8.tn.RGLRU.forward = make_forward(*VARIANTS[name])
        tr, vis = run(model, video, q, o.frames)
        step8.tn.RGLRU.forward = orig
        d = np.abs(tr - ref_tr).max(-1)  # [frames, points] max over (y, x)
        flips = np.mean((vis > 0) != (ref_vis > 0))
        print(f"{name:10s} |d| px mean {d.mean():.4f} p95 {np.percentile(d, 95):.4f} max {d.max():.4f}  last10 mean "
              f"{d[-10:].mean():.4f} max {d[-10:].max():.4f}  vis flips {flips:.3%}", flush=True)


if __name__ == "__main__":
    main()
