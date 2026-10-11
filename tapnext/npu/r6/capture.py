"""Capture real activations at every nonlinearity of the TAPNext++ trunk (round 6, npu-research P8 part 1).

Runs the vendored CPU bf16 model (step8.load_model) on real frames and stores, per block and site, a random subset of
ROWS token rows with their full feature dimension (so 8-value bfp blocks can be formed downstream):
  ssm_gelu    linear_yx's y half before gelu (tanh form)             [rows, 768]
  ffw_gate    ffw_up's gate half before gelu (tanh form)             [rows, 3072]
  ffw_up      ffw_up's up half (multiplies gelu(gate))               [rows, 3072]
  vit_gelu    VitBlock mlp[0] output before nn.GELU (erf form!)      [rows, 3072]
  rglru_gx    input_gate(x) before sigmoid                           [rows, 768]
  rglru_ga    a_gate(x) before sigmoid                               [rows, 768]
  attn_s      q k^T / 8 for the sampled query rows, all heads        [rows, 12, keys]
usage (npu-iron-torch): python3 r6/capture.py --frames 3 --out build/acts.npz
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/npu-tapnext")
import step8  # noqa: E402

ROWS = 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--video", default="/work/tapnet/data/clip30.npy")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    torch.set_num_threads(16)
    model = step8.load_model(12)
    frames = np.load(o.video)[: o.frames]
    video = step8.preprocess(frames)
    rng = np.random.default_rng(0)
    store = {}
    state = dict(block=0, rows=None)

    def put(site, t):
        if site == "attn_s":  # already the sampled rows
            a = t
        else:
            a = t.detach().float().reshape(-1, t.shape[-1]).numpy()[state["rows"]]
        store.setdefault(f"{site}_b{state['block']}", []).append(a.astype(np.float32))

    tn = step8.tn
    orig_gelu = tn.gelu

    def gelu_hook(x):
        put(state["site_gelu"], x)
        return orig_gelu(x)

    tn.gelu = gelu_hook
    hooks = []
    for b, blk in enumerate(model.blocks):
        def pre_block(m, args, b=b):
            state["block"] = b
        hooks.append(blk.register_forward_pre_hook(pre_block))
        rb = blk.ssm_block.recurrent_block
        hooks.append(rb.register_forward_pre_hook(lambda m, a: state.__setitem__("site_gelu", "ssm_gelu")))
        mb = blk.ssm_block.mlp_block

        def mlp_pre(m, a):
            state["site_gelu"] = "ffw_gate"
            gate, up = m.ffw_up(a[0]).chunk(2, dim=-1)
            put("ffw_up", up)
        hooks.append(mb.register_forward_pre_hook(mlp_pre))
        lru = rb.rg_lru
        hooks.append(lru.input_gate.register_forward_hook(lambda m, a, out: put("rglru_gx", out)))
        hooks.append(lru.a_gate.register_forward_hook(lambda m, a, out: put("rglru_ga", out)))
        vb = blk.vit_block
        hooks.append(vb.mlp[0].register_forward_hook(lambda m, a, out: put("vit_gelu", out)))
        sa = vb.self_attention

        def attn_pre(m, a):
            x = a[0]
            n, c = x.shape
            qkv = F.linear(x, m.in_proj_weight, m.in_proj_bias)
            q, k, _ = qkv.view(n, 3, m.num_heads, c // m.num_heads).permute(1, 2, 0, 3)
            s = (q[:, state["rows"]].float() @ k.float().transpose(1, 2)) / 8.0  # [H, rows, keys]
            put("attn_s", s.permute(1, 0, 2).numpy())
        hooks.append(sa.register_forward_pre_hook(attn_pre))

    q = step8.demo_queries()
    caches = None
    with torch.no_grad():
        for t in range(len(video)):
            state["rows"] = np.sort(rng.choice(1024 + 48, ROWS, replace=False))  # image tokens + the query slots
            rel = torch.cat([q[..., :1] - t, q[..., 1:]], -1)
            _, _, _, caches = model.step(video[t], rel, caches)
    for h in hooks:
        h.remove()
    tn.gelu = orig_gelu
    out = {k: np.concatenate(v) for k, v in store.items()}
    np.savez_compressed(o.out, **out)
    print("captured", len(out), "arrays;", ", ".join(f"{k} {v.shape}" for k, v in list(out.items())[:8]), "...")


if __name__ == "__main__":
    main()
