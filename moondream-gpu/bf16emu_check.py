"""Check bf16emu's ops against fp32 on real vision-tower activations (one frame, blocks 0/13/26),
next to the GPU path's ops (fp32 inside, one bf16 rounding), then the encoder output per config.

    venv-rocm714/bin/python bf16emu_check.py
"""

import torch
import torch.nn.functional as F

import bf16emu
import hgemm
import md2fast
from common import SESS, load_model, prep
from md2fast import NTOK, FastMD2

BLOCKS = (0, 13, 26)


def rel(out, ref):
    return ((out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def report(name, ref, gpu, emu):
    print(f"  {name:10} rel rms err  gpu-path {rel(gpu, ref):.2e}   bf16-emu {rel(emu, ref):.2e}")


model = load_model()
fast = FastMD2(model, lin_cls=hgemm.HLin, tlin_cls=hgemm.HI8Lin, prompt_len=48)
fast.set_image(prep(f"{SESS}/sessA/judged_0060.jpg"))

with torch.inference_mode():
    # the default vision path, keeping the inputs of each op in the chosen blocks
    x = fast.img_buf.permute(2, 0, 1)[None].to(torch.bfloat16).div_(255.0).sub_(0.5).div_(0.5)
    x = x.reshape(1, 3, 27, 14, 27, 14).permute(0, 2, 4, 1, 3, 5).reshape(1, NTOK, -1)
    x = fast.patch_emb(x) + fast.pos_emb
    H = fast.vh
    for i, b in enumerate(fast.vblocks):
        h = md2fast._ln(x, b["ln1"])
        q, k, v = [t.view(1, NTOK, H, -1).transpose(1, 2) for t in b["qkv"](h).chunk(3, dim=-1)]
        if i in BLOCKS:
            print(f"block {i}")
            w, bb = (t.float() for t in b["ln1"])
            report("layernorm", F.layer_norm(x.float(), (x.shape[-1],), w, bb), h, bf16emu.ln(x, b["ln1"]))
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
            emu = bf16emu.attn(q, k, v).view(1, NTOK, H, -1).transpose(1, 2)
            bfp = bf16emu.attn_bfp(q, k, v).view(1, NTOK, H, -1).transpose(1, 2)
            report("attention", ref, F.scaled_dot_product_attention(q, k, v), emu)
            report("attn-bfp", ref, F.scaled_dot_product_attention(q, k, v), bfp)
            print(f"  attn-bfp vs bf16-emu attn rel rms {rel(bfp, emu.float()):.2e}")
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, NTOK, -1)
        x = md2fast.lin_res(b["proj"], a, x)
        h2 = md2fast._ln(x, b["ln2"])
        f = b["fc1"](h2)  # pre-activation, bf16
        if i in BLOCKS:
            report("gelu", F.gelu(f.float(), approximate="tanh"), F.gelu(f, approximate="tanh"), bf16emu.gelu(f))
            report("gelu-sig", F.gelu(f.float(), approximate="tanh"), F.gelu(f, approximate="tanh"), bf16emu.gelu_sig(f))
        x = b["fc2"](md2fast.gelu(f)) + x

    # whole encoder output (729 x 2048 projected tokens) per config vs the GPU path
    base = fast._vision(fast.img_buf).float()
    print("encoder output vs GPU path")
    for ops in ("ln", "gelu", "gelusig", "softmax", "all", "ln,gelusig,softmax", "attnbfp", "ln,gelusig,attnbfp"):
        bf16emu.enable(ops)
        out = fast._vision(fast.img_buf).float()
        cos = F.cosine_similarity(out, base, dim=-1)
        print(f"  {ops:8} rel rms err {rel(out, base):.2e}   token cosine mean {cos.mean():.5f} min {cos.min():.5f}")
