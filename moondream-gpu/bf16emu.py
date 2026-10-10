"""bf16-only elementwise numerics in moondream2's vision tower, emulated on the GPU.

The XDNA2 NPU's vector unit has no fp32 arithmetic, so on the NPU every elementwise op rounds its
result to bf16; only reductions (LayerNorm mean/variance, softmax row sum) can accumulate in fp32.
Here each emulated op computes in fp32 and rounds to bf16 right after (`bf`), constants included.
exp is correctly rounded from its bf16 input, i.e. the best a bf16 exp unit could do.

OPS picks the emulated ops ("ln", "gelu", "softmax"); the others run as in md2fast (fp32 inside the
op, one rounding of the output). The vision tower covers the 27 blocks, post_ln and the projection
MLP; the text model is untouched. Residual adds need no emulation: md2fast already adds two bf16
tensors and rounds the sum to bf16, which is exactly a bf16 vector add.

    venv-rocm714/bin/python frame_preds.py --name bf16_all --bf16 all
"""

import math

import torch
import torch.nn.functional as F

import md2fast
from bfpemu import bfp_quantize
from md2fast import GRID, NTOK, PATCH, lin_res

OPS: set[str] = set()
ALL = ("ln", "gelu", "softmax")
# gelusig: gelu as x * sigmoid(2u), same ops in bf16 without the 1 + tanh cancellation
# attnbfp: the NPU attention kernel (npu-moondream p9/attn15.cc), in place of softmax
VARIANTS = ("gelusig", "attnbfp")
BF = torch.bfloat16


def bf(t: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (nearest even), keep computing in fp32."""
    return t.to(BF).float()


def const(c: float) -> float:
    return bf(torch.tensor(c)).item()


GELU_C0 = const(math.sqrt(2 / math.pi))
GELU_C1 = const(0.044715)


def ln(x, wb, eps=1e-5):
    x = x.float()
    mean = bf(x.mean(-1, keepdim=True))  # fp32 reduction; the subtract below takes it as a bf16 operand
    d = bf(x - mean)
    var = (d * d).mean(-1, keepdim=True)  # fp32 reduction of the bf16 differences
    rstd = bf(torch.rsqrt(var + eps))  # scalar rsqrt in fp32, broadcast as bf16
    return bf(bf(bf(d * rstd) * wb[0].float()) + wb[1].float()).to(BF)


def gelu(x):
    """0.5 x (1 + tanh(c0 (x + c1 x^3))), tanh(u) = 1 - 2 / (exp(2u) + 1), every op rounded."""
    x = x.float()
    x3 = bf(bf(x * x) * x)
    u = bf(GELU_C0 * bf(x + bf(GELU_C1 * x3)))
    e = bf(torch.exp(2 * u))  # 2u is exact
    th = bf(1 - 2 * bf(1 / bf(e + 1)))  # 2 * (bf16 reciprocal) is exact
    return bf(bf(0.5 * x) * bf(1 + th)).to(BF)


def gelu_sig(x):
    """0.5 x (1 + tanh(u)) = x / (1 + exp(-2u)), every op rounded."""
    x = x.float()
    x3 = bf(bf(x * x) * x)
    u = bf(GELU_C0 * bf(x + bf(GELU_C1 * x3)))
    return bf(x * bf(1 / bf(1 + bf(torch.exp(-2 * u))))).to(BF)


def attn(q, k, v):
    """q, k, v (1, H, N, hd) bf16 -> (1, N, H*hd). GEMMs take bf16 inputs, accumulate fp32, output bf16."""
    scale = const(1 / math.sqrt(q.shape[-1]))
    s = bf(bf(q.float() @ k.float().transpose(-1, -2)) * scale)
    p = bf(torch.exp(bf(s - s.amax(-1, keepdim=True))))
    p = bf(p * bf(1 / p.sum(-1, keepdim=True)))  # fp32 row sum, bf16 reciprocal-multiply
    o = bf(p @ v.float()).to(BF)
    return o.transpose(1, 2).reshape(1, q.shape[2], -1)


KC = 48  # keys per chunk of the NPU kernel's online softmax


def attn_bfp(q, k, v):
    """attn15.cc: Q, K, V, P as bfp16ebs8 (blocks of 8 along each GEMM's K), fp32 accumulation, online softmax over
    48-key chunks with a bf16 running max, row sums of exactly the bfp P that PV multiplies, one bf16 reciprocal
    of the sum applied after PV. Same in/out as attn."""
    scale = const(1 / math.sqrt(q.shape[-1]))
    n = k.shape[2]
    pad = -n % KC
    qb, kb = bfp_quantize(q), bfp_quantize(F.pad(k.float(), (0, 0, 0, pad)))
    vt = bfp_quantize(F.pad(v.float(), (0, 0, 0, pad)).transpose(-1, -2))  # (1, H, hd, keys): blocks along keys
    m = torch.full((*q.shape[:3], 1), -64.0, device=q.device)  # the kernel's "minus infinity"
    o = torch.zeros(q.shape, device=q.device)
    total = torch.zeros_like(m)
    for c in range(0, n + pad, KC):
        s = bf(bf(qb @ kb[..., c : c + KC, :].transpose(-1, -2)) * scale)
        s[..., max(0, n - c) :] = float("-inf")  # padding keys: out of the max, P = 0
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        f = bf(torch.exp(bf(m - m_new).clamp(min=-40)))  # rescale of O and the sums when the max grew
        o, total = o * f, total * f
        p = bf(torch.exp(bf(s - m_new).clamp(min=-40)))
        p[..., max(0, n - c) :] = 0
        pb = bfp_quantize(p)
        o = o + pb @ vt[..., c : c + KC].transpose(-1, -2)
        total = total + pb.sum(-1, keepdim=True)
        m = m_new
    o = bf(o * bf(1 / bf(total))).to(BF)
    return o.transpose(1, 2).reshape(1, q.shape[2], -1)


def _ln(x, wb):
    return ln(x, wb) if "ln" in OPS else md2fast._ln(x, wb)


def _mlp(fc1, fc2, h, res=None):
    act = gelu if "gelu" in OPS else gelu_sig if "gelusig" in OPS else None
    if act is None:
        return md2fast.mlp(fc1, fc2, h, res=res)
    out = fc2(act(fc1(h)))
    return out if res is None else out + res


def vision(self, img_u8):
    """md2fast.FastMD2._vision with the OPS swapped in."""
    x = img_u8.permute(2, 0, 1)[None].to(self.dtype).div_(255.0).sub_(0.5).div_(0.5)
    x = x.reshape(1, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(1, NTOK, 3 * PATCH * PATCH)
    x = self.patch_emb(x) + self.pos_emb
    H = self.vh
    for b in self.vblocks:
        h = _ln(x, b["ln1"])
        q, k, v = [t.view(1, NTOK, H, -1).transpose(1, 2) for t in b["qkv"](h).chunk(3, dim=-1)]
        if "attnbfp" in OPS:
            a = attn_bfp(q, k, v)
        elif "softmax" in OPS:
            a = attn(q, k, v)
        else:
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, NTOK, -1)
        x = lin_res(b["proj"], a, x)
        x = _mlp(b["fc1"], b["fc2"], _ln(x, b["ln2"]), res=x)
    x = _ln(x, self.post_ln)[0]
    feats = x if self.fold_proj else torch.cat([x, x], dim=-1)
    return _mlp(self.proj_fc1, self.proj_fc2, feats)


def enable(ops: str):
    """ops: comma list of ALL, or "all"."""
    OPS.clear()
    OPS.update(ALL if ops == "all" else ops.split(","))
    assert OPS <= set(ALL + VARIANTS) and not {"gelu", "gelusig"} <= OPS and not {"softmax", "attnbfp"} <= OPS, OPS
    md2fast.FastMD2._vision = vision
