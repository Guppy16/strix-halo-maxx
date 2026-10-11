"""Fused rope + KV-cache write kernel for the moondream2 text model (head_dim 64, rot_dim 32, non-interleaved
input halves, interleaved output -- replicates rope.apply_rotary_emb exactly)."""
import torch, triton, triton.language as tl

@triton.jit
def _rope_cache(QKV, COS, SIN, Q, KC, VC, L, pos0, H: tl.constexpr, HD: tl.constexpr, CTX: tl.constexpr, NQ: tl.constexpr, BCAST: tl.constexpr):
    r = tl.program_id(0)
    b = r // L; t = r % L
    base = QKV + r * (3 * H * HD)
    j = tl.arange(0, 16); h = tl.arange(0, H); p = tl.arange(0, 32); d = tl.arange(0, HD)
    cos = tl.load(COS + t * 16 + j)[None, :]; sin = tl.load(SIN + t * 16 + j)[None, :]
    hoff = h[:, None] * HD
    # ---- q  (out layout: (B, H, L, HD))
    qa = tl.load(base + hoff + j[None, :]).to(tl.float32); qb = tl.load(base + hoff + 16 + j[None, :]).to(tl.float32)
    qp = tl.load(base + hoff + 32 + p[None, :])
    qo = Q + ((b * H + h[:, None]) * L + t) * HD
    tl.store(qo + 2 * j[None, :], (qa * cos - qb * sin).to(Q.dtype.element_ty))
    tl.store(qo + 2 * j[None, :] + 1, (qa * sin + qb * cos).to(Q.dtype.element_ty))
    tl.store(qo + 32 + p[None, :], qp)
    # ---- k / v (cache layout: (NQ, H, CTX, HD))
    kb0 = base + H * HD
    ka = tl.load(kb0 + hoff + j[None, :]).to(tl.float32); kb = tl.load(kb0 + hoff + 16 + j[None, :]).to(tl.float32)
    kp = tl.load(kb0 + hoff + 32 + p[None, :])
    kr = (ka * cos - kb * sin).to(KC.dtype.element_ty); ki = (ka * sin + kb * cos).to(KC.dtype.element_ty)
    v = tl.load(base + 2 * H * HD + hoff + d[None, :])
    if BCAST:
        for bb in tl.static_range(NQ):
            ko = KC + ((bb * H + h[:, None]) * CTX + pos0 + t) * HD
            tl.store(ko + 2 * j[None, :], kr); tl.store(ko + 2 * j[None, :] + 1, ki); tl.store(ko + 32 + p[None, :], kp)
            tl.store(VC + ((bb * H + h[:, None]) * CTX + pos0 + t) * HD + d[None, :], v)
    else:
        ko = KC + ((b * H + h[:, None]) * CTX + pos0 + t) * HD
        tl.store(ko + 2 * j[None, :], kr); tl.store(ko + 2 * j[None, :] + 1, ki); tl.store(ko + 32 + p[None, :], kp)
        tl.store(VC + ((b * H + h[:, None]) * CTX + pos0 + t) * HD + d[None, :], v)

def rope_cache(qkv2d, cos, sin, q_out, kc, vc, pos0, bcast=True):
    """qkv2d: (B*L, 3*H*HD) contiguous; cos/sin: (L,16) fp32; q_out: (B,H,L,HD) contiguous; kc/vc: (NQ,H,CTX,HD).
    bcast=True writes the same k/v into every cache slot (image prefill, B=1); bcast=False writes row-batch b into slot b."""
    B, H, L, HD = q_out.shape; NQ, _, CTX, _ = kc.shape
    _rope_cache[(B * L,)](qkv2d, cos, sin, q_out, kc, vc, L, pos0, H=H, HD=HD, CTX=CTX, NQ=NQ, BCAST=bcast, num_warps=4)
    return q_out

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    H, HD, CTX, n, pos0 = 32, 64, 762, 730, 0
    qkv = torch.randn(n, 3 * H * HD, device="cuda", dtype=torch.bfloat16)
    fc = torch.randn(CTX, 16, 2, device="cuda")
    cos = fc[pos0:pos0 + n, :, 0].contiguous(); sin = fc[pos0:pos0 + n, :, 1].contiguous()
    kc = torch.zeros(1, H, CTX, HD, device="cuda", dtype=torch.bfloat16); vc = torch.zeros_like(kc); q = torch.empty(1, H, n, HD, device="cuda", dtype=torch.bfloat16)
    rope_cache(qkv, cos, sin, q, kc, vc, pos0)
    # reference (from the model's rope.py)
    def ref_rope(x, cos, sin):
        xr, xp = x[..., :32], x[..., 32:]
        a, b = xr[..., :16], xr[..., 16:]
        c = cos[None, None]; s = sin[None, None]
        o = torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)
        return torch.cat([o.to(x.dtype), xp], dim=-1)
    qq, kk, vv = qkv.split([H * HD] * 3, dim=-1)
    qr = ref_rope(qq.view(1, n, H, HD).transpose(1, 2), cos, sin); kr = ref_rope(kk.view(1, n, H, HD).transpose(1, 2), cos, sin)
    print("q err", (q - qr).abs().max().item(), "k err", (kc[:, :, pos0:pos0 + n] - kr).abs().max().item(), "v err", (vc[:, :, pos0:pos0 + n] - vv.view(1, n, H, HD).transpose(1, 2)).abs().max().item())
    import time, statistics
    def bench(fn, k=50):
        for _ in range(3): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(k):
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return statistics.median(ts) * 1e6
    print("fused rope+cache us:", bench(lambda: rope_cache(qkv, cos, sin, q, kc, vc, pos0)))
