"""Triton GEMM kernels for moondream2 linears on gfx1151: bf16/fp16 with fused bias(+gelu),
and int8 W8A8 (per-token dynamic activation scale, per-channel weight scale) with fused dequant epilogue."""
import torch, triton, triton.language as tl

import os
WIDE = os.environ.get("TGEMM_WIDE", "0") == "1"

def _cfgs():
    base = [(128, 128, 64, 4, 2), (128, 128, 32, 4, 2), (64, 128, 64, 4, 2), (128, 64, 64, 4, 2), (64, 64, 64, 4, 2),
            (128, 128, 64, 8, 2), (64, 128, 32, 4, 2), (128, 256, 64, 8, 2), (256, 128, 64, 8, 2), (32, 128, 64, 4, 2), (16, 128, 64, 4, 2), (16, 64, 128, 4, 2)]
    if WIDE:
        base += [(128, 128, 128, 4, 2), (64, 128, 128, 4, 2), (128, 64, 128, 4, 2), (64, 64, 128, 4, 2), (256, 64, 64, 4, 2), (64, 256, 64, 4, 2),
                 (128, 128, 64, 4, 1), (64, 128, 64, 4, 1), (128, 128, 64, 4, 3), (64, 64, 32, 2, 2), (32, 64, 64, 2, 2), (16, 256, 64, 4, 2), (32, 256, 64, 4, 2), (16, 128, 128, 4, 2), (32, 128, 128, 4, 2)]
    out = []
    for bm, bn, bk, w, s in base:
        out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GM": 8}, num_warps=w, num_stages=s))
        if WIDE:
            out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GM": 8, "waves_per_eu": 2}, num_warps=w, num_stages=s))
    return out

@triton.jit
def _gelu_tanh(x):
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * u))  # tanh(u)
    return 0.5 * x * (1.0 + t)

@triton.autotune(configs=_cfgs(), key=["M", "N", "K"], cache_results=True)
@triton.jit
def _gemm_f16(A, B, Bias, R, C, M, N, K, sam, sak, sbk, sbn, scm, scn, HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM); num_n = tl.cdiv(N, BN)
    group = GM * num_n
    gid = pid // group; first_m = gid * GM; gsz = min(num_m - first_m, GM)
    pid_m = first_m + (pid % group) % gsz; pid_n = (pid % group) // gsz
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = A + rm[:, None] * sam + rk[None, :] * sak
    bp = B + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BK), other=0.0)
        b = tl.load(bp, mask=(rk[:, None] < K - k * BK) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        ap += BK * sak; bp += BK * sbk
    if HAS_BIAS:
        acc += tl.load(Bias + rn, mask=rn < N, other=0.0)[None, :].to(tl.float32)
    if GELU:
        acc = _gelu_tanh(acc)
    if HAS_RES:
        acc += tl.load(R + rm[:, None] * scm + rn[None, :] * scn, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0).to(tl.float32)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.autotune(configs=_cfgs(), key=["M", "N", "K"], cache_results=True)
@triton.jit
def _gemm_i8(A, B, SA, SB, Bias, R, C, M, N, K, sam, sak, sbk, sbn, scm, scn, HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM); num_n = tl.cdiv(N, BN)
    group = GM * num_n
    gid = pid // group; first_m = gid * GM; gsz = min(num_m - first_m, GM)
    pid_m = first_m + (pid % group) % gsz; pid_n = (pid % group) // gsz
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = A + rm[:, None] * sam + rk[None, :] * sak
    bp = B + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BK), other=0)
        b = tl.load(bp, mask=(rk[:, None] < K - k * BK) & (rn[None, :] < N), other=0)
        acc += tl.dot(a, b)
        ap += BK * sak; bp += BK * sbk
    sa = tl.load(SA + rm, mask=rm < M, other=0.0); sb = tl.load(SB + rn, mask=rn < N, other=0.0)
    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        out += tl.load(Bias + rn, mask=rn < N, other=0.0)[None, :].to(tl.float32)
    if GELU:
        out = _gelu_tanh(out)
    if HAS_RES:
        out += tl.load(R + rm[:, None] * scm + rn[None, :] * scn, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0).to(tl.float32)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, out.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def _quant_rows(X, Q, S, N, sxm, sqm, BN: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BN)
    x = tl.load(X + row * sxm + cols, mask=cols < N, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    s = tl.maximum(amax, 1e-5) / 127.0
    v = x / s
    q = (v + tl.where(v >= 0, 0.5, -0.5)).to(tl.int32)
    tl.store(Q + row * sqm + cols, q.to(tl.int8), mask=cols < N)
    tl.store(S + row, s)

def quant_rows(x2d):
    M, K = x2d.shape
    q = torch.empty(M, K, device=x2d.device, dtype=torch.int8); s = torch.empty(M, device=x2d.device, dtype=torch.float32)
    _quant_rows[(M,)](x2d, q, s, K, x2d.stride(0), K, BN=triton.next_power_of_2(K), num_warps=8)
    return q, s

class TLin:
    """F.linear replacement: bf16/fp16 triton GEMM, optional fused gelu / residual."""
    fused_gelu = False
    fused_res = True
    def __init__(self, weight, bias, dtype):
        self.wt = weight.detach().to(dtype).t().contiguous()  # (K, N)
        self.b = None if bias is None else bias.detach().to(dtype).contiguous()
        self.N, self.K = weight.shape

    def __call__(self, x, gelu=False, res=None):
        shp = x.shape; x2 = x.reshape(-1, self.K)
        M = x2.shape[0]
        c = torch.empty(M, self.N, device=x.device, dtype=x.dtype)
        r = res.reshape(M, self.N) if res is not None else c
        grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(self.N, meta["BN"]),)
        _gemm_f16[grid](x2, self.wt, self.b if self.b is not None else x2, r, c, M, self.N, self.K, x2.stride(0), x2.stride(1), self.wt.stride(0), self.wt.stride(1), c.stride(0), c.stride(1),
                        HAS_BIAS=self.b is not None, GELU=gelu, HAS_RES=res is not None)
        return c.reshape(*shp[:-1], self.N)

class I8Lin:
    """W8A8: per-channel int8 weights, per-token dynamic int8 activations."""
    fused_gelu = True
    fused_res = True
    def __init__(self, weight, bias, dtype):
        w = weight.detach().float()
        self.N, self.K = w.shape
        s = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        self.wq = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8).t().contiguous()  # (K, N)
        self.sw = s.contiguous()
        self.b = None if bias is None else bias.detach().to(dtype).contiguous()
        self.dtype = dtype

    def quant(self, x):
        """pre-quantise an activation once so several I8Lin layers (qkv + fc1 share the LN output) can reuse it"""
        return quant_rows(x.reshape(-1, self.K))

    def __call__(self, x, gelu=False, res=None, pre=None):
        shp = x.shape; x2 = x.reshape(-1, self.K)
        M = x2.shape[0]
        q, sx = pre if pre is not None else quant_rows(x2)
        c = torch.empty(M, self.N, device=x.device, dtype=x.dtype)
        r = res.reshape(M, self.N) if res is not None else c
        grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(self.N, meta["BN"]),)
        _gemm_i8[grid](q, self.wq, sx, self.sw, self.b if self.b is not None else sx, r, c, M, self.N, self.K, q.stride(0), q.stride(1), self.wq.stride(0), self.wq.stride(1), c.stride(0), c.stride(1),
                       HAS_BIAS=self.b is not None, GELU=gelu, HAS_RES=res is not None)
        return c.reshape(*shp[:-1], self.N)

if __name__ == "__main__":
    import time, statistics, torch.nn.functional as F
    def bench(fn, n=30):
        for _ in range(3): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return statistics.median(ts) * 1e6
    torch.manual_seed(0)
    for M, K, N in [(730, 2048, 6144), (730, 2048, 2048), (730, 2048, 8192), (730, 8192, 2048), (32, 2048, 8192), (32, 8192, 2048), (729, 1152, 4304), (729, 4304, 1152), (729, 2304, 8192)]:
        for dt in (torch.bfloat16, torch.float16):
            x = torch.randn(M, K, device="cuda", dtype=dt); w = torch.randn(N, K, device="cuda", dtype=dt) * 0.02; b = torch.randn(N, device="cuda", dtype=dt)
            ref = F.linear(x, w, b)
            t = TLin(w, b, dt); i8 = I8Lin(w, b, dt)
            o = t(x); oi = i8(x)
            e1 = (o.float() - ref.float()).abs().max().item(); e2 = ((oi.float() - ref.float()).norm() / ref.float().norm()).item()
            u0 = bench(lambda: F.linear(x, w, b)); u1 = bench(lambda: t(x)); u2 = bench(lambda: i8(x)); u3 = bench(lambda: t(x, gelu=True))
            fl = 2 * M * K * N / 1e6
            print(f"{M}x{K}x{N} {str(dt)[6:]}: torch {u0:7.1f}us ({fl/u0:5.1f}T) | triton {u1:7.1f}us ({fl/u1:5.1f}T) err={e1:.3f} | +gelu {u3:7.1f}us | int8 {u2:7.1f}us ({fl/u2:5.1f}T) relerr={e2:.4f}", flush=True)
