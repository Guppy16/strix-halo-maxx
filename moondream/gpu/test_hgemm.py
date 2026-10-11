"""Regression test 1 (kernel level): HIP WMMA GEMMs (hgemm.py) vs the Triton GEMMs (tgemm.py) and vs torch F.linear on random
inputs for every GEMM shape of the moondream2 fast path; plus timing and a HIP-graph capture check.  Run on the host venv-rocm714
or inside the container (same code).  Usage: python test_hgemm.py [--quick]"""
import sys, os, time, statistics, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgemm, hgemm
quick = "--quick" in sys.argv
torch.manual_seed(0)
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6
text = [(730, 2048, 6144), (730, 2048, 2048), (730, 2048, 8192), (730, 8192, 2048), (32, 2048, 6144), (32, 2048, 2048), (32, 2048, 8192), (32, 8192, 2048), (64, 2048, 8192)]
vis = [(729, 1152, 3456), (729, 1152, 1152), (729, 1152, 4304), (729, 4304, 1152), (729, 2304, 8192), (729, 8192, 2048), (729, 588, 1152)]
ok_all = True
print("== int8 W8A8 (text model): HIP vs Triton (same quantised operands -> should match to bf16 rounding) and vs bf16 F.linear")
for M, K, N in text:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02; b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, w, b)
    t = tgemm.I8Lin(w, b, torch.bfloat16); h = hgemm.HI8Lin(w, b, torch.bfloat16)
    for gelu in (False, True):
        ot = t(x, gelu=gelu); oh = h(x, gelu=gelu)
        r = F.gelu(ref, approximate="tanh") if gelu else ref
        d_ht = (oh.float() - ot.float()).abs().max().item(); rel_ht = ((oh.float() - ot.float()).norm() / ot.float().norm()).item()
        rel_h = ((oh.float() - r.float()).norm() / r.float().norm()).item(); rel_t = ((ot.float() - r.float()).norm() / r.float().norm()).item()
        ok = rel_ht < 1e-2 and abs(rel_h - rel_t) < 2e-3
        ok_all &= ok
        if not quick or not gelu:
            ut = bench(lambda: t(x, gelu=gelu)); uh = bench(lambda: h(x, gelu=gelu)); u0 = bench(lambda: F.linear(x, w, b))
            fl = 2 * M * K * N / 1e6
            print(f"{M}x{K}x{N} gelu={int(gelu)}: HIP-vs-Triton maxabs={d_ht:.4f} rel={rel_ht:.5f} | rel err vs bf16 ref: HIP {rel_h:.4f} Triton {rel_t:.4f} | "
                  f"torch {u0:7.1f}us ({fl/u0:4.1f}T) triton-i8 {ut:7.1f}us ({fl/ut:4.1f}T) HIP-i8 {uh:7.1f}us ({fl/uh:4.1f}T) cfg={h.cfgs[M]} {'OK' if ok else 'FAIL'}", flush=True)
print("== bf16 (vision encoder): HIP vs Triton bf16 vs F.linear")
for M, K, N in vis:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02; b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x.float(), w.float(), b.float())
    t = tgemm.TLin(w, b, torch.bfloat16); h = hgemm.HLin(w, b, torch.bfloat16)
    for gelu in (False, True):
        # the fast path applies F.gelu to the bf16 output of TLin (TLin.fused_gelu=False); HLin fuses it but rounds to bf16 first -> must be bit-identical
        ot = F.gelu(t(x), approximate="tanh") if gelu else t(x); oh = h(x, gelu=gelu)
        r = F.gelu(ref, approximate="tanh") if gelu else ref
        rel_h = ((oh.float() - r).norm() / r.norm()).item(); rel_t = ((ot.float() - r).norm() / r.norm()).item(); d = (oh.float() - ot.float()).abs().max().item()
        ok = rel_h < 5e-3 and abs(rel_h - rel_t) < 1e-3
        ok_all &= ok
        if not quick or not gelu:
            ut = bench(lambda: (F.gelu(t(x), approximate="tanh") if gelu else t(x))); uh = bench(lambda: h(x, gelu=gelu)); u0 = bench(lambda: F.linear(x, w, b))
            fl = 2 * M * K * N / 1e6
            print(f"{M}x{K}x{N} gelu={int(gelu)}: HIP-vs-Triton maxabs={d:.4f} | rel err vs fp32 ref: HIP {rel_h:.5f} Triton {rel_t:.5f} | torch {u0:7.1f}us ({fl/u0:4.1f}T) triton {ut:7.1f}us ({fl/ut:4.1f}T) HIP {uh:7.1f}us ({fl/uh:4.1f}T) cfg={h.cfgs.get(M)} {'OK' if ok else 'FAIL'}", flush=True)
print("== HIP graph capture of the ctypes-launched kernels")
M, K, N = 730, 2048, 6144
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02; b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
h = hgemm.HI8Lin(w, b, torch.bfloat16); hb = hgemm.HLin(w, b, torch.bfloat16)
xin = x.clone(); out = {}
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(2): h(xin); hb(xin)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out["i8"] = h(xin); out["bf"] = hb(xin)
x2 = torch.randn_like(x); xin.copy_(x2); g.replay(); torch.cuda.synchronize()
e1 = (out["i8"].float() - h(x2).float()).abs().max().item(); e2 = (out["bf"].float() - hb(x2).float()).abs().max().item()
ug = bench(lambda: g.replay())
print(f"graph replay matches eager: int8 maxabs={e1} bf16 maxabs={e2} | replay of both GEMMs {ug:.1f} us  {'OK' if e1 == 0 and e2 == 0 else 'FAIL'}")
ok_all &= (e1 == 0 and e2 == 0)
print("ALL OK" if ok_all else "SOME CHECKS FAILED")
