import torch, time, statistics
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    return statistics.median(ts)*1e6
for M,K,N in [(730,2048,8192),(730,8192,2048),(32,2048,8192),(729,1152,4304)]:
    a = torch.randint(-127,127,(M,K),device="cuda",dtype=torch.int8); b = torch.randint(-127,127,(K,N),device="cuda",dtype=torch.int8)
    try:
        c = torch._int_mm(a, b); torch.cuda.synchronize()
        us = bench(lambda: torch._int_mm(a, b)); print(f"_int_mm {M}x{K}x{N}: {us:.1f} us {2*M*K*N/us/1e6:.1f} TOPS ok={c.dtype}")
    except Exception as e: print("_int_mm failed:", repr(e)[:200])
    try:
        ab = torch.randn(M,K,device="cuda",dtype=torch.bfloat16); bb = torch.randn(K,N,device="cuda",dtype=torch.bfloat16)
        us = bench(lambda: ab @ bb); print(f"bf16 mm {M}x{K}x{N}: {us:.1f} us {2*M*K*N/us/1e6:.1f} TFLOPS")
    except Exception as e: print("bf16 failed", e)
    try:
        af = torch.randn(M,K,device="cuda").to(torch.float8_e4m3fn); bf = torch.randn(N,K,device="cuda").to(torch.float8_e4m3fn).t()
        sa = torch.ones((),device="cuda"); sb = torch.ones((),device="cuda")
        c = torch._scaled_mm(af, bf, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16); torch.cuda.synchronize()
        us = bench(lambda: torch._scaled_mm(af, bf, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)); print(f"fp8 _scaled_mm {M}x{K}x{N}: {us:.1f} us {2*M*K*N/us/1e6:.1f} TFLOPS")
    except Exception as e: print("fp8 failed:", repr(e)[:300])
# triton int8 dot
import triton, triton.language as tl
@triton.jit
def k_i8(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BM + tl.arange(0,BM); rn = pid_n*BN + tl.arange(0,BN); rk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN), dtype=tl.int32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:,None]*sam + (rk[None,:]+k)*sak, mask=rm[:,None]<M, other=0)
        b = tl.load(B + (rk[:,None]+k)*sbk + rn[None,:]*sbn, mask=rn[None,:]<N, other=0)
        acc += tl.dot(a, b)
    tl.store(C + rm[:,None]*scm + rn[None,:]*scn, acc, mask=(rm[:,None]<M)&(rn[None,:]<N))
@triton.jit
def k_f16(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BM + tl.arange(0,BM); rn = pid_n*BN + tl.arange(0,BN); rk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:,None]*sam + (rk[None,:]+k)*sak, mask=rm[:,None]<M, other=0)
        b = tl.load(B + (rk[:,None]+k)*sbk + rn[None,:]*sbn, mask=rn[None,:]<N, other=0)
        acc += tl.dot(a, b)
    tl.store(C + rm[:,None]*scm + rn[None,:]*scn, acc.to(tl.bfloat16), mask=(rm[:,None]<M)&(rn[None,:]<N))
for M,K,N in [(730,2048,8192),(730,8192,2048)]:
    for BM,BN,BK,nw in [(64,64,64,4),(128,128,64,4),(64,128,64,4),(128,64,32,4),(128,128,32,8)]:
        try:
            a = torch.randint(-127,127,(M,K),device="cuda",dtype=torch.int8); b = torch.randint(-127,127,(K,N),device="cuda",dtype=torch.int8); c = torch.empty(M,N,device="cuda",dtype=torch.int32)
            grid = (triton.cdiv(M,BM), triton.cdiv(N,BN))
            f = lambda: k_i8[grid](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw)
            f(); torch.cuda.synchronize(); ref = torch._int_mm(a,b) if False else None
            us = bench(f); print(f"triton int8 {M}x{K}x{N} BM{BM} BN{BN} BK{BK} w{nw}: {us:.1f} us {2*M*K*N/us/1e6:.1f} TOPS")
        except Exception as e: print(f"triton int8 BM{BM} BN{BN} BK{BK} failed:", repr(e)[:200])
        try:
            a = torch.randn(M,K,device="cuda",dtype=torch.bfloat16); b = torch.randn(K,N,device="cuda",dtype=torch.bfloat16); c = torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
            grid = (triton.cdiv(M,BM), triton.cdiv(N,BN))
            f = lambda: k_f16[grid](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw)
            f(); torch.cuda.synchronize(); err = (c.float() - (a.float()@b.float())).abs().max().item()
            us = bench(f); print(f"triton bf16 {M}x{K}x{N} BM{BM} BN{BN} BK{BK} w{nw}: {us:.1f} us {2*M*K*N/us/1e6:.1f} TFLOPS err={err:.3f}")
        except Exception as e: print(f"triton bf16 BM{BM} BN{BN} BK{BK} failed:", repr(e)[:200])
