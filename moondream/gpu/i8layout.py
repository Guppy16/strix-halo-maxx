import torch, triton, triton.language as tl, time, statistics
@triton.jit
def k_nk(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # B stored (N, K) row-major; load (BN, BK) tile and transpose in-register
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BM + tl.arange(0,BM); rn = pid_n*BN + tl.arange(0,BN); rk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN), dtype=tl.int32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:,None]*K + (rk[None,:]+k), mask=rm[:,None]<M, other=0)
        b = tl.load(B + rn[:,None]*K + (rk[None,:]+k), mask=rn[:,None]<N, other=0)
        acc += tl.dot(a, tl.trans(b))
    tl.store(C + rm[:,None]*N + rn[None,:], acc, mask=(rm[:,None]<M)&(rn[None,:]<N))
@triton.jit
def k_kn(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BM + tl.arange(0,BM); rn = pid_n*BN + tl.arange(0,BN); rk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN), dtype=tl.int32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:,None]*K + (rk[None,:]+k), mask=rm[:,None]<M, other=0)
        b = tl.load(B + (rk[:,None]+k)*N + rn[None,:], mask=rn[None,:]<N, other=0)
        acc += tl.dot(a, b)
    tl.store(C + rm[:,None]*N + rn[None,:], acc, mask=(rm[:,None]<M)&(rn[None,:]<N))
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    return statistics.median(ts)*1e6
for M,K,N in [(730,2048,8192),(730,8192,2048),(730,2048,6144)]:
    a = torch.randint(-127,127,(M,K),device="cuda",dtype=torch.int8); bnk = torch.randint(-127,127,(N,K),device="cuda",dtype=torch.int8); bkn = bnk.t().contiguous(); c = torch.empty(M,N,device="cuda",dtype=torch.int32)
    best = {}
    for BM,BN,BK,nw,ns in [(64,128,64,4,2),(128,128,64,4,2),(64,128,128,4,2),(128,128,128,4,2),(64,64,128,4,2),(128,128,64,4,1),(64,128,64,4,1),(64,128,64,4,3),(128,128,64,8,2),(64,256,64,4,2),(128,256,64,8,2),(64,128,256,4,2),(128,128,128,8,2)]:
        for name,kern,b in [("nk",k_nk,bnk),("kn",k_kn,bkn)]:
            try:
                grid=(triton.cdiv(M,BM),triton.cdiv(N,BN))
                f=lambda: kern[grid](a,b,c,M,N,K,BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
                us=bench(f); best[name]=min(best.get(name,1e9),us)
                print(f"{M}x{K}x{N} {name} BM{BM} BN{BN} BK{BK} w{nw} s{ns}: {us:7.1f} us {2*M*K*N/us/1e6:5.1f} TOPS", flush=True)
            except Exception as e: print(name,BM,BN,BK,nw,ns,"fail",repr(e)[:80])
    print("BEST", M,K,N, best, flush=True)
