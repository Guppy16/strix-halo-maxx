"""Cold-cache GEMM microbench on the host: cycles through R copies of (x, w) so the weights do not stay in the 32 MB MALL
(as in the real pipeline, where 1.2 GB of weights stream through per frame). Times Triton (tgemm) and HIP (hgemm) identically
with CUDA events over a graph-captured loop (no launch overhead).  usage: python coldbench.py [R] [cfg list for HIP, e.g. 17,27,31]"""
import sys, os, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgemm, hgemm
R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfgs = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
torch.manual_seed(0)
shapes = [(730, 2048, 6144), (730, 2048, 2048), (730, 2048, 8192), (730, 8192, 2048), (32, 2048, 6144), (32, 2048, 2048), (32, 2048, 8192), (32, 8192, 2048)]
if os.environ.get("BIGONLY"): shapes = shapes[:4]
def timed(fns, reps=3):
    """graph-capture one pass over all R copies, replay reps times, return us per call"""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fns: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns: f()
    g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / len(fns))
    return best
for M, K, N in shapes:
    xs = [torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for _ in range(R)]
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(R)]
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    tl = [tgemm.I8Lin(w, b, torch.bfloat16) for w in ws]; hl = [hgemm.HI8Lin(w, b, torch.bfloat16) for w in ws]
    pre = [hl[i].quant(xs[i]) for i in range(R)]  # same (row-padded) pre-quantised activations for both
    fl = 2 * M * K * N / 1e6
    ut = timed([lambda i=i: tl[i](xs[i], pre=pre[i]) for i in range(R)])
    line = f"{M}x{K}x{N} cold(R={R}, {R*N*K/2**20:.0f} MB of weights): triton-i8 {ut:7.1f}us ({fl/ut:4.1f}T, {N*K/ut/1e3:5.0f} GB/s)"
    hot = timed([lambda: tl[0](xs[0], pre=pre[0]) for _ in range(R)])
    line += f" [hot {hot:6.1f}us]"
    small = M <= hgemm.SMALL_M
    cl = cfgs or ([3, 4, 5, 6] if small else [17, 27, 31])
    for c in cl:
        for h in hl: h.cfgs[M] = c
        if hl[0]._run(c, pre[0][0], pre[0][1], torch.empty(M, N, device="cuda", dtype=torch.bfloat16), M, False) != 0: continue
        uh = timed([lambda i=i: hl[i](xs[i], pre=pre[i]) for i in range(R)])
        hoth = timed([lambda: hl[0](xs[0], pre=pre[0]) for _ in range(R)])
        line += f" | HIP cfg{c} {uh:7.1f}us ({fl/uh:4.1f}T) [hot {hoth:6.1f}]"
    print(line, flush=True)
    del xs, ws, tl, hl, pre; torch.cuda.empty_cache()
