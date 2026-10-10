"""Microbenchmark of the GEMM shapes moondream2 uses (M,K,N) across BLAS backends and dtypes.
usage: gemm_bench.py [default|hipblaslt|tunable|triton] [bf16|fp16] """
import sys, os, time, torch, statistics
backend = sys.argv[1] if len(sys.argv) > 1 else "default"
dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[sys.argv[2] if len(sys.argv) > 2 else "bf16"]
if backend == "hipblaslt":
    torch.backends.cuda.preferred_blas_library("hipblaslt")
if backend == "tunable":
    torch.cuda.tunable.enable(True); torch.cuda.tunable.tuning_enable(True)
    torch.cuda.tunable.set_filename(os.path.expanduser("~/moondream-opt/tunable_bench.csv"))
shapes = [  # (M, K, N, tag)
    (730, 2048, 6144, "txt qkv prefill"), (730, 2048, 2048, "txt proj prefill"), (730, 2048, 8192, "txt fc1 prefill"), (730, 8192, 2048, "txt fc2 prefill"),
    (14, 2048, 6144, "txt qkv M14"), (14, 2048, 2048, "txt proj M14"), (14, 2048, 8192, "txt fc1 M14"), (14, 8192, 2048, "txt fc2 M14"),
    (1, 2048, 6144, "txt qkv M1"), (1, 2048, 2048, "txt proj M1"), (1, 2048, 8192, "txt fc1 M1"), (1, 8192, 2048, "txt fc2 M1"), (1, 2048, 51200, "lm_head M1"),
    (1458, 1152, 3456, "vis qkv B2"), (1458, 1152, 1152, "vis proj B2"), (1458, 1152, 4304, "vis fc1 B2"), (1458, 4304, 1152, "vis fc2 B2"),
    (729, 1152, 3456, "vis qkv B1"), (729, 1152, 1152, "vis proj B1"), (729, 1152, 4304, "vis fc1 B1"), (729, 4304, 1152, "vis fc2 B1"),
    (729, 588, 1152, "patch_emb"), (729, 2304, 8192, "proj fc1"), (729, 8192, 2048, "proj fc2"),
]
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6
tot = 0
for M, K, N, tag in shapes:
    x = torch.randn(M, K, device="cuda", dtype=dt); w = torch.randn(N, K, device="cuda", dtype=dt); b = torch.randn(N, device="cuda", dtype=dt)
    if backend == "triton":
        import torch._inductor.config as ic
        ic.max_autotune_gemm_backends = "TRITON"
        f = torch.compile(lambda x, w, b: torch.nn.functional.linear(x, w, b), mode="max-autotune-no-cudagraphs", dynamic=False)
    else:
        f = lambda x, w, b: torch.nn.functional.linear(x, w, b)
    us = bench(lambda: f(x, w, b))
    if backend == "tunable":  # re-run after tuning
        us = bench(lambda: f(x, w, b))
    fl = 2 * M * K * N / us / 1e6; gbs = (M * K + K * N + M * N) * 2 / us / 1e3
    tot += us
    print(f"{tag:18s} M={M:5d} K={K:5d} N={N:5d}: {us:8.1f} us  {fl:6.1f} TFLOPS  {gbs:6.1f} GB/s", flush=True)
print(f"[{backend} {dt}] sum of all shapes: {tot/1000:.2f} ms")
if backend == "tunable": torch.cuda.tunable.write_file()
