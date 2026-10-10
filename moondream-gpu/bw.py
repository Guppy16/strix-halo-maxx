import torch, time, statistics
def bench(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    return statistics.median(ts)
for mb in (32, 128, 512, 2048):
    x = torch.empty(mb*2**20, device="cuda", dtype=torch.uint8); y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x)); print(f"copy {mb} MB: {2*mb/1024/t:.0f} GB/s (read+write)")
    xs = x.view(torch.int32) if False else x
    t = bench(lambda: torch.sum(x.view(torch.float16).float() if False else x.view(torch.bfloat16))); print(f"sum-read {mb} MB: {mb/1024/t:.0f} GB/s")
