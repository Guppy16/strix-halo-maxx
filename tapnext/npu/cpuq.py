import time, torch
torch.set_num_threads(8)
Q = 48
shapes = [(768, 1536), (768, 768), (768, 6144), (3072, 768), (768, 2304), (768, 768), (768, 3072), (3072, 768)]
for dt in (torch.bfloat16, torch.float32):
    ws = [torch.randn(n, k, dtype=dt) for k, n in shapes]
    xs = [torch.randn(Q, k, dtype=dt) for k, n in shapes]
    for _ in range(3):
        for x, w in zip(xs, ws): torch.nn.functional.linear(x, w)
    t0 = time.perf_counter(); n = 20
    for _ in range(n):
        for x, w in zip(xs, ws): torch.nn.functional.linear(x, w)
    ms = (time.perf_counter() - t0) * 1e3 / n
    print(f"CPU {dt}: 48 query rows through one block's 8 GEMMs: {ms:.2f} ms -> x12 blocks = {12*ms:.1f} ms/step (8 threads)")
