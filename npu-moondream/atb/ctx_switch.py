"""Cost of alternating between two GEMM xclbins (separate hw_contexts) vs re-running one.
Data content is irrelevant here (timing only)."""

import statistics
import time

import numpy as np
import pyxrt

B = "/work/atb/build/"
DESIGNS = {
    "attn_out": ("768x1152x1152_64x144_384", 768, 1152, 1152),
    "qkv": ("768x1152x3456_64x144_384", 768, 1152, 3456),
}
dev = pyxrt.device(0)
runs = {}
for name, (tag, M, K, N) in DESIGNS.items():
    xb = pyxrt.xclbin(B + tag + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kname = [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0]
    kern = pyxrt.kernel(ctx, kname)
    instr = np.fromfile(B + tag + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bos = [pyxrt.bo(dev, nb, pyxrt.bo.host_only, kern.group_id(g)) for g, nb in ((3, M * K * 2), (4, K * N * 9 // 8), (5, M * N * 2))]
    for b in bos:
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    runs[name] = (kern, bo_i, len(instr), bos, ctx)


def run(name):
    kern, bo_i, n, bos, _ = runs[name]
    t0 = time.perf_counter_ns()
    h = kern(3, bo_i, n, *bos)
    h.wait()
    return (time.perf_counter_ns() - t0) / 1e3


for _ in range(5):
    run("attn_out"), run("qkv")
for pattern in (["attn_out"] * 20, ["qkv"] * 20, ["attn_out", "qkv"] * 20):
    ts = {"attn_out": [], "qkv": []}
    for nm in pattern:
        ts[nm].append(run(nm))
    label = "alternating" if len(set(pattern)) > 1 else f"repeat {pattern[0]}"
    print(label, {k: round(statistics.median(v), 1) for k, v in ts.items() if v})
