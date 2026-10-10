"""Run an ln.py build: LayerNorm of 768x1152 bf16 vs an fp32 reference (torch's F.layer_norm numerics:
fp32 mean / biased variance / rsqrt(var + eps), bf16 out RNE)."""

import json
import sys
import time

import numpy as np
import pyxrt
from ml_dtypes import bfloat16

b = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cfg = json.load(open(b + ".json"))
D, M = cfg["D"], 32 * cfg["rows_per_core"]
reps = cfg["reps"]
rng = np.random.default_rng(0)
x = (rng.standard_normal((M, D)) * 2 + 0.3).astype(bfloat16)
res = cfg.get("res", 0)
if res:
    delta = (rng.standard_normal((M, D)) * 0.5).astype(bfloat16)
    xs = (x.astype(np.float32) + delta.astype(np.float32)).astype(bfloat16)  # bf16 residual add
else:
    xs = x
gamma = (1 + 0.1 * rng.standard_normal(D)).astype(bfloat16)
beta = (0.1 * rng.standard_normal(D)).astype(bfloat16)
xf = xs.astype(np.float32)
mean = xf.mean(axis=1, keepdims=True, dtype=np.float32)
var = ((xf - mean) ** 2).mean(axis=1, keepdims=True, dtype=np.float32)
rb = lambda v: np.asarray(v, np.float32).astype(bfloat16).astype(np.float32)
mean_b = rb(mean)
d = rb(xf - mean_b)
var = (d * d).mean(axis=1, keepdims=True, dtype=np.float32)  # variance of the bf16-rounded differences, fp32
rstd = rb(1.0 / np.sqrt(var + np.float32(1e-5)))
ref = rb(rb(rb(d * rstd) * gamma.astype(np.float32)) + beta.astype(np.float32))

dev = pyxrt.device(0)
xb = pyxrt.xclbin(b + ".xclbin")
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(b + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
xin = np.concatenate([x, delta]) if res else x
bo_x = pyxrt.bo(dev, xin.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_p = pyxrt.bo(dev, 4 * D, pyxrt.bo.host_only, kern.group_id(4))
bo_y = pyxrt.bo(dev, xin.nbytes, pyxrt.bo.host_only, kern.group_id(5))
bo_i.write(instr.tobytes(), 0); bo_x.write(xin.tobytes(), 0); bo_p.write(np.concatenate([gamma, beta]).tobytes(), 0)
for q in (bo_i, bo_x, bo_p, bo_y):
    q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
ts = []
for it in range(iters + 3):
    t0 = time.perf_counter()
    r = kern(3, bo_i, len(instr), bo_x, bo_p, bo_y)
    st = r.wait(3000000) if it == 0 else r.wait()
    ts.append(time.perf_counter() - t0)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
yall = np.frombuffer(bo_y.read(xin.nbytes, 0), dtype=bfloat16)
y = yall[: M * D].reshape(M, D).astype(np.float32)
if res:
    xo = yall[M * D :].reshape(M, D)
    print(f"residual out x' exact: {np.array_equal(xo, xs)}")
exact = ref.astype(bfloat16).astype(np.float32)
err = np.abs(y - ref)
print(f"LN {M}x{D} reps {reps}: us median {1e6 * np.median(ts[3:]):.0f} min {1e6 * min(ts[3:]):.0f} | mismatch_vs_bf16(ref) {np.mean(y != exact):.4%}"
      f" max_abs {err.max():.3e} rms_rel {np.sqrt((err**2).sum() / (ref**2).sum()):.2e} bf16_floor {np.sqrt(((exact - ref)**2).sum() / (ref**2).sum()):.2e}")
