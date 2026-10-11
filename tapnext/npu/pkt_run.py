"""Run a pkt_test build: checks both routes' xor folds and times the dispatch."""

import json
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/xilinx/xrt/python")
import pyxrt  # noqa: E402

b = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cfg = json.load(open(b + ".json"))
n_rep, W, OUT = cfg["n_rep"], cfg["W"], cfg["OUT"]
rng = np.random.default_rng(1)
x = rng.integers(0, 2**32, n_rep * W, dtype=np.uint32)
e = rng.integers(0, 2**32, n_rep * W, dtype=np.uint32)


def fold(v):
    return np.bitwise_xor.reduce(v.reshape(-1, OUT), axis=0)


dev = pyxrt.device(0)
xb = pyxrt.xclbin(b + ".xclbin")
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(b + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_x = pyxrt.bo(dev, x.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_e = pyxrt.bo(dev, e.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_y = pyxrt.bo(dev, OUT * 4, pyxrt.bo.host_only, kern.group_id(5))
bo_z = pyxrt.bo(dev, OUT * 4, pyxrt.bo.host_only, kern.group_id(6))
bo_i.write(instr.tobytes(), 0)
bo_x.write(x.tobytes(), 0)
bo_e.write(e.tobytes(), 0)
for q in (bo_i, bo_x, bo_e):
    q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
ts = []
for it in range(iters):
    t0 = time.perf_counter()
    r = kern(3, bo_i, len(instr), bo_x, bo_e, bo_y, bo_z)
    st = r.wait(3_000_000)
    ts.append(time.perf_counter() - t0)
    if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
        print("RUN FAILED", st, f"iteration {it}")
        raise SystemExit(1)
for q in (bo_y, bo_z):
    q.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
yv = np.frombuffer(bo_y.read(OUT * 4, 0), dtype=np.uint32)
zv = np.frombuffer(bo_z.read(OUT * 4, 0), dtype=np.uint32)
print(f"route A (slot chain) {'OK' if np.array_equal(yv, fold(x)) else 'MISMATCH'}; route E (packet fifo) {'OK' if np.array_equal(zv, fold(e)) else 'MISMATCH'};"
      f" dispatch us median {1e6 * np.median(ts):.0f} min {1e6 * min(ts):.0f} (n_rep {n_rep}, mode {cfg['mode']})")
