"""Time a bw.py build: GB/s delivered from DDR (each stream's words counted once)."""

import json
import sys
import time

import numpy as np
import pyxrt

b = sys.argv[1]
cfg = json.load(open(b + ".json"))
tot = cfg["cols"] * cfg["streams"] * cfg["total"]
dev = pyxrt.device(0)
xb = pyxrt.xclbin(b + ".xclbin")
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(b + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_a = pyxrt.bo(dev, tot * 4, pyxrt.bo.host_only, kern.group_id(3))
wr = cfg.get("write", 0)
osz = cfg["total"] if wr else 16
bo_c = pyxrt.bo(dev, cfg["cols"] * osz * 4, pyxrt.bo.host_only, kern.group_id(4))
bo_i.write(instr.tobytes(), 0)
for x in (bo_i, bo_a, bo_c):
    x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
ts = []
for it in range(25):
    t0 = time.perf_counter()
    r = kern(3, bo_i, len(instr), bo_a, bo_c)
    r.wait()
    ts.append(time.perf_counter() - t0)
t = float(np.median(ts[5:]))
print(f"BW {cfg} us {t * 1e6:.0f} GBps_in {tot * 4 / t / 1e9:.1f} GBps_out {cfg['cols'] * osz * 4 / t / 1e9:.1f} per_stream_GBps {cfg['total'] * 4 / t / 1e9:.2f}")
