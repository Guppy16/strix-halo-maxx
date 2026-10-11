"""Time a bw.py build: GB/s delivered from DDR (each stream's words counted once).

usage: runbw.py BUILD [ITERS]   -- the first dispatch waits with a timeout.
"""

import json
import sys
import time

import numpy as np
import pyxrt

b = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 25
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
for it in range(iters):
    t0 = time.perf_counter()
    r = kern(3, bo_i, len(instr), bo_a, bo_c)
    st = r.wait(3000000) if it == 0 else r.wait()
    if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
        print("RUN FAILED", st)
        raise SystemExit(1)
    ts.append(time.perf_counter() - t0)
t = float(np.median(ts[min(5, iters - 1):]))
print(f"BW {cfg} us {t * 1e6:.0f} min {min(ts[1:] or ts) * 1e6:.0f} GBps_in {tot * 4 / t / 1e9:.1f} per_stream_GBps {cfg['total'] * 4 / t / 1e9:.2f} instr_words {len(instr)}")
