"""Time a spin.py build: usage runspin.py BUILD [iters]."""

import json
import sys
import time

import numpy as np
import pyxrt

b = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cfg = json.load(open(b + ".json"))
dev = pyxrt.device(0)
xb = pyxrt.xclbin(b + ".xclbin")
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(b + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_c = pyxrt.bo(dev, 64, pyxrt.bo.host_only, kern.group_id(3))
bo_i.write(instr.tobytes(), 0)
bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
ts = []
for it in range(iters + 20):
    t0 = time.perf_counter()
    r = kern(3, bo_i, len(instr), bo_c)
    st = r.wait(3000000) if it == 0 else r.wait()
    if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
        print("RUN FAILED", st)
        raise SystemExit(1)
    ts.append(time.perf_counter() - t0)
print(f"SPIN loops {cfg['loops']} kdefs '{cfg['kdefs']}' us median {1e6 * np.median(ts[20:]):.0f} min {1e6 * min(ts[20:]):.0f}")
