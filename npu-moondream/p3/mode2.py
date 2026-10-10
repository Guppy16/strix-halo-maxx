"""Inside one process: time a qkv build, then the spin build (core-clock probe), then qkv again.

usage: mode2.py QKV_BUILD SPIN_BUILD
Tells whether a slow process has a slow core clock (spin also 2x) or slow
completion signalling (spin unchanged).
"""

import json
import sys
import time

import numpy as np
import pyxrt

from design import plan

qb, sb = sys.argv[1], sys.argv[2]
dev = pyxrt.device(0)


def ctx_for(build, n_out):
    xb = pyxrt.xclbin(build + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(build + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return ctx, kern, bo_i, instr


cfg = json.load(open(qb + ".json"))
ops, a_tot, w_tot, c_tot = plan(json.loads(cfg["sched"]), cfg["m"], cfg["k"], cfg["n"], cfg["rho"])
qctx, qk, qi, qinstr = ctx_for(qb, 0)
bos = [pyxrt.bo(dev, n, pyxrt.bo.host_only, qk.group_id(3 + i)) for i, n in enumerate((a_tot * 4, w_tot * cfg["reps"] * 4, c_tot * 2))]
for b in bos:
    b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
sctx, sk, si, sinstr = ctx_for(sb, 1)
so = pyxrt.bo(dev, 64, pyxrt.bo.host_only, sk.group_id(3))


def timed(fn, n):
    ts = []
    for it in range(n):
        t0 = time.perf_counter()
        r = fn()
        st = r.wait(3000000) if it == 0 else r.wait()
        ts.append(time.perf_counter() - t0)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    return 1e6 * np.median(ts[n // 3:])


q1 = timed(lambda: qk(3, qi, len(qinstr), *bos), 24)
s1 = timed(lambda: sk(3, si, len(sinstr), so), 24)
q2 = timed(lambda: qk(3, qi, len(qinstr), *bos), 24)
print(f"qkv {q1:.0f} us, spin {s1:.0f} us, qkv {q2:.0f} us")
