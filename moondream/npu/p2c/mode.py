"""Reproducer for the bimodal per-process timing.

usage: mode.py BUILD [--flags host_only|cacheable|normal] [--sets N] [--iters N] [--nosync]
Zero data (timing only). Prints, per BO set: device addresses of the BOs and the
median dispatch time, so the mode can be correlated with placement.
"""

import argparse
import json
import os
import time

import numpy as np
import pyxrt

from design import plan

p = argparse.ArgumentParser()
p.add_argument("build")
p.add_argument("--flags", default="host_only")
p.add_argument("--sets", type=int, default=1)
p.add_argument("--iters", type=int, default=12)
p.add_argument("--nosync", action="store_true", help="no per-iteration bo_a sync")
p.add_argument("--touch", action="store_true", help="write the whole data BOs from numpy before use")
p.add_argument("--burn", type=float, default=0, help="seconds of single-thread numpy before timing")
p.add_argument("--trace", action="store_true", help="print every iteration's time")
p.add_argument("--rand", action="store_true", help="random operand bytes (MAC array toggles like real data)")
o = p.parse_args()
cfg = json.load(open(o.build + ".json"))
ops, a_tot, w_tot, c_tot = plan(json.loads(cfg["sched"]), cfg["m"], cfg["k"], cfg["n"], cfg["rho"])
sizes = (a_tot * 4, w_tot * cfg["reps"] * 4, c_tot * 2)
flag = {"host_only": pyxrt.bo.host_only, "cacheable": pyxrt.bo.cacheable, "normal": pyxrt.bo.normal}[o.flags]

dev = pyxrt.device(0)
xb = pyxrt.xclbin(o.build + ".xclbin")
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(o.build + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_i.write(instr.tobytes(), 0)
bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
print(f"pid {os.getpid()} flags {o.flags} sizes {sizes} instr_addr {bo_i.address():#x}")
if o.burn:
    t0 = time.perf_counter()
    x = np.random.default_rng(0).standard_normal((2048, 2048)).astype(np.float32)
    while time.perf_counter() - t0 < o.burn:
        x = np.tanh(x @ x * 1e-3)
    print(f"burned {o.burn} s")
keep = []
for s in range(o.sets):
    bos = [pyxrt.bo(dev, n, flag, kern.group_id(3 + i)) for i, n in enumerate(sizes)]
    if o.touch or o.rand:
        rng = np.random.default_rng(s)
        for b, n in zip(bos[:2], sizes[:2]):
            data = rng.integers(0, 256, n, dtype=np.uint8) if o.rand else np.zeros(n, np.uint8)
            if o.rand:
                data.reshape(-1, 9)[:, 8] = 120  # keep block exponents sane (bfp16ebs8: 8 mantissas + 1 exponent)
            b.write(data.tobytes(), 0)
    for b in bos:
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    ts = []
    for it in range(o.iters + 2):
        t0 = time.perf_counter()
        if not o.nosync:
            bos[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = kern(3, bo_i, len(instr), *bos)
        st = run.wait(3000000 if it == 0 else 0)
        t1 = time.perf_counter()
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            print("RUN FAILED", st)
            raise SystemExit(1)
        if it >= 2:
            ts.append(t1 - t0)
        if o.trace:
            print(f"  it {it}: {1e6 * (t1 - t0):.0f}")
    addrs = " ".join(f"{b.address():#x}" for b in bos)
    print(f"set {s}: median {1e6 * np.median(ts):.0f} us min {1e6 * min(ts):.0f} first {1e6 * ts[0]:.0f} addrs {addrs}")
    keep.append(bos)  # keep earlier sets alive so new ones land elsewhere
