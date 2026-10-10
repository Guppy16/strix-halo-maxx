"""NPU side of the pipelined frame: dispatch the vision-encoder GEMM sequences back to back for
`--seconds`, timing every frame. Static data (timing only).

usage: nputloop.py BUILD [BUILD...] --seconds S [--label L]
Each build gets its own hw_context; a frame = one dispatch of each build in order (so the
context switches between xclbins are included in the frame time).
"""

import argparse
import json
import sys
import time

import numpy as np
import pyxrt

from design import plan

p = argparse.ArgumentParser()
p.add_argument("builds", nargs="+")
p.add_argument("--seconds", type=float, default=30)
p.add_argument("--label", default="npu")
p.add_argument("--dump", default=None, help="save per-frame [end_time, frame_ms] rows (npy)")
o = p.parse_args()

dev = pyxrt.device(0)
ctxs = []
for b in o.builds:
    cfg = json.load(open(b + ".json"))
    ops, a_tot, w_tot, c_tot = plan(json.loads(cfg["sched"]), cfg["m"], cfg["k"], cfg["n"], cfg["rho"])
    xb = pyxrt.xclbin(b + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(b + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bos = [pyxrt.bo(dev, n, pyxrt.bo.host_only, kern.group_id(3 + i)) for i, n in enumerate((a_tot * 4, w_tot * cfg["reps"] * 4, c_tot * 2))]
    for x in (bo_i, *bos):
        x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    ctxs.append((b, ctx, kern, bo_i, instr, bos))


def gpu_busy():
    try:
        return int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
    except OSError:
        return -1


def frame():
    parts = []
    for b, ctx, kern, bo_i, instr, bos in ctxs:
        t0 = time.perf_counter()
        r = kern(3, bo_i, len(instr), *bos)
        st = r.wait(3000000) if not parts and frames == 0 else r.wait()
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        parts.append(time.perf_counter() - t0)
    return parts


frames = 0
for _ in range(3):
    frame()
    frames += 1
t_start = time.time()
times, busy, ends = [], [], []
while time.time() - t_start < o.seconds:
    parts = frame()
    times.append(parts)
    ends.append(time.time())
    frames += 1
    if frames % 5 == 0:
        busy.append(gpu_busy())
t_end = time.time()
a = 1e3 * np.array(times)
if o.dump:
    np.save(o.dump, np.column_stack([ends, a.sum(axis=1)]))  # ms, [frame][build]
tot = a.sum(axis=1)
per = " ".join(f"{b.split('/')[-1]} {np.median(a[:, i]):.1f}" for i, (b, *_) in enumerate(ctxs))
print(
    f"NPU[{o.label}] {t_start:.0f}-{t_end:.0f} frames {len(tot)} frame_ms median {np.median(tot):.1f} p10 {np.percentile(tot, 10):.1f}"
    f" p90 {np.percentile(tot, 90):.1f} min {tot.min():.1f} | per build (median ms): {per} | gpu_busy median {np.median(busy) if busy else -1:.0f}%",
    flush=True,
)
