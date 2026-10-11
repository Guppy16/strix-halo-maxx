"""Run a design.py / design3.py build on the NPU: pack, time, check against a float64 reference.

Times: `kernel` = submit + wait of one dispatch (includes the fixed launch cost);
sync in/out reported separately. Reference = decoded bfp operands multiplied in
float64, so the only expected differences are fp32 accumulation order and the
final bf16 rounding.

Safety: every dispatch waits with a timeout (--timeout-ms, default 5 s) and the
first dispatch of a fresh build should be run with --iters 1 --warmup 0.
"""

import argparse
import os
import json
import time

import numpy as np
import pyxrt
from ml_dtypes import bfloat16

from design import COLS, ROWS, plan, words_a
from pack import decode_rows, encode_rows, pack_a, pack_w

M_REAL = 729
RA_K = 1152  # design3 slot size (half an M-tile of A at K = 2 * RA_K)


def ddr_bytes(cfg, ops, m):
    """Bytes each dispatch moves through the shims, per the design's rules."""
    resident = "lookahead" in cfg and "a_depth" not in cfg  # design3 (A in memtiles)
    ra = words_a(m, RA_K)
    a = b = c = 0
    for op in ops:
        cached = resident and op["la"] <= 2 * ra
        a += COLS * op["la"] * 4 * (1 if cached else op["nt"])
        b += ROWS * op["nt"] * op["lb"] * 4
        c += COLS * m * op["N"] * 2
    return a, b, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build")
    p.add_argument("--real", default=None, help="JSON list of [K_real, N_real] per op")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=20, help="the NPU clock ramps over ~20 dispatches")
    p.add_argument("--check", type=int, default=1, help="check reps (0 = none)")
    p.add_argument("--nosync", action="store_true", help="skip the per-iteration A sync")
    p.add_argument("--timeout-ms", type=int, default=-1, help="-1: plain wait(); >0: wait(N) (returns no earlier than ~N us here); 0: wait(0)")
    o = p.parse_args()
    cfg = json.load(open(o.build + ".json"))
    m, k, n, rho, reps = cfg["m"], cfg["k"], cfg["n"], cfg["rho"], cfg["reps"]
    sched = json.loads(cfg["sched"])
    ops, a_tot, w_tot, c_tot = plan(sched, m, k, n, rho)
    real = json.loads(o.real) if o.real else [[K, op["N"]] for (K, _), op in zip(sched, ops)]

    rng = np.random.default_rng(0)
    A_all = np.zeros(a_tot, np.uint32)
    W_all = np.zeros(w_tot * reps, np.uint32)
    refs = []
    for i, (op, (Kr, Nr)) in enumerate(zip(ops, real)):
        K, N = op["K"], op["N"]
        a = np.zeros((COLS * m, K), np.float32)
        a[:M_REAL, :Kr] = rng.standard_normal((M_REAL, Kr)).astype(bfloat16)
        ea = encode_rows(a)
        A_all[op["a"] : op["a"] + COLS * op["la"]] = pack_a(ea, m, k, rho, half="-DBLK12" in cfg.get("kdefs", ""))
        aq = decode_rows(ea).astype(np.float64)
        # Same weights every rep (packing is the slow part); only rep 0 is checked,
        # which is what C holds after the last rep too.
        w = np.zeros((N, K), np.float32)  # W^T
        w[:Nr, :Kr] = (rng.standard_normal((Nr, Kr)) * 0.03).astype(bfloat16)
        ew = encode_rows(w)
        pw = pack_w(ew, n, k)
        for rep in range(reps):
            off = rep * w_tot + op["w"]
            W_all[off : off + ROWS * op["nt"] * op["lb"]] = pw
        refs.append((i, aq[:M_REAL] @ decode_rows(ew).astype(np.float64)[:Nr].T, Nr))
        if os.environ.get("DUMP") and i == 1:
            np.savez(os.environ["DUMP"] + "_in", a=aq[:M_REAL].astype(np.float32), w=decode_rows(ew).astype(np.float32)[:Nr])

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.build + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.build + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_a = pyxrt.bo(dev, A_all.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_w = pyxrt.bo(dev, W_all.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bo_c = pyxrt.bo(dev, c_tot * 2, pyxrt.bo.host_only, kern.group_id(5))
    bo_i.write(instr.tobytes(), 0)
    bo_a.write(A_all.tobytes(), 0)
    bo_w.write(W_all.tobytes(), 0)
    for b in (bo_i, bo_a, bo_w, bo_c):
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def gpu_busy():
        try:
            return int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
        except OSError:
            return -1

    def xdna_irqs():
        try:
            n = 0
            for line in open("/proc/interrupts"):
                if "xdna" in line.lower():
                    n += sum(int(x) for x in line.split()[1:] if x.isdigit())
            return n
        except OSError:
            return -1

    busy0 = gpu_busy()
    irq0 = xdna_irqs()
    tk, ti, to = [], [], []
    for it in range(o.warmup + o.iters):
        t0 = time.perf_counter()
        if not o.nosync:
            bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.perf_counter()
        run = kern(3, bo_i, len(instr), bo_a, bo_w, bo_c)
        st = run.wait() if o.timeout_ms < 0 else run.wait(o.timeout_ms)  # -1: plain wait, no timeout
        t2 = time.perf_counter()
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            print("RUN FAILED", st, f"after {1e6 * (t2 - t1):.0f} us (iteration {it})")
            return 1
        bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t3 = time.perf_counter()
        if it >= o.warmup:
            ti.append(t1 - t0), tk.append(t2 - t1), to.append(t3 - t2)
    us = lambda v: 1e6 * float(np.median(v))
    flops_real = 2 * M_REAL * sum(Kr * Nr for Kr, Nr in real) * reps
    flops_pad = 2 * COLS * m * sum(op["K"] * op["N"] for op in ops) * reps
    busy = max(busy0, gpu_busy())
    irq_per_dispatch = (xdna_irqs() - irq0) / (o.warmup + o.iters) if irq0 >= 0 else -1
    # slow-mode signature: completion quantised at ~1 ms (median near 950 or 1950 us while min is far below)
    slow = us(tk) > 1.6 * 1e6 * min(tk) or (800 < us(tk) < 1100 and 1e6 * min(tk) < 700)
    ab, bb, cb = ddr_bytes(cfg, ops, m)
    mb = (ab + bb + cb) * reps / 1e6
    print(
        f"TIME kernel_us {us(tk):.0f} min {1e6 * min(tk):.0f} sync_in {us(ti):.0f} sync_out {us(to):.0f}"
        f" real_TOPS {flops_real / us(tk) / 1e6:.2f} padded_TOPS {flops_pad / us(tk) / 1e6:.2f}"
        f" ddr_MB {mb:.1f} (A {ab * reps / 1e6:.1f} B {bb * reps / 1e6:.1f} C {cb * reps / 1e6:.1f})"
        f" GBps {mb / us(tk) * 1e3:.1f} gpu_busy% {busy} irq/dispatch {irq_per_dispatch:.2f}{' SLOW-MODE' if slow else ''}"
    )

    C = np.frombuffer(bo_c.read(c_tot * 2, 0), dtype=bfloat16)
    for i, ref, Nr in refs:
        op = ops[i]
        c = C[op["c"] : op["c"] + COLS * m * op["N"]].reshape(COLS * m, op["N"])
        if cfg.get("c_linear"):
            # every 96x96 core tile is still [m/8][n/8][8][8] in memory; un-block it
            tiles = c.reshape(COLS, m, op["N"] // n, n).transpose(0, 2, 1, 3)
            tiles = tiles.reshape(COLS, op["N"] // n, m // 8, n // 8, 8, 8)
            c = tiles.transpose(0, 2, 4, 1, 3, 5).reshape(COLS * m, op["N"])
        c = c[:M_REAL, :Nr].astype(np.float64)
        exact = ref.astype(np.float32).astype(bfloat16).astype(np.float64)
        if os.environ.get("DUMP") and i == 1:
            np.savez(os.environ["DUMP"], c=c, exact=exact)
        err = np.abs(c - ref)
        ulp_mis = np.mean(c != exact)
        print(
            f"CHECK op{i} K={op['K']} N={Nr}: rms_rel {np.sqrt((err**2).sum() / (ref**2).sum()):.2e}"
            f" max_abs/max_ref {err.max() / np.abs(ref).max():.2e}"
            f" bf16-rounding-floor {np.sqrt(((exact - ref) ** 2).sum() / (ref**2).sum()):.2e}"
            f" mismatch_vs_bf16(ref) {ulp_mis:.4%}"
            f" max_err/ulp {np.max(err / np.maximum(np.abs(ref) * 2**-8, 1e-30)):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
