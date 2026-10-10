"""Run a design.py build on the NPU: pack, time, check against a float64 reference.

Times: `kernel` = submit + wait of one dispatch (includes the fixed launch cost);
sync in/out reported separately. Reference = decoded bfp operands multiplied in
float64, so the only expected differences are fp32 accumulation order and the
final bf16 rounding.
"""

import argparse
import json
import time

import numpy as np
import pyxrt
from ml_dtypes import bfloat16

from design import COLS, ROWS, plan
from pack import decode_rows, encode_rows, pack_a, pack_w

M_REAL = 729


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build")
    p.add_argument("--real", default=None, help="JSON list of [K_real, N_real] per op")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--check", type=int, default=1, help="check reps (0 = none)")
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
        A_all[op["a"] : op["a"] + COLS * op["la"]] = pack_a(ea, m, k, rho)
        aq = decode_rows(ea).astype(np.float64)
        for rep in range(reps):
            w = np.zeros((N, K), np.float32)  # W^T
            w[:Nr, :Kr] = (rng.standard_normal((Nr, Kr)) * 0.03).astype(bfloat16)
            ew = encode_rows(w)
            off = rep * w_tot + op["w"]
            W_all[off : off + ROWS * op["nt"] * op["lb"]] = pack_w(ew, n, k)
            if rep == reps - 1 or rep < o.check:
                refs.append((i, rep, aq[:M_REAL] @ decode_rows(ew).astype(np.float64)[:Nr].T, Nr))

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

    tk, ti, to = [], [], []
    for it in range(o.warmup + o.iters):
        t0 = time.perf_counter()
        bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t1 = time.perf_counter()
        run = kern(3, bo_i, len(instr), bo_a, bo_w, bo_c)
        st = run.wait()
        t2 = time.perf_counter()
        bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t3 = time.perf_counter()
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            print("RUN FAILED", st)
            return 1
        if it >= o.warmup:
            ti.append(t1 - t0), tk.append(t2 - t1), to.append(t3 - t2)
    us = lambda v: 1e6 * float(np.median(v))
    flops_real = 2 * M_REAL * sum(Kr * Nr for Kr, Nr in real) * reps
    flops_pad = 2 * COLS * m * sum(op["K"] * op["N"] for op in ops) * reps
    print(
        f"TIME kernel_us {us(tk):.0f} min {1e6 * min(tk):.0f} sync_in {us(ti):.0f} sync_out {us(to):.0f}"
        f" real_TOPS {flops_real / us(tk) / 1e6:.2f} padded_TOPS {flops_pad / us(tk) / 1e6:.2f}"
    )

    C = np.frombuffer(bo_c.read(c_tot * 2, 0), dtype=bfloat16)
    for i, rep, ref, Nr in refs:
        op = ops[i]
        if rep != reps - 1:
            continue  # C is overwritten by every rep; only the last one survives
        c = C[op["c"] : op["c"] + COLS * m * op["N"]].reshape(COLS * m, op["N"])
        c = c[:M_REAL, :Nr].astype(np.float64)
        exact = ref.astype(np.float32).astype(bfloat16).astype(np.float64)
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
