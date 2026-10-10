"""Run a design_i8.py build: pack int8 weights / tokens, time, check against a numpy reference
that mirrors moondream-opt's HI8Lin numerics (int32 exact accumulate, fp32 epilogue in the HIP
kernel's order, tanh-GELU, bf16 RNE).

usage: run_i8.py BUILD --tokens T [--real-n N] [--gelu] [--iters N] ...
The op schedule's "A" rows are weight rows (N_w = ops x 768), the tokens are padded to ROWS*nt*96.
"""

import argparse
import json
import time

import numpy as np
import pyxrt
from ml_dtypes import bfloat16

from design import COLS, ROWS
from design_i8s import plan
from pack_i8 import epilogue, pack_w, pack_x, quant_rows, quant_w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build")
    p.add_argument("--tokens", type=int, default=278, help="real token count (pooled 257+1+~20)")
    p.add_argument("--real-n", type=int, default=None, help="real weight rows (default: all ops' rows)")
    p.add_argument("--gelu", action="store_true", help="the build was compiled with -DGELU")
    p.add_argument("--gelu-ops", default="", help="comma list of op indices whose token tiles set the runtime GELU flag (-DGELU_RT)")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--timeout-ms", type=int, default=-1)
    p.add_argument("--check-ops", type=int, default=2, help="ops to check (0 = none)")
    o = p.parse_args()
    cfg = json.load(open(o.build + ".json"))
    m, k, n, rho, reps = cfg["m"], cfg["k"], cfg["n"], cfg["rho"], cfg["reps"]
    sched = json.loads(cfg["sched"])
    ops, a_tot, w_tot, c_tot = plan(sched, m, k, n, rho)
    T_pad = ops[0]["N"]
    assert all(op["N"] == T_pad for op in ops) and o.tokens <= T_pad, (o.tokens, T_pad)
    R = COLS * m  # weight rows per op
    N_w = len(ops) * R
    real_n = o.real_n or N_w

    rng = np.random.default_rng(0)
    gelu_ops = {int(v) for v in o.gelu_ops.split(",") if v}
    # tokens per distinct K (fc2's input is 8192 wide): bf16 activations quantised per token like quant_rows
    tok = {}
    for K in sorted({op["K"] for op in ops}):
        x = np.zeros((T_pad, K), np.float32)
        x[: o.tokens] = rng.standard_normal((o.tokens, K)).astype(bfloat16).astype(np.float32)
        xq, sa = quant_rows(x)
        xq[o.tokens :] = 0
        tok[K] = (xq, sa, pack_x(xq, sa, n, k), pack_x(xq, sa, n, k, gelu=True))
    A_all = np.zeros(a_tot, np.uint32)
    W_all = np.zeros(w_tot * reps, np.uint32)
    refs = []
    for i, op in enumerate(ops):
        K = op["K"]
        xq, sa, B_plain, B_gelu = tok[K]
        w = np.zeros((R, K), np.float32)
        rows = min(R, max(0, real_n - i * R))
        w[:rows] = (rng.standard_normal((rows, K)) * 0.02).astype(bfloat16).astype(np.float32)
        wq, sw = quant_w(w)
        bias = (rng.standard_normal(R) * 0.1).astype(np.float32)
        bias[rows:] = 0
        A_all[op["a"] : op["a"] + COLS * op["la"]] = pack_w(wq, sw, bias, m, k, rho)
        for rep in range(reps):
            off = rep * w_tot + op["w"]
            W_all[off : off + ROWS * op["nt"] * op["lb"]] = B_gelu if i in gelu_ops else B_plain
        if i < o.check_ops:
            acc = wq.astype(np.int64) @ xq.astype(np.int64).T  # (R, T_pad) exact
            refs.append((i, epilogue(acc, sa, sw, bias, o.gelu or i in gelu_ops), rows))

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

    busy0 = gpu_busy()
    tk = []
    for it in range(o.warmup + o.iters):
        t1 = time.perf_counter()
        run = kern(3, bo_i, len(instr), bo_a, bo_w, bo_c)
        st = run.wait() if o.timeout_ms < 0 else run.wait(o.timeout_ms)
        t2 = time.perf_counter()
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            print("RUN FAILED", st, f"after {1e6 * (t2 - t1):.0f} us (iteration {it})")
            return 1
        if it >= o.warmup:
            tk.append(t2 - t1)
    med, mn = 1e6 * float(np.median(tk)), 1e6 * min(tk)
    slow = med > 1.6 * mn or (800 < med < 1100 and mn < 700)
    macs_real = o.tokens * sum(op["K"] * R for op in ops) * reps * (real_n / N_w)
    w_bytes = sum(op["K"] * R for op in ops) * reps
    print(
        f"TIME kernel_us {med:.0f} min {mn:.0f} real_TOPS {2 * macs_real / med / 1e6:.2f}"
        f" padded_TOPS {2 * T_pad * w_bytes / med / 1e6:.2f} weights_MB {w_bytes / 1e6:.1f}"
        f" weight_GBps {w_bytes / med / 1e3:.1f} gpu_busy% {max(busy0, gpu_busy())}{' SLOW-MODE' if slow else ''}"
    )

    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    C = np.frombuffer(bo_c.read(c_tot * 2, 0), dtype=bfloat16)
    for i, ref, rows in refs:
        op = ops[i]
        c = C[op["c"] : op["c"] + COLS * m * op["N"]].reshape(COLS * m, op["N"])
        if cfg.get("c_linear"):
            tiles = c.reshape(COLS, m, op["N"] // n, n).transpose(0, 2, 1, 3)
            tiles = tiles.reshape(COLS, op["N"] // n, m // 8, n // 8, 8, 8)
            c = tiles.transpose(0, 2, 4, 1, 3, 5).reshape(COLS * m, op["N"])
        c = c[:rows, : o.tokens].astype(np.float32)
        r = ref[:rows, : o.tokens]
        exact = r.astype(bfloat16).astype(np.float32)
        err = np.abs(c - r)
        ulp = np.abs(exact) * 2.0**-8 + 1e-30
        print(
            f"CHECK op{i} rows={rows} tokens={o.tokens}: mismatch_vs_bf16(ref) {np.mean(c != exact):.4%}"
            f" max_err/ulp {np.max(err / ulp):.2f} rms_rel {np.sqrt((err**2).sum() / (r**2).sum() + 1e-30):.2e}"
            f" ref_bf16_floor {np.sqrt(((exact - r) ** 2).sum() / (r**2).sum() + 1e-30):.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
