"""Run a design7 build on the NPU: pack, time, check every op against a float64 reference (+ bias, bf16 RNE).

    PYTHONPATH=/opt/xilinx/xrt/python:/work/npu-tapnext python3 run7.py build/<name> --iters 20 --warmup 20
Prints TIME (kernel us median / min, TOPS, DDR MB, GB/s, gpu_busy) and one CHECK line per op.
"""

import argparse
import json
import os
import time

import numpy as np
import pyxrt
from ml_dtypes import bfloat16

from design7 import CHUNK_SLOTS, CHUNKS, COLS, EW_PARAM_SLOTS, M, ROWS_G, SLOT_W, plan, words_a
from pack7 import decode_rows, encode_hw, encode_rows, pack_a, pack_w7, unblock

M_REAL = int(os.environ.get("M_REAL", "1072"))


def rmsnorm_ref(x, sc1):
    """TAPNext RMSNorm with torch's bf16 numerics: x bf16 [R, D], sc1 = (scale + 1) bf16 [D] -> z bf16 [R, D]."""
    f = lambda v: np.asarray(v, np.float32).astype(bfloat16).astype(np.float32)
    xf = x.astype(np.float32)
    s = f(xf * xf).sum(axis=1, dtype=np.float32)
    mean = f(s / np.float32(x.shape[1]))
    rstd = f(1.0 / np.sqrt(f(mean + np.float32(1e-6))))
    return f(f(xf * rstd[:, None]) * sc1.astype(np.float32)[None]).astype(bfloat16)


def to_layout_l(xc):
    """[m, 768] bf16 of one column -> the blocked chunk layout [rb][q][cb][8][8] as uint32 words."""
    m = xc.shape[0]
    v = xc.reshape(m // 8, 8, 12, 8, 8).transpose(0, 2, 3, 1, 4)
    return np.ascontiguousarray(v).reshape(-1).view(np.uint32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timeout-ms", type=int, default=5000)
    o = p.parse_args()
    cfg = json.load(open(o.build + ".json"))
    m, k, n, rho, reps = cfg["m"], cfg["k"], cfg["n"], cfg["rho"], cfg["reps"]
    sched = json.loads(cfg["sched"])
    ops, a_tot, w_tot, c_tot, x_tot, e_tot = plan(sched)

    rng = np.random.default_rng(0)
    A_all = np.zeros(a_tot, np.uint32)
    W_all = np.zeros(w_tot * reps, np.uint32)
    X_all = np.zeros(x_tot, np.uint32)
    E_exp = {}  # op index -> (offset, expected words)
    refs = []
    ew_a = {}  # id(ew op) -> the decoded A its consumer multiplies
    for i, op in enumerate(ops):
        if "ew" in op:
            x = np.zeros((COLS * m, 768), bfloat16)
            x[:M_REAL] = (rng.standard_normal((M_REAL, 768)) * 1.5 + 0.2).astype(bfloat16)
            sc1 = (1 + 0.1 * rng.standard_normal(768)).astype(bfloat16)
            z = rmsnorm_ref(x, sc1)
            for c in range(COLS):
                base = op["x"] + c * op["xcol"]
                X_all[base: base + EW_PARAM_SLOTS * SLOT_W] = sc1.view(np.uint32)
                X_all[base + EW_PARAM_SLOTS * SLOT_W: base + op["xcol"]] = to_layout_l(x[c * m:(c + 1) * m])
            ez = encode_hw(z.astype(np.float32))  # the device's conversion semantics (exponent bump), not the host codec's saturation
            E_exp[i] = (op["e"], pack_a(ez, m, k, rho, half=True))
            ew_a[id(op)] = decode_rows(ez).astype(np.float64)
            continue
        K, N = op["K"], op["N"]
        if op["src"] is not None:
            aq = ew_a[id(op["src"])]
        else:
            a = np.zeros((COLS * m, K), np.float32)
            a[:M_REAL] = rng.standard_normal((M_REAL, K)).astype(bfloat16)
            ea = encode_rows(a)
            A_all[op["a"]: op["a"] + COLS * op["la"]] = pack_a(ea, m, k, rho, half=True)
            aq = decode_rows(ea).astype(np.float64)
        w = (rng.standard_normal((N, K)) * 0.03).astype(bfloat16).astype(np.float32)  # W^T
        bias = (rng.standard_normal(N) * 0.5).astype(bfloat16).astype(np.float32)
        ew = encode_rows(w)
        pw = pack_w7(ew, bias, n, k)
        assert pw.size == ROWS_G * op["nt"] * op["lw"], (pw.size, ROWS_G * op["nt"] * op["lw"])
        for rep in range(reps):
            off = rep * w_tot + op["w"]
            W_all[off: off + pw.size] = pw
        ref = aq[:M_REAL] @ decode_rows(ew).astype(np.float64).T + bias
        refs.append((i, ref))

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.build + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.build + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_a = pyxrt.bo(dev, A_all.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_w = pyxrt.bo(dev, W_all.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bo_x = pyxrt.bo(dev, X_all.nbytes, pyxrt.bo.host_only, kern.group_id(5))
    bo_e = pyxrt.bo(dev, e_tot * 4, pyxrt.bo.host_only, kern.group_id(6))
    bo_c = pyxrt.bo(dev, c_tot * 2, pyxrt.bo.host_only, kern.group_id(7))
    bo_i.write(instr.tobytes(), 0)
    bo_a.write(A_all.tobytes(), 0)
    bo_w.write(W_all.tobytes(), 0)
    bo_x.write(X_all.tobytes(), 0)
    bo_e.write(np.zeros(e_tot, np.uint32).tobytes(), 0)
    for b in (bo_i, bo_a, bo_w, bo_x, bo_e, bo_c):
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
        run = kern(3, bo_i, len(instr), bo_a, bo_w, bo_x, bo_e, bo_c)
        st = run.wait(o.timeout_ms * 1000)
        t2 = time.perf_counter()
        if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            print("RUN FAILED", st, f"after {1e6 * (t2 - t1):.0f} us (iteration {it})")
            return 1
        if it >= o.warmup:
            tk.append(t2 - t1)
    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    bo_e.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    gops = [op for op in ops if "ew" not in op]
    us = 1e6 * float(np.median(tk))
    flops = 2 * M_REAL * sum(op["K"] * op["N"] for op in gops) * reps
    ab = sum(COLS * (words_a(m, cfg["k_slot"])) * 4 for op in gops) * reps  # every op fills the whole slot
    bb = sum(ROWS_G * op["nt"] * op["lw"] * 4 for op in gops) * reps
    cb = c_tot * 2 * reps
    mb = (ab + bb + cb) / 1e6
    print(f"TIME kernel_us {us:.0f} min {1e6 * min(tk):.0f} real_TOPS {flops / us / 1e6:.2f} ddr_MB {mb:.1f}"
          f" (A {ab / 1e6:.1f} B {bb / 1e6:.1f} C {cb / 1e6:.1f}) GBps {mb / us * 1e3:.1f} gpu_busy% {max(busy0, gpu_busy())}")

    C = np.frombuffer(bo_c.read(c_tot * 2, 0), dtype=bfloat16)
    E = np.frombuffer(bo_e.read(e_tot * 4, 0), dtype=np.uint32)
    rc = 0
    for i, (off, exp) in E_exp.items():
        got = E[off: off + exp.size]
        bad = got != exp
        panels = bad.reshape(-1, 18).any(axis=1)  # 72 B = 18 words per panel
        print(f"CHECK ew{i} {ops[i]['ew']}: words differing {bad.mean():.4%} panels differing {panels.mean():.4%} ({panels.sum()} of {panels.size})")
        rc |= bool(panels.mean() > 1e-2)
        if os.environ.get("EWDBG"):
            pm = panels.reshape(COLS, 12, 9, 2, 8)  # [c][q][s][hf][cb]
            print("  per column:", pm.sum(axis=(1, 2, 3, 4)).tolist())
            print("  per k-step:", pm.sum(axis=(0, 2, 3, 4)).tolist())
            print("  per chunk (s,hf):", pm.sum(axis=(0, 1, 4)).reshape(-1).tolist())
            print("  per cb:", pm.sum(axis=(0, 1, 2, 3)).tolist())
            gp = got.view(np.uint8).reshape(-1, 72)
            ep = exp.view(np.uint8).reshape(-1, 72)
            from aie.utils import bfp
            shown = 0
            for idx in np.nonzero(panels)[0][:400]:
                g = bfp.decode(gp[idx].reshape(8, 9).reshape(1, -1)).reshape(8, 8)
                e = bfp.decode(ep[idx].reshape(8, 9).reshape(1, -1)).reshape(8, 8)
                d = np.abs(g - e)
                rows = np.nonzero(d.max(axis=1))[0].tolist()
                if shown < 12:
                    print(f"  panel {idx} (c{idx // (12*9*2*8)} q{idx // (9*2*8) % 12} s{idx // 16 % 9} hf{idx // 8 % 2} cb{idx % 8}): rows {rows} max|d| {d.max():.3e} max|e| {np.abs(e).max():.3e} exp bytes {ep[idx][:9].tolist()} got {gp[idx][:9].tolist()}")
                    shown += 1
    for i, ref in refs:
        op = ops[i]
        c = unblock(C[op["c"]: op["c"] + COLS * m * op["N"]], m, op["N"], n)[:M_REAL].astype(np.float64)
        exact = ref.astype(np.float32).astype(bfloat16).astype(np.float64)
        err = np.abs(c - ref)
        mis = np.mean(c != exact)
        print(f"CHECK op{i} K={op['K']} N={op['N']}: rms_rel {np.sqrt((err**2).sum() / (ref**2).sum()):.2e}"
              f" mismatch_vs_bf16(ref) {mis:.4%} max_err/ulp {np.max(err / np.maximum(np.abs(ref) * 2**-8, 1e-30)):.2f}")
        rc |= bool(mis > 1e-3)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
