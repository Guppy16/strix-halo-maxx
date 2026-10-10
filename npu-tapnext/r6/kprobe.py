"""Primitive characterisation probe (round 6): every bf16 bit pattern through kprobe.cc's modes on one core.

build (npu-iron, no device):  python3 r6/kprobe.py build --out build/kprobe
run   (npu-iron, NPU):        python3 r6/kprobe.py run --out build/kprobe [--modes 0,1,2]
The run prints, per mode, cycles per 32 lanes (median over chunks, MEASURED with the core's cycle counter) and the
error against the exact function (float64) in bf16 ulps over the inputs where the function is finite and normal.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
CH = 2048
N = 65536
NCH = N // CH
OUT_CH = CH + 32  # results + a 64 B stamp record
MODES = {0: "copy", 1: "vtanh", 2: "gelu_nk8", 3: "exp32_nk8", 4: "aie_exp2", 5: "sigmoid_nk8", 6: "aie_inv", 7: "gelu_lut", 8: "gelu_lut2"}


def build(o):
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2, Tile
    from ml_dtypes import bfloat16

    modes = [int(m) for m in o.modes.split(",")]

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def probe(X: In, Y: Out, *, nm: CompileTime[int] = 1):
        Xo = np.ndarray[(CH,), np.dtype[bfloat16]]
        Yo = np.ndarray[(OUT_CH,), np.dtype[bfloat16]]
        fn = ExternalFunction("kprobe", object_file_name="kprobe.o", source_file=str(HERE / "kprobe.cc"),
                              arg_types=[Xo, Yo, np.int32], compile_flags=[f"-I{INC}", f"-I{HERE}", f"-I{HERE.parent}", f"-DCH={CH}", "-O2"] + o.kdefs.split())
        xf = ObjectFifo(Xo, name="X", depth=2)
        yf = ObjectFifo(Yo, name="Y", depth=2)

        def core(xc, yp, fn):
            for m in modes:
                for _ in range_(NCH):
                    x = xc.acquire(1)
                    y = yp.acquire(1)
                    fn(x, y, m)
                    xc.release(1)
                    yp.release(1)

        w = Worker(core, [xf.cons(), yf.prod(), fn], tile=Tile(0, 2), stack_size=0x1000)

        def seq(x, y, xp, yc):
            tg = TaskGroup()
            for i, _ in enumerate(modes):
                xp.fill(x, tap=TensorAccessPattern((N,), 0, [1, 1, NCH, CH], [0, 0, CH, 1]), group=tg, wait=False)
                yc.drain(y, tap=TensorAccessPattern((len(modes) * NCH * OUT_CH,), i * NCH * OUT_CH, [1, 1, NCH, OUT_CH], [0, 0, OUT_CH, 1]),
                         group=tg, wait=True)
            tg.finish()

        X_ty = np.ndarray[(N,), np.dtype[bfloat16]]
        Y_ty = np.ndarray[(len(modes) * NCH * OUT_CH,), np.dtype[bfloat16]]
        rt = Runtime(seq, [X_ty, Y_ty, xf.prod(), yf.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    probe.specialize(nm=len(modes)).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(modes=modes)))


def timing(o):
    """cycles per 32 lanes by host timing: build per mode with --kdefs=-DREPS=R, then run this on each build;
    prints the median dispatch time (the difference to the copy build / (R x 2048 vectors) is the mode's cost)"""
    import time
    import pyxrt
    x = np.zeros(N, np.uint16)
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    modes = json.loads(Path(o.out + ".json").read_text())["modes"]
    bo_x = pyxrt.bo(dev, N * 2, pyxrt.bo.host_only, kern.group_id(3))
    bo_y = pyxrt.bo(dev, len(modes) * NCH * OUT_CH * 2, pyxrt.bo.host_only, kern.group_id(4))
    bo_x.write(x.tobytes(), 0)
    bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    ts = []
    for it in range(o.iters + 3):
        t0 = time.perf_counter()
        r = kern(3, bo_i, len(instr), bo_x, bo_y)
        st = r.wait(o.timeout_ms * 1000)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        if it >= 3:
            ts.append(time.perf_counter() - t0)
    print(f"TIMING {Path(o.out).name} modes {modes}: dispatch median {np.median(ts) * 1e6:.0f} us min {min(ts) * 1e6:.0f} us")


def ref(name, x):
    x = x.astype(np.float64)
    with np.errstate(all="ignore"):
        if name == "copy":
            return x
        if name == "vtanh":
            return np.tanh(x)
        if name in ("gelu_nk8", "gelu_lut", "gelu_lut2"):
            return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
        if name == "exp32_nk8":
            return np.exp(x)
        if name == "aie_exp2":
            return np.exp2(x)
        if name == "sigmoid_nk8":
            return 1 / (1 + np.exp(-x))
        if name == "aie_inv":
            return 1 / x
    raise ValueError(name)


def ulps(got, exact):
    """|got - exact| in units of the bf16 ulp at exact (2^(e-7))"""
    e = np.floor(np.log2(np.maximum(np.abs(exact), 1e-38)))
    return np.abs(got - exact) / 2.0 ** (e - 7)


def run(o):
    import pyxrt
    from ml_dtypes import bfloat16

    modes = json.loads(Path(o.out + ".json").read_text())["modes"]
    x = np.arange(N, dtype=np.uint32).astype(np.uint16).view(bfloat16)
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    ny = len(modes) * NCH * OUT_CH * 2
    bo_x = pyxrt.bo(dev, N * 2, pyxrt.bo.host_only, kern.group_id(3))
    bo_y = pyxrt.bo(dev, ny, pyxrt.bo.host_only, kern.group_id(4))
    bo_x.write(x.tobytes(), 0)
    bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    r = kern(3, bo_i, len(instr), bo_x, bo_y)
    st = r.wait(o.timeout_ms * 1000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    y = np.frombuffer(bo_y.read(ny, 0), np.uint16).reshape(len(modes), NCH, OUT_CH)
    xf = x.astype(np.float64)
    res = {}
    for i, m in enumerate(modes):
        name = MODES[m]
        cyc = y[i, :, CH:CH + 2].copy().view(np.uint32)[:, 0]
        got = y[i, :, :CH].reshape(-1).view(bfloat16).astype(np.float64)
        ex = ref(name, xf)
        ok = np.isfinite(xf) & np.isfinite(ex) & (np.abs(ex) > 2.0 ** -120) & (np.abs(ex) < 2.0 ** 120)
        if o.range:
            lo, hi = (float(v) for v in o.range.split(","))
            ok &= (xf >= lo) & (xf <= hi)
        u = ulps(got[ok], ex[ok])
        exact_bf = ex[ok].astype(np.float32).astype(bfloat16).astype(np.float64)
        res[name] = dict(cyc32=float(np.median(cyc)) / (CH / 32))
        print(f"{name:12s} cycles/32 lanes {np.median(cyc) / (CH / 32):7.2f}   inputs {ok.sum():6d}  "
              f"!= RNE(exact) {np.mean(got[ok] != exact_bf):6.2%}  ulp: p50 {np.median(u):.2f} p99 {np.percentile(u, 99):.2f} "
              f"max {u.max():.1f} at x={xf[ok][np.argmax(u)]:.4g}")
        if o.save:
            np.save(f"{o.out}_{name}.npy", np.stack([xf, got]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "time"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="0,1,2,3,4,5,6")
    ap.add_argument("--range", default=None, help="run: only inputs in [lo,hi] for the error stats")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--kdefs", default="", help="build: extra kernel flags (space separated)")
    o = ap.parse_args()
    dict(build=build, run=run, time=timing)[o.cmd](o)
