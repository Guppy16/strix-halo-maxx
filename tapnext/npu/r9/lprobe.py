"""Round-9 SSM primitive probe: every bf16 bit pattern through lprobe.cc's modes on one core (from r6/kprobe.py, with
the processor-bus enable so the core timer can be read).

build (npu-iron, no device):  python3 r9/lprobe.py build --out build/r9/lprobe [--ab 1 --cd 3]
run   (npu-iron, NPU):        python3 r9/lprobe.py run --out build/r9/lprobe
Per mode: cycles per 32 lanes (MEASURED, core timer, median over chunks), mismatches against the bit-level emulation
(r9/ssmemu.py: the device must match it exactly) and against the exact function (RNE of float64).
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
OUT_CH = CH + 32
MODES = {0: "copy", 1: "sig_nk8", 2: "sig_lut", 3: "a_nk8", 4: "om_nk8", 5: "a_lut", 6: "om_lut", 7: "sqrt_nk8", 8: "sqrt_lut"}


def build(o):
    import aie.iron as iron
    from aie.dialects.aiex import npu_maskwrite32
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2, Tile
    from ml_dtypes import bfloat16

    modes = [int(m) for m in o.modes.split(",")]
    sec = lambda b: f'__attribute__((section(".aie.bank{b}")))' if b != "data" else ""
    flags = [f"-I{INC}", f"-I{HERE}", f"-I{HERE.parent}", f"-DCH={CH}", "-O2",
             "-DSSM_LUT_EXP", f"-DSSM_LUT_AB={sec(o.ab)}", f"-DSSM_LUT_CD={sec(o.cd)}"] + o.kdefs.split()

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def probe(X: In, Y: Out, *, key: CompileTime[str] = ""):  # key: the flags, so IRON's jit cache tells builds apart
        Xo = np.ndarray[(CH,), np.dtype[bfloat16]]
        Yo = np.ndarray[(OUT_CH,), np.dtype[bfloat16]]
        fn = ExternalFunction("lprobe", object_file_name="lprobe.o", source_file=str(HERE / "lprobe.cc"),
                              arg_types=[Xo, Yo, np.int32], compile_flags=flags)
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
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)  # core timer readable (P15)
            for i, _ in enumerate(modes):  # one group per mode: 2 BDs at a time (the shim has 16)
                tg = TaskGroup()
                xp.fill(x, tap=TensorAccessPattern((N,), 0, [1, 1, NCH, CH], [0, 0, CH, 1]), group=tg, wait=False)
                yc.drain(y, tap=TensorAccessPattern((len(modes) * NCH * OUT_CH,), i * NCH * OUT_CH, [1, 1, NCH, OUT_CH], [0, 0, OUT_CH, 1]),
                         group=tg, wait=True)
                tg.finish()

        X_ty = np.ndarray[(N,), np.dtype[bfloat16]]
        Y_ty = np.ndarray[(len(modes) * NCH * OUT_CH,), np.dtype[bfloat16]]
        rt = Runtime(seq, [X_ty, Y_ty, xf.prod(), yf.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    probe.specialize(key=" ".join(flags)).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(modes=modes)))


def refs(x):
    """(emulation, exact) per mode name; x = every bf16 value as float32"""
    import sys
    sys.path.insert(0, str(HERE))
    import ssmemu
    import ssmlut
    from emu import bf
    x64 = x.astype(np.float64)
    sig_t = ssmemu.sig_table(ssmlut.SIG_LO, ssmlut.SIG_HI, ssmlut.SIG_STEP)
    rr = ssmemu.a_om_rr(ssmlut.EXP_N)
    with np.errstate(all="ignore"):
        sig_ex = bf(1 / (1 + np.exp(-x64)))
        a_ex = bf(np.exp(x64))
        om_ex = bf(1 - bf(np.exp(2 * x64)))
        sq_ex = bf(np.sqrt(x64))
        a_n, om_n, _ = ssmemu.a_om_nk8(x)
        a_l, om_l, _ = rr(x)
        return {"copy": (x, x), "sig_nk8": (ssmemu.sig_nk8(x), sig_ex), "sig_lut": (sig_t(x), sig_ex),
                "a_nk8": (a_n, a_ex), "om_nk8": (om_n, om_ex), "a_lut": (a_l, a_ex), "om_lut": (om_l, om_ex),
                "sqrt_nk8": (ssmemu.sqrt_nk8(x), sq_ex), "sqrt_lut": (ssmemu.sqrt_exact(x), sq_ex)}


def domain(name, x):
    ok = np.isfinite(x)
    if name.startswith(("a_", "om_")):
        ok &= (x <= 0) & (x >= -10)  # log_a: -8 sp sigma with sp <= 1.18
    if name.startswith("sqrt"):
        ok &= (x == 0) | ((x >= 2.0 ** -9) & (x <= 1))  # om = 1 - e2: 0 or >= 2^-8
    if name.startswith("sig"):
        ok &= np.abs(x) < 1e30
    return ok


def run(o):
    import pyxrt
    from ml_dtypes import bfloat16

    modes = json.loads(Path(o.out + ".json").read_text())["modes"]
    xb16 = np.arange(N, dtype=np.uint32).astype(np.uint16).view(bfloat16)
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
    bo_x.write(xb16.tobytes(), 0)
    bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    r = kern(3, bo_i, len(instr), bo_x, bo_y)
    st = r.wait(o.timeout_ms * 1000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    y = np.frombuffer(bo_y.read(ny, 0), np.uint16).reshape(len(modes), NCH, OUT_CH)
    x = xb16.astype(np.float32)
    rec = y[0, 0, CH:CH + 22].copy().view(np.uint32)
    print("bss canary (first call):", [hex(v) for v in rec[2:10]], "calls", rec[10], "(last chunk calls", y[-1, -1, CH + 20:CH + 22].copy().view(np.uint32)[0], ")")
    R = refs(x)
    for i, m in enumerate(modes):
        name = MODES[m]
        cyc = y[i, :, CH:CH + 2].copy().view(np.uint32)[:, 0]
        got = y[i, :, :CH].reshape(-1).view(bfloat16).astype(np.float32)
        emu_v, ex = R[name]
        ok = domain(name, x)
        g, e, t = got[ok], np.asarray(emu_v, np.float32)[ok], np.asarray(ex, np.float32)[ok]
        same = (g == e) | (np.isnan(g) & np.isnan(e))
        bad = np.flatnonzero(~same)
        print(f"{name:9s} cycles/32 lanes {np.median(cyc) / (CH / 32):7.2f}  inputs {ok.sum():6d}  "
              f"!= emulation {np.mean(~same):7.3%}  != RNE(exact) {np.mean(g != t):6.2%}"
              + (f"  first diffs x={x[ok][bad[:3]]} dev={g[bad[:3]]} emu={e[bad[:3]]}" if len(bad) else ""))
        if o.save:
            np.save(f"{o.out}_{name}.npy", np.stack([x, got]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--ab", default="data")
    ap.add_argument("--cd", default="3")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--kdefs", default="")
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
