"""k-step bench (round 9): mm8.cc's mm_kstep on one core, no DMA (r9/ksb.cc), two builds: `ref` (the C++ loop over the
asm mm_core_p, the deployed path) and `ks` (the pipelined asm mm_ks, r9/gen_ks.py). Hashes of all 36 accumulator pieces
must match; cycles per k-step from the core timer.

build (npu-iron, no device):  python3 r9/ksb.py build --out build/r9/ksb
run   (npu-iron, NPU):        python3 r9/ksb.py run --out build/r9/ksb [--calls 1,4]
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
NW = 80
FLAGS = ["-DDIM_M=144", "-DDIM_N=64", "-DDIM_K=64", "-DRHO=9", "-DBLKU=1", "-DATTN_KEYS=1072", "-DATTN_SCALE=0.125f", "-O2",
         "-DMM_ASM", "-DMM_KSTEP", "-DMM_PIECES"]


def obj(out, ks):
    bd = ROOT / "build"
    b = f"{PEANO}/bin"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", f"-I{INC}",
            f"-I{ROOT}"]
    gen = ["python3", str(HERE / "gen_ks.py"), str(bd / "mm_ks.s")] if ks else \
        ["python3", str(ROOT / "r7" / "gen_mm.py"), "--pieces", str(bd / "ksb_core.s")]
    so = bd / ("mm_ks.s" if ks else "ksb_core.s")
    o = Path(out + ".o")
    steps = [gen,
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(so), "-o", str(so.with_suffix(".o"))],
             [f"{b}/clang++"] + base + FLAGS + (["-DMM_KS"] if ks else []) + ["-c", str(HERE / "ksb.cc"), "-o", out + "_cpp.o"],
             [f"{b}/ld.lld", "-r", out + "_cpp.o", str(so.with_suffix(".o")), "-o", str(o)]]
    for cmd in steps:
        subprocess.run(cmd, check=True)
    return str(o)


def build(o):
    import aie.iron as iron
    from aie.iron import CompileTime, Kernel, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.device import NPU2, Tile

    for ks in (False, True):
        out = str(Path(o.out).resolve()) + ("_ks" if ks else "_ref")  # absolute: IRON resolves Kernel objects in the .prj

        @iron.jit(aiecc_flags=["--dynamic-objFifos"])
        def bench(X: In, Y: Out, *, key: CompileTime[str] = ""):
            from aie.iron import Lock as L
            T = np.ndarray[(NW,), np.dtype[np.int32]]
            fn = Kernel("ksb", obj(out, ks), [T, T], stack_size_override=0x800)
            xf = ObjectFifo(T, name="X", depth=1)
            yf = ObjectFifo(T, name="Y", depth=1)
            t = Tile(0, 2)
            a_free = L(tile=t, init=0, name="aF", lock_id=4)
            a_full = L(tile=t, init=0, name="aU", lock_id=5)

            def core(xc, yp, fn, lf, lu):
                x = xc.acquire(1)
                y = yp.acquire(1)
                fn(x, y)
                xc.release(1)
                yp.release(1)

            w = Worker(core, [xf.cons(), yf.prod(), fn, a_free, a_full], tile=t, stack_size=0xA00, data_size=56 * 1024)

            def seq(x, y, xp, yc):
                from aie.dialects.aiex import npu_maskwrite32
                npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
                tg = TaskGroup()
                xp.fill(x, group=tg, wait=False)
                yc.drain(y, group=tg, wait=True)
                tg.finish()

            rt = Runtime(seq, [T, T, xf.prod(), yf.cons()])
            rt.add_lock(a_free)  # (their Tile is not the worker's object: register them so it gets resolved)
            rt.add_lock(a_full)
            return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

        iron.set_current_device(NPU2())
        bench.specialize(key=out).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
        print("built", out)


def call(out, n, timeout_ms):
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_x = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, kern.group_id(3))
    bo_y = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, kern.group_id(4))
    x = np.zeros(NW, np.int32)
    x[0] = n
    bo_x.write(x.tobytes(), 0)
    bo_x.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    st = kern(3, bo_i, len(instr), bo_x, bo_y).wait(timeout_ms * 1000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bo_y.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo_y.read(NW * 4, 0), np.int32).copy()


def run(o):
    res = {}
    for v in o.variants.split(","):
        for n in (int(c) for c in o.calls.split(",")):
            y = call(o.out + "_" + v, n, o.timeout_ms)
            res[v, n] = y
            print(f"{v:4s} {n} k-step(s): last call {int(y[0]) & 0xFFFFFFFF} cycles ({(int(y[0]) & 0xFFFFFFFF) / 9:.1f} per stripe), "
                  f"hash[0:4] {[hex(int(h) & 0xFFFFFFFF) for h in y[2:6]]}", flush=True)
    vs = o.variants.split(",")
    if len(vs) == 2:
        for n in (int(c) for c in o.calls.split(",")):
            a, b = res[vs[0], n][2:38], res[vs[1], n][2:38]
            bad = np.nonzero(a != b)[0]
            print(f"{n} k-step(s): {36 - len(bad)}/36 pieces identical" + (f"; differing (rb, pair): {[(i // 2, i % 2) for i in bad]}" if len(bad) else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", default="1,4")
    ap.add_argument("--variants", default="ref,ks")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
