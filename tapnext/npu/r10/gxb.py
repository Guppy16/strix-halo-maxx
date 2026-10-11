"""Round 10: GELU loop bench, one core, no DMA: cycles per 32 lanes for the deployed loop (mode 0), the forced-II
pipelined loop in place (1) and into another bank (2), calls of n vectors.
    python3 r10/gxb.py build --out build/r10/gxb ; python3 r10/gxb.py run --out build/r10/gxb"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
FLAGS = [f"-I{INC}", f"-I{HERE.parent}", "-DDIM_M=144", "-DDIM_N=64", "-DDIM_K=64", "-DRHO=9", "-DBLKU=1", "-DATTN_KEYS=1072",
         "-DATTN_SCALE=0.125f", "-O2"]


def build(o):
    import aie.iron as iron
    from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.device import NPU2, Tile
    T = np.ndarray[(16,), np.dtype[np.int32]]

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def bench(X: In, Y: Out, *, key: CompileTime[str] = ""):
        fn = ExternalFunction("gxb", object_file_name="gxb.o", source_file=str(HERE / "gxb.cc"), arg_types=[T, T], compile_flags=FLAGS)
        xf, yf = ObjectFifo(T, name="X", depth=1), ObjectFifo(T, name="Y", depth=1)

        def core(xc, yp, fn):
            x, y = xc.acquire(1), yp.acquire(1)
            fn(x, y)
            xc.release(1)
            yp.release(1)

        w = Worker(core, [xf.cons(), yf.prod(), fn], tile=Tile(0, 2), stack_size=0x1000)

        def seq(x, y, xp, yc):
            from aie.dialects.aiex import npu_maskwrite32
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            tg = TaskGroup()
            xp.fill(x, group=tg, wait=False)
            yc.drain(y, group=tg, wait=True)
            tg.finish()

        return Program(iron.get_current_device(), Runtime(seq, [T, T, xf.prod(), yf.cons()]), workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    out = str(Path(o.out).resolve())
    bench.specialize(key=out).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")


def run(o):
    import sys
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_x = pyxrt.bo(dev, 64, pyxrt.bo.host_only, kern.group_id(3))
    bo_y = pyxrt.bo(dev, 64, pyxrt.bo.host_only, kern.group_id(4))
    for mode in (0, 1, 2, 3):
        for n in (8, 16, 144):
            x = np.zeros(16, np.int32)
            x[0], x[1] = mode, n
            bo_x.write(x.tobytes(), 0)
            bo_x.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
            st = kern(3, bo_i, len(instr), bo_x, bo_y).wait(3_000_000)
            assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
            bo_y.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
            y = np.frombuffer(bo_y.read(64, 0), np.uint32)
            print(f"mode {mode} n {n:3d}: {y[0] / 144:6.1f} cycles per 32 lanes, hash {y[1]:#010x}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
