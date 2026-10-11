"""pass 19 RELC: rel.cond's condition = any non-zero value, or one bit? (KTL's lock conditions are flag x counter: values up to 18 x nks.)
build (npu-iron, CPU): python3 r24/relcb.py build ; run (NPU): python3 r24/relcb.py run   (artefacts in r24/build/relc*)"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BD = HERE / "build"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
NW = 32
NOBJ = 8
OUT = str(BD / "relc")


def build():
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import DMAChannelDir
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (Acquire, Bd, Buffer, CompileTime, DmaChannel, Flow, In, Kernel, Lock, ObjectFifo, Out, Program,
                          Release, Runtime, TaskGroup, TileDma, Worker)
    from aie.iron.device import NPU2, Tile

    b = f"{PEANO}/bin"
    BD.mkdir(exist_ok=True)
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "-w", "--target=aie2p-none-unknown-elf", "-O2",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include"]
    for cmd in ([f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(HERE / "relc.s"), "-o", str(BD / "relc_s.o")],
                [f"{b}/clang++"] + base + ["-c", str(HERE / "relc.cc"), "-o", str(BD / "relc_cc.o")],
                [f"{b}/ld.lld", "-r", str(BD / "relc_cc.o"), str(BD / "relc_s.o"), "-o", str(BD / "relc.o")]):
        subprocess.run(cmd, check=True)

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def bench(X: In, Y: Out, A: In, *, key: CompileTime[str] = ""):
        T = np.ndarray[(NW,), np.dtype[np.int32]]
        A_l1 = np.ndarray[(288,), np.dtype[np.uint32]]
        A_host = np.ndarray[(NOBJ * 288,), np.dtype[np.uint32]]
        fn = Kernel("relcb", str(BD / "relc.o"), [T, T, A_l1], stack_size_override=0x400)
        xf, yf = ObjectFifo(T, name="X", depth=1), ObjectFifo(T, name="Y", depth=1)
        t, shim = Tile(0, 2), Tile(0, 0)
        a0 = Buffer(tile=t, type=A_l1, name="a0")
        free = Lock(tile=t, init=0, name="aF", lock_id=4)
        full = Lock(tile=t, init=0, name="aU", lock_id=5)
        fa = Flow(shim, t, src_channel=1, dst_channel=1)
        dma = TileDma(t, [DmaChannel(DMAChannelDir.S2MM, fa.endpoint(t), [Bd(a0, acquires=[Acquire(free)], releases=[Release(full)])])])

        def core(xc, yp, fn, a0):
            x, y = xc.acquire(1), yp.acquire(1)
            fn(x, y, a0)
            xc.release(1)
            yp.release(1)

        w = Worker(core, [xf.cons(), yf.prod(), fn, a0], tile=t, stack_size=0x800)

        def seq(x, y, a, xp, yc):
            tg = TaskGroup()
            xp.fill(x, group=tg, wait=False)
            fa.fill(a, tap=TensorAccessPattern((NOBJ * 288,), 0, [1, 1, NOBJ, 288], [0, 0, 288, 1]), group=tg, wait=False)
            yc.drain(y, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [T, T, A_host, xf.prod(), yf.cons()])
        rt.add_flow(fa)
        rt.add_tile_dma(dma)
        rt.add_lock(free)
        rt.add_lock(full)
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    bench.specialize(key=OUT + str((HERE / "relc.cc").stat().st_mtime) + str((HERE / "relc.s").stat().st_mtime)).compile(
        xclbin_path=OUT + ".xclbin", inst_path=OUT + ".bin")
    print("built", OUT)


def run():
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(OUT + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(OUT + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    a = np.zeros((NOBJ, 288), np.uint32)
    a[:, 0] = 0xA0000000 + np.arange(NOBJ)
    bos = [pyxrt.bo(dev, nb, pyxrt.bo.host_only, kern.group_id(3 + k)) for k, nb in enumerate((NW * 4, NW * 4, a.nbytes))]
    bos[0].write(np.full(NW, 7, np.int32).tobytes(), 0)
    bos[2].write(a.tobytes(), 0)
    for bo in (bos[0], bos[2]):
        bo.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    st = kern(3, bo_i, len(instr), *bos).wait(3_000_000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bos[1].sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    y = np.frombuffer(bos[1].read(NW * 4, 0), np.int32)
    for t in range(4):
        print(f"rel.cond with r26 = {int(y[3 * t]):3d}: released {'YES' if y[3 * t + 1] else 'no '}  (object tag {int(y[3 * t + 2]) & 0xFFFFFFFF:#x})")
    print("objects landed before the cleanup:", int(y[19]))


if __name__ == "__main__":
    dict(build=build, run=run)[sys.argv[1]]()
