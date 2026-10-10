"""GEMM-call bench (round 8): the asm mm_core on one core with its operands at chosen L1 offsets (mmb.cc).

build (npu-iron, no device):  python3 r8/mmb.py build --out build/mmb
run   (npu-iron, NPU):        python3 r8/mmb.py run --out build/mmb [--sweep NAME]
Cycles are the core timer's (P15: processor bus enabled from the runtime sequence).
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
NW = 64
POOL = 56 * 1024
A_B, B_B, C_B = 1152, 4608, 4096  # bytes one call reads: A 16 rows x k 64, B k 64 x 64 cols, C 16 x 64 fp32


def obj(out):
    bd = Path(out).resolve().parent
    b = f"{PEANO}/bin"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", f"-I{INC}"]
    steps = [["python3", str(ROOT / "r7" / "gen_mm.py"), str(bd / "mmb_core.s")],
             ["python3", str(ROOT / "r7" / "gen_mm.py"), str(bd / "mmb_split.s"), "--split"],
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "mmb_core.s"), "-o", str(bd / "mmb_core.o")],
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "mmb_split.s"), "-o", str(bd / "mmb_split.o")],
             [f"{b}/clang++"] + base + ["-O2", f"-DPOOL_BYTES={POOL}", "-c", str(HERE / "mmb.cc"), "-o", str(bd / "mmb_cpp.o")],
             [f"{b}/ld.lld", "-r", str(bd / "mmb_cpp.o"), str(bd / "mmb_core.o"), str(bd / "mmb_split.o"), "-o", str(bd / "mmb.o")]]
    for cmd in steps:
        subprocess.run(cmd, check=True)
    return str(bd / "mmb.o")


def build(o):
    import aie.iron as iron
    from aie.iron import Kernel, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.device import NPU2, Tile

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def bench(X: In, Y: Out):
        T = np.ndarray[(NW,), np.dtype[np.int32]]
        fn = Kernel("mmb", obj(o.out), [T, T], stack_size_override=0x400)
        xf = ObjectFifo(T, name="X", depth=1)
        yf = ObjectFifo(T, name="Y", depth=1)

        def core(xc, yp, fn):
            x = xc.acquire(1)
            y = yp.acquire(1)
            fn(x, y)
            xc.release(1)
            yp.release(1)

        w = Worker(core, [xf.cons(), yf.prod(), fn], tile=Tile(0, 2), stack_size=0x600, data_size=POOL + 1024)

        def seq(x, y, xp, yc):
            from aie.dialects.aiex import npu_maskwrite32
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            tg = TaskGroup()
            xp.fill(x, group=tg, wait=False)
            yc.drain(y, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [T, T, xf.prod(), yf.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    bench.compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")


class Dev:
    def __init__(self, out, timeout_ms):
        import pyxrt

        self.pyxrt = pyxrt
        dev = pyxrt.device(0)
        xb = pyxrt.xclbin(out + ".xclbin")
        dev.register_xclbin(xb)
        self.ctx = pyxrt.hw_context(dev, xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        instr = np.fromfile(out + ".bin", dtype=np.uint32)
        self.n_instr = len(instr)
        self.bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, self.kern.group_id(1))
        self.bo_i.write(instr.tobytes(), 0)
        self.bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_x = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, self.kern.group_id(3))
        self.bo_y = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, self.kern.group_id(4))
        self.timeout = timeout_ms * 1000

    def call(self, n, mode, offs):
        S = self.pyxrt.xclBOSyncDirection
        x = np.zeros(NW, np.int32)
        x[0], x[1] = n, mode
        x[2:2 + len(offs)] = offs
        self.bo_x.write(x.tobytes(), 0)
        self.bo_x.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        st = self.kern(3, self.bo_i, self.n_instr, self.bo_x, self.bo_y).wait(self.timeout)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        self.bo_y.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(self.bo_y.read(NW * 4, 0), np.int32).copy()


def per_call(d, mode, offs, n=200):
    """cycles per call: the slope between n and 2n calls (removes the loop and stamp constant)"""
    c1 = int(d.call(n, mode, offs)[0]) & 0xFFFFFFFF
    c2 = int(d.call(2 * n, mode, offs)[0]) & 0xFFFFFFFF
    return (c2 - c1) / n


def run(o):
    d = Dev(o.out, o.timeout_ms)
    y = d.call(1, 0, [0, 8192, 16384])
    base = int(y[1]) & 0xFFFFFFFF
    print(f"first dispatch OK: pool at 0x{base:05x}, 1 call {int(y[0])} cycles", flush=True)
    K = 1024

    def bank(off):
        return (base + off - 0x70000) // (16 * K)

    def show(label, mode, offs):
        c = per_call(d, mode, offs)
        print(f"{label:46s} offs {[f'{v // K}K' for v in offs]} banks {[bank(v) for v in offs]}: {c:7.1f} cycles/call", flush=True)
        return c

    show("empty loop", 2, [0, 0, 0])
    bs = [b * 16 * K - (base - 0x70000) for b in range(4)]  # pool offset of each 16 KB bank's start (may be < 0)
    print("bank starts in the pool:", bs)

    def at(b, off=0):  # an offset inside bank b, clear of the pool's ends
        return max(bs[b], 0) + off

    if o.sweep in ("all", "banks"):
        # every assignment of A, B, C(load) and C(store) to banks; split mode keeps loads and stores apart
        rows = []
        for ba in range(4):
            for bb in range(4):
                for bl in range(4):
                    for bst in range(4):
                        offs = [at(ba, 0), at(bb, 1536 if bb == ba else 0), at(bl, 6144 if bl in (ba, bb) else 0)]
                        offs.append(at(bst, 10240 if bst in (ba, bb, bl) else 0) if bst != bl else offs[2])
                        if max(offs[0] + A_B, offs[1] + B_B, offs[2] + C_B, offs[3] + C_B) > POOL:
                            continue
                        rows.append(((ba, bb, bl, bst), per_call(d, 1, offs)))
        print("A B Cload Cstore banks -> cycles/call (split call; same bank for Cload/Cstore = in place)")
        for k_, c in sorted(rows, key=lambda t: t[1]):
            print(f"  {k_}  {c:7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--sweep", default="all")
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
