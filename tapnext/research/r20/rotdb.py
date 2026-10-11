"""rotdb.py - one-core k-step bench with a DMA-fed A ring (npu-research pass 15; main's round-18 request).

r19/rotb.py pre-released all nine A slots per call, so the kernel's acquires never blocked and round 18's g2f race (acq.cond one
bundle before the A fill) was invisible there. Here the three A slots are IRON Buffers on the core, filled by a shim -> core
circuit flow whose S2MM BD chain acquires aF (lock 4, init 3) and releases aU (lock 5) per 1,152 B object, the deployed
protocol (design8 core_dmas). The shim streams n x 9 objects from a host buffer at the shim / stream rate, below the core's
need (10,368 B per ~1,400 cycles), so the core waits in its acquires and the acquire -> fill distance is exercised.

Builds (both from r20/rotdb.cc): `ks` = mm_kstep -> mm_ks (deployed before ROT) and `rot` = rot_kstep -> mm_rot (variant --v,
read from --dir). Pre-registered for any correct kernel: the piece hashes of ks and rot agree and equal rotb's (pre-released)
for the same n; a kernel whose acquire sits < 6 bundles before its A fill (g2f) gives DIFFERENT, run-to-run varying hashes.

build (npu-iron, CPU only):  python3 r20/rotdb.py build --v g0a --dir /work/npu-tapnext/r19
run   (npu-iron, the NPU):   python3 r20/rotdb.py run --v g0a [--repeat 5]   (one dispatch each, 3 s timeout)
"""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # tapnext/research/r20
ROOT = HERE.parents[1] / "npu"
BD = HERE / "build"
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
NW = 80
NMAX = 4  # k-steps per dispatch (the kernel consumes all NMAX x 9 objects the shim streams; y[0] times the last)
SEED = 12345
FLAGS = ["-DDIM_M=144", "-DDIM_N=64", "-DDIM_K=64", "-DRHO=9", "-DBLKU=1", "-DATTN_KEYS=1072", "-DATTN_SCALE=0.125f", "-O2",
         "-DMM_ASM", "-DMM_KSTEP", "-DMM_PIECES", "-DMM_KS", "-ffunction-sections", "-fdata-sections"]


def a_host(n):
    """n k-steps x 9 objects of 1,152 B: object p = the slot pattern p % 3 that rotb.cc generates on the core"""
    s = SEED
    out = np.zeros(3 * 1152, np.uint8)
    for i in range(3 * 1152):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        r = s >> 8
        out[i] = (120 + r % 8) if i % 9 == 0 else (r & 0xFF)
    slots = out.reshape(3, 1152)
    return np.concatenate([slots[p % 3] for _ in range(n) for p in range(9)]).view(np.uint32)


def objects(v, rot, src):
    b = f"{PEANO}/bin"
    BD.mkdir(exist_ok=True)
    work = BD / f"inc_{v}"  # mm8_rot.inc includes "build/mm_rot.h": give it the variant's header (round 18 lesson)
    (work / "build").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "build" / "mm_ks.s", BD / "mm_ks.s")
    shutil.copy(Path(src) / f"mm_rot_{v}.s", BD / f"mm_rot_{v}.s")
    shutil.copy(Path(src) / f"mm_rot_{v}.h", work / "build" / "mm_rot.h")
    shutil.copy(ROOT / "r18" / "mm8_rot.inc", work / "mm8_rot.inc")
    out = BD / f"rotdb_{v}_{'rot' if rot else 'ks'}"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", f"-I{INC}",
            f"-I{work}", f"-I{ROOT}"]
    steps = [[f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(BD / "mm_ks.s"), "-o", str(BD / "mm_ks.o")],
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(BD / f"mm_rot_{v}.s"), "-o", str(BD / f"mm_rot_{v}.o")],
             [f"{b}/clang++"] + base + FLAGS + [f"-DROTB_ROT={int(rot)}", f"-DROTDB_N={NMAX}", "-c", str(HERE / "rotdb.cc"), "-o", f"{out}_cpp.o"],
             [f"{b}/ld.lld", "-r", "--gc-sections", "-u", "rotdb", f"{out}_cpp.o", str(BD / "mm_ks.o"), str(BD / f"mm_rot_{v}.o"),
              "-o", f"{out}.o"]]
    for cmd in steps:
        subprocess.run(cmd, check=True)
    return f"{out}.o"


def build(o):
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import DMAChannelDir
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (Acquire, Bd, Buffer, CompileTime, DmaChannel, Flow, In, Kernel, Lock, ObjectFifo, Out, Program,
                          Release, Runtime, TaskGroup, TileDma, Worker)
    from aie.iron.device import NPU2, Tile

    banks = [int(x) for x in o.banks.split(",")] if o.banks else None
    for rot in (False, True):
        obj = objects(o.v, rot, o.dir)
        out = obj[:-2]

        @iron.jit(aiecc_flags=["--dynamic-objFifos"])
        def bench(X: In, Y: Out, A: In, *, key: CompileTime[str] = ""):
            T = np.ndarray[(NW,), np.dtype[np.int32]]
            A_l1 = np.ndarray[(288,), np.dtype[np.uint32]]
            A_host = np.ndarray[(NMAX * 9 * 288,), np.dtype[np.uint32]]
            fn = Kernel("rotdb", obj, [T, T, A_l1, A_l1, A_l1], stack_size_override=0x800)
            xf = ObjectFifo(T, name="X", depth=1)
            yf = ObjectFifo(T, name="Y", depth=1)
            t, shim = Tile(0, 2), Tile(0, 0)
            bufs = [Buffer(tile=t, type=A_l1, name=f"a{i}", **(dict(mem_bank=banks[i]) if banks else {})) for i in range(3)]
            free = Lock(tile=t, init=3, name="aF", lock_id=4)
            full = Lock(tile=t, init=0, name="aU", lock_id=5)
            fa = Flow(shim, t, src_channel=1, dst_channel=1)
            dma = TileDma(t, [DmaChannel(DMAChannelDir.S2MM, fa.endpoint(t),
                                         [Bd(b, acquires=[Acquire(free)], releases=[Release(full)]) for b in bufs])])

            def core(xc, yp, fn, a0, a1, a2):
                x = xc.acquire(1)
                y = yp.acquire(1)
                fn(x, y, a0, a1, a2)
                xc.release(1)
                yp.release(1)

            w = Worker(core, [xf.cons(), yf.prod(), fn, *bufs], tile=t, stack_size=0xA00, data_size=49 * 1024)

            def seq(x, y, a, xp, yc):
                from aie.dialects.aiex import npu_maskwrite32
                npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)  # core timer: processor bus on
                tg = TaskGroup()
                xp.fill(x, group=tg, wait=False)
                # all n x 9 objects in one shim task: [9n objects][288 words] (dims <= 1023)
                fa.fill(a, tap=TensorAccessPattern((NMAX * 9 * 288,), 0, [1, 1, NMAX * 9, 288], [0, 0, 288, 1]), group=tg,
                        wait=False)
                yc.drain(y, group=tg, wait=True)
                tg.finish()

            rt = Runtime(seq, [T, T, A_host, xf.prod(), yf.cons()])
            rt.add_flow(fa)
            rt.add_tile_dma(dma)
            rt.add_lock(free)
            rt.add_lock(full)
            return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

        iron.set_current_device(NPU2())
        h = hashlib.md5(Path(obj).read_bytes()).hexdigest()[:8]  # IRON's jit cache ignores sources: key on the object
        bench.specialize(key=f"{out}_{h}").compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
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
    a = a_host(NMAX)
    bos = [pyxrt.bo(dev, nb, pyxrt.bo.host_only, kern.group_id(3 + k)) for k, nb in enumerate((NW * 4, NW * 4, a.nbytes))]
    x = np.zeros(NW, np.int32)
    x[0] = n
    bos[0].write(x.tobytes(), 0)
    bos[2].write(a.tobytes(), 0)
    for b in (bos[0], bos[2]):
        b.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    st = kern(3, bo_i, len(instr), *bos).wait(timeout_ms * 1000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bos[1].sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bos[1].read(NW * 4, 0), np.int32).copy()


def run(o):
    res = {}
    for var in ("ks", "rot"):
        for n in (int(c) for c in o.calls.split(",")):
            hs = []
            for _ in range(o.repeat):
                y = call(str(BD / f"rotdb_{o.v}_{var}"), n, o.timeout_ms)
                hs.append(tuple(y[2:38]))
            res[var, n] = hs
            stable = len(set(hs)) == 1
            print(f"{var:4s} {n:2d} k-step(s): last call {int(y[0]) & 0xFFFFFFFF} cycles; hashes {'stable' if stable else 'VARY'} "
                  f"over {o.repeat} runs", flush=True)
    for n in (int(c) for c in o.calls.split(",")):
        a, b = np.array(res["ks", n][0]), np.array(res["rot", n][0])
        bad = np.nonzero(a != b)[0]
        print(f"{n} k-step(s): {36 - len(bad)}/36 pieces identical (ks vs rot)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--v", default="g0a")
    ap.add_argument("--dir", default=str(ROOT / "r19"), help="where mm_rot_<v>.s / .h live")
    ap.add_argument("--banks", default="", help="A slot banks, e.g. 1,2,2 (deployed); default: IRON's placement")
    ap.add_argument("--calls", default=f"{NMAX}", help="kept for rotb compatibility: the kernel always runs NMAX k-steps")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout-ms", type=int, default=3000)
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
