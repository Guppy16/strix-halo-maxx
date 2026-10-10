"""Round 18 (research rev 13 E2 / m1, DMAB): does a tile DMA writing into an L1 bank stall the core's accesses to that bank?

One column, one core (0_2). The memtile sends 63 x 1 KB from one slot (filled once from DDR) into a 4-slot ring D in the core's
bank 3 (free lock starts at 63, so the DMA never waits for the core). The core waits for the first slot, then runs NSEG timed
segments of a zero-overhead loop with one 64 B access per cycle (r18/dmab.s: vlda or vst at a fixed address) on a 64 B probe
buffer in bank 3 (same bank as the DMA) or bank 1 (other bank), stamps each segment, waits for all 63 slots and stamps the
DMA's end. Segments that overlap the DMA show the collision cost; the others are the baseline.

    python3 r18/dmab.py build --out build/r18/dmab     (npu-iron)
    python3 r18/dmab.py run --out build/r18/dmab       (npu-iron, NPU)
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
NSEG, TRIPS = 24, 64  # 64 trips x 4 bundles = 256 cycles of accesses per segment
NRES = 8 + 2 * NSEG
SLOTW = 256  # 1 KB per ring slot
GHZ = 1.81


def obj(out):
    b = f"{PEANO}/bin"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", "-O2"]
    for cmd in ([f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(HERE / "dmab.s"), "-o", out + "_s.o"],
                [f"{b}/clang++"] + base + [f"-DNSEG={NSEG}", f"-DTRIPS={TRIPS}", "-c", str(HERE / "dmab.cc"), "-o", out + "_cpp.o"],
                [f"{b}/ld.lld", "-r", out + "_cpp.o", out + "_s.o", "-o", out + ".o"]):
        subprocess.run(cmd, check=True)
    return out + ".o"


def build(o):
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import DMAChannelDir
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (Acquire, Bd, Buffer, CompileTime, DmaChannel, Flow, In, Kernel, Lock, ObjectFifo, Out, Program, Release,
                          Runtime, TaskGroup, TileDma, Worker)
    from aie.iron.dataflow.tile_dma import BdIteration
    from aie.iron.device import NPU2, Tile

    out = str(Path(o.out).resolve())
    Ring = np.ndarray[(4 * SLOTW,), np.dtype[np.uint32]]
    Probe = np.ndarray[(16,), np.dtype[np.uint32]]
    Arg = np.ndarray[(16,), np.dtype[np.int32]]
    Res = np.ndarray[(NRES,), np.dtype[np.uint32]]
    MemSlot = np.ndarray[(SLOTW,), np.dtype[np.uint32]]

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def dmab(SRC: In, ARG: In, RES: Out, *, key: CompileTime[str] = ""):
        from aie.dialects.aiex import npu_maskwrite32

        shim, mem, core = Tile(0, 0), Tile(0, 1), Tile(0, 2)
        k = Kernel("dmab_run", obj(out), [Ring, Probe, Probe, Arg, Res])
        d_in, d_out = Flow(shim, mem), Flow(mem, core)
        mslot = Buffer(tile=mem, type=MemSlot, name="ms")
        ring = Buffer(tile=core, type=Ring, name="dring", mem_bank=3)
        px = Buffer(tile=core, type=Probe, name="px", mem_bank=3)
        py = Buffer(tile=core, type=Probe, name="py", mem_bank=1)
        free = Lock(tile=core, init=63, name="dF", lock_id=6)
        full = Lock(tile=core, init=0, name="dU", lock_id=7)
        cdma = TileDma(core, [DmaChannel(DMAChannelDir.S2MM, d_out.endpoint(core), [Bd(
            ring, tap=TensorAccessPattern((4 * SLOTW,), 0, [1, 1, 1, SLOTW], [0, 0, 0, 1]),
            acquires=[Acquire(free)], releases=[Release(full)], iteration=BdIteration(size=4, stride=SLOTW))])])
        af, rf = ObjectFifo(Arg, name="arg", depth=1), ObjectFifo(Res, name="res", depth=1)

        def core_fn(ring, px, py, ac, rp, k, lf, lu):
            a, r = ac.acquire(1), rp.acquire(1)
            k(ring, px, py, a, r)
            ac.release(1)
            rp.release(1)

        w = Worker(core_fn, [ring, px, py, af.cons(), rf.prod(), k, free, full], tile=core, stack_size=0x400)
        Src = np.ndarray[(SLOTW,), np.dtype[np.uint32]]

        def seq(src, arg, res, ap, rc):
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            me_in, me_out = d_in.endpoint(mem), d_out.endpoint(mem)
            ld = me_in.task(Bd(mslot, tap=TensorAccessPattern((SLOTW,), 0, [1, 1, 1, SLOTW], [0, 0, 0, 1])))
            snd = me_out.task(Bd(mslot, tap=TensorAccessPattern((SLOTW,), 0, [1, 1, 1, SLOTW], [0, 0, 0, 1])))
            tg = TaskGroup()
            ap.fill(arg, group=tg, wait=False)
            d_in.fill(src, tap=TensorAccessPattern((SLOTW,), 0, [1, 1, 1, SLOTW], [0, 0, 0, 1]), group=tg, wait=False)
            ld.start()
            snd.start(repeat_count=62)  # 63 sends of 1 KB: the core's free lock (63) never blocks the DMA
            rc.drain(res, group=tg, wait=True)
            tg.finish()
            for t in (ld, snd):
                t.free()

        rt = Runtime(seq, [Src, Arg, Res, af.prod(), rf.cons()])
        for f in (d_in, d_out):
            rt.add_flow(f)
        rt.add_tile_dma(cdma)
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    key = out + str((HERE / "dmab.s").stat().st_mtime) + str((HERE / "dmab.cc").stat().st_mtime)
    dmab.specialize(key=key).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
    print("built", out)


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
    bo_s = pyxrt.bo(dev, SLOTW * 4, pyxrt.bo.host_only, kern.group_id(3))
    bo_s.write(np.arange(SLOTW, dtype=np.uint32).tobytes(), 0)
    bo_s.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_a = pyxrt.bo(dev, 64, pyxrt.bo.host_only, kern.group_id(4))
    bo_r = pyxrt.bo(dev, NRES * 4, pyxrt.bo.host_only, kern.group_id(5))
    names = {0: "load bank 3 (DMA bank)", 1: "load bank 1 (other)", 2: "store bank 3 (DMA bank)", 3: "store bank 1 (other)",
             4: "no access (nops)"}
    for mode in (int(m) for m in o.modes.split(",")):
        rows = []
        for it in range(o.iters):
            a = np.zeros(16, np.int32)
            a[0] = mode
            bo_a.write(a.tobytes(), 0)
            bo_a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
            st = kern(3, bo_i, len(instr), bo_s, bo_a, bo_r).wait(o.timeout_ms * 1000)
            assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
            bo_r.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
            rows.append(np.frombuffer(bo_r.read(NRES * 4, 0), np.uint32).astype(np.int64).copy())
        r = np.array(rows)
        dma_end, tail = r[:, 1], r[:, 3]  # cycles from the first slot to the DMA's end; the wait after the timed segments
        during, after = r[:, 8:8 + NSEG], r[:, 8 + NSEG:8 + 2 * NSEG]
        print(f"mode {mode} {names[mode]:24s}: DMA {np.median(dma_end):.0f} cycles from the first slot ({63 * 1024 / np.median(dma_end):.2f} "
              f"B/cycle; wait after the segments {np.median(tail):.0f} > 0 = all overlapped); per {4 * TRIPS}-access segment: during "
              f"{np.median(during):.1f} (mean {during.mean():.1f}), after {np.median(after):.1f} cycles", flush=True)
        if o.verbose:
            print("    during", " ".join(str(int(v)) for v in during[-1]), "\n    after ", " ".join(str(int(v)) for v in after[-1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="4,0,1,2,3")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--verbose", action="store_true")
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
