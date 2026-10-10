"""Feasibility probe for design7: one shim MM2S channel carrying two packet-switched routes.

  route A: shim (0,0) ch0 --pkt 1--> memtile (0,1) static slot chain --> core (0,2) (circuit, static ring)
  route E: shim (0,0) ch0 --pkt 2--> core (0,3) directly, as an IRON ObjectFifo(packet=True)
Both cores xor-fold what they receive into a 256 B result; the host checks every word went through.
    python3 pkt_test.py --n-rep 8 --out build/pkt_test
"""

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    ExternalFunction,
    Flow,
    In,
    Lock,
    ObjectFifo,
    Out,
    Program,
    Release,
    Runtime,
    TaskGroup,
    TileDma,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.dataflow.flow import PacketFlow
from aie.iron.dataflow.tile_dma import BdIteration
from aie.iron.device import NPU2, Tile

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
W = 1024  # words per object (4 KB)
OUT = 64


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def pkt_test(X: In, E: In, Y: Out, Z: Out, *, n_rep: CompileTime[int] = 4, mode: CompileTime[int] = 0):
    Slot = np.ndarray[(W,), np.dtype[np.uint32]]
    Res = np.ndarray[(OUT,), np.dtype[np.uint32]]
    shim, mem, c2, c3 = Tile(0, 0), Tile(0, 1), Tile(0, 2), Tile(0, 3)
    flags = [f"-I{INC}", f"-DOBJ={W}", f"-DOUT={OUT}"]
    k = ExternalFunction("xfer_sum", object_file_name="xfer.o", source_file=str(HERE / "xfer.cc"), arg_types=[Slot, Res], compile_flags=flags)
    fin = ExternalFunction("xfer_fin", object_file_name="xfer_fin.o", source_file=str(HERE / "xfer.cc"), arg_types=[Res], compile_flags=flags + ["-DFIN"])
    k3 = ExternalFunction("xfer_sum_at", object_file_name="xfer_at.o", source_file=str(HERE / "xfer.cc"),
                          arg_types=[np.ndarray[(4 * W,), np.dtype[np.uint32]], Res, np.int32], compile_flags=flags + ["-DAT"])

    # route A: packet flow shim -> memtile, static slot, circuit memtile -> core (0,2)
    if mode == 0:
        a_in = PacketFlow(1, shim, mem, src_channel=0, dst_channel=0)
    else:
        a_in = Flow(shim, mem)
    a_out = Flow(mem, c2)
    slot = Buffer(tile=mem, type=Slot, name="slot")
    s_empty = Lock(tile=mem, init=1, name="se")
    s_full = Lock(tile=mem, init=0, name="sf")
    mem_dma = TileDma(mem, [
        DmaChannel(DMAChannelDir.S2MM, a_in.endpoint(mem), [Bd(slot, acquires=[Acquire(s_empty, value=1)], releases=[Release(s_full, value=1)])]),
        DmaChannel(DMAChannelDir.MM2S, a_out.endpoint(mem), [Bd(slot, acquires=[Acquire(s_full, value=1)], releases=[Release(s_empty, value=1)])]),
    ])
    abuf = Buffer(tile=c2, type=Slot, name="abuf")
    a_free = Lock(tile=c2, init=1, name="af")
    a_fill = Lock(tile=c2, init=0, name="au")
    core_dma = TileDma(c2, [DmaChannel(DMAChannelDir.S2MM, a_out.endpoint(c2), [Bd(abuf, acquires=[Acquire(a_free)], releases=[Release(a_fill)])])])

    # route E: packet object fifo on the same shim channel, straight to core (0,3)
    e_dma = None
    if mode == 0:
        e = ObjectFifo(Slot, name="E", depth=2, packet=True, packet_id=2)
        e.prod(tile=shim, channel=0)
    elif mode == 1:
        e = ObjectFifo(Slot, name="E", depth=2)
        e.prod(tile=shim)
    elif mode == 2:  # a second PacketFlow on the same shim channel, hand-built 2-deep ring on core (0,3)
        e = PacketFlow(2, shim, c3, src_channel=0, dst_channel=0)
        ebufs = [Buffer(tile=c3, type=Slot, name=f"eb{i}") for i in range(2)]
        e_free = Lock(tile=c3, init=2, name="ef")
        e_fill = Lock(tile=c3, init=0, name="eu")
        e_dma = TileDma(c3, [DmaChannel(DMAChannelDir.S2MM, e.endpoint(c3),
                                        [Bd(b, acquires=[Acquire(e_free)], releases=[Release(e_fill)]) for b in ebufs])])
    else:  # mode 3: as mode 2 but ONE BD iterating over the 4 slots of one buffer (BdIteration on a core DMA)
        e = PacketFlow(2, shim, c3, src_channel=0, dst_channel=0)
        Ring = np.ndarray[(4 * W,), np.dtype[np.uint32]]
        ering = Buffer(tile=c3, type=Ring, name="ering")
        ebufs = [ering]
        e_free = Lock(tile=c3, init=4, name="ef")
        e_fill = Lock(tile=c3, init=0, name="eu")
        e_dma = TileDma(c3, [DmaChannel(DMAChannelDir.S2MM, e.endpoint(c3), [Bd(
            ering, tap=TensorAccessPattern((4 * W,), 0, [1, 1, 1, W], [0, 0, 0, 1]),
            acquires=[Acquire(e_free)], releases=[Release(e_fill)], iteration=BdIteration(size=4, stride=W))])])
    y = ObjectFifo(Res, name="Y", depth=2)
    yc = y.cons(tile=shim)
    z = ObjectFifo(Res, name="Z", depth=2)
    zc = z.cons(tile=shim)

    def core_a(a_free, a_fill, abuf, yo, k, fin):
        ro = yo.acquire(1)
        fin(ro)
        for _ in range_(n_rep):
            a_fill.acquire(1)
            k(abuf, ro)
            a_free.release(1)
        yo.release(1)

    def core_e(ei, zo, k, fin):
        ro = zo.acquire(1)
        fin(ro)
        for _ in range_(n_rep):
            eo = ei.acquire(1)
            k(eo, ro)
            ei.release(1)
        zo.release(1)

    def core_e2(e_free, e_fill, eb0, eb1, zo, k, fin):
        ro = zo.acquire(1)
        fin(ro)
        for _ in range_(n_rep // 2):
            e_fill.acquire(1)
            k(eb0, ro)
            e_free.release(1)
            e_fill.acquire(1)
            k(eb1, ro)
            e_free.release(1)
        zo.release(1)

    def core_e3(e_free, e_fill, ering, zo, k, fin):  # consume the 4 ring slots in order
        ro = zo.acquire(1)
        fin(ro)
        for _ in range_(n_rep // 4):
            for i in range(4):
                e_fill.acquire(1)
                k3(ering, ro, i)
                e_free.release(1)
        zo.release(1)

    workers = [Worker(core_a, [a_free, a_fill, abuf, y.prod(), k, fin], tile=c2, stack_size=0x400)]
    if mode == 2:
        workers.append(Worker(core_e2, [e_free, e_fill, *ebufs, z.prod(), k, fin], tile=c3, stack_size=0x400))
    elif mode == 3:
        workers.append(Worker(core_e3, [e_free, e_fill, ering, z.prod(), k3, fin], tile=c3, stack_size=0x400))
    else:
        workers.append(Worker(core_e, [e.cons(), z.prod(), k, fin], tile=c3, stack_size=0x400))
    X_ty = np.ndarray[(n_rep * W,), np.dtype[np.uint32]]
    R_ty = np.ndarray[(OUT,), np.dtype[np.uint32]]

    def sequence(x, ee, yy, zz, E_prods, Y_conses, Z_conses):
        tg = TaskGroup()
        for i in range(n_rep):
            tap = TensorAccessPattern((n_rep * W,), i * W, [1, 1, 1, W], [0, 0, 0, 1])
            a_in.fill(x, tap=tap, group=tg, wait=False)
            if mode >= 2:
                e.fill(ee, tap=tap, group=tg, wait=False)
            else:
                E_prods[0].fill(ee, tap=tap, group=tg, wait=False)
        rtap = TensorAccessPattern((OUT,), 0, [1, 1, 1, OUT], [0, 0, 0, 1])
        Y_conses[0].drain(yy, tap=rtap, group=tg, wait=True)
        Z_conses[0].drain(zz, tap=rtap, group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [X_ty, X_ty, R_ty, R_ty, [] if mode >= 2 else [e.prod()], [yc], [zc]])
    for f in ((a_in, a_out, e) if mode >= 2 else (a_in, a_out)):
        rt.add_flow(f)
    for d in ((mem_dma, core_dma, e_dma) if mode >= 2 else (mem_dma, core_dma)):
        rt.add_tile_dma(d)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=4)
    ap.add_argument("--mode", type=int, default=0, help="0: shared packet channel; 1: two circuit channels (control)")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    iron.set_current_device(NPU2())
    pkt_test.specialize(n_rep=o.n_rep, mode=o.mode).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(n_rep=o.n_rep, mode=o.mode, W=W, OUT=OUT)))
