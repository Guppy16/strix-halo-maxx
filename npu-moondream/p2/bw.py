"""DMA bandwidth probe: DDR -> shim -> memtile -> `fan` cores per stream, cores just ack.

`streams` streams per column (1 or 2, i.e. one or both shim MM2S channels), each
carrying `total` words, broadcast from the memtile to `fan` cores of its column.
Optional `out`: each core also sends back the same volume (S2MM path).
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def bw(
    A: In,
    C: Out,
    *,
    cols: CompileTime[int] = 8,
    streams: CompileTime[int] = 1,
    fan: CompileTime[int] = 4,
    l1: CompileTime[int] = 576,
    l2: CompileTime[int] = 1728,
    total: CompileTime[int] = 279936,
    write: CompileTime[int] = 0,
):
    L1 = np.ndarray[(l1,), np.dtype[np.uint32]]
    L2 = np.ndarray[(l2,), np.dtype[np.uint32]]
    n_obj = total // l1
    fifos, l1f = [], {}
    for c in range(cols):
        for s in range(streams):
            f = ObjectFifo(L2, name=f"S{c}_{s}", depth=2)
            fifos.append(f)
            l1f[(c, s)] = f.cons().forward(obj_type=L1, name=f"S{c}_{s}L1", depth=2)
    osz = l1 if write else 16
    out = [ObjectFifo(np.ndarray[(osz,), np.dtype[np.uint32]], name=f"O{c}", depth=2) for c in range(cols)]
    n_out = n_obj if write else 1

    def core_body(ins):
        for _ in range_(n_obj):
            for i in ins:
                i.acquire(1)
            for i in ins:
                i.release(1)

    def core1(i0, o):
        if write:
            for _ in range_(n_obj):
                i0.acquire(1)
                o.acquire(1)
                i0.release(1)
                o.release(1)
        else:
            core_body([i0])
            o.acquire(1)
            o.release(1)

    def core2(i0, i1, o):
        core_body([i0, i1])
        o.acquire(1)
        o.release(1)

    def core1n(i0):
        core_body([i0])

    def core2n(i0, i1):
        core_body([i0, i1])

    workers = []
    for c in range(cols):
        for r in range(fan):
            ins = [l1f[(c, s)].cons() for s in range(streams)]
            if r == 0:
                fn = core1 if streams == 1 else core2
                workers.append(Worker(fn, ins + [out[c].prod()], tile=Tile(c, 2 + r)))
            else:
                fn = core1n if streams == 1 else core2n
                workers.append(Worker(fn, ins, tile=Tile(c, 2 + r)))
    tot = cols * streams * total
    A_ty = np.ndarray[(tot,), np.dtype[np.uint32]]
    C_ty = np.ndarray[(cols * osz * n_out,), np.dtype[np.uint32]]

    def seq(a, cc, prods, conses):
        tg = TaskGroup()
        for j, p in enumerate(prods):
            a0 = 972 if total % 972 == 0 else next(d for d in range(1020, 0, -1) if total % d == 0)
            tap = TensorAccessPattern((tot,), j * total, [1, 1, total // a0, a0], [0, 0, a0, 1])
            p.fill(a, tap=tap, group=tg, wait=False)
        for c, q in enumerate(conses):
            q.drain(cc, tap=TensorAccessPattern((cols * osz * n_out,), c * osz * n_out, [1, 1, n_out, osz], [0, 0, osz, 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(seq, [A_ty, C_ty, [f.prod() for f in fifos], [f.cons() for f in out]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    for a, d in (("cols", 8), ("streams", 1), ("fan", 4), ("l1", 576), ("l2", 1728), ("total", 279936), ("write", 0)):
        p.add_argument(f"--{a}", type=int, default=d)
    p.add_argument("--out", required=True)
    o = p.parse_args()
    iron.set_current_device(NPU2())
    kw = {a: getattr(o, a) for a in ("cols", "streams", "fan", "l1", "l2", "total", "write")}
    bw.specialize(**kw).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    import json
    Path(o.out + ".json").write_text(json.dumps(kw))


if __name__ == "__main__":
    main()
