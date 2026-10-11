"""Fused attention v10 = v8 (K/V broadcast) with K and V on separate fifos (two shim streams per column, two S2MM
channels per core) so the chunk DMA runs at twice the rate; the Q objects ride on the K fifo.

v8 notes: the 4 cores of a column work on the same head in lockstep
(4 query tiles of 48 rows per pass, 4 passes per head, 2 heads per column = 8 items per core), so each K/V chunk
crosses the shim once per column instead of once per core (v7 streamed 33.8 MB per block through the 8 shim
ports, a 0.66 ms floor at ~6.4 GB/s each; v8 streams 9.7 MB). The 4 Q tiles of a pass travel on the same fifo as
4 consecutive objects; core r copies the r-th. All objects are 4224 B (1056 words) of bfp16ebs8 panel groups.
"""

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
INC = "/work/mlir-aie/aie_kernels"
import os
COLS, ROWS = int(os.environ.get("ATTN_COLS", "8")), 4
QT, HD, KC, NKEYS = 48, 72, 48, 729
NCH = (NKEYS + KC - 1) // KC  # 16
PASSES = 4  # query tiles per head / cores per column
HEADS_PER_COL = 2
ITEMS = HEADS_PER_COL * PASSES  # 8 per core
OBJW = 6 * 704 // 4  # 1056 words: Q / K / V object = 6 groups of 9 bfp panels (648 B) padded to 704 B (64 B-aligned seeks)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attention(KQ: In, V: In, O: Out, *, reps: CompileTime[int] = 1, kdefs: CompileTime[str] = "", trace_size: CompileTime[int] = 0):
    Chunk = np.ndarray[(OBJW,), np.dtype[np.uint32]]  # 48 x 72 as bfp16 panels in padded groups: 4224 B = 1056 words (Q, K, V^T)
    Qt = Chunk
    Ot = np.ndarray[(QT * HD,), np.dtype[bfloat16]]  # 48 x 72 bf16 out (hosts half the fp32 accumulator)
    flags = [f"-I{INC}", f"-DATTN_KEYS={NKEYS}", f"-DATTN_SCALE={1.0 / 72 ** 0.5}f"] + kdefs.split()
    k_chunk = ExternalFunction("attn_chunk", object_file_name=f"attn11t_{abs(hash(kdefs)) % 10000}.o", source_file=str(HERE / "attn9.cc"),
                               arg_types=[Qt, Chunk, Chunk], compile_flags=flags)
    k_begin = k_chunk.object_file.bind("attn_begin", [Qt, Ot])
    k_end = k_chunk.object_file.bind("attn_end", [Ot])
    k_copy = k_chunk.object_file.bind("attn_copyq", [Qt, Chunk])
    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    per_item_kq = ROWS + NCH  # K stream per item: 4 Q objects + NCH K chunks
    per_item_v = NCH            # V stream per item
    L2o = np.ndarray[(ROWS * QT * HD,), np.dtype[bfloat16]]
    kq_prods, kq_l1, v_prods, v_l1, o_conses, o_l1 = [], {}, [], {}, [], {}
    for c in range(COLS):
        f = ObjectFifo(Chunk, name=f"K{c}", depth=3)           # Q (4 per item) + K chunks: shim -> memtile -> 4 cores (broadcast)
        kq_prods.append(f.prod(tile=shims[c]))
        l1 = f.cons().forward(tile=mems[c], obj_type=Chunk, name=f"K{c}_L1", depth=int(os.environ.get("ATTN_L1DEPTH", "3")))
        g = ObjectFifo(Chunk, name=f"V{c}", depth=3)           # V^T chunks on their own channel
        v_prods.append(g.prod(tile=shims[c]))
        vl1 = g.cons().forward(tile=mems[c], obj_type=Chunk, name=f"V{c}_L1", depth=int(os.environ.get("ATTN_L1DEPTH", "3")))
        for r in range(ROWS):
            kq_l1[(r, c)] = l1
            v_l1[(r, c)] = vl1
        h = ObjectFifo(L2o, name=f"O{c}", depth=2)
        o_conses.append(h.cons(tile=shims[c]))
        subs = h.prod().join([QT * HD * r for r in range(ROWS)], tile=mems[c], obj_types=[Ot] * ROWS,
                             names=[f"O{c}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        for r in range(ROWS):
            o_l1[(r, c)] = subs[r]

    def core_fn(kq, vf, o, q_buf, chunk, begin, end, copyq, r):
        for _ in range_(reps) if reps > 1 else range(1):
            for _ in range_(ITEMS):
                for i in range(ROWS):  # the pass's 4 Q tiles: core r keeps the r-th
                    qo = kq.acquire(1)
                    if i == r:
                        copyq(q_buf, qo)
                    kq.release(1)
                ob = o.acquire(1)   # the output object hosts half of the fp32 O accumulator for the whole item
                begin(q_buf, ob)
                for _ in range_(NCH):
                    kk = kq.acquire(1)
                    vv = vf.acquire(1)
                    chunk(q_buf, kk, vv)
                    kq.release(1)
                    vf.release(1)
                end(ob)
                o.release(1)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            qb = iron.Buffer(tile=Tile(c, 2 + r), type=Qt, name=f"q{r}{c}")
            workers.append(Worker(core_fn, [kq_l1[(r, c)].cons(), v_l1[(r, c)].cons(), o_l1[(r, c)].prod(), qb,
                                            k_chunk, k_begin, k_end, k_copy, r], tile=Tile(c, 2 + r), stack_size=0x2800))

    n_kq = COLS * ITEMS * per_item_kq * OBJW  # words: K stream, one per column
    n_v = COLS * ITEMS * per_item_v * OBJW
    n_o = COLS * ROWS * ITEMS * QT * HD
    KQ_ty = np.ndarray[(n_kq * reps,), np.dtype[np.uint32]]
    V_ty = np.ndarray[(n_v * reps,), np.dtype[np.uint32]]
    O_ty = np.ndarray[(n_o,), np.dtype[bfloat16]]
    obj = OBJW  # words per bfp object
    oobj = QT * HD  # 3456 bf16 per O object

    def sequence(kq, v, o, KQp, Vp, Oc):
        # host layout: [c][obj index i][r][obj] so each column's L2 object i = [core 0 obj i][core 1 obj i]...
        tg = TaskGroup()
        for rep in range(reps):
            for c in range(COLS):
                nk = ITEMS * per_item_kq
                base = rep * n_kq + c * nk * obj  # one linear stream per column: [object][obj words]
                KQp[c].fill(kq, tap=TensorAccessPattern((n_kq * reps,), base, [1, 1, nk, obj], [0, 0, obj, 1]), group=tg, wait=False)
                nv = ITEMS * per_item_v
                basev = rep * n_v + c * nv * obj
                Vp[c].fill(v, tap=TensorAccessPattern((n_v * reps,), basev, [1, 1, nv, obj], [0, 0, obj, 1]), group=tg, wait=False)
                no = ITEMS
                baseo = c * no * ROWS * oobj
                Oc[c].drain(o, tap=TensorAccessPattern((n_o,), baseo, [1, no, ROWS, oobj], [0, ROWS * oobj, oobj, 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [KQ_ty, V_ty, O_ty, kq_prods, v_prods, o_conses])
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size > 0:  # hardware trace of core (0, 2)
        evs = None
        if os.environ.get("ATTN_TRACE_EVENTS"):
            from aie.utils.trace.events import CoreEvent
            evs = [getattr(CoreEvent, n) for n in os.environ["ATTN_TRACE_EVENTS"].split(",")]
        prog.enable_trace(trace_size=trace_size, workers=[workers[0]], coretile_events=evs)
    return prog.resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--kdefs", default="")
    ap.add_argument("--trace-size", type=int, default=0)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    iron.set_current_device(NPU2())
    attention.specialize(reps=o.reps, kdefs=o.kdefs, trace_size=o.trace_size).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(QT=QT, HD=HD, KC=KC, NKEYS=NKEYS, ITEMS=ITEMS, reps=o.reps)))
