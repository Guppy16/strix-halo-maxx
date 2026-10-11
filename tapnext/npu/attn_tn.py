"""Fused attention v8 = v7's kernel with K/V broadcast: the 4 cores of a column work on the same head in lockstep
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
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
import os
COLS, ROWS = int(os.environ.get("ATTN_COLS", "8")), 4  # ATTN_COLS=1: a one-column build (the trace unit could not be routed beside 8 broadcast columns)
QT, HD, KC, NKEYS = 48, 64, 48, 1072
NCH = (NKEYS + KC - 1) // KC  # 16
PASSES = 4  # query tiles per head / cores per column
HEADS = 12
QTILES = 23          # ceil(1072 / 48) query tiles per head, padded to 24 = 6 passes of 4 cores
ITEMS = HEADS * 6 // COLS  # 9 (head, pass) items per core
OBJW = 8 * 448 // 4  # 896 words: Q / K objects = 6 groups of 8 panels (576 B); V^T objects = 8 groups of 6 panels padded to 448 B


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attention(KQ: In, O: Out, *, reps: CompileTime[int] = 1, kdefs: CompileTime[str] = "", trace_size: CompileTime[int] = 0):
    Chunk = np.ndarray[(OBJW,), np.dtype[np.uint32]]  # 48 x 72 as bfp16 panels in padded groups: 4224 B = 1056 words (Q, K, V^T)
    Qt = Chunk
    Ot = np.ndarray[(QT * HD,), np.dtype[bfloat16]]  # 48 x 72 bf16 out (hosts half the fp32 accumulator)
    flags = [f"-I{INC}", f"-DATTN_KEYS={NKEYS}", f"-DATTN_SCALE={1.0 / HD ** 0.5}f"] + kdefs.split()
    k_chunk = ExternalFunction("attn_chunk_k", object_file_name=f"attn_tn_{abs(hash(kdefs)) % 10000}.o", source_file=str(HERE / "attn_tn.cc"),
                               arg_types=[Qt, Chunk], compile_flags=flags)
    k_chunk_v = k_chunk.object_file.bind("attn_chunk_v", [Chunk])
    k_begin = k_chunk.object_file.bind("attn_begin", [Qt, Ot])
    k_end = k_chunk.object_file.bind("attn_end", [Ot])
    k_copy = k_chunk.object_file.bind("attn_copyq", [Qt, Chunk])
    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    per_item_kq = ROWS + 2 * NCH  # 4 Q objects (one per core of the column) + (K, V^T) pairs per item, one stream per column
    L2o = np.ndarray[(ROWS * QT * HD,), np.dtype[bfloat16]]
    kq_prods, kq_l1, o_conses, o_l1 = [], {}, [], {}
    for c in range(COLS):
        f = ObjectFifo(Chunk, name=f"KV{c}", depth=4)          # shim -> memtile -> the column's 4 cores (broadcast)
        kq_prods.append(f.prod(tile=shims[c]))
        l1 = f.cons().forward(tile=mems[c], obj_type=Chunk, name=f"KV{c}_L1", depth=int(os.environ.get("ATTN_L1DEPTH", "4")))  # one memtile channel, 4 consumers; depth 4: a chunk holds 2 objects, so depth 2 left nothing in flight
        for r in range(ROWS):
            kq_l1[(r, c)] = l1
        h = ObjectFifo(L2o, name=f"O{c}", depth=2)
        o_conses.append(h.cons(tile=shims[c]))
        subs = h.prod().join([QT * HD * r for r in range(ROWS)], tile=mems[c], obj_types=[Ot] * ROWS,
                             names=[f"O{c}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        for r in range(ROWS):
            o_l1[(r, c)] = subs[r]

    def core_fn(kq, o, q_buf, chunk, chunk_v, begin, end, copyq, r):
        for _ in range_(reps) if reps > 1 else range(1):
            for _ in range_(ITEMS):
                for i in range(ROWS):  # the pass's 4 Q tiles: core r keeps the r-th
                    qo = kq.acquire(1)
                    if i == r:
                        copyq(q_buf, qo)
                    kq.release(1)
                ob = o.acquire(1)   # the output object hosts half of the fp32 O accumulator for the whole item
                begin(q_buf, ob)
                for _ in range_(NCH):  # K then V^T, one object held at a time (prefetch needs free slots)
                    kk = kq.acquire(1)
                    chunk(q_buf, kk)
                    kq.release(1)
                    vv = kq.acquire(1)
                    chunk_v(vv)
                    kq.release(1)
                end(ob)
                o.release(1)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            qb = iron.Buffer(tile=Tile(c, 2 + r), type=Qt, name=f"q{r}{c}")
            workers.append(Worker(core_fn, [kq_l1[(r, c)].cons(), o_l1[(r, c)].prod(), qb,
                                            k_chunk, k_chunk_v, k_begin, k_end, k_copy, r], tile=Tile(c, 2 + r), stack_size=0x2800))

    n_kq = COLS * ITEMS * per_item_kq * OBJW  # words: one stream per column
    n_o = COLS * ROWS * ITEMS * QT * HD
    KQ_ty = np.ndarray[(n_kq * reps,), np.dtype[np.uint32]]
    O_ty = np.ndarray[(n_o,), np.dtype[bfloat16]]
    obj = OBJW  # words per bfp object
    oobj = QT * HD  # 3456 bf16 per O object

    def sequence(kq, o, KQp, Oc):
        # host layout: [c][obj index i][r][obj] so each column's L2 object i = [core 0 obj i][core 1 obj i]...
        tg = TaskGroup()
        for rep in range(reps):
            for c in range(COLS):
                nk = ITEMS * per_item_kq
                base = rep * n_kq + c * nk * obj  # one linear stream per column: [object][obj words]
                KQp[c].fill(kq, tap=TensorAccessPattern((n_kq * reps,), base, [1, 1, nk, obj], [0, 0, obj, 1]), group=tg, wait=False)
                no = ITEMS
                baseo = c * no * ROWS * oobj
                Oc[c].drain(o, tap=TensorAccessPattern((n_o,), baseo, [1, no, ROWS, oobj], [0, ROWS * oobj, oobj, 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [KQ_ty, O_ty, kq_prods, o_conses])
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size > 0:  # hardware trace of core (0, 2): event0/event1 (chunk), vector instrs, memory/stream/lock stalls
        evs = None
        if os.environ.get("ATTN_TRACE_EVENTS"):  # e.g. "INSTR_EVENT_0,INSTR_EVENT_1,ACTIVE,GROUP_STALL,INSTR_LOAD,INSTR_STORE,INSTR_VECTOR,INSTR_CALL"
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
    Path(o.out + ".json").write_text(json.dumps(dict(QT=QT, HD=HD, KC=KC, NKEYS=NKEYS, ITEMS=ITEMS, reps=o.reps, COLS=COLS, HEADS=HEADS, QTILES=QTILES)))
