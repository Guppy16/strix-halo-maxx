"""Fused attention v5 (v4 + pipelined exp; see attn5.cc): 16 heads x 16 query tiles of 48 rows = 256 items, 8 per core.
Per item the core receives the Q tile as the first object of the K fifo, then 23 K chunks (pass 0), then 23 K
chunks + 23 V chunks (pass 1), and emits the O tile. Objects are 64 x 72 or 32 x 72 bf16 (both 4608 B = 2304 bf16
when the chunk is 32 keys x 72; the Q tile is 64 x 72 = 2 chunk-objects). All blocked [8x8].
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
COLS, ROWS = 8, 4
QT, HD, KC, NKEYS = 48, 72, 48, 729
NCH = (NKEYS + KC - 1) // KC  # 16
ITEMS = 8  # per core: 16 heads x 16 query tiles / 32 cores
OBJW = 6 * 704 // 4  # 1056 words: Q / K / V object = 6 groups of 9 bfp panels (648 B) padded to 704 B (64 B-aligned seeks)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attention(KQ: In, O: Out, *, reps: CompileTime[int] = 1, kdefs: CompileTime[str] = ""):
    Chunk = np.ndarray[(OBJW,), np.dtype[np.uint32]]  # 48 x 72 as bfp16 panels in padded groups: 4224 B = 1056 words (Q, K, V^T)
    Qt = Chunk
    Ot = np.ndarray[(QT * HD,), np.dtype[bfloat16]]  # 48 x 72 bf16 out (hosts half the fp32 accumulator)
    flags = [f"-I{INC}", f"-DATTN_KEYS={NKEYS}", f"-DATTN_SCALE={1.0 / 72 ** 0.5}f"] + kdefs.split()
    k_chunk = ExternalFunction("attn_chunk", object_file_name=f"attn5_{abs(hash(kdefs)) % 10000}.o", source_file=str(HERE / "attn5.cc"),
                               arg_types=[Qt, Chunk, Chunk], compile_flags=flags)
    k_begin = k_chunk.object_file.bind("attn_begin", [Qt, Ot])
    k_end = k_chunk.object_file.bind("attn_end", [Ot])
    k_copy = k_chunk.object_file.bind("attn_copyq", [Qt, Chunk])
    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    per_item_kq = 1 + 2 * NCH  # Q (1 object) + (K^T, V) pairs: one stream per core
    # (a memtile has 6 MM2S channels: two 4-way splits plus the O path would need 9)
    L2kq = np.ndarray[(ROWS * OBJW,), np.dtype[np.uint32]]
    L2o = np.ndarray[(ROWS * QT * HD,), np.dtype[bfloat16]]
    kq_prods, kq_l1, o_conses, o_l1 = [], {}, [], {}
    for c in range(COLS):
        f = ObjectFifo(L2kq, name=f"KQ{c}", depth=2)
        kq_prods.append(f.prod(tile=shims[c]))
        subs = f.cons().split([OBJW * r for r in range(ROWS)], tile=mems[c], obj_types=[Chunk] * ROWS,
                              names=[f"KQ{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        for r in range(ROWS):
            kq_l1[(r, c)] = subs[r]
        h = ObjectFifo(L2o, name=f"O{c}", depth=2)
        o_conses.append(h.cons(tile=shims[c]))
        subs = h.prod().join([QT * HD * r for r in range(ROWS)], tile=mems[c], obj_types=[Ot] * ROWS,
                             names=[f"O{c}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        for r in range(ROWS):
            o_l1[(r, c)] = subs[r]

    def core_fn(kq, o, q_buf, chunk, begin, end, copyq):
        for _ in range_(reps) if reps > 1 else range(1):
            for _ in range_(ITEMS):
                qo = kq.acquire(1)  # Q tile = one object; copied so the fifo slot is freed for the chunks
                copyq(q_buf, qo)
                kq.release(1)
                ob = o.acquire(1)   # the output object hosts half of the fp32 O accumulator for the whole item
                begin(q_buf, ob)
                for _ in range_(NCH):  # (K^T chunk, V chunk) pairs on the one fifo
                    kv = kq.acquire(2)
                    chunk(q_buf, kv[0], kv[1])
                    kq.release(2)
                end(ob)
                o.release(1)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            qb = iron.Buffer(tile=Tile(c, 2 + r), type=Qt, name=f"q{r}{c}")
            workers.append(Worker(core_fn, [kq_l1[(r, c)].cons(), o_l1[(r, c)].prod(), qb,
                                            k_chunk, k_begin, k_end, k_copy], tile=Tile(c, 2 + r), stack_size=0x2800))

    n_kq = COLS * ROWS * ITEMS * per_item_kq * OBJW  # words
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
                base = rep * n_kq + c * nk * ROWS * obj
                KQp[c].fill(kq, tap=TensorAccessPattern((n_kq * reps,), base, [1, nk, ROWS, obj], [0, ROWS * obj, obj, 1]), group=tg, wait=False)
                no = ITEMS
                baseo = c * no * ROWS * oobj
                Oc[c].drain(o, tap=TensorAccessPattern((n_o,), baseo, [1, no, ROWS, oobj], [0, ROWS * oobj, oobj, 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [KQ_ty, O_ty, kq_prods, o_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--kdefs", default="")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    iron.set_current_device(NPU2())
    attention.specialize(reps=o.reps, kdefs=o.kdefs).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(QT=QT, HD=HD, KC=KC, NKEYS=NKEYS, ITEMS=ITEMS, reps=o.reps)))
