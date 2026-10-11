"""Fused attention on 32 cores: 16 heads x 12 query tiles of 64 rows = 192 items, 6 per core.
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
QT, HD, KC, NKEYS = 64, 72, 32, 729
NCH = (NKEYS + KC - 1) // KC  # 23
ITEMS = 6  # per core: 16 heads x 12 query tiles / 32 cores


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attention(KQ: In, O: Out, *, reps: CompileTime[int] = 1, kdefs: CompileTime[str] = ""):
    Chunk = np.ndarray[(KC * HD,), np.dtype[bfloat16]]  # 32 keys x 72, blocked
    Qt = np.ndarray[(QT * HD,), np.dtype[bfloat16]]     # 64 x 72 = two chunk objects
    flags = [f"-I{INC}", f"-DATTN_KEYS={NKEYS}", f"-DATTN_SCALE={1.0 / 72 ** 0.5}f"] + kdefs.split()
    k_chunk = ExternalFunction("attn_chunk", object_file_name=f"attn{abs(hash(kdefs)) % 10000}.o", source_file=str(HERE / "attn.cc"),
                               arg_types=[Qt, Chunk, Chunk], compile_flags=flags)
    k_begin = k_chunk.object_file.bind("attn_begin", [Qt])
    k_end0 = k_chunk.object_file.bind("attn_end0", [Chunk])
    k_end1 = k_chunk.object_file.bind("attn_end1", [Chunk])
    k_copy = k_chunk.object_file.bind("attn_copyq", [Qt, Chunk, Chunk])
    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    per_item_kq = 2 + 3 * NCH  # Q (2 objects) + K chunks (pass 0) + K, V pairs (pass 1): one stream per core
    # (a memtile has 6 MM2S channels: two 4-way splits plus the O path would need 9)
    L2kq = np.ndarray[(ROWS * KC * HD,), np.dtype[bfloat16]]
    kq_prods, kq_l1, o_conses, o_l1 = [], {}, [], {}
    for c in range(COLS):
        f = ObjectFifo(L2kq, name=f"KQ{c}", depth=2)
        kq_prods.append(f.prod(tile=shims[c]))
        subs = f.cons().split([KC * HD * r for r in range(ROWS)], tile=mems[c], obj_types=[Chunk] * ROWS,
                              names=[f"KQ{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        for r in range(ROWS):
            kq_l1[(r, c)] = subs[r]
        h = ObjectFifo(L2kq, name=f"O{c}", depth=2)
        o_conses.append(h.cons(tile=shims[c]))
        subs = h.prod().join([KC * HD * r for r in range(ROWS)], tile=mems[c], obj_types=[Chunk] * ROWS,
                             names=[f"O{c}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        for r in range(ROWS):
            o_l1[(r, c)] = subs[r]

    def core_fn(kq, o, q_buf, chunk, begin, end0, end1, copyq):
        for _ in range_(reps) if reps > 1 else range(1):
            for _ in range_(ITEMS):
                qq = kq.acquire(2)  # Q tile = two consecutive chunk objects (acquire(n) = hold n in total)
                copyq(q_buf, qq[0], qq[1])
                kq.release(2)
                begin(q_buf)
                for _ in range_(NCH):  # pass 0
                    kc = kq.acquire(1)
                    chunk(q_buf, kc, kc)
                    kq.release(1)
                for _ in range_(NCH):  # pass 1: K chunk then V chunk on the same fifo
                    kv = kq.acquire(2)
                    chunk(q_buf, kv[0], kv[1])
                    kq.release(2)
                oa = o.acquire(1)
                end0(oa)
                o.release(1)
                ob = o.acquire(1)
                end1(ob)
                o.release(1)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            qb = iron.Buffer(tile=Tile(c, 2 + r), type=Qt, name=f"q{r}{c}")
            workers.append(Worker(core_fn, [kq_l1[(r, c)].cons(), o_l1[(r, c)].prod(), qb,
                                            k_chunk, k_begin, k_end0, k_end1, k_copy], tile=Tile(c, 2 + r), stack_size=0x2700))

    n_kq = COLS * ROWS * ITEMS * per_item_kq * KC * HD
    n_o = COLS * ROWS * ITEMS * 2 * KC * HD
    KQ_ty = np.ndarray[(n_kq * reps,), np.dtype[bfloat16]]
    O_ty = np.ndarray[(n_o,), np.dtype[bfloat16]]
    obj = KC * HD  # 2304 bf16 = 1152 words per object

    def sequence(kq, o, KQp, Oc):
        # host layout: [c][obj index i][r][obj] so each column's L2 object i = [core 0 obj i][core 1 obj i]...
        tg = TaskGroup()
        for rep in range(reps):
            for c in range(COLS):
                nk = ITEMS * per_item_kq
                base = rep * n_kq + c * nk * ROWS * obj
                KQp[c].fill(kq, tap=TensorAccessPattern((n_kq * reps,), base, [1, nk, ROWS, obj], [0, ROWS * obj, obj, 1]), group=tg, wait=False)
                no = ITEMS * 2
                baseo = c * no * ROWS * obj
                Oc[c].drain(o, tap=TensorAccessPattern((n_o,), baseo, [1, no, ROWS, obj], [0, ROWS * obj, obj, 1]), group=tg, wait=True)
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
