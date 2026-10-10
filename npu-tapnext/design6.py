"""design3 with ONE memtile A buffer per column (31104 words = a K=1152 M-tile) and one lock pair.

A cached op is one load BD whose "full" release is 2 x nt (the send chain is two
half-buffer BDs, so one N-tile = two sends); a streamed op (K a multiple of 1152)
loads one 1152-wide K-chunk per BD, released by 2 and sent once. fc2 (K 4608) then
needs 4 chunks x 3 N-tiles = 12 load BDs and the whole block chain is 15 loads +
2 sends, inside the 24-BD half. One xclbin runs all 27 blocks; the projection
(K 2304 does not fit one buffer) is left to the GPU. Original docstring follows.


Same mapping as design.py (core column c owns M-tile c, core row r a run of
N-tiles) and the same B / C ObjectFifos, but A reaches the cores from two
slots in the column's memtile instead of from DDR once per N-tile.

design2.py drove the memtile slots with runtime-sequence DMA tasks and
deadlocked. Here the memtile DMA program is static (configured when the
xclbin loads, like an ObjectFifo's) and the per-op repeat lives in the lock
values, not in a task queue:

* Every op's A M-tile is split by K-halves over the two slots, so the
  memtile's send chain is always [slot0, slot1] (one N-tile's A per pass of
  the chain) and loops forever.
* A load into a slot releases its "full" lock by nt (the op's N-tiles per core
  row): the send chain then re-reads the slot nt times before it blocks. The
  next load into that slot acquires "empty" >= nt, i.e. waits until all nt
  sends are done. Different ops therefore just need different lock values on
  their load BDs; the schedule is compile time anyway.
* An op whose A does not fit both slots (K > 2 * RA_K: fc2, proj_fc2) streams
  A from DDR once per N-tile through the slots as a ring (q even chunks per
  pass, each sent once), exactly like design.py's traffic for that op.

Cost: no prefetch of the next op's A while the current op runs (one M-tile
load, ~10-20 us, between ops). Fix later with four half-slots if it matters.
"""

import argparse
import os
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Kernel,
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
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from design import COLS, INC, ROWS, plan, tap1d, words_a

HERE = Path(__file__).resolve().parent
RA_K = int(os.environ.get("RA_K", "1152"))  # one memtile slot holds half an M-tile of A for K <= 2 * RA_K (TAPNext: 768)
L1_WORDS = 288  # one A sub-tile (16 x 64 bfp16ebs8) = the core's L1 object


def slot_unit_unused(ops, ra):
    """Words one send BD moves from a slot: the same for every op of a schedule.

    Cached ops (A fits both slots) fix it at half their M-tile; a schedule
    with only streaming ops takes the largest chunk that fits a slot.
    """
    units = {op["la"] // 2 for op in ops if op["la"] <= 2 * ra}
    assert len(units) <= 1, ("cached ops must share K", units)
    if units:
        return units.pop()
    la = ops[0]["la"]
    q = next(
        q for q in range(4, 65, 2)
        if la % q == 0 and la // q <= ra and (la // q) % L1_WORDS == 0
    )
    return la // q


def a_chunks(op, buf):
    """(cached, chunk_words): how one op's A M-tile goes through the single buffer.

    Cached (la == buf): loaded once, re-sent nt times. Streaming: la / buf
    chunks per N-tile pass, each loaded and sent once.
    """
    la = op["la"]
    cached = la == buf
    if not cached:
        assert la % buf == 0 and la > buf, ("pad K to a multiple of 1152", la, buf)
    return cached, [buf] * (la // buf)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemm_seq6(
    A: In,
    W: In,
    C: Out,
    *,
    sched: CompileTime[str] = "[[1152, 9]]",
    m: CompileTime[int] = 96,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 96,
    rho: CompileTime[int] = 6,
    reps: CompileTime[int] = 1,
    b_depth: CompileTime[int] = 2,
    l2_depth: CompileTime[int] = 2,
    kdefs: CompileTime[str] = "",
    lookahead: CompileTime[int] = 1,
    c_linear: CompileTime[int] = 0,  # 1: C leaves the memtile blocked [m/8][n/8][8][8] (2 BDs instead of 8)
    b_streams: CompileTime[int] = 1,  # >1 needs a memtile join: a core has only two S2MM channels (A, B)
    c_streams: CompileTime[int] = 2,  # shim streams per column for C (each joins ROWS/c_streams core rows)
    a_ring: CompileTime[int] = 6,  # A sub-tile buffers per core; 2 (288 DMA cycles each) was latency-bound
    nosync: CompileTime[int] = 0,  # 1: cores skip the A/B lock handshakes (timing probe: compute + C path only)
):
    assert rho % a_ring == 0, "the sub-tile -> L1 buffer map must repeat every k-step"
    schedule = [tuple(x) for x in json.loads(sched)]
    ops, a_tot, w_tot, c_tot = plan(schedule, m, k, n, rho)
    m_a = m // rho
    wa, wb = words_a(m_a, k), words_a(k, n)
    assert wa == L1_WORDS, wa
    wb0 = next(d for d in range(min(wb, 1020), 0, -1) if wb % d == 0)  # shim wrap <= 1020
    ra = words_a(m, RA_K)  # the one A buffer: a 96 x 1152 M-tile
    unit = ra // 2  # one send BD = half the buffer
    assert unit % L1_WORDS == 0, unit

    A_l1 = np.ndarray[(wa,), np.dtype[np.uint32]]
    B_l1 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    B_l2 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l2 = np.ndarray[(ROWS * m * n,), np.dtype[bfloat16]]
    slot_ty = np.ndarray[(ra,), np.dtype[np.uint32]]

    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}"]
    obj = os.environ.get("GEMM_OBJ")  # phase 11: a prebuilt object (C++ wrapper + hand-scheduled loop merged by ld.lld -r)
    if obj:
        mm = Kernel("mm_acc", obj, [A_l1, B_l1, C_l1], stack_size_override=0x800)
    else:
        mm = ExternalFunction(
            "mm_acc",
            object_file_name="mm_f32acc.o",
            source_file=str(HERE / "mm_f32acc.cc"),
            arg_types=[A_l1, B_l1, C_l1],
            compile_flags=flags + kdefs.split(),
        )
    zero = mm.object_file.bind("acc_zero", [C_l1])
    store = mm.object_file.bind("acc_store", [C_l1])

    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    cores = [[Tile(c, 2 + r) for c in range(COLS)] for r in range(ROWS)]

    # --- A: DDR -> memtile slots -> broadcast to the column's cores ---------
    a_in = [Flow(shims[c], mems[c]) for c in range(COLS)]
    a_out = [Flow(mems[c], [cores[r][c] for r in range(ROWS)]) for c in range(COLS)]

    # One schedule pass of loads: (words, acquire_value, release_value). Every
    # load fills the whole buffer; `pending` = sends (half-buffer BDs) its
    # content still gets = 2 x nt for a cached op, 2 for a streamed chunk.
    load_plan, pending = [], 0
    for op in ops:
        cached, chunks = a_chunks(op, ra)
        sends = 2 * (op["nt"] if cached else 1)
        for _ in range(1 if cached or len(ops) == 1 else op["nt"]):
            for words in chunks:
                load_plan.append((words, pending, sends))
                pending = sends
    empty_init = pending
    load_plan = [(w, acq if acq else empty_init, rel) for (w, acq, rel) in load_plan]
    assert all(v <= 63 for (_, a, r) in load_plan for v in (a, r)), load_plan
    assert len(load_plan) + 2 <= 22, ("A chain does not fit the 24-BD half", len(load_plan))

    mem_dmas = []
    for c in range(COLS):
        slot = Buffer(tile=mems[c], type=slot_ty, name=f"As{c}")
        empty = Lock(tile=mems[c], init=empty_init, name=f"Ae{c}")
        full = Lock(tile=mems[c], init=0, name=f"Af{c}")

        def tap(off, words):
            return TensorAccessPattern((ra,), off, [1, 1, 1, words], [0, 0, 0, 1])

        loads = [
            Bd(slot, tap=tap(0, w), acquires=[Acquire(empty, value=acq)], releases=[Release(full, value=rel)])
            for (w, acq, rel) in load_plan
        ]
        # The send chain is the same two half-buffer BDs for every op; the lock
        # values on the loads decide how often they run.
        sends = [
            Bd(slot, tap=tap(s * unit, unit), acquires=[Acquire(full, value=1)], releases=[Release(empty, value=1)])
            for s in range(2)
        ]
        mem_dmas.append(TileDma(mems[c], [
            DmaChannel(DMAChannelDir.S2MM, a_in[c].endpoint(mems[c]), loads),
            DmaChannel(DMAChannelDir.MM2S, a_out[c].endpoint(mems[c]), sends),
        ]))

    a_l1 = {}
    core_dmas = []
    for r in range(ROWS):
        for c in range(COLS):
            t = cores[r][c]
            bufs = [Buffer(tile=t, type=A_l1, name=f"a{r}{c}_{i}") for i in range(a_ring)]
            free = Lock(tile=t, init=a_ring, name=f"aF{r}{c}")
            fill = Lock(tile=t, init=0, name=f"aU{r}{c}")
            a_l1[(r, c)] = (bufs, free, fill)
            core_dmas.append(TileDma(t, [DmaChannel(
                DMAChannelDir.S2MM, a_out[c].endpoint(t),
                [Bd(b, acquires=[Acquire(free)], releases=[Release(fill)]) for b in bufs],
            )]))

    # --- B and C: ObjectFifos as in design.py, C spread over both shim S2MM --
    # channels of its column (one stream moves ~6.4 GB/s; C over 8 streams was
    # a 104 us floor for qkv). B stays one stream per row: a second one would
    # need a memtile join, and the core's own input port (also ~6.4 GB/s) sees
    # the same bytes anyway -- that floor needs B shared between neighbours.
    assert ROWS * b_streams <= COLS and ROWS % c_streams == 0 and all(
        (K // k) % b_streams == 0 for K, _ in schedule), (b_streams, c_streams, schedule)
    b_prods, b_l2l1 = [], []  # b_prods flat: index r * b_streams + s
    for r in range(ROWS):
        row_l1 = []
        for s in range(b_streams):
            i = r * b_streams + s
            f = ObjectFifo(B_l2, name=f"B{r}_{s}", depth=l2_depth)
            b_prods.append(f.prod(tile=shims[i]))
            row_l1.append(f.cons().forward(tile=mems[i], obj_type=B_l1, name=f"B{r}_{s}L1", depth=b_depth))
        b_l2l1.append(row_l1)
    c_out = TensorAccessPattern((m * n,), 0, [m // 8, 8, n // 8, 8], [8 * n, 8, 64, 1])
    rows_per_c = ROWS // c_streams
    C_l2 = np.ndarray[(rows_per_c * m * n,), np.dtype[bfloat16]]
    c_conses, c_l1l2 = [], [[None] * COLS for _ in range(ROWS)]  # c_conses flat: c * c_streams + s
    for c in range(COLS):
        for s in range(c_streams):
            f = ObjectFifo(C_l2, name=f"C{c}_{s}", depth=2, to_stream=None if c_linear else c_out)
            c_conses.append(f.cons(tile=shims[c]))
            subs = f.prod().join(
                [m * n * i for i in range(rows_per_c)],
                tile=mems[c],
                obj_types=[C_l1] * rows_per_c,
                names=[f"C{c}_{s}_{i}" for i in range(rows_per_c)],
                depths=[1] * rows_per_c,
            )
            for i in range(rows_per_c):
                c_l1l2[s * rows_per_c + i][c] = subs[i]

    def core_fn(a_free, a_full, *rest):
        abufs, rest = rest[:a_ring], rest[a_ring:]
        in_b, (out_c, mm, zero, store) = rest[:b_streams], rest[b_streams:]
        if nosync >= 2:  # timing probe: compute on whatever the B buffers hold
            bstatic = [f.acquire(1) for f in in_b]

        def one_op(K, nt):
            for _ in range_(nt) if nt > 1 else range(1):
                cbuf = out_c.acquire(1)
                zero(cbuf)
                for _ in range_(K // k // b_streams):
                    for s in range(b_streams):
                        b = in_b[s].acquire(1) if nosync < 2 else bstatic[s]
                        for j in range(rho):
                            if not nosync:
                                a_full.acquire(1)
                            mm(abufs[j % a_ring], b, cbuf)
                            if not nosync:
                                a_free.release(1)
                        if nosync < 2:
                            in_b[s].release(1)
                store(cbuf)
                out_c.release(1)

        for _ in range_(reps) if reps > 1 else range(1):
            for K, nt in schedule:
                one_op(K, nt)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            bufs, free, fill = a_l1[(r, c)]
            workers.append(Worker(
                core_fn,
                [free, fill, *bufs, *[f.cons() for f in b_l2l1[r]],
                 c_l1l2[r][c].prod(), mm, zero, store],
                tile=cores[r][c],
                stack_size=0x1400,
            ))

    A_ty = np.ndarray[(a_tot,), np.dtype[np.uint32]]
    W_ty = np.ndarray[(w_tot * reps,), np.dtype[np.uint32]]
    C_ty = np.ndarray[(c_tot,), np.dtype[bfloat16]]

    def sequence(a, w, cc, B_prods, C_conses):
        pending = []
        for rep in range(reps):
            for op in ops:
                tg = TaskGroup()
                cached, _ = a_chunks(op, ra)
                nt = op["nt"]
                for c in range(COLS):
                    a_in[c].fill(
                        a, tap=tap1d(a_tot, op["a"] + c * op["la"], op["la"], 1 if cached else nt),
                        group=tg, wait=False,
                    )
                K = op["K"]
                for r in range(ROWS):
                    for s in range(b_streams):
                        # stream s takes k-steps s, s + b_streams, ... of each N-tile
                        tap = TensorAccessPattern(
                            (w_tot * reps,), rep * w_tot + op["w"] + r * nt * op["lb"] + s * wb,
                            [nt, K // k // b_streams, wb // wb0, wb0],
                            [K // k * wb, b_streams * wb, wb0, 1],
                        )
                        B_prods[r * b_streams + s].fill(w, tap=tap, group=tg, wait=False)
                N = op["N"]
                for c in range(COLS):
                    for s in range(c_streams):
                        tap = TensorAccessPattern(
                            (c_tot,), op["c"] + c * m * N + s * rows_per_c * nt * n,
                            [nt, rows_per_c, m, n], [n, nt * n, N, 1],
                        )
                        C_conses[c * c_streams + s].drain(cc, tap=tap, group=tg, wait=True)
                pending.append(tg)
                if len(pending) >= lookahead:
                    pending.pop(0).finish()
        for tg in pending:
            tg.finish()

    rt = Runtime(
        sequence,
        [A_ty, W_ty, C_ty, b_prods, c_conses],
    )
    for f in (*a_in, *a_out):
        rt.add_flow(f)
    for d in (*mem_dmas, *core_dmas):
        rt.add_tile_dma(d)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sched", default="[[1152, 9]]")
    p.add_argument("-m", type=int, default=96)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=96)
    p.add_argument("--rho", type=int, default=6)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--b-depth", type=int, default=2)
    p.add_argument("--l2-depth", type=int, default=2)
    p.add_argument("--lookahead", type=int, default=1)
    p.add_argument("--kdefs", default="-DOPAQUE_A -DNO_PREFETCH -DPIPE=1")
    p.add_argument("--c-linear", type=int, default=0)
    p.add_argument("--b-streams", type=int, default=1)
    p.add_argument("--c-streams", type=int, default=2)
    p.add_argument("--a-ring", type=int, default=6)
    p.add_argument("--nosync", type=int, default=0)
    p.add_argument("--out", required=True)
    o = p.parse_args()
    iron.set_current_device(NPU2())
    kw = dict(
        sched=o.sched, m=o.m, k=o.k, n=o.n, rho=o.rho, reps=o.reps,
        b_depth=o.b_depth, l2_depth=o.l2_depth, kdefs=o.kdefs, lookahead=o.lookahead,
        c_linear=o.c_linear, b_streams=o.b_streams, c_streams=o.c_streams, a_ring=o.a_ring,
        nosync=o.nosync,
    )
    gemm_seq6.specialize(**kw).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(kw))


if __name__ == "__main__":
    main()
