"""design.py with A (activations) kept resident in the memtiles.

Same mapping as design.py (core column c owns M-tile c, core row r a run of
N-tiles), but A no longer comes from DDR once per N-tile. Each column's M-tile
of A is loaded once into its memtile (two slots of RA words, so the next op's A
can load while this one runs) and the memtile re-sends it to the column's four
cores once per N-tile, with a repeat count set per op from the runtime
sequence. ObjectFifos cannot do that (their repeat is a compile-time constant),
so the A path is explicit: Flows, Buffers, Locks, a static ping-pong DMA
program on each core and runtime-sequence tasks on the memtile.

An op whose A M-tile does not fit both slots (K > 2 * RA_K, i.e. fc2 and
proj_fc2) streams A from DDR once per N-tile through the two slots as a ring.
B and C keep the ObjectFifo paths of design.py.
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
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from design import COLS, INC, ROWS, plan, tap1d, words_a

HERE = Path(__file__).resolve().parent
RA_K = 1152  # one memtile slot holds an M-tile of A for K <= RA_K


def a_passes(op, ra):
    """How one op's A moves through the memtile slots.

    Returns (cached, passes): each pass is a list of (slot, words) chunks.
    Cached: one pass, loaded once, re-sent nt times. Otherwise nt passes,
    each re-loaded from DDR in chunks alternating between the slots.
    """
    la, nt = op["la"], op["nt"]
    if la <= 2 * ra:
        span = 1 if la <= ra else 2
        return True, [[(None, la // span)] * span]
    q = next(q for q in range(3, 64) if la % q == 0 and la // q <= ra)
    return False, [[(None, la // q)] * q for _ in range(nt)]


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemm_seq2(
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
    lookahead: CompileTime[int] = 2,
):
    assert rho % 2 == 0, "A sub-tiles alternate between two L1 buffers per k-step"
    schedule = [tuple(x) for x in json.loads(sched)]
    ops, a_tot, w_tot, c_tot = plan(schedule, m, k, n, rho)
    m_a = m // rho
    wa, wb = words_a(m_a, k), words_a(k, n)
    ra = words_a(m, RA_K)

    A_l1 = np.ndarray[(wa,), np.dtype[np.uint32]]
    B_l1 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    B_l2 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l2 = np.ndarray[(ROWS * m * n,), np.dtype[bfloat16]]
    slot_ty = np.ndarray[(ra,), np.dtype[np.uint32]]

    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}"]
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
    slots = [[Buffer(tile=mems[c], type=slot_ty, name=f"As{c}_{s}") for s in range(2)]
             for c in range(COLS)]
    s_empty = [[Lock(tile=mems[c], init=1, name=f"Ae{c}_{s}") for s in range(2)]
               for c in range(COLS)]
    s_full = [[Lock(tile=mems[c], init=0, name=f"Af{c}_{s}") for s in range(2)]
              for c in range(COLS)]

    a_l1 = {}
    core_dmas = []
    for r in range(ROWS):
        for c in range(COLS):
            t = cores[r][c]
            bufs = [Buffer(tile=t, type=A_l1, name=f"a{r}{c}_{i}") for i in range(2)]
            free = Lock(tile=t, init=2, name=f"aF{r}{c}")
            full = Lock(tile=t, init=0, name=f"aU{r}{c}")
            a_l1[(r, c)] = (bufs, free, full)
            core_dmas.append(TileDma(t, [DmaChannel(
                DMAChannelDir.S2MM, a_out[c].endpoint(t),
                [Bd(b, acquires=[Acquire(free)], releases=[Release(full)]) for b in bufs],
            )]))

    # --- B and C: ObjectFifos as in design.py --------------------------------
    b_l3l2, b_l2l1 = [], []
    for r in range(ROWS):
        f = ObjectFifo(B_l2, name=f"B{r}", depth=l2_depth)
        b_l3l2.append(f)
        b_l2l1.append(f.cons().forward(obj_type=B_l1, name=f"B{r}L1", depth=b_depth))
    c_out = TensorAccessPattern((m * n,), 0, [m // 8, 8, n // 8, 8], [8 * n, 8, 64, 1])
    c_l2l3, c_l1l2 = [], [[None] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        f = ObjectFifo(C_l2, name=f"C{c}", depth=2, to_stream=c_out)
        c_l2l3.append(f)
        subs = f.prod().join(
            [m * n * r for r in range(ROWS)],
            obj_types=[C_l1] * ROWS,
            names=[f"C{c}_{r}" for r in range(ROWS)],
            depths=[1] * ROWS,
        )
        for r in range(ROWS):
            c_l1l2[r][c] = subs[r]

    def core_fn(a0, a1, a_free, a_full, in_b, out_c, mm, zero, store):
        def one_op(K, nt):
            for _ in range_(nt) if nt > 1 else range(1):
                cbuf = out_c.acquire(1)
                zero(cbuf)
                for _ in range_(K // k):
                    b = in_b.acquire(1)
                    for j in range(rho):
                        a_full.acquire(1)
                        mm(a0 if j % 2 == 0 else a1, b, cbuf)
                        a_free.release(1)
                    in_b.release(1)
                store(cbuf)
                out_c.release(1)

        for _ in range_(reps) if reps > 1 else range(1):
            for K, nt in schedule:
                one_op(K, nt)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            bufs, free, full = a_l1[(r, c)]
            workers.append(Worker(
                core_fn,
                [bufs[0], bufs[1], free, full, b_l2l1[r].cons(), c_l1l2[r][c].prod(),
                 mm, zero, store],
                tile=cores[r][c],
                stack_size=0x800,
            ))

    A_ty = np.ndarray[(a_tot,), np.dtype[np.uint32]]
    W_ty = np.ndarray[(w_tot * reps,), np.dtype[np.uint32]]
    C_ty = np.ndarray[(c_tot,), np.dtype[bfloat16]]

    def sequence(a, w, cc, B_prods, C_conses):
        credit = [[1, 1] for _ in range(COLS)]  # what each slot's next load must acquire
        next_slot = 0
        pending = []
        for rep in range(reps):
            for op in ops:
                tg = TaskGroup()
                cached, passes = a_passes(op, ra)
                nt = op["nt"]
                # A: one shim fill per column (re-read nt times when streaming)
                for c in range(COLS):
                    a_in[c].fill(
                        a, tap=tap1d(a_tot, op["a"] + c * op["la"], op["la"], 1 if cached else nt),
                        group=tg, wait=False,
                    )
                mem_tasks = []
                for chunks in passes:
                    loads, sends = [[] for _ in range(COLS)], [[] for _ in range(COLS)]
                    for i, (_, words) in enumerate(chunks):
                        s = next_slot
                        next_slot ^= 1
                        tap = TensorAccessPattern((ra,), 0, [1, 1, 1, words], [0, 0, 0, 1])
                        for c in range(COLS):
                            loads[c].append(Bd(
                                slots[c][s], tap=tap,
                                acquires=[Acquire(s_empty[c][s], value=credit[c][s])],
                                releases=[Release(s_full[c][s], value=nt if cached else 1)],
                            ))
                            sends[c].append(Bd(
                                slots[c][s], tap=tap,
                                acquires=[Acquire(s_full[c][s])],
                                releases=[Release(s_empty[c][s])],
                            ))
                            credit[c][s] = nt if cached else 1
                    for c in range(COLS):
                        tl = a_in[c].endpoint(mems[c]).task(*loads[c], wait=True)
                        ts = a_out[c].endpoint(mems[c]).task(
                            *sends[c], runs=nt if cached else 1, wait=True)
                        tl.start()
                        ts.start()
                        mem_tasks += [tl, ts]
                for r in range(ROWS):
                    lb = nt * op["lb"]
                    B_prods[r].fill(
                        w, tap=tap1d(w_tot * reps, rep * w_tot + op["w"] + r * lb, lb),
                        group=tg, wait=False,
                    )
                N = op["N"]
                for c in range(COLS):
                    tap = TensorAccessPattern(
                        (c_tot,), op["c"] + c * m * N, [nt, ROWS, m, n], [n, nt * n, N, 1]
                    )
                    C_conses[c].drain(cc, tap=tap, group=tg, wait=True)
                pending.append((tg, mem_tasks))
                if len(pending) >= lookahead:
                    tg0, mt0 = pending.pop(0)
                    tg0.finish()
                    for t in mt0:
                        t.await_()
                        t.free()
        for tg0, mt0 in pending:
            tg0.finish()
            for t in mt0:
                t.await_()
                t.free()

    rt = Runtime(
        sequence,
        [A_ty, W_ty, C_ty, [f.prod() for f in b_l3l2], [f.cons() for f in c_l2l3]],
    )
    for f in (*a_in, *a_out):
        rt.add_flow(f)
    for d in core_dmas:
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
    p.add_argument("--lookahead", type=int, default=2)
    p.add_argument("--kdefs", default="-DOPAQUE_A -DNO_PREFETCH -DPIPE=1")
    p.add_argument("--out", required=True)
    o = p.parse_args()
    iron.set_current_device(NPU2())
    kw = dict(
        sched=o.sched, m=o.m, k=o.k, n=o.n, rho=o.rho, reps=o.reps,
        b_depth=o.b_depth, l2_depth=o.l2_depth, kdefs=o.kdefs, lookahead=o.lookahead,
    )
    gemm_seq2.specialize(**kw).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(kw))


if __name__ == "__main__":
    main()
