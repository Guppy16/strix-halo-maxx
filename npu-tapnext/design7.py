"""design7: the fused TAPNext++ step program, built up in stages. Stage 1 = the GEMM frame.

Rows: 1152 = 8 columns x m=144 (1024 image tokens + up to 128 query slots). Core rows 0-2 of each column are
GEMM cores (design6's structure: A M-tile cached in the column's memtile and broadcast to its cores, B N-tiles
streamed per core row and broadcast across the 8 columns, C joined per column); core row 3 is the elementwise core
(stage 2). The A loads arrive on a packet-switched shim channel so the elementwise core's input can share it.

Ops: (K, nt, kind) with K in {768, 1536} and nt = N-tiles per GEMM core (N = 3 * nt * 64). Every op's A occupies
the whole slot (144 x 1536 bfp16 = 243 KB): a K=1536 op fills it once and each N-tile gets both halves; a K=768
op's A is 121 KB and the shim fills it TWICE (one half-slot send per N-tile). So the K=3072 GEMMs run as two K=1536
chunk ops whose bf16 partials the consumer sums, and the one send chain [h0, h1] serves every op (design6's rule
that cached ops share K).

Epilogue: bias. Each N-tile's B stream carries K/k weight tiles then one bias object (64 bf16 in a 4608 B object);
acc_store adds it and rounds to bf16; C tiles leave blocked ([m/8][n/8][8][8]).
"""

import argparse
import json
import os
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
    InOut,
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
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
COLS, ROWS_G = 8, 3  # columns, GEMM core rows per column
M, K_SLOT, KT, NT = 144, 1536, 64, 64  # rows per column, the slot's K, k-step, n-tile
RHO = M // 16  # 9 stripes of 16 rows
L1_WORDS = 288  # one A sub-tile (16 x 64 bfp16ebs8) in words
EW_ROW = 5  # the elementwise core's tile row (core row 3)
SLOT_W = 192  # elementwise ring slot: 768 B
EW_RING_SLOTS = 32
EW_PARAM_SLOTS = 2  # a 768-wide bf16 parameter vector
CHUNKS = M // 8  # 8-row chunks per column
CHUNK_SLOTS = 16  # 12,288 B


def words_a(m, k):  # an (m x k) bfp16ebs8 tile in 32-bit words
    return m * k * 9 // 8 // 4


def tap1d(total, off, length, rep=1):
    """Linear transfer of `length` elements at `off`, repeated `rep` times (shim wraps are 10-bit)."""
    a0 = next(d for d in range(min(length, 1020), 0, -1) if length % d == 0)
    rest = length // a0
    a1 = next(d for d in range(min(rest, 1023), 0, -1) if rest % d == 0)
    a2 = rest // a1
    assert a2 <= 1023, (length, a0, a1, a2)
    return TensorAccessPattern((total,), off, [rep, a2, a1, a0], [0, a1 * a0, a0, 1])


EW_OPS = {"rmsnorm": dict(param_slots=2, out_words=words_a(M, 768))}  # out: the K=768 A layout of the next GEMM


def plan(schedule):
    """Offsets (words / bf16 elements) of each op's buffers: GEMM ops [K, nt] read A from the host-packed A
    buffer, [K, nt, "E"] from the preceding elementwise op's output region in E; elementwise ops ["rmsnorm"] read
    their params + chunks from X and write E. Returns (ops, a_tot, w_tot, c_tot, x_tot, e_tot)."""
    wb = words_a(KT, NT)
    ops, a_off, w_off, c_off, x_off, e_off = [], 0, 0, 0, 0, 0
    last_e = None
    for ent in schedule:
        if isinstance(ent[0], str):
            kind = ent[0]
            spec = EW_OPS[kind]
            xcol = (spec["param_slots"] + CHUNKS * CHUNK_SLOTS) * SLOT_W
            ops.append(dict(ew=kind, x=x_off, xcol=xcol, e=e_off, ecol=spec["out_words"]))
            last_e = ops[-1]
            x_off += COLS * xcol
            e_off += COLS * spec["out_words"]
            continue
        K, nt = ent[0], ent[1]
        assert K in (768, K_SLOT), K
        la = words_a(M, K)  # one column's A M-tile
        lw = (K // KT + 1) * wb  # one N-tile of W: K/k weight tiles + 1 bias object
        N = ROWS_G * nt * NT
        sends = 1 if K == 768 else 2  # half-slot sends per N-tile
        assert sends * nt <= 63, ("lock values are 6-bit: split N", K, nt)
        op = dict(K=K, nt=nt, N=N, a=a_off, w=w_off, c=c_off, la=la, lw=lw, sends=sends, src=None)
        if len(ent) > 2 and ent[2] == "E":
            assert last_e is not None and last_e["ecol"] == la, (last_e, la)
            op["src"] = last_e
        else:
            a_off += COLS * la
        ops.append(op)
        w_off += ROWS_G * nt * lw
        c_off += COLS * M * N
    return ops, max(a_off, 64), w_off, c_off, max(x_off, 64), max(e_off, 64)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def step7(
    A: In,
    W: In,
    X: In,
    E: InOut,
    C: Out,
    *,
    sched: CompileTime[str] = "[[768, 4]]",
    reps: CompileTime[int] = 1,
    a_ring: CompileTime[int] = 3,
    b_depth: CompileTime[int] = 2,
    l2_depth: CompileTime[int] = 4,
    c_depth: CompileTime[int] = 2,
    lookahead: CompileTime[int] = 2,
    kdefs: CompileTime[str] = "",
    packet_a: CompileTime[int] = 1,
):
    m, k, n, rho = M, KT, NT, RHO
    assert rho % a_ring == 0, "the stripe -> L1 buffer map must repeat every k-step"
    schedule = [tuple(x) for x in json.loads(sched)]
    ops, a_tot, w_tot, c_tot, x_tot, e_tot = plan(schedule)
    gops = [op for op in ops if "ew" not in op]
    eops = [op for op in ops if "ew" in op]
    wa, wb = words_a(16, k), words_a(k, n)
    assert wa == L1_WORDS
    wb0 = next(d for d in range(min(wb, 1020), 0, -1) if wb % d == 0)
    ra = words_a(m, K_SLOT)  # the slot
    unit = ra // 2
    assert unit % L1_WORDS == 0 and all(op["la"] in (ra, unit) for op in gops)

    A_l1 = np.ndarray[(wa,), np.dtype[np.uint32]]
    B_l1 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    B_l2 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l2 = np.ndarray[(ROWS_G * m * n,), np.dtype[bfloat16]]
    slot_ty = np.ndarray[(ra,), np.dtype[np.uint32]]

    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}", "-DNO_PREFETCH", "-DBLK12", "-DBLKU=1"]
    mm = ExternalFunction("mm_acc", object_file_name="mm7.o", source_file=str(HERE / "mm7.cc"),
                          arg_types=[A_l1, B_l1, C_l1], compile_flags=flags + kdefs.split())
    zero = mm.object_file.bind("acc_zero", [C_l1])
    store = mm.object_file.bind("acc_store_bias", [C_l1, B_l1])

    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    cores = [[Tile(c, 2 + r) for c in range(COLS)] for r in range(ROWS_G)]

    # --- A: DDR -> memtile slot (packet-switched shim channel 0) -> broadcast to the column's GEMM cores -----
    if packet_a:
        a_in = [PacketFlow(1, shims[c], mems[c], src_channel=0, dst_channel=0, name=f"Ain{c}") for c in range(COLS)]
    else:
        a_in = [Flow(shims[c], mems[c]) for c in range(COLS)]
    a_out = [Flow(mems[c], [cores[r][c] for r in range(ROWS_G)]) for c in range(COLS)]

    # One load BD per op per schedule pass: it fills the whole slot and releases `full` by the sends its content
    # gets (sends per N-tile x nt); it first waits for all of the previous op's sends (`empty`).
    load_plan, pending = [], 0
    for op in gops:
        sends = op["sends"] * op["nt"]
        load_plan.append((pending, sends))
        pending = sends
    empty_init = pending
    load_plan = [(acq if acq else empty_init, rel) for (acq, rel) in load_plan]
    assert all(v <= 63 for (a, r) in load_plan for v in (a, r)), load_plan
    assert len(load_plan) + 2 <= 22, ("A chain does not fit the 24-BD half", len(load_plan))

    mem_dmas = []
    for c in range(COLS):
        slot = Buffer(tile=mems[c], type=slot_ty, name=f"As{c}")
        empty = Lock(tile=mems[c], init=empty_init, name=f"Ae{c}")
        full = Lock(tile=mems[c], init=0, name=f"Af{c}")

        def tap(off, words):
            return TensorAccessPattern((ra,), off, [1, 1, 1, words], [0, 0, 0, 1])

        loads = [Bd(slot, tap=tap(0, ra), acquires=[Acquire(empty, value=acq)], releases=[Release(full, value=rel)])
                 for (acq, rel) in load_plan]
        sends = [Bd(slot, tap=tap(s * unit, unit), acquires=[Acquire(full, value=1)], releases=[Release(empty, value=1)])
                 for s in range(2)]
        mem_dmas.append(TileDma(mems[c], [
            DmaChannel(DMAChannelDir.S2MM, a_in[c].endpoint(mems[c]), loads),
            DmaChannel(DMAChannelDir.MM2S, a_out[c].endpoint(mems[c]), sends),
        ]))

    a_l1, core_dmas = {}, []
    for r in range(ROWS_G):
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

    # --- B (one stream per GEMM row, forwarded by memtile r, broadcast across the columns) and C (join) ------
    b_prods, b_l1 = [], []
    for r in range(ROWS_G):
        f = ObjectFifo(B_l2, name=f"B{r}", depth=l2_depth)
        b_prods.append(f.prod(tile=shims[r]))
        b_l1.append(f.cons().forward(tile=mems[r], obj_type=B_l1, name=f"B{r}L1", depth=b_depth))
    c_conses, c_l1 = [], [[None] * COLS for _ in range(ROWS_G)]
    for c in range(COLS):
        f = ObjectFifo(C_l2, name=f"C{c}", depth=c_depth)
        c_conses.append(f.cons(tile=shims[c]))
        subs = f.prod().join([m * n * i for i in range(ROWS_G)], tile=mems[c], obj_types=[C_l1] * ROWS_G,
                             names=[f"C{c}_{i}" for i in range(ROWS_G)], depths=[1] * ROWS_G)
        for i in range(ROWS_G):
            c_l1[i][c] = subs[i]

    # --- the elementwise core (row 3): packet input ring on the shim's channel 0, circuit output to channel 1 --
    Ring = np.ndarray[(EW_RING_SLOTS * SLOT_W,), np.dtype[np.uint32]]
    ew_flags = [f"-I{INC}", f"-DEW_RING_SLOTS={EW_RING_SLOTS}", "-DL_IN_FREE=0", "-DL_IN_FILL=1", "-DL_OUT_FREE=2", "-DL_OUT_FILL=3"]
    ew_rms = ExternalFunction("ew_rmsnorm", object_file_name="ew7.o", source_file=str(HERE / "ew7.cc"),
                              arg_types=[Ring, Ring, np.int32], compile_flags=ew_flags)
    ew_tiles = [Tile(c, EW_ROW) for c in range(COLS)]
    e_in = [PacketFlow(2, shims[c], ew_tiles[c], src_channel=0, dst_channel=0, name=f"Ein{c}") for c in range(COLS)]
    e_out = [Flow(ew_tiles[c], shims[c]) for c in range(COLS)]
    ew_workers = []
    if eops:
        for c in range(COLS):
            t = ew_tiles[c]
            rin = Buffer(tile=t, type=Ring, name=f"ewi{c}")
            rout = Buffer(tile=t, type=Ring, name=f"ewo{c}")
            in_free = Lock(tile=t, init=EW_RING_SLOTS, name=f"eif{c}", lock_id=0)
            in_fill = Lock(tile=t, init=0, name=f"eiu{c}", lock_id=1)
            out_free = Lock(tile=t, init=EW_RING_SLOTS, name=f"eof{c}", lock_id=2)
            out_fill = Lock(tile=t, init=0, name=f"eou{c}", lock_id=3)
            slot_tap = TensorAccessPattern((EW_RING_SLOTS * SLOT_W,), 0, [1, 1, 1, SLOT_W], [0, 0, 0, 1])
            core_dmas.append(TileDma(t, [
                DmaChannel(DMAChannelDir.S2MM, e_in[c].endpoint(t), [Bd(
                    rin, tap=slot_tap, acquires=[Acquire(in_free)], releases=[Release(in_fill)],
                    iteration=BdIteration(size=EW_RING_SLOTS, stride=SLOT_W))]),
                DmaChannel(DMAChannelDir.MM2S, e_out[c].endpoint(t), [Bd(
                    rout, tap=slot_tap, acquires=[Acquire(out_fill)], releases=[Release(out_free)],
                    iteration=BdIteration(size=EW_RING_SLOTS, stride=SLOT_W))]),
            ]))

            def ew_fn(rin, rout, ew_rms):
                for _ in range_(reps) if reps > 1 else range(1):
                    for op in eops:
                        if op["ew"] == "rmsnorm":
                            ew_rms(rin, rout, CHUNKS)

            ew_workers.append(Worker(ew_fn, [rin, rout, ew_rms], tile=t, stack_size=0x800))

    def core_fn(a_free, a_full, *rest):
        abufs, (in_b, out_c, mm, zero, store) = rest[:a_ring], rest[a_ring:]

        def one_op(K, nt):
            for _ in range_(nt) if nt > 1 else range(1):
                cbuf = out_c.acquire(1)
                zero(cbuf)
                for _ in range_(K // k):
                    b = in_b.acquire(1)
                    for _ in range_(rho // a_ring) if rho > a_ring else range(1):
                        for j in range(a_ring):
                            a_full.acquire(1)
                            mm(abufs[j], b, cbuf)
                            a_free.release(1)
                    in_b.release(1)
                bias = in_b.acquire(1)
                store(cbuf, bias)
                in_b.release(1)
                out_c.release(1)

        for _ in range_(reps) if reps > 1 else range(1):
            for op in gops:
                one_op(op["K"], op["nt"])

    workers = []
    for r in range(ROWS_G):
        for c in range(COLS):
            bufs, free, fill = a_l1[(r, c)]
            workers.append(Worker(core_fn, [free, fill, *bufs, b_l1[r].cons(), c_l1[r][c].prod(), mm, zero, store],
                                  tile=cores[r][c], stack_size=0x1000))
    workers += ew_workers

    A_ty = np.ndarray[(a_tot,), np.dtype[np.uint32]]
    W_ty = np.ndarray[(w_tot * reps,), np.dtype[np.uint32]]
    X_ty = np.ndarray[(x_tot,), np.dtype[np.uint32]]
    E_ty = np.ndarray[(e_tot,), np.dtype[np.uint32]]
    C_ty = np.ndarray[(c_tot,), np.dtype[bfloat16]]

    def sequence(a, w, x, e, cc, B_prods, C_conses):
        pending = []
        for rep in range(reps):
            for op in ops:
                if "ew" in op:
                    # an elementwise op depends on everything before it and everything after depends on it
                    for tg in pending:
                        tg.finish()
                    pending = []
                    tg = TaskGroup()
                    for c in range(COLS):
                        e_in[c].fill(x, tap=tap1d(x_tot, op["x"] + c * op["xcol"], op["xcol"]), group=tg, wait=False)
                        # 18 chunks x 12 k-steps x 576 B of panels -> the A layout [k-step][stripe][half][576 B]
                        e_out[c].drain(e, tap=TensorAccessPattern((e_tot,), op["e"] + c * op["ecol"], [9, 2, 12, 144], [288, 144, 2592, 1]),
                                       group=tg, wait=True)
                    tg.finish()
                    continue
                tg = TaskGroup()
                nt, K, N = op["nt"], op["K"], op["N"]
                for c in range(COLS):
                    if op["src"] is None:
                        a_in[c].fill(a, tap=tap1d(a_tot, op["a"] + c * op["la"], op["la"], ra // op["la"]), group=tg, wait=False)
                    else:
                        src = op["src"]
                        a_in[c].fill(e, tap=tap1d(e_tot, src["e"] + c * src["ecol"], op["la"], ra // op["la"]), group=tg, wait=False)
                for r in range(ROWS_G):
                    tap = TensorAccessPattern((w_tot * reps,), rep * w_tot + op["w"] + r * nt * op["lw"],
                                              [nt, K // k + 1, wb // wb0, wb0], [op["lw"], wb, wb0, 1])
                    B_prods[r].fill(w, tap=tap, group=tg, wait=False)
                for c in range(COLS):
                    tap = TensorAccessPattern((c_tot,), op["c"] + c * m * N, [nt, ROWS_G, m, n], [n, nt * n, N, 1])
                    C_conses[c].drain(cc, tap=tap, group=tg, wait=True)
                pending.append(tg)
                if len(pending) >= lookahead:
                    pending.pop(0).finish()
        for tg in pending:
            tg.finish()

    rt = Runtime(sequence, [A_ty, W_ty, X_ty, E_ty, C_ty, b_prods, c_conses])
    for f in (*a_in, *a_out, *(e_in + e_out if eops else [])):
        rt.add_flow(f)
    for d in (*mem_dmas, *core_dmas):
        rt.add_tile_dma(d)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sched", default="[[768, 4]]")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--a-ring", type=int, default=3)
    p.add_argument("--b-depth", type=int, default=2)
    p.add_argument("--l2-depth", type=int, default=4)
    p.add_argument("--c-depth", type=int, default=2)
    p.add_argument("--lookahead", type=int, default=2)
    p.add_argument("--kdefs", default="")
    p.add_argument("--packet-a", type=int, default=1)
    p.add_argument("--out", required=True)
    o = p.parse_args()
    iron.set_current_device(NPU2())
    kw = dict(sched=o.sched, reps=o.reps, a_ring=o.a_ring, b_depth=o.b_depth, l2_depth=o.l2_depth, c_depth=o.c_depth,
              lookahead=o.lookahead, kdefs=o.kdefs, packet_a=o.packet_a)
    step7.specialize(**kw).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(kw, m=M, k=KT, n=NT, rho=RHO, rows_g=ROWS_G, k_slot=K_SLOT)))


if __name__ == "__main__":
    main()
