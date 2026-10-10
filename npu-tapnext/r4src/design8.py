"""design8: the fused TAPNext++ step program. design7's frame (3 GEMM core rows + one elementwise core per
column, packet-shared shim channel for the A loads and the EW inputs) driven by a plan8 op list.

Buffers: W (In: weights, bias/flag objects, EW parameter objects), ACT (InOut: every activation, the SSM state, the
host-written frame patches / pos-emb / query tokens, the final LN rows). The plan (plan8.Plan) gives every op its
DMA access patterns in bytes; this file only turns them into TAPs and generates the workers' control code.

Ops: gemm [K, nt, epilogue flags per tile], attn (Q objects through the A slot, K/V through the B fifos, O through
C), ew norm / patch / ssm (row-3 cores). Phases: `pre` ops run once, `body` ops `reps` times (identical op kinds
per rep: the GEMM cores' code is one block), `post` ops once.
"""

import argparse
import json
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import arith
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.ir import IndexType, IntegerType
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
COLS, ROWS_G = 8, 3
M, K_SLOT, KT, NT = 144, 1536, 64, 64
RHO = M // 16
L1_WORDS = 288
EW_ROW = 5
SLOT_W = 192
EW_IN_SLOTS, EW_OUT_SLOTS = 20, 12
PATCH_PARAM_B, SSM_PARAM_B = 19 * 768, 14 * 768
EW_MODE = dict(none=0, rms=1, ln=2, ln_out=3)


def words_a(m, k):
    return m * k * 9 // 8 // 4


def tap_of(t, totals):
    """plan8 byte TAP -> TensorAccessPattern on the uint32 buffer."""
    sizes = list(t["sizes"])
    strides = [s // 4 for s in t["strides"][:3]] + [1]
    sizes[3] //= 4
    # fold the inner dim under the shim's 10-bit limits when needed
    if sizes[3] > 1023:
        assert sizes[2] == 1 and strides[2] == 0, t
        a0 = next(d for d in range(1020, 0, -1) if sizes[3] % d == 0)
        sizes[2], strides[2], sizes[3] = sizes[3] // a0, a0, a0
    assert all(s <= 1023 for s in sizes[1:]) and sizes[0] <= 1023, (t, sizes)
    return TensorAccessPattern((totals[t["buf"]],), t["off"] // 4, sizes, strides)


@iron.jit(aiecc_flags=["--dynamic-objFifos"] + os.environ.get("AIECC_EXTRA", "").split())
def step8(
    W: In,
    ACT: InOut,
    *,
    sched_path: CompileTime[str] = "",
    w_words: CompileTime[int] = 64,
    act_words: CompileTime[int] = 64,
    pre_n: CompileTime[int] = 0,
    body_n: CompileTime[int] = 0,
    reps: CompileTime[int] = 1,
    a_ring: CompileTime[int] = 3,
    b_depth: CompileTime[int] = 2,
    l2_depth: CompileTime[int] = 4,
    c_depth: CompileTime[int] = 2,
    lookahead: CompileTime[int] = 2,
    kdefs: CompileTime[str] = "",
    attn_keys: CompileTime[int] = 1072,
):
    m, k, n, rho = M, KT, NT, RHO
    assert rho % a_ring == 0
    ops = json.loads(Path(sched_path).read_text())
    assert len(ops) == pre_n + body_n * reps + (len(ops) - pre_n - body_n * reps)
    post_n = len(ops) - pre_n - body_n * reps
    pre, body, post = ops[:pre_n], ops[pre_n:pre_n + body_n], ops[len(ops) - post_n:]
    for rep in range(reps):
        for i, op in enumerate(ops[pre_n + rep * body_n:pre_n + (rep + 1) * body_n]):
            b = body[i]
            assert op["kind"] == b["kind"] and op.get("K") == b.get("K") and op.get("nt") == b.get("nt") and op.get("fn") == b.get("fn") \
                and op.get("epi") == b.get("epi") and op.get("mode") == b.get("mode"), (i, op["name"])
    totals = dict(w=w_words, act=act_words)
    for op in ops:  # a stale schedule (plan changed, sched not regenerated) hangs the EW core: check the param sizes
        if op["kind"] == "ew" and op["fn"] == "norm":
            assert op["fills"][0][0]["sizes"][3] == (5 if op["mode"] in ("ln", "ln_out") else 3) * 768, ("stale sched: norm params", op["name"])
    # the GEMM cores run: pre gemm/attn ops, then `reps` x the body's gemm/attn ops, then post's
    core_pre = [o for o in pre if o["kind"] in ("gemm", "attn")]
    core_body = [o for o in body if o["kind"] in ("gemm", "attn")]
    core_post = [o for o in post if o["kind"] in ("gemm", "attn")]
    ew_pre = [o for o in pre if o["kind"] == "ew"]
    ew_body = [o for o in body if o["kind"] == "ew"]
    ew_post = [o for o in post if o["kind"] == "ew"]
    wa, wb = words_a(16, k), words_a(k, n)
    assert wa == L1_WORDS
    wb0 = next(d for d in range(min(wb, 1020), 0, -1) if wb % d == 0)
    ra = words_a(m, K_SLOT)
    unit = ra // 2

    A_l1 = np.ndarray[(wa,), np.dtype[np.uint32]]
    B_l1 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    B_l2 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l2 = np.ndarray[(ROWS_G * m * n,), np.dtype[bfloat16]]
    slot_ty = np.ndarray[(ra,), np.dtype[np.uint32]]

    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}", "-DBLKU=1", f"-DATTN_KEYS={attn_keys}",
             "-DATTN_SCALE=0.125f", "-Os"] + kdefs.split()
    mm = ExternalFunction("mm_acc", object_file_name="mm8.o", source_file=str(HERE / "mm8.cc"),
                          arg_types=[A_l1, B_l1, C_l1], compile_flags=flags)
    zero = mm.object_file.bind("acc_zero", [C_l1])
    epi = mm.object_file.bind("acc_epilogue", [C_l1, B_l1])
    at_begin = mm.object_file.bind("attn_begin", [C_l1, np.int32])
    at_qcopy = mm.object_file.bind("attn_qcopy", [A_l1, A_l1, A_l1, np.int32, np.int32, np.int32, np.int32])
    at_chunk_k = mm.object_file.bind("attn_chunk_k", [B_l1])
    at_chunk_v = mm.object_file.bind("attn_chunk_v", [B_l1])
    at_end = mm.object_file.bind("attn_end", [C_l1])

    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]
    cores = [[Tile(c, 2 + r) for c in range(COLS)] for r in range(ROWS_G)]

    # --- A: DDR -> memtile slot (packet-switched shim channel 0) -> broadcast to the column's GEMM cores -----
    a_in = [PacketFlow(1, shims[c], mems[c], src_channel=0, dst_channel=0, name=f"Ain{c}") for c in range(COLS)]
    a_out = [Flow(mems[c], [cores[r][c] for r in range(ROWS_G)]) for c in range(COLS)]

    # one load BD per slot-op per chain pass: the chain = pre ops, body ops (looped by the hardware), post ops is not
    # expressible as one cyclic chain unless pre/post have no slot ops -> require that (patch / final norm are EW ops)
    assert not core_pre and not core_post, "slot ops must all be in the body"
    load_plan, pending = [], 0
    for op in core_body:
        load_plan.append((pending, op["nsend"]))
        pending = op["nsend"]
    empty_init = pending
    load_plan = [(acq if acq else empty_init, rel) for (acq, rel) in load_plan]
    assert all(v <= 63 for (a, r_) in load_plan for v in (a, r_)), load_plan
    assert len(load_plan) + 2 <= 22, len(load_plan)

    mem_dmas = []
    for c in range(COLS):
        slot = Buffer(tile=mems[c], type=slot_ty, name=f"As{c}")
        empty = Lock(tile=mems[c], init=empty_init, name=f"Ae{c}")
        full = Lock(tile=mems[c], init=0, name=f"Af{c}")

        def tap(off, words):
            return TensorAccessPattern((ra,), off, [1, 1, 1, words], [0, 0, 0, 1])

        loads = [Bd(slot, tap=tap(0, ra), acquires=[Acquire(empty, value=acq)], releases=[Release(full, value=rel)])
                 for (acq, rel) in load_plan]
        sends = [Bd(slot, tap=tap(s_ * unit, unit), acquires=[Acquire(full, value=1)], releases=[Release(empty, value=1)])
                 for s_ in range(2)]
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
    RingIn = np.ndarray[(EW_IN_SLOTS * SLOT_W,), np.dtype[np.uint32]]
    RingOut = np.ndarray[(EW_OUT_SLOTS * SLOT_W,), np.dtype[np.uint32]]
    ew_flags = [f"-I{INC}", f"-DEW_IN_SLOTS={EW_IN_SLOTS}", f"-DEW_OUT_SLOTS={EW_OUT_SLOTS}", "-DL_IN_FREE=0", "-DL_IN_FILL=1",
                "-DL_OUT_FREE=2", "-DL_OUT_FILL=3", f"-DPATCH_PARAM_B={PATCH_PARAM_B}", f"-DSSM_PARAM_B={SSM_PARAM_B}", os.environ.get("EW_OPT", "-Oz")] + os.environ.get("EW_DEFS", "").split()
    ew_norm = ExternalFunction("ew_norm", object_file_name="ew8.o", source_file=str(HERE / "ew8.cc"),
                               arg_types=[RingIn, RingOut, np.int32], compile_flags=ew_flags)
    ew_patch = ew_norm.object_file.bind("ew_patch", [RingIn, RingOut])
    ew_ssm = ew_norm.object_file.bind("ew_ssm", [RingIn, RingOut])
    ew_tiles = [Tile(c, EW_ROW) for c in range(COLS)]
    e_in = [PacketFlow(2, shims[c], ew_tiles[c], src_channel=0, dst_channel=0, name=f"Ein{c}") for c in range(COLS)]
    e_out = [Flow(ew_tiles[c], shims[c]) for c in range(COLS)]
    ew_workers = []
    any_ew = bool(ew_pre or ew_body or ew_post)
    if any_ew:
        for c in range(COLS):
            t = ew_tiles[c]
            rin = Buffer(tile=t, type=RingIn, name=f"ewi{c}")
            rout = Buffer(tile=t, type=RingOut, name=f"ewo{c}")
            in_free = Lock(tile=t, init=EW_IN_SLOTS, name=f"eif{c}", lock_id=0)
            in_fill = Lock(tile=t, init=0, name=f"eiu{c}", lock_id=1)
            out_free = Lock(tile=t, init=EW_OUT_SLOTS, name=f"eof{c}", lock_id=2)
            out_fill = Lock(tile=t, init=0, name=f"eou{c}", lock_id=3)
            core_dmas.append(TileDma(t, [
                DmaChannel(DMAChannelDir.S2MM, e_in[c].endpoint(t), [Bd(
                    rin, tap=TensorAccessPattern((EW_IN_SLOTS * SLOT_W,), 0, [1, 1, 1, SLOT_W], [0, 0, 0, 1]),
                    acquires=[Acquire(in_free)], releases=[Release(in_fill)],
                    iteration=BdIteration(size=EW_IN_SLOTS, stride=SLOT_W))]),
                DmaChannel(DMAChannelDir.MM2S, e_out[c].endpoint(t), [Bd(
                    rout, tap=TensorAccessPattern((EW_OUT_SLOTS * SLOT_W,), 0, [1, 1, 1, SLOT_W], [0, 0, 0, 1]),
                    acquires=[Acquire(out_fill)], releases=[Release(out_free)],
                    iteration=BdIteration(size=EW_OUT_SLOTS, stride=SLOT_W))]),
            ]))

            def ew_fn(rin, rout, ew_norm, ew_patch, ew_ssm):
                def run(oplist):
                    for op in oplist:
                        if op["fn"] == "norm":
                            ew_norm(rin, rout, EW_MODE[op["mode"]])
                        elif op["fn"] == "patch":
                            ew_patch(rin, rout)
                        elif op["fn"] == "ssm":
                            ew_ssm(rin, rout)
                run(ew_pre)
                if ew_body:
                    for _ in range_(reps) if reps > 1 else range(1):
                        run(ew_body)
                run(ew_post)

            ew_workers.append(Worker(ew_fn, [rin, rout, ew_norm, ew_patch, ew_ssm], tile=t, stack_size=0x1800, data_size=26112))

    # the GEMM cores' control code must stay small (16 KB of program memory incl. the kernels): the ops are a
    # table [k-steps, C objects, tiles per object] walked by one generic loop; the attention op has its own loop
    segments = []  # ("gemm", first op index, count) | ("attn", op)
    gemm_ops = [o for o in core_body if o["kind"] == "gemm"]
    gi = 0
    for o in core_body:
        if o["kind"] == "gemm":
            if segments and segments[-1][0] == "gemm":
                segments[-1] = ("gemm", segments[-1][1], segments[-1][2] + 1)
            else:
                segments.append(("gemm", gi, 1))
            gi += 1
        else:
            segments.append(("attn", o))
    tbl_k = np.array([o["K"] // k for o in gemm_ops] or [0], np.int32)
    tbl_n = np.array([(o["nt"] // 2 if o["epi"] == "gated" else o["nt"]) for o in gemm_ops] or [0], np.int32)
    tbl_t = np.array([(2 if o["epi"] == "gated" else 1) for o in gemm_ops] or [0], np.int32)

    def to_index(v):
        return arith.IndexCastOp(IndexType.get(), v).result

    def to_i32(v):
        return arith.IndexCastOp(IntegerType.get_signless(32), v).result

    def make_core_fn(myrow):
        def core_fn(a_free, a_full, *rest):
            abufs = rest[:a_ring]
            in_b, out_c, mm, zero, epi, at_begin, at_qcopy, at_chunk_k, at_chunk_v, at_end, bk, bn, bt = rest[a_ring:]

            def gemm_seg(i0, n):
                for i in range_(i0, i0 + n):
                    ksteps, nobj, tpo = to_index(bk[i]), to_index(bn[i]), to_index(bt[i])
                    for _ in range_(nobj):
                        cbuf = out_c.acquire(1)
                        for _ in range_(tpo):
                            zero(cbuf)
                            for _ in range_(ksteps):
                                b = in_b.acquire(1)
                                for _ in range_(rho // a_ring) if rho > a_ring else range(1):
                                    for j in range(a_ring):
                                        a_full.acquire(1)
                                        mm(abufs[j], b, cbuf)
                                        a_free.release(1)
                                in_b.release(1)
                            bias = in_b.acquire(1)
                            epi(cbuf, bias)
                            in_b.release(1)
                        out_c.release(1)

            def one_attn(op):
                # per pass the 36 Q objects stream through the A ring in [head][tile] order (3 sub-tiles each);
                # this core keeps those of heads myrow + 3 i and processes each head after its triple arrived.
                # pass 0 = query tiles {0, 1}, pass 1 = tile {2}: ntiles = 2 - pass
                two = arith.ConstantOp(IndexType.get(), 2).result
                for pi in range_(len(op["passes"])):
                    ntiles = to_i32(arith.SubIOp(two, pi).result)
                    pi32 = to_i32(pi)
                    for _ in range_(4):
                        for hh in range_(3):
                            for t in range_(3):
                                a_full.acquire(1)
                                a_full.acquire(1)
                                a_full.acquire(1)
                                at_qcopy(abufs[0], abufs[1], abufs[2], to_i32(hh), to_i32(t), pi32, myrow)
                                a_free.release(1)
                                a_free.release(1)
                                a_free.release(1)
                        cbuf = out_c.acquire(1)
                        at_begin(cbuf, ntiles)
                        for _ in range_(24):
                            kobj = in_b.acquire(1)
                            at_chunk_k(kobj)
                            in_b.release(1)
                            vobj = in_b.acquire(1)
                            at_chunk_v(vobj)
                            in_b.release(1)
                        at_end(cbuf)
                        out_c.release(1)

            for _ in range_(reps) if reps > 1 else range(1):
                for seg in segments:
                    if seg[0] == "gemm":
                        gemm_seg(seg[1], seg[2])
                    else:
                        one_attn(seg[1])

        return core_fn

    workers = []
    for r in range(ROWS_G):
        for c in range(COLS):
            bufs, free, fill = a_l1[(r, c)]
            t = cores[r][c]
            tabs = [Buffer(tile=t, initial_value=a, name=f"tb{n}{r}{c}") for n, a in (("k", tbl_k), ("n", tbl_n), ("t", tbl_t))]
            workers.append(Worker(make_core_fn(r), [free, fill, *bufs, b_l1[r].cons(), c_l1[r][c].prod(), mm, zero, epi,
                                                    at_begin, at_qcopy, at_chunk_k, at_chunk_v, at_end, *tabs],
                                  tile=t, stack_size=0x1800, data_size=25088))
    workers += ew_workers

    W_ty = np.ndarray[(w_words,), np.dtype[np.uint32]]
    ACT_ty = np.ndarray[(act_words,), np.dtype[np.uint32]]

    def sequence(w, act, B_prods, C_conses):
        bufs = dict(w=w, act=act)

        # static accounting of the shim BD pool (16 per shim tile) as this sequence uses it: every task occupies
        # a BD until its group is finished; the worst case and the op where it happens go to the build log
        bd_live = {}
        bd_of_group = {}
        bd_peak = [0, ""]
        cur_op = [""]

        def account(key, group):
            bd_live[key] = bd_live.get(key, 0) + 1
            bd_of_group.setdefault(id(group), []).append(key)
            if bd_live[key] > bd_peak[0]:
                bd_peak[0], bd_peak[1] = bd_live[key], f"{cur_op[0]} ({key})"

        def finished(group):
            for key in bd_of_group.pop(id(group), []):
                bd_live[key] -= 1

        class G(TaskGroup):  # a TaskGroup that reports its finish to the accounting
            def finish(self):
                finished(self)
                super().finish()

        def fill(ep, t, **kw):
            account(ep_shim(ep), kw["group"])
            ep.fill(bufs[t["buf"]], tap=tap_of(t, totals), **kw)

        def drain(ep, t, **kw):
            account(ep_shim(ep), kw["group"])
            ep.drain(bufs[t["buf"]], tap=tap_of(t, totals), **kw)

        def ep_shim(ep):
            for c in range(COLS):
                if ep is a_in[c] or ep is e_in[c] or ep is e_out[c] or ep is C_conses[c]:
                    return f"shim{c}"
            for r in range(ROWS_G):
                if ep is B_prods[r]:
                    return f"shim{r}"
            return "?"

        pending = []  # [(name, TaskGroup)]

        def finish_all():
            for _, tg in pending:
                tg.finish()
            pending.clear()

        def finish_deps(op):  # a consumer waits for exactly its producers' groups
            for name, tg in list(pending):
                if name in op.get("deps", []):
                    tg.finish()
                    pending.remove((name, tg))

        def push(name, tg):
            pending.append((name, tg))
            while len(pending) > lookahead - 1:
                pending.pop(0)[1].finish()

        for op in ops:
            cur_op[0] = op["name"]
            if op["kind"] == "gemm":
                finish_deps(op)
                qkv = isinstance(op["c"][0], list)
                tg = G()
                # finish() frees a group's unwaited fills WITHOUT awaiting them: a fill may only sit in a group that is
                # finished after the fill completed (its drains), else it must be waited (wait=True)
                for c in range(COLS):
                    fill(a_in[c], op["a"][c], group=tg, wait=qkv)
                nt, K = op["nt"], op["K"]
                lw = (K // k + 1) * wb
                for r in range(ROWS_G):
                    tp = TensorAccessPattern((w_words,), op["w"] // 4 + r * nt * lw, [nt, K // k + 1, wb // wb0, wb0], [lw, wb, wb0, 1])
                    account(f"shim{r}", tg)
                    B_prods[r].fill(w, tap=tp, group=tg, wait=qkv)
                if qkv:  # 12 drains per column, issued 4 tiles at a time, 2 groups in flight; the fills ride in the last group
                    finish_all()
                    groups = []
                    for g in range(3):
                        tgd = tg if g == 2 else G()
                        for c in range(COLS):
                            for j in range(4 * g, 4 * g + 4):
                                drain(C_conses[c], op["c"][c][j], group=tgd, wait=True)
                        groups.append(tgd)
                        if len(groups) > 2:
                            groups.pop(0).finish()
                    for tgd in groups:
                        pending.append((op["name"], tgd))
                else:
                    for c in range(COLS):
                        drain(C_conses[c], op["c"][c], group=tg, wait=True)
                    push(op["name"], tg)
            elif op["kind"] == "attn":
                finish_deps(op)
                tg = G()
                for c in range(COLS):
                    fill(a_in[c], op["a"][c], group=tg, wait=False)
                for pi in range(len(op["passes"])):
                    for r in range(ROWS_G):
                        for t in op["b"][r]:
                            fill(B_prods[r], t, group=tg, wait=False)
                    for c in range(COLS):
                        drain(C_conses[c], op["c"][pi][c], group=tg, wait=True)
                push(op["name"], tg)
            elif op["kind"] == "ew":
                finish_all()
                if op["fn"] == "norm":
                    tg = G()
                    for c in range(COLS):
                        for f in op["fills"][c]:
                            fill(e_in[c], f, group=tg, wait=False)
                        drain(e_out[c], op["c"][c], group=tg, wait=True)
                    tg.finish()
                else:  # patch / ssm: one drain per column, per-head fill pairs with a lookahead of 2
                    tgd = G()
                    for c in range(COLS):
                        drain(e_out[c], op["c"][c], group=tgd, wait=True)
                    groups = []
                    for q in range(len(op["fills"][0])):
                        tg = G()
                        for c in range(COLS):
                            for f in op["fills"][c][q]:
                                fill(e_in[c], f, group=tg, wait=True)  # finished two heads later: must be awaited, not just freed
                        groups.append(tg)
                        if len(groups) > 2:
                            groups.pop(0).finish()
                    for tg in groups:
                        tg.finish()
                    tgd.finish()
            else:
                raise NotImplementedError(op["kind"])
        finish_all()
        print(f"[design8] shim BD peak {bd_peak[0]} at {bd_peak[1]} (limit 16)")

    rt = Runtime(sequence, [W_ty, ACT_ty, b_prods, c_conses])
    for f in (*a_in, *a_out, *(e_in + e_out if any_ew else [])):
        rt.add_flow(f)
    for d in (*mem_dmas, *core_dmas):
        rt.add_tile_dma(d)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sched", required=True, help="plan json (ops) + meta json next to it")
    p.add_argument("--a-ring", type=int, default=3)
    p.add_argument("--b-depth", type=int, default=2)
    p.add_argument("--l2-depth", type=int, default=4)
    p.add_argument("--c-depth", type=int, default=2)
    p.add_argument("--lookahead", type=int, default=2)
    p.add_argument("--kdefs", default="")
    p.add_argument("--attn-keys", type=int, default=1072)
    p.add_argument("--out", required=True)
    o = p.parse_args()
    meta = json.loads(Path(o.sched + ".meta").read_text())
    iron.set_current_device(NPU2())
    kw = dict(sched_path=o.sched, w_words=meta["w_words"], act_words=meta["act_words"], pre_n=meta["pre_n"], body_n=meta["body_n"],
              reps=meta["reps"], a_ring=o.a_ring, b_depth=o.b_depth, l2_depth=o.l2_depth, c_depth=o.c_depth, lookahead=o.lookahead,
              kdefs=o.kdefs, attn_keys=o.attn_keys)
    import shutil
    try:
        step8.specialize(**kw).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    except Exception:
        for f in ("aie.mlir",):  # keep the lowered design of a failed build for inspection
            src = Path(o.out + ".prj") / f
            if src.exists():
                shutil.copy(src, o.out + ".failed." + f)
        raise
    Path(o.out + ".json").write_text(json.dumps(dict(kw, m=M, k=KT, n=NT, rho=RHO, rows_g=ROWS_G, k_slot=K_SLOT)))


if __name__ == "__main__":
    main()
