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
    Kernel,
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
from aie.iron.dataflow.flow import FlowEndpoint, PacketDest, PacketFlow
from aie.iron.dataflow.tile_dma import BdIteration
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
COLS, ROWS_G = 8, 3
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"


def asm_object(flags):
    """mm8.cc (as the design compiles it) + the generated mm_core.s, merged by ld.lld -r -> build/mm8asm.o"""
    import subprocess

    b, bd = f"{PEANO}/bin", HERE / "build"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include",
            "-fstack-size-section"]  # round 11: frame sizes for r11/stackaudit.py (a non-alloc section; text unchanged)
    ks = "-DMM_KS" in flags  # round 9: the k-step generator (r9/gen_ks.py) writes build/mm_ks.s and build/mm_ks.h
    gen = ["python3", str(HERE / "r9" / "gen_ks.py"), str(bd / "mm_ks.s")] if ks else \
        ["python3", str(HERE / "r7" / "gen_mm.py")] + (["--pieces"] if "-DMM_PIECES" in flags else []) + [str(bd / "mm_core.s")]
    so = bd / ("mm_ks.s" if ks else "mm_core.s")
    extra = []
    gm = "-DGELU_MS" in flags and "-DBGELU" not in flags  # round 11: the GELU loop as a hand modulo schedule (r11/gen_gelu.py)
    # (round 12: with -DBGELU every GELU epilogue runs bgelu and gelu_ms is not linked)
    if gm:  # round 12: with the flat-edge tables (-DGELU_FLAT) the flat loop (no xc ring; r12 apb bit-exact at II 12)
        flat = ["--flat", "--rel", "S:a,PQ:b", "--seeds", "8"] if "-DGELU_FLAT" in flags else []
        extra = [["python3", str(HERE / "r11" / "gen_gelu.py"), str(bd / "gelu_ms.s")] + flat,
                 [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "gelu_ms.s"), "-o", str(bd / "gelu_ms.o")]]
    fit = ["--fit"] if os.environ.get("ASM_FIT") == "1" else []  # round 13: compact loop bodies (r12/asmfit.py), -208 B of text
    gu = "-DGUP" in flags  # round 12: the gated up / product / push loop (r12/gen_gup.py, schedule from r12/gup.json)
    if gu:
        extra += [["python3", str(HERE / "r12" / "gen_gup.py"), str(bd / "gup.s"), "--times", str(HERE / "r12" / "gup.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "gup.s"), "-o", str(bd / "gup.o")]]
    cp = "-DCPY32" in flags  # round 12: the gated_b stash copy (r12/gen_copy.py)
    if cp:
        extra += [["python3", str(HERE / "r12" / "gen_copy.py"), str(bd / "cpy32.s")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "cpy32.s"), "-o", str(bd / "cpy32.o")]]
    pn = "-DPAN" in flags  # round 12: the panel pushes of flags 4 / 5 / 6 (r12/gen_pan.py, schedule r12/pan.json)
    if pn:
        extra += [["python3", str(HERE / "r12" / "gen_pan.py"), str(bd / "pan.s"), "--times", str(HERE / "r12" / "pan.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "pan.s"), "-o", str(bd / "pan.o")]]
    bg = "-DBGELU" in flags  # round 12: bias + GELU fused (r12/gen_bgelu.py, schedule r12/bgelu.json)
    if bg:
        extra += [["python3", str(HERE / "r12" / "gen_bgelu.py"), str(bd / "bgelu.s"), "--times", str(HERE / "r12" / "bgelu.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "bgelu.s"), "-o", str(bd / "bgelu.o")]]
    ak = "-DATTK" in flags  # round 13: the attention K phase's QK^T and exp + push loops (r13/gen_attqk.py, r13/gen_aexp.py)
    if ak:
        extra += [["python3", str(HERE / "r13" / "gen_attqk.py"), str(bd / "attqk.s"), "--times", str(HERE / "r13" / "attqk.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attqk.s"), "-o", str(bd / "attqk.o")],
                  ["python3", str(HERE / "r13" / "gen_aexp.py"), str(bd / "aexp.s"), "--times", str(HERE / "r13" / "aexp.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "aexp.s"), "-o", str(bd / "aexp.o")]]
    ab = "-DATTKB" in flags  # round 14: the batched fused K loop (r14/gen_attkb.py, schedule r14/attkb.json) replaces attk
    af = "-DATTKF" in flags and not ab  # round 13: the fused K loop (r13/gen_attk.py, schedule r13/attk.json)
    if ab:
        extra += [["python3", str(HERE / "r14" / "gen_attkb.py"), str(bd / "attkb.s"), "--times", str(HERE / "r14" / "attkb.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attkb.s"), "-o", str(bd / "attkb.o")]]
    av = "-DATTV" in flags  # round 14: the V phase in asm (r14/gen_attv.py, schedule r14/attv.json)
    if av:
        extra += [["python3", str(HERE / "r14" / "gen_attv.py"), str(bd / "attv.s"), "--times", str(HERE / "r14" / "attv.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attv.s"), "-o", str(bd / "attv.o")],
                  ["python3", str(HERE / "r14" / "gen_attv.py"), str(bd / "attvs.s"), "--sums", "--times", str(HERE / "r14" / "attvs.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attvs.s"), "-o", str(bd / "attvs.o")]]
    vk = "-DVTK" in flags  # round 14: the V^T epilogue in asm (r14/gen_vt.py, schedule r14/vt.json)
    if vk:
        extra += [["python3", str(HERE / "r14" / "gen_vt.py"), str(bd / "vtk.s"), "--times", str(HERE / "r14" / "vt.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "vtk.s"), "-o", str(bd / "vtk.o")]]
    za = "-DZASM" in flags  # round 14: acc_zero's fills as zero-overhead loops (r14/zero2.s)
    if za:
        extra += [[f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(HERE / "r14" / "zero2.s"), "-o", str(bd / "zero2.o")]]
    rk = "-DRESCK" in flags  # round 15 (research rev 10 S1): rescale_rb's O update in asm (r15/gen_resc.py, schedule r15/resc.json)
    if rk:
        extra += [["python3", str(HERE / "r15" / "gen_resc.py"), str(bd / "resc.s"), "--times", str(HERE / "r15" / "resc.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "resc.s"), "-o", str(bd / "resc.o")]]
    if af:
        extra += [["python3", str(HERE / "r13" / "gen_attk.py"), str(bd / "attk.s"), "--times", str(HERE / "r13" / "attk.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attk.s"), "-o", str(bd / "attk.o")]]
    steps = [gen,
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(so), "-o", str(so.with_suffix(".o"))]] + extra + [
             [f"{b}/clang++"] + base + flags + [f"-I{HERE}", "-c", str(HERE / "mm8.cc"), "-o", str(bd / "mm8cpp.o")],
             [f"{b}/ld.lld", "-r", str(bd / "mm8cpp.o"), str(so.with_suffix(".o"))] + ([str(bd / "gelu_ms.o")] if gm else []) + ([str(bd / "gup.o")] if gu else []) + ([str(bd / "cpy32.o")] if cp else []) + ([str(bd / "pan.o")] if pn else []) + ([str(bd / "bgelu.o")] if bg else [])
             + ([str(bd / "attqk.o"), str(bd / "aexp.o")] if ak else []) + ([str(bd / "attk.o")] if af else []) + ([str(bd / "attkb.o")] if ab else []) + ([str(bd / "attv.o"), str(bd / "attvs.o")] if av else []) + ([str(bd / "vtk.o")] if vk else []) + ([str(bd / "zero2.o")] if za else []) + ([str(bd / "resc.o")] if rk else [])
             + ["-o", str(bd / "mm8asm.o")]]
    for cmd in steps:
        subprocess.run(cmd, check=True)
    return str(bd / "mm8asm.o")
def ew_object(flags):
    """round 12: ew8.cc compiled here and merged with generated asm kernels (-DSIGK: r12/gen_sig.py, schedule r12/sig.json)
    -> build/ew8asm.o, for Kernel() (ExternalFunction cannot link asm)"""
    import subprocess

    b, bd = f"{PEANO}/bin", HERE / "build"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include",
            "-fstack-size-section", f"-I{HERE}"]
    objs, steps = [str(bd / "ew8cpp.o")], []
    if "-DSIGK" in flags:
        steps += [["python3", str(HERE / "r12" / "gen_sig.py"), str(bd / "sigk.s"), "--times", str(HERE / "r12" / "sig.json")],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "sigk.s"), "-o", str(bd / "sigk.o")]]
        objs.append(str(bd / "sigk.o"))
    if "-DSSX" in flags:  # round 15 (P29b): the SSM log_a + exp + 1 - a^2 chain as one asm loop (r15/gen_ssx.py, schedule r15/ssx.json)
        steps += [["python3", str(HERE / "r15" / "gen_ssx.py"), str(bd / "ssx.s"), "--times", str(HERE / "r15" / "ssx.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "ssx.s"), "-o", str(bd / "ssx.o")]]
        objs.append(str(bd / "ssx.o"))
    if "-DCONVK" in flags:  # round 15: the SSM conv as one asm loop (r15/gen_conv.py, schedule r15/convk.json)
        steps += [["python3", str(HERE / "r15" / "gen_conv.py"), str(bd / "convk.s"), "--times", str(HERE / "r15" / "convk.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "convk.s"), "-o", str(bd / "convk.o")]]
        objs.append(str(bd / "convk.o"))
    if "-DRECK" in flags:  # round 15: the SSM recurrence as one asm loop (r15/gen_rec.py, schedule r15/reck4.json: U 4, II 11)
        steps += [["python3", str(HERE / "r15" / "gen_rec.py"), str(bd / "reck.s"), "--times", str(HERE / "r15" / "reck4.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "reck.s"), "-o", str(bd / "reck.o")]]
        objs.append(str(bd / "reck.o"))
    if "-DNRMK" in flags:  # round 15: the norm pass-2 math (r15/gen_nrm.py, schedules r15/nrm_rms.json, nrm_ln.json)
        for md in ("rms", "ln"):
            steps += [["python3", str(HERE / "r15" / "gen_nrm.py"), str(bd / f"nrm_{md}.s"), "--mode", md, "--times", str(HERE / "r15" / f"nrm_{md}.json"), "--fit"],
                      [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / f"nrm_{md}.s"), "-o", str(bd / f"nrm_{md}.o")]]
            objs.append(str(bd / f"nrm_{md}.o"))
    if "-DCPY64" in flags:  # round 15: the SSM out copies as an II-1 asm copy (r15/gen_cpy64.py)
        steps += [["python3", str(HERE / "r15" / "gen_cpy64.py"), str(bd / "cpy64.s")],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "cpy64.s"), "-o", str(bd / "cpy64.o")]]
        objs.append(str(bd / "cpy64.o"))
    steps += [[f"{b}/clang++"] + base + flags + ["-c", str(HERE / "ew8.cc"), "-o", str(bd / "ew8cpp.o")],
              [f"{b}/ld.lld", "-r"] + objs + ["-o", str(bd / "ew8asm.o")]]
    for cmd in steps:
        subprocess.run(cmd, check=True)
    return str(bd / "ew8asm.o")


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
             "-DATTN_SCALE=0.125f", "-Os"] + kdefs.replace(",", " ").split()
    if "-DMM_ASM" in flags:  # round 7: the GEMM call hand-scheduled (r7/gen_mm.py), merged with the C++ object
        mm = Kernel("mm_acc", asm_object(flags), [A_l1, B_l1, C_l1], stack_size_override=0x900)
    else:
        mm = ExternalFunction("mm_acc", object_file_name="mm8.o", source_file=str(HERE / "mm8.cc"),
                              arg_types=[A_l1, B_l1, C_l1], compile_flags=flags)
    zero = mm.object_file.bind("acc_zero", [C_l1])
    kstep = "-DMM_KSTEP" in flags
    mmk = mm.object_file.bind("mm_kstep", [A_l1, A_l1, A_l1, B_l1, C_l1]) if kstep else None
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
    a_half = os.environ.get("A_HALF") == "1"
    # round 10 (research P5): A_RT=1 drives the memtile A slot from the runtime sequence (IRON DmaEndpoint.task) instead of
    # static BD chains: per op a chunked load (one k-step chunk per BD run: acquire a free chunk, release a loaded one), tile 0's
    # sends chunk by chunk behind the load, the middle tiles' whole sends without locks, the last tile's sends chunk by chunk
    # freeing each chunk for the next op's load (bench: r10/rtm.py). Needs the A_HALF plan (single fills, reuse_a).
    a_rt = os.environ.get("A_RT") == "1"
    assert not a_rt or a_half, "A_RT uses the A_HALF plan"
    if a_rt:
        a_in = [PacketFlow(1, shims[c], mems[c], src_channel=0, dst_channel=0, name=f"Ain{c}") for c in range(COLS)]
        a_inb = a_in
    elif a_half:
        # round 10 (research P5a): the slot's halves are written by two memtile S2MM channels (0 and 5), each with its own
        # lock pair, both fed by ONE packet flow multicast from shim channel 0 (id 4; 2 = the EW ring). A K=768 op's A
        # (and attention's Q objects) leaves DDR once: 124 KB, one copy per half. A K=1536 op streams its 248 KB and both
        # channels write it over the whole slot (identical values; the multicast keeps them in step, and each channel
        # starts only after its own half's empty lock).
        # (IRON 1.4.4's router drops a multicast destination when unicast flows share the two DMA ports: r10/pf1.mlir.)
        a_inb = [PacketFlow(4, shims[c], mems[c], src_channel=0, dst_channel=0, extra_dsts=[PacketDest(mems[c], channel=5)],
                            name=f"Ainb{c}") for c in range(COLS)]
        a_in = a_inb
    else:
        a_in = [PacketFlow(1, shims[c], mems[c], src_channel=0, dst_channel=0, name=f"Ain{c}") for c in range(COLS)]
    a_out = [Flow(mems[c], [cores[r][c] for r in range(ROWS_G)]) for c in range(COLS)]

    # one load BD per slot-op per chain pass: the chain = pre ops, body ops (looped by the hardware), post ops is not
    # expressible as one cyclic chain unless pre/post have no slot ops -> require that (patch / final norm are EW ops)
    assert not core_pre and not core_post, "slot ops must all be in the body"
    load_plan, pending = [], 0
    if a_half:  # per half: sends alternate halves and every op sends an even count, so each half gets nsend / 2
        assert all(op["nsend"] % 2 == 0 for op in core_body), [op["nsend"] for op in core_body]
        assert not core_body[0].get("reuse_a")
        for op in core_body:
            if op.get("reuse_a"):  # no load: the previous load releases this op's sends too
                acq, rel, kk = load_plan[-1]
                load_plan[-1] = (acq, rel + op["nsend"] // 2, kk)
            else:
                load_plan.append((0, op["nsend"] // 2, op.get("K", 768)))
        load_plan = [(load_plan[i - 1][1], rel, kk) for i, (_, rel, kk) in enumerate(load_plan)]
        empty_init = load_plan[-1][1]
    else:
        for op in core_body:
            load_plan.append((pending, op["nsend"]))
            pending = op["nsend"]
        empty_init = pending
        load_plan = [(acq if acq else empty_init, rel) for (acq, rel) in load_plan]
    assert all(v <= 63 for (a, r_, *_) in load_plan for v in (a, r_)), load_plan
    assert len(load_plan) + 2 <= 22, len(load_plan)
    print(f"[design8] A load plan (acquire empty, release full){' per half' if a_half else ''}: {load_plan}")

    mem_dmas = []
    rt_locks, rt_slots = [], []
    for c in range(COLS):
        slot = Buffer(tile=mems[c], type=slot_ty, name=f"As{c}")
        if a_rt:  # F: loaded chunks (load -> tile 0), F2: tile 0's chunks (-> last tile), E lo / hi: free chunks 0-11 / 12-23
            lk = dict(F=Lock(tile=mems[c], init=0, name=f"AF{c}"), F2=Lock(tile=mems[c], init=0, name=f"AF2_{c}"),
                      Elo=Lock(tile=mems[c], init=12, name=f"AElo{c}"), Ehi=Lock(tile=mems[c], init=12, name=f"AEhi{c}"))
            rt_locks.append(lk)
            rt_slots.append(slot)
            continue

        def tap(off, words):
            return TensorAccessPattern((ra,), off, [1, 1, 1, words], [0, 0, 0, 1])

        if a_half:
            empty = [Lock(tile=mems[c], init=empty_init, name=f"Ae{h}_{c}") for h in range(2)]
            full = [Lock(tile=mems[c], init=0, name=f"Af{h}_{c}") for h in range(2)]
            def load_tap(h, kk):  # K=1536: both channels write the whole slot (identical data; stride 0 is refused in tile BDs)
                return tap(0, ra) if kk == 1536 else tap(h * unit, unit)

            loads = [[Bd(slot, tap=load_tap(h, kk), acquires=[Acquire(empty[h], value=acq)], releases=[Release(full[h], value=rel)])
                      for (acq, rel, kk) in load_plan] for h in range(2)]
            sends = [Bd(slot, tap=tap(s_ * unit, unit), acquires=[Acquire(full[s_], value=1)], releases=[Release(empty[s_], value=1)])
                     for s_ in range(2)]
            mem_dmas.append(TileDma(mems[c], [
                DmaChannel(DMAChannelDir.S2MM, FlowEndpoint(a_inb[c], 1), loads[0]),
                DmaChannel(DMAChannelDir.S2MM, FlowEndpoint(a_inb[c], 2), loads[1]),
                DmaChannel(DMAChannelDir.MM2S, a_out[c].endpoint(mems[c]), sends),
            ]))
            continue
        empty = Lock(tile=mems[c], init=empty_init, name=f"Ae{c}")
        full = Lock(tile=mems[c], init=0, name=f"Af{c}")

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
            # round 8: A_BANKS pins the A ring slots to L1 banks (r8/mmb.py: an A pop and a B pop in one bank cost the
            # asm GEMM call ~14 cycles; the default allocator put slots 1 and 2 in the B fifo's bank 3)
            a_banks = [int(v) for v in os.environ.get("A_BANKS", "").split(",") if v]
            bufs = [Buffer(tile=t, type=A_l1, name=f"a{r}{c}_{i}", **(dict(mem_bank=a_banks[i]) if a_banks else {}))
                    for i in range(a_ring)]
            kstep = "-DMM_KSTEP" in kdefs.replace(",", " ").split()  # round 9: mm_kstep takes these locks by id
            free = Lock(tile=t, init=a_ring, name=f"aF{r}{c}", **(dict(lock_id=4) if kstep else {}))
            fill = Lock(tile=t, init=0, name=f"aU{r}{c}", **(dict(lock_id=5) if kstep else {}))
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
    if "-DSIGK" in ew_flags:  # round 12: with generated asm (the SSM sigmoid) the object is prebuilt
        ew_op = Kernel("ew_op", ew_object(ew_flags), [RingIn, RingOut])
    else:
        ew_op = ExternalFunction("ew_op", object_file_name="ew8.o", source_file=str(HERE / "ew8.cc"),
                                 arg_types=[RingIn, RingOut], compile_flags=ew_flags)
    assert not any(o["fn"] == "patch" for o in ew_pre + ew_body + ew_post), "ew_op has no patch kind (the patch embedding is on the host)"
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

            def ew_fn(rin, rout, ew_op):
                # one op per worker-loop iteration: ew_op reads the op kind from the op's params header, so this code
                # does not depend on the op list (unrolling the block loop here once cost 2 KB of program memory)
                ew_op(rin, rout)

            ew_workers.append(Worker(ew_fn, [rin, rout, ew_op], tile=t, stack_size=0x1800, data_size=27648))  # round 8: +1.5 KB (ew8 g_ptmp, g_panels' pad)

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
    # round 8 (npu-research P22): ONE loop over the block's ops; attention is an entry of the same table (bn = 0 skips
    # the GEMM nest, ba = 1 runs the attention body), so the GEMM loop nest exists once in program memory (two
    # segments around attention cost two copies, ~1 KB). The attention loops take their trip counts from tbl_c (an
    # L1 table: constants let LLVM unroll the 3 x 3 Q-copy loop, 9 calls)
    one_seg = os.environ.get("ONE_SEG", "1") == "1"
    attn_ops = [o for o in core_body if o["kind"] == "attn"]
    assert len({json.dumps(o["passes"]) for o in attn_ops}) <= 1
    if one_seg:
        tops = core_body
    else:
        tops = gemm_ops
    tbl_k = np.array([(o["K"] // k if o["kind"] == "gemm" else 0) for o in tops] or [0], np.int32)
    tbl_n = np.array([((o["nt"] // 2 if o["epi"] == "gated" else o["nt"]) if o["kind"] == "gemm" else 0) for o in tops] or [0], np.int32)
    tbl_t = np.array([((2 if o["epi"] == "gated" else 1) if o["kind"] == "gemm" else 0) for o in tops] or [0], np.int32)
    tbl_a = np.array([(1 if o["kind"] == "attn" else 0) for o in tops] or [0], np.int32)
    tbl_c = np.array([3, 4, 24, 2], np.int32)  # Q-copy loops (3 x 3), head groups 4, key chunks 24, passes 2

    def to_index(v):
        return arith.IndexCastOp(IndexType.get(), v).result

    def to_i32(v):
        return arith.IndexCastOp(IntegerType.get_signless(32), v).result

    def make_core_fn(myrow):
        def core_fn(a_free, a_full, *rest):
            abufs = rest[:a_ring]
            in_b, out_c, mm, zero, epi, at_begin, at_qcopy, at_chunk_k, at_chunk_v, at_end, bk, bn, bt, ba, bc = rest[a_ring:a_ring + 15]
            mmk = rest[a_ring + 15] if kstep else None

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
                three = to_index(bc[0])
                for pi in range_(to_index(bc[3]) if one_seg else len(op["passes"])):
                    ntiles = to_i32(arith.SubIOp(two, pi).result)
                    pi32 = to_i32(pi)
                    for _ in range_(to_index(bc[1]) if one_seg else 4):
                        for hh in range_(three if one_seg else 3):
                            for t in range_(three if one_seg else 3):
                                a_full.acquire(1)
                                a_full.acquire(1)
                                a_full.acquire(1)
                                at_qcopy(abufs[0], abufs[1], abufs[2], to_i32(hh), to_i32(t), pi32, myrow)
                                a_free.release(1)
                                a_free.release(1)
                                a_free.release(1)
                        cbuf = out_c.acquire(1)
                        at_begin(cbuf, ntiles)
                        for _ in range_(to_index(bc[2]) if one_seg else 24):
                            kobj = in_b.acquire(1)
                            at_chunk_k(kobj)
                            in_b.release(1)
                            vobj = in_b.acquire(1)
                            at_chunk_v(vobj)
                            in_b.release(1)
                        at_end(cbuf)
                        out_c.release(1)

            # one block per worker-loop iteration (IRON wraps the body in an endless loop): the core code does not
            # depend on the depth, so one xclbin serves every depth; only the runtime sequence (.bin) differs
            if one_seg:
                for i in range_(len(tops)):
                    ksteps, nobj, tpo = to_index(bk[i]), to_index(bn[i]), to_index(bt[i])
                    for _ in range_(nobj):
                        cbuf = out_c.acquire(1)
                        for _ in range_(tpo):
                            zero(cbuf)
                            for _ in range_(ksteps):
                                b = in_b.acquire(1)
                                if mmk is not None:  # round 9: the A ring's locks inside the kernel, one call per k-step
                                    mmk(abufs[0], abufs[1], abufs[2], b, cbuf)
                                else:
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
                    if attn_ops:
                        for _ in range_(to_index(ba[i])):
                            one_attn(attn_ops[0])
            else:
                for seg in segments:
                    if seg[0] == "gemm":
                        gemm_seg(seg[1], seg[2])
                    else:
                        one_attn(seg[1])

        return core_fn

    workers = []
    ks_core = "-DMM_KS" in flags  # round 9: mm_ks's record table (232 B): 256 B from the stack (deepest chain 2.24 KB)
    for r in range(ROWS_G):
        for c in range(COLS):
            bufs, free, fill = a_l1[(r, c)]
            t = cores[r][c]
            tabs = [Buffer(tile=t, initial_value=a, name=f"tb{n}{r}{c}") for n, a in (("k", tbl_k), ("n", tbl_n), ("t", tbl_t), ("a", tbl_a), ("c", tbl_c))]
            workers.append(Worker(make_core_fn(r), [free, fill, *bufs, b_l1[r].cons(), c_l1[r][c].prod(), mm, zero, epi,
                                                    at_begin, at_qcopy, at_chunk_k, at_chunk_v, at_end, *tabs]
                                  + ([mmk] if kstep else []),
                                  tile=t, stack_size=(0xA00 if "-DGELU_FLAT" in flags else 0xB00) if ks_core else 0xC00,
                                  data_size=28160 + (256 if ks_core else 0) + (256 if "-DGELU_FLAT" in flags else 0)))
            # round 12: the flat-edge tables' ab copies (+256 B) from the stack (stackaudit: 1,664 of 2,816 B used)  # stack: deepest chain 2.24 KB measured (r7); data: +2 KB the GELU tables' ab copies (cd: bank 3), +1.25 KB the exp table (r7)
    workers += ew_workers

    W_ty = np.ndarray[(w_words,), np.dtype[np.uint32]]
    ACT_ty = np.ndarray[(act_words,), np.dtype[np.uint32]]

    def sequence(w, act, B_prods, C_conses):
        bufs = dict(w=w, act=act)
        if "-DPROF" in kdefs.replace(",", " ").split():  # round 7 timing build: enable every GEMM core's processor bus
            from aie.dialects.aiex import npu_maskwrite32  # (Core_Processor_Bus 0x32038 bit 0) so read_tm does not hang

            for c in range(COLS):
                for r in range(ROWS_G):
                    npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=2 + r)
        if "-DPROF" in os.environ.get("EW_DEFS", "").split():  # round 8: the EW cores' phase stamps (ew8.cc PROF)
            from aie.dialects.aiex import npu_maskwrite32

            for c in range(COLS):
                npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=EW_ROW)

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

        def fill_a(op, c, **kw):  # the A slot load of column c (A_HALF: per half, or once into both; none for reuse_a)
            if not a_half:
                fill(a_in[c], op["a"][c], **kw)
                return
            if op.get("reuse_a"):
                return
            for t in op["a"][c]:
                fill(a_inb[c], t, **kw)

        CH = words_a(m, k)  # words per k-step chunk of the slot (144 rows x 64 bfp16 = 2592)
        rt_tasks = []
        if a_rt:
            for c in range(COLS):
                lk, slot = rt_locks[c], rt_slots[c]
                me_in, me_out = a_in[c].endpoint(mems[c]), a_out[c].endpoint(mems[c])

                def chunk(h, acq, rel):
                    return Bd(slot, tap=TensorAccessPattern((ra,), 12 * CH * h, [1, 1, 1, CH], [0, 0, 0, 1]), acquires=[Acquire(acq)],
                              releases=[Release(rel)], iteration=BdIteration(size=12, stride=CH))

                E = [lk["Elo"], lk["Ehi"]]
                t = dict(ld=[me_in.task(chunk(h, E[h], lk["F"]), runs=12) for h in range(2)],
                         t0=[me_out.task(chunk(h, lk["F"], lk["F2"]), runs=12) for h in range(2)],
                         last=[me_out.task(chunk(h, lk["F2"], E[h]), runs=12) for h in range(2)],
                         mid={K_: me_out.task(Bd(slot, tap=TensorAccessPattern((ra,), 0, [1, 1, 1, (K_ // k) * CH], [0, 0, 0, 1])))
                              for K_ in (768, 1536)})
                rt_tasks.append(t)
        slot_ops = [o for o in ops if o["kind"] in ("gemm", "attn")]

        def a_loads(op):  # A_RT: start column c's chunked loads (after the op's fills)
            if a_rt and not op.get("reuse_a"):
                for c in range(COLS):
                    for h in range(op.get("K", 768) // 768):
                        rt_tasks[c]["ld"][h].start()

        def a_sends(op):  # A_RT: the op's sends: tile 0 chunked, middle tiles whole, last tile chunked (none around reuse_a)
            if not a_rt:
                return
            i = next(j for j, o in enumerate(slot_ops) if o is op)
            reused_next = i + 1 < len(slot_ops) and slot_ops[i + 1].get("reuse_a")
            K = op.get("K", 768)
            tiles = op["nt"] if op["kind"] == "gemm" else len(op["passes"])
            nmid = tiles - (0 if op.get("reuse_a") else 1) - (0 if reused_next else 1)
            assert nmid >= 0, op["name"]
            for c in range(COLS):
                t = rt_tasks[c]
                if not op.get("reuse_a"):
                    for h in range(K // 768):
                        t["t0"][h].start()
                if nmid:
                    t["mid"][K].start(repeat_count=nmid - 1)
                if not reused_next:
                    for h in range(K // 768):
                        t["last"][h].start()

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

        p43 = os.environ.get("P43") == "1"  # round 11 (research P43): a GEMM's B (weights) leaves before its dependency wait

        for op in ops:
            cur_op[0] = op["name"]
            if op["kind"] == "gemm":
                qkv = isinstance(op["c"][0], list)
                tg = G()
                nt, K = op["nt"], op["K"]
                lw = (K // k + 1) * wb

                def fill_b():
                    for r in range(ROWS_G):
                        tp = TensorAccessPattern((w_words,), op["w"] // 4 + r * nt * lw, [nt, K // k + 1, wb // wb0, wb0], [lw, wb, wb0, 1])
                        account(f"shim{r}", tg)
                        B_prods[r].fill(w, tap=tp, group=tg, wait=qkv)

                if p43:  # B has no data dependency; its group is finished after its drains as before
                    fill_b()
                finish_deps(op)
                # finish() frees a group's unwaited fills WITHOUT awaiting them: a fill may only sit in a group that is
                # finished after the fill completed (its drains), else it must be waited (wait=True)
                for c in range(COLS):
                    fill_a(op, c, group=tg, wait=qkv)
                a_loads(op)
                if not p43:
                    fill_b()
                a_sends(op)  # before any wait of this op: a blocked push may only wait on work issued already
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
                    fill_a(op, c, group=tg, wait=False)
                a_loads(op)
                for pi in range(len(op["passes"])):
                    for r in range(ROWS_G):
                        for t in op["b"][r]:
                            fill(B_prods[r], t, group=tg, wait=False)
                    if pi == 0:
                        a_sends(op)
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
        for t in rt_tasks:
            for task in (*t["ld"], *t["t0"], *t["last"], *t["mid"].values()):
                task.free()
        print(f"[design8] shim BD peak {bd_peak[0]} at {bd_peak[1]} (limit 16)")

    rt = Runtime(sequence, [W_ty, ACT_ty, b_prods, c_conses])
    for f in (*a_in, *a_out, *(e_in + e_out if any_ew else [])):
        rt.add_flow(f)
    for d in (*mem_dmas, *core_dmas):
        rt.add_tile_dma(d)
    for lk in rt_locks:
        for lock in lk.values():
            rt.add_lock(lock)
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
    Path(o.out + ".json").write_text(json.dumps(dict(kw, m=M, k=KT, n=NT, rho=RHO, rows_g=ROWS_G, k_slot=K_SLOT,
                                                     a_half=os.environ.get("A_HALF") == "1", a_rt=os.environ.get("A_RT") == "1", p23=os.environ.get("P23") == "1", p60g=os.environ.get("P60G") == "1",
                                                     p43=os.environ.get("P43") == "1")))


if __name__ == "__main__":
    main()
