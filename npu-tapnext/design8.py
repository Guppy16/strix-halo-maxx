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
import fcntl
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


E7_YW = 1056  # round 21 E7'': the Y buffer (4,224 B: the DMA writes 4,032; the attention's -m rows use the K object's 3,456 .. 4,224)
E7_XB = 1152  # the two X buffers (576 B each) the stack gives up after r21/e7patch.py

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
        s2 = ["--s2", "--times", str(HERE / os.environ.get("ATTKS_JSON", "r17/attks.json"))] if "-DATTKS2MAX" in flags else ["--times", str(HERE / "r14" / "attkb.json")]
        extra += [["python3", str(HERE / "r14" / "gen_attkb.py"), str(bd / "attkb.s")] + s2 + fit,  # round 17: --s2 = the running max
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attkb.s"), "-o", str(bd / "attkb.o")]]
    k2 = "-DK2" in flags  # round 22 (research K2): chunk 0's running maxima by a QK-only loop (r22/gen_attkq.py, schedule r22/attkq.json)
    if k2:
        extra += [["python3", str(HERE / "r22" / "gen_attkq.py"), str(bd / "attkq.s"), "--times", str(HERE / "r22" / "attkq.json")] + fit,
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "attkq.s"), "-o", str(bd / "attkq.o")]]
    av = "-DATTV" in flags  # round 14: the V phase in asm (r14/gen_attv.py, schedule r14/attv.json)
    if av:
        extra += [["python3", str(HERE / "r14" / "gen_attv.py"), str(bd / "attv.s"), "--times", str(HERE / os.environ.get("ATTV_JSON", "r14/attv.json"))] + fit
                  + (["--two"] if "-DV2T" in flags else []),  # round 18: both tiles per call
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
    if "-DMM_ROT" in flags:  # round 18 (research pass 14 ROT): the rotating-accumulator k-step (r18/mm_rot_g2f.*, gen_rot.py output)
        import shutil

        v = os.environ.get("ROT_V", "g2f")  # round 18: g2a = re-solved with 6 bundles from acq.cond to the A fill (r18/gen_rot.py)
        rs = f"mm_rot_{v}_rel.s" if "-DMM_ROT_REL" in flags else f"mm_rot_{v}.s"
        rd = next(HERE / d for d in ("r23", "r22", "r21", "r19", "r18") if (HERE / d / rs).exists())  # round 19: g1a / g0a (r19/build_rot.sh); round 21: g0z (r21/mkzload.py)
        for src, dst in ((rd / rs, "mm_rot.s"), (rd / f"mm_rot_{v}.h", "mm_rot.h"), (HERE / "r18" / "mm8_rot.inc", "mm_rot.inc")):
            shutil.copy(src, bd / dst)
        extra += [[f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "mm_rot.s"), "-o", str(bd / "mm_rot.o")]]
    steps = [gen,
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(so), "-o", str(so.with_suffix(".o"))]] + extra + [
             [f"{b}/clang++"] + base + flags + [f"-I{HERE}", "-c", str(HERE / "mm8.cc"), "-o", str(bd / "mm8cpp.o")],
             [f"{b}/ld.lld", "-r", str(bd / "mm8cpp.o")] + ([] if "-DMM_ROT" in flags else [str(so.with_suffix(".o"))]) + ([str(bd / "gelu_ms.o")] if gm else []) + ([str(bd / "gup.o")] if gu else []) + ([str(bd / "cpy32.o")] if cp else []) + ([str(bd / "pan.o")] if pn else []) + ([str(bd / "bgelu.o")] if bg else [])
             + ([str(bd / "attqk.o"), str(bd / "aexp.o")] if ak else []) + ([str(bd / "attk.o")] if af else []) + ([str(bd / "attkb.o")] if ab else []) + ([str(bd / "attkq.o")] if k2 else []) + ([str(bd / "attv.o"), str(bd / "attvs.o")] if av else []) + ([str(bd / "vtk.o")] if vk else []) + ([str(bd / "zero2.o")] if za else []) + ([str(bd / "resc.o")] if rk else []) + ([str(bd / "mm_rot.o")] if "-DMM_ROT" in flags else [])
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
        sig_json = HERE / "r16" / "sig10.json" if "-DSIG10" in flags else HERE / "r12" / "sig.json"  # round 16: II 10, U 4 (was II 12, U 2)
        steps += [["python3", str(HERE / "r12" / "gen_sig.py"), str(bd / "sigk.s"), "--times", str(sig_json)],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "sigk.s"), "-o", str(bd / "sigk.o")]]
        objs.append(str(bd / "sigk.o"))
    if "-DSSX" in flags:  # round 15 (P29b): the SSM log_a + exp + 1 - a^2 chain as one asm loop (r15/gen_ssx.py, schedule r15/ssx.json)
        steps += [["python3", str(HERE / "r15" / "gen_ssx.py"), str(bd / "ssx.s"), "--times", str(HERE / "r15" / "ssx.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "ssx.s"), "-o", str(bd / "ssx.o")]]
        objs.append(str(bd / "ssx.o"))
    if "-DN1K" in flags:  # round 16 (research N1): norm pass 1 + statistics per unit (r16/gen_n1k.py, r16/n1k_ln.json / n1k_rms.json)
        for md in ("ln", "rms"):
            steps += [["python3", str(HERE / "r16" / "gen_n1k.py"), str(bd / f"n1k_{md}.s"), "--mode", md, "--times", str(HERE / "r16" / f"n1k_{md}.json"), "--fit"],
                      [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / f"n1k_{md}.s"), "-o", str(bd / f"n1k_{md}.o")]]
            objs.append(str(bd / f"n1k_{md}.o"))
    if "-DGGK" in flags:  # round 16: both SSM gate GEMMs as one asm loop (r16/gen_ggk.py, schedule r16/ggk2.json)
        steps += [["python3", str(HERE / "r16" / "gen_ggk.py"), str(bd / "ggk.s"), "--times", str(HERE / "r16" / "ggk2.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "ggk.s"), "-o", str(bd / "ggk.o")]]
        objs.append(str(bd / "ggk.o"))
    if "-DSSQ" in flags:  # round 16: ssx + sqk fused (r16/gen_ssq.py, schedule r16/ssq.json: II 30, U 2)
        steps += [["python3", str(HERE / "r16" / "gen_ssq.py"), str(bd / "ssq.s"), "--times", str(HERE / "r16" / "ssq.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "ssq.s"), "-o", str(bd / "ssq.o")]]
        objs.append(str(bd / "ssq.o"))
    if "-DSQK" in flags:  # round 16: the SSM sqrt as one asm loop (r16/gen_sqk.py, schedule r16/sqk.json)
        steps += [["python3", str(HERE / "r16" / "gen_sqk.py"), str(bd / "sqk.s"), "--times", str(HERE / "r16" / "sqk.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "sqk.s"), "-o", str(bd / "sqk.o")]]
        objs.append(str(bd / "sqk.o"))
    if "-DCONVK" in flags:  # round 15: the SSM conv as one asm loop (r15/gen_conv.py, schedule r15/convk.json)
        steps += [["python3", str(HERE / "r15" / "gen_conv.py"), str(bd / "convk.s"), "--fit"]  # round 18 CGF: + the bfp push (r18/convkp.json)
                  + (["--push", "--times", str(HERE / "r18" / "convkp.json")] if "-DCGF" in flags else ["--times", str(HERE / "r15" / "convk.json")]),
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "convk.s"), "-o", str(bd / "convk.o")]]
        objs.append(str(bd / "convk.o"))
    if "-DRECK" in flags:  # round 15: the SSM recurrence as one asm loop (r15/gen_rec.py, schedule r15/reck4.json: U 4, II 11)
        steps += [["python3", str(HERE / "r15" / "gen_rec.py"), str(bd / "reck.s"), "--times", str(HERE / "r15" / "reck4.json"), "--fit"]
                  + (["--ring"] if "-DHRING" in flags else []),  # round 18: h straight into the output ring
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "reck.s"), "-o", str(bd / "reck.o")]]
        objs.append(str(bd / "reck.o"))
    if "-DZK" in flags:  # round 18: the SSM z panels as one asm loop (r15/gen_zk.py, schedule r18/zk.json: II 5, U 2)
        steps += [["python3", str(HERE / "r15" / "gen_zk.py"), str(bd / "zk.s"), "--times", str(HERE / "r18" / "zk.json"), "--fit"],
                  [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(bd / "zk.s"), "-o", str(bd / "zk.o")]]
        objs.append(str(bd / "zk.o"))
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
PATCH_PARAM_B, SSM_PARAM_B = 19 * 768, (12288 if os.environ.get("AL1") == "1" else 14 * 768)  # round 17 AL1: 3 KB multiple
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
            want = (5 if op["mode"] in ("ln", "ln_out") else 3) * 768
            if os.environ.get("AL1") == "1":  # round 17: 1 KB multiples
                want = (want + 3071) // 3072 * 3072
            assert op["fills"][0][0]["sizes"][3] == want, ("stale sched: norm params", op["name"])
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
    e7 = os.environ.get("E7", "0") == "1"  # round 21 E7'' (B on both FIFOs): L1 B objects keep only the Y part (r21/e7patch.py splits the BDs)
    B_l1 = np.ndarray[(E7_YW if e7 else wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    B_l2 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l2 = np.ndarray[(ROWS_G * m * n,), np.dtype[bfloat16]]
    yxssm = os.environ.get("YXSSM") == "1"  # round 23 (main): the SSM starts during yx; z panels into half 1
    oph1 = os.environ.get("OPH1") == "1"  # round 23: out_proj's A in half 1 (its last tile can then be split: norm3's NRES uses half 0)
    a3 = os.environ.get("A3") == "1"  # round 22 (research 3A): a third half-slot region after the slot (chunks 24-35)
    ra_buf = ra * 3 // 2 if a3 else ra  # slot buffer words (TAP shapes over it)
    slot_ty = np.ndarray[(ra_buf,), np.dtype[np.uint32]]

    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}", "-DBLKU=1", f"-DATTN_KEYS={attn_keys}",
             "-DATTN_SCALE=0.125f", "-Os"] + kdefs.replace(",", " ").split()
    if "-DMM_ASM" in flags:  # round 7: the GEMM call hand-scheduled (r7/gen_mm.py), merged with the C++ object
        mm = Kernel("mm_acc", asm_object(flags), [A_l1, B_l1, C_l1], stack_size_override=0x900)
    else:
        mm = ExternalFunction("mm_acc", object_file_name="mm8.o", source_file=str(HERE / "mm8.cc"),
                              arg_types=[A_l1, B_l1, C_l1], compile_flags=flags)
    zero = mm.object_file.bind("acc_zero", [C_l1])
    kstep = "-DMM_KSTEP" in flags
    # round 18 (research P35): -DP35 runs a tile's k-steps in one call (mm_ktile: the B locks by id, the buffers by address; the
    # count per tile is even, so IRON's own B index - bias objects, attention - stays right)
    p35 = "-DP35" in flags
    mmt = mm.object_file.bind("mm_ktile", [A_l1, A_l1, A_l1, C_l1, np.int32]) if p35 else None
    zhi_on = "-DZHI" in flags  # round 19: acc_hi zeroed after the C release (mm8.cc acc_zero_hi)
    zhi = mm.object_file.bind("acc_zero_hi", []) if zhi_on else None
    # (with P35 mm_kstep is not bound: an unbound entry point and what only it calls - mm_ks under MM_ROT - leave the elf)
    mmk = mm.object_file.bind("mm_kstep", [A_l1, A_l1, A_l1, B_l1, C_l1]) if kstep and not p35 else None
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
            if os.environ.get("IL") == "1":  # round 20 IL: half 1's loaded chunks get their own full lock (it may load first)
                lk["Fhi"] = Lock(tile=mems[c], init=0, name=f"AFhi{c}")
            if a3:  # round 22 3A: region 3's free / loaded chunks
                lk["E3"] = Lock(tile=mems[c], init=12, name=f"AE3_{c}")
                lk["F3"] = Lock(tile=mems[c], init=0, name=f"AF3_{c}")
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
                "-DL_OUT_FREE=2", "-DL_OUT_FILL=3", "-DL_P_FREE=4", "-DL_P_FILL=5", f"-DPATCH_PARAM_B={PATCH_PARAM_B}", f"-DSSM_PARAM_B={SSM_PARAM_B}", os.environ.get("EW_OPT", "-Oz")] + os.environ.get("EW_DEFS", "").split()
    nres = "-DNRES" in ew_flags  # round 20 NRES: a third ring (the panels, 4 x 576 B) drained by the EW core's MM2S 1
    assert nres == bool(os.environ.get("NRES")), "NRES (plan) and -DNRES (EW_DEFS) go together"
    RingP = np.ndarray[(2 * 576 // 4,), np.dtype[np.uint32]]  # 2 slots (L1: the EW stack gives up 1 KB for it, bank 0)
    if "-DSIGK" in ew_flags:  # round 12: with generated asm (the SSM sigmoid) the object is prebuilt
        ew_op = Kernel("ew_op", ew_object(ew_flags), [RingIn, RingOut] + ([RingP] if nres else []))
    else:
        ew_op = ExternalFunction("ew_op", object_file_name="ew8.o", source_file=str(HERE / "ew8.cc"),
                                 arg_types=[RingIn, RingOut], compile_flags=ew_flags)
    assert not any(o["fn"] == "patch" for o in ew_pre + ew_body + ew_post), "ew_op has no patch kind (the patch embedding is on the host)"
    ew_tiles = [Tile(c, EW_ROW) for c in range(COLS)]
    e_in = [PacketFlow(2, shims[c], ew_tiles[c], src_channel=0, dst_channel=0, name=f"Ein{c}") for c in range(COLS)]
    eout_pkt = nres or os.environ.get("EOUTPKT") == "1"  # (EOUTPKT: the X' packet flow alone, a debug twin)
    if eout_pkt:  # round 20 NRES: the core tile's 4 south ports carry C x 3 + the EW output already: X' (packet 3) and the panels
        # (packet 5) share one of them as two packet flows (EW MM2S 0 -> shim S2MM 1, EW MM2S 1 -> memtile S2MM 5). IRON 1.4.4's
        # pathfinder finds no route for two packet sources on one tile, so Pn is DECLARED from MM2S 0 and r20/nresfix.py moves its
        # rule to the DMA 1 slave in the lowered MLIR (the unpatched xclbin must never run: MM2S 1's packets have no rule)
        e_out = [PacketFlow(3, ew_tiles[c], shims[c], src_channel=0, dst_channel=1, name=f"Eout{c}") for c in range(COLS)]
        p_flow = [PacketFlow(5, ew_tiles[c], mems[c], src_channel=0, dst_channel=5, name=f"Pn{c}") for c in range(COLS)] if nres else []
    else:
        e_out = [Flow(ew_tiles[c], shims[c]) for c in range(COLS)]
        p_flow = []
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
            p_chan = []
            if nres:
                rp = Buffer(tile=t, type=RingP, name=f"ewp{c}", mem_bank=0)
                p_free = Lock(tile=t, init=2, name=f"epf{c}", lock_id=4)
                p_fill = Lock(tile=t, init=0, name=f"epu{c}", lock_id=5)
                p_chan = [DmaChannel(DMAChannelDir.MM2S, 1, [Bd(
                    rp, tap=TensorAccessPattern((2 * 144,), 0, [1, 1, 1, 144], [0, 0, 0, 1]),
                    acquires=[Acquire(p_fill)], releases=[Release(p_free)], iteration=BdIteration(size=2, stride=144), packet=(0, 5))])]
            core_dmas.append(TileDma(t, p_chan + [
                DmaChannel(DMAChannelDir.S2MM, e_in[c].endpoint(t), [Bd(
                    rin, tap=TensorAccessPattern((EW_IN_SLOTS * SLOT_W,), 0, [1, 1, 1, SLOT_W], [0, 0, 0, 1]),
                    acquires=[Acquire(in_free)], releases=[Release(in_fill)],
                    iteration=BdIteration(size=EW_IN_SLOTS, stride=SLOT_W))]),
                DmaChannel(DMAChannelDir.MM2S, e_out[c].endpoint(t), [Bd(
                    rout, tap=TensorAccessPattern((EW_OUT_SLOTS * SLOT_W,), 0, [1, 1, 1, SLOT_W], [0, 0, 0, 1]),
                    acquires=[Acquire(out_fill)], releases=[Release(out_free)],
                    iteration=BdIteration(size=EW_OUT_SLOTS, stride=SLOT_W), **(dict(packet=(0, 3)) if eout_pkt else {}))]),
            ]))

            def ew_fn(rin, rout, ew_op):
                # one op per worker-loop iteration: ew_op reads the op kind from the op's params header, so this code
                # does not depend on the op list (unrolling the block loop here once cost 2 KB of program memory)
                ew_op(rin, rout)

            def ew_fn_p(rin, rout, rp, ew_op):
                ew_op(rin, rout, rp)

            if nres:
                ew_workers.append(Worker(ew_fn_p, [rin, rout, rp, ew_op], tile=t, stack_size=0x1300, data_size=27648))  # (deepest ~3.2 KB; bank 0 = stack + ewo + ewp + 1 KB tables)
            else:
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
    iv32 = os.environ.get("IV32", "1") == "1"  # round 21 (research rev 13 IV32): i32 loop counters in the GEMM core body
    attn_ops = [o for o in core_body if o["kind"] == "attn"]
    assert len({json.dumps(o["passes"]) for o in attn_ops}) <= 1
    if one_seg:
        tops = core_body
    else:
        tops = gemm_ops
    # round 23 MSPLIT-P: a split op's last tile is two C objects (rows 0-5, rows 6-17); mm_ktile gets the op's object count in bits 8+
    tbl_k = np.array([((o["K"] // k) | ((o["nt"] + 1) << 8 if o.get("msplit") else 0) if o["kind"] == "gemm" else 0) for o in tops] or [0], np.int32)
    tbl_n = np.array([((o["nt"] // 2 if o["epi"] == "gated" else o["nt"] + bool(o.get("msplit"))) if o["kind"] == "gemm" else 0) for o in tops] or [0], np.int32)
    tbl_t = np.array([((2 if o["epi"] == "gated" else 1) if o["kind"] == "gemm" else 0) for o in tops] or [0], np.int32)
    tbl_a = np.array([(1 if o["kind"] == "attn" else 0) for o in tops] or [0], np.int32)
    tbl_c = np.array([3, 4, 24, 2], np.int32)  # Q-copy loops (3 x 3), head groups 4, key chunks 24, passes 2

    def to_index(v):
        return arith.IndexCastOp(IndexType.get(), v).result

    def to_i32(v):
        return arith.IndexCastOp(IntegerType.get_signless(32), v).result

    def range32(stop):
        """round 21 IV32: an scf.for with i32 bounds (IRON's range_ casts everything to index, which lowers to i64)"""
        from aie.dialects.scf import ForOp, yield_
        from aie.ir import InsertionPoint
        i32 = IntegerType.get_signless(32)
        if isinstance(stop, int):
            stop = arith.ConstantOp(i32, stop).result
        f = ForOp(arith.ConstantOp(i32, 0).result, stop, arith.ConstantOp(i32, 1).result)
        with InsertionPoint(f.body):
            yield f.induction_variable
            yield_([])

    rng = range32 if iv32 else range_

    def make_core_fn(myrow):
        def core_fn(a_free, a_full, *rest):
            abufs = rest[:a_ring]
            in_b, out_c, mm, zero, epi, at_begin, at_qcopy, at_chunk_k, at_chunk_v, at_end, bk, bn, bt, ba, bc = rest[a_ring:a_ring + 15]
            mmk = rest[a_ring + 15] if kstep and not p35 else None
            mmt = rest[a_ring + 15] if p35 else None
            zhi_k = rest[-1] if zhi_on else None

            def gemm_seg(i0, n):
                for i in range_(i0, i0 + n):
                    ksteps, nobj, tpo = to_index(bk[i]), to_index(bn[i]), to_index(bt[i])
                    for _ in range_(nobj):
                        cbuf = out_c.acquire(1)
                        for _ in range_(tpo):
                            zero(cbuf)
                            if mmt is not None:
                                mmt(abufs[0], abufs[1], abufs[2], cbuf, bk[i])
                            for _ in range(0) if mmt is not None else range_(ksteps):
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
                        if zhi_k is not None:
                            zhi_k()

            def one_attn(op):
                # per pass the 36 Q objects stream through the A ring in [head][tile] order (3 sub-tiles each);
                # this core keeps those of heads myrow + 3 i and processes each head after its triple arrived.
                # pass 0 = query tiles {0, 1}, pass 1 = tile {2}: ntiles = 2 - pass
                # (round 21 IV32: i32 loop counters straight from the L1 tables; IRON's index type lowers to i64)
                two = arith.ConstantOp(IntegerType.get_signless(32) if iv32 else IndexType.get(), 2).result
                three = bc[0] if iv32 else to_index(bc[0])
                for pi in rng((bc[3] if iv32 else to_index(bc[3])) if one_seg else len(op["passes"])):
                    ntiles = arith.SubIOp(two, pi).result if iv32 else to_i32(arith.SubIOp(two, pi).result)
                    pi32 = pi if iv32 else to_i32(pi)
                    for _ in rng((bc[1] if iv32 else to_index(bc[1])) if one_seg else 4):
                        for hh in rng(three if one_seg else 3):
                            for t in rng(three if one_seg else 3):
                                a_full.acquire(1)
                                a_full.acquire(1)
                                a_full.acquire(1)
                                at_qcopy(abufs[0], abufs[1], abufs[2], hh if iv32 else to_i32(hh), t if iv32 else to_i32(t), pi32, myrow)
                                a_free.release(1)
                                a_free.release(1)
                                a_free.release(1)
                        cbuf = out_c.acquire(1)
                        at_begin(cbuf, ntiles)
                        for _ in rng((bc[2] if iv32 else to_index(bc[2])) if one_seg else 24):
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
                for i0 in rng(len(tops)):
                    i = to_index(i0) if iv32 else i0
                    ksteps, nobj, tpo = (bk[i], bn[i], bt[i]) if iv32 else (to_index(bk[i]), to_index(bn[i]), to_index(bt[i]))
                    for _ in rng(nobj):
                        cbuf = out_c.acquire(1)
                        for _ in rng(tpo):
                            zero(cbuf)
                            if mmt is not None:
                                mmt(abufs[0], abufs[1], abufs[2], cbuf, bk[i])
                            for _ in range(0) if mmt is not None else range_(ksteps):
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
                        if zhi_k is not None:
                            zhi_k()
                    if attn_ops:
                        for _ in rng(ba[i] if iv32 else to_index(ba[i])):
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
            tabs = [Buffer(tile=t, initial_value=a, name=f"tb{n}{r}{c}", **(dict(mem_bank=1) if e7 else {}))  # E7: bank 3 is full
                    for n, a in (("k", tbl_k), ("n", tbl_n), ("t", tbl_t), ("a", tbl_a), ("c", tbl_c))]
            workers.append(Worker(make_core_fn(r), [free, fill, *bufs, b_l1[r].cons(), c_l1[r][c].prod(), mm, zero, epi,
                                                    at_begin, at_qcopy, at_chunk_k, at_chunk_v, at_end, *tabs]
                                  + ([mmk] if mmk is not None else []) + ([mmt] if p35 else []) + ([zhi] if zhi_on else []),
                                  tile=t, stack_size=((0xA00 if "-DGELU_FLAT" in flags else 0xB00) if ks_core else 0xC00) + (E7_XB if e7 else 0),
                                  data_size=28160 + (256 if ks_core else 0) + (256 if "-DGELU_FLAT" in flags else 0) - (E7_XB if e7 else 0)))
            # round 12: the flat-edge tables' ab copies (+256 B) from the stack (stackaudit: 1,664 of 2,816 B used)  # stack: deepest chain 2.24 KB measured (r7); data: +2 KB the GELU tables' ab copies (cd: bank 3), +1.25 KB the exp table (r7)
    workers += ew_workers

    W_ty = np.ndarray[(w_words,), np.dtype[np.uint32]]
    ACT_ty = np.ndarray[(act_words,), np.dtype[np.uint32]]

    def sequence(w, act, B_prods, C_conses):
        bufs = dict(w=w, act=act)
        if "-DKSP38" in kdefs.replace(",", " ").split():  # round 18 (research rev 13 KS-P38): the bus + core events counted per cycle
            from aie.dialects.aiex import npu_maskwrite32, npu_write32  # Performance_Control0 / 1: start = stop = the event

            ev = [int(v) for v in os.environ.get("KSP38_EV", "23,26,1").split(",")]
            for c in range(COLS):
                for r in range(ROWS_G):
                    npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=2 + r)
                    npu_write32(address=0x31500, value=ev[0] | ev[0] << 8 | ev[1] << 16 | ev[1] << 24, column=c, row=2 + r)
                    npu_write32(address=0x31504, value=ev[2] | ev[2] << 8, column=c, row=2 + r)
        if {"-DPROF", "-DTPROF", "-DATPROF"} & set(kdefs.replace(",", " ").split()):  # round 7 timing build (round 19: TPROF): enable every GEMM core's processor bus
            from aie.dialects.aiex import npu_maskwrite32  # (Core_Processor_Bus 0x32038 bit 0) so read_tm does not hang

            for c in range(COLS):
                for r in range(ROWS_G):
                    npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=2 + r)
            if os.environ.get("CPST_OP"):  # round 20 probe (mm8 -DCPST): core 0_2 spins on its flag word until the CP passes the
                from aie.dialects.aiex import npu_write32  # dependency wait of CPST_OP; every other core's flag is preset to 1

                for c in range(COLS):
                    for r in range(ROWS_G):
                        npu_write32(address=int(os.environ["CPST_ADDR"], 0) - 0x70000, value=2 if (c or r) else 0, column=c, row=2 + r)
        if {"-DPROF", "-DNPROF", "-DNIO2"} & set(os.environ.get("EW_DEFS", "").split()):  # round 8: the EW cores' phase stamps (ew8.cc PROF)
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

        seqlog = os.environ.get("SEQLOG") == "1"  # round 23: print every group finish (the CP waits) with the op being issued

        class G(TaskGroup):  # a TaskGroup that reports its finish to the accounting
            def finish(self):
                if seqlog:
                    nm = next((n_ for n_, g_ in pending if g_ is self), "?")
                    print(f"[seq] {cur_op[0]}: finish {nm}")
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
                    return Bd(slot, tap=TensorAccessPattern((ra_buf,), 12 * CH * h, [1, 1, 1, CH], [0, 0, 0, 1]), acquires=[Acquire(acq)],
                              releases=[Release(rel)], iteration=BdIteration(size=12, stride=CH))

                E = [lk["Elo"], lk["Ehi"]] + ([lk["E3"]] if a3 else [])
                F = [lk["F"], lk.get("Fhi", lk["F"])] + ([lk["F3"]] if a3 else [])
                t = dict(ld=[me_in.task(chunk(h, E[h], F[h]), runs=12) for h in range(len(E))],
                         t0=[me_out.task(chunk(h, F[h], lk["F2"]), runs=12) for h in range(len(E))],
                         last=[None if h == 2 and any(o.get("msplit") and o.get("r3") for o in ops) else me_out.task(chunk(h, lk["F2"], E[h]), runs=12)
                               for h in range(len(E))],
                         mid={K_: me_out.task(Bd(slot, tap=TensorAccessPattern((ra,), 0, [1, 1, 1, (K_ // k) * CH], [0, 0, 0, 1])))
                              for K_ in (768, 1536)})
                if a3:  # round 22 3A: a K=1536 op's middle tiles read k-steps 0-11 from half 0 and 12-23 from region 3
                    assert (12 * CH) % 972 == 0
                    t["mid"]["r3"] = me_out.task(Bd(slot, tap=TensorAccessPattern((ra_buf,), 0, [1, 2, 12 * CH // 972, 972],
                                                                                   [0, 24 * CH, 972, 1])))
                ms3 = any(o.get("msplit") and o.get("r3") for o in ops)
                ms1 = any(o.get("msplit") and o["K"] == 768 for o in ops if o["kind"] == "gemm")  # (K = 768 split ops: A in half 1)
                if ms3:  # round 23 MSPLIT-P: rows 0-5 of every chunk of both regions (lock-free: the
                    # chunks are loaded, tile 0 took them), then rows 6-17 chunk by chunk, releasing each chunk as the last tile does
                    assert a3 and CH % 3 == 0
                    t["pa"] = me_out.task(Bd(slot, tap=TensorAccessPattern((ra_buf,), 0, [1, 2, 12, CH // 3], [0, 24 * CH, CH, 1])))
                    # (region 3's last-tile task is the second pass's: only the split ops use region 3)
                    assert all(o.get("msplit") for o in ops if o.get("r3"))
                if ms1:  # (half 1's rows 0-5 of its 12 chunks)
                    t["pa1"] = me_out.task(Bd(slot, tap=TensorAccessPattern((ra_buf,), 12 * CH, [1, 1, 12, CH // 3], [0, 0, CH, 1])))
                if ms3 or ms1:
                    t["lastb"] = [None if not ((h == 1 and ms1) or (h != 1 and ms3)) else me_out.task(Bd(slot, tap=TensorAccessPattern((ra_buf,), 12 * CH * h + CH // 3, [1, 1, 1, 2 * CH // 3],
                                                                                                    [0, 0, 0, 1]),
                                                                     acquires=[Acquire(lk["F2"])], releases=[Release(E[h])],
                                                                     iteration=BdIteration(size=12, stride=CH)), runs=12) for h in range(3)]
                if p_flow:  # round 20 NRES: the norm's units arrive rb-major (rb, q); a 3D TAP puts unit (rb, q) at chunk q, row block rb
                    # (the usual chunk-major half 0: the op's sends and the last tile's chunk-wise release stay as they are; an rb-major
                    # half let the next op's chunk-0 load overwrite every chunk's first row blocks while the last tile still read them)
                    zh = 1 if yxssm else 0  # round 23 YXSSM: the z panels into half 1 (half 0 holds yx's A until its last tile)
                    if os.environ.get("ZCH") == "1":  # round 23 ZCH: one chunk (head) per run, released as it lands: out's k-step q follows head q
                        t["ldz"] = p_flow[c].endpoint(mems[c]).task(Bd(slot, tap=TensorAccessPattern((ra,), 12 * CH * zh, [1, 1, 1, CH], [0, 0, 0, 1]),
                                                                        acquires=[Acquire(E[zh])], releases=[Release(F[zh])],
                                                                        iteration=BdIteration(size=12, stride=CH)), runs=12)
                    else:
                        t["ldz"] = p_flow[c].endpoint(mems[c]).task(Bd(slot, tap=TensorAccessPattern((ra,), 12 * CH * zh, [1, 1, 216, 144], [0, 0, 144, 1]),
                                                                        acquires=[Acquire(E[zh], value=12)], releases=[Release(F[zh], value=12)]))
                    if yxssm or oph1:  # (out's / out_proj's middle tiles from half 1)
                        t["mid"]["h1"] = me_out.task(Bd(slot, tap=TensorAccessPattern((ra,), 12 * CH, [1, 1, 1, 12 * CH], [0, 0, 0, 1])))
                    # (SRES: the SSM's z panels arrive head-major = chunk-major already)
                    t["ldn"] = p_flow[c].endpoint(mems[c]).task(Bd(slot, tap=TensorAccessPattern((ra,), 0, [1, 18, 12, 144], [0, 144, CH, 1]),
                                                                    acquires=[Acquire(lk["Elo"], value=12)], releases=[Release(lk["F"], value=12)]))
                rt_tasks.append(t)
        slot_ops = [o for o in ops if o["kind"] in ("gemm", "attn")]

        def a_loads(op):  # A_RT: start column c's chunked loads (after the op's fills)
            if a_rt and not op.get("reuse_a") and not op.get("nres"):  # (NRES: the norm's op pushed the S2MM 5 task)
                hs = list(range(op.get("K", 768) // 768))
                if op["kind"] == "attn" and aqpre:  # round 20 AQPRE: the attention's Q lives in half 1 (qkv still holds half 0)
                    hs = [1]
                if op.get("h1first"):  # round 20 IL: the fill sends chunks 12-23 first
                    assert os.environ.get("IL") == "1"  # (the per-half full locks)
                    hs = hs[::-1]
                if op.get("r3"):  # round 22 3A: k-steps 12-23 into region 3 first (free since the last r3 op), then 0-11 into half 0
                    hs = [2, 0]
                if oph1 and op["name"].endswith("_out_proj"):  # round 23 OPH1: out_proj's A into half 1 (norm3's NRES panels: half 0)
                    hs = [1]
                for c in range(COLS):
                    for h in hs:
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
            hs = [1] if op["kind"] == "attn" and aqpre else list(range(K // 768))
            if op.get("r3"):  # round 22 3A: k order = half 0, then region 3
                hs = [0, 2]
            z1 = yxssm and op.get("nres") and i > 0 and slot_ops[i - 1]["name"].endswith("_yx")  # round 23 YXSSM: out's A (the z panels) in half 1
            z1 = z1 or (oph1 and op["name"].endswith("_out_proj"))  # (round 23 OPH1: out_proj too)
            if z1:
                hs = [1]
            assert not (nmid and hs == [1]) or z1  # (the attention: two passes = t0 + last, no whole-slot mid send)
            for c in range(COLS):
                t = rt_tasks[c]
                if not op.get("reuse_a"):
                    for h in hs:
                        t["t0"][h].start()
                if nmid:
                    t["mid"]["h1" if z1 else "r3" if op.get("r3") else K].start(repeat_count=nmid - 1)
                if op.get("msplit") and hs == [1]:  # round 23: a K = 768 split op from half 1 (four tasks: no deferral)
                    assert not reused_next
                    if msdefer and op["name"] in outpre:  # round 24 MSDEFER: pushed by the SSM after yx's last drain group
                        deferred.append(t)
                        continue
                    t["pa1"].start()
                    t["lastb"][1].start()
                elif op.get("msplit"):  # round 23 MSPLIT-P: the last tile's rows 0-5, then rows 6-17 with the chunk releases
                    assert not reused_next and hs == [0, 2]
                    t["pa"].start()
                    t["lastb"][0].start()
                elif not reused_next:
                    for h in hs:
                        t["last"][h].start()
            # (a column's sixth task only after every column has its first five: a push into a full queue waits for that column's
            # tile 0, whose B multicast needs every column's A sends issued; pushed column by column it hung, r23d)
            for c in range(COLS) if op.get("msplit") and op.get("r3") else ():
                rt_tasks[c]["lastb"][2].start()

        pending = []  # [(name, TaskGroup)]
        aqpre = os.environ.get("AQPRE") == "1"
        akvpre = os.environ.get("AKVPRE") == "1"  # round 20: head group 0's K / V fills after qkv's step-8 drain group
        assert not akvpre or aqpre  # round 20: attention Q in half 1, filled before the wait on qkv's K / V drains
        assert not aqpre or os.environ.get("IL") == "1", "AQPRE needs the per-half full locks (IL)"

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

        p43 = os.environ.get("P43") == "1"
        seqpre = os.environ.get("SEQPRE", "0") != "0"
        ms_names = {o["name"] for o in ops if o.get("msplit")}
        assert not yxssm or (os.environ.get("SRES") == "1" and os.environ.get("IL") == "1" and seqpre)
        MS_RB = 6  # (plan8.MS_RB: mm8.cc's first row pass)
        assert not ms_names or seqpre and os.environ.get("NSEQ") == "1"
        sp_mode = int(os.environ.get("SEQPRE", "0"))  # round 19: bit 1 = slot loads before the wait, bit 2 = drains before it  # round 11 (research P43): a GEMM's B (weights) leaves before its dependency wait

        seq_ops, outpre = list(ops), set()
        ops_by_name = {o["name"]: o for o in ops}
        msdefer = os.environ.get("MSDEFER") == "1"  # round 24
        oppre2 = os.environ.get("OPPRE2") == "1"  # round 24: out_proj's A per attention head group (pass 1)
        assert not oppre2 or (seqpre and os.environ.get("SEQPRE") == "1" and a_half), "OPPRE2: SEQPRE=1 (a_loads before the waits), A_HALF"
        deferred = []  # (rt_tasks entries whose pass sends wait for the SSM)

        def push_deferred():
            for t in deferred:
                t["pa1"].start()
                t["lastb"][1].start()
            if seqlog and deferred:
                print(f"[seq] {cur_op[0]}: deferred pass sends pushed ({len(deferred)} columns)")
            deferred.clear()
        if os.environ.get("OUTPRE") == "1":  # round 23: out before the SSM in the CP stream (locks / ZCH chunks pace its first tile)
            assert yxssm and os.environ.get("ZCH") == "1"
            for i in range(len(seq_ops) - 1):
                a_, b_ = seq_ops[i], seq_ops[i + 1]
                if a_["kind"] == "ew" and a_.get("sres") and b_["kind"] == "gemm" and b_.get("nres") and not b_.get("deps"):
                    seq_ops[i], seq_ops[i + 1] = b_, a_
                    outpre.add(b_["name"])
        for op in seq_ops:
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
                        if op.get("msplit"):  # round 23 MSPLIT-P: the last tile's B again for its second row pass
                            tp = TensorAccessPattern((w_words,), op["w"] // 4 + r * nt * lw + (nt - 1) * lw, [1, K // k + 1, wb // wb0, wb0],
                                                     [lw, wb, wb0, 1])
                            account(f"shim{r}", tg)
                            B_prods[r].fill(w, tap=tp, group=tg, wait=False)

                if p43:  # B has no data dependency; its group is finished after its drains as before
                    fill_b()
                if os.environ.get("CPRATE") and op["name"].endswith("_mlp3_a"):  # (before the wait: timing probe only)  # round 19 probe (research rev 15 CPRATE): N dummy
                    from aie.dialects.aiex import npu_write32  # CP writes on mlp3_a's start path (Performance_Control0 of a GEMM core = 0)

                    if os.environ.get("CPRATE_POLL") == "1":  # the same count of maskpolls (mask 0: always satisfied)
                        from aie.dialects.aiex import npu_maskpoll
                        from aie.ir import IntegerType

                        i32c = lambda v: arith.ConstantOp(IntegerType.get_signless(32), v).result
                        for i in range(int(os.environ["CPRATE"])):
                            npu_maskpoll(i32c(0x31500), i32c(0), i32c(0), column=i % 8, row=2)
                    else:
                        for i in range(int(os.environ["CPRATE"])):
                            npu_write32(address=0x31500, value=0, column=i % 8, row=2)
                # finish() frees a group's unwaited fills WITHOUT awaiting them: a fill may only sit in a group that is
                # finished after the fill completed (its drains), else it must be waited (wait=True)
                if seqpre:  # round 19 (research rev 15 SEQPRE): the op's memtile tasks (lock-governed) and its drains before the
                    # producer wait, so only the shim A fills sit on the post-sync path
                    # v2: the memtile SENDS stay after the A fills (a K=1536 op pushes 5 tasks on the 4-deep MM2S queue: a push that
                    # blocks before the fills waits on work not yet issued; v1 with the sends first: +1.1 ms)
                    if sp_mode & 1:
                        a_loads(op)
                    if not qkv and sp_mode & 2:
                        for c in range(COLS):
                            drain(C_conses[c], op["c"][c], group=tg, wait=True)
                    op2 = oppre2 and any(n_ == d_ + "#h0" for n_, _ in pending for d_ in op.get("deps", []))
                    if op2:  # round 24 OPPRE2: out_proj's A part g after the attention's pass 0 and pass-1 head group g
                        an = next(d_ for d_ in op["deps"] if any(n_ == d_ + "#h0" for n_, _ in pending))
                        if os.environ.get("OPPRE2_SENDS", "1") == "1":  # (v2: the sends before the waits)
                            a_sends(op)
                        for g in range(-1, 4):
                            want = an + ("#p0" if g < 0 else f"#h{g}")
                            for n_, g_ in list(pending):
                                if n_ == want:
                                    g_.finish()
                                    pending.remove((n_, g_))
                            for c in range(COLS) if g >= 0 else ():
                                (t_,) = op["a"][c]
                                assert t_["sizes"][0] == 12, t_
                                fill(a_inb[c], dict(t_, sizes=[3] + t_["sizes"][1:], off=t_["off"] + 3 * g * t_["strides"][0]), group=tg, wait=False)
                    finish_deps(op)
                    if op["name"] == os.environ.get("CPST_OP"):  # round 20 probe: the CP passed this op's dependency wait
                        from aie.dialects.aiex import npu_write32

                        npu_write32(address=int(os.environ["CPST_ADDR"], 0) - 0x70000, value=1, column=0, row=2)
                    split = op.get("split", 0)
                    tg0 = G() if split else tg  # round 20 IL: the A fills + steps [0, split) drain in "<name>#0", B + the rest in tg
                    for c in range(COLS) if not op2 else ():
                        fill_a(op, c, group=tg0, wait=qkv)
                    if not sp_mode & 1:
                        a_loads(op)
                    if not p43:
                        fill_b()
                    if not (op2 and os.environ.get("OPPRE2_SENDS", "1") == "1"):
                        a_sends(op)
                    if split:
                        assert p43 and not sp_mode & 2 and not qkv
                        for part, g_ in ((0, tg0), (1, tg)):
                            for c in range(COLS):
                                t_ = dict(op["c"][c])
                                n0 = t_["sizes"][0]
                                t_["sizes"] = [split if part == 0 else n0 - split] + t_["sizes"][1:]
                                t_["off"] = t_["off"] + (split * t_["strides"][0] if part else 0)
                                drain(C_conses[c], t_, group=g_, wait=True)
                        push(op["name"] + "#0", tg0)
                        push(op["name"], tg)
                        continue
                    if yxssm and op["name"].endswith("_yx"):  # round 23 YXSSM: y + x steps 4 / 5 / 6 / 7 (heads 0-2 / 3-5 / 6-8 / 9-11)
                        assert p43 and not sp_mode & 2 and not qkv and op.get("rr")
                        gs = [G(), G(), G(), tg]
                        if op.get("yxh"):  # round 24 YXH: drain groups head 0 | heads 1-2 | 3-5 | 6-11 (four TAPs: a shim S2MM channel
                            # queues 5 tasks, out's drain is the fifth; seven yx TAPs blocked the CP before the SSM's fills, r24b +0.44)
                            for c in range(COLS):
                                t_ = op["c"][c]
                                assert t_["sizes"] == [12, 2, 18, 1024], t_
                                for g, (h0, nh) in enumerate(((0, 1), (1, 2), (3, 3), (6, 6))):
                                    drain(C_conses[c], dict(t_, sizes=[nh, 2, 18, 1024], off=t_["off"] + h0 * t_["strides"][0]), group=gs[g], wait=True)
                            push(op["name"] + "#0", gs[0])
                            pending.extend([(op["name"] + "#1", gs[1]), (op["name"] + "#2", gs[2]), (op["name"], gs[3])])  # (past the lookahead)
                            continue
                        for c in range(COLS):
                            t_ = op["c"][c]
                            if op.get("perm"):  # round 23 YXR: group g = steps 2 g, 2 g + 1 (y and x of heads 3 g .. 3 g + 2)
                                assert t_["sizes"] == [4, 2, 54, 1024], t_
                                for g in range(4):
                                    drain(C_conses[c], dict(t_, sizes=[1, 2, 54, 1024], off=t_["off"] + g * t_["strides"][0]), group=gs[g], wait=True)
                                continue
                            assert t_["sizes"] == [2, 12, 18, 1024], t_
                            drain(C_conses[c], dict(t_, sizes=[1, 12, 18, 1024]), group=gs[0], wait=True)
                            for g in range(4):
                                drain(C_conses[c], dict(t_, sizes=[1, 3, 18, 1024], off=t_["off"] + t_["strides"][0] + 3 * g * t_["strides"][1]),
                                      group=gs[g], wait=True)
                        push(op["name"] + "#0", gs[0])
                        pending.extend([(op["name"] + "#1", gs[1]), (op["name"] + "#2", gs[2]), (op["name"], gs[3])])  # (past the lookahead)
                        continue
                    if op.get("msplit"):  # round 23 MSPLIT-P: the norm after it waits only for "#0" before its rows 0-5
                        assert p43 and not sp_mode & 2 and not qkv
                        tgm = G()
                        for c in range(COLS):
                            t_ = op["c"][c]
                            assert t_["sizes"][0] == nt
                            drain(C_conses[c], t_, group=tgm, wait=True)
                            drain(C_conses[c], dict(t_, sizes=[1] + t_["sizes"][1:], off=t_["off"] + (nt - 1) * t_["strides"][0]), group=tg, wait=True)
                        if op["name"] in outpre:  # (OUTPRE: no lookahead finish here, as below)
                            pending.append((op["name"] + "#0", tgm))
                        else:
                            push(op["name"] + "#0", tgm)
                        pending.append((op["name"], tg))  # (past the lookahead: the norm finishes "#0" alone first)
                        continue
                    if not qkv and not sp_mode & 2:
                        for c in range(COLS):
                            drain(C_conses[c], op["c"][c], group=tg, wait=True)
                    if not qkv:
                        if op["name"] in outpre:  # (no lookahead finish here: the yx groups the SSM waits for one by one)
                            pending.append((op["name"], tg))
                        else:
                            push(op["name"], tg)
                        continue
                else:
                    finish_deps(op)
                    for c in range(COLS):
                        fill_a(op, c, group=tg, wait=qkv)
                    a_loads(op)
                    if not p43:
                        fill_b()
                    a_sends(op)  # before any wait of this op: a blocked push may only wait on work issued already
                if qkv:  # 12 drains per column, issued 4 tiles at a time, 2 groups in flight; the fills ride in the last group
                    finish_all()
                    groups = []
                    # round 20 AKVPRE: V's first step (8 = heads 0-2: the attention's head group 0) is a group of its own, so the
                    # attention can fill head group 0's K / V before qkv's last three steps are drained
                    spans = [(0, 4), (4, 8), (8, 9), (9, 12)] if akvpre else [(0, 4), (4, 8), (8, 12)]
                    for g, (j0, j1) in enumerate(spans):
                        tgd = tg if g == len(spans) - 1 else G()
                        for c in range(COLS):
                            for j in range(j0, j1):
                                drain(C_conses[c], op["c"][c][j], group=tgd, wait=True)
                        groups.append(tgd)
                        if len(groups) > 2:
                            groups.pop(0).finish()
                    for k_, tgd in enumerate(groups):
                        pending.append((op["name"] + ("#v0" if akvpre and k_ == 0 else ""), tgd))
                else:
                    for c in range(COLS):
                        drain(C_conses[c], op["c"][c], group=tg, wait=True)
                    push(op["name"], tg)
            elif op["kind"] == "attn" and seqpre:  # round 19 SEQPRE: memtile tasks and drains before the wait
                tg = G()
                a_loads(op)
                op2 = oppre2 and len(op["passes"]) == 2  # round 24 OPPRE2: pass 0's drain alone now, pass 1's per head group later
                p0g = G() if op2 else tg
                for pi in range(1 if op2 else len(op["passes"])):
                    for c in range(COLS):
                        drain(C_conses[c], op["c"][pi][c], group=p0g, wait=True)
                if aqpre:  # round 20: qkv's Q drains (steps 0-3, its first drain group) were finished inside qkv's sequence when its
                    # third group was issued; the Q fill (into half 1) goes before the wait on the K / V groups
                    assert ops[ops.index(op) - 1].get("rr") and ops[ops.index(op) - 1]["epi"] == "qkv"
                    for c in range(COLS):
                        fill_a(op, c, group=tg, wait=False)
                    a_sends(op)  # (2 tasks behind qkv's mid + last on the 4-deep queue; lock-governed)
                if akvpre:  # V of heads 0-2 (step 8) is drained: head group 0's K / V fills (pass 0) go now
                    qn = ops[ops.index(op) - 1]["name"] + "#v0"
                    for name, g_ in list(pending):
                        if name == qn:
                            g_.finish()
                            pending.remove((name, g_))
                    for r in range(ROWS_G):
                        fill(B_prods[r], op["b"][r][0], group=tg, wait=False)
                finish_deps(op)
                for c in range(COLS) if not aqpre else ():
                    fill_a(op, c, group=tg, wait=False)
                for pi in range(len(op["passes"])):
                    for r in range(ROWS_G):
                        for i_, t in enumerate(op["b"][r]):
                            if not (akvpre and pi == 0 and i_ == 0):
                                fill(B_prods[r], t, group=tg, wait=False)
                    if pi == 0 and not aqpre:
                        a_sends(op)
                if op2:  # round 24 OPPRE2: pass 1's O in four head-group drains ([4 groups][3 rows][object]), groups past the lookahead
                    hg = [G() for _ in range(4)]
                    for g in range(4):
                        for c in range(COLS):
                            t_ = op["c"][1][c]
                            assert t_["sizes"] == [4, 3, 1, t_["sizes"][3]], t_
                            drain(C_conses[c], dict(t_, sizes=[1] + t_["sizes"][1:], off=t_["off"] + g * t_["strides"][0]), group=hg[g], wait=True)
                    pending.extend([(op["name"] + "#p0", p0g)] + [(op["name"] + f"#h{g}", hg[g]) for g in range(4)] + [(op["name"], tg)])
                    continue
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
                nseq = op["fn"] == "norm" and os.environ.get("NSEQ") == "1"
                yxg = [(n_, g_) for n_, g_ in pending if yxssm and op["fn"] == "ssm" and n_.split("#")[0].endswith("_yx")]
                if yxg:  # round 23 YXSSM: everything but yx's groups now; yx's group g before head 3 g's fill
                    for n_, g_ in list(pending):
                        if (n_, g_) not in yxg and n_.split("#")[0] not in outpre:  # (OUTPRE: out completes only after this SSM)
                            g_.finish()
                            pending.remove((n_, g_))
                elif not nseq:
                    finish_all()
                if op["fn"] == "norm":
                    tg = G()
                    if op.get("nres"):  # round 20 NRES: the memtile takes the panels (S2MM 5) before the norm can push any
                        for c in range(COLS):
                            rt_tasks[c]["ldn"].start()
                    if nseq:  # round 20 (research rev 16 NSEQ): params (from W) and drains before the producer wait; after it only
                        # the input fills, so the CP issues column 7's input ~4 instructions after column 0's (it was ~12)
                        assert len(op["fills"][0]) == 2 and op["fills"][0][0]["buf"] == "w"
                        for c in range(COLS):
                            fill(e_in[c], op["fills"][c][0], group=tg, wait=False)
                            drain(e_out[c], op["c"][c], group=tg, wait=True)
                        ntwin = [x for x in os.environ.get("NTWIN", "").split(",") if x]  # round 22 timing twin (research NTWIN, outputs wrong):
                        ms = [(n_, g_) for n_, g_ in pending if n_ in ms_names]
                        if ms:  # round 23 MSPLIT-P: row blocks 0-5 while the producer computes rows 6-17
                            for n_, g_ in list(pending):
                                if (n_, g_) not in ms:
                                    g_.finish()
                                    pending.remove((n_, g_))
                            for c in range(COLS):
                                f_ = op["fills"][c][1]
                                fill(e_in[c], dict(f_, sizes=[MS_RB] + f_["sizes"][1:]), group=tg, wait=False)
                            finish_all()
                            for c in range(COLS):
                                f_ = op["fills"][c][1]
                                fill(e_in[c], dict(f_, sizes=[f_["sizes"][0] - MS_RB] + f_["sizes"][1:], off=f_["off"] + MS_RB * f_["strides"][0]),
                                     group=tg, wait=False)
                        else:
                            if op["name"].split("_", 1)[-1] not in ntwin:  # these norms do not wait for their producer's drains
                                finish_all()
                            for c in range(COLS):
                                fill(e_in[c], op["fills"][c][1], group=tg, wait=False)
                    for c in range(COLS) if not nseq else ():
                        for f in op["fills"][c]:
                            fill(e_in[c], f, group=tg, wait=False)
                        drain(e_out[c], op["c"][c], group=tg, wait=True)
                    if op.get("nres"):  # the next GEMM issues its B, sends and drains before this norm is awaited
                        push(op["name"], tg)
                    else:
                        tg.finish()
                else:  # patch / ssm: one drain per column, per-head fill pairs with a lookahead of 2
                    if op.get("sres"):  # round 20 SRES: the memtile takes the z panels (S2MM 5)
                        for c in range(COLS):
                            rt_tasks[c]["ldz"].start()
                    tgd = G()
                    for c in range(COLS):
                        drain(e_out[c], op["c"][c], group=tgd, wait=True)
                    groups = []
                    for q in range(len(op["fills"][0])):
                        yxh_ = yxg and ops_by_name[yxg[0][0].split("#")[0]].get("yxh")
                        if yxg and (q in (0, 1, 3, 6) if yxh_ else q % 3 == 0):  # (yx's group: YXR "#0" .. "#2" at q 0 / 3 / 6, the op's own
                            gi = {0: 0, 1: 1, 3: 2}.get(q, 3) if yxh_ else q // 3  # name at 9; YXH "#0" .. "#2" at 0 / 1 / 3, its name at 6)
                            want = yxg[0][0].split("#")[0] + (f"#{gi}" if gi < 3 else "")
                            for n_, g_ in list(pending):
                                if n_ == want:
                                    g_.finish()
                                    pending.remove((n_, g_))
                            if gi == 3:  # round 24 MSDEFER: yx is drained, its memtile tasks are done
                                push_deferred()
                        tg = G()
                        for c in range(COLS):
                            for f in op["fills"][c][q]:
                                fill(e_in[c], f, group=tg, wait=True)  # finished two heads later: must be awaited, not just freed
                        groups.append(tg)
                        if len(groups) > 2:
                            groups.pop(0).finish()
                    for tg in groups:
                        tg.finish()
                    if op.get("sres") and any(n_.split("#")[0] in outpre for n_, _ in pending):  # (OUTPRE: a lookahead finish here
                        pending.append((op["name"], tgd))  # would wait for all of out, also its second row pass, before norm1)
                    elif op.get("sres"):  # the GEMM after it issues its work before the SSM's drains are awaited
                        push(op["name"], tgd)
                    else:
                        tgd.finish()
            else:
                raise NotImplementedError(op["kind"])
        assert not deferred, "MSDEFER: pass sends never pushed"
        finish_all()
        for t in rt_tasks:
            for task in (*t["ld"], *t["t0"], *[x for x in t["last"] if x is not None], *t["mid"].values(), *[t[x] for x in ("ldn", "ldz", "pa", "pa1") if x in t],
                         *[x for x in t.get("lastb", []) if x is not None]):
                task.free()
        print(f"[design8] shim BD peak {bd_peak[0]} at {bd_peak[1]} (limit 16)")

    rt = Runtime(sequence, [W_ty, ACT_ty, b_prods, c_conses])
    for f in (*a_in, *a_out, *(e_in + e_out if any_ew else []), *p_flow):
        rt.add_flow(f)
    for d in (*mem_dmas, *core_dmas):
        rt.add_tile_dma(d)
    for lk in rt_locks:
        for lock in lk.values():
            rt.add_lock(lock)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def ksc_check(out, kdefs):
    """round 18 KSC: -DKS_ACC_HI / KS_C / KS_A0..2 (L1 byte addresses as the core sees them, 0x70000 + offset) against every GEMM
    core's linked acc_hi (llvm-nm) and its A slot / C object buffers in input_with_addresses.mlir"""
    import re
    import subprocess

    want = {k: int(v, 0) for k, v in re.findall(r"-D(KS_ACC_HI|KS_C|KS_A[012]|KS_B[01]|KS_BPROD|KS_BCONS)=(0x[0-9a-fA-F]+|\d+)",
                                                 kdefs.replace(",", " "))}
    assert len(want) in (5, 9), f"KSC needs KS_ACC_HI, KS_C, KS_A0..2 (P35: + KS_B0, KS_B1, KS_BPROD, KS_BCONS): {want}"
    prj = Path(out + ".prj")
    mlir = (prj / "input_with_addresses.mlir").read_text()
    bad = []
    for r in range(ROWS_G):
        for c in range(COLS):
            tile = f"tile_{c}_{r + 2}"
            for i in range(3):
                m_ = re.search(rf"%a{r}{c}_{i} = aie.buffer\(%{tile}\) {{address = (\d+)", mlir)
                if not m_ or 0x70000 + int(m_.group(1)) != want[f"KS_A{i}"]:
                    bad.append(f"{tile} a slot {i}: {m_ and hex(0x70000 + int(m_.group(1)))}")
            if "KS_B0" in want:  # round 18 P35: the B fifo's L1 buffers and locks
                for i in range(2):
                    m_ = re.search(rf"%B{r}L1_{c}_cons_buff_{i} = aie.buffer\(%{tile}\) {{address = (\d+)", mlir)
                    if not m_ or 0x70000 + int(m_.group(1)) != want[f"KS_B{i}"]:
                        bad.append(f"{tile} B buffer {i}: {m_ and hex(0x70000 + int(m_.group(1)))}")
                for nm, key in (("prod", "KS_BPROD"), ("cons", "KS_BCONS")):
                    m_ = re.search(rf"%B{r}L1_{c}_cons_{nm}_lock_0 = aie.lock\(%{tile}, (\d+)\)", mlir)
                    if not m_ or int(m_.group(1)) != want[key]:
                        bad.append(f"{tile} B {nm} lock: {m_ and m_.group(1)}")
            m_ = re.search(rf"%C{c}_{r}_buff_0 = aie.buffer\(%{tile}\) {{address = (\d+)", mlir)
            if not m_ or 0x70000 + int(m_.group(1)) != want["KS_C"]:
                bad.append(f"{tile} C: {m_ and hex(0x70000 + int(m_.group(1)))}")
            elf = prj / f"elfs_main_core_{c}_{r + 2}" / f"elfs_main_core_{c}_{r + 2}.elf"
            nm = subprocess.run([f"{PEANO}/bin/llvm-nm", str(elf)], capture_output=True, text=True, check=True).stdout
            m_ = re.search(r"^([0-9a-f]+) [bBdD] _ZL6acc_hi$", nm, re.M)
            if not m_ or int(m_.group(1), 16) != want["KS_ACC_HI"]:
                bad.append(f"{tile} acc_hi: {m_ and hex(int(m_.group(1), 16))}")
    assert not bad, "KSC placement moved, rebuild with: " + "; ".join(sorted(set(bad))[:6])
    print(f"[design8] KSC placement checked on {ROWS_G * COLS} GEMM cores: " + ", ".join(f"{k} {v:#x}" for k, v in want.items()))


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
    # One design build at a time per build dir (held until exit): asm_object / ew_object write fixed names (build/mm8asm.o,
    # ew8cpp.o, the generated <k>.s / <k>.h that mm8.cc includes as "build/<k>.h", asmfit's <k>_fit.s) that aiecc links
    # later in this process; two builds in parallel raced on them ("the last body bundle is not 16 B aligned", round 13).
    lock = open(HERE / "build" / ".design8.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
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
    if "-DKSC" in o.kdefs:  # round 18: the k-step records were built from these addresses: fail the build if the placement moved
        ksc_check(o.out, o.kdefs)
    Path(o.out + ".json").write_text(json.dumps(dict(kw, m=M, k=KT, n=NT, rho=RHO, rows_g=ROWS_G, k_slot=K_SLOT, rot_v=os.environ.get("ROT_V", "g2f"),
                                                     a_half=os.environ.get("A_HALF") == "1", a_rt=os.environ.get("A_RT") == "1", p23=os.environ.get("P23") == "1", p60g=os.environ.get("P60G") == "1",
                                                     p43=os.environ.get("P43") == "1", xrot=int(os.environ.get("XROT", "0") or 0), al1=os.environ.get("AL1") == "1",
                                                     il=os.environ.get("IL") == "1", dapre=os.environ.get("DAPRE") == "1", a3=os.environ.get("A3") == "1",
                                                     nres=[x for x in os.environ.get("NRES", "").split(",") if x], sres=os.environ.get("SRES") == "1",
                                                     msplit=[x for x in os.environ.get("MSPLIT", "").split(",") if x], yxr=os.environ.get("YXR") == "1", yxh=os.environ.get("YXH") == "1", zdrop=os.environ.get("ZDROP") == "1")))


if __name__ == "__main__":
    main()
