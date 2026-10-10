"""32-core bfp16 GEMM with an fp32 accumulator over all of K (AIE2P / Strix).

Mapping: core column c owns M-tile c (8 x m rows = 768 >= 729), core row r owns
a contiguous run of N-tiles. A (activations) streams per column and is
broadcast to the 4 cores of that column; B (weights) streams per row and is
broadcast to the 8 cores of that row, so each weight byte leaves DDR once.

A schedule is a sequence of GEMMs ("ops") run back to back from one runtime
sequence: op = (K, nt) with K padded to a multiple of k and nt = N-tiles per
core row (N = 4 * nt * n). Every op's A, B and C live at offsets in three
flat buffers (A_all, W_all, C_all), all pre-packed on the host in exactly
the order the cores consume them (see pack.py), so every L3->L2->L1 transfer
is linear except C, which the memtile un-blocks into row-major.
"""

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
INC = "/work/mlir-aie/aie_kernels"
COLS, ROWS = 8, 4


def words_a(m, k):  # one A-stripe sub-tile (m x k) bfp16ebs8 in 32-bit words
    return m * k * 9 // 8 // 4


def plan(schedule, m, k, n, rho):
    """Offsets (in words / bf16 elements) of each op's A, W and C."""
    ops, a_off, w_off, c_off = [], 0, 0, 0
    for K, nt in schedule:
        la = words_a(m, K)  # one M-tile of A, all K
        lb = words_a(n, K)  # one N-tile of W, all K
        N = ROWS * nt * n
        ops.append(dict(K=K, nt=nt, N=N, a=a_off, w=w_off, c=c_off, la=la, lb=lb))
        a_off += COLS * la
        w_off += ROWS * nt * lb
        c_off += COLS * m * N
    return ops, a_off, w_off, c_off


def tap1d(total, off, length, rep=1):
    """Linear transfer of `length` elements at `off`, repeated `rep` times."""
    a0 = next(d for d in range(min(length, 1020), 0, -1) if length % d == 0)
    a1 = length // a0
    assert a1 <= 1023, (length, a0, a1)
    return TensorAccessPattern((total,), off, [rep, 1, a1, a0], [0, 0, a0, 1])


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemm_seq(
    A: In,
    W: In,
    C: Out,
    *,
    sched: CompileTime[str] = "[[1152, 9]]",
    m: CompileTime[int] = 96,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 96,
    rho: CompileTime[int] = 3,
    reps: CompileTime[int] = 1,
    a_depth: CompileTime[int] = 2,
    b_depth: CompileTime[int] = 2,
    l2_depth: CompileTime[int] = 2,
    kdefs: CompileTime[str] = "",
    c_linear: CompileTime[int] = 0,
    mode: CompileTime[int] = 0,  # 0 GEMM, 1 A-stream only, 2 B-stream only (DMA probes)
):
    schedule = [tuple(x) for x in json.loads(sched)]
    ops, a_tot, w_tot, c_tot = plan(schedule, m, k, n, rho)
    m_a = m // rho
    wa, wb = words_a(m_a, k), words_a(k, n)

    A_l1 = np.ndarray[(wa,), np.dtype[np.uint32]]
    B_l1 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    # L2 objects: one k-step (rho A sub-tiles / one B tile).
    A_l2 = np.ndarray[(rho * wa,), np.dtype[np.uint32]]
    B_l2 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l2 = np.ndarray[(ROWS * m * n,), np.dtype[bfloat16]]

    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}"] + kdefs.split()
    mm = ExternalFunction(
        "mm_acc",
        object_file_name="mm_f32acc.o",
        source_file=str(HERE / "mm_f32acc.cc"),
        arg_types=[A_l1, B_l1, C_l1],
        compile_flags=flags,
    )
    zero = mm.object_file.bind("acc_zero", [C_l1])
    store = mm.object_file.bind("acc_store", [C_l1])

    a_l3l2, a_l2l1 = [], []
    for c in range(COLS):
        f = ObjectFifo(A_l2, name=f"A{c}", depth=l2_depth)
        a_l3l2.append(f)
        a_l2l1.append(f.cons().forward(obj_type=A_l1, name=f"A{c}L1", depth=a_depth))
    b_l3l2, b_l2l1 = [], []
    for r in range(ROWS):
        f = ObjectFifo(B_l2, name=f"B{r}", depth=l2_depth)
        b_l3l2.append(f)
        b_l2l1.append(f.cons().forward(obj_type=B_l1, name=f"B{r}L1", depth=b_depth))

    # C: un-block [r][m/8][n/8][8][8] -> [r*m rows][n] on the memtile's way out.
    c_out = TensorAccessPattern(
        (m * n,), 0, [m // 8, 8, n // 8, 8], [8 * n, 8, 64, 1]
    )
    c_l2l3, c_l1l2 = [], [[None] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        f = ObjectFifo(C_l2, name=f"C{c}", depth=2, to_stream=None if c_linear else c_out)
        c_l2l3.append(f)
        subs = f.prod().join(
            [m * n * r for r in range(ROWS)],
            obj_types=[C_l1] * ROWS,
            names=[f"C{c}_{r}" for r in range(ROWS)],
            depths=[1] * ROWS,
        )
        for r in range(ROWS):
            c_l1l2[r][c] = subs[r]

    def core_fn(in_a, in_b, out_c, mm, zero, store):
        def one_op(K, nt):
            for _ in range_(nt) if nt > 1 else range(1):
                cbuf = out_c.acquire(1)
                zero(cbuf)
                for _ in range_(K // k):
                    if mode != 1:
                        b = in_b.acquire(1)
                    for _ in range(rho):
                        if mode != 2:
                            a = in_a.acquire(1)
                        if mode == 0:
                            mm(a, b, cbuf)
                        if mode != 2:
                            in_a.release(1)
                    if mode != 1:
                        in_b.release(1)
                store(cbuf)
                out_c.release(1)

        for _ in range_(reps) if reps > 1 else range(1):
            for K, nt in schedule:
                one_op(K, nt)

    grid = Worker.grid(
        ROWS,
        COLS,
        lambda r, c: Worker(
            core_fn,
            [a_l2l1[c].cons(), b_l2l1[r].cons(), c_l1l2[r][c].prod(), mm, zero, store],
            stack_size=0x800,
        ),
    )
    workers = [w for row in grid for w in row]

    A_ty = np.ndarray[(a_tot,), np.dtype[np.uint32]]
    W_ty = np.ndarray[(w_tot * reps,), np.dtype[np.uint32]]
    C_ty = np.ndarray[(c_tot,), np.dtype[bfloat16]]

    def sequence(a, w, cc, A_prods, B_prods, C_conses):
        for rep in range(reps):
            for op in ops:
                tg = TaskGroup()
                for c in range(COLS if mode != 2 else 0):
                    A_prods[c].fill(
                        a, tap=tap1d(a_tot, op["a"] + c * op["la"], op["la"], op["nt"]),
                        group=tg, wait=False,
                    )
                for r in range(ROWS if mode != 1 else 0):
                    lb = op["nt"] * op["lb"]
                    B_prods[r].fill(
                        w, tap=tap1d(w_tot * reps, rep * w_tot + op["w"] + r * lb, lb),
                        group=tg, wait=False,
                    )
                N, nt = op["N"], op["nt"]
                for c in range(COLS):
                    # step i of column c holds N-tiles r*nt + i for r = 0..3
                    tap = TensorAccessPattern(
                        (c_tot,), op["c"] + c * m * N, [nt, ROWS, m, n],
                        [n, nt * n, N, 1],
                    )
                    C_conses[c].drain(cc, tap=tap, group=tg, wait=True)
                tg.finish()

    rt = Runtime(
        sequence,
        [
            A_ty,
            W_ty,
            C_ty,
            [f.prod() for f in a_l3l2],
            [f.prod() for f in b_l3l2],
            [f.cons() for f in c_l2l3],
        ],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sched", default="[[1152, 9]]")
    p.add_argument("-m", type=int, default=96)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=96)
    p.add_argument("--rho", type=int, default=3)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--a-depth", type=int, default=2)
    p.add_argument("--b-depth", type=int, default=2)
    p.add_argument("--l2-depth", type=int, default=2)
    p.add_argument("--kdefs", default="")
    p.add_argument("--c-linear", type=int, default=0)
    p.add_argument("--mode", type=int, default=0)
    p.add_argument("--out", required=True)
    o = p.parse_args()
    iron.set_current_device(NPU2())
    kw = dict(
        sched=o.sched, m=o.m, k=o.k, n=o.n, rho=o.rho, reps=o.reps,
        a_depth=o.a_depth, b_depth=o.b_depth, l2_depth=o.l2_depth, kdefs=o.kdefs, c_linear=o.c_linear, mode=o.mode,
    )
    gemm_seq.specialize(**kw).compile(
        xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin"
    )
    Path(o.out + ".json").write_text(json.dumps(kw))


if __name__ == "__main__":
    main()
