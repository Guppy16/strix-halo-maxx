"""LayerNorm of a (M x D) bf16 matrix on 32 cores (moondream vision tower: D 1152, M 729 -> 768).

Column c's shim streams rows [c*96, c*96+96) into its memtile, which splits them
over the column's 4 cores (24 rows each, in 6 objects of 4 rows); the outputs
join back the same way. gamma/beta arrive once per core as one bf16 block.
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


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def layernorm(X: In, P: In, Y: Out, *, D: CompileTime[int] = 1152, rows_per_core: CompileTime[int] = 24,
              rpo: CompileTime[int] = 4, eps: CompileTime[float] = 1e-5, reps: CompileTime[int] = 1):
    assert rows_per_core % rpo == 0
    n_obj = rows_per_core // rpo
    M = COLS * ROWS * rows_per_core
    Obj = np.ndarray[(rpo * D,), np.dtype[bfloat16]]
    L2 = np.ndarray[(ROWS * rpo * D,), np.dtype[bfloat16]]
    Par = np.ndarray[(2 * D,), np.dtype[bfloat16]]
    flags = [f"-I{INC}", f"-DLN_D={D}", f"-DLN_ROWS={rpo}", f"-DLN_EPS={eps}f"]
    ln = ExternalFunction("ln_rows", object_file_name="ln.o", source_file=str(HERE / "ln.cc"),
                          arg_types=[Obj, Obj], compile_flags=flags)
    params = ln.object_file.bind("ln_params", [Par])
    shims = [Tile(c, 0) for c in range(COLS)]
    mems = [Tile(c, 1) for c in range(COLS)]

    xin, xl1, yout, yl1, pin, yc = [], {}, [], {}, [], []
    for c in range(COLS):
        f = ObjectFifo(L2, name=f"X{c}", depth=2)
        f.prod(tile=shims[c])
        xin.append(f)
        subs = f.cons().split([rpo * D * r for r in range(ROWS)], tile=mems[c], obj_types=[Obj] * ROWS,
                              names=[f"X{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        for r in range(ROWS):
            xl1[(r, c)] = subs[r]
        g = ObjectFifo(L2, name=f"Y{c}", depth=2)
        yc.append(g.cons(tile=shims[c]))
        yout.append(g)
        subs = g.prod().join([rpo * D * r for r in range(ROWS)], tile=mems[c], obj_types=[Obj] * ROWS,
                             names=[f"Y{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        for r in range(ROWS):
            yl1[(r, c)] = subs[r]
        # params: one block per column, broadcast to its 4 cores through the memtile
        pf = ObjectFifo(Par, name=f"P{c}", depth=1)
        pf.prod(tile=shims[c])
        pin.append(pf)
        pl1 = pf.cons().forward(tile=mems[c], obj_type=Par, name=f"P{c}L1", depth=1)
        for r in range(ROWS):
            xl1[(r, c, "p")] = pl1

    def core_fn(pin, xin, yout, ln, params):
        pb = pin.acquire(1)
        params(pb)
        pin.release(1)
        for _ in range_(reps) if reps > 1 else range(1):
            for _ in range_(n_obj):
                xo = xin.acquire(1)
                yo = yout.acquire(1)
                ln(xo, yo)
                xin.release(1)
                yout.release(1)

    workers = []
    for r in range(ROWS):
        for c in range(COLS):
            workers.append(Worker(core_fn, [xl1[(r, c, "p")].cons(), xl1[(r, c)].cons(), yl1[(r, c)].prod(), ln, params],
                                  tile=Tile(c, 2 + r), stack_size=0x800))

    X_ty = np.ndarray[(M * D,), np.dtype[bfloat16]]
    P_ty = np.ndarray[(2 * D,), np.dtype[bfloat16]]

    def sequence(x, p, y, X_prods, P_prods, Y_conses):
        tg = TaskGroup()
        for c in range(COLS):
            P_prods[c].fill(p, tap=TensorAccessPattern((2 * D,), 0, [1, 1, 1, 2 * D], [0, 0, 0, 1]), group=tg, wait=False)
        for rep in range(reps):
            for c in range(COLS):
                # column c, L2 object i = rows (c*ROWS*rpc) + [r*rpc + i*rpo ...] for r in 0..3: gather per object
                base = c * ROWS * rows_per_core * D
                X_prods[c].fill(x, tap=TensorAccessPattern((M * D,), base, [n_obj, ROWS, rpo, D],
                                                           [rpo * D, rows_per_core * D, D, 1]), group=tg, wait=False)
                Y_conses[c].drain(y, tap=TensorAccessPattern((M * D,), base, [n_obj, ROWS, rpo, D],
                                                             [rpo * D, rows_per_core * D, D, 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [X_ty, P_ty, X_ty, [f.prod() for f in xin], [f.prod() for f in pin], yc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    iron.set_current_device(NPU2())
    layernorm.specialize(reps=o.reps).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(D=1152, rows_per_core=24, rpo=4, reps=o.reps)))
