"""Transfer-cost probe: a trivial 8-core program (one core per column copies its column's share of an input
buffer through L1 and emits one 256 B object per column), so the host side of a step can be timed:
bo write / sync / dispatch / read-back, at the sizes a tracker step moves (a 256x256x3 frame as uint8 or bf16,
a full camera frame, 48 points back).

    python3 xfer.py --n-obj 1 --out build/xfer_1      # the smallest dispatch: 4 KB in, 2 KB out
    python3 xfer.py --n-obj 6 --out build/xfer_6      # 8 x 6 x 4 KB = 192 KB (256x256x3 uint8)
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

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
COLS = 8
OBJ = 1024  # words per input object (4 KB)
OUT = 64  # words per output object (256 B)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def xfer(X: In, Y: Out, *, n_obj: CompileTime[int] = 1):
    Obj = np.ndarray[(OBJ,), np.dtype[np.uint32]]
    Res = np.ndarray[(OUT,), np.dtype[np.uint32]]
    k = ExternalFunction("xfer_sum", object_file_name="xfer.o", source_file=str(HERE / "xfer.cc"),
                         arg_types=[Obj, Res], compile_flags=[f"-I{INC}", f"-DOBJ={OBJ}", f"-DOUT={OUT}"])
    fin = ExternalFunction("xfer_fin", object_file_name="xfer_fin.o", source_file=str(HERE / "xfer.cc"),
                           arg_types=[Res], compile_flags=[f"-I{INC}", f"-DOBJ={OBJ}", f"-DOUT={OUT}", "-DFIN"])
    xin, yout = [], []
    workers = []
    for c in range(COLS):
        f = ObjectFifo(Obj, name=f"X{c}", depth=2)
        f.prod(tile=Tile(c, 0))
        xin.append(f)
        g = ObjectFifo(Res, name=f"Y{c}", depth=2)
        yout.append(g.cons(tile=Tile(c, 0)))

        def core_fn(xi, yo, k, fin):
            ro = yo.acquire(1)
            fin(ro)
            for _ in range_(n_obj) if n_obj > 1 else range(1):
                xo = xi.acquire(1)
                k(xo, ro)
                xi.release(1)
            yo.release(1)

        workers.append(Worker(core_fn, [f.cons(), g.prod(), k, fin], tile=Tile(c, 2), stack_size=0x400))

    X_ty = np.ndarray[(COLS * n_obj * OBJ,), np.dtype[np.uint32]]
    Y_ty = np.ndarray[(COLS * OUT,), np.dtype[np.uint32]]

    def sequence(x, y, X_prods, Y_conses):
        tg = TaskGroup()
        for c in range(COLS):
            X_prods[c].fill(x, tap=TensorAccessPattern((COLS * n_obj * OBJ,), c * n_obj * OBJ, [1, 1, n_obj, OBJ], [0, 0, OBJ, 1]),
                            group=tg, wait=False)
            Y_conses[c].drain(y, tap=TensorAccessPattern((COLS * OUT,), c * OUT, [1, 1, 1, OUT], [0, 0, 0, 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [X_ty, Y_ty, [f.prod() for f in xin], yout])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obj", type=int, default=1)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    iron.set_current_device(NPU2())
    xfer.specialize(n_obj=o.n_obj).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(n_obj=o.n_obj, OBJ=OBJ, OUT=OUT, COLS=COLS)))
