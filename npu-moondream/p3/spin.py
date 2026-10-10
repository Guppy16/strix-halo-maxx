"""Clock / kernel-cycle probe: one core runs mm_acc `loops` times on resident L1
buffers (no input DMA), then emits 16 words. Host time = launch + loops x
(kernel cycles) / f. Two loop counts give f and the cycles per call.
"""

import argparse
import json
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import Buffer, CompileTime, ExternalFunction, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
INC = "/work/mlir-aie/aie_kernels"


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def spin(C: Out, *, loops: CompileTime[int] = 1000, kdefs: CompileTime[str] = "", m: CompileTime[int] = 96,
         k: CompileTime[int] = 64, n: CompileTime[int] = 96, rho: CompileTime[int] = 6):
    m_a = m // rho
    wa, wb = m_a * k * 9 // 32, k * n * 9 // 32
    A_l1 = np.ndarray[(wa,), np.dtype[np.uint32]]
    B_l1 = np.ndarray[(wb,), np.dtype[np.uint32]]
    C_l1 = np.ndarray[(m * n,), np.dtype[bfloat16]]
    O = np.ndarray[(16,), np.dtype[np.uint32]]
    flags = [f"-I{INC}", f"-DDIM_M={m}", f"-DDIM_N={n}", f"-DDIM_K={k}", f"-DRHO={rho}"] + kdefs.split()
    mm = ExternalFunction("mm_acc", object_file_name="mm_f32acc.o", source_file=str(HERE / "mm_f32acc.cc"),
                          arg_types=[A_l1, B_l1, C_l1], compile_flags=flags)
    zero = mm.object_file.bind("acc_zero", [C_l1])
    t = Tile(0, 2)
    a = Buffer(tile=t, type=A_l1, name="a")
    b = Buffer(tile=t, type=B_l1, name="b")
    c = Buffer(tile=t, type=C_l1, name="c")
    out = ObjectFifo(O, name="o", depth=2)

    def body(a, b, c, o, mm, zero):
        zero(c)
        for _ in range_(loops):
            mm(a, b, c)
        ob = o.acquire(1)
        o.release(1)

    w = Worker(body, [a, b, c, out.prod(), mm, zero], tile=t)

    def seq(cc, cons):
        cons.drain(cc, wait=True)

    rt = Runtime(seq, [np.ndarray[(16,), np.dtype[np.uint32]], out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loops", type=int, default=1000)
    p.add_argument("--kdefs", default="-DNO_PREFETCH -DBLK12")
    p.add_argument("--out", required=True)
    o = p.parse_args()
    iron.set_current_device(NPU2())
    spin.specialize(loops=o.loops, kdefs=o.kdefs).compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")
    Path(o.out + ".json").write_text(json.dumps(dict(loops=o.loops, kdefs=o.kdefs)))
