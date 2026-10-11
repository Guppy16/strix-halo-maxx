"""Round 19 DDRCAP (research rev 15 rank 0'): the NPU's aggregate DDR rate. Per column c: NIN input streams shim c MM2S 0 / 1 ->
core (c, 2) S2MM 0 / 1 (2 KB objects, depth 2, the core only acquires / releases), and optionally one drain core -> shim S2MM.
Each input moves MB_IN MB per column, the drain MB_OUT. Host time of one dispatch (median of N) minus an empty-ish dispatch.
usage (npu-iron, NPU): python3 r19/ddrcap.py build --out build/r19/ddrcap_<cfg> --nin 2 --drain 1 ; ... run --out ... --nin 2 --drain 1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

OBJ_W = 512  # 2 KB objects
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "run"])
ap.add_argument("--out", required=True)
ap.add_argument("--nin", type=int, default=2)
ap.add_argument("--drain", type=int, default=1)
ap.add_argument("--nobj", type=int, default=2048)  # objects per input stream per column (2048 x 2 KB = 4 MB)
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--iters", type=int, default=20)
o = ap.parse_args()
COLS, NOBJ = o.cols, o.nobj
IN_W = COLS * NOBJ * OBJ_W  # words per input buffer
OUT_W = COLS * (NOBJ // 2) * OBJ_W


def tap_for(c, n):
    from aie.helpers.taplib import TensorAccessPattern
    a = n // 512
    return TensorAccessPattern((IN_W if n == NOBJ else OUT_W,), c * n * OBJ_W, [1, a, 512, OBJ_W], [0, 512 * OBJ_W, OBJ_W, 1])


if o.cmd == "build":
    import aie.iron as iron
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2, Tile

    O_ty = np.ndarray[(OBJ_W,), np.dtype[np.uint32]]
    I_ty = np.ndarray[(IN_W,), np.dtype[np.uint32]]
    Q_ty = np.ndarray[(OUT_W,), np.dtype[np.uint32]]

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def ddr(A: In, B: In, C: Out, *, key: CompileTime[str] = ""):
        ins = [[ObjectFifo(O_ty, name=f"I{c}_{j}", depth=2) for j in range(o.nin)] for c in range(COLS)]
        outs = [ObjectFifo(O_ty, name=f"O{c}", depth=2) for c in range(COLS)] if o.drain else []
        workers = []
        for c in range(COLS):
            def body(*eps, nin=o.nin, drain=o.drain):
                for _ in range_(NOBJ // 2):
                    for _ in range(2):
                        for e in eps[:nin]:
                            e.acquire(1)
                            e.release(1)
                    if drain:
                        eps[nin].acquire(1)
                        eps[nin].release(1)
            args = [f.cons() for f in ins[c]] + ([outs[c].prod()] if o.drain else [])
            workers.append(Worker(body, args, tile=Tile(c, 2), while_true=True))

        def seq(a, b, cc, *eps):
            tg = TaskGroup()
            k = 0
            for c in range(COLS):
                for j in range(o.nin):
                    eps[k].fill(a if j == 0 else b, tap=tap_for(c, NOBJ), group=tg, wait=not o.drain)
                    k += 1
            for c in range(COLS):
                if o.drain:
                    eps[k].drain(cc, tap=tap_for(c, NOBJ // 2), group=tg, wait=True)
                    k += 1
            tg.finish()

        eps = [f.prod() for c in range(COLS) for f in ins[c]] + [f.cons() for f in outs]
        return Program(iron.get_current_device(), Runtime(seq, [I_ty, I_ty, Q_ty, *eps]), workers=workers).resolve_program()

    iron.set_current_device(NPU2())
    out = str(Path(o.out).resolve())
    ddr.specialize(key=out).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
    print("built", out)
else:
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bos = [pyxrt.bo(dev, n * 4, pyxrt.bo.host_only, kern.group_id(3 + i)) for i, n in enumerate((IN_W, IN_W, OUT_W))]
    for b in bos[:2]:
        b.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    ts = []
    for it in range(o.iters + 3):
        t0 = time.perf_counter()
        r = kern(3, bo_i, len(instr), *bos)
        st = r.wait(5000 * 1000)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        if it >= 3:
            ts.append(time.perf_counter() - t0)
    t = float(np.median(ts))
    rd = COLS * o.nin * NOBJ * OBJ_W * 4
    wr = COLS * (NOBJ // 2) * OBJ_W * 4 if o.drain else 0
    print(f"DDRCAP nin {o.nin} drain {o.drain} cols {COLS}: {t * 1e3:.3f} ms median (min {min(ts) * 1e3:.3f}); read {rd / 2**20:.0f} MiB "
          f"-> {rd / t / 1e9:.1f} GB/s read, {(rd + wr) / t / 1e9:.1f} GB/s read+write")
