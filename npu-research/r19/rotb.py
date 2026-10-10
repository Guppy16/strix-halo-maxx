"""rotb.py - one-core bench of ROT (npu-research pass 14, E1) after npu-tapnext r9/ksb.py: r19/rotb.cc built twice, `ks` (the
deployed mm_kstep -> mm_ks, B in pair4 order) and `rot` (mm_rot from r19/gen_rot.py, B repacked by mm_rot_border). Hashes of
all 36 accumulator pieces must match (bit-identical); cycles of the last k-step from the core timer. Objects and xclbins go to
r19/build/ (main's npu-tapnext/build/mm_ks.s / .h are only read).

build (npu-iron, CPU only):  python3 r19/rotb.py build [--v g2f]
run   (npu-iron, the NPU):   python3 r19/rotb.py run [--v g2f] [--calls 1,4]
place (no device):           python3 r19/rotb.py place [--v g2f]   # the bench's own L1 addresses -> gen_rot.py check --place
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # npu-research/r19
ROOT = HERE.parents[1] / "npu-tapnext"
BD = HERE / "build"
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
NW = 80
FLAGS = ["-DDIM_M=144", "-DDIM_N=64", "-DDIM_K=64", "-DRHO=9", "-DBLKU=1", "-DATTN_KEYS=1072", "-DATTN_SCALE=0.125f", "-O2",
         "-DMM_ASM", "-DMM_KSTEP", "-DMM_PIECES", "-DMM_KS", "-ffunction-sections", "-fdata-sections"]


def objects(v, rot):
    """the merged bench object of one build (absolute path: IRON resolves Kernel objects inside the .prj)"""
    b = f"{PEANO}/bin"
    BD.mkdir(exist_ok=True)
    shutil.copy(ROOT / "build" / "mm_ks.s", BD / "mm_ks.s")  # the deployed generator output, read-only use
    shutil.copy(HERE / f"mm_rot_{v}.s", BD / "mm_rot.s")
    shutil.copy(HERE / f"mm_rot_{v}.h", BD / "mm_rot.h")
    out = BD / f"rotb_{v}_{'rot' if rot else 'ks'}"
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", f"-I{INC}",
            f"-I{ROOT}", f"-I{HERE}"]
    steps = [[f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(BD / "mm_ks.s"), "-o", str(BD / "mm_ks.o")],
             [f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(BD / "mm_rot.s"), "-o", str(BD / "mm_rot.o")],
             [f"{b}/clang++"] + base + FLAGS + [f"-DROTB_ROT={int(rot)}", "-c", str(HERE / "rotb.cc"), "-o", f"{out}_cpp.o"],
             [f"{b}/ld.lld", "-r", "--gc-sections", "-u", "rotb", f"{out}_cpp.o", str(BD / "mm_ks.o"), str(BD / "mm_rot.o"),
              "-o", f"{out}.o"]]
    for cmd in steps:
        subprocess.run(cmd, check=True)
    return f"{out}.o"


def build(o):
    import aie.iron as iron
    from aie.iron import CompileTime, Kernel, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.device import NPU2, Tile

    for rot in (False, True):
        obj = objects(o.v, rot)
        out = obj[:-2]

        @iron.jit(aiecc_flags=["--dynamic-objFifos"])
        def bench(X: In, Y: Out, *, key: CompileTime[str] = ""):
            from aie.iron import Lock as L
            T = np.ndarray[(NW,), np.dtype[np.int32]]
            fn = Kernel("rotb", obj, [T, T], stack_size_override=0x800)
            xf = ObjectFifo(T, name="X", depth=1)
            yf = ObjectFifo(T, name="Y", depth=1)
            t = Tile(0, 2)
            a_free = L(tile=t, init=0, name="aF", lock_id=4)
            a_full = L(tile=t, init=0, name="aU", lock_id=5)

            def core(xc, yp, fn, lf, lu):
                x = xc.acquire(1)
                y = yp.acquire(1)
                fn(x, y)
                xc.release(1)
                yp.release(1)

            w = Worker(core, [xf.cons(), yf.prod(), fn, a_free, a_full], tile=t, stack_size=0xA00, data_size=52 * 1024)  # ksb_br sits in .aie.bank3 outside it

            def seq(x, y, xp, yc):
                from aie.dialects.aiex import npu_maskwrite32
                npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)  # core timer: processor bus on
                tg = TaskGroup()
                xp.fill(x, group=tg, wait=False)
                yc.drain(y, group=tg, wait=True)
                tg.finish()

            rt = Runtime(seq, [T, T, xf.prod(), yf.cons()])
            rt.add_lock(a_free)
            rt.add_lock(a_full)
            return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

        iron.set_current_device(NPU2())
        import hashlib

        h = hashlib.md5(Path(obj).read_bytes()).hexdigest()[:8]  # IRON's jit cache ignores sources: key on the object
        bench.specialize(key=f"{out}_{h}").compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
        print("built", out)


def place(o):
    """the rot build's L1 addresses (llvm-nm of the linked core elf) for gen_rot.py check --place"""
    elf = BD / f"rotb_{o.v}_rot.prj" / "elfs_main_core_0_2" / "elfs_main_core_0_2.elf"
    nm = subprocess.run([f"{PEANO}/bin/llvm-nm", str(elf)], capture_output=True, text=True, check=True).stdout
    a = {ln.split()[-1]: int(ln.split()[0], 16) for ln in nm.splitlines() if len(ln.split()) == 3}

    def sym(part):
        return next(v for k, v in a.items() if part in k)

    slot0 = sym("ksb_a")
    print(f"--place c_obj={sym('ksb_c'):#x},acc_hi={sym('acc_hi'):#x},slots={slot0:#x}:{slot0 + 1152:#x}:{slot0 + 2304:#x},"
          f"rec={sym('g_rot_tab'):#x},b={sym('ksb_br'):#x}")


def call(out, n, timeout_ms):
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_x = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, kern.group_id(3))
    bo_y = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, kern.group_id(4))
    x = np.zeros(NW, np.int32)
    x[0] = n
    bo_x.write(x.tobytes(), 0)
    bo_x.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    st = kern(3, bo_i, len(instr), bo_x, bo_y).wait(timeout_ms * 1000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bo_y.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo_y.read(NW * 4, 0), np.int32).copy()


def run(o):
    res = {}
    for var in ("ks", "rot"):
        for n in (int(c) for c in o.calls.split(",")):
            y = call(str(BD / f"rotb_{o.v}_{var}"), n, o.timeout_ms)
            res[var, n] = y
            print(f"{var:4s} {n} k-step(s): last call {int(y[0]) & 0xFFFFFFFF} cycles, hash[0:4] "
                  f"{[hex(int(h) & 0xFFFFFFFF) for h in y[2:6]]}", flush=True)
    for n in (int(c) for c in o.calls.split(",")):
        a, b = res["ks", n][2:38], res["rot", n][2:38]
        bad = np.nonzero(a != b)[0]
        print(f"{n} k-step(s): {36 - len(bad)}/36 pieces identical" + (f"; differing (rb, pair): {[(i // 2, i % 2) for i in bad]}" if len(bad) else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "place"])
    ap.add_argument("--v", default="g2f", help="gen_rot variant: r19/mm_rot_<v>.s / .h")
    ap.add_argument("--calls", default="1,4")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    o = ap.parse_args()
    dict(build=build, run=run, place=place)[o.cmd](o)
