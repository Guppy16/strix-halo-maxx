"""Round 11: one-core bench for hand-asm elementwise kernels (asm probe bench).

The .s given to `build` defines up to 8 functions probe0..probe7 with the signature
    void probeK(const bfloat16 *in /*p0*/, bfloat16 *out /*p1*/, int n /*r0*/, const float *ab /*p2*/, const float *cd /*p3*/)
(missing ones become stubs). r11/apb.cc times each over 144 vectors of 32 lanes in calls of n and compares every output
with the deployed C++ GELU (mm8.cc gelu_lut_n).

    python3 r11/apb.py build --asm build/r11/x.s --out build/r11/apbX        (npu-iron, no device)
    python3 r11/apb.py run --out build/r11/apbX --probes 0,1 --n 8,144      (npu-iron, NPU)
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
FLAGS = ["-DDIM_M=144", "-DDIM_N=64", "-DDIM_K=64", "-DRHO=9", "-DBLKU=1", "-DATTN_KEYS=1072", "-DATTN_SCALE=0.125f", "-O2"]
NW = 1040
GC = False  # round 13: set by a bench (attkb) whose C++ would not fit program memory otherwise
STUB = """	.section	.text.{f},"ax",@progbits
	.globl	{f}
	.p2align	4
	.type	{f},@function
{f}:
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end_{f}:
	.size	{f}, .Lfunc_end_{f}-{f}
"""


def defined(asm):
    return set(re.findall(r"^(probe\d):", asm, re.M))


def obj(asm_path, out, cc=None):
    b = f"{PEANO}/bin"
    asm = open(asm_path).read()
    have = defined(asm)
    full = asm + "".join(STUB.format(f=f"probe{k}") for k in range(8) if f"probe{k}" not in have)
    s = Path(out + "_probes.s")
    s.write_text(full)
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "--target=aie2p-none-unknown-elf",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", f"-I{INC}",
            f"-I{ROOT}"]
    gc = ["-ffunction-sections", "-fdata-sections"] if GC else []  # round 13: drop the C++ entry points the bench never calls
    for cmd in ([f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", str(s), "-o", out + "_probes.o"],
                [f"{b}/clang++"] + base + FLAGS + gc + ["-c", str(cc or HERE / "apb.cc"), "-o", out + "_cpp.o"],
                [f"{b}/ld.lld", "-r"] + (["--gc-sections", "-u", "apb"] if GC else []) + [out + "_cpp.o", out + "_probes.o", "-o", out + ".o"]):
        subprocess.run(cmd, check=True)
    return out + ".o"


def build(o):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Kernel, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.device import NPU2, Tile

    out = str(Path(o.out).resolve())
    global NW
    NW = getattr(o, "nw", NW)  # round 13: smaller X / Y objects when L1 is tight (the run side sets the same)
    T = np.ndarray[(NW,), np.dtype[np.int32]]

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def bench(X: In, Y: Out, *, key: CompileTime[str] = ""):
        fn = Kernel("apb", obj(o.asm, out, getattr(o, "cc", None)), [T, T], stack_size_override=getattr(o, "stack_override", 0x600))
        xf, yf = ObjectFifo(T, name="X", depth=1), ObjectFifo(T, name="Y", depth=1)

        def core(xc, yp, fn):
            x, y = xc.acquire(1), yp.acquire(1)
            fn(x, y)
            xc.release(1)
            yp.release(1)

        w = Worker(core, [xf.cons(), yf.prod(), fn], tile=Tile(0, 2), stack_size=getattr(o, "stack", 0x800),
                   data_size=getattr(o, "data_size", 16 * 1024))  # round 13: attkb needs ~50 KB of .bss (mm8's acc_hi + the C object)

        def seq(x, y, xp, yc):
            from aie.dialects.aiex import npu_maskwrite32
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            tg = TaskGroup()
            xp.fill(x, group=tg, wait=False)
            yc.drain(y, group=tg, wait=True)
            tg.finish()

        return Program(iron.get_current_device(), Runtime(seq, [T, T, xf.prod(), yf.cons()]), workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    cc = getattr(o, "cc", None) or str(HERE / "apb.cc")  # round 12: the C++ side's mtime too (a cc-only change hit the cache)
    bench.specialize(key=out + str(Path(o.asm).stat().st_mtime) + str(Path(cc).stat().st_mtime)).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
    print("built", out)


class Dev:
    def __init__(self, out):
        import sys
        sys.path.insert(0, "/opt/xilinx/xrt/python")
        import pyxrt
        self.x, S = pyxrt, pyxrt.xclBOSyncDirection
        self.S = S
        dev = pyxrt.device(0)
        xb = pyxrt.xclbin(out + ".xclbin")
        dev.register_xclbin(xb)
        self.ctx = pyxrt.hw_context(dev, xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        instr = np.fromfile(out + ".bin", dtype=np.uint32)
        self.n_instr = len(instr)
        self.bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, self.kern.group_id(1))
        self.bo_i.write(instr.tobytes(), 0)
        self.bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_x = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, self.kern.group_id(3))
        self.bo_y = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, self.kern.group_id(4))

    def call(self, args, timeout_ms=3000):
        x = np.zeros(NW, np.int32)
        x[:len(args)] = args
        self.bo_x.write(x.tobytes(), 0)
        self.bo_x.sync(self.S.XCL_BO_SYNC_BO_TO_DEVICE)
        st = self.kern(3, self.bo_i, self.n_instr, self.bo_x, self.bo_y).wait(timeout_ms * 1000)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        self.bo_y.sync(self.S.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(self.bo_y.read(NW * 4, 0), np.int32).copy()


def run(o):
    d = Dev(o.out)
    for k in (int(p) for p in o.probes.split(",")):
        for lay in (int(c) for c in o.layout.split(",")):
            for n in ((int(c) for c in o.n.split(",")) if lay == 0 else (72,)):
                for inplace in (int(i) for i in o.inplace.split(",")):
                    u = dict((int(a), int(b)) for a, b in (kv.split("=") for kv in o.u.split(","))).get(k, 0) if o.u else 0
                    bases = range(0, 65536, 4608) if o.sweep else (0,)
                    bad, cycs, msg = 0, [], ""
                    for base in bases:  # --sweep: every bf16 bit pattern (15 runs), else the default fill
                        y = d.call([k, n, inplace, o.erf, lay, u, o.flat, 1 if o.sweep else 0, base], o.timeout_ms)
                        cycs.append((int(y[0]) & 0xFFFFFFFF) / (144 if lay == 0 else 72))
                        if y[1] and not msg:
                            msg = f"first at base {base} + {y[2]}: got {y[3] & 0xFFFF:#06x} want {y[4] & 0xFFFF:#06x}"
                        bad += int(y[1])
                    cyc = min(cycs)
                    ok = "OK" if bad == 0 else f"BAD {bad} elems, {msg}"
                    print(f"probe {k} {'pieces' if lay else 'contig'} n {n:3d} {'inplace' if inplace else 'separate'}: "
                          f"{cyc:6.2f} cycles per 32 lanes  {ok}", flush=True)
                    if o.dump:
                        np.save(f"{o.out}_p{k}_l{lay}_n{n}_i{inplace}.npy", y[16:].view(np.uint16).copy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--asm")
    ap.add_argument("--probes", default="0")
    ap.add_argument("--n", default="8,144")
    ap.add_argument("--inplace", default="0")
    ap.add_argument("--erf", type=int, default=0)
    ap.add_argument("--layout", default="0", help="0 contiguous, 1 pieces (one call, jump 576)")
    ap.add_argument("--u", default="", help="per probe U, e.g. 5=1,6=2 (n argument = n / U trips; the generator prints U); "
                    "a wrong U runs the loop past its buffer (round 11: one timeout)")
    ap.add_argument("--flat", type=int, default=0, help="round 12: pass the flat-edge tables to the probe")
    ap.add_argument("--sweep", action="store_true", help="round 12: inputs = every finite bf16 (15 runs per case)")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--dump", action="store_true", help="save the first 96 outputs of each run")
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
