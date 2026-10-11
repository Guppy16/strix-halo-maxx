"""pass 19 KTLP-D: the fixed KTL kernel with real locks, A and B fed by shim DMAs (one core). Two-pass build: the kernel's buffer
toggles must use the bench's own B buffer addresses, read from the first build's input_with_addresses.mlir.
build (npu-iron, CPU): python3 r24/ktlpd.py build --json r24/e7k_r70b_n16.json      (writes r24/build/ktlpd*)
run   (npu-iron, NPU): python3 r24/ktlpd.py run"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BD = HERE / "build"
PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie"
N = 3
OUT = str(BD / "ktlpd")
NW = 64


def rng_bytes():
    s = 12345
    out = []

    def rnd():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s >> 8
    a = np.zeros((3, 1152), np.uint8)
    for k in range(3):
        for i in range(1152):
            a[k, i] = (120 + rnd() % 8) if i % 9 == 0 else rnd() & 0xFF
    lb = np.zeros((64, 72), np.uint8)
    for p in range(64):
        for j in range(72):
            lb[p, j] = (120 + rnd() % 8) if j % 9 == 0 else rnd() & 0xFF
    return a, lb


def border(h):
    m = re.search(r"mm_e7k_border\[64\] = \{([^}]*)\}", Path(h).read_text())
    return [int(x) for x in m.group(1).split(",")]


def objects(kernel_s):
    b = f"{PEANO}/bin"
    BD.mkdir(exist_ok=True)
    base = ["-std=c++20", "-D__AIE_API_AIE_ADF_HPP__", "-Wno-attributes", "-w", "--target=aie2p-none-unknown-elf", "-O2",
            f"-I{PEANO}/include/aie2p-none-unknown-elf", "-I/opt/ironenv/lib/python3.13/site-packages/mlir_aie/include", f"-I{HERE}"]
    for cmd in ([f"{b}/clang", "--target=aie2p-none-unknown-elf", "-c", kernel_s, "-o", str(BD / "ktlpd_k.o")],
                [f"{b}/clang++"] + base + [f"-DKTLPD_N={N}", "-c", str(HERE / "ktlpd.cc"), "-o", str(BD / "ktlpd_cc.o")],
                [f"{b}/ld.lld", "-r", str(BD / "ktlpd_cc.o"), str(BD / "ktlpd_k.o"), "-o", str(BD / "ktlpd.o")]):
        subprocess.run(cmd, check=True)
    return str(BD / "ktlpd.o")


def build_once(kernel_s, tag):
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import DMAChannelDir
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Acquire, Bd, Buffer, CompileTime, DmaChannel, Flow, In, Kernel, Lock, ObjectFifo, Out, Program, Release, Runtime, TaskGroup, TileDma, Worker
    from aie.iron.device import NPU2, Tile

    obj = objects(kernel_s)

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def bench(Y: Out, AH: In, BH: In, *, key: CompileTime[str] = ""):
        T = np.ndarray[(NW,), np.dtype[np.int32]]
        A_l1 = np.ndarray[(288,), np.dtype[np.uint32]]
        B_l1 = np.ndarray[(4608,), np.dtype[np.uint8]]
        A_host = np.ndarray[(N * 9 * 288,), np.dtype[np.uint32]]
        B_host = np.ndarray[(N * 1152,), np.dtype[np.uint32]]
        fn = Kernel("ktlpd", obj, [T, A_l1, A_l1, A_l1, B_l1, B_l1], stack_size_override=0x600)
        yf = ObjectFifo(T, name="Y", depth=1)
        t, shim = Tile(0, 2), Tile(0, 0)
        abufs = [Buffer(tile=t, type=A_l1, name=f"a{i}") for i in range(3)]
        bbufs = [Buffer(tile=t, type=B_l1, name=f"b{i}") for i in range(2)]
        bp = Lock(tile=t, init=2, name="bP", lock_id=0)
        bc = Lock(tile=t, init=0, name="bC", lock_id=1)
        af = Lock(tile=t, init=3, name="aF", lock_id=4)
        au = Lock(tile=t, init=0, name="aU", lock_id=5)
        fa = Flow(shim, t, src_channel=1, dst_channel=1)
        fb = Flow(shim, t, src_channel=0, dst_channel=0)
        dma = TileDma(t, [DmaChannel(DMAChannelDir.S2MM, fa.endpoint(t), [Bd(x, acquires=[Acquire(af)], releases=[Release(au)]) for x in abufs]),
                          DmaChannel(DMAChannelDir.S2MM, fb.endpoint(t), [Bd(x, acquires=[Acquire(bp)], releases=[Release(bc)]) for x in bbufs])])

        def core(yp, fn, a0, a1, a2, b0, b1):
            y = yp.acquire(1)
            fn(y, a0, a1, a2, b0, b1)
            yp.release(1)

        w = Worker(core, [yf.prod(), fn, *abufs, *bbufs], tile=t, stack_size=0x800, data_size=40 * 1024)

        def seq(y, ah, bh, yc):
            from aie.dialects.aiex import npu_maskwrite32
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            tg = TaskGroup()
            fa.fill(ah, tap=TensorAccessPattern((N * 9 * 288,), 0, [1, 1, N * 9, 288], [0, 0, 288, 1]), group=tg, wait=False)
            fb.fill(bh, tap=TensorAccessPattern((N * 1152,), 0, [1, 1, N, 1152], [0, 0, 1152, 1]), group=tg, wait=False)
            yc.drain(y, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [T, A_host, B_host, yf.cons()])
        for f in (fa, fb):
            rt.add_flow(f)
        rt.add_tile_dma(dma)
        for lk in (bp, bc, af, au):
            rt.add_lock(lk)
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    import hashlib
    h = hashlib.md5(Path(obj).read_bytes()).hexdigest()[:8]
    bench.specialize(key=f"{OUT}_{tag}_{h}").compile(xclbin_path=OUT + ".xclbin", inst_path=OUT + ".bin")
    mlir = Path(OUT + ".prj") / "input_with_addresses.mlir"
    if not mlir.exists():
        cands = sorted(Path(OUT).parent.glob("**/input_with_addresses.mlir"), key=lambda p: p.stat().st_mtime)
        mlir = cands[-1]
    txt = mlir.read_text()
    addr = {nm: 0x70000 + int(a) for a, nm in re.findall(r'address = (\d+) : i32,[^}]*sym_name = "(b[01])"', txt)}
    locks = re.findall(r'aie\.lock\(%tile_0_2, (\d+)\) \{init = (\d+) : i32, sym_name = "(\w+)"', txt)
    print(f"built {OUT} ({tag}): B buffers {[hex(addr.get(k, 0)) for k in ('b0', 'b1')]}; locks {locks}")
    return addr


def build(o):
    d = json.loads(Path(o.json).read_text())
    first = HERE / "build" / "ktlpd_pass1.s"
    subprocess.run(["python3", str(HERE.parent / "r22" / "gen_e7k.py"), "fit", o.json, "--out", str(first), "--place",
                    "y=0x7be00:0x7d000,x=0x7cdc0:0x7dfc0"], check=True, stdout=subprocess.DEVNULL)
    (HERE / "e7k.h").write_text(first.with_suffix(".h").read_text())
    addr = build_once(str(first), "p1")
    y0, y1 = addr["b0"], addr["b1"]
    final = HERE / "build" / "ktlpd_k.s"
    subprocess.run(["python3", str(HERE.parent / "r22" / "gen_e7k.py"), "fit", o.json, "--out", str(final), "--place",
                    f"y={y0:#x}:{y1:#x},x={y0 + 4032:#x}:{y1 + 4032:#x}"], check=True)
    (HERE / "e7k.h").write_text(final.with_suffix(".h").read_text())
    addr2 = build_once(str(final), "p2")
    assert addr2 == addr, ("placement moved between passes", addr, addr2)


def run(o):
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    a, lb = rng_bytes()
    bo_ = border(HERE / "e7k.h")
    A_h = np.concatenate([a[s % 3] for _ in range(N) for s in range(9)]).view(np.uint32)
    obj = np.concatenate([lb[bo_[i]] for i in range(64)])
    B_h = np.concatenate([obj for _ in range(N)]).view(np.uint32)
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(OUT + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(OUT + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bos = [pyxrt.bo(dev, nb, pyxrt.bo.host_only, kern.group_id(3 + k)) for k, nb in enumerate((NW * 4, A_h.nbytes, B_h.nbytes))]
    bos[1].write(A_h.tobytes(), 0)
    bos[2].write(B_h.tobytes(), 0)
    for bo in bos[1:]:
        bo.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    st = kern(3, bo_i, len(instr), *bos).wait(3_000_000)
    assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bos[0].sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    y = [int(v) for v in np.frombuffer(bos[0].read(NW * 4, 0), np.int32)]
    ref = json.loads((BD / "ktlp_ref.json").read_text())["0,3"]
    same = sum(p == q for p, q in zip(y[:36], ref))
    print(f"KTLP-D (real locks, DMA-fed A and B, one call of {y[39]} k-steps): {y[36]} cycles = {y[36] / y[39]:.0f} per k-step; "
          f"{same}/36 piece hashes equal to the lock-free E7 reference; B buffers {y[37] & 0xFFFFF:#x} / {y[38] & 0xFFFFF:#x}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--json", default=str(HERE / "e7k_r70b_n16.json"))
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
