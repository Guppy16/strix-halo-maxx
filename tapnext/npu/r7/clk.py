"""Clock probe (round 7): settle the AIE core clock without reading the core timer (read_tm hung the core in round 6).

clk.cc runs a pure-register hardware loop (scalar LCG: 4 bundles per trip; dependent bf16 vmac: 3 bundles per trip,
counts from r6/asmstat.py on the -S output) with the trip count from the input buffer. No loads/stores in the loop, so
nothing can stall it: core cycles = trips x bundles + a constant. Host-time dispatches at several trip counts and fit
time = a + trips x bundles / f: the slope gives f, the intercept the dispatch overhead.

build (npu-iron, no device):  python3 r7/clk.py build --out build/clk
run   (npu-iron, NPU):        python3 r7/clk.py run --out build/clk
"""

import argparse
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INC = "/work/npu-moondream/mlir-aie/aie_kernels"
NW = 64  # int32 words in and out
BUNDLES = {0: 4, 1: 3}  # bundles per trip (asmstat: chain_s .LBB1_2, chain_v .LBB2_2)


def build(o):
    import aie.iron as iron
    from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.device import NPU2, Tile

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def probe(X: In, Y: Out):
        T = np.ndarray[(NW,), np.dtype[np.int32]]
        fn = ExternalFunction("clk", object_file_name="clk.o", source_file=str(HERE / "clk.cc"), arg_types=[T, T],
                              compile_flags=[f"-I{INC}", "-O2"] + (["-DSTAMP"] if o.stamp else []))
        xf = ObjectFifo(T, name="X", depth=1)
        yf = ObjectFifo(T, name="Y", depth=1)

        def core(xc, yp, fn):
            x = xc.acquire(1)
            y = yp.acquire(1)
            fn(x, y)
            xc.release(1)
            yp.release(1)

        w = Worker(core, [xf.cons(), yf.prod(), fn], tile=Tile(0, 2), stack_size=0x400)

        def seq(x, y, xp, yc):
            if o.stamp:  # P15: enable the core's processor bus (read_tm hangs the core without it)
                from aie.dialects.aiex import npu_maskwrite32
                npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            tg = TaskGroup()
            xp.fill(x, group=tg, wait=False)
            yc.drain(y, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [T, T, xf.prod(), yf.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    probe.compile(xclbin_path=o.out + ".xclbin", inst_path=o.out + ".bin")


def run(o):
    import pyxrt

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_x = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, kern.group_id(3))
    bo_y = pyxrt.bo(dev, NW * 4, pyxrt.bo.host_only, kern.group_id(4))

    def once(trips, mode):
        x = np.zeros(NW, np.int32)
        x[0], x[1], x[2] = trips, mode, 12345
        x[16:48] = 0x3F803F80 if mode else 0  # bf16 1.0 pairs: acc += 1 per lane per trip (exact up to 2^24)
        bo_x.write(x.tobytes(), 0)
        bo_x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t0 = time.perf_counter()
        r = kern(3, bo_i, len(instr), bo_x, bo_y)
        st = r.wait(o.timeout_ms * 1000)
        dt = time.perf_counter() - t0
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
        bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return dt, np.frombuffer(bo_y.read(NW * 4, 0), np.int32).copy()

    # first: one short dispatch (new design rule), and check the result is what the loop computes
    dt, y = once(1000, 0)
    xs = 12345
    for _ in range(1000):
        xs = (xs * 1664525 + 1013904223) & 0xFFFFFFFF
    print(f"first dispatch {dt * 1e6:.0f} us, scalar result {'OK' if (int(y[0]) & 0xFFFFFFFF) == xs else 'WRONG'}", flush=True)
    dt, y = once(1000, 1)
    print(f"vector result {y[16:48].view(np.float32)[:2]} (expect 1000)", flush=True)
    for _ in range(20):  # DPM ramp
        once(10000, 0)
    if o.stamp:  # P15: cycles from the core timer vs bundles x trips
        for mode, b in ((2, 4), (3, 3)):
            for n in (1000, 100000, 4000000):
                dt, y = once(n, mode)
                cyc = int(y[1]) & 0xFFFFFFFF
                print(f"STAMP mode {mode} trips {n:>8d}: timer {cyc:>10d} cycles, bundles x trips {b * n:>10d}, "
                      f"ratio {cyc / (b * n):.4f}; host {dt * 1e6:.0f} us -> {cyc / (dt * 1e6) / 1e3:.3f} GHz lower bound", flush=True)
        return
    trips = [int(t) for t in o.trips.split(",")]
    for mode in (0, 1):
        rows = []
        for n in trips:
            ts = [once(n, mode)[0] for _ in range(o.iters)]
            rows.append((n, float(np.median(ts)), float(np.min(ts))))
            print(f"mode {mode} trips {n:>10d}: median {rows[-1][1] * 1e3:9.3f} ms  min {rows[-1][2] * 1e3:9.3f} ms", flush=True)
        n = np.array([r[0] for r in rows], float) * BUNDLES[mode]
        for k, lab in ((1, "median"), (2, "min")):
            t = np.array([r[k] for r in rows])
            slope, icpt = np.polyfit(n, t, 1)
            res = t - (icpt + slope * n)
            print(f"mode {mode} ({BUNDLES[mode]} bundles/trip) fit on {lab}: clock {1e-9 / slope:.4f} GHz, "
                  f"intercept {icpt * 1e6:.1f} us, max |residual| {np.abs(res).max() * 1e6:.1f} us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--trips", default="0,1000000,4000000,16000000,32000000")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--stamp", action="store_true", help="build: P15 timer reads (modes 2/3) + the bus enable")
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
