"""Round 10 bench: the A slot driven by RUNTIME memtile tasks (IRON DmaEndpoint.task), one column, one core.

Per op (K = 768: 12 k-step chunks of 10,368 B; K = 1536: 24), nt tiles each reading the op's whole A:
  load  (memtile S2MM): chunk by chunk, acquire E (free chunk), release F (+1 per loaded chunk)
  tile 0 sends: chunk by chunk, acquire F, release F2 (the first tile starts after the first chunk)
  middle tiles: the whole region, no locks            (tile 0 has seen every chunk loaded)
  last tile   : chunk by chunk, acquire F2, release E (the next op's load of chunk q waits only for chunk q)
(a BD has one acquire and one release field, or none: F2 carries tile 0's per-chunk token to the last tile)
E is split into E_lo (chunks 0-11) and E_hi (12-23) so K = 768 ops never take K = 1536's upper tokens.
`reuse`: no load; the previous op's last-tile sends become middle ones.

    python3 r10/rtm.py build --out build/r10/rtm      (npu-iron)
    python3 r10/rtm.py run --out build/r10/rtm        (npu-iron, NPU)
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = 2592  # words per k-step chunk (144 rows x 64 bfp16)
PLAN = [(768, 4, 0), (1536, 4, 0), (768, 2, 0), (768, 4, 1), (1536, 2, 0), (768, 3, 0), (768, 16, 0), (1536, 4, 0)]
GHZ = 1.81


def regions(plan):
    offs, o = [], 0
    for K, nt, reuse in plan:
        if reuse:
            offs.append(offs[-1])
            continue
        offs.append(o)
        o += (K // 64) * CH
    return offs, o


def build(o):
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import DMAChannelDir
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (Acquire, Bd, Buffer, CompileTime, DmaChannel, ExternalFunction, Flow, In, Lock, ObjectFifo, Out,
                          Program, Release, Runtime, TaskGroup, TileDma, Worker)
    from aie.iron.dataflow.tile_dma import BdIteration
    from aie.iron.device import NPU2, Tile

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def rtm(ACT: In, RES: Out, *, plan: CompileTime[str] = "[]"):
        from aie.dialects.aiex import npu_maskwrite32

        ops = json.loads(plan)
        offs, total = regions(ops)
        nres = 3 * len(ops)
        Slot = np.ndarray[(288,), np.dtype[np.uint32]]
        Tab = np.ndarray[(1 + len(ops),), np.dtype[np.int32]]
        Res = np.ndarray[(nres,), np.dtype[np.uint32]]
        MemSlot = np.ndarray[(24 * CH,), np.dtype[np.uint32]]
        shim, mem, core = Tile(0, 0), Tile(0, 1), Tile(0, 2)
        k = ExternalFunction("rtm_run", object_file_name="rtm.o", source_file=str(HERE / "rtm.cc"),
                             arg_types=[Slot, Slot, Slot, Tab, Res], compile_flags=["-O2"])
        a_in = Flow(shim, mem)
        a_out = Flow(mem, core)
        slot = Buffer(tile=mem, type=MemSlot, name="As")
        F = Lock(tile=mem, init=0, name="F")
        Elo = Lock(tile=mem, init=12, name="Elo")
        Ehi = Lock(tile=mem, init=12, name="Ehi")
        F2 = Lock(tile=mem, init=0, name="F2")  # a BD takes both a lock acquire and a release, or neither: tile 0 -> last tile
        bufs = [Buffer(tile=core, type=Slot, name=f"a{i}") for i in range(3)]
        free = Lock(tile=core, init=3, name="aF", lock_id=4)
        fill = Lock(tile=core, init=0, name="aU", lock_id=5)
        cdma = TileDma(core, [DmaChannel(DMAChannelDir.S2MM, a_out.endpoint(core),
                                         [Bd(b, acquires=[Acquire(free)], releases=[Release(fill)]) for b in bufs])])
        tab = Buffer(tile=core, initial_value=np.array([len(ops)] + [nt * (K // 64) * 9 for K, nt, _ in ops], np.int32), name="tab")
        res = ObjectFifo(Res, name="res", depth=1)

        def core_fn(b0, b1, b2, tab, rp, k, lf, lu):
            r = rp.acquire(1)
            k(b0, b1, b2, tab, r)
            rp.release(1)

        w = Worker(core_fn, [*bufs, tab, res.prod(), k, free, fill], tile=core, stack_size=0x800)
        ACT_ty = np.ndarray[(max(total, 64),), np.dtype[np.uint32]]

        def seq(act, rs, rc):
            npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)
            me_in, me_out = a_in.endpoint(mem), a_out.endpoint(mem)

            def chunk(h, *, acq=None, rel=None):
                return Bd(slot, tap=TensorAccessPattern((24 * CH,), 12 * CH * h, [1, 1, 1, CH], [0, 0, 0, 1]),
                          acquires=[Acquire(acq)] if acq else [], releases=[Release(rel)] if rel else [],
                          iteration=BdIteration(size=12, stride=CH))

            E = [Elo, Ehi]
            ld = [me_in.task(chunk(h, acq=E[h], rel=F), runs=12) for h in range(2)]
            t0 = [me_out.task(chunk(h, acq=F, rel=F2), runs=12) for h in range(2)]
            last = [me_out.task(chunk(h, acq=F2, rel=E[h]), runs=12) for h in range(2)]
            mid = {K: me_out.task(Bd(slot, tap=TensorAccessPattern((24 * CH,), 0, [1, 1, 1, (K // 64) * CH], [0, 0, 0, 1]))) for K in (768, 1536)}
            tg = TaskGroup()
            for i, (K, nt, reuse) in enumerate(ops):
                nh = K // 768
                reused_next = i + 1 < len(ops) and ops[i + 1][2]
                if not reuse:
                    a_in.fill(act, tap=TensorAccessPattern((max(total, 64),), offs[i], [1, 1, 1, (K // 64) * CH], [0, 0, 0, 1]), group=tg)
                    for h in range(nh):
                        ld[h].start()
                nmid = nt - (0 if reuse else 1) - (0 if reused_next else 1)
                if not reuse:
                    for h in range(nh):
                        t0[h].start()
                if nmid > 0:
                    mid[K].start(repeat_count=nmid - 1)
                if not reused_next:
                    for h in range(nh):
                        last[h].start()
            rc.drain(rs, group=tg, wait=True)
            tg.finish()
            for t in ld + t0 + last + list(mid.values()):
                t.free()

        rt = Runtime(seq, [ACT_ty, Res, res.cons()])
        for f in (a_in, a_out):
            rt.add_flow(f)
        rt.add_tile_dma(cdma)
        for lk in (F, Elo, Ehi, F2):
            rt.add_lock(lk)
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

    iron.set_current_device(NPU2())
    out = str(Path(o.out).resolve())
    rtm.specialize(plan=json.dumps(PLAN)).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
    print("built", out)


def run(o):
    import sys
    sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    S = pyxrt.xclBOSyncDirection
    offs, total = regions(PLAN)
    rng = np.random.default_rng(1)
    act = rng.integers(0, 2 ** 32, total, dtype=np.uint64).astype(np.uint32)
    exp = []
    for (K, nt, reuse), off in zip(PLAN, offs):
        w = act[off:off + (K // 64) * CH].reshape(-1, 288)
        sums = (w.astype(np.uint64).sum(1) & 0xFFFFFFFF).astype(np.uint64)
        h = 0
        for _ in range(nt):
            for s in sums:
                h = (h * 31 + int(s)) & 0xFFFFFFFF
        exp.append(h)
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(o.out + ".xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(o.out + ".bin", dtype=np.uint32)
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_a = pyxrt.bo(dev, act.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_a.write(act.tobytes(), 0)
    bo_a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    nres = 3 * len(PLAN)
    bo_r = pyxrt.bo(dev, nres * 4, pyxrt.bo.host_only, kern.group_id(4))
    for it in range(o.iters):
        st = kern(3, bo_i, len(instr), bo_a, bo_r).wait(o.timeout_ms * 1000)
        assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED", st
    bo_r.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    r = np.frombuffer(bo_r.read(nres * 4, 0), np.uint32).reshape(-1, 3)
    ok = all(int(r[i, 0]) == exp[i] for i in range(len(PLAN)))
    print(f"RTM {'OK' if ok else 'WRONG'}: per op (K, nt, reuse): hash ok, op-start wait us, stripes us, B/cycle into the core")
    for i, (K, nt, reuse) in enumerate(PLAN):
        nb = nt * (K // 64) * CH * 4
        print(f"  {K:5d} {nt:3d} {reuse} {int(r[i, 0]) == exp[i]!s:5s} wait {r[i, 1] / GHZ / 1e3:8.2f} run {r[i, 2] / GHZ / 1e3:8.2f}  "
              f"{nb / max(int(r[i, 2]), 1):5.2f} B/cycle")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--timeout-ms", type=int, default=3000)
    o = ap.parse_args()
    dict(build=build, run=run)[o.cmd](o)
