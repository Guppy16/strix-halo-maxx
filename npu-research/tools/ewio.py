#!/usr/bin/env python3
"""ewio.py - the I/O model of the EW core's ops (norms, SSM): compute vs input port vs output port (npu-research rev 12).

One column's EW core runs an op as a sequence of steps; its input ring (ewi, 20 x 768 B slots, bank 3) is filled by a
shim -> core packet flow, its output ring (ewo, 12 x 768 B slots, bank 0) drained by core MM2S -> shim S2MM. The slot /
lock protocol is ew8.cc's: in_need waits until `bytes` of whole slots are filled, in_done frees whole consumed slots,
out_begin waits for free slots covering the unit, out_end hands whole written slots to the DMA. The DMAs move one slot
at a time at R_in / R_out bytes per cycle (when a slot is free / full). Compute per step comes from tcall.py (static,
whole-call, 1 bundle = 1 cycle). The simulation is event-driven per slot; one op per call, rings start empty / free.

    ewio.py fit                 # the calibration table (r15h, r15i, NRMK, N1K, NPROF) over an R_in x R_out grid
    ewio.py norm --p1 263 --p2 320 --stats 1110 --nin 3 --rin 6 --rout 2.4     # one norm op
    ewio.py ssm --chunk 2950 --out-kb 6 --rin 6 --rout 2.4                      # one SSM op
    ewio.py levers --rin R --rout R                                             # rev-12 levers through the model

Units: cycles at 1.81 GHz; norm ops are 18 row blocks x 12 units (unit = n_in KB in, 1,600 B out); the SSM 12 heads x
18 chunks (10,752 B params per head; chunk 7 KB in, 6 KB out).
"""

import argparse
import itertools
import sys

SLOT, IN_SLOTS, OUT_SLOTS = 768, 20, 12
KB, UNIT_OUT, QS, RBN = 1024, 1600, 12, 18
GHZ = 1.81


class Ring:
    """a slot ring between a DMA and the core; `full` = slots the producer finished, `free` = slots the consumer released"""

    def __init__(self, slots, rate, ovh=0.0):
        self.slots, self.rate, self.ovh = slots, rate, ovh


def simulate(steps, rin, rout, in_ovh=0.0, out_ovh=0.0):
    """steps: list of ("need", bytes) | ("done", bytes) | ("cpu", cycles) | ("obegin", bytes) | ("oend", bytes).
    Returns (end cycle, waits dict). The input DMA always has data to send (the shim streams ahead); the output DMA
    drains whatever is full. Event-driven at slot granularity."""
    t = 0.0
    # input side: slot k (k = 0, 1, ...) becomes full at in_full[k]; it may start only when slot k - IN_SLOTS was freed
    in_full, in_freed = [], []  # times
    in_dma_t = 0.0  # when the input DMA finished its last slot
    consumed_b, avail_slots_used = 0, 0  # bytes consumed; slots released so far
    held_b = 0  # bytes acquired (whole slots) by in_need
    out_written_b, out_slot_done = 0, []  # out slot k handed to the DMA at out_slot_done[k]
    out_drained = []  # out slot k drained (freed) at
    out_dma_t = 0.0
    out_acq_b = 0  # bytes of output slots acquired
    waits = dict(in_need=0.0, out_begin=0.0)
    tail = 0

    def fill_in_until(k):
        """make sure input slots 0..k have fill times"""
        nonlocal in_dma_t
        while len(in_full) <= k:
            j = len(in_full)
            start = in_dma_t
            if j >= IN_SLOTS:
                if j - IN_SLOTS >= len(in_freed):
                    return False  # depends on a future release
                start = max(start, in_freed[j - IN_SLOTS])
            in_dma_t = start + SLOT / rin + in_ovh
            in_full.append(in_dma_t)
        return True

    def drain_out_until(k):
        nonlocal out_dma_t
        while len(out_drained) <= k and len(out_drained) < len(out_slot_done):
            j = len(out_drained)
            out_dma_t = max(out_dma_t, out_slot_done[j]) + SLOT / rout + out_ovh
            out_drained.append(out_dma_t)
        return len(out_drained) > k

    for kind, v in steps:
        if kind == "cpu":
            t += v
        elif kind == "need":
            need_b = consumed_b + v  # bytes that must be resident (from the op start)
            k = -(-need_b // SLOT) - 1  # last slot index needed
            ok = fill_in_until(k)
            assert ok, "input slot depends on a release that has not happened (ring too small for one unit)"
            if in_full[k] > t:
                waits["in_need"] += in_full[k] - t
                t = in_full[k]
        elif kind == "done":
            consumed_b += v
            while len(in_freed) < consumed_b // SLOT:
                in_freed.append(t)
        elif kind == "obegin":
            need_b = out_written_b + v
            k = -(-need_b // SLOT) - 1  # slot index that must be free: slot k needs k - OUT_SLOTS drained
            j = k - OUT_SLOTS
            if j >= 0:
                if not drain_out_until(j):
                    raise RuntimeError("output deadlock")
                if out_drained[j] > t:
                    waits["out_begin"] += out_drained[j] - t
                    t = out_drained[j]
        elif kind == "oend":
            out_written_b += v
            while len(out_slot_done) < out_written_b // SLOT:
                out_slot_done.append(t)
    # the op ends when its last output slot is drained
    if out_written_b % SLOT:
        out_slot_done.append(t)
    drain_out_until(len(out_slot_done) - 1)
    end = max(t, out_drained[-1] if out_drained else t)
    tail = end - t
    waits["tail"] = tail
    return end, waits


def norm_steps(nin, p1, p2, stats, params=3 * SLOT, rbs=RBN):
    st = [("need", params), ("cpu", 200), ("done", params)]
    for _ in range(rbs):
        for _ in range(QS):
            st += [("need", nin * KB), ("cpu", p1), ("done", nin * KB)]
        st.append(("cpu", stats))
        for _ in range(QS):
            st += [("obegin", UNIT_OUT), ("cpu", p2), ("oend", UNIT_OUT)]
    return st


def norm_steps_pipe(nin, p1, p2, stats, params=3 * SLOT, rbs=RBN, swap=16):
    """NPIPE: pass 2 of row block rb interleaved unit by unit with pass 1 of rb + 1 (stage slot q is free once nrm(rb, q)
    read it); `swap` = cycles per unit to save / restore the n1k accumulators around nrm"""
    st = [("need", params), ("cpu", 200), ("done", params)]
    for _ in range(QS):
        st += [("need", nin * KB), ("cpu", p1), ("done", nin * KB)]
    st.append(("cpu", stats))
    for rb in range(rbs):
        last = rb == rbs - 1
        for _ in range(QS):
            st += [("obegin", UNIT_OUT), ("cpu", p2), ("oend", UNIT_OUT)]
            if not last:
                st += [("need", nin * KB), ("cpu", p1 + swap), ("done", nin * KB)]
        if not last:
            st.append(("cpu", stats))
    return st


def ssm_steps(chunk, out_kb=6, params=10752, heads=12, chunks=18, head_cpu=700):
    st = []
    for _ in range(heads):
        st += [("need", params), ("cpu", head_cpu), ("done", params)]
        for _ in range(chunks):
            st += [("need", 7 * KB), ("cpu", chunk * 0.9), ("obegin", out_kb * KB), ("cpu", chunk * 0.1),
                   ("oend", out_kb * KB), ("done", 7 * KB)]
    return st


# static whole-call costs per unit / row block (tcall.py on the deployed elfs; branch-pinned per mode), cycles
# p1 = pass-1 unit iteration, p2 = pass-2 unit iteration, s = per-row-block rest (stats, begin / end, loop glue)
BUILDS = {
    # r15h / r15i: template norm_rows<LN> / <RMS>; rb iteration from tcall (whole call / 18)
    "r15h": dict(ln=dict(p1=362, p2=632, rb=235258 / 18), rms=dict(p1=309, p2=503, rb=188485 / 18)),
    "r15i": dict(ln=dict(p1=362, p2=539, rb=215170 / 18), rms=dict(p1=309, p2=410, rb=168397 / 18)),
    "r16f": dict(ln=dict(p1=273, p2=325, rb=8788), rms=dict(p1=273, p2=257, rb=7483)),  # = r15k3 / r15m norms (NRMK)
    "r16h": dict(ln=dict(p1=256, p2=317, rb=8089), rms=dict(p1=256, p2=249, rb=6867)),  # = r16z (N1K)
}
# the step's norm ops: (mode, n_in, count) per depth-12 step (plan8 full_step)
OPS = [("rms", 1, 1), ("rms", 3, 11), ("rms", 2, 12), ("ln", 3, 12), ("ln", 2, 12), ("ln", 3, 1)]


def op_cycles(b, mode, nin, rin, rout, fixed=0, pipe=False, dp1=0, dp2=0, ds=0):
    c = BUILDS[b][mode]
    stats = c["rb"] - QS * (c["p1"] + c["p2"]) + ds
    params = 3 * SLOT if mode == "rms" else 5 * SLOT
    mk = norm_steps_pipe if pipe else norm_steps
    end, w = simulate(mk(nin, c["p1"] + dp1, c["p2"] + dp2, stats, params), rin, rout)
    return end + fixed, w


def step_ms(b, rin, rout, fixed=0, **kw):
    tot, wn, wo = 0.0, 0.0, 0.0
    for mode, nin, n in OPS:
        e, w = op_cycles(b, mode, nin, rin, rout, fixed, **kw)
        tot += n * e
        wn += n * w["in_need"]
        wo += n * w["out_begin"]
    units = sum(n for *_, n in OPS) * RBN * QS
    return tot / GHZ / 1e6, wn / units, wo / units


def cmd_fit(o):
    print("measured: r15h -> r15i -0.53, r15i -> NRMK (r15k3) -0.37..-0.42, r16f -> r16h (N1K) -0.16..-0.28 ms; r15h norms "
          "n_in-invariant (LN n3 136 / n2 138, RMS n3 119 / n2 117 us prev-last -> last); NPROF r16hN in_need 104.8 per unit "
          "(static in_need 28-41 + one stamp ~10-20: wait ~45-65)")
    print("static (no I/O): r15h->r15i %.2f, NRMK %.2f, N1K %.2f ms" % tuple(
        step_ms(b2, 1e9, 1e9)[0] - step_ms(b1, 1e9, 1e9)[0] for b1, b2 in (("r15h", "r15i"), ("r15i", "r16f"), ("r16f", "r16h"))))
    print(f"{'R_in':>5s} {'R_out':>5s} | {'r15i':>6s} {'NRMK':>6s} {'N1K':>6s} | r15h LN n3-n2 us | r16h in_need/unit out_begin/unit | score")
    rows = []
    for rin, rout in itertools.product(o.rins, o.routs):
        s = {b: step_ms(b, rin, rout) for b in BUILDS}
        d1 = s["r15i"][0] - s["r15h"][0]
        d2 = s["r16f"][0] - s["r15i"][0]
        d3 = s["r16h"][0] - s["r16f"][0]
        ln3 = op_cycles("r15h", "ln", 3, rin, rout)[0] / GHZ / 1e3
        ln2 = op_cycles("r15h", "ln", 2, rin, rout)[0] / GHZ / 1e3
        wn, wo = s["r16h"][1], s["r16h"][2]
        # score: squared misses against the measured deltas (ms) + the n-invariance (us / 10) + the in_need wait (per unit / 100)
        sc = (d1 + 0.53) ** 2 + (d2 + 0.40) ** 2 + (d3 + 0.22) ** 2 + ((ln3 - ln2) / 10) ** 2 / 100 + ((wn - 55) / 100) ** 2
        rows.append((sc, rin, rout, d1, d2, d3, ln3 - ln2, wn, wo))
    for sc, rin, rout, d1, d2, d3, dn, wn, wo in sorted(rows)[:o.top]:
        print(f"{rin:5.2f} {rout:5.2f} | {d1:6.2f} {d2:6.2f} {d3:6.2f} | {dn:16.1f} | {wn:16.0f} {wo:16.0f} | {sc:.4f}")


def cmd_norm(o):
    stats = o.stats
    end, w = simulate(norm_steps(o.nin, o.p1, o.p2, stats, o.params), o.rin, o.rout)
    comp = 200 + RBN * (QS * (o.p1 + o.p2) + stats)
    print(f"norm n_in {o.nin}: {end:.0f} cycles = {end / GHZ / 1e3:.1f} us (compute {comp} = {comp / GHZ / 1e3:.1f} us); "
          f"waits per unit: in_need {w['in_need'] / 216:.0f}, out_begin {w['out_begin'] / 216:.0f}; drain tail {w['tail']:.0f}")
    print(f"  floors per op: input {o.nin * 216 * KB / o.rin / GHZ / 1e3:.1f} us, output {216 * UNIT_OUT / o.rout / GHZ / 1e3:.1f} us")


def cmd_ssm(o):
    end, w = simulate(ssm_steps(o.chunk, o.out_kb), o.rin, o.rout)
    comp = 12 * (700 + 18 * o.chunk)
    print(f"ssm: {end:.0f} cycles = {end / GHZ / 1e3:.1f} us (compute {comp / GHZ / 1e3:.1f} us); waits per chunk: in_need "
          f"{w['in_need'] / 216:.0f}, out_begin {w['out_begin'] / 216:.0f}; floors per chunk: input {7 * KB / o.rin:.0f}, output "
          f"{o.out_kb * KB / o.rout:.0f} cycles")


LEVERS = [  # (name, kwargs on r16h) - the rev-12 norm levers
    ("NPIPE (pass 2 of rb || pass 1 of rb+1)", dict(pipe=True)),
    ("NG1 pass-1 glue inline (-80 per unit)", dict(dp1=-80)),
    ("NG2 pass-2 glue inline (-100 per unit)", dict(dp2=-100)),
    ("NS stats in asm (LN -860 / RMS -570 per rb: -700 avg)", dict(ds=-700)),
    ("NPIPE + NG1 + NG2", dict(pipe=True, dp1=-80, dp2=-100)),
    ("NPIPE + NG1 + NG2 + NS", dict(pipe=True, dp1=-80, dp2=-100, ds=-700)),
]


def cmd_levers(o):
    base = step_ms("r16h", o.rin, o.rout)[0]
    print(f"norms at r16h/z, R_in {o.rin} R_out {o.rout}: {base:.2f} ms per step (+ ~8 us fixed per op); static floor of the "
          f"output port {49 * 216 * UNIT_OUT / o.rout / GHZ / 1e6:.2f} ms")
    for name, kw in LEVERS:
        v = step_ms("r16h", o.rin, o.rout, **kw)
        print(f"  {name:52s} {v[0] - base:+.2f} ms   (in_need {v[1]:.0f} / out_begin {v[2]:.0f} per unit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--rins", type=float, nargs="+", default=[4.0, 5.0, 6.0, 7.0, 8.0, 12.0])
    f.add_argument("--routs", type=float, nargs="+", default=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0])
    f.add_argument("--top", type=int, default=12)
    n = sub.add_parser("norm")
    for k, d in (("p1", 256), ("p2", 317), ("stats", 1110), ("nin", 3), ("rin", 6.0), ("rout", 2.4), ("params", 5 * SLOT)):
        n.add_argument(f"--{k}", type=type(d), default=d)
    s = sub.add_parser("ssm")
    for k, d in (("chunk", 2950), ("out_kb", 6), ("rin", 6.0), ("rout", 2.4)):
        s.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(d), default=d)
    lv = sub.add_parser("levers")
    lv.add_argument("--rin", type=float, default=6.0)
    lv.add_argument("--rout", type=float, default=2.35)
    o = ap.parse_args()
    {"fit": cmd_fit, "norm": cmd_norm, "ssm": cmd_ssm, "levers": cmd_levers}[o.cmd](o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
