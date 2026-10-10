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


# ---------------------------------------------------------------------------------------------------------------------
# Revision 13 (research pass 13): the refit against round 17's NPROF stamps (in_need AND out_begin) and the timeline's
# per-kind norm walls. Changes to the model:
#   * in-situ per-unit costs: p1 += D1 (n1k step stamped 161.5 vs static 134 + call 6 + stamp ~4: +17), p2 += D2 (the
#     one-core nrm bench excess, RMS +19 / LN +33 per unit, scaled by s2);
#   * NPIPE with groups of G units (ew8.cc: n1k _end / _begin + g_nacc per group, GCOST cycles; one pair is already in
#     the serial build's per-row-block rest);
#   * the stamps measure the call too: net wait = stamped - (in_need 35 + jl 6 + stamp ~3 = 44) / (out_begin 31 + 6 + 3 = 40);
#   * a fixed per-op latency F (us) for the timeline walls (prev op's last write -> this op's last write);
#   * optional DDR coupling (--ddr D): while both DMAs of a column are busy they share D B/cycle (8 columns run in step,
#     so a column's share of the DRAM peak is ~55-60 GB/s / 8 = 3.8-4.1 B/cycle).
# Targets (round 17, MEASURED): NPROF r16z in_need 101.5 / out_begin 45.0 per unit (net ~57.5 / ~5), NPIPE_G=2 61.0 / 46.6
# (net ~17 / ~6.6); r16z timeline walls RMS n2 83.8, RMS n3 103.1, LN n2 100.6, LN n3 115.4 us (12 ops each, +-3); deltas
# r15i -0.53, NRMK -0.40, N1K -0.22, NPIPE G2 +0.07 (noisy), G6 ~-0.10 ms.
C_IN, C_OUT = 44, 40           # static cycles inside the in_need / out_begin stamp intervals (call + stamp)
D1 = 17                        # n1k step in situ above static (NPROF)
D2 = dict(rms=19, ln=33)       # nrm_* one-core bench above static
GCOST = 45                     # NPIPE: n1k _end + _begin (+ g_nacc) per group (main, round 17)
PARAMS13 = dict(rms=3 * SLOT, ln=5 * SLOT)   # r16z param fills (AL1 made them 3 KB / 6 KB in r17d)
OPS13 = [("rms", 2, 1), ("rms", 3, 11), ("rms", 2, 12), ("ln", 3, 12), ("ln", 2, 12), ("ln", 3, 1)]  # the n1 op as n2


def norm_steps_g(nin, p1, p2, stats, params, G, gcost=GCOST, rbs=RBN):
    """NPIPE as built in round 17 (ew8.cc -DNPIPE): rb = -1 runs pass 1 of row block 0; then for each row block, per
    group of G units: G x pass 2 of rb, then (if a next row block exists) the group cost and G x pass 1 of rb + 1."""
    st = [("need", params), ("cpu", 200), ("done", params)]
    for rb in range(-1, rbs):
        more = rb + 1 < rbs
        for _q0 in range(0, QS, G):
            if rb >= 0:
                for _ in range(G):
                    st += [("obegin", UNIT_OUT), ("cpu", p2), ("oend", UNIT_OUT)]
            if not more:
                continue
            st.append(("cpu", gcost))
            for _ in range(G):
                st += [("need", nin * KB), ("cpu", p1), ("done", nin * KB)]
        if more:
            st.append(("cpu", stats - GCOST))  # the serial build's one _begin / _end pair is inside `stats`
    return st


def simulate_ddr(steps, rin, rout, ddr):
    """simulate() with the two DMAs coupled: a slot transfer that overlaps the other direction's activity runs at the
    shared rate. Approximation: iterate - first pass with independent rates, then rescale each DMA's slot time by
    max(1, (rin_eff + rout_eff) / ddr) over the fraction of time both are busy."""
    end, w = simulate(steps, rin, rout)
    if not ddr or rin + rout <= ddr:
        return end, w
    # busy fractions from the first pass (bytes / rate / end)
    bi = sum(v for k, v in steps if k == "need") / rin / end
    bo = sum(v for k, v in steps if k == "oend") / rout / end
    both = max(0.0, bi + bo - 1.0)  # lower bound of the overlap
    f = (rin + rout) / ddr
    ri = rin / (1 + both / max(bi, 1e-9) * (f - 1))
    ro = rout / (1 + both / max(bo, 1e-9) * (f - 1))
    return simulate(steps, ri, ro)


def op13(b, mode, nin, rin, rout, s2=1.0, G=None, params=None, d1=D1, dp1=0, dp2=0, ds=0, gcost=GCOST, ddr=0.0):
    c = BUILDS[b][mode]
    insitu = b in ("r16f", "r16h")
    p1 = c["p1"] + (d1 if b == "r16h" else 0) + dp1
    p2 = c["p2"] + (s2 * D2[mode] if insitu else 0) + dp2
    stats = c["rb"] - QS * (c["p1"] + c["p2"]) + ds
    params = params or PARAMS13[mode]
    steps = norm_steps_g(nin, p1, p2, stats, params, G, gcost) if G else norm_steps(nin, p1, p2, stats, params)
    return simulate_ddr(steps, rin, rout, ddr)


def step13(b, rin, rout, fold=False, **kw):
    tot = wn = wo = 0.0
    per = {}
    for mode, nin, n in OPS13:
        nn = 2 if (fold and nin == 3) else nin
        e, w = op13(b, mode, nn, rin, rout, **kw)
        tot += n * e
        wn += n * w["in_need"]
        wo += n * w["out_begin"]
        per[(mode, nin)] = e
    units = sum(n for *_, n in OPS13) * RBN * QS
    return tot / GHZ / 1e6, wn / units, wo / units, per


WALLS = {("rms", 2): 83.8, ("rms", 3): 103.1, ("ln", 2): 100.6, ("ln", 3): 115.4}  # r16z timeline (us, 12 ops each)


def score13(rin, rout, s2, ddr=0.0):
    kw = dict(s2=s2, ddr=ddr)
    z = step13("r16h", rin, rout, **kw)
    g2 = step13("r16h", rin, rout, G=2, **kw)
    g6 = step13("r16h", rin, rout, G=6, **kw)
    d = {b: step13(b, rin, rout, **kw)[0] for b in ("r15h", "r15i", "r16f")}
    d_r15i, d_nrmk, d_n1k = d["r15i"] - d["r15h"], d["r16f"] - d["r15i"], z[0] - d["r16f"]
    model = {k: z[3][k] / GHZ / 1e3 for k in WALLS}
    F = sum(WALLS[k] - model[k] for k in WALLS) / len(WALLS)  # one per-op constant, least squares
    wall_rms = (sum((WALLS[k] - model[k] - F) ** 2 for k in WALLS) / len(WALLS)) ** 0.5
    n3n2 = (model[("rms", 3)] - model[("rms", 2)], model[("ln", 3)] - model[("ln", 2)])
    terms = dict(
        need=((z[1] - 57.5) / 8) ** 2, obeg=((z[2] - 5) / 8) ** 2,
        need2=((g2[1] - 17) / 8) ** 2, obeg2=((g2[2] - 6.6) / 8) ** 2,
        walls=(wall_rms / 3) ** 2,
        r15i=((d_r15i + 0.53) / 0.06) ** 2, nrmk=((d_nrmk + 0.40) / 0.06) ** 2, n1k=((d_n1k + 0.22) / 0.06) ** 2,
        g2=((g2[0] - z[0] - 0.07) / 0.10) ** 2, g6=((g6[0] - z[0] + 0.10) / 0.08) ** 2)
    return dict(rin=rin, rout=rout, s2=s2, ddr=ddr, sc=sum(terms.values()), need=z[1], obeg=z[2], need2=g2[1], obeg2=g2[2],
                F=F, wall_rms=wall_rms, n3n2=n3n2, r15i=d_r15i, nrmk=d_nrmk, n1k=d_n1k, g2=g2[0] - z[0], g6=g6[0] - z[0],
                terms=terms, norms_ms=z[0])


def fmt13(r):
    return (f"{r['rin']:5.2f} {r['rout']:5.2f} {r['s2']:4.1f} {r['ddr']:4.1f} | {r['need']:5.0f} {r['obeg']:5.0f} | {r['need2']:5.0f} "
            f"{r['obeg2']:5.0f} | {r['n3n2'][0]:6.1f} {r['n3n2'][1]:6.1f} {r['F']:5.1f} {r['wall_rms']:5.1f} | {r['r15i']:6.2f} "
            f"{r['nrmk']:6.2f} {r['n1k']:6.2f} {r['g2']:+6.2f} {r['g6']:+6.2f} | {r['sc']:7.2f}")


HDR13 = (f"{'R_in':>5s} {'R_out':>5s} {'s2':>4s} {'ddr':>4s} | {'need':>5s} {'obeg':>5s} | {'need2':>5s} {'obeg2':>5s} | "
         f"{'n3-n2 RMS / LN us':>18s} {'F':>4s} {'wrms':>5s} | {'r15i':>6s} {'NRMK':>6s} {'N1K':>6s} {'G2':>6s} {'G6':>6s} | score")


def cmd_fit13(o):
    print("rev 13 refit. targets: NPROF r16z in_need net 57.5 / out_begin net 5 per unit; NPIPE_G=2 17 / 6.6; r16z walls RMS2 "
          "83.8 RMS3 103.1 LN2 100.6 LN3 115.4 us (n3 - n2: RMS +19.3, LN +14.8); deltas r15i -0.53, NRMK -0.40, N1K -0.22, "
          "NPIPE G2 +0.07, G6 -0.10 ms")
    rows = [score13(ri, ro, s2, dd) for ri, ro, s2, dd in itertools.product(o.rins, o.routs, o.s2s, o.ddrs)]
    rows.sort(key=lambda r: r["sc"])
    print(HDR13)
    for r in rows[:o.top]:
        print(fmt13(r))
    print("best terms:", {k: round(v, 2) for k, v in rows[0]["terms"].items()})
    print("rev-12 point (R_in 6.0, R_out 2.35, s2 1):")
    print(fmt13(score13(6.0, 2.35, 1.0)))
    print("R_out 2.35 at its best R_in / s2:")
    print(fmt13(min((r for r in rows if r["rout"] == 2.35), key=lambda r: r["sc"])) if any(r["rout"] == 2.35 for r in rows) else "-")


LEVERS13 = [  # (name, kwargs on r16h; G = NPIPE group size) - deltas against the deployed r17d norm config (NPIPE G 6)
    ("serial order (r16z)", dict()),
    ("NPIPE G 2", dict(G=2)),
    ("NPIPE G 12 (one group per row block)", dict(G=12)),
    ("FOLD: n_in 3 -> 2 for the 24 n3 ops (D1+D2, D3+D4 upstream), G 6", dict(G=6, fold=True)),
    ("FOLD on the serial order", dict(fold=True)),
    ("NKEEP: n1k state kept across nrm (group cost 45 -> 8), G 1", dict(G=1, gcost=8)),
    ("NKEEP, G 2", dict(G=2, gcost=8)),
    ("NKEEP + FOLD, G 1", dict(G=1, gcost=8, fold=True)),
    ("NG1 pass-1 glue -80 per unit (G 6)", dict(G=6, dp1=-80)),
    ("NG2 pass-2 glue -100 per unit (G 6)", dict(G=6, dp2=-100)),
    ("NS stats in asm -700 per row block (G 6)", dict(G=6, ds=-700)),
    ("NG1 + NG2 + NS (G 6)", dict(G=6, dp1=-80, dp2=-100, ds=-700)),
    ("FOLD + NG1 + NG2 + NS (G 6)", dict(G=6, fold=True, dp1=-80, dp2=-100, ds=-700)),
]


def cmd_levers13(o):
    base = step13("r16h", o.rin, o.rout, s2=o.s2, G=6, ddr=o.ddr)
    print(f"norm levers, rev-13 fit R_in {o.rin} R_out {o.rout} s2 {o.s2} ddr {o.ddr}: deployed r17d norms (NPIPE G 6) {base[0]:.2f} "
          f"ms per step + fixed; in_need {base[1]:.0f} / out_begin {base[2]:.0f} per unit")
    for name, kw in LEVERS13:
        v = step13("r16h", o.rin, o.rout, s2=o.s2, ddr=o.ddr, **kw)
        print(f"  {name:66s} {v[0] - base[0]:+.2f} ms   (in_need {v[1]:.0f} / out_begin {v[2]:.0f} per unit)")


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
    f13 = sub.add_parser("fit13")
    f13.add_argument("--rins", type=float, nargs="+", default=[3.0, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0])
    f13.add_argument("--routs", type=float, nargs="+", default=[2.35, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
    f13.add_argument("--s2s", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    f13.add_argument("--ddrs", type=float, nargs="+", default=[0.0])
    f13.add_argument("--top", type=int, default=15)
    l13 = sub.add_parser("levers13")
    l13.add_argument("--rin", type=float, default=4.25)
    l13.add_argument("--rout", type=float, default=4.0)
    l13.add_argument("--s2", type=float, default=1.0)
    l13.add_argument("--ddr", type=float, default=0.0)
    o = ap.parse_args()
    {"fit": cmd_fit, "norm": cmd_norm, "ssm": cmd_ssm, "levers": cmd_levers, "fit13": cmd_fit13,
     "levers13": cmd_levers13}[o.cmd](o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
