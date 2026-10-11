#!/usr/bin/env python3
"""collide.py - time-resolved DMA-collision model of the ROT GEMM k-step (npu-research pass 15).

ksplan's collision model (rev 13) multiplied each DMA stream's write count by the AVERAGE core busy fraction of the bank it
writes. Pass 14 priced ROT with it (raw 269 per k-step, x 0.79 = 212); round 18 measured MEMORY_STALL 167.5 around mm_rot
(KS-P38). This model replays WHEN each 32 B DMA write lands against WHICH banks the core touches in that cycle:

  * the core: the value-level replay of the emitted mm_rot (r20/gen_rot.check) gives, per static cycle of one call, the
    banks of every memory access, the cycles of the 9 conditional acquires (A slot) and releases; between calls a gap of G
    cycles (no access in banks 1-3);
  * A: one memtile -> core stream, 9 objects of 1,152 B per k-step into a 3-slot ring (object n -> slot n % 3); object n
    starts when object n - 3's slot is released (+ LAT_REL) and the previous object is done; 36 writes of 32 B at R_A B/cycle;
    the core's acquire of object n waits until it is complete (LOCK stall);
  * B (P35): the object for k-step k + 1 goes into buffer (k + 1) % 2 once k-step k - 1 released it (its end) + LAT_REL;
    144 writes at R_B; the next call waits for it;
  * a DMA write in a cycle in which the core accesses the same bank costs the core ALPHA cycles (DMAB: ~1 cycle per 32 B write
    with the core accessing that bank every cycle; ALPHA is fitted to KS-P38's 167.5 on the deployed placement).

    python3 r20/collide.py                      # calibrate on full12r18m + price the layouts in LAYOUTS
    python3 r20/collide.py --alpha 0.7 --show   # per-stream split, per-row-block collision map

Placement dicts as gen_rot's (c_obj, acc_hi, slots, rec, b, b1). Nothing here runs on the NPU.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_rot as G  # noqa: E402

GHZ, KSTEPS = 1.81, 15552
MS_PER_CYCLE = KSTEPS / GHZ / 1e6  # 1 cycle per k-step = 8.59 us per depth-12 step
A_OBJ, B_OBJ, W = 1152, 4608, 32
MEAS = dict(true=1603.2, mem=167.5, lock=13.4, timer=1613.2)  # KS-P38 around mm_rot g2a, full12r18m (round 18)
STATIC_G2A = 1383


def bank(a):
    return (a - 0x70000) >> 14


_PROF = {}


def profile(s_path, j_path, pl):
    """per-static-cycle bank accesses of one call on placement pl (+ acquire / release cycles, internal stalls)"""
    import json

    key = (str(s_path), tuple(sorted((k, str(v)) for k, v in pl.items())))
    if key in _PROF:
        return _PROF[key]

    d = json.loads(Path(j_path).read_text())
    st = G.check(Path(s_path).read_text(), d["info"], pl=pl, pad=0, order=[tuple(x) for x in d["order"]],
                 border=[tuple(x) for x in d["border"]])
    tr = [dict(x[2]) for x in st["trace"]]
    _PROF[key] = dict(trace=tr, acq=list(st["acq"]), rel=list(st["rel"]), stalls=st["stalls"], n=len(tr))
    return _PROF[key]


def simulate(profs, slots, b_bufs, alpha, gap=39, r_a=7.95, r_b=7.95, lat_rel=12, lat_acq=4, ksteps=16, warm=6,
             detail=False):
    """profs[k % 2] = the call's profile while B buffer k % 2 is popped. Returns per-k-step averages over the steady part."""
    t = 0.0
    a_ready = {}  # object -> completion time
    rel_t = {}  # object -> release time
    # A DMA state
    a_obj, a_i, a_next = 0, 0, 0.0  # object being written, its next write index, time of that write
    a_start = {}
    # B DMA state: object k (for k-step k) into buffer k % 2
    b_k, b_i, b_next = 1, 0, 0.0  # object 0 is in place before the first call
    b_ready = {0: -1e9}
    call_end = {}
    mem = dict(a=0.0, b=0.0)
    lock = 0.0
    per_k = []
    rb_map = {}

    def a_can_start(n):
        if n < 3:
            return 0.0
        return rel_t.get(n - 3, None)

    for k in range(ksteps):
        pr = profs[k % 2]
        tr, acq, rel = pr["trace"], pr["acq"], pr["rel"]
        acq_ix = {c: i for i, c in enumerate(acq)}
        rel_ix = {c: i for i, c in enumerate(rel)}
        # B for this call must be in
        k_start = t
        mem0, lock0 = dict(mem), lock
        if b_ready.get(k, 1e18) > t:
            pass
        # B acquire at the call start
        while b_ready.get(k) is None:
            # advance DMAs without the core until B k is done (start-up only)
            t += 1.0
            b_k, b_i, b_next = _b_step(t, b_k, b_i, b_next, b_ready, call_end, b_bufs, lat_rel, r_b, None, mem, alpha, k)
        if b_ready[k] > t:
            lock += b_ready[k] - t
            t = b_ready[k]
        for c in range(pr["n"]):
            # acquire of A object 9k + i (the first row block of pair i)
            if c in acq_ix:
                n = 9 * k + acq_ix[c]
                # make sure the DMA progressed to it: run DMAs forward while waiting
                while a_ready.get(n) is None:
                    a_obj, a_i, a_next = _a_step(t, t + 1.0, a_obj, a_i, a_next, a_ready, rel_t, a_start, slots, r_a, lat_rel,
                                                 None, mem, alpha)
                    b_k, b_i, b_next = _b_step(t + 1.0, b_k, b_i, b_next, b_ready, call_end, b_bufs, lat_rel, r_b, None,
                                               mem, alpha, k)
                    lock += 1.0
                    t += 1.0
                if a_ready[n] + lat_acq > t:
                    lock += a_ready[n] + lat_acq - t
                    t = a_ready[n] + lat_acq
            acc = tr[c]
            t_end = t + 1.0
            # DMA writes inside [t, t_end): each colliding write costs alpha
            hit_a = []
            a_obj, a_i, a_next = _a_step(t, t_end, a_obj, a_i, a_next, a_ready, rel_t, a_start, slots, r_a, lat_rel, acc, hit_a,
                                         alpha)
            hit_b = []
            b_k, b_i, b_next = _b_step(t_end, b_k, b_i, b_next, b_ready, call_end, b_bufs, lat_rel, r_b, acc, hit_b, alpha, k,
                                       t0=t)
            st = alpha * (len(hit_a) + len(hit_b))
            mem["a"] += alpha * len(hit_a)
            mem["b"] += alpha * len(hit_b)
            if detail and k >= warm and (hit_a or hit_b):
                rb_map.setdefault(c, [0, 0])
                rb_map[c][0] += len(hit_a)
                rb_map[c][1] += len(hit_b)
            t = t_end + st
            if c in rel_ix:
                rel_t[9 * k + rel_ix[c]] = t
        t += pr["stalls"] * 0  # internal stalls are added statically by the caller
        call_end[k] = t
        t += gap
        per_k.append(dict(dur=t - k_start, mem_a=mem["a"] - mem0["a"], mem_b=mem["b"] - mem0["b"], lock=lock - lock0))
    steady = per_k[warm:]
    avg = {key: sum(x[key] for x in steady) / len(steady) for key in steady[0]}
    if detail:
        avg["map"] = rb_map
    return avg


def _a_step(t0, t1, obj, i, nxt, ready, rel_t, start, slots, r_a, lat_rel, acc, hits, alpha):
    """emit the A writes with time in [t0, t1); acc = the core's banks this cycle (None: no core access)"""
    while True:
        if i == 0:
            s = 0.0 if obj < 3 else (rel_t.get(obj - 3) + lat_rel if obj - 3 in rel_t else None)
            if s is None:
                return obj, i, nxt
            nxt = max(nxt, s)
            start[obj] = nxt
        if nxt >= t1:
            return obj, i, nxt
        if acc is not None and nxt >= t0:
            b = bank(slots[obj % 3] + W * i)
            if acc.get(b):
                hits.append(b)
        i += 1
        nxt += W / r_a
        if i == A_OBJ // W:
            ready[obj] = nxt
            obj, i = obj + 1, 0


def _b_step(t1, k, i, nxt, ready, call_end, bufs, lat_rel, r_b, acc, hits, alpha, kcur, t0=None):
    """B object k into buffer k % 2: starts when k-step k - 2 (the buffer's previous user) ended (+ lat_rel)"""
    while True:
        if i == 0:
            if k <= 1:
                s = 0.0
            elif (k - 2) in call_end:
                s = call_end[k - 2] + lat_rel
            else:
                return k, i, nxt
            nxt = max(nxt, s)
        if nxt >= t1:
            return k, i, nxt
        if acc is not None and t0 is not None and nxt >= t0:
            b = bank(bufs[k % 2] + W * i)
            if acc.get(b):
                hits.append(b)
        i += 1
        nxt += W / r_b
        if i == B_OBJ // W:
            ready[k] = nxt
            k, i = k + 1, 0


PHASES = [(lr, g) for lr in (8, 14, 20, 26) for g in (35, 39, 43)]  # DMA start latency x gap between calls: the phase of the
# DMA writes against the core's access pattern (a single phase is +-10 cycles of noise; the mean over these 12 is stable)


def price(name, s_path, j_path, pl, alpha, show=False, phases=True, **kw):
    b0, b1 = pl["b"], pl.get("b1", pl["b"] + 4608)
    profs = [profile(s_path, j_path, dict(pl, b=b0)), profile(s_path, j_path, dict(pl, b=b1))]
    if phases:
        rs = []
        for lr, g in PHASES:
            k2 = dict(kw, lat_rel=lr, gap=g)
            rs.append(simulate(profs, pl["slots"], (b0, b1), alpha, **k2))
        r = {k: sum(x[k] for x in rs) / len(rs) for k in ("dur", "mem_a", "mem_b", "lock")}
        tot = [x["mem_a"] + x["mem_b"] for x in rs]
        r["spread"] = max(tot) - min(tot)
    else:
        r = simulate(profs, pl["slots"], (b0, b1), alpha, detail=show, **kw)
    internal = (profs[0]["stalls"] + profs[1]["stalls"]) / 2
    r["internal"] = internal
    r["static"] = profs[0]["n"]
    return r


# full12r18m and the candidate layouts (see PROPOSALS rev 15, section "GEMM L1 layout")
R18M = dict(G.PLACE_R18M)
LAYOUTS = {
    "r18m (deployed, slots b1 b2 b2)": R18M,
    "slots b1 b2 b0 (scan best)": dict(R18M, slots=(0x77B80, 0x78000, 0x72C00)),
    "slots b0 b1 b2": dict(R18M, slots=(0x72C00, 0x77B80, 0x78000)),
    "all slots b2": dict(R18M, slots=(0x78000, 0x78480, 0x7AC00)),
    "bound: A slots in a private bank": dict(R18M, slots=(0x80000, 0x80480, 0x80900)),
}


def piece_banks(pl):
    """banks of (pair 0, pair 1) per row block (a straddling piece lists both banks)"""
    out = []
    for rb in range(G.NRB):
        bs = []
        for pair in (0, 1):
            a = G.piece(rb, pair, pl)
            bs.append("".join(sorted({str(bank(a)), str(bank(a + 1023))})))
        out.append("/".join(bs))
    return out


def _scan_one(args):
    s, j, pl, alpha, kw = args
    r = price("", s, j, pl, alpha, **kw)
    return r


def scan(s, j, alpha, kw, procs=4):
    """every assignment of the 3 A slots to banks 0-2, on several piece placements (acc_hi / C object bases)"""
    import itertools
    from multiprocessing import Pool

    pieces = {"deployed": dict(acc_hi=0x71980, c_obj=0x78900), "C@0x79000": dict(acc_hi=0x71980, c_obj=0x79000),
              "acc_hi@0x71000": dict(acc_hi=0x71000, c_obj=0x78900), "acc_hi@0x71000 C@0x79000": dict(acc_hi=0x71000, c_obj=0x79000)}
    jobs, names = [], []
    for pn, pc in pieces.items():
        for sb in itertools.product((0, 1, 2), repeat=3):
            slots = tuple(0x70000 + 0x4000 * b + 0x2C00 + 0x480 * i for i, b in enumerate(sb))
            pl = dict(R18M, slots=slots, **pc)
            jobs.append((s, j, pl, alpha, kw))
            names.append((pn, sb, pl))
    with Pool(procs) as pool:
        res = pool.map(_scan_one, jobs)
    rows = []
    for (pn, sb, pl), r in zip(names, res):
        rows.append((r["mem_a"] + r["mem_b"] + r["internal"], pn, sb, r))
    for pn, pc in pieces.items():
        print(f"pieces {pn}: rb -> banks (pair0/pair1): {' '.join(piece_banks(dict(R18M, **pc)))}")
    rows.sort(key=lambda x: x[0])
    for mem, pn, sb, r in rows:
        print(f"{pn:26s} slots in banks {sb}: internal {r['internal']:5.1f} A {r['mem_a']:6.1f} B {r['mem_b']:6.1f} = {mem:6.1f} "
              f"LOCK {r['lock']:4.1f} k-step {r['dur']:7.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s", default=str(HERE / "mm_rot_g2a.s"))
    ap.add_argument("--json", default=str(HERE / "rot_g2a.json"))
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--lat-rel", type=float, default=12)
    ap.add_argument("--r-a", type=float, default=7.95)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--scan", action="store_true", help="slot banks x piece placements")
    o = ap.parse_args()
    kw = dict(r_a=o.r_a, r_b=o.r_a)
    if o.alpha is None:
        lo, hi = 0.3, 1.2
        for _ in range(12):
            mid = (lo + hi) / 2
            r = price("cal", o.s, o.json, R18M, mid, **kw)
            (lo, hi) = (mid, hi) if r["mem_a"] + r["mem_b"] < MEAS["mem"] else (lo, mid)
        alpha = (lo + hi) / 2
        print(f"calibrated alpha = {alpha:.3f} (MEMORY_STALL {MEAS['mem']} on full12r18m)")
    else:
        alpha = o.alpha
    if o.scan:
        scan(o.s, o.json, alpha, kw)
        return
    for name, pl in LAYOUTS.items():
        r = price(name, o.s, o.json, pl, alpha, show=o.show, **kw)
        mem = r["mem_a"] + r["mem_b"] + r["internal"]
        print(f"{name:34s} static {r['static']} internal {r['internal']:5.1f} MEM A {r['mem_a']:6.1f} B {r['mem_b']:6.1f} "
              f"= {mem:6.1f} (phase spread {r.get('spread', 0):4.1f})  LOCK {r['lock']:5.1f}  k-step {r['dur']:7.1f}")


if __name__ == "__main__":
    main()
