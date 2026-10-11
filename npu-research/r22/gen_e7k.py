#!/usr/bin/env python3
"""gen_e7k.py - KTL: E7'' as ONE modulo loop over all of a tile's k-steps (research pass 17).

Today mm_ktile calls the k-step kernel once per k-step: every call pays setup + prologue + ret (61 cycles on ROT g0a, ~100 on
E7'' R 68) plus 13 of loop glue, and the pipeline drains at every k-step end. KTL keeps the modulo loop running across k-steps:

  * the schedule comes from `gen_e7.py solve --ktl` (implies --aligned: every stream segment instance serves ONE row block, since
    consecutive k-steps read different B buffers; lf1 is split L | N so that no lf1 instance straddles two row blocks);
  * records wrap per k-step: the body's record loads are `lda.2d rX, [p2], d2` (dn2 = 6 x 18 - 1, dj2 = 4, m2 = -4 dn2); the prologue's
    are plain post-increment loads of a separate head (rb 0's stage -1 fields), so the table is [head][one k-step];
  * the B buffer of every stream segment is a base register (Y r16, L r17, N r18, X r19) toggled by `xor` with (B0 ^ B1) in the
    row block whose record carries the B flag (rb 17); restarts are `movs p0/p1, rBase`;
  * B locks inside the kernel: `rel.cond #48` (B prod) after the last pop of every instance serving rb 17, `acq.cond #49` (B cons)
    >= 6 bundles before the first fill of every instance serving the next rb 0;
  * the trip after the last (ROT's dummy row block) must not acquire: r27 counts the remaining row blocks; the A acquire condition
    is r26 x r27 and the B acquire's (r27 - 1) x flag (no B beyond the tile). acq.cond / rel.cond take their condition in r26
    only: the four r26 groups (A acquire, A release, B acquire, B release) are disjoint in the schedule.

mm_e7k(const int32_t *rec, float *pc, int32_t cfg, int32_t trips, int32_t y0, int32_t x0): p0 = the record table, p1 = pc' =
piece(0, 0) - J0(continue) (so rb 0's jump lands on piece(0, 0)), r0 = 781 (ZLOAD: k-step 0's kp-0 MACs zero-load; frame 17's
`and r6, r6, ~flag` turns it into 780 for the rest of the tile), r1 = 18 x nks, r2 / r3 = k-step 0's Y / X buffers. The caller
acquires B(0) before the call; the kernel acquires B(1..nks-1) and releases every B itself.

usage:  python3 r22/gen_e7k.py emit r22/e7k_r70_n16.json --out r22/mm_e7k_r70.s       (+ .h, replay of 3 k-steps)
        docker ... npu-iron python3 r22/gen_e7k.py fit r22/e7k_r70_n16.json --out r22/mm_e7k_r70.s
        python3 r22/gen_e7k.py check r22/mm_e7k_r70.s --json r22/e7k_r70_n16.json [--nks 5]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_e7 as E  # noqa: E402
from gen_e7 import (ACQ_GAP, AREG, BREG, NCB, NK, NQ, NRB, PLACE, PTR_GAP, QN, REL_GAP, SCL, Fail, bank, caddr, cfg_of,  # noqa: E402
                    fifo_names, layout, piece, sname)
from gen_rot import a_ptr, F_PTR, F_J0, F_J1, F_ACQ, F_REL  # noqa: E402

F_B = 5
FIELDS = 6
BASE = dict(Y="r16", L="r17", N="r18", X="r19")
DELTA = dict(Y="r20", L="r21", N="r22", X="r23")
TEMP = dict(L="r0", X="r1", Y="r2", N="r7")


def record(r, f, pl=PLACE):
    """record field f of row block r (any k-step: r mod 18); rb 0's jump is the 'continue from rb 17' value"""
    r %= NRB
    if f == F_PTR:
        return a_ptr(r, pl)
    if f == F_J0:
        return piece(r, 0, pl) - piece((r - 1) % NRB, 1, pl) - 1024
    if f == F_J1:
        return piece(r, 1, pl) - piece(r, 0, pl) - 1024
    if f == F_ACQ:
        return int(r % 2 == 0)
    if f == F_REL:
        return int(r % 2 == 1)
    return int(r == NRB - 1)


def bases(cfg, pl, buf):
    y = pl["y"][buf]
    return dict(Y=y, L=y + 72 * cfg.ny, N=y + 72 * (cfg.ny + cfg.n1 - cfg.n1b), X=pl["x"][buf])


def all_ops(times, cfg, pr, prologue):
    """(time, slot, name, text, record field) of row block 0; FIFO ops as gen_e7 (slot 'fifo')"""
    out = []
    for t, sl, n, txt, fld, fifo, ispop in E.all_ops(times, cfg, pr):
        if n in ("XP", "YP", "BP", "NP"):
            seg = {"XP": "X", "YP": "Y", "BP": "L", "NP": "N"}[n]
            txt, sl = f"movs p{0 if seg in 'XY' else 1}, {BASE[seg]}", "st"
        elif n == "ACQ":
            txt = "acq.cond #53, r4, r26"
        elif fld is not None:
            dst = txt.split()[1].rstrip(",")
            txt = f"lda {dst}, [p2], #4" if prologue(t) else f"lda.2d {dst}, [p2], d2"
        out.append((t, sl, n, txt, fld, fifo, ispop))
    T = times
    x = dict(CNT=("alu", "add r27, r27, #-1"), MA=("alu", "mul r26, r26, r27"), MB=("alu", "add r30, r27, #-1"),
             MB2=("alu", "mul r26, r29, r30"), CB=("mv", "mov r26, r29"), ACQB=("alu", "acq.cond #49, r4, r26"),
             RELB=("alu", "rel.cond #48, r5, r26"), ZX=("alu", "xor r31, r29, r4"), ZA=("alu", "and r6, r6, r31"))
    for n, (sl, txt) in x.items():
        out.append((T[n], sl, n, txt, None, None, False))
    t = T["RQ2"]
    out.append((t, "lda", "RQ2", "lda r29, [p2], #4" if prologue(t) else "lda.2d r29, [p2], d2", F_B, None, False))
    for sg in cfg.segs:
        if sg == "A":
            continue
        out.append((T[f"TM{sg}"], "alu", f"TM{sg}", f"mul {TEMP[sg]}, r29, {DELTA[sg]}", None, None, False))
        out.append((T[f"TX{sg}"], "alu", f"TX{sg}", f"xor {BASE[sg]}, {BASE[sg]}, {TEMP[sg]}", None, None, False))
    return out


def emit(times, info, name="mm_e7k", full_pro=frozenset(), pl=PLACE):
    cfg = cfg_of(info)
    assert cfg.ktl and cfg.n1b, "KTL needs a --ktl schedule with the lf1 split"
    R, F = cfg.R, info["F"]
    pr, off, order_b = layout(times, cfg)
    ops = all_ops(times, cfg, pr, lambda t: t < F)
    body = defaultdict(list)
    for t, sl, n, txt, fld, fifo, ispop in ops:
        stage = (t - F) // R
        assert stage in (-1, 0), (n, t, F)
        if fld is not None and t < F:  # the body's copy of a stage -1 record load is the 2D form
            txt = txt.replace(", #4", ", d2").replace("lda ", "lda.2d ")
        body[(t - F) % R].append((sl, txt, fifo, txt if sl == "fifo" else None, stage, fld))
    # prologue: rb 0's stage -1 ops + the previous period's instances of segments with delta -1 (they serve rb 0)
    fn = {nm: (fifo, seg, i, o) for nm, fifo, seg, i, o in fifo_names(cfg)}
    need = {}
    for sg, d in cfg.delta.items():
        if sg in cfg.segs and d == -1:
            r_ = {"L": ("BR", "BP"), "N": ("NR", "NP")}.get(sg, (sg + "R", sg + "P"))
            need[sg] = min([times[sname(sg, 0)]] + [times[x] for x in r_]) - R
    pro = defaultdict(list)
    for t, sl, n, txt, fld, fifo, ispop in ops:
        if t < F:
            pro[t].append((sl, txt, fifo, txt if sl == "fifo" else None, fld))
    # every op of that FIFO from the needed instance's restart on (incl. the restart of the next segment, e.g. L after N),
    # except the A segment (rb -1's A belongs to no one)
    fifo_of = {"NR": "lf1", "NP": "lf1", "BR": "lf1", "BP": "lf1", "XR": "lf0", "XP": "lf0", "YR": "lf0", "YP": "lf0"}
    needf = {}
    for sg, t0 in need.items():
        f = "lf1" if sg in "LN" else "lf0"
        needf[f] = min(needf.get(f, t0), t0)
    for t, sl, n, txt, fld, fifo, ispop in ops:
        f = fifo if n in fn and fn[n][1] != "A" else fifo_of.get(n)
        if f in needf and needf[f] <= t - R < F:
            pro[t - R].append((sl, txt, fifo, txt if sl == "fifo" else None, None))
    E_ = min(list(pro) + [F])
    b1 = bases(cfg, pl, 0)
    b2 = bases(cfg, pl, 1)
    nrec = FIELDS * NRB
    lines = ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    setup = ["mov r6, r0", "mov r27, r1", "mov r16, r2", "mov r19, r3", f"movxm r20, #{72 * cfg.ny}",
             f"movxm r21, #{72 * (cfg.ny + cfg.n1 - cfg.n1b)}", "nop", "add r17, r16, r20", "add r18, r16, r21",
             f"movxm r20, #{b1['Y'] ^ b2['Y']}", f"movxm r21, #{b1['L'] ^ b2['L']}", f"movxm r22, #{b1['N'] ^ b2['N']}",
             f"movxm r23, #{b1['X'] ^ b2['X']}", "movs p3, p1", "movs p4, p1", "movs p2, p0", "mova r3, #780", "mova r25, #0",
             "mova r24, #0", "mova r4, #-1", "mova r5, #1", f"movxm m2, #{-4 * (nrec - 1)}", f"mova dn2, #{nrec - 1}",
             "mova dj2, #4", "mova dc2, #0", f"movxm ls, #.L{name}_body", f"movxm le, #.L{name}_end", "add.nc lc, r27, #0"]
    lines += ["\t" + s for s in setup]
    lines += ["\t.p2align\t4", f".L{name}_pro:"]
    head = []
    for i, a in enumerate(range(E_, F)):
        lines.append("\t" + E.bundle([(sl, txt, fifo, reg) for sl, txt, fifo, reg, _ in pro.get(a, [])], i in full_pro))
        for sl, txt, fifo, reg, fld in pro.get(a, []):
            if fld is not None:
                head.append((0, fld))
    lines += ["\t.p2align\t4", f".L{name}_body:", f"{name}_start:"]
    for u in range(R):
        if u == R - 1:
            lines += ["\t.p2align\t4", f".L{name}_end:", f"{name}_last:"]
        lines.append("\t" + E.bundle([(sl, txt, fifo, reg) for sl, txt, fifo, reg, _, _ in body[u]], True))
    period = []
    for f in range(NRB):
        for u in range(R):
            for sl, txt, fifo, reg, stage, fld in body[u]:
                if fld is not None:
                    period.append(((f - stage) % NRB, fld))
    assert len(period) == nrec, (len(period), nrec)
    lines += ["\tret lr"] + ["\tnop"] * 5 + [f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return "\n".join(lines) + "\n", head, period, order_b


def table(head, period, pl=PLACE):
    return [record(r, f, pl) for r, f in head] + [record(r, f, pl) for r, f in period]


def header(head, period, order_b, cfg, name="mm_e7k"):
    nm = name.upper()
    return "\n".join([
        f"// generated by npu-research/r22/gen_e7k.py (KTL, R = {cfg.R}, lf1 {cfg.n1 - cfg.n1b} + {cfg.n1b} / lf0-X {cfg.nx} / "
        f"lf0-Y {cfg.ny} panels): record table = head (the prologue's loads) + one k-step (the body's lda.2d wraps there)",
        "// fields: 0 = A pointer, 1 = jump into (rb, pair 0) (rb 0: from rb 17), 2 = jump into (rb, pair 1), 3 = A acquire flag, "
        "4 = A release flag, 5 = B flag (rb 17: release B(j), acquire B(j + 1), toggle the buffer bases)",
        f"#define {nm}_NHEAD {len(head)}",
        f"#define {nm}_NPER {len(period)}",
        f"#define {nm}_YB {cfg.yb}",
        f"static constexpr int8_t {name}_order[{nm}_NHEAD + {nm}_NPER][2] = {{" +
        ", ".join(f"{{{a}, {b}}}" for a, b in head + period) + "};",
        f"static constexpr int8_t {name}_border[64] = {{" +
        ", ".join(str((cb >> 2) * 32 + k * 4 + (cb & 3)) for cb, k in order_b) + "};",
    ]) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
def parse_s(text, name="mm_e7k"):
    pre, body, tail = [], [], []
    cur = pre
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("//", "#")):
            continue
        if s == f".L{name}_body:":
            cur = body
            continue
        if s.startswith(".") or s.endswith(":"):
            continue
        if cur is body and s.startswith("ret"):
            cur = tail
        ins = [i.strip() for i in s.split(";") if i.strip() and not re.fullmatch(r"nop[abxmvs]?", i.strip())]
        cur.append(ins)
    return pre, body, tail


def check(text, info, head, period, order_b, nks=4, pl=PLACE, pad=0, name="mm_e7k", verbose=False):
    """replay one KTL call over a whole tile (k-steps 0..nks-1; k-step 0 zero-loads its accumulators: cfg 781 in r6 until
    frame 17's `and r6, r6, ~flag`)"""
    cfg = cfg_of(info)
    pre, body, tail = parse_s(text, name)
    trips = NRB * nks
    seq = [(b, -1) for b in pre] + [([], -1)] * pad + [(b, f) for f in range(trips) for b in body] + [(b, trips) for b in tail]
    mem = {}
    for rb in range(NRB):
        for cb in range(NCB):
            for q in range(NQ):
                mem[caddr(rb, cb, q, pl)] = ("acc", rb, cb, q, 0)  # whatever the pieces hold: ZLOAD ignores it
    cq = set(mem)
    tab = table(head, period, pl)
    rec0 = pl["rec"]
    regs = defaultdict(list)
    errs, notes = [], []
    st = dict(macs=0, stalls=0, stall_where=defaultdict(int), acq=[], rel=[], slot_owner={}, last_read=defaultdict(int),
              stores=0, busy=defaultdict(int), body_cycles=0, bacq=[], brel=[])
    fifo = {}
    B = {b: bases(cfg, pl, b) for b in (0, 1)}
    # buffer state: k-step 0 in buffer 0 (acquired by the caller), k-step 1 already in buffer 1
    buf = {0: dict(k=0, held=True), 1: dict(k=1, held=False)}
    held_q = [0]  # acquisition order
    next_acq = dict(k=1)
    pc0 = piece(0, 0, pl) - record(0, F_J0, pl)
    init = {"r0": 781, "r1": trips, "r2": B[0]["Y"], "r3": B[0]["X"], "p0": rec0, "p1": pc0}
    for r, v in init.items():
        regs[r].append((-100, 1, 1, v))

    def err(c, msg):
        errs.append(f"cycle {c}: {msg}")
        if len(errs) > 30:
            raise Fail("\n".join(notes[:10] + errs))

    def wr(c, r, v, wmin, wmax=None):
        regs[r].append((c, wmin, wmax if wmax is not None else wmin, v))

    def rd(c, r, stage=1, ok_tie=False):
        best, bl, unsure = None, None, []
        for ci, wmin, wmax, v in regs[r][-12:]:
            if ci + wmax <= c + stage - 1:
                land = ci + wmax
                if bl is None or land > bl or (land == bl and ci >= best[0]):
                    if bl is not None and land == bl and best[3] != v and not ok_tie:
                        err(c, f"{r}: two writes land together ({best[3]} / {v})")
                    best, bl = (ci, wmin, wmax, v), land
            elif ci + wmin <= c + stage - 1:
                unsure.append((ci, v))
        if unsure:
            err(c, f"{r} read while a write issued at {unsure[0][0]} may or may not have landed")
        if best is None:
            err(c, f"{r} read before any write")
            return None
        return best[3]

    def a_val(c, addr):
        for j, s0 in enumerate(pl["slots"]):
            if s0 <= addr < s0 + 1152:
                o = addr - s0
                if o % 72:
                    err(c, f"A pop at slot {j} + {o}")
                own = st["slot_owner"].get(j)
                if own is None or own[1]:
                    return "G"
                st["last_read"][own[0]] = c
                i = o // 72
                s = own[0]
                return ("A", 2 * (s % 9) + i // 8, i % 8, s // 9)
        return None

    def b_val(c, addr):
        for b in (0, 1):
            y, x = pl["y"][b], pl["x"][b]
            for base, lo, n in ((y, 0, cfg.yb), (x, cfg.yb, 4608 - cfg.yb)):
                if base <= addr < base + n:
                    o = addr - base
                    if o % 72:
                        err(c, f"B pop at buffer {b} + {o}")
                        return "G"
                    if not buf[b]["held"]:  # only the dummy trip may (its pops feed no counted MAC: the MAC check catches the rest)
                        st["unheld_pops"] = st.get("unheld_pops", 0) + 1
                    buf[b]["last"] = c
                    cb, k = order_b[(lo + o) // 72]
                    return ("B", cb, k, buf[b]["k"])
        return None

    def buffer_of(addr):
        for b in (0, 1):
            if pl["y"][b] <= addr < pl["y"][b] + cfg.yb or pl["x"][b] <= addr < pl["x"][b] + 4608 - cfg.yb:
                return b
        return None

    def stream(c, lf, ptr, pos, is_pop):
        p = rd(c, ptr)
        ps = rd(c, pos, ok_tie=True)
        if p is None:
            return None, p
        if ps == "Z":
            if is_pop:
                err(c, f"pop on {lf} straight after a restart (no fill)")
                return None, p
            fifo[lf] = dict(base=p, lines=1, C=0)
            val = None
        else:
            s = fifo.get(lf)
            if s is None:
                err(c, f"{lf} used before a fill")
                return None, p
            if p != s["base"] + 64 * s["lines"]:
                err(c, f"{lf}: pointer {p:#x} != stream line {s['base'] + 64 * s['lines']:#x}")
            s["lines"] += 1
            val = None
            if is_pop:
                if 64 * s["lines"] < s["C"] + 72:
                    err(c, f"{lf}: FIFO underflow")
                val = s["base"] + s["C"]
                s["C"] += 72
        wr(c, ptr, p + 64, 1)
        wr(c, pos, "S", 1)
        return val, p

    def lda(c, dst, two_d):
        p = rd(c, "p2")
        i = (p - rec0) // 4
        v = tab[i] if 0 <= i < len(tab) else None
        if v is None:
            err(c, f"record {i} outside the table")
            v = 0
        if two_d:
            dc, dn = rd(c, "dc2"), rd(c, "dn2")
            if dc < dn:
                wr(c, "p2", p + rd(c, "dj2"), 1)
                wr(c, "dc2", dc + 1, 1)
            else:
                wr(c, "p2", p + rd(c, "m2"), 1)
                wr(c, "dc2", 0, 1)
        else:
            wr(c, "p2", p + 4, 1)
        return v, p

    c = -1
    for bund, frame in seq:
        c += 1
        acc_banks = []
        writes = []
        for ins in bund:
            mm = None
            if (mm := re.fullmatch(r"mov (r\d+), (r\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)), 1, 1))
            elif (mm := re.fullmatch(r"movs (p\d), ([pr]\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)), 1, 2))
            elif (mm := re.fullmatch(r"mova (\w+), #(-?\d+)", ins)):
                v = int(mm.group(2))
                writes.append((mm.group(1), "Z" if mm.group(1) in ("r24", "r25") and v == 0 else v, 1, 1))
            elif (mm := re.fullmatch(r"movxm (\w+), #(\S+)", ins)):
                v = mm.group(2)
                writes.append((mm.group(1), int(v) if re.fullmatch(r"-?\d+", v) else v, 1, 1))
            elif ins.startswith("add.nc lc"):
                if rd(c, "r27") != trips:
                    err(c, "lc != trips")
            elif (mm := re.fullmatch(r"mov (r2[45]), #0", ins)):
                writes.append((mm.group(1), "Z", 1, 1))
            elif (mm := re.fullmatch(r"add (r\d+), (r\d+), #(-?\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)) + int(mm.group(3)), 1, 1))
            elif (mm := re.fullmatch(r"add (r\d+), (r\d+), (r\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)) + rd(c, mm.group(3)), 1, 1))
            elif (mm := re.fullmatch(r"mul (r\d+), (r\d+), (r\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)) * rd(c, mm.group(3)), 2, 2))
            elif (mm := re.fullmatch(r"and (r\d+), (r\d+), (r\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)) & rd(c, mm.group(3)), 1, 1))
            elif (mm := re.fullmatch(r"xor (r\d+), (r\d+), (r\d+)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)) ^ rd(c, mm.group(3)), 1, 1))
            elif (mm := re.fullmatch(r"vld([ab])\.fill\.512 \[(p[01]), (lf[01]), (r2[45])\]", ins)):
                _, p = stream(c, mm.group(3), mm.group(2), mm.group(4), False)
                acc_banks.append(bank(p))
                if p is not None and mm.group(3) == "lf0" and a_val(c, p) is not None:
                    pa = st.pop("pending_acq", None)
                    if pa is not None and c - pa < ACQ_GAP:
                        err(c, f"A fill {c - pa} cycle(s) after its acquire")
                b = buffer_of(p) if p is not None else None
                if b is not None and buf[b].get("fresh") is not None:
                    if c - buf[b]["fresh"] < ACQ_GAP:
                        err(c, f"B fill {c - buf[b]['fresh']} cycle(s) after the B acquire")
                    buf[b]["fresh"] = None
            elif (mm := re.fullmatch(r"vld([ab])\.pop\.576 ex(\d+), \[(p[01]), (lf[01]), (r2[45])\]", ins)):
                a, p = stream(c, mm.group(4), mm.group(3), mm.group(5), True)
                acc_banks.append(bank(p))
                v = "G"
                if a is not None:
                    v = a_val(c, a)
                    if v is None:
                        v = b_val(c, a)
                    if v is None:
                        err(c, f"pop from {a:#x}")
                        v = "G"
                writes.append((f"ex{mm.group(2)}", v, 8, 8))
            elif (mm := re.fullmatch(r"vmac\.f dm(\d), dm\1, ex(\d+), ex(\d+), (r[36])", ins)):
                d = mm.group(1)
                accs = [rd(c, f"dm{d}.{q}", 4) for q in range(NQ)]
                av, bv = rd(c, f"ex{mm.group(2)}"), rd(c, f"ex{mm.group(3)}")
                cfgv = rd(c, mm.group(4))
                rb = cb = n = None
                if cfgv == 781:  # zero_acc: the accumulator input is ignored; only k-step 0's kp-0 MACs may run like this
                    ok = isinstance(av, tuple) and isinstance(bv, tuple) and av[2] == 0 and av[3] == 0 and bv[2:] == (0, 0)
                    if ok:
                        rb, cb, n = av[1], bv[1], 0
                        ok = all(isinstance(x, tuple) and x[0] == "acc" and x[1:3] == (rb, cb) for x in accs) or True
                    elif 0 <= frame < trips and isinstance(av, tuple):
                        err(c, f"cfg 781 on a MAC that must accumulate ({av}, {bv})")
                else:
                    if cfgv != 780:
                        err(c, f"{mm.group(4)} = {cfgv}")
                    ok = all(isinstance(x, tuple) and x[0] == "acc" for x in accs) and len({x[1:3] + x[4:] for x in accs}) == 1 \
                        and [x[3] for x in accs] == [0, 1, 2, 3]
                    if ok:
                        _, rb, cb, _, n = accs[0]
                        ok = n > 0 and av == ("A", rb, n % NK, n // NK) and bv == ("B", cb, n % NK, n // NK)
                if ok:
                    st["macs"] += 1
                elif verbose and 0 <= frame < trips and all(isinstance(x, tuple) for x in accs):
                    notes.append(f"cycle {c} frame {frame}: {ins}: acc {accs[0]}, A {av}, B {bv}")
                for q in range(NQ):
                    writes.append((f"dm{d}.{q}", ("acc", rb, cb, q, n + 1) if ok else "G", 6, 6))
            elif (mm := re.fullmatch(r"vlda (bm\w\w)(\d), \[p3\], #64", ins)):
                p = rd(c, "p3")
                acc_banks.append(bank(p))
                writes.append((f"dm{mm.group(2)}.{QN.index(mm.group(1))}", mem.get(p, "G"), 7, 7))
                writes.append(("p3", p + 64, 1, 1))
            elif (mm := re.fullmatch(r"vst (bm\w\w)(\d), \[p4\], #64", ins)):
                p = rd(c, "p4")
                acc_banks.append(bank(p))
                v = rd(c, f"dm{mm.group(2)}.{QN.index(mm.group(1))}")
                if p not in cq:
                    err(c, f"stray store to {p:#x} ({v})")
                else:
                    want = mem[p][:4] + ((mem[p][4] // NK + 1) * NK,)
                    if v != want:
                        err(c, f"store to {p:#x}: {v}, want {want}")
                    mem[p] = v
                    st["stores"] += 1
                writes.append(("p4", p + 64, 1, 1))
            elif (mm := re.fullmatch(r"padds \[(p[34])\], (m\d)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(1)) + rd(c, mm.group(2)), 1, PTR_GAP))
            elif (mm := re.fullmatch(r"lda(\.2d)? (\w+), \[p2\], (#4|d2)", ins)):
                v, p = lda(c, mm.group(2), bool(mm.group(1)))
                acc_banks.append(bank(p))
                writes.append((mm.group(2), v, 1, SCL))
                st.setdefault("scalar_at", []).append(c)
            elif ins == "acq.cond #53, r4, r26":
                if rd(c, "r26"):
                    s = len(st["acq"])
                    st["acq"].append(c)
                    st["pending_acq"] = c
                    st["slot_owner"][s % 3] = (s, False)
            elif ins == "rel.cond #52, r5, r26":
                if rd(c, "r26"):
                    s = len(st["rel"])
                    st["rel"].append(c)
                    if st["slot_owner"].get(s % 3, (None,))[0] != s:
                        err(c, f"release of stripe {s}: slot holds {st['slot_owner'].get(s % 3)}")
                    else:
                        st["slot_owner"][s % 3] = (s, True)
                    if c < st["last_read"].get(s, -99) + REL_GAP:
                        err(c, f"release of stripe {s} {c - st['last_read'][s]} cycles after its last A read")
            elif ins == "acq.cond #49, r4, r26":
                if rd(c, "r26"):
                    k = next_acq["k"]
                    b = k % 2
                    if buf[b]["held"] or buf[b]["k"] != k:
                        err(c, f"B acquire of k-step {k}: buffer {b} holds {buf[b]}")
                    buf[b].update(held=True, k=k, fresh=c)
                    held_q.append(b)
                    next_acq["k"] = k + 1
                    st["bacq"].append(c)
            elif ins == "rel.cond #48, r5, r26":
                if rd(c, "r26"):
                    if not held_q:
                        err(c, "B release with nothing held")
                    else:
                        b = held_q.pop(0)
                        if c < buf[b].get("last", -99) + REL_GAP:
                            err(c, f"B release {c - buf[b]['last']} cycles after the last pop of buffer {b}")
                        buf[b].update(held=False, k=buf[b]["k"] + 2)  # the DMA refills it with k-step + 2 at once (worst case)
                    st["brel"].append(c)
            elif ins.startswith("ret") or ins == "nop":
                pass
            else:
                raise Fail(f"unknown instruction {ins!r}")
        for r, v, wmin, wmax in writes:
            wr(c, r, v, wmin, wmax)
        cnt = defaultdict(int)
        for bk in acc_banks:
            cnt[bk] += 1
        if 0 <= frame < trips:
            st["body_cycles"] += 1
            for bk in cnt:
                st["busy"][bk] += 1
        sl = max(cnt.values(), default=1) - 1
        if sl > 0:
            st["stalls"] += sl
            st["stall_where"]["prologue" if frame < 0 else "body" if frame < trips else "tail"] += sl
    bad = [(a, v) for a, v in mem.items() if not (isinstance(v, tuple) and v[4] == NK * nks)]
    if bad:
        err(c, f"{len(bad)} quarters not finished, e.g. {bad[0][0]:#x}: {bad[0][1]}")
    want_macs = NRB * NCB * NK * nks
    if st["macs"] != want_macs:
        err(c, f"{st['macs']} valid MACs, want {want_macs}")
    if len(st["acq"]) != 9 * nks or len(st["rel"]) != 9 * nks:
        err(c, f"A: {len(st['acq'])} acquires / {len(st['rel'])} releases, want {9 * nks}")
    if len(st["bacq"]) != nks - 1 or len(st["brel"]) != nks:
        err(c, f"B: {len(st['bacq'])} acquires / {len(st['brel'])} releases, want {nks - 1} / {nks}")
    if errs:
        raise Fail("\n".join(notes[:12] + errs))
    st["cycles"] = c + 1
    st["bundles_pre"] = len(pre)
    return st


def fit(times, info, work, name="mm_e7k"):
    """gen_e7.fit's prologue alignment on the KTL emission"""
    text, head, period, order_b = emit(times, info, name=name)
    _, dis, addr = E.G.assemble(text, work, name)
    offs = E.G.bundle_addrs(dis)
    ls = text.splitlines()
    a, b = ls.index(f"{name}:"), ls.index(f".L{name}_pro:")
    n_setup = sum(1 for x in ls[a + 1:b] if not x.strip().startswith(".") and not x.strip().endswith(":"))
    p0 = next(o for o in offs[n_setup:] if o % 16 == 0)
    c0, c1 = ls.index(f".L{name}_pro:"), ls.index(f".L{name}_body:")
    npro = sum(1 for x in ls[c0 + 1:c1] if not x.strip().startswith(".") and not x.strip().endswith(":"))
    i0 = offs.index(p0)
    pro = offs[i0:i0 + npro + 1]
    sizes = [y - x for x, y in zip(pro, pro[1:])]
    need = (-(pro[-1] - pro[0])) % 16
    best = {0: (0, frozenset())}
    for i, sz in enumerate(sizes):
        gain = 16 - sz
        if gain <= 0:
            continue
        nxt = dict(best)
        for r, (cost, sel) in best.items():
            r2, c2 = (r + gain) % 16, cost + gain
            if r2 not in nxt or c2 < nxt[r2][0]:
                nxt[r2] = (c2, sel | {i})
        best = nxt
    assert need in best
    full = best[need][1]
    text, head, period, order_b = emit(times, info, name=name, full_pro=full)
    o, dis, addr = E.G.assemble(text, work, name)
    offs = E.G.bundle_addrs(dis)
    i0 = offs.index(pro[0])
    assert offs[i0 + npro] == addr[f"{name}_start"], "a pad bundle sits between the prologue and the body"
    assert addr[f"{name}_last"] % 16 == 0 and addr[f"{name}_start"] % 16 == 0
    return text, head, period, order_b, dict(expanded=sorted(full), added=best[need][0], dis=dis)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("emit", "fit"):
        b = sub.add_parser(nm)
        b.add_argument("json")
        b.add_argument("--out", required=True)
        b.add_argument("--nks", type=int, default=4)
    c = sub.add_parser("check")
    c.add_argument("s")
    c.add_argument("--json", required=True)
    c.add_argument("--nks", type=int, default=4)
    c.add_argument("--place")
    o = ap.parse_args()
    d = json.loads(Path(o.json).read_text())
    cfg = cfg_of(d["info"])
    if o.cmd in ("emit", "fit"):
        out = Path(o.out)
        if o.cmd == "emit":
            text, head, period, order_b = emit(d["times"], d["info"])
        else:
            text, head, period, order_b, fi = fit(d["times"], d["info"], out.parent)
            out.with_suffix(".dis").write_text(fi.pop("dis"))
            d["fit"] = fi
        out.write_text(text)
        out.with_suffix(".h").write_text(header(head, period, order_b, cfg))
        d["head"], d["period"], d["order_b"] = head, period, order_b
        Path(o.json).write_text(json.dumps(d, indent=1))
        for nks in sorted({1, 2, 3, o.nks}):
            st = check(text, d["info"], head, period, order_b, nks=nks, verbose=True)
            print(f"nks {nks}: {st['cycles']} cycles per call ({nks} k-steps: {(st['cycles']) / nks:.1f} per k-step; "
                  f"setup + prologue {st['bundles_pre']}), {st['macs']} MACs checked, {st['stores']} stores, bank stalls "
                  f"{st['stalls']} {dict(st['stall_where'])}, busy {dict((k, round(v / st['body_cycles'], 2)) for k, v in sorted(st['busy'].items()))}")
    else:
        text = Path(o.s).read_text()
        pl = E.parse_place(o.place)
        st = check(text, d["info"], [tuple(x) for x in d["head"]], [tuple(x) for x in d["period"]],
                   [tuple(x) for x in d["order_b"]], nks=o.nks, pl=pl)
        print(f"nks {o.nks}: {st['cycles']} cycles per call, {st['macs']} MACs, bank stalls {st['stalls']} {dict(st['stall_where'])}")


if __name__ == "__main__":
    main()
