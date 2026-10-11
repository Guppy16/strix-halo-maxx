"""One GEMM k-step in hand-scheduled AIE2P assembly, the 9 stripes software-pipelined (round 9, npu-research P20).

  extern "C" void mm_ks(const int32_t *rec /*p0*/, const bfp16ebs8 *pB /*p1*/, float *piece0 /*p2*/);

Same arithmetic and MAC order as r7/gen_mm.py's mm_core (4 accumulator chains, B in pack8's pair4 order): bit-identical.
Differences:
  - one call per k-step: a zero-overhead hardware loop over the 9 stripes (16-row blocks of the 144-row tile). The loop body
    is one period of a modulo schedule: stripe s's 4 groups, the previous stripe's last stores and the next stripe's ramp
    (A slot load, A fill + row-block-0 pops, its first piece's accumulator loads) overlap the stripe boundary.
  - the A ring's locks are taken inside: acq.cond #53, -1 (aU, full; the record's flag in r26) before a stripe's first A access,
    rel #52 (aF, free) >= 6 cycles after its last pop. design8 pins the ring's locks (KSTEP).
  - accumulators as 1 KB pieces (mm8.cc MM_PIECES): the load / store cursors (p3 / p4) jump between pieces by deltas from a
    per-stripe record.
Per stripe s = 0..9 (9 = a dummy: its A slot is readable, its flag 0 skips the acq) the kernel loads,
through one post-incremented pointer and in the order the generated header lists:
  w0 A slot address, w1..w4 the cursor jump into group 0..3 (both cursors), w5 the acquire flag (1; 0 for the dummy)
The loop runs 9 times; the prologue is frame -1 restricted to stripe 0's ramp, the epilogue frame 9 restricted to
stripe 8's last stores. Frame 0 also runs "stripe -1"'s last stores that spill past the frame: p4 starts at piece 0 and
stripe 0's jump into its G0 (MM_KS_P4_BACK in the generated .h) brings it back, so those garbage stores rewrite stripe 0's first piece with the
values just loaded from it, before its real stores (harmless).

usage: python3 r9/gen_ks.py out.s [--check-only]
"""

import sys
from collections import defaultdict

AREG = [0, 1, 2, 3, 4, 6, 8, 10]
BREG = [5, 7, 9, 11]
Q = ["bmll", "bmlh", "bmhl", "bmhh"]
POP_LAT, LOAD_LAT, ACC_READ, ACC_WRITE, STORE_AFTER = 8, 7, 3, 6, 6
SCL_LAT = 8  # scalar lda -> usable (pointer / modifier / r26): margin over the load latency
NG = 4
F_SLOT, F_JUMP, F_FLAG = 0, 1, 5  # record fields: A slot address, jumps into groups 0..3, lock flag


class Infeasible(Exception):
    pass


class Sched:
    """ops of ONE stripe at stripe-relative times t; resources reserved modulo the period P"""

    def __init__(self, P):
        self.P = P
        self.res = defaultdict(set)  # (t mod P) -> slots
        self.ops = []  # (t, slot, text, tag)

    def free(self, t, slot):
        return slot not in self.res[t % self.P]

    def put(self, t, slot, text, tag=None):
        if not self.free(t, slot):
            raise Infeasible(f"slot {slot} at {t} ({t % self.P}) taken: {text}")
        self.res[t % self.P].add(slot)
        self.ops.append((t, slot, text, tag))
        return t

    def first_free(self, lo, hi, slot):
        for t in range(lo, hi + 1):
            if self.free(t, slot):
                return t
        raise Infeasible(f"no {slot} slot in [{lo}, {hi}]")


def build(S, P):
    """S: group start times (B fill of the group's first pop block), P: period"""
    sc = Sched(P)
    mac, pop = {}, {}
    # B: per group 4 fills + 32 pops, MAC at pop + 8 (chain j % 4), one per cycle
    for G in range(NG):
        for j in range(32):
            if j % 8 == 0:
                sc.put(S[G] + 9 * (j // 8), "ldb", "vldb.fill.512 [p1, lf1, r25]")
            c = S[G] + 1 + j + j // 8
            sc.put(c, "ldb", f"vldb.pop.576 ex{BREG[j % 4]}, [p1, lf1, r25]", ("Bpop", G, j))
            pop[G, j] = c
            mac[G, j] = sc.put(c + POP_LAT, "vec", f"vmac.f dm{j % 4}, dm{j % 4}, ex{AREG[j // 4]}, ex{BREG[j % 4]}, r3", ("mac", G, j))
    # B rewinds: before row block 1 (G2) and before the next stripe's G0 (S[0] + P)
    for G, nxt in ((1, S[2]), (3, S[0] + P)):
        last = pop[G, 31]
        if nxt < last + 3:
            raise Infeasible("B rewind")
        sc.put(last, "mv", "mov r25, #0")
        sc.put(last + 1, "mv", "mov p1, p5")
    # stores: memory order per group (cursor p4), >= the chain's last MAC + 6
    store = {}
    c = -10 ** 6
    for G in range(NG):
        for i in range(4):
            for qq in range(4):
                c = max(c, mac[G, 28 + i] + STORE_AFTER)
                c = sc.first_free(c, c + P, "st")
                store[G, i, qq] = sc.put(c, "st", f"vst {Q[qq]}{i}, [p4], #64", ("st", G, i, qq))
                c += 1
    # p4 jumps after each group's last store (into the next piece; G3 -> the next stripe's G0), st slot
    p4jump = {}
    for G in range(NG):
        lo = store[G, 3, 3] + 1
        nxt_first = store[(G + 1) % NG, 0, 0] + (P if G == NG - 1 else 0)
        t = sc.first_free(lo, nxt_first - 3, "st")
        p4jump[G] = sc.put(t, "st", f"padds [p4], m{(G + 1) % NG}", ("p4j", G))
    # lda: accumulator loads (memory order, cursor p3) and the A stream, earliest deadline first
    accs = []
    for G in range(NG):
        for i in range(4):
            for qq in range(4):
                if G == 0:
                    early = store[NG - 1, i, qq] - P + 2 - LOAD_LAT
                else:
                    early = store[G - 1, i, qq] + 2 - LOAD_LAT
                dead = mac[G, i] + ACC_READ - LOAD_LAT
                accs.append((early, dead, f"vlda {Q[qq]}{i}, [p3], #64", ("ld", G, i, qq)))
    astr = []
    # row block 0: after the previous stripe's last reads of ex_q (its G3, MAC 4q+3, at - P)
    astr.append((mac[NG - 1, 3] - P - LOAD_LAT - 1, mac[0, 0] - POP_LAT - 1, "vlda.fill.512 [p0, lf0, r24]", ("Afill", 0)))
    for q in range(8):
        astr.append((mac[NG - 1, 4 * q + 3] - P - POP_LAT + 1, mac[0, 4 * q] - POP_LAT, f"vlda.pop.576 ex{AREG[q]}, [p0, lf0, r24]", ("Apop", 0, q)))
    astr.append((mac[1, 3] - LOAD_LAT - 1, mac[2, 0] - POP_LAT - 1, "vlda.fill.512 [p0, lf0, r24]", ("Afill", 1)))
    for q in range(8):
        astr.append((mac[1, 4 * q + 3] - POP_LAT + 1, mac[2, 4 * q] - POP_LAT, f"vlda.pop.576 ex{AREG[q]}, [p0, lf0, r24]", ("Apop", 1, q)))
    # accumulator loads first (the tightest: each group's 16 quarters between the previous group's stores and its
    # first MACs), in memory order; then the A stream into the remaining lda cycles (fill before pops, pops in order).
    # Placed modulo P, so ops of neighbouring stripes compete for the same cycles.
    lda_t = {}
    t = -10 ** 6
    for e, d, txt, tag in accs:
        t = sc.first_free(max(t + 1, e), d, "lda")
        sc.put(t, "lda", txt, tag)
        lda_t[tag] = t
    t = -10 ** 6
    for e, d, txt, tag in astr:
        lo = max(t + 1, e)
        while True:
            t = sc.first_free(lo, d, "lda")
            if tag[0] != "Afill" or sc.free(t - 1, "mv"):
                break
            lo = t + 1
        if tag[0] == "Afill":
            sc.put(t - 1, "mv", "mov r24, #0", ("r24", tag[1]))
        sc.put(t, "lda", txt, tag)
        lda_t[tag] = t
    # p3 jumps after each group's last load, >= 3 before the next group's first load (st slot)
    for G in range(NG):
        lo = lda_t["ld", G, 3, 3] + 1
        hi = lda_t["ld", (G + 1) % NG, 0, 0] + (P if G == NG - 1 else 0) - 3
        tj = sc.first_free(lo, hi, "st")
        sc.put(tj, "st", f"padds [p3], m{(G + 1) % NG}", ("p3j", G))
    # locks: acq.cond before the stripe's first A access (its rb-0 fill), rel >= 6 after its last pop (alu slot)
    fill0 = lda_t["Afill", 0]
    last_pop = lda_t["Apop", 1, 7]
    t_acq = sc.first_free(fill0 - 6, fill0 - 1, "alu")
    # acquire_greater_equal(aU, 1) when r26 != 0 (Peano lowers the conditional intrinsic to exactly this); r26 = 0 for the
    # dummy stripe. (A plain acq with value 0 is acquire_EQUAL 0: it waits for an empty ring and hung, round 9.)
    sc.put(t_acq, "alu", "acq.cond #53, r4, r26", ("acq",))
    t_rel = sc.first_free(last_pop + 6, last_pop + 40, "alu")
    sc.put(t_rel, "alu", "rel #52, r5", ("rel",))
    if t_rel >= P:
        raise Infeasible("rel in stage +1")
    # record loads (lda slot; the record pointer p2 moves by 64 at the end of every frame).
    # p0 = this stripe's A slot: after the previous stripe's last pop, >= SCL_LAT before this stripe's rb-0 fill;
    # r26 = this stripe's lock flag before the acq; m_k for each jump after the previous stripe's use of m_k.
    # record loads (lda slot, post-increment through one table: the C++ side writes the values in the order the loads
    # execute, emitted in the generated header). p0 = this stripe's A slot: after the previous stripe's last pop, >= SCL_LAT
    # before this stripe's rb-0 fill; r26 = its lock flag before the acq; m_k = the jump into group k (shared by the p3
    # and p4 cursors, which visit the same pieces): after the previous stripe's last use, before this stripe's first.
    rec_loads = [("p0", F_SLOT, last_pop - P + 1, fill0 - SCL_LAT), ("r26", F_FLAG, t_acq - P + 1, t_acq - SCL_LAT)]
    uses = defaultdict(list)
    for tt, sl, txt, tag in sc.ops:
        if tag and tag[0] in ("p3j", "p4j"):
            k = (tag[1] + 1) % NG
            uses[k].append(tt - (P if k == 0 else 0))  # (k = 0: the jump at the end of the previous stripe's G3)
    for k in range(NG):
        rec_loads.append((f"m{k}", F_JUMP + k, max(uses[k]) - P + 1, min(uses[k]) - SCL_LAT))
    for reg, field, lo, hi in rec_loads:  # the latest free cycle: keeps the prologue short
        lo = max(lo, -P)
        tl = hi
        while tl >= lo and not sc.free(tl, "lda"):
            tl -= 1
        if tl < lo:
            raise Infeasible(f"record load {reg}")
        if tl >= P:
            raise Infeasible(f"record load {reg} in stage +1")
        sc.put(tl, "lda", f"lda {reg}, [p2], #4", ("rec", reg, field))
    n1 = sum(1 for (G, i, qq), t in store.items() if G == NG - 1 and t >= P)
    assert all(t < P for (G, i, qq), t in store.items() if G < NG - 1), "only G3's stores may spill into the next frame"
    return sc, dict(mac=mac, pop=pop, store=store, lda=lda_t, acq=t_acq, rel=t_rel, p4_back=64 * n1)


def frame_ops(sc, P):
    """ops by frame-local time and stage: stage = floor(t / P); an op of stripe s at t runs in frame s + stage"""
    out = []
    for t, slot, txt, tag in sc.ops:
        stage = t // P
        out.append((t - stage * P, stage, slot, txt, tag))
    return out


def text_of(txt, tag, stage, lt, Bt=None):
    return txt


def full_bundle(ops):
    """all six slot families named, in Peano's order (lda, ldb, st, alu, mv, vec), nop where idle"""
    by = {slot: t for slot, t, _, _ in ops}
    order = [("lda", "nopa"), ("ldb", "nopb"), ("st", "nops"), ("alu", "nopx"), ("mv", "nopm"), ("vec", "nopv")]
    assert set(by) <= {k for k, _ in order}, by
    return ";\t\t".join(by.get(k, n) for k, n in order)


def emit(sc, P, name="mm_ks", nstripes=9):
    fo = frame_ops(sc, P)
    stages = sorted({st for _, st, *_ in fo})
    assert stages[0] >= -1 and stages[-1] <= 1, stages
    body = defaultdict(list)
    for lt, st, slot, txt, tag in fo:
        body[lt].append((slot, txt, st, tag))
    pro_t = sorted({lt for lt, st, *_ in fo if st == -1})
    pro_lo = min(pro_t) if pro_t else P
    epi_t = sorted({lt for lt, st, *_ in fo if st == 1})
    epi_hi = max(epi_t) if epi_t else -1
    lines = ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    setup = ["movs p3, p2", "movs p4, p2", "movs p5, p1", "mova r3, #780", "mova r25, #0", "mova r4, #-1", "mova r5, #1",
             "movs p2, p0", "movxm ls, #.Lks_body", "movxm le, #.Lks_end", f"mova r0, #{nstripes}",
             "add.nc lc, r0, #0"]
    for s in setup:
        lines.append("\t" + s)
    lines += ["\tnop"] * 4
    # prologue: frame -1, stripe 0's stage -1 ops only, plus the bump at P - 1
    for lt in range(pro_lo, P):
        ops = [o for o in body[lt] if o[2] == -1]
        lines.append("\t" + ";\t\t".join(t for _, t, _, _ in ops) if ops else "\tnop")
    # the loop: one frame. As Peano's loops: the start 16 B aligned, the first and the last bundle in the full 16-byte
    # format (every slot named, nop where idle) and the last one 16 B aligned too (the assembler pads with nop bundles:
    # one extra cycle per frame at most, which only adds slack to the schedule's dependences)
    lines.append("\t.p2align\t4")
    lines.append(".Lks_body:")
    for lt in range(P):
        ops = body[lt]
        if lt in (0, P - 1):
            if lt == P - 1:
                lines.append("\t.p2align\t4")
                lines.append(".Lks_end:")
            lines.append("\t" + full_bundle(ops))
        else:
            lines.append("\t" + ";\t\t".join(t for _, t, _, _ in ops) if ops else "\tnop")
    # epilogue: frame 9, stripe 8's stage +1 ops
    for lt in range(0, epi_hi + 1):
        ops = [o for o in body[lt] if o[2] == 1]
        lines.append("\t" + ";\t\t".join(t for _, t, _, _ in ops) if ops else "\tnop")
    lines += ["\tret lr"] + ["\tnop"] * 5 + [f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return lines, dict(prologue=P - pro_lo, body=P, epilogue=epi_hi + 1)


def simulate(lines, nstripes=9):
    """execute the emitted stream symbolically (hardware loop unrolled): check that every MAC reads the intended A and
    B panels and accumulator, every load / store hits the intended piece offset, locks in order. Returns stats."""
    import re
    # flatten: setup + prologue + body x n + epilogue
    i0 = lines.index(f"mm_ks:") + 1 if "mm_ks:" in lines else 5
    body_start = lines.index(".Lks_body:")
    end_label = lines.index(".Lks_end:")
    seq = []
    pre = [l for l in lines[i0:body_start]]
    body = [l for l in lines[body_start + 1:] if not l.startswith(".Lks_end")]
    body = body[:end_label - body_start]  # P bundles
    post = lines[lines.index(".Lks_end:") + 2:]
    post = [l for l in post if not l.startswith(".Lfunc") and not l.startswith("\t.size")]
    seq = [(l, -1) for l in pre] + [(l, f) for f in range(nstripes) for l in body] + [(l, nstripes) for l in post]
    return seq


def check(sc, P, info, nstripes=9):
    """time-domain check on the unrolled stream: stripe s's op at t executes at absolute time T(s, t) = s P + t
    (prologue/epilogue keep the same relative timing). Conveyor / register reads and memory offsets."""
    events = []  # (abs time, slot, text, stripe, tag)
    for s in range(-1, nstripes + 1):
        for t, slot, txt, tag in sc.ops:
            stage = t // P
            frame = s + stage
            # which ops actually execute: frames -1 (prologue: stage -1 ops of stripe 0), 0..n-1 (all), n (stage +1 of n-1)
            if frame == -1 and not (s == 0 and stage == -1):
                continue
            if frame == nstripes and not (s == nstripes - 1 and stage == 1):
                continue
            if frame < -1 or frame > nstripes:
                continue
            events.append((s * P + t, slot, txt, s, tag))
    events.sort(key=lambda e: e[0])
    # B conveyor + A registers: each MAC (stripe s, G, j) must see its own pops
    land = defaultdict(list)  # reg -> [(land time, (kind, stripe, ...))]
    for T, slot, txt, s, tag in events:
        if tag and tag[0] == "Bpop":
            land[f"ex{BREG[tag[2] % 4]}"].append((T + POP_LAT, ("B", s, tag[1], tag[2])))
        if tag and tag[0] == "Apop":
            land[f"ex{AREG[tag[2]]}"].append((T + POP_LAT, ("A", s, tag[1], tag[2])))
    for k in land:
        land[k].sort()

    def seen(reg, T):
        best = None
        for lt, what in land[reg]:
            if lt <= T:
                best = what
            else:
                break
        return best

    n = 0
    for T, slot, txt, s, tag in events:
        if tag and tag[0] == "mac" and 0 <= s < nstripes:
            G, j = tag[1], tag[2]
            b = seen(f"ex{BREG[j % 4]}", T)
            assert b == ("B", s, G, j), f"MAC s{s} G{G} j{j} at {T}: B reg holds {b}"
            a = seen(f"ex{AREG[j // 4]}", T)
            assert a == ("A", s, G // 2, j // 4), f"MAC s{s} G{G} j{j} at {T}: A reg holds {a}"
            n += 1
    # accumulators: the load of (s, G, i, qq) lands after the previous store of that register read it, and before the
    # group's first MAC on chain i reads it; the store reads after the chain's last MAC wrote it
    st, ld = {}, {}
    for T, slot, txt, s, tag in events:
        if tag and tag[0] == "st":
            st[s, tag[1], tag[2], tag[3]] = T
        if tag and tag[0] == "ld":
            ld[s, tag[1], tag[2], tag[3]] = T
    macT = {(s, tag[1], tag[2]): T for T, slot, txt, s, tag in events if tag and tag[0] == "mac"}
    for (s, G, i, qq), T in ld.items():
        if not 0 <= s < nstripes:
            continue
        assert T + LOAD_LAT <= macT[s, G, i] + ACC_READ, f"load s{s} G{G} {i}{qq} lands late"
        prev = (s, G - 1, i, qq) if G > 0 else (s - 1, NG - 1, i, qq)
        if prev in st and prev[0] >= 0:  # (stripe -1's garbage stores rewrite piece 0 with what was just loaded from it)
            assert T + LOAD_LAT >= st[prev] + 2, f"load s{s} G{G} {i}{qq} overwrites before the store {prev} read"
    for (s, G, i, qq), T in st.items():
        if 0 <= s < nstripes:
            assert T >= macT[s, G, 28 + i] + STORE_AFTER, f"store s{s} G{G} {i}{qq} early"
    # cursors: simulate p3 / p4 as (piece index, offset) with the record jumps (stripe s piece G = 4 s + G)
    for cur, kind, jt in (("p3", "ld", "p3j"), ("p4", "st", "p4j")):
        pos = (0, 0) if cur == "p3" else (0, 0)  # p4: frame 0 first stores 16 garbage quarters of "stripe -1"
        piece, off = 0, 0
        if cur == "p4":
            off = 0
        for T, slot, txt, s, tag in events:
            if tag and tag[0] == kind:
                want = (4 * s + tag[1], 64 * (4 * tag[2] + tag[3]))
                if s >= 0:
                    assert (piece, off) == want, f"{cur} at {T} ({tag}, stripe {s}): at piece {piece} +{off}, want {want}"
                off += 64
            if tag and tag[0] == jt:
                G = tag[1]
                if s < 0:  # stripe -1's jump into stripe 0 G0: record 0's w5 brings p4 back over its stage +1 stores
                    off -= info["p4_back"]
                else:
                    piece, off = 4 * s + G + 1, 0
    # record registers: every use sees the load of the right record and field (landed, not yet overwritten)
    loads = defaultdict(list)  # reg -> [(land time, record index, field)]
    for T, slot, txt, s, tag in events:
        if tag and tag[0] == "rec":
            loads[tag[1]].append((T + SCL_LAT, s, tag[2]))  # text_of's offset reads record s, field tag[2]
    for k in loads:
        loads[k].sort()

    def reg_at(reg, T):
        cur = None
        for lt_, rs, fld in loads[reg]:
            if lt_ <= T:
                cur = (rs, fld)
        return cur

    for T, slot, txt, s, tag in events:
        if not tag:
            continue
        need = None
        if tag[0] == "p3j":  # jump into group (G + 1) % 4: of stripe s, or of s + 1 after G3
            k = (tag[1] + 1) % NG
            need = (f"m{k}", s + (1 if tag[1] == NG - 1 else 0), F_JUMP + k)
        elif tag[0] == "p4j":
            k = (tag[1] + 1) % NG
            need = (f"m{k}", s + (1 if tag[1] == NG - 1 else 0), F_JUMP + k)
        elif tag[0] == "acq":
            need = ("r26", s, F_FLAG)
        elif tag[0] in ("Afill", "Apop"):
            need = ("p0", s, F_SLOT)
        if need and 0 <= need[1] <= nstripes:
            got = reg_at(need[0], T)
            assert got == (need[1], need[2]), f"{tag} of stripe {s} at {T}: {need[0]} holds {got}, wants {need[1:]}"
    # locks: per stripe acq before its first A access, rel after its last pop + 6
    for s in range(nstripes):
        acq = [T for T, sl, tx, ss, tag in events if ss == s and tag and tag[0] == "acq"]
        rel = [T for T, sl, tx, ss, tag in events if ss == s and tag and tag[0] == "rel"]
        afirst = min(T for T, sl, tx, ss, tag in events if ss == s and tag and tag[0] in ("Afill", "Apop"))
        alast = max(T for T, sl, tx, ss, tag in events if ss == s and tag and tag[0] == "Apop")
        assert len(acq) == 1 and acq[0] < afirst and len(rel) == 1 and rel[0] >= alast + 6, (s, acq, rel, afirst, alast)
    order = [(s, tag[2]) for T, slot, txt, s, tag in events if tag and tag[0] == "rec"]
    return n, order


def main(out, check_only=False):
    best = None
    for D in range(44, 60):  # group spacing
        S = [0, D, 2 * D, 3 * D]
        P = 4 * D
        try:
            sc, info = build(S, P)
        except Infeasible as e:
            last_err = (D, str(e))
            continue
        best = (D, P, sc, info)
        break
    if best is None:
        raise SystemExit(f"no schedule: {last_err}")
    D, P, sc, info = best
    nm, order = check(sc, P, info)
    lines, sizes = emit(sc, P)
    print(f"{out}: group spacing {D}, period {P} bundles per stripe (mm_core: 225 per call), prologue {sizes['prologue']}, "
          f"epilogue {sizes['epilogue']}; {nm} MACs checked (9 stripes)")
    if not check_only:
        open(out, "w").write("\n".join(lines) + "\n")
        # record 0's p4 jump (w5): back over the stage +1 stores of "stripe -1" that frame 0 runs into piece 0
        hdr = [f"// generated by r9/gen_ks.py: mm_ks's record stream (post-increment loads in execution order)",
               f"#define MM_KS_P4_BACK {-info['p4_back']}  // stripe 0's jump into its G0 (see gen_ks.py)",
               f"#define MM_KS_NREC {len(order)}",
               "// {stripe, field}: field 0 = A slot address, 1..4 = cursor jump into group 0..3, 5 = lock flag",
               "static constexpr int8_t mm_ks_order[MM_KS_NREC][2] = {" + ", ".join(f"{{{a}, {b}}}" for a, b in order) + "};"]
        open(out.rsplit(".", 1)[0] + ".h", "w").write("\n".join(hdr) + "\n")
        print(f"  record 0 w5 (p4 back) = {-info['p4_back']}")


if __name__ == "__main__" and "--banks" not in sys.argv:
    main(sys.argv[1] if len(sys.argv) > 1 else "mm_ks.s", "--check-only" in sys.argv)


def bank_cost(sc, P, piece_addr, slot_addr, b_bank=3, nstripes=9):
    """extra cycles per k-step from same-bundle bank conflicts (npu-research bankmodel rule: a bundle costs the most
    accesses it makes to one 16 KB bank). piece_addr(stripe, G) -> address of the group's 1 KB piece,
    slot_addr(stripe) -> the A slot. Accumulator quarter q of a group at piece + 64 q."""
    from collections import Counter
    bundles = defaultdict(Counter)
    for s in range(-1, nstripes + 1):
        for t, slot, txt, tag in sc.ops:
            if not tag:
                continue
            stage = t // P
            frame = s + stage
            if frame < 0 or frame >= nstripes:  # count the steady frames only
                continue
            T = s * P + t
            k = tag[0]
            if k in ("Apop", "Afill"):
                ss = min(max(s, 0), nstripes - 1)
                b = (slot_addr(ss) - 0x70000) >> 14
            elif k in ("Bpop",) or (txt and txt.startswith("vldb.fill")):
                b = b_bank
            elif k in ("ld", "st"):
                if s < 0 or s >= nstripes:
                    continue
                G, i, qq = tag[1], tag[2], tag[3]
                b = (piece_addr(s, G) + 64 * (4 * i + qq) - 0x70000) >> 14
            else:
                continue
            bundles[T][b] += 1
    return sum(max(c.values()) - 1 for c in bundles.values())


def explore(argv):
    """python3 r9/gen_ks.py --banks [ACC_HI C SLOT0 SLOT1 SLOT2]: conflict cost of the layouts for the given addresses"""
    vals = [int(v, 16) for v in argv] if argv else [0x71900, 0x78900, 0x77B80, 0x78000, 0x78480]
    h0, c0, s0, s1, s2 = vals
    slots = [s0, s1, s2]
    D = 48
    sc, info = build([0, D, 2 * D, 3 * D], 4 * D)

    def affine(s, G):
        rb, pair = 2 * s + G // 2, G % 2
        return h0 + 1024 * (6 + rb) if pair else (c0 + 1024 * rb if rb < 12 else h0 + 1024 * (rb - 12))

    def old(s, G):
        rb, pair = 2 * s + G // 2, G % 2
        base = c0 + 2048 * rb if rb < 6 else h0 + 2048 * (rb - 6)
        return base + 1024 * pair

    for name, lay in (("old contiguous rows", old), ("affine pieces", affine)):
        print(f"{name:22s} extra cycles per k-step {bank_cost(sc, 4 * D, lay, lambda s: slots[s % 3])}")


if __name__ == "__main__" and "--banks" in sys.argv:
    explore(sys.argv[sys.argv.index("--banks") + 1:])
