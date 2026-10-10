#!/usr/bin/env python3
"""gen_rot.py - ROT: the GEMM k-step (mm_ks) with a rotating accumulator, scheduled exactly by CP-SAT (research pass 14, E1).

The deployed mm_ks (npu-tapnext r9/gen_ks.py) runs 4 accumulator chains in lockstep (B in pack8's pair4 order: per k-panel the
4 column blocks of a pair), so the 4 accumulators end together and the MACs stop for 12.75 bundles per 32 MACs while 16
quarters are stored and 16 loaded (459 cycles per k-step). ROT:

  * a JOB = one accumulator (row block rb, column block cb) over the k-step's 8 k-panels: 8 dependent vmacs, kp 0..7 in order
    (the deployed per-accumulator order), its 4 C quarters loaded before and stored after;
  * jobs start every 9 cycles (II 9 per job = the bank-3 bound: 8 B pops + 1 fill per job, all in B's bank), so 2-3 chains
    interleave and one accumulator finishes every 9 cycles: dm[cb % 4] rotates through the swap, the MACs never stop;
  * B is popped in the order the MACs need it, so B is REPACKED on the host (`border`: stream position -> panel), one pass
    (64 pops + 8 fills = the 4,608 B object) per row block, restarted at every row block;
  * A (the row block's 8 panels in ex0-4/6/8/10) is refreshed in place at the row-block change: pop k lands between the last
    MAC on panel k of (rb - 1, cb 7) and the first of (rb, cb 0); the A stream restarts per row block (mov r24 + fill);
  * the zero-overhead loop body is ONE ROW BLOCK (R = 72 + g bundles), 18 trips; stores sit in the last stage, so there is no
    epilogue; the last trip runs the early stage of a dummy row block 18 (reads only, its lock flags 0).

Restart gap g (bundles per row block in which the B bank idles while the B stream restarts):
  g = 2  `mov r25, #0` with the pass's last pop, `mov p1, p5` next, the next fill 2 later: gen_ks's mechanism (deployed);
  g = 1  `padda [p1], m2` (m2 = -4608) instead of the mov, the fill 1 later: .td latency 1, UNVERIFIED (probe in PROPOSALS);
  g = 0  the pass's last pop is `vldb.pop.576.2d .., [p1, lf1, r25, d2]` (m2 = -4544, dn2 = 0): the pointer wraps in the pop
         itself, UNVERIFIED (probe).

solve(): one CP-SAT model: issue cycles of one job's 24 ops (copied to the 8 jobs of a row block at 9-cycle offsets) and of the
row block's 21 control ops; slots and itinerary ports modulo R; the B stream's fixed fill / pop / gap pattern; .td latencies
with the visibility rule (w - r + 1) and the old-value windows (accumulator swap, B and A registers); scalar loads 8 cycles,
pointer moves 3 (gen_ks margins); stages (stores in the last); bank conflicts on the deployed placement (soft, counted per
k-step). check(): replays the EMITTED asm bundle by bundle over the whole call (setup, prologue, 18 trips, both .p2align pads):
FIFO streams byte by byte (fills, pops, FIFO-POP9, restarts), registers with latencies and uncertain windows, locks, records,
every MAC's operands, final memory = 8 MACs per accumulator in kp order, bank stalls on real addresses.

usage (testbox; results of pass 14 in r19/: rot_<v>.json, mm_rot_<v>.s / .h / .dis for v = g2f g1f g0f g0u):
  uv run --no-project --with ortools --with numpy python r19/gen_rot.py solve --g 2 --free --json r19/rot_g2f.json
      (--free: each job of the row block its own cycles; without it the 8 jobs share one template, which is feasible for
      g = 0 only: with g > 0 the B pass's exact cover forces each job's 8 pops into consecutive cycles)
  docker ... npu-iron python3 r19/gen_rot.py fit r19/rot_g2f.json --out r19/mm_rot_g2f.s
      (emit + expand prologue bundles until the prologue ends 16 B aligned, so the .p2align before the body adds no pad
      bundle (asserted on the disassembly) + the .h (records, B order) + the replay of the whole call)
  python3 r19/gen_rot.py check r19/mm_rot_g2f.s --json r19/rot_g2f.json [--place c_obj=..,acc_hi=..,slots=a:b:c,rec=..,b=..]
      (replay on another L1 placement: correctness + the bank stalls it costs; rotb.py place prints the bench's)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

NRB, NCB, NK, NQ, RHO = 18, 8, 8, 4, 9  # row blocks, column blocks, k-panels, accumulator quarters, A objects per k-step
JOB = 9  # cycles between job starts inside a row block
AREG = [0, 1, 2, 3, 4, 6, 8, 10]  # A panels kp 0..7 (the deployed registers)
BREG = [5, 7, 9, 11]
QN = ["bmll", "bmlh", "bmhl", "bmhh"]
# deployed placement (r17d input_with_addresses.mlir + llvm-nm; all 24 GEMM cores share it, MACHINE R13.4)
PLACE = dict(c_obj=0x78900, acc_hi=0x718C0, slots=(0x77B80, 0x78000, 0x78480), rec=0x778C8, b=0x7D100)
# .td latencies (aie2p_lat.json) under the visibility rule: a write of latency w is seen by a read at stage r of an op issued
# >= w - r + 1 later
D_POP_MAC = 8 - 1 + 1  # pop dst 8 -> vmac operand (stage 1)
D_LD_MAC = 7 - 4 + 1  # vlda bm dst 7 -> vmac accumulator (stage 4)
D_MAC_MAC = 6 - 4 + 1  # vmac dst 6 -> dependent vmac (3; "2 through the one-deep bypass" is not used)
D_MAC_ST = 6 - 1 + 1  # vmac -> vst (stage 1)
OLD_ST = 7 - 1  # a store reads the old quarter iff issued <= load + 6 (load dst 7, store stage 1)
SCL = 8  # scalar lda -> first use (gen_ks margin; Peano uses 4-6)
PTR_GAP = 3  # padds -> next load / store through that pointer (gen_ks margin)
MACS = NRB * NCB * NK
F_PTR, F_J0, F_J1, F_ACQ, F_REL = range(5)  # record fields per row block
FIELD_NAMES = ["A pointer", "jump into (rb, pair 0)", "jump into (rb, pair 1)", "acquire flag", "release flag"]
# instruction names for the itinerary ports (aie2p_lat.json); scalar lda gets the spill load's itinerary (round 13 rule)
INSTR = dict(P="VLDB_POP_576_normal_pop", M="VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX", L="VLDA_dmx_lda_bm_pstm_nrm_imm",
             S="VST_dmx_sts_bm_pstm_nrm_imm", AF="VLDA_FILL_512", A="VLDA_POP_576_normal_pop", ACQ="ACQ_COND_mLockId_imm",
             REL="REL_COND_mLockId_imm", R="LDA_dmv_lda_q_spill")


def _ports(name):
    """[(stage, port)] of an instruction (npu-tapnext r11/msched.ports), ports at stage 0 / 1 that only the slot uses dropped"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "npu-tapnext" / "r11"))
    import msched

    return [(st, p) for st, p in msched.ports(name)]


# ---------------------------------------------------------------------------------------------------------------------
# geometry
def piece(rb, pair, pl=PLACE):
    """mm8.cc MM_PIECES: the 1 KB accumulator piece of (row block, column pair)"""
    if pair:
        return pl["acc_hi"] + 1024 * (6 + rb)
    return pl["c_obj"] + 1024 * rb if rb < 12 else pl["acc_hi"] + 1024 * (rb - 12)


def caddr(rb, cb, q, pl=PLACE):
    return piece(rb, cb >> 2, pl) + 256 * (cb & 3) + 64 * q


def bank(a):
    return -1 if a is None else (a - 0x70000) >> 14


def a_ptr(r, pl=PLACE):
    """the A stream's start for row block r (record field 0): slot of stripe r // 2, + 576 for the stripe's second row block"""
    return pl["slots"][(r // 2) % 3] + 576 * (r % 2) if r < NRB else pl["slots"][1]  # dummy: a bank-2 slot (rb 17 stores bank 0 / 1)


def record(r, f, pl=PLACE):
    """record field f of row block r (r = 18.. the dummy). The C++ side computes the same values (see the .h)."""
    if f == F_PTR:
        return a_ptr(r, pl)
    if f == F_J0:
        if r == 0:
            return 0
        if r < NRB:
            return piece(r, 0, pl) - piece(r - 1, 1, pl) - 1024
        # the dummy reads row blocks 0-1 of the C object (bank 2) while rb 17 stores into banks 0 / 1
        return piece(0, 0, pl) - piece(NRB - 1, 1, pl) - 1024 if r == NRB else 0
    if f == F_J1:
        return piece(r, 1, pl) - piece(r, 0, pl) - 1024 if r < NRB else 0
    if f == F_ACQ:
        return int(r < NRB and r % 2 == 0)
    return int(r < NRB and r % 2 == 1)


# ---------------------------------------------------------------------------------------------------------------------
# the CP model
TPL = [f"P{k}" for k in range(NK)] + [f"M{k}" for k in range(NK)] + [f"L{q}" for q in range(NQ)] + [f"S{q}" for q in range(NQ)]
RBO = ["AR", "AF"] + [f"A{k}" for k in range(NK)] + ["ACQ", "REL", "RQ0", "RQ1", "RP", "RM0", "RM1", "J30", "J31", "J40", "J41"]
SLOT = dict(P="ldb", M="vec", L="lda", S="st", AR="mv", AF="lda", A="lda", ACQ="alu", REL="alu", RQ="lda", RP="lda",
            RM="lda", J3="st", J4="st")


def kind(name):
    if name.startswith(("RQ", "RM", "J3", "J4")):
        return name[:2]
    if name in ("AR", "AF", "ACQ", "REL", "RP", "BF", "BR", "BP"):
        return name
    return name[0]


def port_list(name):
    k = kind(name)
    ins = dict(P="P", M="M", L="L", S="S", AF="AF", A="A", ACQ="ACQ", REL="REL", RQ="R", RP="R", RM="R").get(k)
    return _ports(INSTR[ins]) if ins else []


def fixed_ops(g):
    """the B stream's per-row-block ops at fixed cycles (pass start = 0): fills, restart"""
    out = [(9 * i, "ldb", "BF") for i in range(8)] + [(71, "mv", "BR")]
    if g == 2:
        out.append((72, "mv", "BP"))
    elif g == 1:
        out.append((72, "lda", "BP"))
    return out


def solve(g=2, time_limit=300.0, workers=12, dmin=D_MAC_MAC, nb=4, emax=6, margin=1, hard_banks=False, pl=PLACE,
          verbose=False, hint=None, uniform=True, pls=None):
    """uniform: the 8 jobs of a row block share one template at 9-cycle offsets (g = 0 only: with g > 0 the exact cover of
    the B pass forces each job's 8 pops into 8 consecutive cycles, which no MAC chain can follow); otherwise every job of the
    row block has its own issue cycles (the body is one row block, so they may differ)."""
    from ortools.sat.python import cp_model

    R = 72 + g
    m = cp_model.CpModel()
    T, X = {}, {}

    def var(name, lo, hi):
        T[name] = m.NewIntVar(lo, hi, name)
        X[name] = {t: m.NewBoolVar(f"{name}@{t}") for t in range(lo, hi + 1)}
        m.AddExactlyOne(X[name].values())
        m.Add(T[name] == sum(t * b for t, b in X[name].items()))

    for n in TPL:  # job ops in row-block coordinates (pass start 0)
        for cb in range(NCB):
            var(f"{n}_{cb}", -48 + JOB * cb - (0 if uniform else 8), JOB * cb + R + 24 + (0 if uniform else 8))
            if uniform and cb:
                m.Add(T[f"{n}_{cb}"] == T[f"{n}_0"] + JOB * cb)
    for n in RBO:
        var(n, -R, 2 * R - 1)
    F = m.NewIntVar(0, R - 1, "F")
    # ---- resources modulo R: slots + itinerary ports
    use = defaultdict(list)
    for name, lits in X.items():
        base = name.split("_")[0]
        sl, ports = SLOT[kind(base)], port_list(base)
        for t, b in lits.items():
            use[sl, t % R].append(b)
            for st, p in ports:
                use[p, (t + st) % R].append(b)
    fixed = defaultdict(int)
    for c, sl, _ in fixed_ops(g):
        fixed[sl, c] += 1
    for (res, c), lits in use.items():
        if res != "ldb":
            m.Add(sum(lits) <= 1 - fixed.get((res, c), 0))
    for c in range(R):  # B pops exactly cover the pass's pop cycles (fills at 9 i, gap at 72..)
        m.Add(sum(use.get(("ldb", c), [])) == int(c < 72 and c % 9 != 0))

    def J(n, cb):
        return T[f"{n}_{cb}"]

    # ---- jobs
    for cb in range(NCB):
        P = [J(f"P{k}", cb) for k in range(NK)]
        M = [J(f"M{k}", cb) for k in range(NK)]
        L = [J(f"L{q}", cb) for q in range(NQ)]
        S = [J(f"S{q}", cb) for q in range(NQ)]
        for k in range(NK):
            m.Add(M[k] >= P[k] + D_POP_MAC)
            m.Add(M[k] <= P[k] + D_POP_MAC + emax)
            if k:
                m.Add(M[k] >= M[k - 1] + dmin)
        for q in range(NQ):
            m.Add(L[q] <= M[0] - D_LD_MAC)
            m.Add(S[q] >= M[NK - 1] + D_MAC_ST)
            if q:
                m.Add(L[q] >= L[q - 1] + 1)
                m.Add(S[q] >= S[q - 1] + 1)
        # p3 / p4 walk the quarters in memory order, job after job; dm[cb % 4] is reloaded by job cb + 4 (next row block's
        # cb - 4 for cb >= 4): the store of a quarter must read it before that load lands
        nxt, off = (cb + 1, 0) if cb < NCB - 1 else (0, R)
        m.Add(L[3] <= J("L0", nxt) + off - 1)
        m.Add(S[3] <= J("S0", nxt) + off - 1)
        succ, off4 = (cb + 4, 0) if cb < 4 else (cb - 4, R)
        for q in range(NQ):
            m.Add(S[q] <= J(f"L{q}", succ) + off4 + OLD_ST - margin)
    # B registers: pops (rb offset dr, cb, k) into BREG[k % nb]; a later pop into the same register may land only after the
    # earlier one's MAC read it (old-value window: MAC <= later pop + 7)
    pops = [(dr, cb, k) for dr in (-1, 0, 1) for cb in range(NCB) for k in range(NK)]
    for x in [p for p in pops if p[0] == 0]:
        for y in pops:
            if y == x or y[2] % nb != x[2] % nb or (y[0] == 0 and y <= x):
                continue
            if abs((y[0] * R + JOB * y[1]) - JOB * x[1]) > 90:
                continue
            tx, ty = J(f"P{x[2]}", x[1]), y[0] * R + J(f"P{y[2]}", y[1])
            mx, my = J(f"M{x[2]}", x[1]), y[0] * R + J(f"M{y[2]}", y[1])
            b = m.NewBoolVar("")
            m.Add(tx < ty).OnlyEnforceIf(b)
            m.Add(tx > ty).OnlyEnforceIf(b.Not())
            m.Add(mx <= ty + D_POP_MAC - 1).OnlyEnforceIf(b)
            m.Add(my <= tx + D_POP_MAC - 1).OnlyEnforceIf(b.Not())
    # ---- the row block's control ops (the previous row block's jobs at -R)
    A = [T[f"A{k}"] for k in range(NK)]
    for k in range(NK):
        for cb in range(NCB):
            m.Add(A[k] >= J(f"M{k}", cb) - R - (D_POP_MAC - 1))  # every old reader issues <= pop + 7
            m.Add(A[k] <= J(f"M{k}", cb) - D_POP_MAC)  # every new reader sees it
        m.Add(A[k] >= (A[k - 1] if k else T["AF"]) + 1)
    m.Add(T["AR"] == T["AF"] - 1)  # mov r24, #0 right before the fill (deployed)
    m.Add(T["AR"] >= A[NK - 1] - R + 1)
    # round 18 (main): the first in-situ ROT (acq.cond one bundle before the A fill) raced: wrong, run-to-run varying tokens while the
    # one-core bench (all A slots pre-released) was bit-exact. Deployed mm_ks has 6 bundles from acq.cond to the fill: ACQ_GAP
    import os
    gap = int(os.environ.get("ROT_ACQ_GAP", "1"))
    m.Add(T["ACQ"] <= T["AF"] - gap)
    m.Add(T["ACQ"] >= T["AF"] - gap - 5)
    m.Add(T["RQ0"] <= T["ACQ"] - SCL)
    m.Add(T["RQ0"] >= T["REL"] - R + 1)
    m.Add(T["RQ1"] >= T["ACQ"] + 1)
    m.Add(T["REL"] >= T["RQ1"] + SCL)
    m.Add(T["REL"] >= A[NK - 1] + int(os.environ.get("ROT_REL_GAP", "6")))
    m.Add(T["RP"] <= T["AF"] - SCL)
    m.Add(T["RP"] >= A[NK - 1] - R + 1)
    for i, (first, prev, poff) in enumerate(((0, NCB - 1, -R), (4, 3, 0))):
        j3, j4, rm = T[f"J3{i}"], T[f"J4{i}"], T[f"RM{i}"]
        m.Add(j3 >= J("L3", prev) + poff + 1)
        m.Add(j3 <= J("L0", first) - PTR_GAP)
        m.Add(j4 >= J("S3", prev) + poff + 1)
        m.Add(j4 <= J("S0", first) - PTR_GAP)
        for j in (j3, j4):
            m.Add(rm <= j - SCL)
            m.Add(rm >= j - R + 1)
    # ---- stages: stores in the last stage [F, F + R); nothing earlier than F - R or later than F + R - 1
    for name in X:
        if name.startswith("S") and "_" in name:
            m.Add(T[name] >= F)
        m.Add(T[name] >= F - R)
        m.Add(T[name] <= F + R - 1)
    E = m.NewIntVar(-2 * R, 0, "E")  # the prologue starts at E (rb 0 coordinates)
    for name in X:
        m.Add(E <= T[name])
    pro = m.NewIntVar(0, 3 * R, "pro")
    m.Add(pro == F - E)
    # ---- bank conflicts (soft unless hard_banks): co-issued lda + st in one bank, counted over the 18 row blocks and summed
    # over the placements pls (default: the one pl)
    pls = pls or [pl]
    cost = []

    def penal(expr, v, cnt, tag):
        if cnt == 0:
            return
        b = m.NewBoolVar(tag)
        m.Add(expr == v).OnlyEnforceIf(b)
        m.Add(expr != v).OnlyEnforceIf(b.Not())
        if hard_banks:
            m.Add(b == 0)
        else:
            cost.append(cnt * b)

    for q in range(NQ):
        for cb in range(NCB):
            for q2 in range(NQ):
                for cb2 in range(NCB):
                    for dr in (-1, 0, 1):  # load (r, cb) and store (r + dr, cb2) co-issue iff L - S == dr R
                        cnt = sum(1 for pl in pls for r in range(NRB) if 0 <= r + dr < NRB and
                                  bank(caddr(r, cb, q, pl)) == bank(caddr(r + dr, cb2, q2, pl)))
                        penal(J(f"L{q}", cb) - J(f"S{q2}", cb2), dr * R, cnt, "")
    for n in ["AF"] + [f"A{k}" for k in range(NK)] + ["RQ0", "RQ1", "RP", "RM0", "RM1"]:
        for q2 in range(NQ):
            for cb2 in range(NCB):
                for dr in (-2, -1, 0, 1):
                    cnt = 0
                    for pl in pls:
                        for r in range(NRB):
                            if 0 <= r + dr < NRB:
                                bu = bank(pl["rec"]) if n.startswith("R") else bank(a_ptr(r, pl))
                                cnt += bu == bank(caddr(r + dr, cb2, q2, pl))
                    penal(T[n] - J(f"S{q2}", cb2), dr * R, cnt, "")
    conf = sum(cost) if cost else 0
    m.Minimize(100 * conf + pro)
    if hint:
        for n, v in hint.items():
            if n in T:
                m.AddHint(T[n], v)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = verbose
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, dict(status=s.StatusName(st), g=g, R=R, uniform=uniform, wall=s.WallTime())
    times = {n: int(s.Value(v)) for n, v in T.items()}
    info = dict(status=s.StatusName(st), g=g, R=R, F=int(s.Value(F)), E=int(s.Value(E)), prologue=int(s.Value(pro)),
                bank_conflicts_model=int(s.Value(conf)) if cost else 0, nb=nb, dmin=dmin, margin=margin, uniform=uniform,
                bound=s.BestObjectiveBound(), objective=s.ObjectiveValue(), wall=s.WallTime())
    return times, info


# ---------------------------------------------------------------------------------------------------------------------
# emission
def op_text(n, cb, g, R, a, nb):
    k = kind(n)
    d = cb % 4 if cb is not None else None
    if k == "P":
        kk = int(n[1:])
        if g == 0 and a % R == 71:
            return f"vldb.pop.576.2d ex{BREG[kk % nb]}, [p1, lf1, r25, d2]"
        return f"vldb.pop.576 ex{BREG[kk % nb]}, [p1, lf1, r25]"
    if k == "M":
        kk = int(n[1:])
        return f"vmac.f dm{d}, dm{d}, ex{AREG[kk]}, ex{BREG[kk % nb]}, r3"
    if k == "L":
        return f"vlda {QN[int(n[1:])]}{d}, [p3], #64"
    if k == "S":
        return f"vst {QN[int(n[1:])]}{d}, [p4], #64"
    return dict(AR="mov r24, #0", AF="vlda.fill.512 [p0, lf0, r24]", ACQ="acq.cond #53, r4, r26", REL="rel.cond #52, r5, r26",
                RQ0="lda r26, [p2], #4", RQ1="lda r26, [p2], #4", RP="lda p0, [p2], #4", RM0="lda m0, [p2], #4",
                RM1="lda m1, [p2], #4", J30="padds [p3], m0", J31="padds [p3], m1", J40="padds [p4], m0",
                J41="padds [p4], m1", BF="vldb.fill.512 [p1, lf1, r25]", BR="mov r25, #0",
                BP="mov p1, p5" if g == 2 else "padda [p1], m2").get(n) or f"vlda.pop.576 ex{AREG[int(n[1:])]}, [p0, lf0, r24]"


REC_FIELD = dict(RP=F_PTR, RM0=F_J0, RM1=F_J1, RQ0=F_ACQ, RQ1=F_REL)
ORDER = ["lda", "ldb", "st", "alu", "mv", "vec"]
NOPS = dict(lda="nopa", ldb="nopb", st="nops", alu="nopx", mv="nopm", vec="nopv")


def all_ops(times, info):
    """every op of row block 0 as (absolute cycle in rb-0 coordinates, slot, name, cb or None, text, rec field or None)"""
    g, R, nb = info["g"], info["R"], info["nb"]
    out = []
    for n in TPL:
        for cb in range(NCB):
            a = times[f"{n}_{cb}"]
            out.append((a, SLOT[kind(n)], n, cb, op_text(n, cb, g, R, a, nb), None))
    for n in RBO:
        out.append((times[n], SLOT[kind(n)], n, None, op_text(n, None, g, R, times[n], nb), REC_FIELD.get(n)))
    for c, sl, n in fixed_ops(g):
        out.append((c, sl, n, None, op_text(n, None, g, R, c, nb), None))
    return out


def bundle(ops, full):
    by = {}
    for sl, txt in ops:
        assert sl not in by, (sl, txt, by)
        by[sl] = txt
    if full:
        return ";\t\t".join(by.get(s, NOPS[s]) for s in ORDER)
    return ";\t\t".join(by[s] for s in ORDER if s in by) if by else "nop"


def emit(times, info, name="mm_rot", full_pro=frozenset()):
    """asm text + the record order [(rb, field)] + the B order (stream position -> (cb, kp)). full_pro: prologue bundle
    indices written in the full 16-byte form (alignment fitting, see fit())."""
    g, R, F, nb = info["g"], info["R"], info["F"], info["nb"]
    ops = all_ops(times, info)
    body = defaultdict(list)  # frame cycle u -> [(slot, text, stage, rec field)]
    for a, sl, n, cb, txt, fld in ops:
        stage = (a - F) // R
        assert stage in (-1, 0), (n, cb, a, F)
        body[(a - F) % R].append((sl, txt, stage, fld))
    # prologue: frame -1 = rb 0's stage -1 ops, plus the B stream from the last 576 B line boundary before rb 0's first pop:
    # rb 0's earliest pops sit in pass -1, so pass -1 restarts there (p1 = B + 576 i0, a fresh stream: MACHINE R11 "restarts
    # after 576 B may continue") and every B-stream op of rb -1 from then on runs (its pops land in registers nobody reads)
    neg = [a + R for a, sl, n, cb, txt, fld in ops if kind(n) == "P" and a < 0]
    i0 = min(neg) // 9 if neg else 8
    s0 = -R + 9 * i0 if neg else 0
    pro = defaultdict(list)
    for a, sl, n, cb, txt, fld in ops:
        if a < F:
            pro[a].append((sl, txt, fld))
        if kind(n) in ("P", "BF", "BR", "BP") and s0 <= a - R < F:
            pro[a - R].append((sl, txt, None))
    E = min(list(pro) + [F])
    lines = ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    setup = ["movs p3, p2", "movs p4, p2", "movs p5, p1", "mova r3, #780", "mova r25, #0", "mova r24, #0", "mova r4, #-1",
             "mova r5, #1", "movs p2, p0", f"movxm ls, #.L{name}_body", f"movxm le, #.L{name}_end", f"mova r0, #{NRB}",
             "add.nc lc, r0, #0"]
    if neg:
        setup += [f"movxm m3, #{576 * i0}", "padda [p1], m3"]
    if g == 1:
        setup += ["movxm m2, #-4608"]
    if g == 0:
        setup += ["movxm m2, #-4544", "mova dn2, #0", "mova dj2, #64", "mova dc2, #0"]
    lines += ["\t" + s for s in setup]
    lines += ["\t.p2align\t4", f".L{name}_pro:"]
    order = []
    for i, a in enumerate(range(E, F)):
        lines.append("\t" + bundle([(sl, t) for sl, t, _ in pro.get(a, [])], i in full_pro))
        for sl, t, fld in sorted(pro.get(a, []), key=lambda o: ORDER.index(o[0])):
            if fld is not None:
                order.append((0, fld))
    lines += ["\t.p2align\t4", f".L{name}_body:", f"{name}_start:"]
    for u in range(R):
        if u == R - 1:
            lines += ["\t.p2align\t4", f".L{name}_end:", f"{name}_last:"]
        lines.append("\t" + bundle([(sl, t) for sl, t, _, _ in body[u]], True))
    for f in range(NRB):
        for u in range(R):
            for sl, t, stage, fld in body[u]:
                if fld is not None:
                    order.append((f - stage, fld))
    lines += ["\tret lr"] + ["\tnop"] * 5 + [f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    # B order: the pops of one pass in time order (pass = cycles [0, 72) of a row block; any row block's jobs)
    bo = {}
    for dr in (-1, 0, 1):
        for cb in range(NCB):
            for k in range(NK):
                a = dr * R + times[f"P{k}_{cb}"]
                if 0 <= a < 72:
                    bo[a] = (cb, k)
    border = [bo[a] for a in sorted(bo)]
    assert len(border) == 64 and len(set(border)) == 64, len(border)
    return "\n".join(lines) + "\n", order, border


def header(order, border, info, name="mm_rot"):
    nm = name.upper()
    out = [f"// generated by npu-research/r19/gen_rot.py (g = {info['g']}, R = {info['R']} bundles per row block): {name}'s "
           "record stream and B order",
           f"// records: one int32 per (row block, field) in execution order; fields: " +
           ", ".join(f"{i} = {s}" for i, s in enumerate(FIELD_NAMES)),
           f"//   row block 18 = the dummy: A pointer = slot 1, jump 0 = piece(0, 0) - piece(17, 1) - 1024 (reads in bank 2), jump 1 = 0, flags 0",
           f"#define {nm}_NREC {len(order)}",
           f"static constexpr int8_t {name}_order[{nm}_NREC][2] = {{" + ", ".join(f"{{{a}, {b}}}" for a, b in order) + "};",
           "// B stream position i (72 B each) holds panel pair4[...] of pack8's pair4 order: pair * 32 + kp * 4 + cb % 4",
           f"static constexpr int8_t {name}_border[64] = {{" +
           ", ".join(str((cb >> 2) * 32 + k * 4 + (cb & 3)) for cb, k in border) + "};"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# replay of the emitted asm
class Fail(Exception):
    pass


def parse_s(text, name="mm_rot"):
    """bundles (lists of instruction strings) of setup + prologue, body, tail; trip count"""
    pre, body, tail, cur = [], [], [], None
    cur = pre
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("//", "#")):
            continue
        if s == f".L{name}_body:":
            cur = body
            continue
        if s.startswith(".") or s.endswith(":"):
            if s.startswith(".p2align") and cur is body and body:
                continue
            continue
        if s.startswith("ret"):
            cur = tail
        ins = [i.strip() for i in s.split(";") if i.strip() and not re.fullmatch(r"nop[abxmvs]?", i.strip())]
        cur.append(ins)
    return pre, body, tail


def check(text, info, pl=PLACE, pad=0, name="mm_rot", order=None, border=None, verbose=False):
    """replay one call; returns stats or raises Fail. pad = the .p2align nop bundle before the body (0 / 1)."""
    pre, body, tail = parse_s(text, name)
    trips = None
    for b in pre:
        for i in b:
            mm = re.fullmatch(r"mova r0, #(\d+)", i)
            if mm:
                trips = int(mm.group(1))
    assert trips == NRB
    seq = [(b, -1) for b in pre] + [([], -1)] * pad + [(b, f) for f in range(trips) for b in body] + [(b, trips) for b in tail[:6]]
    # memory: C quarters, A slots, B object, records
    mem = {}
    for rb in range(NRB):
        for cb in range(NCB):
            for q in range(NQ):
                mem[caddr(rb, cb, q, pl)] = ("acc", rb, cb, q, 0)
    cq = set(mem)
    table = [record(r, f, pl) for r, f in order]
    b0 = pl["b"]
    regs = defaultdict(list)  # reg -> [(issue, wmin, wmax, value)]
    errs, notes = [], []
    st = dict(macs=0, stalls=0, stall_where=defaultdict(int), acq=[], rel=[], slot_owner={}, last_read=defaultdict(int),
              stores=0, loads=0)
    fifo = {}
    for r, v in {"p0": pl["rec"], "p1": pl["b"], "p2": piece(0, 0, pl)}.items():  # mm_rot(rec, pB, piece0)
        regs[r].append((-100, 1, 1, v))

    def err(c, msg):
        errs.append(f"cycle {c}: {msg}")
        if len(errs) > 30:
            raise Fail("\n".join(errs))

    def wr(c, r, v, wmin, wmax=None):
        regs[r].append((c, wmin, wmax if wmax is not None else wmin, v))

    def rd(c, r, stage=1, ok_tie=False):
        best, bl, unsure = None, None, []
        for ci, wmin, wmax, v in regs[r]:
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
                if own is None or own[1]:  # never acquired / released
                    return "G"
                st["last_read"][own[0]] = c
                i = o // 72
                return ("A", 2 * own[0] + i // 8, i % 8)
        return "G"

    def b_val(c, addr):
        o = addr - b0
        if not (0 <= o < 4608 and o % 72 == 0):
            err(c, f"B pop at B + {o}")
            return "G"
        return ("B",) + tuple(border[o // 72])

    def stream(c, lf, ptr, pos, is_pop, two_d=False):
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
                    err(c, f"{lf}: FIFO underflow (pop {s['C'] // 72 + 1} after {s['lines'] - 1} lines)")
                val = s["base"] + s["C"]
                s["C"] += 72
        step = 64
        if two_d:
            if rd(c, "dn2") == 0:
                step = rd(c, "m2")
        wr(c, ptr, p + step, 1)
        wr(c, pos, "S", 1)
        return val, p

    c = -1
    for bund, frame in seq:
        c += 1
        acc_banks = []
        writes = []
        for ins in bund:
            mm = None
            if (mm := re.fullmatch(r"movs (p\d), (p\d)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)) if regs[mm.group(2)] else ARGS[mm.group(2)], 1, 2))
            elif (mm := re.fullmatch(r"mova (\w+), #(-?\d+)", ins)):
                v = int(mm.group(2))
                writes.append((mm.group(1), "Z" if mm.group(1) in ("r24", "r25") and v == 0 else v, 1, 1))
            elif (mm := re.fullmatch(r"movxm (\w+), #(\S+)", ins)):
                v = mm.group(2)
                writes.append((mm.group(1), int(v) if re.fullmatch(r"-?\d+", v) else v, 1, 1))
            elif ins.startswith("add.nc lc"):
                pass
            elif (mm := re.fullmatch(r"mov (r2[45]), #0", ins)):
                writes.append((mm.group(1), "Z", 1, 1))
            elif ins == "mov p1, p5":
                writes.append(("p1", rd(c, "p5"), 1, 2))
            elif (mm := re.fullmatch(r"padda \[p1\], (m[23])", ins)):
                writes.append(("p1", rd(c, "p1") + rd(c, mm.group(1)), 1, 1 if mm.group(1) == "m2" else PTR_GAP))
            elif (mm := re.fullmatch(r"vldb\.fill\.512 \[p1, lf1, r25\]", ins)):
                _, p = stream(c, "lf1", "p1", "r25", False)
                acc_banks.append(bank(p))
            elif (mm := re.fullmatch(r"vldb\.pop\.576(\.2d)? ex(\d+), \[p1, lf1, r25(, d2)?\]", ins)):
                a, p = stream(c, "lf1", "p1", "r25", True, bool(mm.group(1)))
                acc_banks.append(bank(p))
                writes.append((f"ex{mm.group(2)}", b_val(c, a) if a is not None else "G", 8, 8))
            elif ins == "vlda.fill.512 [p0, lf0, r24]":
                _, p = stream(c, "lf0", "p0", "r24", False)
                acc_banks.append(bank(p))
                a_val(c, p)
            elif (mm := re.fullmatch(r"vlda\.pop\.576 ex(\d+), \[p0, lf0, r24\]", ins)):
                a, p = stream(c, "lf0", "p0", "r24", True)
                acc_banks.append(bank(p))
                writes.append((f"ex{mm.group(1)}", a_val(c, a) if a is not None else "G", 8, 8))
            elif (mm := re.fullmatch(r"vmac\.f dm(\d), dm\1, ex(\d+), ex(\d+), r3", ins)):
                d = mm.group(1)
                accs = [rd(c, f"dm{d}.{q}", 4) for q in range(NQ)]
                av, bv = rd(c, f"ex{mm.group(2)}"), rd(c, f"ex{mm.group(3)}")
                if rd(c, "r3") != 780:
                    err(c, "r3 != 780")
                ok = all(isinstance(x, tuple) and x[0] == "acc" for x in accs) and len({x[1:3] + x[4:] for x in accs}) == 1 \
                    and [x[3] for x in accs] == [0, 1, 2, 3]
                if ok:
                    _, rb, cb, _, n = accs[0]
                    ok = av == ("A", rb, n) and bv == ("B", cb, n) and n < NK
                if ok:
                    st["macs"] += 1
                elif all(isinstance(x, tuple) for x in accs) and verbose:
                    notes.append(f"cycle {c} frame {frame}: {ins}: acc {accs[0]}, A {av}, B {bv}")
                for q in range(NQ):
                    writes.append((f"dm{d}.{q}", ("acc", rb, cb, q, n + 1) if ok else "G", 6, 6))
            elif (mm := re.fullmatch(r"vlda (bm\w\w)(\d), \[p3\], #64", ins)):
                p = rd(c, "p3")
                acc_banks.append(bank(p))
                q = QN.index(mm.group(1))
                v = mem.get(p, "G")
                writes.append((f"dm{mm.group(2)}.{q}", v, 7, 7))
                writes.append(("p3", p + 64, 1, 1))
                st["loads"] += 1
            elif (mm := re.fullmatch(r"vst (bm\w\w)(\d), \[p4\], #64", ins)):
                p = rd(c, "p4")
                acc_banks.append(bank(p))
                q = QN.index(mm.group(1))
                v = rd(c, f"dm{mm.group(2)}.{q}")
                if p not in cq:
                    err(c, f"stray store to {p:#x} ({v})")
                else:
                    want = mem[p][:4] + (NK,)
                    if v != want:
                        err(c, f"store to {p:#x}: {v}, want {want}")
                    mem[p] = v
                    st["stores"] += 1
                writes.append(("p4", p + 64, 1, 1))
            elif (mm := re.fullmatch(r"padds \[(p[34])\], (m\d)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(1)) + rd(c, mm.group(2)), 1, PTR_GAP))
            elif (mm := re.fullmatch(r"lda (\w+), \[p2\], #4", ins)):
                p = rd(c, "p2")
                acc_banks.append(bank(p))
                i = (p - pl["rec"]) // 4
                if not 0 <= i < len(table):
                    err(c, f"record {i} out of the table")
                    v = 0
                else:
                    v = table[i]
                writes.append((mm.group(1), v, 1, SCL))
                writes.append(("p2", p + 4, 1, 1))
            elif ins == "acq.cond #53, r4, r26":
                if rd(c, "r26"):
                    if rd(c, "r4") != -1:
                        err(c, "acq r4 != -1")
                    s = len(st["acq"])
                    st["acq"].append(c)
                    st["slot_owner"][s % 3] = (s, False)
            elif ins == "rel.cond #52, r5, r26":
                if rd(c, "r26"):
                    s = len(st["rel"])
                    st["rel"].append(c)
                    if st["slot_owner"].get(s % 3, (None,))[0] != s:
                        err(c, f"release of stripe {s}: slot holds {st['slot_owner'].get(s % 3)}")
                    else:
                        st["slot_owner"][s % 3] = (s, True)
                    if c < st["last_read"].get(s, -99) + 6:
                        err(c, f"release of stripe {s} {c - st['last_read'][s]} cycles after its last A read")
            elif ins.startswith("ret"):
                pass
            else:
                raise Fail(f"unknown instruction {ins!r}")
        for r, v, wmin, wmax in writes:
            wr(c, r, v, wmin, wmax)
        cnt = defaultdict(int)
        for bk in acc_banks:
            cnt[bk] += 1
        if 0 <= frame < NRB:
            st["body_cycles"] = st.get("body_cycles", 0) + 1
            for bk in cnt:
                st.setdefault("busy", defaultdict(int))[bk] += 1
        sl = max(cnt.values(), default=1) - 1
        if sl > 0:
            st["stalls"] += sl
            st["stall_where"]["prologue" if frame < 0 else "body" if frame < NRB else "tail"] += sl
            st.setdefault("stall_list", []).append((c, frame, sl, " | ".join(bund)))
    bad = [(a, v) for a, v in mem.items() if a in cq and not (isinstance(v, tuple) and v[4] == NK)]
    if bad:
        err(c, f"{len(bad)} quarters not finished, e.g. {bad[0][0]:#x}: {bad[0][1]}")
    if st["macs"] != MACS:
        err(c, f"{st['macs']} valid MACs, want {MACS}")
    if len(st["acq"]) != RHO or len(st["rel"]) != RHO:
        err(c, f"{len(st['acq'])} acquires / {len(st['rel'])} releases, want {RHO}")
    if errs:
        raise Fail("\n".join(notes[:12] + errs))
    st["cycles"] = c + 1
    st["bundles_pre"] = len(pre)
    return st


ARGS = {"p0": PLACE["rec"], "p1": PLACE["b"], "p2": PLACE["c_obj"]}  # mm_rot(const int32_t *rec, const bfp16ebs8 *pB, float *piece0)


def run_check(text, info, order, border, name="mm_rot"):
    res = {}
    for pad in (0, 1):
        res[pad] = check(text, info, pad=pad, name=name, order=order, border=border, verbose=True)
    return res




PEANO = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin"


def assemble(text, work, name="mm_rot"):
    """assemble in the npu-iron container; returns (object path, -z disassembly, symbol table {name: addr})"""
    import subprocess

    work = Path(work)
    s, o = work / f".fit_{name}.s", work / f".fit_{name}.o"
    s.write_text(text)
    subprocess.run([f"{PEANO}/clang", "--target=aie2p-none-unknown-elf", "-c", str(s), "-o", str(o)], check=True)
    dis = subprocess.run([f"{PEANO}/llvm-objdump", "-d", "-z", "--no-show-raw-insn", str(o)], capture_output=True, text=True,
                         check=True).stdout
    syms = subprocess.run([f"{PEANO}/llvm-objdump", "-t", str(o)], capture_output=True, text=True, check=True).stdout
    addr = {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+)\s.*\s(\S+)$", syms, re.M)}
    return o, dis, addr


def bundle_addrs(dis):
    return [int(m.group(1), 16) for m in re.finditer(r"^\s*([0-9a-f]+):", dis, re.M)]


def fit(times, info, work, name="mm_rot"):
    """expand the cheapest set of prologue bundles to the full 16-byte form so that the prologue ends 16 B aligned: the
    .p2align before the loop body then inserts nothing (a pad bundle shifts the prologue a cycle against the body)"""
    text, order, border = emit(times, info, name=name)
    _, dis, addr = assemble(text, work, name)
    offs = bundle_addrs(dis)
    p0 = _pro_start(text, offs, addr, name)
    npro = -_E(text, name)
    i0 = offs.index(p0)
    pro = offs[i0:i0 + npro + 1]
    sizes = [b - a for a, b in zip(pro, pro[1:])]
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
    assert need in best, f"no expansion set reaches residue {need}"
    full = best[need][1]
    text, order, border = emit(times, info, name=name, full_pro=full)
    o, dis, addr = assemble(text, work, name)
    offs = bundle_addrs(dis)
    i0 = offs.index(pro[0])
    assert offs[i0 + npro] == addr[f"{name}_start"], "a pad bundle sits between the prologue and the body"
    assert addr[f"{name}_last"] % 16 == 0 and addr[f"{name}_start"] % 16 == 0
    return text, order, border, dict(expanded=sorted(full), added=best[need][0], text_bytes=addr[f".Lfunc_end_{name}"]
                                     if f".Lfunc_end_{name}" in addr else None, obj=str(o), dis=dis)


def _E(text, name):
    """prologue length in bundles = lines between the prologue label and the body's .p2align"""
    ls = text.splitlines()
    a = ls.index(f".L{name}_pro:")
    b = ls.index(f".L{name}_body:")
    return -sum(1 for x in ls[a + 1:b] if not x.strip().startswith(".") and not x.strip().endswith(":"))


def _pro_start(text, offs, addr, name):
    """the prologue's first bundle: setup bundles, then the first 16 B aligned bundle (the setup's .p2align pad, if any,
    is a separate executable nop bundle before it)"""
    ls = text.splitlines()
    a = ls.index(f"{name}:")
    b = ls.index(f".L{name}_pro:")
    n_setup = sum(1 for x in ls[a + 1:b] if not x.strip().startswith(".") and not x.strip().endswith(":"))
    return next(o for o in offs[n_setup:] if o % 16 == 0)


def parse_place(spec):
    """"c_obj=0x..,acc_hi=0x..,slots=0x..:0x..:0x..,rec=0x..,b=0x.." (missing keys: the r17d placement)"""
    pl = dict(PLACE)
    for kv in (spec.split(",") if spec else []):
        k, v = kv.split("=")
        pl[k] = tuple(int(x, 16) for x in v.split(":")) if k == "slots" else int(v, 16)
    return pl


# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("solve")
    a.add_argument("--g", type=int, default=2)
    a.add_argument("--time", type=float, default=300)
    a.add_argument("--workers", type=int, default=12)
    a.add_argument("--dmin", type=int, default=D_MAC_MAC)
    a.add_argument("--nb", type=int, default=4)
    a.add_argument("--hard-banks", action="store_true")
    a.add_argument("--free", action="store_true", help="every job of the row block its own issue cycles (needed for g > 0)")
    a.add_argument("--place", action="append", default=[], help="placements to keep conflict-free (repeatable; check's format); "
                   "default the r17d one")
    a.add_argument("--json", required=True)
    a.add_argument("-v", action="store_true")
    b = sub.add_parser("emit")
    b.add_argument("json")
    b.add_argument("--out", required=True)
    b.add_argument("--full-pro", default="", help="comma list of prologue bundle indices in the full form (fit)")
    f = sub.add_parser("fit", help="emit + prologue alignment fit + assemble (npu-iron container) + check pad 0")
    f.add_argument("json")
    f.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("s")
    c.add_argument("--json", required=True)
    c.add_argument("--place", help="c_obj=0x..,acc_hi=0x..,slots=0x..:0x..:0x..,rec=0x..,b=0x.. (default: the deployed r17d)")
    o = ap.parse_args()
    if o.cmd == "solve":
        times, info = solve(o.g, o.time, o.workers, dmin=o.dmin, nb=o.nb, hard_banks=o.hard_banks, verbose=o.v,
                            uniform=not o.free, pls=[parse_place(x) for x in o.place] or None)
        info["places"] = o.place
        print(json.dumps(info))
        if times is None:
            raise SystemExit(1)
        Path(o.json).write_text(json.dumps(dict(times=times, info=info), indent=1))
        for cb in range(NCB):
            print(f"job cb {cb}:", {n: times[f"{n}_{cb}"] for n in TPL})
        print("row-block ops:", {n: times[n] for n in RBO})
    elif o.cmd == "emit":
        d = json.loads(Path(o.json).read_text())
        fp = frozenset(int(x) for x in o.full_pro.split(",") if x)
        text, order, border = emit(d["times"], d["info"], full_pro=fp)
        Path(o.out).write_text(text)
        Path(o.out).with_suffix(".h").write_text(header(order, border, d["info"]))
        d["order"], d["border"] = order, border
        Path(o.json).write_text(json.dumps(d, indent=1))
        res = run_check(text, d["info"], order, border)
        for pad, st in res.items():
            print(f"pad {pad}: {st['cycles']} cycles per call (setup + prologue {st['bundles_pre'] + pad}, {NRB} x {d['info']['R']}, "
                  f"ret 6), {st['macs']} MACs checked, {st['stores']} stores, bank stalls {st['stalls']} "
                  f"{dict(st['stall_where'])}")
    elif o.cmd == "fit":
        d = json.loads(Path(o.json).read_text())
        out = Path(o.out)
        text, order, border, fi = fit(d["times"], d["info"], out.parent)
        out.write_text(text)
        out.with_suffix(".h").write_text(header(order, border, d["info"]))
        out.with_suffix(".dis").write_text(fi.pop("dis"))
        d["order"], d["border"], d["fit"] = order, border, fi
        Path(o.json).write_text(json.dumps(d, indent=1))
        st = check(text, d["info"], pad=0, order=order, border=border, verbose=True)
        print(f"fit: prologue bundles {fi['expanded']} in the full form (+{fi['added']} B), no pad before the body")
        print(f"check pad 0: {st['cycles']} cycles per call, {st['macs']} MACs, {st['stores']} stores, bank stalls {st['stalls']} "
              f"{dict(st['stall_where'])}")
    else:
        d = json.loads(Path(o.json).read_text())
        text = Path(o.s).read_text()
        pl = parse_place(o.place)
        st = check(text, d["info"], pl=pl, pad=0, order=[tuple(x) for x in d["order"]], border=[tuple(x) for x in d["border"]])
        print(f"{st['cycles']} cycles per call (no stalls), {st['macs']} MACs, bank stalls {st['stalls']} "
              f"{dict(st['stall_where'])}: {st['cycles'] + st['stalls']} on this placement")


if __name__ == "__main__":
    main()
