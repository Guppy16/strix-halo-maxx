#!/usr/bin/env python3
"""gen_e7.py - E7'': the ROT k-step with B on BOTH FIFOs, A merged into lf0, B split over two L1 banks, R = 64 (research pass 17).

ROT (r20/gen_rot.py, deployed as g0a) is bound at R = 72 bundles per row block by ONE thing: the 72 B line reads of a row block
(64 pops + 8 fills of the 4,608 B object) all come from lf1 in bank 3. The MAC bound is 64. E7'' removes the 72:

  * B is popped from BOTH FIFOs: lf1 carries N1 panels (one segment), lf0 carries A (fill + 8 pops, as today) and two B
    segments, X (NX panels) and Y (NY panels); every segment restarts with `mov r2x, #0` + an explicit pointer + a fill
    (MACHINE R7 / SKILL: restarts with an explicit pointer are always safe);
  * the B object is split over two banks by the core's S2MM: object bytes [0, YB) -> the Y buffer in bank 3 (lf0-Y panels, then
    lf1 panels), [YB, 4608) -> the X buffer in bank 0 (the lf0-X panels). Bank 3 then sees N1 + NY panels + their fills per row
    block (63 for 32 / 24), bank 0 the X segment (9) + records (5) + the C pieces that live there (<= 32);
  * a FIFO op may issue on the lda or the ldb slot (vlda.pop / vldb.pop on either FIFO: POPLF1 MEASURED for lf1; lf0 on ldb is
    the same encoding class, UNVERIFIED), so the 118 loads of a row block (32 C quarters + 5 records + 81 line reads) share 128
    load slots;
  * jobs, accumulator rotation, record stream, A refresh, lock flags and the dummy row block are ROT's (gen_rot), with the
    round-18 lock distances as HARD constraints: acquire >= ACQ_GAP (6) bundles before the A fill, release >= REL_GAP (8) after
    the last A pop.

The CP model matches the 64 job pops to the stream's B pop positions by residue (mod R): the host packs the B object in stream
order, so the only requirement is that a pop happens at the time the job wants it; the panel <-> position pairing (border) is
read off the solution.

usage (test machine; CP-SAT via uv, nice, <= 12 workers):
  nice -n 10 ~/.local/bin/uv run --no-project --with ortools --with numpy python r22/gen_e7.py solve --R 64 --json r22/e7_r64.json
  docker ... npu-iron python3 r22/gen_e7.py fit r22/e7_r64.json --out r22/mm_e7_r64.s     (emit + alignment fit + assemble + replay)
  python3 r22/gen_e7.py check r22/mm_e7_r64.s --json r22/e7_r64.json [--buf 1] [--place acc_hi=0x..]   (replay only)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "r20"))  # main round 22: copy of research r22/gen_e7.py
import gen_rot as G  # noqa: E402
from gen_rot import (ACQ_GAP, AREG, BREG, D_LD_MAC, D_MAC_MAC, D_MAC_ST, D_POP_MAC, F_ACQ, F_J0, F_J1, F_PTR, F_REL,  # noqa: E402
                     FIELD_NAMES, MACS, NCB, NK, NQ, NRB, OLD_ST, PTR_GAP, QN, REL_GAP, RHO, SCL, Fail, a_ptr, bank, caddr,
                     piece, record)

# E7L placement (PROPOSALS rev 17): X buffers in bank 0 before core_data (aexp_ab moved to .aie.bank3), core_data from 0x70E80,
# so rot_rec ~0x71784 and acc_hi ~0x71940; Y buffers 4,224 B (the DMA writes 4,032; the attention's -m rows keep kt + 3456..4224).
PLACE = dict(c_obj=0x78900, acc_hi=0x71940, slots=(0x77B80, 0x78000, 0x78480), rec=0x71784, x=(0x70A00, 0x70C40),
             y=(0x7D100, 0x7E180))
OBJ = 4608
DROP = set(filter(None, os.environ.get("E7_DROP", "").split(",")))  # diagnostics: constraint groups to leave out


def seg_ops(n):
    """a stream segment of n pops: a fill before every 8 pops (FIFO-POP9). Returns ['F', 'P', ...]"""
    out = []
    for i in range(n):
        if i % 8 == 0:
            out.append("F")
        out.append("P")
    return out


class Cfg:
    """n1b > 0 splits lf1 into L (n1 - n1b panels) and N (n1b) with a restart between them. aligned: every stream segment
    instance serves the pops of ONE row block (offset delta[s] periods): what a k-step boundary inside one modulo loop (KTL)
    needs, because consecutive k-steps read different B buffers. Residue matching (aligned = False) lets an lf1 instance serve
    two row blocks (fine inside one k-step)."""

    def __init__(self, R=64, n1=32, nx=8, n1b=0, aligned=False, delta=None, ktl=False):
        self.R, self.n1, self.nx, self.n1b, self.aligned, self.ktl = R, n1, nx, n1b, aligned or ktl, ktl
        self.ny = 64 - n1 - nx
        assert self.ny > 0 and nx > 0 and n1 > n1b >= 0
        self.segs = {"L": seg_ops(n1 - n1b), "X": seg_ops(nx), "Y": seg_ops(self.ny)}
        if n1b:
            self.segs["N"] = seg_ops(n1b)
        self.delta = {**dict(L=0, N=-1, X=0, Y=0), **(delta or {})}
        self.yb = 72 * (self.ny + n1)  # bytes of the object in the Y buffer
        assert OBJ - self.yb == 72 * nx

    def d(self):
        return dict(R=self.R, n1=self.n1, nx=self.nx, n1b=self.n1b, aligned=self.aligned, delta=self.delta, ktl=self.ktl)


def cfg_of(info):
    return Cfg(info["R"], info["n1"], info["nx"], info.get("n1b", 0), info.get("aligned", False), info.get("delta"),
               info.get("ktl", False))


# KTL (one modulo loop over a tile's k-steps 1..nks-1, research pass 17): per-row-block ops beside ROT's
KTL_ALU = ("CNT", "MA", "MB", "MB2")  # counter, A-acquire mask, B-acquire mask (r27 - 1) x flag
RESTART = dict(X="XP", Y="YP", L="BP", N="NP")


def ktl_ops(cfg):
    return KTL_ALU + ("ACQB", "RELB", "RQ2") + tuple(f"T{m}{sg}" for sg in cfg.segs for m in "MX")


def sname(s, i):
    return f"{s}{i:02d}"


JOBTPL = [f"P{k}" for k in range(NK)] + [f"M{k}" for k in range(NK)] + [f"L{q}" for q in range(NQ)] + [f"S{q}" for q in range(NQ)]
LDA_ONLY = ("RQ0", "RQ1", "RP", "RM0", "RM1")
MV = ("AR", "XR", "XP", "YR", "YP", "BR", "BP", "NR")
REC = ("RQ0", "RQ1", "RP", "RM0", "RM1")
REC_FIELD = dict(RP=F_PTR, RM0=F_J0, RM1=F_J1, RQ0=F_ACQ, RQ1=F_REL)


def fifo_names(cfg):
    """[(name, fifo, segment, index, 'F'|'P')] of one row block's stream ops in stream order per FIFO"""
    out = [("AF", "lf0", "A", 0, "F")] + [(f"A{k}", "lf0", "A", k + 1, "P") for k in range(NK)]
    for s in ("X", "Y"):
        out += [(sname(s, i), "lf0", s, i, o) for i, o in enumerate(cfg.segs[s])]
    for s in ("L", "N"):
        if s in cfg.segs:
            out += [(sname(s, i), "lf1", s, i, o) for i, o in enumerate(cfg.segs[s])]
    return out


def solve(cfg, time_limit=600.0, workers=12, emax=6, nb=4, margin=1, hard_banks=False, pls=None, verbose=False, absolute=False,
          hint=None):
    from ortools.sat.python import cp_model

    R = cfg.R
    pls = pls or [PLACE]
    m = cp_model.CpModel()
    T, X = {}, {}

    def var(name, lo, hi):
        T[name] = m.NewIntVar(lo, hi, name)
        X[name] = {t: m.NewBoolVar(f"{name}@{t}") for t in range(lo, hi + 1)}
        m.AddExactlyOne(X[name].values())
        m.Add(T[name] == sum(t * b for t, b in X[name].items()))

    per = R // NCB
    for n in JOBTPL:
        for cb in range(NCB):
            var(f"{n}_{cb}", -56 + per * cb, per * cb + R + 32)
    FN = fifo_names(cfg)
    for name, *_ in FN:
        var(name, -R, 2 * R - 1)
    for n in ("ACQ", "REL", "J30", "J31", "J40", "J41") + REC + tuple(x for x in MV if x != "NR" or cfg.n1b) + \
            (("NP",) if cfg.n1b else ()):
        var(n, -R, 2 * R - 1)
    if cfg.ktl:
        for n in ktl_ops(cfg):
            var(n, -R, 2 * R - 1)
    F = m.NewIntVar(0, R - 1, "F")

    def J(n, cb):
        return T[f"{n}_{cb}"]

    def res(name):
        return defaultdict(list, {t % R: [] for t in range(R)})

    # ---- per-residue literal lists
    lits = defaultdict(lambda: defaultdict(list))  # group -> residue -> [bool]
    bank3x = []
    for name, xs in X.items():
        base = name.split("_")[0]
        groups = []
        if base[0] == "M" and "_" in name:
            groups.append("vec")
        elif base[0] == "L" and "_" in name:
            groups.append("lda")
        elif base[0] == "S" and "_" in name:
            groups.append("st")
        elif base[0] == "P" and "_" in name:
            groups.append("bdemand")
        elif name in LDA_ONLY or name == "RQ2":
            groups += ["lda", "scalar"]
        elif cfg.ktl and (name in KTL_ALU or name in ("ACQB", "RELB") or (name[0] == "T" and name[1] in "MX")):
            groups.append("alu")
        elif cfg.ktl and name in ("XP", "YP", "BP"):  # KTL restarts: movs pN, rBase (st slot), the bases toggle per k-step
            groups.append("st")
        elif name in ("J30", "J31", "J40", "J41", "NP"):
            groups.append("st")
        elif name in MV:
            groups.append("mv")
        elif name in ("ACQ", "REL"):
            groups.append("alu")
        for t, b in xs.items():
            for gname in groups:
                lits[gname][t % R].append(b)
    for name, fifo, seg, i, o in FN:
        for t, b in X[name].items():
            lits["fifo"][t % R].append(b)
            if o == "P":
                lits["fpop"][t % R].append(b)
                if seg != "A":
                    lits["bpos"][t % R].append(b)
            if seg in ("L", "N", "Y"):
                lits["bank3"][t % R].append(b)
            if seg == "X":
                lits["bank0"][t % R].append(b)
    for n in REC + (("RQ2",) if cfg.ktl else ()):
        for t, b in X[n].items():
            lits["bank0"][t % R].append(b)
    for n in ("ACQ", "REL") + (("ACQB", "RELB") if cfg.ktl else ()):  # LCKREQ at stages 0..3 (aie2p_isa_timing.json): an acquire and a release >= 4 apart
        for t, b in X[n].items():
            for st in range(4):
                lits["lckreq"][(t + st) % R].append(b)
    for c in range(R):
        L = lits
        m.Add(sum(L["vec"][c]) <= 1)
        m.Add(sum(L["lda"][c]) <= 1)
        if "slots" not in DROP:
            m.Add(sum(L["fifo"][c]) + sum(L["lda"][c]) <= 2)
        m.Add(sum(L["st"][c]) <= 1)
        m.Add(sum(L["mv"][c]) <= 1)
        m.Add(sum(L["alu"][c]) <= 1)
        m.Add(sum(L["lckreq"][c]) <= 1)
        if "bank3soft" in DROP:  # the cap as a cost (excess bank-3 reads per residue): LNS finds 0 more easily than a hard cap
            ex = m.NewIntVar(0, 2, f"ex3_{c}")
            m.Add(ex >= sum(L["bank3"][c]) - 1)
            bank3x.append(ex)
        elif "bank3" not in DROP:
            m.Add(sum(L["bank3"][c]) <= 1)  # lf1 and lf0-Y both read bank 3
        if "bank0" not in DROP:
            m.Add(sum(L["bank0"][c]) <= 1)  # lf0-X and the record loads both read bank 0
        if not absolute and not cfg.aligned:
            m.Add(sum(L["bdemand"][c]) == sum(L["bpos"][c]))  # every job pop finds a B pop position at its time (mod R)
    if cfg.aligned:  # each segment instance serves one row block: match job pops to positions at time + delta * R
        dem, pos = defaultdict(list), defaultdict(list)
        for name, xs in X.items():
            if name[0] == "P" and "_" in name:
                for t, b in xs.items():
                    dem[t].append(b)
        for name, fifo, seg, i, o in FN:
            if o == "P" and seg != "A":
                for t, b in X[name].items():
                    pos[t + cfg.delta[seg] * R].append(b)
        for t in set(dem) | set(pos):
            m.Add(sum(dem[t]) == sum(pos[t]))
    elif absolute:  # rb 0s pops are served by rb 0s own stream instances: the prologue needs no earlier stream segment
        dem, pos = defaultdict(list), defaultdict(list)
        for name, xs in X.items():
            if name[0] == "P" and "_" in name:
                for t, b in xs.items():
                    dem[t].append(b)
        for name, fifo, seg, i, o in FN:
            if o == "P" and seg != "A":
                for t, b in X[name].items():
                    pos[t].append(b)
        for t in set(dem) | set(pos):
            m.Add(sum(dem[t]) == sum(pos[t]))
    # QEY_WA: a scalar lda (write port at stage 6) one cycle after a cycle with two FIFO pops (one of them on lda: stage 7)
    for n in (LDA_ONLY + (("RQ2",) if cfg.ktl else ())) if "qey" not in DROP else ():
        for t, b in X[n].items():
            m.Add(sum(lits["fpop"][(t - 1) % R]) <= 1).OnlyEnforceIf(b)
    # ---- streams: ops of one segment in order, restarts between segments
    def chain(names):
        for a, b in zip(names, names[1:]):
            m.Add(T[b] >= T[a] + 1)

    segn = {s: [sname(s, i) for i in range(len(cfg.segs[s]))] for s in cfg.segs}
    A = ["AF"] + [f"A{k}" for k in range(NK)]
    chain(A)
    chain(segn["X"])
    chain(segn["Y"])
    chain(segn["L"])
    if cfg.n1b:
        chain(segn["N"])
    # lf0: A -> X -> Y -> (next row block) A
    for (prev_last, R_, P_, first) in ((A[-1], "XR", "XP", segn["X"][0]), (segn["X"][-1], "YR", "YP", segn["Y"][0])):
        m.Add(T[R_] >= T[prev_last])
        m.Add(T[R_] <= T[first] - 1)
        m.Add(T[P_] >= T[prev_last] + 1)
        m.Add(T[first] >= T[P_] + 2)
    ylast = segn["Y"][-1]
    m.Add(T["AR"] == T["AF"] - 1)
    m.Add(T["AR"] >= T[ylast] - R)
    m.Add(T["RP"] >= T[ylast] - R + 1)
    m.Add(T["RP"] <= T["AF"] - SCL)
    # lf1: one segment per row block, restart BR (mov r25, #0) / BP (mov p1, p5)
    l0, ll = segn["L"][0], segn["N" if cfg.n1b else "L"][-1]
    if cfg.n1b:  # L -> N restart (mov r25, #0; movs p1, r16 = N's base)
        lx, n0 = segn["L"][-1], segn["N"][0]
        m.Add(T["NR"] >= T[lx])
        m.Add(T["NR"] <= T[n0] - 1)
        m.Add(T["NP"] >= T[lx] + 1)
        m.Add(T[n0] >= T["NP"] + 2)

    m.Add(T["BR"] >= T[ll])
    m.Add(T["BR"] <= T[l0] + R - 1)
    m.Add(T["BP"] >= T[ll] + 1)
    m.Add(T[l0] + R >= T["BP"] + 2)
    # ---- jobs (gen_rot)
    for cb in range(NCB):
        P = [J(f"P{k}", cb) for k in range(NK)]
        M = [J(f"M{k}", cb) for k in range(NK)]
        Lq = [J(f"L{q}", cb) for q in range(NQ)]
        S = [J(f"S{q}", cb) for q in range(NQ)]
        for k in range(NK):
            m.Add(M[k] >= P[k] + D_POP_MAC)
            m.Add(M[k] <= P[k] + D_POP_MAC + emax)
            if k:
                m.Add(M[k] >= M[k - 1] + D_MAC_MAC)
        for q in range(NQ):
            m.Add(Lq[q] <= M[0] - D_LD_MAC)
            m.Add(S[q] >= M[NK - 1] + D_MAC_ST)
            if q:
                m.Add(Lq[q] >= Lq[q - 1] + 1)
                m.Add(S[q] >= S[q - 1] + 1)
        nxt, off = (cb + 1, 0) if cb < NCB - 1 else (0, R)
        m.Add(Lq[3] <= J("L0", nxt) + off - 1)
        m.Add(S[3] <= J("S0", nxt) + off - 1)
        succ, off4 = (cb + 4, 0) if cb < 4 else (cb - 4, R)
        for q in range(NQ):
            m.Add(S[q] <= J(f"L{q}", succ) + off4 + OLD_ST - margin)
    pops = [(dr, cb, k) for dr in (-1, 0, 1) for cb in range(NCB) for k in range(NK)] if "nb" not in DROP else []
    for x in [p for p in pops if p[0] == 0]:
        for y in pops:
            if y == x or y[2] % nb != x[2] % nb or (y[0] == 0 and y <= x):
                continue
            if abs((y[0] * R + per * y[1]) - per * x[1]) > 90:
                continue
            tx, ty = J(f"P{x[2]}", x[1]), y[0] * R + J(f"P{y[2]}", y[1])
            mx, my = J(f"M{x[2]}", x[1]), y[0] * R + J(f"M{y[2]}", y[1])
            b = m.NewBoolVar("")
            m.Add(tx < ty).OnlyEnforceIf(b)
            m.Add(tx > ty).OnlyEnforceIf(b.Not())
            m.Add(mx <= ty + D_POP_MAC - 1).OnlyEnforceIf(b)
            m.Add(my <= tx + D_POP_MAC - 1).OnlyEnforceIf(b.Not())
    # ---- A refresh (gen_rot): pop k lands between the last MAC on panel k of rb - 1 and the first of rb
    Ak = [T[f"A{k}"] for k in range(NK)]
    for k in range(NK):
        for cb in range(NCB):
            m.Add(Ak[k] >= J(f"M{k}", cb) - R - (D_POP_MAC - 1))
            m.Add(Ak[k] <= J(f"M{k}", cb) - D_POP_MAC)
    # ---- locks (HARD: round 18) and records (gen_rot)
    acq_gap = max(ACQ_GAP, int(os.environ.get("ROT_ACQ_GAP", ACQ_GAP)))
    rel_gap = max(REL_GAP, int(os.environ.get("ROT_REL_GAP", REL_GAP)))
    m.Add(T["ACQ"] <= T["AF"] - acq_gap)
    m.Add(T["ACQ"] >= T["AF"] - acq_gap - 5)
    if not cfg.ktl:
        m.Add(T["RQ0"] <= T["ACQ"] - SCL)
        m.Add(T["RQ1"] >= T["ACQ"] + 1)
    m.Add(T["RQ0"] >= T["REL"] - R + 1)
    m.Add(T["REL"] >= T["RQ1"] + SCL)
    m.Add(T["REL"] >= Ak[NK - 1] + rel_gap)
    m.Add(T["REL"] <= Ak[NK - 1] + R - 1)
    if cfg.ktl:
        # A acquire masked by the row-block counter (zero on the trip after the last): r28 = r26 x r27
        m.Add(T["MA"] >= T["RQ0"] + SCL)
        m.Add(T["MA"] <= T["RQ1"])  # reads r26 before RQ1 overwrites it
        m.Add(T["ACQ"] >= T["MA"] + 2)  # mul latency 2
        m.Add(T["ACQ"] <= T["MA"] + R - 1)
        for x in ("MA", "MB"):  # read this row block's counter value: after the previous CNT landed (alu -> alu >= 2), not after this one
            m.Add(T[x] <= T["CNT"])
            m.Add(T[x] >= T["CNT"] - R + 2)
        m.Add(T["MB2"] >= T["MB"] + 2)  # r30 = (r27 - 1) x r29: zero on the last k-step's rb 17 (no B beyond the tile)
        m.Add(T["MB2"] >= T["RQ2"] + SCL)
        m.Add(T["MB2"] <= T["MB"] + R - 2)
        m.Add(T["ACQB"] >= T["MB2"] + 2)
        m.Add(T["ACQB"] <= T["MB2"] + R - 1)
        m.Add(T["RELB"] >= T["RQ2"] + SCL)
        segs = [sg for sg in cfg.segs if sg in RESTART]
        r29 = ["MB2", "RELB"] + [f"TM{sg}" for sg in segs]
        for x in r29:
            m.Add(T[x] >= T["RQ2"] + SCL)
            m.Add(T[x] <= T["RQ2"] + R)  # before the next row block's RQ2 lands
        for sg in segs:
            m.Add(T[f"TX{sg}"] >= T[f"TM{sg}"] + 2)
        for sg in segs:  # each toggle has its own temp (r0 / r1 / r2 / r7, free after the setup): TM -> TX within a period
            m.Add(T[f"TX{sg}"] <= T[f"TM{sg}"] + R - 1)
        for sg in segs:
            d, P = cfg.delta[sg], T[RESTART[sg]]
            first, last = T[sname(sg, 0)], T[sname(sg, len(cfg.segs[sg]) - 1)]
            # frame 17's toggle: after the restart of the instance serving rb 17, before the one serving the next rb 0. BP (lf1's
            # L restart) sits after N in the period, so the BP of frame k restarts L's instance k + 1
            dp = d - 1 if sg == "L" else d
            m.Add(T[f"TX{sg}"] >= P + dp * R)
            m.Add(T[f"TX{sg}"] <= P + (1 + dp) * R - 1)
            # B(j) released >= REL_GAP after the last op of every instance serving rb 17; B(j + 1) acquired >= ACQ_GAP before
            # the first fill of every instance serving the next rb 0
            m.Add(T["RELB"] >= last + d * R + rel_gap)
            m.Add(T["ACQB"] <= first + (1 + d) * R - acq_gap)
    for i, (first, prev, poff) in enumerate(((0, NCB - 1, -R), (4, 3, 0))):
        j3, j4, rm = T[f"J3{i}"], T[f"J4{i}"], T[f"RM{i}"]
        m.Add(j3 >= J("L3", prev) + poff + 1)
        m.Add(j3 <= J("L0", first) - PTR_GAP)
        m.Add(j4 >= J("S3", prev) + poff + 1)
        m.Add(j4 <= J("S0", first) - PTR_GAP)
        for j in (j3, j4):
            m.Add(rm <= j - SCL)
            m.Add(rm >= j - R + 1)
    # ---- stages: stores in the last stage; everything inside two stages
    for name in X:
        if name.startswith("S") and "_" in name:
            m.Add(T[name] >= F)
        m.Add(T[name] >= F - R)
        m.Add(T[name] <= F + R - 1)
    E = m.NewIntVar(-3 * R, 0, "E")
    for name in X:
        m.Add(E <= T[name])
    pro = m.NewIntVar(0, 4 * R, "pro")
    m.Add(pro == F - E)
    # ---- bank conflicts between memory ops of different slots in one bundle (counted over the 18 row blocks and placements)
    cost = []

    def penal(expr, v, cnt):
        if cnt == 0 or "soft" in DROP:
            return
        b = m.NewBoolVar("")
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
                    for dr in (-1, 0, 1):
                        cnt = sum(1 for pl in pls for r in range(NRB) if 0 <= r + dr < NRB and
                                  bank(caddr(r, cb, q, pl)) == bank(caddr(r + dr, cb2, q2, pl)))
                        penal(J(f"L{q}", cb) - J(f"S{q2}", cb2), dr * R, cnt)

    def mem_bank_of(n, r, pl):
        if n in REC or n == "RQ2":
            return bank(pl["rec"])
        if n == "AF" or re.fullmatch(r"A\d", n):
            return bank(a_ptr(r, pl))
        if n.startswith("X"):
            return bank(pl["x"][0])
        return None

    others = A + segn["X"] + list(REC) + (["RQ2"] if cfg.ktl else [])
    for n in others:
        for kind in ("S", "L"):
            if kind == "L" and (n in REC or n == "RQ2"):
                continue  # same (lda) slot
            for q2 in range(NQ):
                for cb2 in range(NCB):
                    for dr in (-2, -1, 0, 1):
                        cnt = sum(1 for pl in pls for r in range(NRB) if 0 <= r + dr < NRB and
                                  mem_bank_of(n, r, pl) == bank(caddr(r + dr, cb2, q2, pl)))
                        penal(T[n] - J(f"{kind}{q2}", cb2), dr * R, cnt)
    conf = sum(cost) if cost else 0
    # main round 22: a bank-3 excess unit is one stall per row block (18 per call), as a conf unit is one stall per call: weigh them alike
    w3 = int(os.environ.get("E7_W3", "10000"))
    m.Minimize(100 * conf + pro + w3 * sum(bank3x))
    if hint:  # e.g. a relaxed model's solution (E7_DROP=soft,bank3): CP-SAT repairs it
        for n, v in hint.items():
            if n in T:
                m.AddHint(T[n], v)
                if n in X and v in X[n]:
                    for t, b in X[n].items():
                        m.AddHint(b, int(t == v))
    s = cp_model.CpSolver()
    if hint:
        s.parameters.repair_hint = True
        s.parameters.hint_conflict_limit = 100000
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = verbose
    st = s.Solve(m)
    info = dict(status=s.StatusName(st), **cfg.d(), nb=nb, emax=emax, margin=margin, wall=s.WallTime(), hard_banks=hard_banks,
                absolute=absolute)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, info
    times = {n: int(s.Value(v)) for n, v in T.items()}
    info.update(F=int(s.Value(F)), E=int(s.Value(E)), prologue=int(s.Value(pro)),
                bank_conflicts_model=int(s.Value(conf)) if cost else 0, bank3_excess=int(sum(s.Value(e) for e in bank3x)), bound=s.BestObjectiveBound(), objective=s.ObjectiveValue())
    return times, info


# ---------------------------------------------------------------------------------------------------------------------
# emission
NOPS = dict(lda="nopa", ldb="nopb", st="nops", alu="nopx", mv="nopm", vec="nopv")
ORDER = ["lda", "ldb", "st", "alu", "mv", "vec"]
FIFO_REGS = dict(lf0=("p0", "r24"), lf1=("p1", "r25"))


def pairing(times, cfg):
    """stream B pop position -> (cb, kp): the job pops and the positions matched residue by residue"""
    R = cfg.R
    key = (lambda t: t) if cfg.aligned else (lambda t: t % R)
    jobs = defaultdict(list)
    for cb in range(NCB):
        for k in range(NK):
            jobs[key(times[f"P{k}_{cb}"])].append((times[f"P{k}_{cb}"], cb, k))
    pos = defaultdict(list)
    for name, fifo, seg, i, o in fifo_names(cfg):
        if o == "P" and seg != "A":
            pos[key(times[name] + (cfg.delta[seg] * R if cfg.aligned else 0))].append((fifo, name))
    out = {}
    for c, js in jobs.items():
        ps = sorted(pos[c])
        assert len(ps) == len(js), (c, ps, js)
        for (fifo, name), (_, cb, k) in zip(ps, sorted(js)):
            out[name] = (cb, k)
    return out


def layout(times, cfg):
    """byte offset in the 4,608 B object of every B position (Y buffer: lf0-Y panels then lf1; X buffer: lf0-X) and the
    object order [(cb, kp)] * 64"""
    pr = pairing(times, cfg)
    off, order = {}, [None] * 64
    nxt = dict(Y=0, L=72 * cfg.ny, N=72 * (cfg.ny + cfg.n1 - cfg.n1b), X=cfg.yb)
    for name, fifo, seg, i, o in fifo_names(cfg):
        if o == "P" and seg != "A":
            off[name] = nxt[seg]
            order[nxt[seg] // 72] = pr[name]
            nxt[seg] += 72
    assert all(order) and len(set(order)) == 64
    return pr, off, order


def all_ops(times, cfg, pr):
    """[(time, slot or 'fifo', name, text-or-None, rec field, fifo, is_pop)] of row block 0"""
    R = cfg.R
    out = []
    for n in JOBTPL:
        if n[0] == "P":
            continue
        for cb in range(NCB):
            d = cb % 4
            k = int(n[1:])
            if n[0] == "M":
                txt, sl = f"vmac.f dm{d}, dm{d}, ex{AREG[k]}, ex{BREG[k % 4]}, {'r6' if k == 0 else 'r3'}", "vec"
            elif n[0] == "L":
                txt, sl = f"vlda {QN[k]}{d}, [p3], #64", "lda"
            else:
                txt, sl = f"vst {QN[k]}{d}, [p4], #64", "st"
            out.append((times[f"{n}_{cb}"], sl, f"{n}_{cb}", txt, None, None, False))
    for name, fifo, seg, i, o in fifo_names(cfg):
        if o == "F":
            reg = None
        elif seg == "A":
            reg = f"ex{AREG[i - 1]}"
        else:
            reg = f"ex{BREG[pr[name][1] % 4]}"
        out.append((times[name], "fifo", name, reg, None, fifo, o == "P"))
    txt = dict(AR="mov r24, #0", XR="mov r24, #0", YR="mov r24, #0", XP="mov p0, p6", YP="mov p0, p7", BR="mov r25, #0",
               BP="mov p1, p5", ACQ="acq.cond #53, r4, r26", REL="rel.cond #52, r5, r26", RQ0="lda r26, [p2], #4",
               RQ1="lda r26, [p2], #4", RP="lda p0, [p2], #4", RM0="lda m0, [p2], #4", RM1="lda m1, [p2], #4",
               J30="padds [p3], m0", J31="padds [p3], m1", J40="padds [p4], m0", J41="padds [p4], m1")
    if cfg.n1b:
        txt.update(NR="mov r25, #0", NP="movs p1, r16")
    slot = {**{n: "mv" for n in MV}, **{n: "lda" for n in REC}, "ACQ": "alu", "REL": "alu",
            **{n: "st" for n in ("J30", "J31", "J40", "J41", "NP")}}
    for n, t in txt.items():
        out.append((times[n], slot[n], n, t, REC_FIELD.get(n), None, False))
    return out


def fifo_text(slot, fifo, reg):
    p, r = FIFO_REGS[fifo]
    u = "a" if slot == "lda" else "b"
    if reg is None:
        return f"vld{u}.fill.512 [{p}, {fifo}, {r}]"
    return f"vld{u}.pop.576 {reg}, [{p}, {fifo}, {r}]"


def bundle(ops, full):
    """ops [(slot or 'fifo', text, fifo, reg)] -> bundle text; FIFO ops go to the free load slot(s); with two FIFO ops the
    fill (if any) takes lda (no QEY_WA write)"""
    by = {}
    fifos = [o for o in ops if o[0] == "fifo"]
    for sl, txt, _, _ in ops:
        if sl != "fifo":
            assert sl not in by, (sl, txt, by)
            by[sl] = txt
    free = [s for s in ("ldb", "lda") if s not in by]
    assert len(fifos) <= len(free), (ops, by)
    fifos.sort(key=lambda o: o[3] is not None)  # fills first -> lda when both slots are free
    if len(fifos) == 2:
        free = ["lda", "ldb"]
    for (sl, _, fifo, reg), s in zip(fifos, free):
        by[s] = fifo_text(s, fifo, reg)
    if full:
        return ";\t\t".join(by.get(s, NOPS[s]) for s in ORDER)
    return ";\t\t".join(by[s] for s in ORDER if s in by) if by else "nop"


def emit(times, info, name="mm_e7", full_pro=frozenset()):
    cfg = cfg_of(info)
    R, F = cfg.R, info["F"]
    pr, off, order_b = layout(times, cfg)
    ops = all_ops(times, cfg, pr)
    body = defaultdict(list)
    for t, sl, n, txt, fld, fifo, ispop in ops:
        stage = (t - F) // R
        assert stage in (-1, 0), (n, t, F)
        body[(t - F) % R].append((sl, txt, fifo, txt if sl == "fifo" else None, stage, fld))
    # prologue: rb 0's stage -1 ops, plus the B-stream segments of the previous period that hold rb 0's early pops (from the
    # segment's restart on, every op of that FIFO up to F; their other pops land in registers nobody reads yet)
    fn = {nm: (fifo, seg, i, o) for nm, fifo, seg, i, o in fifo_names(cfg)}
    need = defaultdict(lambda: None)  # fifo -> earliest previous-period time needed
    for cb in range(NCB):
        for k in range(NK):
            a = times[f"P{k}_{cb}"]
            for nm, (cb2, k2) in pr.items():
                if (cb2, k2) == (cb, k) and times[nm] - R == a:  # served by the previous period's instance
                    fifo, seg, i, o = fn[nm]
                    seg0 = sname(seg, 0)
                    start = times[seg0] - R
                    restart = [x for x in (("BR", "BP") if seg == "L" else (seg + "R", seg + "P"))]
                    start = min([start] + [times[x] - R for x in restart])
                    need[fifo] = start if need[fifo] is None else min(need[fifo], start)
    pro = defaultdict(list)
    for t, sl, n, txt, fld, fifo, ispop in ops:
        if t < F:
            pro[t].append((sl, txt, fifo, txt if sl == "fifo" else None, fld))
    prev_ops = []
    for t, sl, n, txt, fld, fifo, ispop in ops:
        f = fifo
        if n in ("BR", "BP", "NR", "NP"):
            f = "lf1"
        elif n in ("XR", "XP", "YR", "YP"):
            f = "lf0"
        if f is None or n in ("AF",) or re.fullmatch(r"A\d", n) or n in ("AR",):
            continue
        if need[f] is not None and need[f] <= t - R < F:
            prev_ops.append((t - R, sl, n, txt, fifo))
    for t, sl, n, txt, fifo in prev_ops:
        pro[t].append((sl, txt, fifo, txt if sl == "fifo" else None, None))
    E = min(list(pro) + [F])
    lines = ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    # mm_e7(const int32_t *rec, const bfp16ebs8 *pY, float *piece0, const bfp16ebs8 *pX, int32_t cfg): p0 p1 p2 p3, r0.
    # ZLOAD (main round 21): the kp-0 MACs take their config from r6 = cfg (781 = 780 | zero_acc on the first k-step of a tile)
    setup = ["mov r6, r0", "paddxm [sp], #64", f"movxm m3, #{72 * cfg.ny}", "st p6, [sp, #-64]", "st p7, [sp, #-60]", "movs p6, p3",
             "movs p7, p1", "padda [p1], m3", "movs p3, p2", "movs p4, p2", "mova r3, #780", "movs p5, p1", "mova r25, #0",
             "mova r24, #0", "mova r4, #-1", "mova r5, #1", "movs p2, p0", f"movxm ls, #.L{name}_body", f"movxm le, #.L{name}_end",
             f"mova r0, #{NRB}", "add.nc lc, r0, #0"]
    if cfg.n1b:  # N's base (Y + 72 (ny + n1 - n1b)) is the 6th argument (r1)
        setup = ["mov r16, r1"] + setup
    lines += ["\t" + s for s in setup]
    lines += ["\t.p2align\t4", f".L{name}_pro:"]
    order = []
    for i, a in enumerate(range(E, F)):
        b = [(sl, txt, fifo, reg) for sl, txt, fifo, reg, _ in pro.get(a, [])]
        lines.append("\t" + bundle(b, i in full_pro))
        for sl, txt, fifo, reg, fld in pro.get(a, []):
            if fld is not None:
                order.append((0, fld))
    lines += ["\t.p2align\t4", f".L{name}_body:", f"{name}_start:"]
    for u in range(R):
        if u == R - 1:
            lines += ["\t.p2align\t4", f".L{name}_end:", f"{name}_last:"]
        lines.append("\t" + bundle([(sl, txt, fifo, reg) for sl, txt, fifo, reg, _, _ in body[u]], True))
    for f in range(NRB):
        for u in range(R):
            for sl, txt, fifo, reg, stage, fld in sorted(body[u], key=lambda o: ORDER.index(o[0]) if o[0] in ORDER else 0):
                if fld is not None:
                    order.append((f - stage, fld))
    lines += ["\tlda p6, [sp, #-64]", "\tlda p7, [sp, #-60]", "\tpaddxm [sp], #-64", "\tnop", "\tnop", "\tret lr"] + ["\tnop"] * 5 + [f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return "\n".join(lines) + "\n", order, order_b, off


def header(order, order_b, cfg, name="mm_e7"):
    nm = name.upper()
    return "\n".join([
        f"// generated by tapnext/research/r22/gen_e7.py (R = {cfg.R}, lf1 {cfg.n1} / lf0-X {cfg.nx} / lf0-Y {cfg.ny} panels): {name}'s "
        "record stream and B object order",
        "// records: one int32 per (row block, field) in execution order; fields: " + ", ".join(f"{i} = {s}" for i, s in enumerate(FIELD_NAMES)),
        f"#define {nm}_NREC {len(order)}",
        f"#define {nm}_YB {cfg.yb}  // object bytes [0, YB) -> the Y buffer (bank 3), [YB, 4608) -> the X buffer (bank 0)",
        f"static constexpr int8_t {name}_order[{nm}_NREC][2] = {{" + ", ".join(f"{{{a}, {b}}}" for a, b in order) + "};",
        "// object position i (72 B each) holds panel pair4[...] of pack8's pair4 order: pair * 32 + kp * 4 + cb % 4",
        f"static constexpr int8_t {name}_border[64] = {{" + ", ".join(str((cb >> 2) * 32 + k * 4 + (cb & 3)) for cb, k in order_b) + "};",
    ]) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# replay of the emitted asm (one whole call, bundle by bundle)
def parse_s(text, name="mm_e7"):
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
        if cur is body and len(body) and s.startswith("lda p6"):
            cur = tail
        ins = [i.strip() for i in s.split(";") if i.strip() and not re.fullmatch(r"nop[abxmvs]?", i.strip())]
        cur.append(ins)
    return pre, body, tail


def check(text, info, order, order_b, pl=PLACE, buf=0, pad=0, name="mm_e7", verbose=False):
    cfg = cfg_of(info)
    pre, body, tail = parse_s(text, name)
    trips = NRB
    seq = [(b, -1) for b in pre] + [([], -1)] * pad + [(b, f) for f in range(trips) for b in body] + [(b, trips) for b in tail]
    mem = {}
    for rb in range(NRB):
        for cb in range(NCB):
            for q in range(NQ):
                mem[caddr(rb, cb, q, pl)] = ("acc", rb, cb, q, 0)
    cq = set(mem)
    table = [record(r, f, pl) for r, f in order]
    X0, Y0 = pl["x"][buf], pl["y"][buf]
    regs = defaultdict(list)
    errs, notes = [], []
    st = dict(macs=0, stalls=0, stall_where=defaultdict(int), acq=[], rel=[], slot_owner={}, last_read=defaultdict(int),
              stores=0, loads=0, busy=defaultdict(int), lda_fifo=0)
    fifo = {}
    sp0 = 0x70400
    smem = {}
    init = {"r0": 780, "r1": Y0 + 72 * (cfg.ny + cfg.n1 - cfg.n1b), "p0": pl["rec"], "p1": Y0, "p2": piece(0, 0, pl), "p3": X0, "p6": "P6", "p7": "P7", "sp": sp0}
    for r, v in init.items():
        regs[r].append((-100, 1, 1, v))

    def err(c, msg):
        errs.append(f"cycle {c}: {msg}")
        if len(errs) > 30:
            raise Fail("\n".join(notes[:10] + errs))

    def wr(c, r, v, wmin, wmax=None):
        if os.environ.get("E7DBG") == r:
            print("write", r, c, v, wmin, wmax)
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
                if own is None or own[1]:
                    return "G"
                st["last_read"][own[0]] = c
                i = o // 72
                return ("A", 2 * own[0] + i // 8, i % 8)
        return None

    def b_val(c, addr):
        for base, lo, n in ((Y0, 0, cfg.yb), (X0, cfg.yb, OBJ - cfg.yb)):
            if base <= addr < base + n:
                o = addr - base
                if o % 72:
                    err(c, f"B pop at buffer + {o}")
                    return "G"
                return ("B",) + tuple(order_b[(lo + o) // 72])
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
                    err(c, f"{lf}: FIFO underflow (pop {s['C'] // 72 + 1} after {s['lines'] - 1} lines)")
                val = s["base"] + s["C"]
                s["C"] += 72
        wr(c, ptr, p + 64, 1)
        wr(c, pos, "S", 1)
        return val, p

    c = -1
    for bund, frame in seq:
        c += 1
        acc_banks = []
        writes = []
        nlda_pop = 0
        for ins in bund:
            mm = None
            if ins in ("mov r6, r0", "mov r16, r1"):
                d_, s_ = ins[4:].split(", ")
                writes.append((d_, rd(c, s_), 1, 1))
            elif ins == "movs p1, r16":
                writes.append(("p1", rd(c, "r16"), 1, 2))
            elif (mm := re.fullmatch(r"movs (p\d), (p\d)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)), 1, 2))
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
            elif (mm := re.fullmatch(r"mov (p[01]), (p[567])", ins)):
                writes.append((mm.group(1), rd(c, mm.group(2)), 1, 2))
            elif (mm := re.fullmatch(r"padda \[(p\d)\], (m3)", ins)):
                writes.append((mm.group(1), rd(c, mm.group(1)) + rd(c, mm.group(2)), 1, PTR_GAP))
            elif ins == "paddxm [sp], #64":
                writes.append(("sp", rd(c, "sp") + 64, 1, 1))
            elif ins == "paddxm [sp], #-64":
                writes.append(("sp", rd(c, "sp") - 64, 1, 1))
            elif (mm := re.fullmatch(r"st (p[67]), \[sp, #(-?\d+)\]", ins)):
                smem[rd(c, "sp") + int(mm.group(2))] = rd(c, mm.group(1))
            elif (mm := re.fullmatch(r"lda (p[67]), \[sp, #(-?\d+)\]", ins)):
                writes.append((mm.group(1), smem.get(rd(c, "sp") + int(mm.group(2))), 1, SCL))
            elif (mm := re.fullmatch(r"vld([ab])\.fill\.512 \[(p[01]), (lf[01]), (r2[45])\]", ins)):
                if (mm.group(2), mm.group(4)) != FIFO_REGS[mm.group(3)]:
                    err(c, f"unexpected FIFO registers {ins}")
                _, p = stream(c, mm.group(3), mm.group(2), mm.group(4), False)
                acc_banks.append(bank(p))
                if mm.group(3) == "lf0" and p is not None and a_val(c, p) is not None:
                    st.setdefault("afill", []).append(c)
                    pa = st.pop("pending_acq", None)
                    if pa is not None and c - pa < ACQ_GAP:
                        err(c, f"A fill {c - pa} cycle(s) after its acquire (need >= {ACQ_GAP})")
            elif (mm := re.fullmatch(r"vld([ab])\.pop\.576 ex(\d+), \[(p[01]), (lf[01]), (r2[45])\]", ins)):
                a, p = stream(c, mm.group(4), mm.group(3), mm.group(5), True)
                acc_banks.append(bank(p))
                if mm.group(1) == "a":
                    nlda_pop += 1
                    st["lda_fifo"] += 1
                v = "G"
                if a is not None:
                    v = a_val(c, a)
                    if v is None:
                        v = b_val(c, a)
                    if v is None:
                        err(c, f"pop from {a:#x}: neither an A slot nor a B buffer")
                        v = "G"
                writes.append((f"ex{mm.group(2)}", v, 8, 8))
            elif (mm := re.fullmatch(r"vmac\.f dm(\d), dm\1, ex(\d+), ex(\d+), (r[36])", ins)):
                d = mm.group(1)
                accs = [rd(c, f"dm{d}.{q}", 4) for q in range(NQ)]
                av, bv = rd(c, f"ex{mm.group(2)}"), rd(c, f"ex{mm.group(3)}")
                if rd(c, mm.group(4)) != 780:
                    err(c, f"{mm.group(4)} != 780")
                ok = all(isinstance(x, tuple) and x[0] == "acc" for x in accs) and len({x[1:3] + x[4:] for x in accs}) == 1 \
                    and [x[3] for x in accs] == [0, 1, 2, 3]
                rb = cb = n = None
                if ok:
                    _, rb, cb, _, n = accs[0]
                    ok = av == ("A", rb, n) and bv == ("B", cb, n) and n < NK
                if ok:
                    st["macs"] += 1
                elif all(isinstance(x, tuple) for x in accs) and verbose and 0 <= frame < NRB:
                    notes.append(f"cycle {c} frame {frame}: {ins}: acc {accs[0]}, A {av}, B {bv}")
                for q in range(NQ):
                    writes.append((f"dm{d}.{q}", ("acc", rb, cb, q, n + 1) if ok else "G", 6, 6))
            elif (mm := re.fullmatch(r"vlda (bm\w\w)(\d), \[p3\], #64", ins)):
                p = rd(c, "p3")
                acc_banks.append(bank(p))
                q = QN.index(mm.group(1))
                writes.append((f"dm{mm.group(2)}.{q}", mem.get(p, "G"), 7, 7))
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
                st.setdefault("scalar_at", []).append(c)
            elif ins == "acq.cond #53, r4, r26":
                if rd(c, "r26"):
                    if rd(c, "r4") != -1:
                        err(c, "acq r4 != -1")
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
            elif ins.startswith("ret") or ins == "nop":
                pass
            else:
                raise Fail(f"unknown instruction {ins!r}")
        if nlda_pop:
            st.setdefault("lda_pop_at", []).append(c)
        for r, v, wmin, wmax in writes:
            wr(c, r, v, wmin, wmax)
        cnt = defaultdict(int)
        for bk in acc_banks:
            cnt[bk] += 1
        if 0 <= frame < NRB:
            st["body_cycles"] = st.get("body_cycles", 0) + 1
            for bk in cnt:
                st["busy"][bk] += 1
        sl = max(cnt.values(), default=1) - 1
        if sl > 0:
            st["stalls"] += sl
            st["stall_where"]["prologue" if frame < 0 else "body" if frame < NRB else "tail"] += sl
    for s in st.get("scalar_at", []):  # QEY_WA: scalar lda (stage 6) one cycle after an lda pop (stage 7)
        if s - 1 in set(st.get("lda_pop_at", [])):
            err(s, "QEY_WA port: scalar lda one cycle after an lda pop")
    bad = [(a, v) for a, v in mem.items() if a in cq and not (isinstance(v, tuple) and v[4] == NK)]
    if bad:
        err(c, f"{len(bad)} quarters not finished, e.g. {bad[0][0]:#x}: {bad[0][1]}")
    if st["macs"] != MACS:
        err(c, f"{st['macs']} valid MACs, want {MACS}")
    if len(st["acq"]) != RHO or len(st["rel"]) != RHO:
        err(c, f"{len(st['acq'])} acquires / {len(st['rel'])} releases, want {RHO}")
    for r, want in (("p6", "P6"), ("p7", "P7"), ("sp", sp0)):
        if rd(c + 20, r) != want:
            err(c, f"{r} not restored")
    if errs:
        raise Fail("\n".join(notes[:12] + errs))
    st["cycles"] = c + 1
    st["bundles_pre"] = len(pre)
    return st


def run_check(text, info, order, order_b, pl=PLACE):
    res = {}
    for pad in (0, 1):  # pad 1 = the .p2align nop before the body: fit() removes it; reported, not fatal
        for buf in (0, 1):
            try:
                res[pad, buf] = check(text, info, order, order_b, pl=pl, buf=buf, pad=pad, verbose=True)
            except Fail as e:
                if pad == 0:
                    raise
                print(f"pad 1 buf {buf}: FAILS (fit must remove the pad): {str(e).splitlines()[-1]}")
    return res


def fit(times, info, work, name="mm_e7"):
    """as gen_rot.fit: expand prologue bundles until the prologue ends 16 B aligned (no pad bundle before the body)"""
    text, order, order_b, off = emit(times, info, name=name)
    _, dis, addr = G.assemble(text, work, name)
    offs = G.bundle_addrs(dis)
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
    assert need in best, f"no expansion set reaches residue {need}"
    full = best[need][1]
    text, order, order_b, off = emit(times, info, name=name, full_pro=full)
    o, dis, addr = G.assemble(text, work, name)
    offs = G.bundle_addrs(dis)
    i0 = offs.index(pro[0])
    assert offs[i0 + npro] == addr[f"{name}_start"], "a pad bundle sits between the prologue and the body"
    assert addr[f"{name}_last"] % 16 == 0 and addr[f"{name}_start"] % 16 == 0
    return text, order, order_b, dict(expanded=sorted(full), added=best[need][0], obj=str(o), dis=dis,
                                      text_bytes=addr.get(f".Lfunc_end_{name}"))


def parse_place(spec):
    pl = dict(PLACE)
    for kv in (spec.split(",") if spec else []):
        k, v = kv.split("=")
        pl[k] = tuple(int(x, 16) for x in v.split(":")) if k in ("slots", "x", "y") else int(v, 16)
    return pl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("solve")
    a.add_argument("--R", type=int, default=64)
    a.add_argument("--n1", type=int, default=32)
    a.add_argument("--nx", type=int, default=8)
    a.add_argument("--n1b", type=int, default=0, help="lf1 split: N segment of n1b panels after L (restart between)")
    a.add_argument("--aligned", action="store_true", help="every stream segment instance serves ONE row block (KTL)")
    a.add_argument("--delta", help='per-segment instance offsets, JSON, e.g. {"N": -1}')
    a.add_argument("--ktl", action="store_true", help="schedule KTL's extra per-row-block ops (implies --aligned)")
    a.add_argument("--time", type=float, default=600)
    a.add_argument("--workers", type=int, default=12)
    a.add_argument("--emax", type=int, default=6)
    a.add_argument("--hard-banks", action="store_true")
    a.add_argument("--place", action="append", default=[])
    a.add_argument("--json", required=True)
    a.add_argument("--hint", help="a solution json (any R / drop set) to start from")
    a.add_argument("-v", action="store_true")
    a.add_argument("--absolute", action="store_true", help="rb 0 pops served by rb 0 stream instances (short prologue; R = 64 INFEASIBLE, r22/diag64.out)")
    for nm in ("emit", "fit"):
        b = sub.add_parser(nm)
        b.add_argument("json")
        b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("s")
    c.add_argument("--json", required=True)
    c.add_argument("--place")
    o = ap.parse_args()
    if o.cmd == "solve":
        cfg = Cfg(o.R, o.n1, o.nx, o.n1b, o.aligned, json.loads(o.delta) if o.delta else None, o.ktl)
        times, info = solve(cfg, o.time, o.workers, emax=o.emax, hard_banks=o.hard_banks, verbose=o.v,
                            pls=[parse_place(x) for x in o.place] or None, absolute=o.absolute,
                            hint=json.loads(Path(o.hint).read_text())["times"] if o.hint else None)
        info["places"] = o.place
        print(json.dumps(info))
        if times is None:
            raise SystemExit(1)
        Path(o.json).write_text(json.dumps(dict(times=times, info=info), indent=1))
    elif o.cmd in ("emit", "fit"):
        d = json.loads(Path(o.json).read_text())
        out = Path(o.out)
        cfg = cfg_of(d["info"])
        if o.cmd == "emit":
            text, order, order_b, off = emit(d["times"], d["info"])
        else:
            text, order, order_b, fi = fit(d["times"], d["info"], out.parent)
            out.with_suffix(".dis").write_text(fi.pop("dis"))
            d["fit"] = fi
        out.write_text(text)
        out.with_suffix(".h").write_text(header(order, order_b, cfg))
        d["order"], d["order_b"] = order, order_b
        Path(o.json).write_text(json.dumps(d, indent=1))
        res = run_check(text, d["info"], order, order_b)
        for (pad, buf), st in res.items():
            print(f"pad {pad} buf {buf}: {st['cycles']} cycles per call (setup + prologue {st['bundles_pre'] + pad}, {NRB} x "
                  f"{cfg.R}, tail), {st['macs']} MACs checked, {st['stores']} stores, lda-slot FIFO ops {st['lda_fifo']}, "
                  f"bank stalls {st['stalls']} {dict(st['stall_where'])}, body busy per bank "
                  f"{ {k: round(v / max(1, st['body_cycles']), 2) for k, v in sorted(st['busy'].items())} }")
    else:
        d = json.loads(Path(o.json).read_text())
        text = Path(o.s).read_text()
        pl = parse_place(o.place)
        for buf in (0, 1):
            st = check(text, d["info"], [tuple(x) for x in d["order"]], [tuple(x) for x in d["order_b"]], pl=pl, buf=buf)
            print(f"buf {buf}: {st['cycles']} cycles per call, {st['macs']} MACs, bank stalls {st['stalls']} "
                  f"{dict(st['stall_where'])}: {st['cycles'] + st['stalls']} on this placement")


if __name__ == "__main__":
    main()
