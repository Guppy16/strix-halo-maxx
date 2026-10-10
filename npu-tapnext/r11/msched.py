"""Round 11: modulo scheduler, register allocator, emitter and value-level checker for hand-written AIE2P loops.

A loop body is a list of Ops (one iteration) over virtual registers. The scheduler places them at iteration-relative
cycles with slots (and optionally memory banks) reserved modulo the initiation interval II, the allocator gives every
virtual register U rotating physical copies (modulo variable expansion; the loop body holds U iterations' worth of
U * II bundles), and the emitter writes a prologue (the first iterations' early stages, iterations >= 0 only), a zero-overhead
loop body run T = n / U times and a drain. The checker replays the emitted program bundle by bundle: every operand read
must see the value of ITS iteration (the right def, landed, not yet overwritten), every load / store must hit its
iteration's address (2D post-modify emulated), and no loop op may write a constant register.

Machine model (SKILL §3, aie2p_lat.json): one op per slot per bundle (a = lda, b = ldb, s = st, x = alu, m = mv,
v = vec) plus the itinerary port resources (ports()); a register written by an op issued at cycle c with .td latency w is
visible to an operand read at stage r of an op issued at c2 iff c2 + r >= c + w + 1 (LLVM's latency w - r + 1; measured for
vlda, vldb.4x64, vadd.32, vconv, vfloor, vshuffle, vmin_ge by r11/mk_haz.py); until then reads see the old value (no
interlocks) - EXCEPT that a value from an mv-slot op reaches mv-slot readers one cycle early (bypass, measured), so a
register stays reserved from c + w and a read is accepted only outside the one-cycle window. vmac reads its accumulator at the .td's stage 4 (= "stage 3" in the older c + r >= c' + w convention). Register atoms: x_n = wl_n + wh_n; dm_n = 4 quarters (cml = q0 q1, bmll q0, bmlh q1).

Usage: build Ops, then
    s = Loop(ops, consts, classes); s.schedule(II); s.allocate(); text = s.emit(...); s.check(n)
"""

import math
import re
from collections import defaultdict

SLOTS = {"lda": "a", "ldb": "b", "st": "s", "alu": "x", "mv": "m", "vec": "v"}
SLOT_ORDER = ["a", "b", "s", "x", "m", "v"]
NOP = {"a": "nopa", "b": "nopb", "s": "nops", "x": "nopx", "m": "nopm", "v": "nopv"}
PARTS = {  # part -> (register class, atoms as offsets)
    "x": ("x", ("l", "h")), "wl": ("x", ("l",)), "wh": ("x", ("h",)),
    "dm": ("dm", (0, 1, 2, 3)), "cml": ("dm", (0, 1)), "cmh": ("dm", (2, 3)), "bmll": ("dm", (0,)), "bmlh": ("dm", (1,)),
    # round 13: a bfp panel register ex_n (576 bits, written by pops, read by the bfp mmul) ALIASES x_n (Peano spills live x
    # registers around pop loops; a loop that held Q in ex0-7 beside x consts computed garbage): same class, both atoms
    "ex": ("x", ("l", "h")),
}
PHYS = {"x": lambda n, p: f"ex{n}" if p == "ex" else f"{p}{n}", "dm": lambda n, p: f"{p}{n}"}
VIS = 1  # measured (r11/mk_haz.py): a write with .td latency w is seen by an op issued >= w - r + 1 later (LLVM's formula)
MEM_W, MEM_R = 4, 1
MIN_BODY = 3  # bundles: measured (r12/gen_copy.py + cpyb): a body of 2 bundles runs once, 3 loops  # checker model: a store's data reach memory 4 cycles after issue, a load reads at issue + 1


class Infeasible(Exception):
    pass


_TD = None


def td():
    global _TD
    if _TD is None:
        import json
        from pathlib import Path
        _TD = json.load(open(Path(__file__).resolve().parent / "aie2p_lat.json"))
    return _TD


def ports(instr):
    """[(stage, resource)] from the instruction's itinerary in llvm-aie's AIE2PGenSchedule.td (aie2p_lat.json):
    SimpleCycle<R> = R for one cycle then the next stage; PrefixCycle<R> = R in the same cycle as the next stage;
    EmptyCycles<n> = n stages without resources; InstrStage<n, [R]> = R for n cycles. Stage 0 = the issue cycle (a write
    port at stage k is a result with latency k + 1). AvoidPartWordStore / IsPartWordStore only matter for part-word stores."""
    d = td()["defs"][instr]
    st = td()["itin"][d["itin"]].get("stages", "")
    out, t = [], 0
    for kind, arg in re.findall(r"(SimpleCycle|PrefixCycle|EmptyCycles|InstrStage|AvoidPartWordStore|IsPartWordStore)(?:<([^>]*)>)?", st):
        if kind == "SimpleCycle":
            out.append((t, arg))
            t += 1
        elif kind == "PrefixCycle":
            out.append((t, arg))
        elif kind == "EmptyCycles":
            t += int(arg)
        elif kind == "InstrStage":
            n, r = re.match(r"\s*(\d+)\s*,\s*\[(\w+)\]", arg).groups()
            for k in range(int(n)):
                out.append((t + k, r))
            t += int(n)
    return out


class Op:
    """fmt: asm text with {vreg.part} placeholders (e.g. "vmac.f {s.dm}, {sacc.dm}, {slope.x}, {xc.x}, r1").
    defs: [(vreg, part, latency)]; uses: [(vreg, part, stage)] (part = the atoms the op really reads: a vmac reads its
    accumulator as dm in the text but only cml matters -> give "cml" here). mem: ("ld" | "st", stream name) for address
    checks; banks: memory banks the op touches (at most one access per bank per bundle)."""

    def __init__(self, name, slot, fmt, defs=(), uses=(), mem=None, banks=(), after=(), spill=None, fill=None, instr=None, cuses=()):
        """after: [(op name, min issue distance)] extra edges (memory dependences); spill: (vreg, part) this store writes
        to its stream; fill: (vreg, part) whose spilled value this load must read back (checked through memory)"""
        self.name, self.slot, self.fmt = name, SLOTS[slot], fmt
        self.defs, self.uses, self.mem, self.banks = list(defs), list(uses), mem, tuple(banks)
        self.after, self.spill, self.fill = list(after), spill, fill
        self.instr = instr  # the .td instruction: its itinerary's port resources are reserved too
        # round 16: carried uses [(vreg, part, stage)]: reads of the PREVIOUS iteration's value (a loop-carried accumulator,
        # s(i) = s(i - 1) + ...). Loops with carried uses need U = 1 (one register holds the chain), the reading op in the last
        # stage (the drain's garbage iterations must not update it) and the initial value in the register before the loop.
        self.cuses = list(cuses)
        self.ports = ports(instr) if instr else []
        self.t = None

    def __repr__(self):
        return f"{self.name}@{self.t}"


def atoms_of(cls, n, part):
    return [(cls, n, a) for a in PARTS[part][1]]


class Loop:
    def __init__(self, ops, consts=None, classes=None, acc_read_stage=None):
        """consts: physical atoms never written by the loop, e.g. {"x": [4, 5]}; classes: allocatable physical registers
        per class, e.g. {"x": [0, 1, ...], "dm": [0..4]}"""
        self.ops = ops
        self.consts = consts or {}
        self.classes = classes
        self.by_name = {o.name: o for o in ops}
        assert len(self.by_name) == len(ops), "op names must be unique"
        self.vclass = {}
        for o in ops:  # the text must name exactly the registers the def / use lists declare
            named = set(re.findall(r"\{(\w+)\.\w+\}", o.fmt))
            declared = {v for v, *_ in o.defs + o.uses + o.cuses}
            assert named == declared, f"{o.name}: text names {sorted(named)}, defs / uses declare {sorted(declared)}"
        for o in ops:
            for v, p, *_ in o.defs + o.uses + o.cuses:
                c = PARTS[p][0]
                assert self.vclass.setdefault(v, c) == c, (v, p)
        self._edges()

    # ---- dependences -------------------------------------------------------------------------------------------
    def _edges(self):
        """reaching defs per (use, atom) in op order; edges def -> use with min distance lat - stage; def -> later def of
        an overlapping atom (the later one must land after the earlier one)"""
        self.reach = {}  # (op name, use index, atom offset) -> def op name
        self.preds = defaultdict(list)  # op name -> [(pred name, min distance)]
        last = {}  # (vreg, atom) -> (op, lat)
        for o in self.ops:
            for ui, (v, p, st) in enumerate(o.uses):
                for a in PARTS[p][1]:
                    if (v, a) not in last:
                        raise ValueError(f"{o.name} reads {v}.{a} before any def")
                    d, lat = last[v, a]
                    self.reach[o.name, ui, a] = d.name
                    self.preds[o.name].append((d.name, lat + VIS - st))
            for v, p, lat in o.defs:
                for a in PARTS[p][1]:
                    if (v, a) in last:
                        d, dl = last[v, a]
                        self.preds[o.name].append((d.name, dl - lat + 1))
                    last[v, a] = (o, lat)
            for n, dist in o.after:
                self.preds[o.name].append((n, dist))
            if o.spill:
                v, p = o.spill
                self.spilled = getattr(self, "spilled", {})
                self.spilled[o.name] = {a: last[v, a][0].name for a in PARTS[p][1]}
        self.cpreds = defaultdict(list)  # round 16: op -> [(def op, lat + VIS - stage)]: t_op + II >= t_def + that
        for o in self.ops:
            for v, p, st in o.cuses:
                for a in PARTS[p][1]:
                    d, lat = last[v, a]  # the iteration's (last) def of v
                    self.cpreds[o.name].append((d.name, lat + VIS - st))
        self.succs = defaultdict(list)
        for n, ps in self.preds.items():
            for p, dist in ps:
                self.succs[p].append((n, dist))

    # ---- scheduling --------------------------------------------------------------------------------------------
    def res_mii(self, detail=False):
        cnt = defaultdict(int)
        for o in self.ops:
            for _, k in self._res(o):
                cnt[k] += 1
        return dict(sorted(cnt.items(), key=lambda kv: -kv[1])) if detail else max(cnt.values())

    @staticmethod
    def _res(o):
        """(stage, resource) pairs one op holds: its issue slot, its banks, its itinerary ports"""
        return [(0, o.slot)] + [(0, ("bank", b)) for b in o.banks] + [(st, ("port", r)) for st, r in o.ports]

    def _free(self, tab, o, t):
        return all(k not in tab[(t + st) % self.II] for st, k in self._res(o))

    def _take(self, tab, o, t, on=True):
        for st, k in self._res(o):
            (tab[(t + st) % self.II].add if on else tab[(t + st) % self.II].discard)(k)

    def schedule(self, II, sink=True, horizon=400):
        self.II = II
        tab = defaultdict(set)
        for o in self.ops:  # ops are in a topological order (defs before uses)
            lo = max([self.by_name[p].t + d for p, d in self.preds[o.name]], default=0)
            for t in range(lo, lo + horizon):
                if self._free(tab, o, t):
                    o.t = t
                    self._take(tab, o, t)
                    break
            else:
                raise Infeasible(f"{o.name}: no slot at II {II}")
        if sink:
            self._sink(tab)
        t0 = min(o.t for o in self.ops)
        for o in self.ops:
            o.t -= t0
        self.tab = tab
        return self

    def place(self, II, times):
        """round 12: take a schedule from elsewhere (r12/cpsched.py's CP-SAT model, or a saved JSON): {op name: t}.
        Checks dependences and the modulo resource table, then behaves as schedule() without the sink pass."""
        self.II = II
        missing = {o.name for o in self.ops} - set(times)
        assert not missing, f"no time for {sorted(missing)}"
        for o in self.ops:
            o.t = times[o.name]
        for o in self.ops:
            for p, d in self.preds[o.name]:
                assert o.t >= self.by_name[p].t + d, f"{o.name}@{o.t} breaks {p}@{self.by_name[p].t} + {d}"
            for p, d in self.cpreds.get(o.name, []):
                assert o.t + II >= self.by_name[p].t + d, f"{o.name}@{o.t} + II breaks the carried {p}@{self.by_name[p].t} + {d}"
        tab = defaultdict(set)
        for o in self.ops:
            assert self._free(tab, o, o.t), f"{o.name}@{o.t}: resource taken (II {II})"
            self._take(tab, o, o.t)
        self.tab = tab
        return self

    slack = 0  # extra cycles every register stays reserved after its last read (debug / margin against model errors)

    def schedule_ims(self, II, budget=20000, sink=True, seed=0):
        """Iterative modulo scheduling (Rau 1994): ops by height; an op that finds no free slot within [estart, estart + II)
        takes the first time >= estart (after its previous try) and evicts every op holding a resource it needs, and every
        scheduled successor whose dependence it now breaks."""
        import random
        rnd = random.Random(seed)
        self.II = II
        names = [o.name for o in self.ops]
        height = {}
        for o in reversed(self.ops):
            height[o.name] = max([d + height[sn] for sn, d in self.succs[o.name]], default=0)
        tab = defaultdict(set)
        holder = defaultdict(dict)  # mod slot -> resource -> op name
        sched, last = {}, {}

        def place(o, t):
            for st, k in self._res(o):
                r = (t + st) % II
                tab[r].add(k)
                holder[r][k] = o.name
            sched[o.name] = t
            o.t = t

        def unplace(n):
            o = self.by_name[n]
            for st, k in self._res(o):
                r = (sched[n] + st) % II
                tab[r].discard(k)
                holder[r].pop(k, None)
            del sched[n]

        todo = sorted(names, key=lambda n: (-height[n], names.index(n)))
        steps = 0
        while todo:
            steps += 1
            if steps > budget:
                raise Infeasible(f"IMS budget exhausted at II {II}")
            n = todo.pop(0)
            o = self.by_name[n]
            est = max([sched[p] + d for p, d in self.preds[n] if p in sched], default=0)
            t = next((t for t in range(est, est + II) if self._free(tab, o, t)), None)
            if t is None:
                t = max(est, last.get(n, est - 1) + 1)
                for st, k in self._res(o):
                    v = holder[(t + st) % II].get(k)
                    if v is not None and v in sched:
                        unplace(v)
                        todo.append(v)
            place(o, t)
            last[n] = t
            for sn, d in self.succs[n]:  # successors whose dependence broke
                if sn in sched and sched[sn] < t + d:
                    unplace(sn)
                    todo.append(sn)
            for pn, d in self.preds[n]:  # predecessors placed after us (possible after evictions)
                if pn in sched and t < sched[pn] + d:
                    unplace(pn)
                    todo.append(pn)
            todo.sort(key=lambda m: (-height[m], names.index(m)))
        t0 = min(o.t for o in self.ops)
        for o in self.ops:
            o.t -= t0
        for o in self.ops:  # rebuild the table at the normalised times for the sink pass
            pass
        tab = defaultdict(set)
        for o in self.ops:
            self._take(tab, o, o.t)
        if sink:
            self._sink(tab)
        self.tab = tab
        return self

    def lifetimes(self):
        """vreg -> (first landing, last read + slack) in iteration-relative cycles"""
        lo, hi = {}, {}
        for o in self.ops:
            for v, p, lat in o.defs:
                lo[v] = min(lo.get(v, 1 << 30), o.t + lat)  # may already be visible at c + w (the mv->mv bypass)
            for v, p, st in o.uses:
                hi[v] = max(hi.get(v, -(1 << 30)), o.t + st)
            for v, p, st in o.cuses:  # round 16: read by the NEXT iteration
                hi[v] = max(hi.get(v, -(1 << 30)), o.t + st + self.II)
        return {v: (lo[v], hi.get(v, lo[v]) + self.slack) for v in lo}

    def maxlive(self, lt=None):
        """class -> the most registers live in any cycle of the steady state (all overlapping iterations' copies)"""
        lt = lt or self.lifetimes()
        out = {}
        for c in set(self.vclass.values()):
            cnt = [0] * self.II
            for v, (l, h) in lt.items():
                if self.vclass[v] == c:
                    for t in range(l, h + 1):
                        cnt[t % self.II] += 1
            out[c] = max(cnt)
        return out

    def _cost(self):
        """lexicographic: registers over the class budgets, the peaks, then the summed lifetimes"""
        lt = self.lifetimes()
        ml = self.maxlive(lt)
        over = sum(max(0, ml[c] - len(self.classes[c])) for c in ml)
        return (over, sum(ml.values()), sum(h - l + 1 for l, h in lt.values()))

    def _sink(self, tab):
        """move ops later (or earlier) inside their dependence window when that shortens the summed register lifetimes"""
        changed = True
        while changed:
            changed = False
            for o in reversed(self.ops):
                lo = max([self.by_name[p].t + d for p, d in self.preds[o.name]], default=o.t)
                hi = min([self.by_name[s].t - d for s, d in self.succs[o.name]], default=o.t)
                best, bt = self._cost(), o.t
                self._take(tab, o, o.t, on=False)
                for t in range(lo, hi + 1):
                    if t == bt or not self._free(tab, o, t):
                        continue
                    old = o.t
                    o.t = t
                    c = self._cost()
                    o.t = old
                    if c < best:
                        best, bt = c, t
                if bt != o.t:
                    o.t = bt
                    changed = True
                self._take(tab, o, o.t)

    # ---- register allocation (modulo variable expansion) -----------------------------------------------------
    def allocate(self, U=None, tries=None):
        II = self.II
        if any(o.cuses for o in self.ops):  # round 16: one register per carried chain
            assert U in (None, 1), "carried uses need U = 1"
            U = 1
        lt = self.lifetimes()
        need = max(math.ceil((h - l + 1) / II) for l, h in lt.values())
        need = max(need, math.ceil(MIN_BODY / II))  # round 12: a 2-bundle zero-overhead loop body ran once (r12/cpyb)
        assert not U or U * II >= MIN_BODY, f"U {U} x II {II}: the loop body needs >= {MIN_BODY} bundles"
        for u in ([U] if U else range(need, need + 4)):
            res = self._color(lt, u)
            if res:
                self.U, self.alloc = u, res
                return self
        raise Infeasible(f"no register allocation at II {II} (copies needed {need}, pressure {self.pressure()}, maxlive {self.maxlive()})")

    def pressure(self):
        lt = self.lifetimes()
        out = {}
        for c in set(self.vclass.values()):
            out[c] = sum(h - l + 1 for v, (l, h) in lt.items() if self.vclass[v] == c) / self.II
        return out

    def _color(self, lt, U):
        for order in (lambda a: (-a[1], a[0]), lambda a: (a[0], -a[1]), lambda a: ((a[0] + a[1]) % (U * self.II), -a[1])):
            res = self._color1(lt, U, order)
            if res:
                return res
        return None

    def _color1(self, lt, U, order):
        P = U * self.II
        alloc = {}
        for c in set(self.vclass.values()):
            regs = self.classes[c]
            arcs = []
            for v, (l, h) in lt.items():
                if self.vclass[v] != c:
                    continue
                if h - l + 1 > P:
                    return None
                for k in range(U):
                    arcs.append(((l + k * self.II) % P, h - l + 1, v, k))
            arcs.sort(key=order)
            occ = {r: [] for r in regs}

            def clash(r, s, L):
                for s2, L2 in occ[r]:
                    # circular overlap of [s, s+L) and [s2, s2+L2) modulo P
                    for sh in (-P, 0, P):
                        if s < s2 + sh + L2 and s2 + sh < s + L:
                            return True
                return False

            for s, L, v, k in arcs:
                for r in regs:
                    if not clash(r, s, L):
                        occ[r].append((s, L))
                        alloc[v, k] = r
                        break
                else:
                    return None
        return alloc

    # ---- emission ----------------------------------------------------------------------------------------------
    def stages(self):
        return max(o.t for o in self.ops) // self.II + 1

    def text(self, o, i):
        k = i % self.U

        def sub(m):
            v, p = m.group(1), m.group(2)
            return PHYS[self.vclass[v]](self.alloc[v, k], p)

        return re.sub(r"\{(\w+)\.(\w+)\}", sub, o.fmt)

    def timeline(self, n_iter):
        """cycle -> [(op, iteration)] for iterations 0..n_iter-1, straight-line"""
        tl = defaultdict(list)
        for i in range(n_iter):
            for o in self.ops:
                tl[i * self.II + o.t].append((o, i))
        return tl

    def program(self):
        """(prologue bundles, body bundles): each bundle = [(op, iteration)], body iterations relative to trip 0"""
        S, II, U = self.stages(), self.II, self.U
        P = (S - 1) * II
        tl = self.timeline(S + U + 1)
        pro = [tl.get(c, []) for c in range(P)]
        body = [tl.get(c, []) for c in range(P, P + U * II)]
        for c, b in enumerate(body):
            for o, i in b:
                assert i >= 0
        return pro, body

    @staticmethod
    def bundle(texts_slots, full=False):
        """one bundle line from [(slot, text)]; full = the six-slot form (first / last bundle of a zero-overhead loop)"""
        bys = dict(texts_slots)
        assert len(bys) == len(texts_slots), texts_slots
        if full:
            return "\t" + ";\t\t".join(bys.get(s, NOP[s]) for s in SLOT_ORDER)
        if not bys:
            return "\tnop"
        return "\t" + ";\t\t".join(bys[s] for s in SLOT_ORDER if s in bys)

    body_full = None  # round 12: None = every body bundle in the full form; else the set to expand (r12/asmfit.py)
    kernel_only = False  # round 12: no prologue; the loop runs extra_trips() more trips whose early iterations are negative

    def extra_trips(self):
        """kernel-only: trips that replace the prologue ((S - 1) II cycles = ceil((S - 1) / U) trips). EVERY stream with
        negative-iteration accesses (check() reports them in self.neg_steps, loads included) must start that many steps
        early - round 12 sigk offset only its store pointer and came out shifted by 3 vectors."""
        return -(-(self.stages() - 1) // self.U)

    def emit_loop(self, label, pre_lines=(), drain=8):
        """asm lines: prologue (none if kernel_only), aligned loop body (the caller sets ls / le / lc), drain nops"""
        pro, body = self.program()
        if self.kernel_only:
            pro = []
        out = [f"\t// modulo schedule: II {self.II}, U {self.U}, stages {self.stages()}, ops {len(self.ops)}"
               + (f", kernel-only (+{self.extra_trips()} trips)" if self.kernel_only else "")]
        for b in pro:
            out.append(self.bundle([(o.slot, self.text(o, i)) for o, i in b]))
        out.append("\t.p2align\t4")
        out.append(f"{label}_start:")
        for c, b in enumerate(body):
            if c == len(body) - 1:
                out.append(f"{label}_last:")
            # the full 16-byte form for every body bundle (default): the start is 16 B aligned, so the last bundle is too (the
            # hardware loop needs an aligned last bundle; assembler padding inside the body would shift the schedule). Round 12
            # (r12/asmfit.py): body_full = the bundles to expand (first and last always), chosen from the assembled sizes so
            # the last bundle still lands aligned, the rest compact (text).
            full = self.body_full is None or c in self.body_full or c in (0, len(body) - 1)
            out.append(self.bundle([(o.slot, self.text(o, i)) for o, i in b], full=full))
        out += ["\tnop"] * drain
        return out

    # ---- checking ----------------------------------------------------------------------------------------------
    PADS = (0, 1)  # round 12: `.p2align 4` before the body inserts one executable nop bundle (<= 15 B) after the prologue (bgelu:
    # one 12 B bundle). They delay the body against ops already issued in the prologue, which broke a U = 2 schedule
    # (a prologue write to a rotating register landed before an older iteration's delayed read). check() replays every pad.

    def check(self, n, streams, drain=8, pads=None):
        """check_one for every prologue -> body padding in PADS; returns the pad-0 result"""
        res = None
        for pad in (self.PADS if pads is None else pads):
            try:
                r = self.check_one(n, streams, drain, pad)
            except AssertionError as e:
                raise AssertionError(f"with {pad} padding bundle(s) after the prologue: {e}") from None
            res = res or r
        return res

    def check_one(self, n, streams, drain=8, pad=0):
        """Replay prologue + T = n / U trips. streams: name -> (address function of iteration, 2D params dict
        {base, dn, dj, m}) for every mem stream. Events (register writes landing, operand reads) are replayed in time
        order, writes before reads within a cycle. Raises AssertionError on the first wrong read / address / clobber.
        Returns (cycles issued, last landing cycle)."""
        assert n % self.U == 0
        pro, body = self.program()
        T = n // self.U
        if self.kernel_only:  # extra_trips() trips run the body over negative iterations first (no prologue, no padding)
            seq = []
            for t in range(-self.extra_trips(), T):
                seq += [(b, t * self.U) for b in body]
        else:
            seq = [(b, 0) for b in pro] + [([], 0)] * pad
            for t in range(T):
                seq += [(b, t * self.U) for b in body]
        # kernel-only: a stream's pointer has advanced once per negative-iteration access before iteration 0's; start it that
        # many (uniform) steps back. self.neg_steps reports the counts for the generator's preamble.
        self.neg_steps = defaultdict(int)
        for b, ibase in seq:
            for o, i0 in b:
                if o.mem and i0 + ibase < 0:
                    self.neg_steps[o.mem[1]] += 1
        const_atoms = {(c, r, a) for c, rs in self.consts.items() for r in rs for a in (PARTS["x"][1] if c == "x" else range(4))}
        ptr = {}
        for s_, (f, prm) in streams.items():
            k = self.neg_steps.get(s_, 0)
            assert not k or prm["dj"] == prm["m"], f"stream {s_}: negative-iteration accesses need a uniform step"
            ptr[s_] = dict(p=prm["base"] - k * prm["dj"], dc=0, **prm)
            # round 14: per-op modifiers (an op with its own 2D modifier register, or a post-increment immediate), each with its
            # own counter: prm["ops"] = {op name: dict(dn, dj, m)}; ops of kind "pa" (padda.2d etc.) only move the pointer
            ptr[s_]["dcs"] = defaultdict(int)
            for on, om in prm.get("ops", {}).items():  # round 18: a 2D counter that starts mid-period (a ring wrap inside the call)
                if "dc0" in om:
                    ptr[s_]["dcs"][on] = om["dc0"]
        # round 12: a stream with several ops per iteration: f(i, k) gets k = the op's index among the stream's ops (op order)
        kth = {}
        for o in self.ops:
            if o.mem:
                kth[o.name] = sum(1 for x in self.ops[:self.ops.index(o)] if x.mem and x.mem[1] == o.mem[1] and x.mem[0] != "pa")
        nargs = {s: f.__code__.co_argcount - len(f.__defaults__ or ()) for s, (f, prm) in streams.items()}
        done = defaultdict(set)
        ev = []  # (time, 0 = write / 1 = read, atom, value or wanted, text)
        busy = {}  # (cycle, resource) -> op text: itinerary ports along the REAL replay (with the padding after the prologue)
        for cyc, (b, ibase) in enumerate(seq):
            for o, i0 in b:
                i = i0 + ibase
                for st, k in self._res(o):
                    if isinstance(k, tuple) and k[0] == "port":
                        assert (cyc + st, k) not in busy, f"cycle {cyc} {o.name} it {i}: port {k[1]} at {cyc + st} taken by {busy[cyc + st, k]}"
                        busy[cyc + st, k] = f"{o.name} it {i}"
                k = i % self.U
                for v, p, st in o.cuses:  # round 16: the previous iteration's value; iteration 0 reads the preamble's ("init")
                    c = self.vclass[v]
                    for a in PARTS[p][1]:
                        if 0 <= i < n:
                            want = (v, i - 1, dict(self.cpreds[o.name])and next(dn for dn, _ in self.cpreds[o.name])) if i else (v, -1, "init")
                            ev.append((cyc + st, 1, (c, self.alloc[v, 0], a), want, f"cycle {cyc} {o.name} it {i} (carried)"))
                for ui, (v, p, st) in enumerate(o.uses):
                    c = self.vclass[v]
                    for a in PARTS[p][1]:
                        if 0 <= i < n:  # garbage iterations (>= n, or < 0 kernel-only) may read anything
                            ev.append((cyc + st, 1, (c, self.alloc[v, k], a), (v, i, self.reach[o.name, ui, a]), f"cycle {cyc} {o.name} it {i}"))
                for v, p, lat in o.defs:
                    c = self.vclass[v]
                    for a in PARTS[p][1]:
                        atom = (c, self.alloc[v, k], a)
                        assert atom not in const_atoms, f"{o.name} writes constant {atom}"
                        ev.append((cyc + lat + VIS, 0, atom, (v, i, o.name), f"cycle {cyc} {o.name} it {i}"))
                if o.mem:
                    kind, s = o.mem
                    f, prm = streams[s]
                    q = ptr[s]
                    if 0 <= i < n and kind != "pa":
                        want = f(i) if nargs[s] == 1 else f(i, kth[o.name])
                        assert q["p"] == prm["base"] + want, f"{o.name} it {i}: address {q['p'] - prm['base']} wants {want}"
                        done[s].add(i)
                    if o.spill:  # the store's data are written MEM_W cycles after issue (assumed; the schedule keeps margin)
                        v, p = o.spill
                        for a in PARTS[p][1]:
                            ev.append((cyc + MEM_W, 0, ("mem", q["p"], a), (v, i, self.spilled[o.name][a]), f"cycle {cyc} {o.name} it {i}"))
                    if o.fill and 0 <= i < n:
                        v, p = o.fill
                        for a in PARTS[p][1]:
                            src = next(x for x in self.ops if x.spill and x.spill[0] == v)
                            ev.append((cyc + MEM_R, 1, ("mem", q["p"], a), (v, i, self.spilled[src.name][a]), f"cycle {cyc} {o.name} it {i}"))
                    # 2D post-modify (aie2p_addr.h add_2d): count < dn: p += dj, count++; else p += m, count = 0
                    om = prm.get("ops", {}).get(o.name)
                    if om is not None and "dn1" in om:  # round 14: 3D post-modify (inner dn / dj, then dn1 / dj1, then m)
                        c0, c1 = q["dcs"][o.name, 0], q["dcs"][o.name, 1]
                        if c0 < om["dn"]:
                            q["p"] += om["dj"]
                            q["dcs"][o.name, 0] = c0 + 1
                        elif c1 < om["dn1"]:
                            q["p"] += om["dj1"]
                            q["dcs"][o.name, 0], q["dcs"][o.name, 1] = 0, c1 + 1
                        else:
                            q["p"] += om["m"]
                            q["dcs"][o.name, 0], q["dcs"][o.name, 1] = 0, 0
                    elif om is not None:
                        if q["dcs"][o.name] < om["dn"]:
                            q["p"] += om["dj"]
                            q["dcs"][o.name] += 1
                        else:
                            q["p"] += om["m"]
                            q["dcs"][o.name] = 0
                    elif q["dc"] < q["dn"]:
                        q["p"] += q["dj"]
                        q["dc"] += 1
                    else:
                        q["p"] += q["m"]
                        q["dc"] = 0
        for o in self.ops:  # round 16: carried chains start from the preamble's value
            for v, p, st in o.cuses:
                for a in PARTS[p][1]:
                    ev.append((-1, 0, (self.vclass[v], self.alloc[v, 0], a), (v, -1, "init"), "preamble"))
        # a register write (event time = c + w + 1) may already be seen at c + w (measured: the mv -> mv bypass), so a read
        # at T is certain only if its write is visible by T (time <= T) and no later write could be (time - 1 > T)
        writes = defaultdict(list)
        for tm, kind, atom, val, txt in ev:
            if kind == 0:
                writes[atom].append((tm, val, txt))
        for atom, ws in writes.items():
            ws.sort(key=lambda w: w[0])
            for a, b in zip(ws, ws[1:]):
                assert a[0] != b[0], f"two writes land on {atom} at {a[0]}: {a[2]} / {b[2]}"
        for tm, kind, atom, val, txt in ev:
            if kind != 1:
                continue
            ws = writes.get(atom, [])
            if atom[0] == "mem":  # memory: no bypass model; the last write with time <= T
                got = max((w for w in ws if w[0] <= tm), default=(None, None, None), key=lambda w: w[0])[1]
                assert got == val, f"{txt} reads {atom} at {tm} = {got}, wants {val}"
                continue
            seen = [w for w in ws if w[0] <= tm]
            assert seen and seen[-1][1] == val, f"{txt} reads {atom} at {tm} = {seen[-1][1] if seen else None}, wants {val}"
            nxt = [w for w in ws if w[0] > tm]
            # round 12: the one-cycle window is the measured mv -> mv bypass only (an mv-slot writer, an mv-slot reader);
            # applied to every pair it flagged reads that run bit-exact on the device
            early = nxt and self.by_name[nxt[0][2].split()[2]].slot == "m" and self.by_name[txt.split()[2]].slot == "m"
            assert not nxt or nxt[0][0] - (1 if early else 0) > tm, \
                f"{txt} reads {atom} at {tm}: the next write ({nxt[0][2]}) may already be visible"
        for s in streams:
            assert done[s] == set(range(n)), f"stream {s}: iterations {sorted(set(range(n)) - done[s])[:5]} missing"
        return len(seq), max(e[0] for e in ev if e[1] == 0)
