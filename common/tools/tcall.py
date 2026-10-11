#!/usr/bin/env python3
"""tcall.py - static call census and whole-call cycle counts for AIE2P cores (npu-research rev 12, T-CALL).

Round 15's largest gain (div768, -1.05 ms) was a minsize helper called once per unit of work that no price counted, and
rounds 15-16 missed four prices by the per-call overhead (prologue, preamble, call / ret delay slots). This tool reads a
`llvm-objdump -d -z` disassembly (asmlint's parser) and reports, statically:

  * per function: whole-call cycles from entry to the end of `ret`'s delay slots (min / max over branches, loops at their
    trip counts) and the part outside zero-overhead-loop bodies ("glue": setup, prologue, preamble, restores, delay slots,
    helper calls) - the per-call cost a loop-only price misses;
  * per call site: the loop nest around it (CFG natural loops + ZOLs, trip counts inferred or given), so calls per
    invocation of a root function and calls x whole-call cycles;
  * a census: roots x invocations per step (spec) -> the top call sites by static cycles per step.

    tcall.py r17/r16z_0_5.dis --func norm_rows_k --spec tools/tcall_r16z.json   # cost, loops, call sites of one function
    tcall.py r17/r16z_0_5.dis --census tools/tcall_r16z.json                     # top call sites per step
    tcall.py r17/r16z_0_5.dis --all                                              # every function's whole-call cost

Model: one bundle = one cycle; NO stalls (lock waits, bank conflicts, DMA back-pressure: an in-situ stamp minus this count
measures them). j / jz / jnz / jl / ret execute 5 delay-slot bundles before the transfer. A ZOL body runs `lc` times (the
value written by `add.nc lc, rX, #imm`: Peano writes n - 1 and peels one iteration after the loop; generated loops write
the trip count). CFG loops are natural loops (dominator back edges, so rotated loops entered in the middle are found);
trip counts come from `mov rX, #n` before the loop + `add rX, rX, #-1` + a `jnz rX` latch, else from the spec
({"trips": {"func": {"0xHEAD": n}}}, HEAD = the loop header's address as --func prints it); unknown = 1, marked '?'.
A loop costs trips x its longest (shortest) iteration; the partial exit iteration is ignored (a few bundles).
Host-only (python 3, no toolchain) from a .dis.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asmlint  # noqa: E402  (the parser and the ZOL finder)

DS = asmlint.DELAY_SLOTS
IMM_RE = re.compile(r"#(-?0x[0-9a-f]+|-?\d+)")
Z = (0, 0, 0, 0)


class StubModel:
    """asmlint's parser wants an instruction model; the census needs only mnemonics and operands"""

    optypes = {}

    def match(self, mn, ops):
        return None


def imm(s):
    m = IMM_RE.search(s)
    return int(m.group(1), 0) if m else None


def add(*ts):
    return tuple(sum(x) for x in zip(*ts))


def short(n, w=26):
    m = re.match(r"_ZL\d+(\w+?)(P[KS]?[A-Za-z0-9]|N3aie|i?$|v$)", n)
    n = m.group(1) if m else n
    return n[:w]


class Fn:
    """one function: bundle graph (delay slots modelled), dominators, natural loops, ZOLs"""

    def __init__(self, prog, f, spec, force=None):
        self.P, self.f, self.name = prog, f, f.name
        self.B = f.bundles
        self.n = len(self.B)
        self.ix = {b.addr: k for k, b in enumerate(self.B)}
        self.call = {}  # bundle -> callee name (jl, or a tail jump out of the function)
        self.succ = [[] for _ in range(self.n)]
        self.exit = set()
        pend = {}  # bundle index whose delay slots end here -> (kind, target index or None)
        for k, b in enumerate(self.B):
            br = prog.branch(b)
            if br:
                kind, t, reg = br
                end = min(k + DS, self.n - 1)
                if kind == "jl":
                    cf = prog.by_addr.get(t)
                    if cf:
                        self.call[k] = cf.name
                elif kind == "ret":
                    pend[end] = ("ret", None)
                else:
                    tk = self.ix.get(t) if t is not None else None
                    if tk is None and t is not None and prog.by_addr.get(t):
                        self.call[k] = prog.by_addr[t].name  # tail call
                        pend[end] = ("ret" if kind == "j" else "cond_out", None)
                    else:
                        fz = (force or {}).get(f"0x{b.addr:x}")
                        if fz is not None and kind in ("jz", "jnz"):
                            kind = "j" if fz else "never"
                        pend[end] = (kind, tk)
            if k in pend:
                kind, tk = pend.pop(k)
                if kind == "ret":
                    self.exit.add(k)
                    continue
                if kind == "never":
                    if k + 1 < self.n:
                        self.succ[k].append(k + 1)
                    continue
                if tk is not None:
                    self.succ[k].append(tk)
                if kind in ("jz", "jnz", "cond_out") and k + 1 < self.n:
                    self.succ[k].append(k + 1)
                if kind == "cond_out" and k + 1 >= self.n:
                    self.exit.add(k)
            elif k + 1 < self.n:
                self.succ[k].append(k + 1)
            else:
                self.exit.add(k)
        # ZOL bodies: straight-line ranges executed lc times
        self.zol = []
        fspec = spec.get(self.name) or spec.get(short(self.name, 99)) or {}
        zspec = {int(a, 16): n for a, n in fspec.items()}
        for ls, le, setup in asmlint.zol_loops(f):
            i0, i1 = self.ix.get(ls), self.ix.get(le)
            if i0 is None or i1 is None:
                continue
            n = zspec.get(ls)
            if n is None and "lc" in setup:
                for i in self.B[setup["lc"]].instrs:
                    m = re.match(r"\s*lc\s*,\s*(\w+)\s*,\s*(#\S+)", i.ops)
                    if i.mn == "add.nc" and m:
                        v = self.const_before(setup["lc"] - 1, m.group(1))
                        n = None if v is None else v + imm(m.group(2))
            known = n is not None
            if not known:
                n = 1
                prog.unknown.add((self.name, ls, "zol"))
            self.zol.append(dict(head=i0, tail=i1, trips=n, known=known))
        self.inzol = {}
        for z in self.zol:
            for k in range(z["head"], z["tail"] + 1):
                self.inzol[k] = z
        self.reach = self._reachable()
        self.dom = self._dominators()
        self.loops = self._loops(fspec)

    def const_before(self, k, reg):
        """last immediate move to `reg` at or before bundle k in address order (None if another write comes first)"""
        for j in range(k, -1, -1):
            for i in self.B[j].instrs:
                ops = [o.strip() for o in i.ops.split(",")]
                if ops and ops[0] == reg and not i.is_store() and not i.mn.startswith(("j", "ret")):
                    if i.mn in ("mova", "movx", "movxm", "mov") and len(ops) == 2 and ops[1].startswith("#"):
                        return imm(ops[1])
                    return None
        return None

    def _reachable(self):
        seen, st = {0}, [0]
        while st:
            k = st.pop()
            for s in self.succ[k]:
                if s not in seen:
                    seen.add(s)
                    st.append(s)
        return seen

    def _dominators(self):
        nodes = sorted(self.reach)
        pred = {k: [] for k in nodes}
        for k in nodes:
            for s in self.succ[k]:
                pred[s].append(k)
        self.pred = pred
        dom = {k: set(nodes) for k in nodes}
        dom[0] = {0}
        changed = True
        while changed:
            changed = False
            for k in nodes[1:]:
                ps = [dom[p] for p in pred[k] if p in dom]
                d = set.intersection(*ps) | {k} if ps else {k}
                if d != dom[k]:
                    dom[k], changed = d, True
        return dom

    def _loops(self, spec):
        heads = {}
        for k in self.reach:
            for s in self.succ[k]:
                if s in self.dom[k]:  # back edge k -> s
                    heads.setdefault(s, []).append(k)
        loops = []
        for h, latches in heads.items():
            body, st = {h}, list(latches)
            while st:
                x = st.pop()
                if x not in body:
                    body.add(x)
                    st.extend(self.pred[x])
            L = dict(head=h, latches=latches, body=body, addr=self.B[h].addr)
            n = spec.get(f"0x{self.B[h].addr:x}")
            known = n is not None
            if n is None:
                n = self._infer(L)
                known = n is not None
            if not known:
                n = 1
                self.P.unknown.add((self.name, self.B[h].addr, "loop"))
            L["trips"], L["known"] = n, known
            L["exits"] = sorted({s for x in body for s in self.succ[x] if s not in body})
            loops.append(L)
        loops.sort(key=lambda L: len(L["body"]))  # inner first
        for L in loops:
            L["parent"] = next((M for M in loops if M is not L and L["body"] < M["body"]), None)
        return loops

    def _infer(self, L):
        """counted loop: a latch `jnz rX` (to the header) with `add rX, rX, #-1` in the body and `mov rX, #n` before"""
        for lat in L["latches"]:
            for j in range(lat, max(-1, lat - DS - 1), -1):
                br = self.P.branch(self.B[j])
                if br and br[0] == "jnz" and br[2]:
                    reg = br[2]
                    dec = [x for x in L["body"] for i in self.B[x].instrs
                           if i.mn == "add" and re.match(rf"\s*{reg}\s*,\s*{reg}\s*,\s*#-(0x)?1\s*$", i.ops)]
                    if len(dec) != 1:
                        return None
                    pre = [p for p in self.pred[L["head"]] if p not in L["body"]]
                    if not pre:
                        return None
                    v = self.const_before(max(pre), reg)
                    if v is None:
                        return None
                    # decrement before the test in iteration order (header .. test) -> n trips, after it -> n + 1
                    return v if L["head"] <= dec[0] <= j or (dec[0] >= L["head"] and j < L["head"] and dec[0] <= j) else v + 1
        return None

    # ---- costs -----------------------------------------------------------------------------------------------------
    def wt(self, k):
        """(min, max, glue_min, glue_max) of one execution of bundle k incl. a callee"""
        z = self.inzol.get(k)
        n = z["trips"] if z else 1
        c = self.P.cost(self.call[k]) if k in self.call else Z
        own = (n, n, 0, 0) if z else (1, 1, 1, 1)
        return add(own, tuple(x * n for x in c))

    def seg(self, a, b):
        """min / max cost from bundle address a up to (not incl.) bundle address b, the function's loops collapsed"""
        ka, kb = self.ix[a], self.ix[b]
        enc = [L for L in self.loops if ka in L["body"] and kb in L["body"]]
        if enc:  # walk inside the innermost loop holding both ends (one iteration; the back edge ends the walk)
            L = min(enc, key=lambda L: len(L["body"]))
            return self.region(ka, L["body"], stop_at_head=L["head"], stop=kb)
        return self.region(ka, self.reach, stop=kb)

    def region(self, start, nodes, stop_at_head=None, stop=None):
        """min / max cost of paths from `start` through `nodes` (a node set) to an end: an exit of the function, a node
        whose successors leave `nodes`, or (iteration mode) a latch back to stop_at_head. Inner loops are collapsed."""
        inner = [L for L in self.loops if L["body"] < nodes or (stop_at_head is None and L["body"] <= nodes)]
        inner = [L for L in inner if L["head"] != stop_at_head]
        top = [L for L in inner if not any(L["body"] < M["body"] for M in inner)]
        owner = {}
        for L in top:
            for x in L["body"]:
                owner[x] = L
        memo = {}

        def go(k):
            if k == stop:
                return Z
            if k in memo:
                return memo[k]
            memo[k] = Z  # cycle guard (irreducible leftovers)
            L = owner.get(k)
            if L is not None:
                it = self.region(L["head"], L["body"], stop_at_head=L["head"])
                tot = tuple(x * L["trips"] for x in it)
                outs = [s for s in L["exits"] if s in nodes]
                r = self._best([go(s) for s in outs]) if outs else Z
                memo[k] = add(tot, r)
                return memo[k]
            here = self.wt(k)
            nxt = []
            for s in self.succ[k]:
                if stop_at_head is not None and s == stop_at_head:
                    nxt.append(Z)  # iteration ends
                elif s in nodes:
                    nxt.append(go(s))
                elif stop_at_head is None:
                    nxt.append(Z)
            if stop_at_head is not None:
                # inside an iteration, exits are not taken (unless nothing else)
                inl = [go(s) if s != stop_at_head else Z for s in self.succ[k] if s in nodes or s == stop_at_head]
                nxt = inl or nxt
            memo[k] = add(here, self._best(nxt) if nxt else Z)
            return memo[k]

        return go(start)

    @staticmethod
    def _best(rs):
        return (min(r[0] for r in rs), max(r[1] for r in rs), min(r[2] for r in rs), max(r[3] for r in rs))

    def loops_around(self, k):
        out = [L for L in self.loops if k in L["body"]]
        out.sort(key=lambda L: -len(L["body"]))
        return out


class Prog:
    def __init__(self, text, spec=None, branches=None):
        self.spec = spec or {}
        self.branches = branches or {}
        funcs, _ = asmlint.parse_dis(text, StubModel())
        self.funcs = funcs
        self.by_addr = {f.start: f for f in funcs.values()}
        self.unknown = set()
        self.memo = {}
        self.fn = {}

    def get(self, name):
        if name not in self.fn:
            br = self.branches.get(name) or self.branches.get(short(name, 99))
            self.fn[name] = Fn(self, self.funcs[name], self.spec, br)
        return self.fn[name]

    def branch(self, b):
        for i in b.instrs:
            if i.mn in ("j", "jz", "jnz", "jl", "ret"):
                t = imm(i.ops) if "#" in i.ops else None
                reg = i.ops.split(",")[0].strip() if i.mn in ("jz", "jnz") else None
                return i.mn, t, reg
        return None

    def cost(self, name):
        if name in self.memo:
            return self.memo[name]
        self.memo[name] = Z  # recursion guard
        F = self.get(name)
        self.memo[name] = F.region(0, F.reach) if F.n else Z
        return self.memo[name]

    def find(self, name):
        return next((n for n in self.funcs if n == name or short(n, 99) == name), None)

    def census(self, root, per_root, path=(), acc=None):
        acc = [] if acc is None else acc
        F = self.get(root)
        for k, callee in sorted(F.call.items()):
            if k not in F.reach:
                continue
            m = 1
            nest = F.loops_around(k)
            for L in nest:
                m *= L["trips"]
            z = F.inzol.get(k)
            m *= z["trips"] if z else 1
            n = m * per_root
            c = self.cost(callee)
            acc.append(dict(caller=root, addr=F.B[k].addr, callee=callee, calls_per_caller=m, calls=n, cost=c,
                            cyc=(n * c[0], n * c[1]), glue=(n * c[2], n * c[3]),
                            nest=[(L["addr"], L["trips"], L["known"]) for L in nest]))
            if callee not in path + (root,) and len(path) < 6:
                self.census(callee, n, path + (root,), acc)
        return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dis")
    ap.add_argument("--func", help="report one function: whole-call cost, loops, call sites")
    ap.add_argument("--all", action="store_true", help="every function's whole-call cost")
    ap.add_argument("--spec", help="json: {trips: {func: {0xHEAD: n}}, branches: {func: {0xBRANCH: taken?}}, roots: {func: calls per "
                    "step}, clock_ghz}; branches pin a mode (e.g. LN vs RMS paths of one function)")
    ap.add_argument("--seg", action="append", default=[], help="FUNC:0xA:0xB cost of the path A -> B (B excluded); repeatable")
    ap.add_argument("--census", action="store_true", help="top call sites per step from the spec's roots")
    ap.add_argument("--top", type=int, default=25)
    o = ap.parse_args()
    spec = json.load(open(o.spec)) if o.spec else {}
    P = Prog(Path(o.dis).read_text(), spec.get("trips"), spec.get("branches"))
    ghz = spec.get("clock_ghz", 1.81)
    if o.all:
        print(f"{'function':28s} {'min':>7s} {'max':>7s} {'glue':>13s}  loops (header x trips)")
        for f in sorted(P.funcs.values(), key=lambda f: f.start):
            if not f.bundles:
                continue
            c = P.cost(f.name)
            F = P.get(f.name)
            lp = " ".join([f"L@0x{L['addr']:x}x{L['trips']}{'' if L['known'] else '?'}" for L in F.loops] +
                          [f"Z@0x{F.B[z['head']].addr:x}x{z['trips']}{'' if z['known'] else '?'}" for z in F.zol])
            print(f"{short(f.name, 28):28s} {c[0]:7d} {c[1]:7d} {c[2]:6d}-{c[3]:<6d}  {lp}")
    if o.func:
        name = P.find(o.func)
        if name is None:
            sys.exit(f"tcall: no function {o.func}")
        F = P.get(name)
        c = P.cost(name)
        print(f"{name}: whole call {c[0]}..{c[1]} cycles (glue {c[2]}..{c[3]}), {F.n} bundles")
        for L in sorted(F.loops, key=lambda L: L["addr"]):
            it = F.region(L["head"], L["body"], stop_at_head=L["head"])
            ex = ",".join(f"0x{F.B[s].addr:x}" for s in L["exits"])
            print(f"  loop @0x{L['addr']:x} ({len(L['body'])} bundles) trips {L['trips']}{'' if L['known'] else ' (unknown: spec it)'}"
                  f" iteration {it[0]}..{it[1]} (glue {it[2]}..{it[3]}) exits {ex}")
        for z in F.zol:
            print(f"  zol  @0x{F.B[z['head']].addr:x} body {z['tail'] - z['head'] + 1} bundles x {z['trips']}{'' if z['known'] else '?'}")
        for k, callee in sorted(F.call.items()):
            if k not in F.reach:
                continue
            cc = P.cost(callee)
            nest = F.loops_around(k)
            m = 1
            for L in nest:
                m *= L["trips"]
            s = " > ".join(f"0x{L['addr']:x}x{L['trips']}{'' if L['known'] else '?'}" for L in nest) or "-"
            print(f"  call @0x{F.B[k].addr:x} -> {short(callee):26s} x{m:<5d} [{s}] callee {cc[0]}..{cc[1]} (glue {cc[2]}..{cc[3]})")
    for sg in o.seg:
        fn, a, b = sg.split(":")
        F = P.get(P.find(fn))
        c = F.seg(int(a, 16), int(b, 16))
        print(f"seg {fn} {a}..{b}: {c[0]}..{c[1]} cycles (glue {c[2]}..{c[3]})")
    if o.census:
        rows = []
        for root, per in spec["roots"].items():
            name = P.find(root)
            if name is None:
                print(f"tcall: root {root} not found", file=sys.stderr)
                continue
            rc = P.cost(name)
            print(f"root {short(name):24s} x{per:<4d} per step: {rc[0]}..{rc[1]} cycles per call (glue {rc[2]}..{rc[3]}) = "
                  f"{rc[0] * per / ghz / 1e6:.2f}..{rc[1] * per / ghz / 1e6:.2f} ms per step (static, no waits)")
            rows += P.census(name, per)
        rows.sort(key=lambda r: -r["cyc"][1])
        print(f"\ntop {o.top} call sites by static cycles per step ({ghz} GHz). glue = outside ZOL bodies (call overhead in the wide sense)")
        print(f"{'caller':20s} {'@':>5s} {'callee':22s} {'/caller':>7s} {'/step':>8s} {'cyc/call':>11s} {'ms/step':>11s} {'glue ms':>11s}")
        for r in rows[:o.top]:
            print(f"{short(r['caller'], 20):20s} {r['addr']:5x} {short(r['callee'], 22):22s} {r['calls_per_caller']:7d} {r['calls']:8d} "
                  f"{r['cost'][0]:5d}-{r['cost'][1]:<5d} {r['cyc'][0] / ghz / 1e6:5.2f}-{r['cyc'][1] / ghz / 1e6:<5.2f} "
                  f"{r['glue'][0] / ghz / 1e6:5.2f}-{r['glue'][1] / ghz / 1e6:<5.2f}")
    if P.unknown:
        print("\nunknown trip counts (1 assumed; give them in the spec):", file=sys.stderr)
        for fn, a, kind in sorted(P.unknown):
            print(f"  {short(fn, 40)} {kind} head 0x{a:x}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
