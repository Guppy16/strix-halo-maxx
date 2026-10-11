#!/usr/bin/env python3
"""asmlint.py - offline hazard linter for AIE2P (XDNA2) kernels (npu-research rev 11).

Checks the hardware rules of the xdna2-npu skill that can be decided from the instruction stream alone, on built elfs,
merged objects, generated .s files or `llvm-objdump -d -z` text. Nothing runs on the NPU.

    asmlint.py build/full12r15m.prj/elfs_main_core_0_5/elfs_main_core_0_5.elf       # every function of a core
    asmlint.py build/r15/ssx.s build/r15/reck.s                                     # generated kernels before a build
    asmlint.py X.dis --funcs ssx,reck                                                # pre-made disassembly (host side)
    asmlint.py --prj build/full12r16a.prj                                            # one GEMM + one EW core of a build
    asmlint.py ... --asm-only        only generated / hand asm functions (those with <f>_start / <f>_last labels or from .s)
    asmlint.py ... --json            machine-readable findings (try.sh uses it)
    asmlint.py --rules               the rule table: id, severity, status (verified / from the .td / unverified), source

Needs Peano (llvm-objdump, clang, ld.lld) for .elf / .o / .s inputs: run it in the npu-iron container (CPU only):
    docker run --rm --network none -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron python3 $HOME/git/app/strix-halo-maxx/common/tools/asmlint.py ...
A .dis input needs no toolchain (made with `llvm-objdump -d -z --no-show-raw-insn`; -z matters: without it runs of
2-byte nops print as `...` and every distance is wrong).

Model (SKILL 3 / 4): one bundle = one issue cycle (stalls freeze the whole pipeline, so relative timing holds); a write with
.td latency w issued at c is seen by an operand read at stage r of an op issued at c2 iff c2 + r - 1 >= c + w (LLVM's
w - r + 1, measured round 11), except that an mv-slot write reaches mv-slot readers one cycle early; until then readers see
the older value (no interlocks). Ports: llvm-aie's itineraries (aie2p_isa_timing.json), scalar `lda` given the DM_ADA itinerary
that round 13 measured. Control flow: a stream per function in address order, reset after calls (jl + 5 delay slots) and at
branch targets; every zero-overhead loop body is replayed 3x (prologue -> body -> body -> body -> after the loop) so
loop-carried hazards and the prologue's ops (incl. the .p2align pad, which is in the binary) are seen.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PEANO_DEFAULT = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin"
DELAY_SLOTS = 5  # j / jz / jnz / jl / ret: 5 delay-slot bundles (Peano's output)
CALLEE_SAVED = [f"r{i}" for i in range(8, 16)] + ["p6", "p7"]

# ---------------------------------------------------------------------------------------------------------------------
# The rule table. status: "measured" (a hardware run taught it; the round is named), "td" (llvm-aie's .td / assembler),
# "peano" (what Peano's own output always does), "unverified" (a hypothesis; the named microbench decides it).
RULES = {
    "ZOL-END-ALIGN": ("error", "measured r11", "the last body bundle of a zero-overhead loop (le) must be 16 B aligned, else the body runs about once"),
    "ZOL-START-ALIGN": ("error", "rule r9 (all deployed loops comply)", "the loop start (ls) must be 16 B aligned (.p2align 4)"),
    "ZOL-MIN-BODY": ("error", "measured r12", "a zero-overhead loop body needs >= 3 bundles (2 ran once)"),
    "ZOL-SETUP": ("error", "measured r14", "<= 52 B of instructions between the last lc / ls / le write and the loop end (le): the body runs once"),
    "ZOL-SETUP-MARGIN": ("warning", "peano", "setup -> loop end between the measured bad 52 B and the smallest deployed 80 B (unverified gap)"),
    "ZOL-FULL-EDGE": ("warning", "peano", "first / last body bundle not in the full 16-byte form"),
    "ZOL-NO-LC": ("warning", "rule r9", "ls / le set but no lc write before the loop in this function"),
    "LAT-OLD-SCALAR": ("error", "measured r11 / r14", "a scalar register is read while a newer write to it is still in flight: the read sees the OLD value"),
    "LAT-OLD-VECTOR": ("info", "measured r11", "a vector read sees an older value while a newer write is in flight (modulo-scheduled conveyor reads do this on purpose)"),
    "LAT-UNLANDED": ("error", "measured r11", "a read before ANY write to the register in this stream has landed (sees the caller's / stale value)"),
    "LAT-MV-BYPASS": ("error", "measured r11", "an mv-slot reader one cycle before an mv-slot write lands sees the NEW value (bypass), but an older def was meant"),
    "LAT-WAW": ("warning", "measured r11", "an older long-latency write lands after a younger short one: the reader sees the older value (landing order)"),
    "SCALAR-PLUS1": ("warning", "measured once r14 (unverified scope)", "an alu op reads an alu result one cycle later inside a zero-overhead loop body (attkb FO -> FA saw the previous iteration's r4); Peano does alu -> alu at +1 outside ZOL bodies and deployed kernels do alu -> mv at +1: microbench named in MACHINE rev 11"),
    "PORT": ("error", "td itineraries + measured r11 / r13", "two ops use one itinerary port (DM_ADA, QEY_*, ...) in the same cycle"),
    "EX-ALIAS": ("error", "measured r13", "ex_n aliases x_n: an exN read sees an x / w write of the same number (or an xN read sees a pop into exN)"),
    "ABI-R0": ("error", "measured r13", "a value-returning function reads r0 as an argument (r0 is reserved for the result; the first scalar arg is r1)"),
    "ABI-CALLEE-SAVED": ("error", "measured r15 + ABI", "a callee-saved register (r8-r15, p6, p7) is written without a save / restore through sp"),
    "ABI-LR": ("error", "ABI", "a function that calls (jl) does not save / restore lr"),
    "IMM-RANGE": ("error", "td + assembler", "an immediate outside its .td operand type (range / step); the assembler aborts on these too"),
    "STAGE-SIDE-EFFECT": ("warning", "measured r12", "a store / push / flush / release in a non-last stage of a loop without an epilogue (msched style): the drain's garbage iterations write past the buffer"),
    "ACQ-EQUAL": ("warning", "measured r9 (lock semantics)", "`acq #id, rV` with rV >= 0 is acquire-EQUAL (a 'no-op' acq of 0 waits for 0: hang)"),
    "PUSH-P2": ("error", "measured r12", "bfp output-stream pushes / flushes must use [p2, sf, r26]"),
    "ASSEMBLE": ("error", "assembler", "the .s does not assemble (out-of-range immediates, a push not on p2, ...): the other checks did not run"),
    "FIFO-POP9": ("error", "measured r16 (popprobe 6 / 7)", "more than 8 dense 576-bit pops from one FIFO (lf0 / lf1) without a fill on it: a pop consumes 72 B and refills 64, so pops 9-12 read zeros / half exponents; put a fill at the same position after every 8 pops (restarts of <= 8 pops are fine)"),
    "RAW-BYTES": ("error", "measured (hung the core twice)", "raw .byte / .word data in a .text section of a .s"),
    "LOCK-ACQ-GAP": ("error", "measured r18 (ROT g2f raced in situ; mm_ks / g2a with 6 are exact)", "a FIFO fill that opens freshly "
                     "loaded memory (its pointer came from a scalar lda since the stream's last fill) issues < 6 bundles after a lock "
                     "acquire: the fill reads the slot before the acquire has stalled the core (one-core benches with pre-released "
                     "locks cannot see it)"),
    "LOCK-REL-GAP": ("warning", "unverified threshold (g2a deploys 8; 6 = the pop's memory stage 5 + 1)", "a lock release < 6 "
                     "bundles after the last pop / fill of a stream opened under a lock: the DMA may overwrite the slot before the "
                     "pop has read it"),
    "IMM-SRC": ("error", "measured r18 (V2T: `add r1, r1, #64` assembled SILENTLY as #-64, a hang)", "an immediate in the .s SOURCE "
                "outside its .td operand type: the assembler may wrap it without an error (add takes c7s = [-64, 63]), so the "
                "disassembly looks legal; checked on the source text"),
}
LOCK_ACQ_GAP, LOCK_REL_GAP = 6, 6
NOT_STATIC = [
    ("bank conflicts (two accesses to one bank in a bundle stall)", "needs addresses: one-core bench with operands at chosen offsets (r8/mmb.py)"),
    ("a drain's garbage store landing on data a later stage still reads", "needs the buffer map: msched check / guard-zone benches"),
    ("FIFO pop restarts after a non-64-multiple byte count", "needs the stream byte position: generator-level check (gen_attv)"),
    ("lock order / counts across cores, ring slot accounting", "dynamic: one dispatch with a short timeout"),
    ("output-ring store cost in situ (~3.5 cycles per vector)", "dynamic: EW PROF stamps"),
    ("kernel-only loops' negative-iteration accesses", "needs trip counts and buffer bounds: msched kernel_only / neg_steps"),
    ("stack / program-memory budgets", "r11/stackaudit.py, llvm-size (already in bb.sh)"),
    ("block-vector stream alignment (64 B) and no calls while an OutS is live", "C++ level: read the call graph of the stream's live range"),
]

# ---------------------------------------------------------------------------------------------------------------------
# Registers

REG_RE = re.compile(r"\b(bm[lh][lh]\d+|cm[lh]\d+|dm\d+|ex\d+|wl\d+|wh\d+|x\d+|y\d+|e[lh]\d+|e\d+|q\d+|r\d+|p\d+|m\d+|dj\d+|dc\d+|dn\d+|"
                    r"d\d+|s\d+|lf[lh]?\d|lfe|sf[lh]?|lc|ls|le|lr|sp|cr[a-z0-9]+|sr[a-z0-9]+)\b")


def atoms(reg):
    """A register name -> the storage atoms it covers (x_n = wl_n + wh_n; ex_n = x_n + e_n; y_k = x_2k + x_2k+1;
    dm_n = 4 quarters, cml / cmh halves, bm.. quarters; d_n = the 2D modifier m_n + dn_n + dj_n + dc_n)."""
    m = re.fullmatch(r"(bm[lh][lh]|cm[lh]|dm|ex|wl|wh|x|y|e[lh]|e|d)(\d+)", reg)
    if not m:
        return (reg,)
    k, n = m.group(1), int(m.group(2))
    if k in ("x",):
        return (f"x{n}.l", f"x{n}.h")
    if k == "wl":
        return (f"x{n}.l",)
    if k == "wh":
        return (f"x{n}.h",)
    if k == "ex":
        return (f"x{n}.l", f"x{n}.h", f"e{n}")
    if k in ("e", "el", "eh"):
        return (f"e{n}",)
    if k == "y":
        return (f"x{2 * n}.l", f"x{2 * n}.h", f"x{2 * n + 1}.l", f"x{2 * n + 1}.h")
    if k == "dm":
        return tuple(f"a{n}.{i}" for i in range(4))
    if k == "cml":
        return (f"a{n}.0", f"a{n}.1")
    if k == "cmh":
        return (f"a{n}.2", f"a{n}.3")
    if k == "d":
        return (f"m{n}", f"dn{n}", f"dj{n}", f"dc{n}")
    return (f"a{n}.{['bmll', 'bmlh', 'bmhl', 'bmhh'].index(k)}",)


def is_scalar(atom):
    return bool(re.fullmatch(r"(r|p|m|dn|dj|dc|s)\d+|lc|ls|le|lr|sp|cr\w+|sr\w+", atom))


def xnum(reg):
    m = re.fullmatch(r"(ex|x|wl|wh)(\d+)", reg)
    return (m.group(1), int(m.group(2))) if m else (None, None)


# ---------------------------------------------------------------------------------------------------------------------
# The machine model: llvm-aie's .td (latencies / itineraries from aie2p_isa_timing.json, operand types from the InstrInfo .td)


def find_file(rel):
    here = Path(os.path.abspath(__file__)).parent
    roots = [here.parent.parent, Path(__file__).resolve().parent.parent.parent, Path("/work"),
             Path.home() / "git/app/.scratch", Path("~/git/app/.scratch")]
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


def slot_of_cls(cls):
    return cls.split("_")[2] if cls and cls.count("_") >= 2 else "?"


def parse_ports(stages):
    """[(stage, resource)] from an itinerary string (same reading as r11/msched.ports)."""
    out, t = [], 0
    for kind, arg in re.findall(r"(SimpleCycle|PrefixCycle|EmptyCycles|InstrStage|AvoidPartWordStore|IsPartWordStore)(?:<([^>]*)>)?", stages or ""):
        if kind == "SimpleCycle":
            out.append((t, arg))
            t += 1
        elif kind == "PrefixCycle":
            out.append((t, arg))
        elif kind == "EmptyCycles":
            t += int(arg)
        elif kind == "InstrStage":
            n, r = re.match(r"\s*(\d+)\s*,\s*\[(\w+)\]", arg).groups()
            out += [(t + k, r) for k in range(int(n))]
            t += int(n)
    return out


# ports that the hardware arbitrates itself (multi-cycle lock / scheduling resources): not structural hazards for us
SOFT_PORTS = {"LCKREQ", "SCD", "SCD_CTL", "RSRC_EXECUTION_TRACE"}
# operand-class families: td register-operand class -> register-name regex (for picking among same-text defs)
CLASS_RE = [
    (r"OP_mX|OP_mShflXDst|OP_mQX", r"x\d+"), (r"OP_mW", r"w[lh]\d+"), (r"OP_mEX|OP_mQEX", r"ex\d+"), (r"OP_mE[hl]", r"e\d+"),
    (r"OP_mBM|OP_mShflBMDst", r"bm[lh][lh]\d+"), (r"OP_mCM", r"cm[lh]\d+"), (r"^eDM$", r"dm\d+"), (r"^eY$", r"y\d+"),
    (r"^eR$|mR\d", r"r\d+"), (r"^eP$|mPfs", r"p\d+"), (r"^eM$", r"m\d+"), (r"^eD$", r"d\d+"), (r"^eDN$", r"dn\d+"),
    (r"^eDJ$", r"dj\d+"), (r"^eDC$", r"dc\d+"), (r"^eS$", r"s\d+"), (r"OP_mQQ", r"q\d+"),
]


class Model:
    def __init__(self, lat_path=None, td_path=None):
        lat_path = lat_path or find_file("common/tools/aie2p_isa_timing.json")
        td_path = td_path or find_file("moondream/npu/p10/td/AIE2PGenInstrInfo.td")
        if not lat_path:
            sys.exit("asmlint: aie2p_isa_timing.json not found (common/tools/); pass --lat")
        J = json.load(open(lat_path))
        self.defs, self.itin = J["defs"], J["itin"]
        self.optypes = {}  # def -> {operand: td type}
        if td_path:
            t = open(td_path).read()
            for name, outs, ins in re.findall(r"^def (\w+)\s*:\s*\w+\s*<\(outs([^)]*)\),\s*\(ins([^)]*)\)", t, re.M):
                d = {}
                for x in (outs + "," + ins).split(","):
                    if ":" in x:
                        ty, nm = x.strip().split(":", 1)
                        d[nm.strip().lstrip("$")] = ty.strip()
                self.optypes[name] = d
        self.by_mn = defaultdict(list)
        for name, d in self.defs.items():
            self.by_mn[d["mn"]].append((name, d, self._template(d["args"])))
        self.cache, self.unknown, self.ambiguous = {}, defaultdict(int), defaultdict(int)

    @staticmethod
    def _template(args):
        """'$dst, [$ptr, #$imm]' -> (regex, [operand names]); whitespace-free matching"""
        names, out, i = [], "", 0
        a = re.sub(r"\s+", "", args)
        while i < len(a):
            m = re.match(r"\$\{?(\w+)\}?", a[i:])
            if m:
                nm = m.group(1)
                names.append(nm)
                imm = out.endswith("\\#")
                out += f"(?P<g{len(names) - 1}>[^,\\[\\]]+?)" if imm else f"(?P<g{len(names) - 1}>[^,\\[\\]#][^,\\[\\]]*?)"
                i += m.end()
            else:
                out += re.escape(a[i])
                i += 1
        return re.compile(out + "$"), names

    def match(self, mn, ops):
        key = (mn, ops)
        if key in self.cache:
            return self.cache[key]
        o = re.sub(r"\s+", "", ops)
        cands = []
        for name, d, (rx, names) in self.by_mn.get(mn, []):
            m = rx.match(o)
            if not m:
                continue
            bind = {}
            for k, nm in enumerate(names):
                bind[nm] = m.group(f"g{k}")
            score = 0
            types = self.optypes.get(name, {})
            ok = True
            for nm, val in bind.items():
                ty = types.get(nm)
                if not ty or not REG_RE.fullmatch(val):
                    continue
                for crx, rrx in CLASS_RE:
                    if re.search(crx, ty):
                        if re.fullmatch(rrx, val):
                            score += 2
                        else:
                            ok = False
                        break
            if not ok:
                continue
            score += 1 if d["itin"] else 0
            cands.append((score, name, d, bind))
        if not cands:
            self.unknown[mn] += 1
            self.cache[key] = None
            return None
        cands.sort(key=lambda c: -c[0])
        if len(cands) > 1 and cands[0][0] == cands[1][0]:
            a, b = cands[0][2], cands[1][2]
            if a["lat"] != b["lat"] or a["itin"] != b["itin"]:
                self.ambiguous[mn] += 1
        self.cache[key] = cands[0][1:]
        return self.cache[key]


# ---------------------------------------------------------------------------------------------------------------------
# Program representation


class Instr:
    __slots__ = ("mn", "ops", "text", "slot", "reads", "writes", "ports", "defname", "imms")

    def __init__(self, text):
        self.text = text.strip()
        p = self.text.split(None, 1)
        self.mn, self.ops = p[0], (p[1].strip() if len(p) > 1 else "")
        self.slot, self.reads, self.writes, self.ports, self.defname, self.imms = "?", [], [], [], None, {}

    def is_nop(self):
        return self.mn.startswith("nop")

    def is_branch(self):
        return self.mn in ("j", "jz", "jnz", "jl", "ret", "jnzd") or re.fullmatch(r"j[a-z]*", self.mn) is not None

    def is_store(self):
        return self.mn.startswith(("vst", "st")) and not self.mn.startswith("stream")

    def is_side_effect(self):
        return self.is_store() or self.mn.startswith(("rel", "vst.push", "vst.flush"))


class Bundle:
    __slots__ = ("addr", "size", "instrs", "func", "src_line")

    def __init__(self, addr, instrs, func):
        self.addr, self.instrs, self.func, self.size, self.src_line = addr, instrs, func, 0, None

    def real(self):
        return [i for i in self.instrs if not i.is_nop()]


class Func:
    def __init__(self, name, start, end):
        self.name, self.start, self.end, self.bundles, self.generated = name, start, end, [], False
        self.zol_idx = set()  # bundle indices inside zero-overhead loop bodies


def annotate(model, ins):
    """slot, reads [(reg, stage)], writes [(reg, latency)], ports [(stage, port)], immediates {operand: (value, type)}"""
    r = model.match(ins.mn, ins.ops)
    regs = REG_RE.findall(ins.ops)
    if r is None:
        # unknown op: first register written (unless a store / control op), the rest read, latency 1
        ctrl = ins.is_store() or ins.mn.startswith(("acq", "rel", "j", "ret", "event", "done", "sched"))
        ins.writes = [] if ctrl or not regs else [(regs[0], 1)]
        ins.reads = [(g, 1) for g in (regs if ctrl else regs[1:])]
        ins.slot = "?"
        return
    name, d, bind = r
    ins.defname, ins.slot = name, slot_of_cls(d["cls"])
    lat = d["lat"]
    types = model.optypes.get(name, {})
    load = ins.slot in ("lda", "ldb") and ("ld" in ins.mn or "pop" in ins.mn)
    for nm, val in bind.items():
        ty = types.get(nm, "")
        if re.fullmatch(r"c\d+[sun](_step\d+)?", ty):
            try:
                ins.imms[nm] = (int(val.lstrip("#"), 0), ty)
            except ValueError:
                pass
    for nm in d["outs"]:
        src = nm[:-4] if nm.endswith("_out") else nm
        val = bind.get(nm) or bind.get(src)
        if val is None and nm.startswith("dc") and "mod" in bind and re.fullmatch(r"d\d+", bind["mod"]):
            val = "dc" + bind["mod"][1:]  # 2D / 3D post-modify updates the modifier's counter
        if val and REG_RE.fullmatch(val):
            w = lat.get(nm, lat.get(src, 7 if (load and nm == "dst") else 1))
            ins.writes.append((val, w))
    for nm in d["ins"]:
        val = bind.get(nm)
        if val and REG_RE.fullmatch(val):
            ins.reads.append((val, lat.get(nm, 1)))
    # registers in the text not bound by the template (literal r26, sp, lf0, ...): reads at stage 1
    bound = set(bind.values())
    for g in regs:
        if g not in bound and g != "sp":
            ins.reads.append((g, 1))
    if d["itin"]:
        ins.ports = [(s, p) for s, p in parse_ports(model.itin.get(d["itin"], {}).get("stages", "")) if p not in SOFT_PORTS]
    elif ins.mn in ("lda", "lda.s16", "lda.u16", "lda.s8", "lda.u8") and "[" in ins.ops:
        # round 13 (measured): a scalar load uses the A data port at stage 1 like LDA_dmv_lda_q_spill
        ins.ports = [(1, "DM_ADA")]
    if ins.mn.startswith("vmov") and re.match(r"\s*x\d+,\s*bm", ins.ops):
        ins.ports = [(0, "DM_RM_L0_PORT")]  # round 15: vmov x <- bm has its own itinerary (latency 2)
        ins.writes = [(w, 2) for w, _ in ins.writes]


def parse_dis(text, model, syms=None):
    """llvm-objdump -d -z text -> {name: Func}. syms: {name: (addr, size)} from the symbol table (size 0 = label)."""
    heads, rows = [], []
    for ln in text.splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:", ln)
        if m:
            heads.append((int(m.group(1), 16), m.group(2)))
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", ln)
        if m:
            body = re.sub(r"^(?:[0-9a-f]{2} )+\s*", "", m.group(2))
            rows.append((int(m.group(1), 16), body))
    if not rows:
        return {}, {}
    names = {n for _, n in heads}

    def is_label(n):
        if syms is not None and n in syms:
            return syms[n][1] == 0 and not n.startswith("core_")
        return n.startswith(".") or (re.match(r"(.+)_(start|last)$", n) and re.match(r"(.+)_(start|last)$", n).group(1) in names)

    funcs, labels = [], defaultdict(list)
    for a, n in heads:
        (labels[a].append(n) if is_label(n) else funcs.append((a, n)))
    # aliases at one address (main / core_0_5 / _ctors_*): keep the last header printed
    fa = {}
    for a, n in funcs:
        fa[a] = n
    starts = sorted(fa)
    end_all = rows[-1][0] + 16
    out = {}
    for k, a in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else end_all
        if syms and fa[a] in syms and syms[fa[a]][1]:
            e = min(e, a + syms[fa[a]][1]) if k + 1 >= len(starts) else e
        out[fa[a]] = Func(fa[a], a, e)
    fl = sorted(out.values(), key=lambda f: f.start)
    fi = 0
    for a, body in rows:
        while fi + 1 < len(fl) and a >= fl[fi + 1].start:
            fi += 1
        if not fl or a < fl[0].start:
            continue
        f = fl[fi]
        ins = [Instr(t) for t in body.split(";") if t.strip()]
        for i in ins:
            annotate(model, i)
        f.bundles.append(Bundle(a, ins, f.name))
    for f in fl:
        for b, nb in zip(f.bundles, f.bundles[1:]):
            b.size = nb.addr - b.addr
        if f.bundles:
            f.bundles[-1].size = max(2, f.end - f.bundles[-1].addr) if f.end - f.bundles[-1].addr <= 16 else 16
        # hand / generated asm keeps its local labels (<f>_start, zero2_a ...) in the linked elf; Peano's .LBB labels are gone
        f.generated = any(f.start <= a < f.end and any(n.startswith(f.name + "_") for n in v) for a, v in labels.items())
    return out, labels


# ---------------------------------------------------------------------------------------------------------------------
# Toolchain


class Tools:
    def __init__(self, peano=None, work=None):
        self.bin = peano or os.environ.get("PEANO_BIN") or PEANO_DEFAULT
        self.work = work

    def ok(self):
        return Path(self.bin, "llvm-objdump").exists()

    def run(self, *cmd):
        return subprocess.run(list(cmd), capture_output=True, text=True)

    def symbols(self, path):
        r = self.run(f"{self.bin}/llvm-nm", "-n", "-S", str(path))
        syms = {}
        for ln in r.stdout.splitlines():
            p = ln.split()
            if len(p) == 4:
                syms[p[3]] = (int(p[0], 16), int(p[1], 16))
            elif len(p) == 3:
                syms[p[2]] = (int(p[0], 16), 0)
        return syms

    def disassemble(self, path):
        r = self.run(f"{self.bin}/llvm-objdump", "-d", "-z", "--no-show-raw-insn", str(path))
        if r.returncode:
            raise RuntimeError(r.stderr.strip())
        return r.stdout

    def assemble_link(self, s_path):
        """a .s -> a linked elf in a PRIVATE temp dir (parallel runs never share files); returns (elf, tmpdir)"""
        base = Path(self.work) if self.work else Path(s_path).resolve().parent
        tmp = Path(tempfile.mkdtemp(prefix=".asmlint_", dir=base))
        o, e = tmp / "k.o", tmp / "k.elf"
        r = self.run(f"{self.bin}/clang", "--target=aie2p-none-unknown-elf", "-c", str(s_path), "-o", str(o))
        if r.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError("assembler: " + (r.stderr.strip().splitlines() or ["?"])[0])
        r = self.run(f"{self.bin}/ld.lld", str(o), "-o", str(e), "--unresolved-symbols=ignore-all", "--noinhibit-exec", "-e", "0")
        if not e.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError("ld.lld: " + r.stderr.strip())
        return e, tmp


# ---------------------------------------------------------------------------------------------------------------------
# The checks


# Rules that must hold for Peano's output too (a finding in Peano code there means the RULE is wrong). Every other rule is
# about intent (which value a read means), which only the generator / hand author knows: in Peano functions those findings
# are kept as "calib" (calibration of the model against the compiler that knows the true latencies), never as errors.
UNIVERSAL = {"ZOL-END-ALIGN", "ZOL-START-ALIGN", "ZOL-MIN-BODY", "ZOL-SETUP", "ZOL-SETUP-MARGIN", "ZOL-FULL-EDGE", "PORT",
             "ABI-CALLEE-SAVED", "ABI-LR", "IMM-RANGE", "PUSH-P2", "RAW-BYTES"}


class Report:
    def __init__(self, generated=None):
        self.items, self.seen, self.generated = [], set(), generated

    def add(self, rule, func, addr, msg, key=None):
        k = (rule, func, addr, key or msg)
        if k in self.seen:
            return
        self.seen.add(k)
        sev = RULES[rule][0]
        if self.generated is not None and func not in self.generated and rule not in UNIVERSAL:
            sev = "calib"
        self.items.append(dict(rule=rule, severity=sev, func=func, addr=addr, msg=msg))


def zol_loops(f):
    """[(ls_addr, le_addr, {"lc" / "ls" / "le": index of the last write before the loop start})], one entry per loop,
    decided when the walk reaches the loop start (setup writes may come in any order before it)"""
    loops, last = [], {}
    for k, b in enumerate(f.bundles):
        if last.get("ls_val") is not None and b.addr == last["ls_val"] and last.get("le_val") is not None:
            loops.append((last["ls_val"], last["le_val"], {r: last[r] for r in ("lc", "ls", "le") if r in last}))
            last = {"lc": last["lc"]} if "lc" in last else {}
        for i in b.instrs:
            m = re.match(r"\s*(lc|ls|le)\s*,\s*(.*)$", i.ops)
            if m and not i.is_store():
                last[m.group(1)] = k
                if m.group(1) in ("ls", "le"):
                    v = re.search(r"#(-?0x[0-9a-f]+|-?\d+)", m.group(2))
                    last[m.group(1) + "_val"] = int(v.group(1), 0) if v else None
    return loops


def index_of(f, addr):
    for k, b in enumerate(f.bundles):
        if b.addr == addr:
            return k
    return None


def check_zol(f, rep, stats):
    for ls, le, setup in zol_loops(f):
        i0, i1 = index_of(f, ls), index_of(f, le)
        if i0 is None or i1 is None or i1 < i0:
            rep.add("ZOL-NO-LC", f.name, ls, f"loop bounds 0x{ls:x} / 0x{le:x} not inside this function")
            continue
        body = f.bundles[i0:i1 + 1]
        nb, by = len(body), body[-1].addr + body[-1].size - body[0].addr
        if ls % 16:
            rep.add("ZOL-START-ALIGN", f.name, ls, f"loop start 0x{ls:x} not 16 B aligned")
        if le % 16:
            rep.add("ZOL-END-ALIGN", f.name, le, f"last body bundle 0x{le:x} not 16 B aligned (the body runs ~once)")
        if nb < 3:
            rep.add("ZOL-MIN-BODY", f.name, ls, f"body of {nb} bundle(s) ({by} B): runs once; need >= 3")
        if body[-1].size != 16:  # Peano's own loops start with 10 / 12 B bundles, but always end with a full one
            rep.add("ZOL-FULL-EDGE", f.name, le, f"last body bundle {body[-1].size} B (every deployed loop ends with a 16 B bundle)")
        if "lc" not in setup:
            rep.add("ZOL-NO-LC", f.name, ls, "no lc write before the loop in this function")
        # setup distances: from the bundle AFTER the last setup write to the loop end, in cycles and bytes
        k = max(setup.get("lc", -1), setup.get("ls", -1), setup.get("le", -1))
        wb = f.bundles[k]
        d_cyc_ls = i0 - k
        d_byt_ls = ls - (wb.addr + wb.size)
        d_cyc_le = i1 - k
        d_byt_le = le - (wb.addr + wb.size)
        lc_k = setup.get("lc", k)
        st = dict(func=f.name, ls=ls, le=le, body_bundles=nb, body_bytes=by, first=body[0].size, last=body[-1].size,
                  setup_cyc_to_ls=d_cyc_ls, setup_bytes_to_ls=d_byt_ls, setup_cyc_to_le=d_cyc_le, setup_bytes_to_le=d_byt_le,
                  lc_cyc_to_ls=i0 - lc_k, lc_bytes_to_ls=ls - (f.bundles[lc_k].addr + f.bundles[lc_k].size),
                  generated=f.generated)
        stats.append(st)
        # ZOL-SETUP (MACHINE rev 11): the distance that matters is from the last lc / ls / le write to the loop END (le):
        # Peano writes lc 2 bundles before ls (ew_ssm, bias_pcs) when the body is long, so "setup -> ls" is not the rule.
        # r14's zero2 v1 (6 two-byte nops, 3-bundle body) ran once at 9 bundles / 52 B to le; every deployed loop has
        # >= 11 bundles and >= 84 B. Error at or below the bad case on both axes, warning outside the known-safe region.
        if d_byt_le <= SETUP_BAD_BYTES:
            rep.add("ZOL-SETUP", f.name, ls, f"setup -> loop end {d_byt_le} B ({d_cyc_le} bundles): at or below r14's zero2 v1 "
                    f"({SETUP_BAD_BYTES} B), which ran its body once; pad with full-form (16 B) nop bundles")
        elif d_byt_le < SETUP_SAFE_BYTES:
            rep.add("ZOL-SETUP-MARGIN", f.name, ls, f"setup -> loop end {d_byt_le} B ({d_cyc_le} bundles): between the measured bad "
                    f"{SETUP_BAD_BYTES} B and the smallest deployed {SETUP_SAFE_BYTES} B", key="warn")


# The bad and the good zero2 have the SAME bundle count from the lc write to le (10) and differ only in bytes (52 vs 132):
# the limit is in instruction bytes (fetch runs ahead of issue), not cycles. Peano's own loops go down to 7 bundles / 96 B.
SETUP_BAD_BYTES = 52   # r14 zero2 v1 (reconstructed: tapnext/research/r16/hist/zero2_v1.s)
SETUP_SAFE_BYTES = 80  # the smallest over every loop of full12r15m (cpy32: 17 bundles / 80 B)


def streams(f):
    """bundle-index sequences to replay: the function in address order (resets marked None) plus each ZOL replayed 3x
    and each backward branch loop 2x"""
    n = len(f.bundles)
    targets = set()
    addr_ix = {b.addr: k for k, b in enumerate(f.bundles)}
    for b in f.bundles:
        for i in b.instrs:
            if i.is_branch() and i.mn != "jl":
                m = re.search(r"#(0x[0-9a-f]+)", i.ops)
                if m and int(m.group(1), 16) in addr_ix:
                    targets.add(addr_ix[int(m.group(1), 16)])
    lin, k, pending = [], 0, None
    while k < n:
        if k in targets:
            lin.append(None)
        lin.append(k)
        if any(i.mn == "jl" for i in f.bundles[k].instrs):
            pending = DELAY_SLOTS
        elif pending is not None:
            pending -= 1
            if pending == 0:
                lin.append(None)  # after the call: everything landed, values unknown
                lin.append("CALL")
                pending = None
        k += 1
    # kernel-only loops (no prologue between the lc write and ls): the first trip reads registers no iteration wrote yet,
    # by design (garbage iterations) -> no LAT-UNLANDED there ("quiet" positions of a sequence)
    quiet_lin = set()
    zl = []
    for ls, le, setup in zol_loops(f):
        i0, i1 = index_of(f, ls), index_of(f, le)
        if i0 is None or i1 is None or i1 < i0:
            continue
        k = setup.get("lc", i0 - 1)
        ko = not any(b.real() for b in f.bundles[k + 1:i0])
        zl.append((i0, i1, ko))
        if ko:
            quiet_lin |= set(range(i0, i1 + 1))
    out = [(lin, {p for p, x in enumerate(lin) if x in quiet_lin})]
    for i0, i1, ko in zl:
        pre_start = max([t for t in targets if t <= i0] + [0])
        pre = list(range(pre_start, i0))
        post_end = min([t for t in targets if t > i1] + [n])
        post = list(range(i1 + 1, min(post_end, i1 + 12)))
        seq = pre + list(range(i0, i1 + 1)) * 3 + post
        out.append((seq, set(range(len(pre), len(pre) + i1 + 1 - i0)) if ko else set()))
    for k, b in enumerate(f.bundles):
        for i in b.instrs:
            if i.mn in ("j", "jnz", "jz"):
                m = re.search(r"#(0x[0-9a-f]+)", i.ops)
                t = addr_ix.get(int(m.group(1), 16)) if m else None
                if t is not None and t < k:
                    body = list(range(t, min(n, k + DELAY_SLOTS + 1)))
                    out.append((body * 2, set()))
    return out


def replay(f, seq, rep, calib, quiet=frozenset()):
    """walk one bundle sequence: register visibility, ports, ex aliasing. Every read sees the write that LANDED last
    (landing order, round 11); `pending` = writes issued before the read that land after it."""
    hist = {}  # atom -> [(land, issue, addr, slot, reg, mn)]
    ports = {}
    cyc = 0
    for pos, k in enumerate(seq):
        if k is None:
            hist, ports = {}, {}
            continue
        if k == "CALL":
            continue
        b = f.bundles[k]
        cyc += 1
        writes = []
        for ins in b.instrs:
            if ins.is_nop():
                continue
            for st, p in ins.ports:
                key = (cyc + st, p)
                if key in ports and ports[key][0] != id(ins):
                    o = ports[key]
                    rep.add("PORT", f.name, b.addr, f"'{ins.text}' and '{o[1]}' (0x{o[2]:x}) both use {p} in one cycle", key=(p, o[2]))
                else:
                    ports[key] = (id(ins), ins.text, b.addr)
            for reg, stage in ins.reads:
                rt = cyc + stage - 1
                for a in atoms(reg):
                    ws = hist.get(a)
                    if not ws:
                        continue
                    byp = ins.slot == "mv"

                    def eff(w):
                        return w[0] - (1 if (byp and w[3] == "mv") else 0)
                    landed = [w for w in ws if eff(w) <= rt]
                    seen = max(landed, key=lambda w: (w[0], w[1])) if landed else None
                    pending = [w for w in ws if eff(w) > rt]
                    newest = ws[-1]
                    if calib is not None and not f.generated and newest in pending:
                        calib.append((newest[5], newest[4][0], ins.mn, reg[0], cyc - newest[1], newest[0] - newest[1], stage))
                    if newest in pending:
                        if seen is None and pos in quiet:
                            pass
                        elif seen is None:
                            rep.add("LAT-UNLANDED", f.name, b.addr, f"'{ins.text}' reads {reg} {cyc - newest[1]} cycle(s) after '{newest[5]}' "
                                    f"(0x{newest[2]:x}, latency {newest[0] - newest[1]}): no write to it has landed yet", key=(reg, newest[2]))
                        else:
                            rule = "LAT-OLD-SCALAR" if is_scalar(a) else "LAT-OLD-VECTOR"
                            rep.add(rule, f.name, b.addr, f"'{ins.text}' reads {reg} {cyc - newest[1]} cycle(s) after '{newest[5]}' (0x{newest[2]:x}, "
                                    f"latency {newest[0] - newest[1]}): it sees the older 0x{seen[2]:x} ('{seen[5]}')", key=(reg, newest[2]))
                    elif byp and seen is not None and seen[3] == "mv" and eff(seen) == rt and seen[0] > rt:
                        older = [w for w in ws if w is not seen and w[1] < seen[1]]
                        if older:
                            rep.add("LAT-MV-BYPASS", f.name, b.addr, f"'{ins.text}' reads {reg} one cycle before '{seen[5]}' (0x{seen[2]:x}) lands: "
                                    f"the mv bypass gives its NEW value (the .td says the old one)", key=(reg, seen[2]))
                    if seen is not None and is_scalar(a) and seen[3] == "alu" and ins.slot == "alu" and cyc - seen[1] == 1 \
                            and k in f.zol_idx:
                        rep.add("SCALAR-PLUS1", f.name, b.addr, f"'{ins.text}' reads {reg} one cycle after '{seen[5]}' (0x{seen[2]:x}) inside a "
                                f"zero-overhead loop body", key=(reg, seen[2]))
                    # ex aliasing (round 13): the value an exN read sees must come from an ex write (a pop), and an x read
                    # must not see a pop's bits
                    kind, n = xnum(reg)
                    if kind is not None and seen is not None and a == f"x{n}.l":
                        sk, _ = xnum(seen[4])
                        if kind == "ex" and sk in ("x", "wl", "wh") and "pop" not in seen[5]:
                            rep.add("EX-ALIAS", f.name, b.addr, f"'{ins.text}' reads {reg}, but the value there is '{seen[5]} {seen[4]}' "
                                    f"(0x{seen[2]:x}): ex{n} aliases x{n}", key=(reg, seen[2]))
                        elif kind in ("x", "wl", "wh") and sk == "ex" and "pop" in seen[5]:
                            rep.add("EX-ALIAS", f.name, b.addr, f"'{ins.text}' reads {reg}, but the value there is the pop into {seen[4]} "
                                    f"(0x{seen[2]:x})", key=(reg, seen[2]))
                    break
            for reg, w in ins.writes:
                writes.append((reg, cyc + w, cyc, b.addr, ins.slot, ins.mn))
        for reg, land, issue, addr, slot, mn in writes:
            for a in atoms(reg):
                hist.setdefault(a, []).append((land, issue, addr, slot, reg, mn))
                if len(hist[a]) > 6:
                    hist[a] = hist[a][-6:]


def check_abi(f, funcs, rep, callers_r0):
    """ABI-R0 (generated functions): r0 read before written AND the function returns a value (callers read r0 after the
    jl, or r0 is written and never read again before ret = live out). Callee-saved registers / lr: every function."""
    seq = [i for b in f.bundles for i in b.instrs if not i.is_nop()]
    first = None
    for i in seq:
        if any(r == "r0" for r, _ in i.reads):
            first = ("read", i)
            break
        if any(r == "r0" for r, _ in i.writes):
            first = ("write", i)
            break
    live_out = False  # r0 written near a ret (incl. its delay slots) and not read again: the return value
    for k, b in enumerate(f.bundles):
        if any(i.mn == "ret" for i in b.instrs):
            last_w = None
            for bb in f.bundles[max(0, k - 12):k + DELAY_SLOTS + 1]:
                for i in bb.real():
                    if any(r == "r0" for r, _ in i.reads):
                        last_w = None
                    if any(r == "r0" for r, _ in i.writes):
                        last_w = i
            live_out = live_out or last_w is not None
    returns = f.name in callers_r0 or live_out
    if f.generated and first and first[0] == "read" and returns:
        why = "callers read r0 after jl" if f.name in callers_r0 else "r0 is written and live at ret"
        addr = next(b.addr for b in f.bundles if first[1] in b.instrs)
        rep.add("ABI-R0", f.name, addr, f"'{first[1].text}' reads r0 as an input, but the function returns a value ({why}): "
                f"with a return value Peano passes the first scalar argument in r1")
    written = {r for i in seq for r, _ in i.writes if r in CALLEE_SAVED}
    saves = {r for i in seq if i.is_store() and "[sp" in i.ops for r in REG_RE.findall(i.ops.split("[")[0])}
    restores = {r for i in seq if i.mn.startswith("lda") and "[sp" in i.ops for r in REG_RE.findall(i.ops.split("[")[0])}
    for r in sorted(written):
        if r not in saves or r not in restores:
            rep.add("ABI-CALLEE-SAVED", f.name, f.start, f"{r} is written but not {'saved' if r not in saves else 'restored'} through sp")
    if any(i.mn == "jl" for i in seq) and ("lr" not in saves or "lr" not in restores) and f.name not in ("__start",):
        rep.add("ABI-LR", f.name, f.start, "calls (jl) without saving / restoring lr")


def check_misc(f, rep):
    regdef = {}
    for b in f.bundles:
        for i in b.instrs:
            if i.is_nop():
                continue
            if i.mn.startswith("acq"):
                m = re.match(r"\s*#?\w+\s*,\s*(r\d+)", i.ops)
                if m and m.group(1) in regdef and regdef[m.group(1)] >= 0:
                    rep.add("ACQ-EQUAL", f.name, b.addr, f"'{i.text}': {m.group(1)} = {regdef[m.group(1)]} >= 0 makes this an acquire-EQUAL")
            if i.mn.startswith(("vst.push", "vst.flush")) and not re.search(r"\[\s*p2\s*,\s*sf", i.ops):
                rep.add("PUSH-P2", f.name, b.addr, f"'{i.text}': output-stream pushes / flushes use [p2, sf, r26]")
            for nm, (v, ty) in i.imms.items():
                m = re.fullmatch(r"c(\d+)([sun])(?:_step(\d+))?", ty)
                bits, kind, step = int(m.group(1)), m.group(2), int(m.group(3) or 1)
                lo, hi = {"s": (-(1 << (bits - 1)), (1 << (bits - 1)) - step), "u": (0, (1 << bits) - step),
                          "n": (-(1 << (bits - 1)), 0)}[kind]
                if v < lo or v > hi or v % step:
                    rep.add("IMM-RANGE", f.name, b.addr, f"'{i.text}': #{v} outside {ty} = [{lo}, {hi}] step {step}")
            for r, _ in i.writes:
                regdef.pop(r, None)
            m = re.match(r"(mova?|movxm)$", i.mn)
            if m:
                mm = re.match(r"\s*(r\d+)\s*,\s*#(-?(?:0x)?[0-9a-f]+)", i.ops)
                if mm:
                    regdef[mm.group(1)] = int(mm.group(2), 0)


def check_stage(f, rep):
    """side effects only in the last stage of an msched-style loop (no epilogue: the drain after le is nops only)"""
    for ls, le, setup in zol_loops(f):
        i0, i1 = index_of(f, ls), index_of(f, le)
        if i0 is None or i1 is None:
            continue
        after = f.bundles[i1 + 1:i1 + 5]
        if len(after) < 4 or any(b.real() and not any(i.mn == "ret" for i in b.real()) for b in after[:3]):
            continue  # an epilogue exists (Peano's pipeliner): not this rule's shape
        k = setup.get("lc", i0 - 1)
        pro = [i for b in f.bundles[k + 1:i0] for i in b.real()]
        if not pro:
            continue  # kernel-only loop: the garbage iterations' stores are by design (stage rule not checkable statically)
        # pointer provenance before the loop: a pointer copied from sp addresses a stack ring (bgelu's x ring: a garbage
        # store there lands in the ring's slack by design)
        stackp = set()
        for b in f.bundles[:i0]:
            for i in b.real():
                for r, _ in i.writes:
                    if re.fullmatch(r"p\d+", r):
                        src = REG_RE.findall(i.ops)[1:]
                        if "padd" in i.mn:
                            continue
                        if any(x == "sp" or x in stackp for x in src):
                            stackp.add(r)
                        else:
                            stackp.discard(r)
        for b in f.bundles[i0:i1 + 1]:
            for i in b.real():
                if not i.is_side_effect():
                    continue
                mem = re.search(r"\[[^\]]*\].*$", i.ops)
                memk = re.sub(r"\s+", "", mem.group(0)) if mem else i.ops
                base = re.search(r"\[\s*(p\d+)", i.ops)
                if base and base.group(1) in stackp:
                    continue
                for p in pro:
                    pm = re.search(r"\[[^\]]*\].*$", p.ops)
                    if p.mn == i.mn and pm and re.sub(r"\s+", "", pm.group(0)) == memk:
                        rep.add("STAGE-SIDE-EFFECT", f.name, b.addr, f"'{i.text}' also runs in the prologue, so it is not in the last stage: "
                                f"the drain's garbage iterations repeat it past the end (fine only into slack: a score / stack ring)")
                        break


POP_RE = re.compile(r"vld[ab]\.pop\.576(?:\.\w+)*$")


def check_fifo(f, rep):
    """FIFO-POP9 (round 16): at most 8 dense 576-bit pops per lf between fills. Walks the replay streams (each ZOL body 3x
    after its prologue); the count restarts at a fill of that lf and at unknown points (branch targets, calls)."""
    seen = set()
    for seq, _ in streams(f):
        cnt = {}
        for x in seq:
            if x is None or x == "CALL":
                cnt = {}
                continue
            b = f.bundles[x]
            for i in b.instrs:
                m = re.search(r"\b(lf[01])\b", i.ops)
                if not m:
                    continue
                lf = m.group(1)
                if ".fill" in i.mn:
                    cnt[lf] = 0
                elif POP_RE.match(i.mn):
                    cnt[lf] = cnt.get(lf, 0) + 1
                    if cnt[lf] == 9 and (b.addr, lf) not in seen:
                        seen.add((b.addr, lf))
                        rep.add("FIFO-POP9", f.name, b.addr, f"'{i.text}' is the 9th 576-bit pop from {lf} since its last fill "
                                f"(or since an unknown point): add a fill at the same position every 8 pops")


def check_lock(f, rep):
    """LOCK-ACQ-GAP / LOCK-REL-GAP (round 18). A fill `v*.fill* [pN, lfK, ..]` whose pointer register was written by a
    scalar load (`lda pN, ..`: the next slot's address from a record) since lfK's previous fill opens freshly acquired memory:
    it must issue >= LOCK_ACQ_GAP bundles after the most recent acquire. A release must come >= LOCK_REL_GAP bundles after
    the last pop / fill of a stream opened that way."""
    seen = set()
    for seq, _ in streams(f):
        cyc, last_acq, loaded, guarded, last_use, moved = 0, None, set(), set(), {}, set()
        for x in seq:
            if x is None or x == "CALL":
                last_acq, loaded, guarded, last_use, moved = None, set(), set(), {}, set()
                continue
            b = f.bundles[x]
            cyc += 1
            for i in b.instrs:
                if i.is_nop():
                    continue
                m = re.search(r"\[\s*(p\d+)\s*,\s*(lf[01])\b", i.ops)
                if i.mn.startswith("acq"):
                    last_acq = (cyc, b.addr, i.text)
                elif i.mn.startswith("rel"):
                    for lf in guarded:
                        if lf in last_use and cyc - last_use[lf][0] < LOCK_REL_GAP and (b.addr, "rel") not in seen:
                            seen.add((b.addr, "rel"))
                            rep.add("LOCK-REL-GAP", f.name, b.addr, f"'{i.text}' {cyc - last_use[lf][0]} bundle(s) after "
                                    f"'{last_use[lf][2]}' (0x{last_use[lf][1]:x}), the last access of the {lf} stream opened under "
                                    f"the lock (deployed ROT g2a: 8)")
                elif m and ".fill" in i.mn:
                    ptr, lf = m.groups()
                    if ptr in moved:  # pass 17 (E7''): a restart from a register copy (mov p0, p6) opens an unguarded stream
                        moved.discard(ptr)
                        guarded.discard(lf)
                        last_use.pop(lf, None)
                    elif ptr in loaded:
                        loaded.discard(ptr)
                        guarded.add(lf)
                        last_use[lf] = (cyc, b.addr, i.text)
                        if last_acq is not None and cyc - last_acq[0] < LOCK_ACQ_GAP and (b.addr, "acq") not in seen:
                            seen.add((b.addr, "acq"))
                            rep.add("LOCK-ACQ-GAP", f.name, b.addr, f"'{i.text}' opens a freshly loaded {ptr} "
                                    f"{cyc - last_acq[0]} bundle(s) after '{last_acq[2]}' (0x{last_acq[1]:x}): need >= {LOCK_ACQ_GAP}")
                    elif lf in guarded:
                        last_use[lf] = (cyc, b.addr, i.text)
                elif m and POP_RE.match(i.mn) and m.group(2) in guarded:
                    last_use[m.group(2)] = (cyc, b.addr, i.text)
                for r, _ in i.writes:
                    if re.fullmatch(r"p\d+", r):
                        if i.mn == "lda":
                            loaded.add(r)
                            moved.discard(r)
                        elif i.mn in ("mov", "movs"):
                            moved.add(r)
                            loaded.discard(r)


def check_source(path, rep, model=None):
    """text-level rules on a .s"""
    sec = ".text"
    for n, ln in enumerate(open(path).read().splitlines(), 1):
        s = ln.split("//")[0].strip()
        if s.startswith((".section", ".text", ".data", ".bss", ".rodata")):
            sec = s
        if re.match(r"\.(byte|2byte|4byte|word|long|short|quad)\b", s) and ".text" in sec:
            rep.add("RAW-BYTES", Path(path).name, n, f"line {n}: '{s}' in {sec.split(',')[0]}")
        for slot in s.split(";"):
            if re.match(r"\s*vst\.(push|flush)", slot) and not re.search(r"\[\s*p2\s*,\s*sf", slot):
                rep.add("PUSH-P2", Path(path).name, n, f"line {n}: '{slot.strip()}': output-stream pushes / flushes use [p2, sf, r26]")
            # IMM-SRC (round 18): immediates as WRITTEN, against the .td operand types (the assembler may wrap silently)
            txt = slot.strip()
            if model is None or not txt or txt.startswith((".", "#")) or txt.endswith(":") or not re.search(r"#-?\d", txt):
                continue
            ins = Instr(txt)
            annotate(model, ins)
            for nm, (v, ty) in ins.imms.items():
                mm = re.fullmatch(r"c(\d+)([sun])(?:_step(\d+))?", ty)
                bits, kind, step = int(mm.group(1)), mm.group(2), int(mm.group(3) or 1)
                lo, hi = {"s": (-(1 << (bits - 1)), (1 << (bits - 1)) - step), "u": (0, (1 << bits) - step),
                          "n": (-(1 << (bits - 1)), 0)}[kind]
                if v < lo or v > hi or v % step:
                    rep.add("IMM-SRC", Path(path).name, n, f"line {n}: '{txt}': #{v} outside {ty} = [{lo}, {hi}] step {step} "
                            f"(the assembler may wrap it silently)")


# ---------------------------------------------------------------------------------------------------------------------


def lint(funcs, labels, rep, only=None, asm_only=False, zstats=None, calib=None):
    zstats = [] if zstats is None else zstats
    calib = [] if calib is None else calib
    rep.generated = {f.name for f in funcs.values() if f.generated}
    # call sites that read r0 before writing it -> the callee returns a value
    callers_r0 = set()
    byaddr = {f.start: f.name for f in funcs.values()}
    for f in funcs.values():
        bl = f.bundles
        for k, b in enumerate(bl):
            for i in b.instrs:
                if i.mn == "jl":
                    m = re.search(r"#(0x[0-9a-f]+)", i.ops)
                    tgt = byaddr.get(int(m.group(1), 16)) if m else None
                    for bb in bl[k + DELAY_SLOTS + 1:k + DELAY_SLOTS + 12]:
                        rd = any(r == "r0" for j in bb.instrs for r, _ in j.reads)
                        wr = any(r == "r0" for j in bb.instrs for r, _ in j.writes)
                        if rd and tgt:
                            callers_r0.add(tgt)
                            break
                        if wr or any(j.mn == "jl" for j in bb.instrs):
                            break
    for f in funcs.values():
        if only and f.name not in only and not any(f.name.startswith(o) for o in only):
            continue
        if asm_only and not f.generated:
            continue
        if not f.bundles:
            continue
        for ls_, le_, _ in zol_loops(f):
            i0, i1 = index_of(f, ls_), index_of(f, le_)
            if i0 is not None and i1 is not None:
                f.zol_idx |= set(range(i0, i1 + 1))
        check_zol(f, rep, zstats)
        for seq, quiet in streams(f):
            replay(f, seq, rep, calib, quiet)
        check_abi(f, funcs, rep, callers_r0)
        check_misc(f, rep)
        check_stage(f, rep)
        check_fifo(f, rep)
        check_lock(f, rep)
    return zstats, calib


def load_input(path, tools, model, rep, keep=False):
    p = Path(path)
    if p.suffix == ".dis" or p.suffix == ".txt":
        funcs, labels = parse_dis(p.read_text(), model)
        return funcs, labels, []
    if not tools.ok():
        sys.exit(f"asmlint: {path} needs Peano ({tools.bin}); run inside npu-iron or pass a .dis")
    tmp = None
    src = None
    if p.suffix == ".s":
        check_source(p, rep, model)
        src = p
        try:
            p, tmp = tools.assemble_link(p)
        except RuntimeError as e:
            rep.add("ASSEMBLE", Path(path).name, 0, f"does not assemble: {e}")
            return {}, {}, []
    try:
        syms = tools.symbols(p)
        funcs, labels = parse_dis(tools.disassemble(p), model, syms)
        if src is not None and not re.search(r"^\s*// %bb\.|\.Lfunc_begin|^\s*\.file\s", src.read_text(), re.M):
            for f in funcs.values():  # a hand-written / generated .s (not `clang -S` output): every function is ours
                f.generated = True
    finally:
        if tmp and not keep:
            shutil.rmtree(tmp, ignore_errors=True)
    return funcs, labels, []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--prj", help="an IRON build dir (<name>.prj): lint core 0_2 (GEMM) and 0_5 (EW)")
    ap.add_argument("--funcs", help="comma list of function names / prefixes")
    ap.add_argument("--asm-only", action="store_true", help="only generated / hand-written asm functions")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rules", action="store_true", help="print the rule table and exit")
    ap.add_argument("--zolstats", action="store_true", help="print every zero-overhead loop's geometry")
    ap.add_argument("--info", action="store_true", help="also print info-level findings")
    ap.add_argument("--calib", action="store_true", help="print the findings in Peano functions + the distances Peano uses "
                    "for reads the .td model calls early (the latency calibration table)")
    ap.add_argument("--lat")
    ap.add_argument("--td")
    ap.add_argument("--peano")
    ap.add_argument("--work", help="where .s inputs are assembled (a private temp dir is made inside)")
    ap.add_argument("--fail-on", default="error", choices=["error", "warning", "never"])
    o = ap.parse_args()
    if o.rules:
        for k, (sev, st, txt) in RULES.items():
            print(f"{k:18s} {sev:7s} [{st}] {txt}")
        print("\nnot checkable statically:")
        for what, how in NOT_STATIC:
            print(f"  - {what}: {how}")
        return 0
    inputs = list(o.inputs)
    if o.prj:
        for c in ("0_2", "0_5"):
            inputs.append(str(Path(o.prj) / f"elfs_main_core_{c}" / f"elfs_main_core_{c}.elf"))
    if not inputs:
        ap.error("no inputs")
    model = Model(o.lat, o.td)
    tools = Tools(o.peano, o.work)
    only = o.funcs.split(",") if o.funcs else None
    allrep, zall, pall = [], [], []
    for path in inputs:
        rep = Report()
        funcs, labels, _ = load_input(path, tools, model, rep)
        zs, p1 = lint(funcs, labels, rep, only, o.asm_only)
        zall += [dict(z, input=path) for z in zs]
        pall += p1
        for it in rep.items:
            it["input"] = path
        allrep += rep.items
    sev_rank = {"error": 0, "warning": 1, "info": 2, "calib": 3}
    shown = [it for it in allrep if (o.info or it["severity"] != "info") and (o.calib or it["severity"] != "calib")]
    if o.json:
        print(json.dumps(dict(findings=shown, counts=count(allrep), zol=zall if o.zolstats else None,
                              unknown=dict(model.unknown), ambiguous=dict(model.ambiguous)), indent=1))
    else:
        if o.zolstats:
            for z in zall:
                print("ZOL {func:28.28s} body {body_bundles:3d} b {body_bytes:4d} B edges {first}/{last} | setup->ls {setup_cyc_to_ls:3d} b "
                      "{setup_bytes_to_ls:4d} B | lc->ls {lc_cyc_to_ls:3d} b {lc_bytes_to_ls:4d} B | setup->le {setup_cyc_to_le:3d} b "
                      "{setup_bytes_to_le:4d} B {gen}".format(**z, gen="asm" if z["generated"] else "peano"))
        for it in sorted(shown, key=lambda t: (sev_rank[t["severity"]], t["input"], t["func"], t["addr"])):
            print(f"{it['severity'].upper():7s} {it['rule']:18s} {Path(it['input']).name}:{it['func']}@0x{it['addr']:x}  {it['msg']}")
        c = count(allrep)
        if o.calib:
            tab = defaultdict(list)
            for pm, pc, cm, cc, d, lat, st in pall:
                tab[(pm, pc, cm, cc, lat, st)].append(d)
            print("calib: Peano reads that the model calls early: producer (dst class) -> consumer (reg class): latency, read stage, "
                  "distances used [count]")
            for (pm, pc, cm, cc, lat, st), ds in sorted(tab.items(), key=lambda kv: -len(kv[1])):
                print(f"  {pm:22s} {pc} -> {cm:22s} {cc}: lat {lat} stage {st}: min {min(ds)} {sorted(set(ds))[:6]} [{len(ds)}]")
        print(f"asmlint: {len(inputs)} input(s): {c.get('error', 0)} error(s), {c.get('warning', 0)} warning(s), {c.get('info', 0)} info, "
              f"{c.get('calib', 0)} calib (Peano functions)")
        if model.unknown:
            print("  unmatched mnemonics (latency 1 / no ports assumed):", dict(sorted(model.unknown.items(), key=lambda kv: -kv[1])[:12]))
    c = count(allrep)
    if o.fail_on == "never":
        return 0
    if c.get("error") or (o.fail_on == "warning" and c.get("warning")):
        return 1
    return 0


def count(items):
    c = defaultdict(int)
    for it in items:
        c[it["severity"]] += 1
    return dict(c)


if __name__ == "__main__":
    sys.exit(main())
