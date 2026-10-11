"""Latency checker for hand-written / emitted AIE2P assembly.

The core has no interlocks: an operand read at stage r of an instruction issued in cycle c sees the register
as it is in cycle c + r, and a producer issued in cycle c' writes its result in cycle c' + w. The checker walks
the bundles of one function (one bundle = one cycle, delay slots are explicit) and flags every read with
c + r < c' + w, using the per-operand latencies from llvm-aie's AIE2PGenSchedule.td (common/tools/aie2p_isa_timing.json from
common/tools/gen_isa_timing.py) and this register aliasing model:
  x_n (512) = wl_n + wh_n; ex_n = x_n + e_n; y_k = x_{2k} + x_{2k+1}
  dm_n (2048) = cml_n + cmh_n; cml_n = bmll_n + bmlh_n; cmh_n = bmhl_n + bmhh_n
Usage: asmcheck.py file.s function [--loop LABEL] [--lat aie2p_isa_timing.json] [-v]
  --loop LABEL: also run the block starting at LABEL twice in a row (loop-carried hazards).
"""
import json, re, sys
from pathlib import Path

REG = re.compile(r"\b(bm[lh][lh]\d+|cm[lh]\d+|dm\d+|ex\d+|wl\d+|wh\d+|x\d+|y\d+|e\d+|q\d+|r\d+|p\d+|m\d+|dj\d+|dc\d+|dn\d+|ds\d+|lf[lh]\d|lfe|lf\d|sf[lh]|sf|s\d|lc|ls|le|lr|sp|cr[a-z0-9]+|sr[a-z0-9]+)\b")


def atoms(reg):
    """A register name -> the set of storage atoms it covers."""
    m = re.fullmatch(r"(bm[lh][lh]|cm[lh]|dm|ex|wl|wh|x|y|e)(\d+)", reg)
    if not m:
        return {reg}
    kind, n = m.group(1), int(m.group(2))
    if kind == "x" or kind in ("wl", "wh"):
        return {f"x{n}"}
    if kind == "ex":
        return {f"x{n}", f"e{n}"}
    if kind == "e":
        return {f"e{n}"}
    if kind == "y":
        return {f"x{2 * n}", f"x{2 * n + 1}"}
    if kind == "dm":
        return {f"a{n}_{i}" for i in range(4)}
    if kind == "cml":
        return {f"a{n}_0", f"a{n}_1"}
    if kind == "cmh":
        return {f"a{n}_2", f"a{n}_3"}
    q = {"bmll": 0, "bmlh": 1, "bmhl": 2, "bmhh": 3}[kind]
    return {f"a{n}_{q}"}


def load_table(path):
    J = json.load(open(path))["defs"]
    by_mn = {}
    for name, d in J.items():
        by_mn.setdefault(d["mn"], []).append((name, d))
    return by_mn


def parse_bundles(path, func):
    """The bundles of one function: list of (lineno, [instr strings]); labels kept as ('label', name)."""
    lines = open(path).read().split("\n")
    out, inside = [], False
    for i, raw in enumerate(lines, 1):
        line = raw.split("//")[0].rstrip()
        s = line.strip()
        if not s:
            continue
        if s.endswith(":") and not s.startswith("."):
            inside = s[:-1] == func
            continue
        if not inside:
            continue
        if s.startswith(".") and s.endswith(":"):
            out.append((i, ("label", s[:-1])))
            continue
        if s.startswith("."):
            if s.startswith(".size") or s.startswith(".Lfunc_end"):
                inside = False
            continue
        instrs = [t.strip() for t in s.split(";") if t.strip()]
        out.append((i, instrs))
    return out


def split_instr(text):
    toks = text.split(None, 1)
    return toks[0], (toks[1] if len(toks) > 1 else "")


def match_def(by_mn, mn, ops):
    """Pick the def whose operand template has the same register positions; returns (def, {opname: reg})."""
    regs = REG.findall(ops)
    best = None
    for name, d in by_mn.get(mn, []):
        tregs = re.findall(r"\$\{?(\w+)\}?|\b(r\d+|p\d+|sf|lf\d)\b", d["args"])
        tnames = [a or b for a, b in tregs]
        if len(tnames) != len(regs):
            continue
        binding = {}
        ok = True
        for tn, rg in zip(tnames, regs):
            if tn.startswith("$") or tn in d["ins"] or tn in d["outs"] or tn in d["lat"]:
                binding[tn] = rg
            elif tn != rg:  # literal register in the template
                ok = False
                break
        if ok:
            best = (d, binding)
            break
    return best


LEARN = {}   # (producer mn, consumer mn, operand) -> min observed distance (compiler output = ground truth)
USE_LEARNED = None


def check(path, func, table, loop=None, verbose=False, learn=False):
    bundles = parse_bundles(path, func)
    if not bundles:
        print(f"no bundles for {func}")
        return 1
    seq = [b for b in bundles if b[1] and b[1][0] != "label"] if False else bundles
    if loop:
        idx = [k for k, b in enumerate(bundles) if b[1] and b[1][0] == "label" and b[1][1] == loop]
        if idx:
            seq = bundles + bundles[idx[0]:]  # second pass of the loop body
    hist = {}         # atom -> list of (landing cycle, line, instr, issue cycle, mn) in issue order
    violations, infos, unknown, cycle = 0, 0, {}, 0
    for lineno, instrs in seq:
        if instrs and instrs[0] == "label":
            continue
        cycle += 1
        writes = []
        for ins in instrs:
            mn, ops = split_instr(ins)
            if mn.startswith("nop"):
                continue
            m = match_def(table, mn, ops)
            if not m:
                unknown[mn] = unknown.get(mn, 0) + 1
                continue
            d, binding = m
            lat = d["lat"]
            # reads: the value seen is the latest write that has landed by cycle + rl (no interlocks)
            for tn, rg in binding.items():
                if tn in d["ins"] or (tn not in d["outs"]):
                    rl = lat.get(tn, 1)
                    for a in atoms(rg):
                        if a in hist:
                            landed = [w for w in hist[a] if w[0] <= cycle + rl]
                            latest = hist[a][-1]
                            if learn:
                                continue
                            if not landed:
                                violations += 1
                                print(f"line {lineno} cycle {cycle}: '{ins}' reads {rg} before any write landed ('{latest[2]}' line {latest[1]} lands in cycle {latest[0]}: {latest[0] - cycle - rl} early)")
                            elif landed[-1] is not latest:
                                infos += 1
                                if verbose:
                                    print(f"  info line {lineno} cycle {cycle}: '{ins}' reads {rg} written by line {landed[-1][1]} (line {latest[1]} lands later, cycle {latest[0]})")
            # writes (apply after all reads of this bundle)
            for on in d["outs"]:
                src = on[:-4] if on.endswith("_out") else on
                rg = binding.get(on) or binding.get(src)
                if not rg:
                    continue
                writes.append((rg, cycle + lat.get(on, 1), lineno, ins, cycle, mn))
        for rg, land, l, i, c, m in writes:
            for a in atoms(rg):
                hist.setdefault(a, []).append((land, l, i, c, m))
    print(f"{func}: {cycle} cycles ({len([b for b in bundles if b[1] and b[1][0] != 'label'])} bundles), {violations} latency violation(s), {infos} read(s) of a value with a newer write in flight (-v lists them)")
    if unknown:
        print("  unmatched mnemonics (not checked):", dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:15]))
    return violations


if __name__ == "__main__":
    args = sys.argv[1:]
    lat = str(Path(__file__).resolve().parents[3] / "common/tools/aie2p_isa_timing.json")
    loop = None
    if "--lat" in args:
        i = args.index("--lat"); lat = args[i + 1]; del args[i:i + 2]
    if "--loop" in args:
        i = args.index("--loop"); loop = args[i + 1]; del args[i:i + 2]
    verbose = "-v" in args
    args = [a for a in args if a != "-v"]
    table = load_table(lat)
    if "--learn" in args:  # learn min distances from compiler output: asmcheck.py file.s --learn out.json f1 f2 ...
        i = args.index("--learn"); out = args[i + 1]
        for f in args[i + 2:]:
            check(args[0], f, table, None, False, learn=True)
        json.dump({"|".join(k): v for k, v in LEARN.items()}, open(out, "w"), indent=0)
        print(f"learned {len(LEARN)} producer/consumer pairs -> {out}")
        sys.exit(0)
    if "--learned" in args:
        i = args.index("--learned"); USE_LEARNED = {tuple(k.split("|")): v for k, v in json.load(open(args[i + 1])).items()}; del args[i:i + 2]
    sys.exit(1 if check(args[0], args[1], table, loop, verbose) else 0)
