"""Static schedule statistics of a Peano AIE2P assembly file (clang -S output).

usage: python3 asmstat.py file.s [func ...] [--blocks]

Per function: bundles (one non-empty asm line = one VLIW bundle, delay slots included), nop-only bundles, stack
references (spills / reloads: `[sp`), calls, and every loop:
  - hardware loops (zero-overhead: `.LBBx_y:` ... `.L_LEndN:`): body bundles = bundles per iteration;
  - branch loops (a j/jz/jnz back to an earlier label): body = label .. jump incl. its 5 delay slots; the static
    count of every block on the way (both arms of an if inside), so it is an upper bound per iteration.
`--blocks` also prints each basic block with its bundle count and call targets.
Calls are reported by (demangled-ish) target name, so a loop that calls a helper shows where its time goes.
"""

import re
import sys

LABEL = re.compile(r"^([.\w$]+):")
JUMP = re.compile(r"\b(j|jz|jnz|jnzd)\s+(?:r\d+,\s*)?#?([.\w$]+)")
CALL = re.compile(r"\bjl\s+#?([.\w$]+)")


def short(sym):
    """_ZL8blk_bias..., _ZN3nk8L8gelu32x2E... -> blk_bias, nk8::gelu32x2 (enough of the Itanium mangling for here)."""
    if not sym.startswith("_Z"):
        return sym
    s = sym[2:]
    nested = s.startswith("N")
    s = s[1:] if nested else s
    out = []
    while True:
        if s.startswith("L"):
            s = s[1:]
        m = re.match(r"\d+", s)
        if not m:
            break
        n = int(m.group())
        s = s[len(m.group()):]
        out.append(s[:n])
        s = s[n:]
        if not nested:
            break
    return "::".join(out) or sym


def parse(path):
    funcs = {}
    cur = None
    for raw in open(path):
        line = raw.rstrip("\n")
        code = line.split("//")[0].strip()
        m = LABEL.match(line)
        if m and not line.startswith("\t"):
            lab = m.group(1)
            if not lab.startswith("."):
                cur = funcs.setdefault(lab, [])
            elif cur is not None:
                cur.append(("label", lab))
            continue
        if cur is None:
            continue
        if code.startswith(".") or not code:
            if ".Lfunc_end" in code:
                cur = None
            continue
        cur.append(("bundle", code, line))
    return funcs


def is_nop(code):
    slots = [s.strip() for s in code.split(";") if s.strip()]
    return all(s.startswith("nop") for s in slots)


def analyse(name, items, show_blocks=False):
    labels = {}
    pos = 0
    seq = []  # (bundle index, code) and label positions
    for it in items:
        if it[0] == "label":
            labels[it[1]] = pos
        else:
            seq.append(it[1])
            pos += 1
    nb = len(seq)
    nops = sum(is_nop(c) for c in seq)
    spills = sum("[sp" in c for c in seq)
    calls = [short(m.group(1)) for c in seq for m in CALL.finditer(c)]
    print(f"{name} ({short(name)}): {nb} bundles, {nops} nop-only, {spills} stack refs, {len(calls)} call sites "
          f"{sorted(set(calls))}")
    loops = []
    # hardware loops: the label just before .L_LEnd's matching start; Peano emits `movxm le, #.L_LEndN` and the body
    # starts at the label that follows the loop setup. Take the last label before the bundle that sets ls.
    for lab, p in labels.items():
        if lab.startswith(".L_LEnd"):
            # start = the position of the label referenced by `movxm ls, #<label>`
            start = None
            for i, c in enumerate(seq):
                m = re.search(r"movxm\s+ls,\s*#([.\w$]+)", c)
                if m and i < p:
                    end_m = None
                    for j in range(i, min(i + 40, nb)):
                        e = re.search(r"movxm\s+le,\s*#([.\w$]+)", seq[j])
                        if e:
                            end_m = e.group(1)
                    if end_m == lab:
                        start = labels.get(m.group(1))
            if start is not None:
                loops.append(("hw", start, p + 1, seq[start:p + 1]))  # .L_LEnd labels the last bundle
    for i, c in enumerate(seq):
        m = JUMP.search(c)
        if m and m.group(2) in labels and labels[m.group(2)] <= i:
            s = labels[m.group(2)]
            loops.append(("br", s, i + 6, seq[s:i + 6]))
    for kind, s, e, body in sorted(loops, key=lambda x: (x[1], -x[2])):
        lab = next((l for l, p in labels.items() if p == s and not l.startswith(".L_LEnd")), "?")
        bc = [short(m.group(1)) for c in body for m in CALL.finditer(c)]
        print(f"  loop {kind} {lab:12s} bundles {e - s:5d}  nop-only {sum(is_nop(c) for c in body):4d}  "
              f"stack refs {sum('[sp' in c for c in body):3d}  calls {bc if bc else '-'}")
    if show_blocks:
        inv = sorted((p, l) for l, p in labels.items())
        bounds = [p for p, _ in inv] + [nb]
        for (p, l), q in zip(inv, bounds[1:]):
            body = seq[p:q]
            bc = [short(m.group(1)) for c in body for m in CALL.finditer(c)]
            print(f"    {l:14s} @{p:5d} bundles {q - p:4d} nop-only {sum(is_nop(c) for c in body):3d} calls {bc if bc else '-'}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    funcs = parse(args[0])
    want = args[1:]
    for f, items in funcs.items():
        if want and not any(w == f or w in short(f) for w in want):
            continue
        if not any(it[0] == "bundle" for it in items):
            continue
        analyse(f, items, "--blocks" in sys.argv)


if __name__ == "__main__":
    main()
