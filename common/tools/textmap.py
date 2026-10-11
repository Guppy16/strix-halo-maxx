"""Program-memory map of an AIE2P elf from its `llvm-objdump -d -z --no-show-raw-insn` output (npu-research rev 8).

Per function (symbols not starting with '.'; asm labels such as gup_start are folded into the function they sit in when
--fold is given): bytes, bundles, bytes per bundle, bundles that are all-nop, full-form (16 B) bundles, callers, and the
bytes that would go if every bundle of a function were packed to its minimum (an upper bound for asmfit: we do not know
each bundle's minimum encoding from the listing, so the bound uses the narrowest size seen for the same slot mix).

usage: python3 textmap.py X.dis [--fold gup,pan,bgelu,cpy32,sigk] [--detail FUNC]
"""

import argparse
import re
from collections import Counter, defaultdict


def parse(path):
    txt = open(path).read().splitlines()
    syms, bundles, cur = [], [], None
    for ln in txt:
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:", ln)
        if m:
            a, n = int(m.group(1), 16), m.group(2)
            syms.append((a, n))
            if not n.startswith("."):
                cur = n
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", ln)
        if m:
            bundles.append([int(m.group(1), 16), m.group(2).strip(), cur])
    for b, nb in zip(bundles, bundles[1:] + [None]):
        b.append((nb[0] if nb else b[0] + 16) - b[0])
    return syms, bundles


def slots(text):
    ops = [o.strip() for o in text.split(";") if o.strip()]
    return ops


def is_nop(text):
    return all(o.split()[0].startswith("nop") for o in slots(text))


def mix(text):
    return tuple(sorted(o.split()[0].split(".")[0] for o in slots(text) if not o.split()[0].startswith("nop")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dis")
    ap.add_argument("--fold", default="")
    ap.add_argument("--detail")
    o = ap.parse_args()
    syms, bundles = parse(o.dis)
    fold = [f for f in o.fold.split(",") if f]
    owner = {}
    for b in bundles:
        n = b[2]
        for f in fold:
            if n and n != f and n.startswith(f + "_"):
                n = f
        owner[b[0]] = n
    name_at = {a: n for a, n in syms if not n.startswith(".")}
    # narrowest size seen per non-nop slot mix
    best = {}
    for a, t, _, sz in bundles:
        k = mix(t)
        best[k] = min(best.get(k, 99), sz)
    stat = defaultdict(lambda: Counter())
    callers = defaultdict(Counter)
    for a, t, _, sz in bundles:
        f = owner[a]
        s = stat[f]
        s["bytes"] += sz
        s["bundles"] += 1
        s["nop"] += is_nop(t)
        s["nop_bytes"] += sz if is_nop(t) else 0
        s["full16"] += sz == 16
        s["pack"] += sz - best[mix(t)]
        m = re.search(r"\bjl\s+#0x([0-9a-f]+)", t)
        if m:
            tgt = name_at.get(int(m.group(1), 16), m.group(1))
            for f2 in fold:
                if tgt.startswith(f2 + "_"):
                    tgt = f2
            callers[tgt][f] += 1
    tot = sum(s["bytes"] for s in stat.values())
    print(f"{'bytes':>6} {'bund':>5} {'B/b':>5} {'nop':>4} {'nopB':>5} {'16B':>4} {'pack<=':>6}  function  <- callers")
    for f, s in sorted(stat.items(), key=lambda kv: -kv[1]["bytes"]):
        c = ", ".join(f"{k} x{v}" for k, v in callers[f].items())
        print(f"{s['bytes']:6d} {s['bundles']:5d} {s['bytes'] / s['bundles']:5.1f} {s['nop']:4d} {s['nop_bytes']:5d} {s['full16']:4d}"
              f" {s['pack']:6d}  {f}  <- {c}")
    print(f"{tot:6d} total; nop bundles {sum(s['nop'] for s in stat.values())} = {sum(s['nop_bytes'] for s in stat.values())} B;"
          f" full-form bundles {sum(s['full16'] for s in stat.values())}")
    sz = Counter(b[3] for b in bundles)
    print("bundle sizes: " + ", ".join(f"{k} B x{v}" for k, v in sorted(sz.items())))
    if o.detail:
        for a, t, f, sz in bundles:
            if owner[a] == o.detail:
                print(f"  {a:5x} {sz:2d}  {t}")


if __name__ == "__main__":
    main()
