"""Zero-overhead loops in an AIE2P disassembly (npu-research rev 8): for each `movxm ls, #A` / `movxm le, #B` pair, the
function, the body bundles [A, B] and the loop count source (the nearest preceding `lc` write). Static cycles per trip =
body bundles (no stalls). usage: python3 loops.py X.dis [--func NAME]"""
import re
import sys

txt = open(sys.argv[1]).read().splitlines()
want = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--func" else None
cur, rows, addrs, ls = None, [], [], None
for ln in txt:
    m = re.match(r"^([0-9a-f]+) <([^>]+)>:", ln)
    if m:
        if not m.group(2).startswith("."):
            cur = m.group(2)
        continue
    m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", ln)
    if not m:
        continue
    a, t = int(m.group(1), 16), m.group(2)
    addrs.append((a, cur, t))
lc = {}
for a, f, t in addrs:
    m = re.search(r"movxm\s+ls, #0x([0-9a-f]+)", t)
    if m:
        ls = (int(m.group(1), 16), f)
    m = re.search(r"(add\.nc\s+lc,[^;]*|mov[a-z]*\s+lc,[^;]*)", t)
    if m:
        lc[f] = m.group(1).strip()
    m = re.search(r"movxm\s+le, #0x([0-9a-f]+)", t)
    if m and ls:
        lo, hi = ls[0], int(m.group(1), 16)
        n = sum(1 for b, _, _ in addrs if lo <= b <= hi)
        if want is None or f == want or (f or "").startswith(want):
            print(f"{f[:60]:60s} body [0x{lo:x}, 0x{hi:x}] {n:3d} bundles  lc: {lc.get(f, '?')}")
