"""rev 10: per-slot op counts of the zero-overhead loops in a disassembly (-z), with the .td slot classes.
    python3 slots.py X.dis FUNC_SUBSTR..."""
import json, re, sys
from pathlib import Path
DEFS = json.load(open(Path(__file__).resolve().parents[3] / "common/tools/aie2p_isa_timing.json"))["defs"]
def slot_of(mn):
    cl = sorted({v["cls"].split("_")[2] for v in DEFS.values() if v["mn"] == mn})
    return "/".join(cl) if cl else None
cache = {}
txt = open(sys.argv[1]).read().splitlines()
want = sys.argv[2:]
cur = None; lines = []
for ln in txt:
    m = re.match(r"^([0-9a-f]+) <([^>]+)>:", ln)
    if m:
        if not m.group(2).startswith("."): cur = m.group(2)
        continue
    m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", ln)
    if m: lines.append((int(m.group(1), 16), cur, m.group(2)))
ls = None
for i, (a, f, t) in enumerate(lines):
    m = re.search(r"movxm\s+ls, #(0x[0-9a-f]+)", t)
    if m: ls = int(m.group(1), 16)
    m = re.search(r"movxm\s+le, #(0x[0-9a-f]+)", t)
    if m and ls is not None and any(w in f for w in want):
        le = int(m.group(1), 16)
        body = [x for x in lines if ls <= x[0] <= le]
        cnt = {}
        for _, _, tt in body:
            for op in tt.split(";"):
                mn = op.strip().split()[0] if op.strip() else ""
                if not mn or mn.startswith("nop"): continue
                if mn not in cache: cache[mn] = slot_of(mn)
                s = cache[mn] or mn
                cnt[s] = cnt.get(s, 0) + 1
        print(f"{f[:40]:40s} {len(body):4d} bundles  ops by slot: {cnt}")
