"""Round 20: print the CP instructions of a .bin between two token-wait runs (segment index), with push targets.
usage (npu-iron): python3 r20/cpdump.py X.bin SEG [SEG2 ...] | python3 r20/cpdump.py X.bin --list"""
import sys
from aie.utils import txn_trace as T
w = T.load(sys.argv[1])
ops = T.decode(w)[1:]
pushes = [e for e in T.trace(w) if e.kind == "push"]
pidx = [i for i, o in enumerate(ops) if o.kind == "write32"]
pk = dict(zip(pidx, pushes))
runs, i = [], 0
while i < len(ops):
    if ops[i].kind == "tct":
        j = i
        while j < len(ops) and ops[j].kind == "tct":
            j += 1
        runs.append((i, j)); i = j
    else:
        i += 1
if sys.argv[2] == "--list":
    for k, (a, b) in enumerate(runs[:-1]):
        nxt = runs[k + 1][0]
        kinds = {}
        for o in ops[b:nxt]:
            kinds[o.kind] = kinds.get(o.kind, 0) + 1
        print(k, "tct", b - a, "then", nxt - b, kinds)
    sys.exit()
for s in map(int, sys.argv[2:]):
    a, b = runs[s]
    nxt = runs[s + 1][0] if s + 1 < len(runs) else len(ops)
    print(f"--- segment {s}: {b - a} tct, then {nxt - b} instructions")
    for i in range(a, nxt):
        o = ops[i]
        extra = ""
        if i in pk:
            e = pk[i]
            extra = f"PUSH row {e.row} col {e.col} {e.direction} ch {e.channel} len {e.bd.length if e.bd else 0}"
        print(i, o.kind, getattr(o, "addr", ""), extra)
