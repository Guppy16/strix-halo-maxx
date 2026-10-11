"""Round 25: compact view of the CP runs after each sync of a .bin: pushes in issue order, consecutive pushes of the same kind
(shim / mem, M / S, channel, length, configured in the run or bare) merged with their column list, plus the instruction offset of
each item inside the run. usage (npu-iron): python3 r25/cpcompact.py BIN first_sync n"""
import sys

from aie.utils import txn_trace as T

w = T.load(sys.argv[1])
ops = T.decode(w)[1:]
pushes = [e for e in T.trace(w) if e.kind == "push"]
pidx = [i for i, o in enumerate(ops) if o.kind == "write32"]
assert len(pidx) == len(pushes)
tag = {i: e for i, e in zip(pidx, pushes)}
syncs, i = [], 0
while i < len(ops):
    if ops[i].kind == "tct":
        j = i
        while j < len(ops) and ops[j].kind == "tct":
            j += 1
        syncs.append((i, j))
        i = j
    else:
        i += 1
k0, n = int(sys.argv[2]), int(sys.argv[3])
for k in range(k0, min(k0 + n, len(syncs) - 1)):
    a, b = syncs[k][1], syncs[k + 1][0]
    items, bw, ncfg = [], False, 0
    for x in range(a, b):
        o = ops[x]
        if o.kind == "blockwrite":
            bw = True
            ncfg += 1
        e = tag.get(x)
        if e is None:
            continue
        kind = ("shim" if e.row == 0 else "mem" if e.row == 1 else "core") + e.direction[0] + str(e.channel)
        cfg = bw and e.row == 0
        ncfg -= cfg
        key = (("cfg " if cfg else "") + kind, e.bd.length if e.bd else 0)
        bw = False
        if items and items[-1][0] == key:
            items[-1][1].append(e.col)
        else:
            items.append([key, [e.col], x - a])
    s = "; ".join(f"@{off} {k_[0]}:{k_[1]} c{''.join(map(str, cs))}" for k_, cs, off in items)
    print(f"sync {k:3d} (+{syncs[k][1] - syncs[k][0]} tct) {b - a:4d} instr | {s}" + (f" | +{ncfg} blockwrites w/o push" if ncfg > 0 else ""))
