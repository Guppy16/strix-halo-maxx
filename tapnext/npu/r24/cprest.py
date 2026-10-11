"""Round 24: per sync run of a .bin (one block's worth), the post-sync instructions up to the next sync: total, shim pushes that were
configured in the run (blockwrite before them) by channel, and the instruction index of the last shim MM2S push.
usage (npu-iron): python3 r24/cprest.py BIN first_sync n"""
import sys
from collections import Counter

from aie.utils import txn_trace as T

w = T.load(sys.argv[1])
ops = T.decode(w)[1:]
pushes = [e for e in T.trace(w) if e.kind == "push"]
pidx = [i for i, o in enumerate(ops) if o.kind == "write32"]
tag = {i: (({0: "shim", 1: "mem"}.get(e.row, "core")), e.direction[0], e.channel) for i, e in zip(pidx, pushes)}
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
    cfg, plain, last_m = Counter(), Counter(), None
    bw = False
    for x in range(a, b):
        if ops[x].kind == "blockwrite":
            bw = True
        t = tag.get(x)
        if t:
            key = f"{t[0]}{t[1]}{t[2]}"
            (cfg if bw and t[0] == "shim" else plain)[key] += 1
            bw = False
            if t[0] == "shim" and t[1] == "M":
                last_m = x - a
    print(f"sync {k:3d}: {b - a:4d} instr; configured shim pushes {dict(cfg)}; bare pushes {dict(plain)}; last shim MM2S at +{last_m}")
