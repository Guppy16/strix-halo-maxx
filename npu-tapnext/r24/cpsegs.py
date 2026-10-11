"""Round 24: the CP instruction runs after each token-wait run (sync) of a .bin, for a range of syncs: instruction kinds with the
push tags (shim / mem, M / S channel, column, length). usage (npu-iron): python3 r24/cpsegs.py BIN [first_sync] [n_syncs]"""
import sys

from aie.utils import txn_trace as T

w = T.load(sys.argv[1])
ops = T.decode(w)[1:]
pushes = [e for e in T.trace(w) if e.kind == "push"]
pidx = [i for i, o in enumerate(ops) if o.kind == "write32"]
assert len(pidx) == len(pushes)
tag = {}
for i, e in zip(pidx, pushes):
    tag[i] = f"{ {0: 'shim', 1: 'mem'}.get(e.row, 'core')}{e.direction[0]}{e.channel}c{e.col}:{e.bd.length if e.bd else 0}"
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
k0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
print(f"{len(ops)} instructions, {len(syncs)} sync runs")
from collections import Counter
print("kinds:", Counter(o.kind for o in ops))
for k in range(k0, min(k0 + n, len(syncs) - 1)):
    a, b = syncs[k][1], syncs[k + 1][0]
    seq = []
    for x in range(a, b):
        seq.append(tag.get(x, ops[x].kind))
    print(f"## sync {k} (+{syncs[k][1] - syncs[k][0]} tct), {b - a} instr: " + " ".join(seq))
