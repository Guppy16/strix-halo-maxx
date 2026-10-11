import sys, collections
from aie.utils import txn_trace as T
w = T.load(sys.argv[1])
ops = T.decode(w)
print("opcodes", collections.Counter(type(o).__name__ if not hasattr(o,'opcode') else str(o.opcode) for o in ops).most_common())
ev = T.trace(w)
# max pushes on a channel between two waits on the same channel (upper bound of queue occupancy if nothing completes)
cnt = collections.Counter(); worst = collections.Counter(); where = {}
for i, e in enumerate(ev):
    k = (e.col, e.row, e.direction, e.channel)
    if e.kind == "push":
        cnt[k] += 1
        if cnt[k] > worst[k]: worst[k] = cnt[k]; where[k] = i
    elif e.kind == "wait":
        cnt[k] = 0
for k in sorted(worst): print(k, "max pushes between waits", worst[k], "at event", where[k])
