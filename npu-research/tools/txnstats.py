import sys, collections
from aie.utils import txn_trace as T
w = T.load(sys.argv[1])
ev = T.trace(w)
kinds = collections.Counter(e.kind for e in ev)
print("events", dict(kinds), "words", len(w))
bursts = collections.Counter(); bytes_ch = collections.Counter(); pushes_ch = collections.Counter()
lens = collections.Counter()
def nbytes(e):
    n = e.bd.length * 4 * max(1, e.repeat + 1 if e.repeat else 1)
    for t in e.chain: n += t.length * 4
    return n
for e in ev:
    if e.kind == "push":
        k = (e.col, e.row, e.direction, e.channel)
        bursts[e.bd.burst_axcache] += 1
        bytes_ch[k] += nbytes(e); pushes_ch[k] += 1
print("burst/axcache", dict(bursts))
for k in sorted(bytes_ch): print(k, "pushes", pushes_ch[k], "MB", round(bytes_ch[k] / 1e6, 1))
# wait groups: consecutive waits = one sync point
sync, prev = 0, None
for e in ev:
    if e.kind == "wait" and prev != "wait": sync += 1
    prev = e.kind
print("sync points (runs of waits)", sync)
# pushes between sync points
seg, segs = 0, []
prev = None
for e in ev:
    if e.kind == "push": seg += 1
    if e.kind == "wait" and prev != "wait": segs.append(seg); seg = 0
    prev = e.kind
print("pushes per sync segment: min", min(segs), "max", max(segs), "mean", round(sum(segs)/len(segs),1))
writes = [e for e in ev if e.kind == "write"]
print("register writes", len(writes), collections.Counter((e.row) for e in writes))
