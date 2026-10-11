"""Rev 16: one line per post-sync segment of a .bin (block given): instruction count and the pushes in order, run-length
compressed by (tile kind, direction+channel, length). Also the length of the tct run that ends the segment.
python3 r21/cpsig.py BIN [first_seg] [count]   (npu-iron: aie.utils.txn_trace)"""
import sys

sys.path.insert(0, "/work/npu-research/r20")
from aie.utils import txn_trace as T  # noqa: E402

w = T.load(sys.argv[1])
ops = T.decode(w)[1:]
pushes = [e for e in T.trace(w) if e.kind == "push"]
pidx = [i for i, o in enumerate(ops) if o.kind == "write32"]
tag = {i: ({0: "shim", 1: "mem"}.get(e.row, "core"), f"{e.direction[0]}{e.channel}", e.col, e.bd.length if e.bd else 0) for i, e in zip(pidx, pushes)}
runs, i = [], 0
while i < len(ops):
    if ops[i].kind == "tct":
        j = i
        while j < len(ops) and ops[j].kind == "tct":
            j += 1
        runs.append((i, j))
        i = j
    else:
        i += 1
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cnt = int(sys.argv[3]) if len(sys.argv) > 3 else 30
print(f"{len(runs)} tct runs; {len(ops)} instructions")
for k in range(first, min(first + cnt, len(runs))):
    a = runs[k - 1][1] if k else 0
    b = runs[k][0]
    sig, prev = [], None
    for x in range(a, b):
        t = tag.get(x)
        if t is None:
            continue
        key = (t[0], t[1], t[3])
        if sig and sig[-1][0] == key:
            sig[-1][1] += 1
        else:
            sig.append([key, 1])
    s = " ".join(f"{kk[0]}.{kk[1]}:{kk[2]}x{n}" for kk, n in sig)
    print(f"seg {k:3d} n={b - a:4d} tct={runs[k][1] - runs[k][0]:3d} | {s}")
