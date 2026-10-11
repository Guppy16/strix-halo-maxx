"""Round 25: the C hand-off ("hand": epilogue end -> the next C object acquired) per GEMM op of a TPROF npz (block 11): median per
tile over cores, the tile count, and the per-block total of the median hand on the slowest core path (sum over tiles)."""
import sys

import numpy as np

d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
F = dict(op=0, col=1, byte=2, first=3, ks=4, nks=5, mid=6, epi=7, hand=8, t0=9, n=10)
U = 1810.0
tot = 0
for nm in sorted(set(names)):
    if not nm.startswith(sys.argv[2] + "_"):
        continue
    r = a[names == nm].astype(np.int64)
    ns = sorted(set(r[:, F["n"]]))
    per = [np.median(r[r[:, F["n"]] == k, F["hand"]]) / U for k in ns]
    mx = [np.max(r[r[:, F["n"]] == k, F["hand"]]) / U for k in ns]
    s = sum(per[:-1])  # the last tile's hand leads into the next op
    tot += s
    print(f"{nm:14s} tiles {len(ns):2d}  hand median per tile {np.median(per):5.2f} us (max {max(mx):5.2f})  sum (tiles 0..n-2) {s:6.2f} us")
print(f"total {tot:.1f} us per block")
