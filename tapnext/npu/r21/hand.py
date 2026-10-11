"""Round 21: per-op epilogue / hand (prev epilogue end -> mm_ktile entry) medians from a tprof20 dump, non-first tiles.
usage: python3 r21/hand.py X.npz [block]"""
import sys
import numpy as np
d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
blk = sys.argv[2] if len(sys.argv) > 2 else "b11"
tot = 0
for op in sorted(set(a[:, 0])):
    m = a[:, 0] == op
    nm = names[m][0]
    if not nm.startswith(blk + "_"):
        continue
    r = a[m]
    n0 = r[:, 10].min()
    rr = r[r[:, 10] > n0]  # not the op s first tile
    rf = r[r[:, 10] == n0]
    if len(rr) == 0:
        rr = r
    per_core = len(r) / 24
    ex = np.median(rr[:, 8]) - 676
    tot += ex * per_core
    print(f"{nm:14s} tiles/core {per_core:5.1f} epi {np.median(rr[:,7]):6.0f} hand {np.median(rr[:,8]):6.0f} (p10 {np.percentile(rr[:,8],10):6.0f} p90 {np.percentile(rr[:,8],90):6.0f})  first-tile hand {np.median(rf[:,8]):7.0f}  exposed/core {ex*per_core:8.0f}")
print(f"total exposed hand above 676 per core per block: {tot:.0f} cycles = {tot/1810:.1f} us; x12 = {tot*12/1.81e6:.3f} ms")
