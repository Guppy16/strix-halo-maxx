"""Round 24: per-tile TPROF timeline of chosen ops in one block (KTL build: ks field = absolute k-steps end, t0 = entry).
usage: tpk.py X.npz BLOCK op1,op2,..  -> per op and tile index (core tile counter order): median over cores of entry, k-end, epi end,
in us relative to the first listed op's first tile entry (1.81 GHz)."""
import sys

import numpy as np

d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
blk, ops = sys.argv[2], sys.argv[3].split(",")
F = dict(op=0, col=1, byte=2, first=3, ks=4, nks=5, mid=6, epi=7, hand=8, t0=9, n=10)
U = 1810.0
ref = None
for o in ops:
    nm = f"{blk}_{o}"
    m = names == nm
    if not m.any():
        print(nm, "no records")
        continue
    r = a[m].astype(np.int64)
    t0 = r[:, F["t0"]] & 0xFFFFFFFF
    kend = r[:, F["ks"]] & 0xFFFFFFFF
    epi_end = kend + r[:, F["mid"]] + r[:, F["epi"]]
    if ref is None:
        ref = np.median(t0[r[:, F["n"]] == r[:, F["n"]].min()])
    ns = sorted(set(r[:, F["n"]]))
    print(f"{nm}: {len(r)} recs, tiles {len(ns)}")
    for k in ns:
        mk = r[:, F["n"]] == k
        e, ke, ee = np.median(t0[mk]), np.median(kend[mk]), np.median(epi_end[mk])
        print(f"   tile {k - ns[0]:2d}: entry {(e - ref) / U:8.2f}  k-end {(ke - ref) / U:8.2f} (+{(ke - e) / U:6.2f})  epi-end {(ee - ref) / U:8.2f}"
              f"  hand {np.median(r[mk, F['hand']]) / U:6.2f}  entry spread {(t0[mk].max() - t0[mk].min()) / U:5.2f}  nks {np.median(r[mk, F['nks']]):.0f}")
