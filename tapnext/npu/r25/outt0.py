"""Round 25: out's tile 0 per column from a TPROF npz (r24tp): k-end, bias wait (mid), epilogue end, next tile entry, by row."""
import sys

import numpy as np

d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
F = dict(op=0, col=1, byte=2, first=3, ks=4, nks=5, mid=6, epi=7, hand=8, t0=9, n=10)
U = 1810.0
for blk in sys.argv[2].split(","):
    for opn in sys.argv[3].split(","):
        nm = f"{blk}_{opn}"
        r = a[names == nm].astype(np.int64)
        if not len(r):
            print(nm, "none")
            continue
        ns = sorted(set(r[:, F["n"]]))
        print(f"{nm}: {len(r)} recs; cols {sorted(set(r[:, F['col']]))[:8]}")
        k0 = r[:, F["n"]] == ns[0]
        r0 = r[k0]
        ref = (r0[:, F["ks"]] & 0xFFFFFFFF).min()
        for x in sorted(r0, key=lambda x: (x[F["col"]])):
            ke = (x[F["ks"]] & 0xFFFFFFFF) - ref
            print(f"  col {x[F['col']]:3d} k-end {ke / U:7.2f} mid {x[F['mid']] / U:6.2f} epi {x[F['epi']] / U:6.2f} -> epi-end {(ke + x[F['mid']] + x[F['epi']]) / U:7.2f} hand {x[F['hand']] / U:5.2f}")
        if len(ns) > 1:
            r1 = r[r[:, F["n"]] == ns[1]]
            t1 = (r1[:, F["t0"]] & 0xFFFFFFFF) - ref
            k1 = (r1[:, F["ks"]] & 0xFFFFFFFF) - ref
            print(f"  tile 1 entry {np.min(t1) / U:.2f}..{np.max(t1) / U:.2f}  k-end {np.min(k1) / U:.2f}..{np.max(k1) / U:.2f}")
