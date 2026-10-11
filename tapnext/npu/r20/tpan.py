"""Round 20: per-op first-tile analysis of a tprof20 dump. usage: python3 r20/tpan.py X.npz [block]"""
import sys
import numpy as np
d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
blk = sys.argv[2] if len(sys.argv) > 2 else "b11"
F = dict(op=0, col=1, byte=2, first=3, ks=4, nks=5, mid=6, epi=7, hand=8, t0=9, n=10)
print("ops with records:", len(set(names)))
for op in sorted(set(a[:, 0])):
    m = a[:, 0] == op
    nm = names[m][0]
    if not nm.startswith(blk + "_"):
        continue
    r = a[m]
    n0 = r[:, 10].min()
    print(f"{nm:14s} recs {m.sum():4d} n {n0}..{r[:,10].max()} cols {sorted(set(r[:,1]))[:8]}")
    for k in sorted(set(r[:, 10]))[:3]:
        rk = r[r[:, 10] == k]
        ksk = rk[:, 4] / np.maximum(rk[:, 5] - 1, 1)
        print(f"   tile n={k}: {len(rk):3d} recs  first {np.median(rk[:,3]):7.0f} [{rk[:,3].min()}..{rk[:,3].max()}]  ks/k {np.median(ksk):6.0f} [{ksk.min():.0f}..{ksk.max():.0f}]"
              f"  nks {np.median(rk[:,5]):.0f} epi {np.median(rk[:,7]):6.0f} hand {np.median(rk[:,8]):7.0f} [{rk[:,8].min()}..{rk[:,8].max()}]  t0 spread {rk[:,9].max()-rk[:,9].min()}")
