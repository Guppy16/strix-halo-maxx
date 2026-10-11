"""Round 20: op boundaries from TPROF timers (the 24 GEMM cores' timers agree within ~1k cycles). Per GEMM op of a block:
end of the previous GEMM op's last epilogue, this op's first k-step end, the first tile's k-step rate, the A-pacing excess.
usage: python3 r20/tpgap.py X.npz [block] [steady]"""
import sys
import numpy as np
d = np.load(sys.argv[1])
a, names = d["a"], d["names"]
blk = sys.argv[2] if len(sys.argv) > 2 else "b11"
ST = float(sys.argv[3]) if len(sys.argv) > 3 else 1576.0
ops = []
for op in sorted(set(a[:, 0])):
    m = a[:, 0] == op
    if names[m][0].startswith(blk + "_"):
        ops.append((names[m][0], a[m]))
prev_end = None
tot = 0.0
print(f"{'op':13s} {'nks':>3s} {'ks1/k':>6s} {'pace_exc_us':>11s} {'k0_end-prevGEMMend_us':>22s} {'tile0 end->tile1 start':>22s} {'last tile end':>14s}")
for nm, r in ops:
    t0 = r[:, 9]
    end = t0 + r[:, 3] + r[:, 4] + r[:, 6] + r[:, 7]
    n0 = r[:, 10].min()
    f = r[r[:, 10] == n0]
    nks = int(np.median(f[:, 5]))
    ksk = np.median(f[:, 4] / np.maximum(f[:, 5] - 1, 1))
    exc = (ksk - ST) * (nks - 1) / 1810.0
    k0e = np.median(f[:, 9] + f[:, 3])
    g = (k0e - prev_end) / 1810.0 if prev_end is not None else float("nan")
    print(f"{nm:13s} {nks:3d} {ksk:6.0f} {exc:11.2f} {g:22.2f} {'':22s} {np.max(end):14d}")
    tot += max(exc, 0)
    prev_end = np.max(end)
print(f"first-tile A-pacing excess, sum over the block's GEMMs: {tot:.1f} us -> x12 = {tot*12/1e3:.2f} ms")
