import sys, numpy as np
d = np.load(sys.argv[1]); c, ex = d["c"], d["exact"]
ok = np.isclose(c, ex, rtol=2**-7, atol=1e-6); fin = np.isfinite(c)
print("ok", ok.mean(), "finite", fin.mean())
t = ok[:96, :96].reshape(12, 8, 12, 8).mean(axis=(1, 3))
print("first tile ok per 8x8 block (rows=row blocks 0..11, cols=col blocks 0..11):"); print(np.array2string(t, precision=2, max_line_width=200))
f = fin[:96, :96].reshape(12, 8, 12, 8).mean(axis=(1, 3)); print("finite per block:"); print(np.array2string(f, precision=2, max_line_width=200))
rbok = ok[:728].reshape(91, 8, -1).mean(axis=(1, 2)); print("ok per row block (first 24):", np.array2string(rbok[:24], precision=2, max_line_width=200))
cbok = ok[:, :1152].reshape(-1, 144, 8).mean(axis=(0, 2)); print("ok per col block (first 24):", np.array2string(cbok[:24], precision=2, max_line_width=200))
i = np.load(sys.argv[1].replace(".npz", "") + "_in.npz"); a, w = i["a"].astype(np.float64), i["w"].astype(np.float64)
bad = (~ok) & fin
for name, mask in (("odd", (np.arange(a.shape[1]) // 8) % 2 == 1), ("even", (np.arange(a.shape[1]) // 8) % 2 == 0), ("full", np.arange(a.shape[1]) >= 0)):
    ref = a[:, mask] @ w[:, mask].T
    e = np.sqrt(((c - ref)[bad] ** 2).sum() / (ref[bad] ** 2).sum()); print(f"wrong-finite entries vs {name}: rms_rel {e:.3e}")
