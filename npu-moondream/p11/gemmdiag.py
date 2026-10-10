import sys, numpy as np
d = np.load(sys.argv[1]); c, ex = d["c"], d["exact"]
ok = np.isclose(c, ex, rtol=2**-7, atol=1e-6)
print("overall ok fraction", ok.mean(), "shape", c.shape)
t = ok[:96, :96]
print("first 96x96 tile: ok fraction per 8x8 block (rows = row blocks, cols = col blocks):")
print(np.array2string(t.reshape(12, 8, 12, 8).mean(axis=(1, 3)), precision=2, max_line_width=200))
print("ok per row-in-block:", np.array2string(t.reshape(12, 8, 12, 8).mean(axis=(0, 2, 3)), precision=2))
print("ok per col-in-block:", np.array2string(t.reshape(12, 8, 12, 8).mean(axis=(0, 1, 2)), precision=2))
print("ok per row block (first 728 rows):", np.array2string(ok[:728].reshape(-1, 8, ok.shape[1]).mean(axis=(1, 2))[:24], precision=2, max_line_width=200))
r = c[:16, :16]; e = ex[:16, :16]
print("block(0,0) got:\n", np.array2string(r[:4, :8], precision=3), "\nexpected:\n", np.array2string(e[:4, :8], precision=3))
# is the wrong output a plausible partial sum? ratio statistics
fin = np.isfinite(c) & (np.abs(ex) > 1e-3)
print("ratio got/expected median", np.median((c / ex)[fin]), "fraction |got| huge (>1e3):", (np.abs(c) > 1e3).mean())
