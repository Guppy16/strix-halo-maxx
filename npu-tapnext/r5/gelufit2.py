# gelu(x) = relu(x) - y s, y = |x|, s = u/(1+u), u = exp(-|t|), t = y (a + b y^2): polynomial for s(u) on [0, 1]
import numpy as np
from ml_dtypes import bfloat16 as B
def bf(v): return np.asarray(v, np.float32).astype(B).astype(np.float32)
def gelu(x): x = np.asarray(x, np.float64); return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
A, Bc = 2 * 0.7978845608028654, 2 * 0.7978845608028654 * 0.044715
uu = np.linspace(0, 1, 20001)
xs = np.linspace(-8, 8, 400001)
ref = gelu(xs)
rng = np.random.default_rng(0)
for d in (3, 4, 5, 6):
    w = np.ones_like(uu)
    for it in range(40):
        V = np.vander(uu, d + 1, increasing=True)[:, 1:]  # s(0) = 0
        c = np.linalg.lstsq(V * w[:, None], uu / (1 + uu) * w, rcond=None)[0]
        e = V @ c - uu / (1 + uu)
        w = w * (1 + 4 * np.abs(e) / np.abs(e).max())
    for uerr in (0.0, 2e-3, 5e-3):
        y = np.minimum(np.abs(xs), 8)
        y2 = bf(y * y)
        wv = bf(np.float32(A) + y2 * bf(Bc))
        t = y * wv  # fp32
        u = np.exp(-t.astype(np.float64)) * (1 + uerr * rng.uniform(-1, 1, t.shape))
        u = bf(u)
        q = np.full_like(u, bf(c[-1]))
        for k in range(d - 2, -1, -1):
            q = bf(np.float32(c[k]) + u * q)
        s = u * q  # fp32 (s = u (c1 + u (...)))
        sh = bf(s); sl = bf(s - sh)
        g = bf(np.maximum(xs, 0) - np.abs(xs) * sh - np.abs(xs) * sl)
        err = np.abs(g - ref)
        scale = np.maximum(np.abs(ref), 2.0 ** -6)
        print(f"deg {d} uerr {uerr}: fit {np.abs(e).max():.1e} coefs {np.round(c, 4)}; max abs {err.max():.2e} max rel(>=2^-6) {np.max(err / scale):.2e}; "
              f"bf16 mismatch {(g != bf(ref)).mean() * 100:.1f}% (> 1 ulp {(np.abs(g - bf(ref)) > np.abs(bf(ref)) * 2 ** -8 + 1e-30).mean() * 100:.2f}%)")
# the current nk8 gelu as a baseline needs no emulation: its documented result was 15% flips <= 2 ulp
