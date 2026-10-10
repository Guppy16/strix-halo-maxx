# h(y) = gelu_tanh(y) - relu(y) for y = |x| (h is even in this form: gelu(x) = relu(x) + h(|x|)), polynomial fits with
# bf16 Horner emulation (fp32 accumulate, bf16 round per step, as on the AIE)
import numpy as np
from ml_dtypes import bfloat16 as B
def bf(v): return np.asarray(v, np.float32).astype(B).astype(np.float32)
def gelu(x): x = np.asarray(x, np.float64); return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
def h(y): return gelu(y) - y
for Y in (4.0, 4.5, 5.0):
    y = np.linspace(0, Y, 40001)
    for d in (6, 7, 8, 9, 10):
        # minimax-ish absolute fit via iteratively reweighted least squares on Chebyshev basis then to monomials
        t = 2 * y / Y - 1
        w = np.ones_like(y)
        for it in range(40):
            V = np.polynomial.chebyshev.chebvander(t, d)
            c = np.linalg.lstsq(V * w[:, None], h(y) * w, rcond=None)[0]
            e = V @ c - h(y)
            w = w * (1 + 4 * np.abs(e) / np.abs(e).max())
        mono = np.polynomial.chebyshev.cheb2poly(c)  # in t
        # Horner in t with bf16 coefficients and bf16 intermediates; t = bf(y * 2/Y - 1) in fp32 then rounded
        xs = np.concatenate([np.linspace(-8, 8, 200001)])
        yy = np.abs(xs)
        tt = bf(yy * (2.0 / Y) - 1.0)
        cb = bf(mono)
        q = np.full_like(tt, cb[-1])
        for k in range(d - 1, -1, -1):
            q = bf(np.float32(mono[k]) + tt * q) if k > 0 else (np.float32(mono[0]) + tt * q)  # last step stays fp32
        hq = np.where(yy < Y, q, 0.0)
        g = bf(np.maximum(xs, 0) + hq)
        ref = gelu(xs)
        err = np.abs(g - ref)
        scale = np.maximum(np.abs(ref), 2.0 ** -6)
        print(f"Y {Y} deg {d}: fit max {np.abs(e).max():.2e}; emulated max abs {err.max():.2e} max rel(>=2^-6) {np.max(err/scale):.2e} at x={xs[np.argmax(err/scale)]:.3f}; "
              f"bf16-exact-mismatch {(g != bf(ref)).mean()*100:.1f}%")
