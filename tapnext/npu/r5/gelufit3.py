# the planned gelu_f op sequence, emulated (fp32 accumulators, bf16 roundings where the kernel converts)
import numpy as np
from ml_dtypes import bfloat16 as B
def bf(v): return np.asarray(v, np.float32).astype(B).astype(np.float32)
def split(a): h = bf(a); return h, bf(np.float32(a) - h)
def gelu(x): x = np.asarray(x, np.float64); return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
L = 1.4426950408889634
A, Bc = 2 * 0.7978845608028654, 2 * 0.7978845608028654 * 0.044715
KAh, KAl = split(np.float32(-L * A)); KBh, KBl = split(np.float32(-L * Bc))
M0 = np.float32(12582912.0)
def gelu_f(x, P, cub=(0.6931472, 0.2402265, 0.0555041)):
    x = bf(x)
    y = np.abs(x); yc = np.minimum(y, 8.0)
    y2 = bf(yc * yc); y3 = bf(yc * y2)
    za = (yc * KAh + yc * KAl + y3 * KBh + y3 * KBl).astype(np.float32)
    zn = (za + M0).astype(np.float32)
    nb = bf(zn - M0)
    f = bf(za - nb)
    c1, c2, c3 = [bf(c) for c in cub]
    t1 = bf(c2 + f * c3); t2 = bf(c1 + f * t1); p = np.float32(1.0) + f * t2
    u = bf(p) * np.float32(2.0) ** nb
    q = bf(np.float32(P[-2]) + u * bf(P[-1]))
    for c in P[-3::-1]:
        q = bf(np.float32(c) + u * q)
    s = (u * q).astype(np.float32)
    sh, sl = split(s)
    return bf(np.maximum(x, 0) - y * sh - y * sl)
uu = np.linspace(0, 1, 20001)
for d in (4, 5):
    w = np.ones_like(uu)
    for it in range(40):
        V = np.vander(uu, d + 1, increasing=True)[:, 1:]
        c = np.linalg.lstsq(V * w[:, None], uu / (1 + uu) * w, rcond=None)[0]
        e = V @ c - uu / (1 + uu); w = w * (1 + 4 * np.abs(e) / np.abs(e).max())
    cb = [float(bf(v)) for v in c]
    for name, xs in (("uniform[-8,8]", np.linspace(-8, 8, 400001)), ("N(0,1)", np.random.default_rng(0).standard_normal(400000)), ("N(0,3)", 3 * np.random.default_rng(1).standard_normal(400000))):
        xs = bf(xs)
        g = gelu_f(xs, c)
        r = bf(gelu(xs))
        ulp = np.maximum(np.abs(r), 1e-30) * 2.0 ** -7
        ue = np.abs(g - r) / (2.0 ** np.floor(np.log2(np.maximum(np.abs(r), 1e-30))) * 2.0 ** -7)
        rel = np.abs(g - gelu(xs)) / np.maximum(np.abs(gelu(xs)), 1e-3)
        print(f"deg {d} {name:14s}: mismatch {(g != r).mean()*100:5.1f}%  >1ulp {(ue > 1.01).mean()*100:5.2f}%  max ulp {ue[np.abs(r) > 1e-3].max():.1f}  rms rel {np.sqrt(np.mean(rel**2)):.2e}  coefs {np.round(c,4).tolist()}")
