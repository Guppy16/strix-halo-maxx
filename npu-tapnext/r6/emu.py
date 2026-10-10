"""Bit-level numpy models of the AIE2P vector numerics used by nk8.h (round 6).

The MAC unit multiplies bf16 operands exactly (8 x 8-bit mantissas fit fp32) and accumulates in fp32 with
round-to-nearest-even; conversions accumulator -> bf16 round to nearest even (the kernels set conv_even).
int16 bit tricks act on the bf16 bit patterns. Each kernel model below mirrors its nk8.h op sequence line by line,
so it can be checked bit-exactly against the device (kprobe.py dumps every bf16 input through the real kernels).

Arrays: fp32 accumulators are np.float32, bf16 vectors are np.float32 holding bf16-representable values.
"""

import numpy as np
from ml_dtypes import bfloat16

F32 = np.float32


def bf(a):
    """accumulator / fp32 -> bf16 (RNE), returned as float32"""
    return np.asarray(a, F32).astype(bfloat16).astype(F32)


def mul(a, b):
    """bf16 x bf16 -> fp32 accumulator (exact product)"""
    return (np.asarray(a, F32) * np.asarray(b, F32)).astype(F32)


def mac(acc, a, b):
    """acc + a*b with an exact product, one fp32 rounding"""
    return (np.asarray(acc, np.float64) + np.asarray(a, np.float64) * np.asarray(b, np.float64)).astype(F32)


def msc(acc, a, b):
    return (np.asarray(acc, np.float64) - np.asarray(a, np.float64) * np.asarray(b, np.float64)).astype(F32)


def split(a):
    h = bf(a)
    return h, bf(msc(a, h, F32(1.0)))


def bits16(v):
    return np.asarray(v, F32).astype(bfloat16).view(np.int16).astype(np.int32)


def from16(i):
    return (np.asarray(i).astype(np.int16)).view(bfloat16).astype(F32)


def c_bf(v):
    """a float constant as the kernel's bfloat16 literal"""
    return F32(bfloat16(v))


M0 = F32(12582912.0)
L_HI = c_bf(1.4426950408889634)
L_LO = c_bf(1.4426950408889634 - float(L_HI))


def exp32_i(d, dl=None):
    """nk8::exp32_i"""
    d = np.clip(np.asarray(d, F32), F32(-60), F32(60))
    za = mul(d, L_HI)
    za = mac(za, d, L_LO)
    if dl is not None:
        za = mac(za, dl, L_HI)
    zn = mac(za, F32(1), M0)
    nb = bf(mac(zn, F32(1), -M0))
    f = bf(mac(za, nb, F32(-1)))
    n16 = (zn.view(np.int32) & 0xFFFF).astype(np.int16).astype(np.int32)  # low 16 bits of the fp32 word
    p2n = from16((n16 << 7) + 0x3F80)
    f2 = bf(mul(f, f))
    f3 = bf(mul(f2, f))
    p = mac(F32(1.0), f, c_bf(0.69314718))
    p = mac(p, f2, c_bf(0.24022651))
    p = mac(p, f3, c_bf(0.05550411))
    return bf(mul(bf(p), p2n))


def recipa_i(d):
    """nk8::recipa_i: 1/d as fp32 accumulator"""
    dh, dl = split(d)
    r0 = from16(0x7EF3 - bits16(dh))
    eps = msc(F32(1.0), dh, r0)
    eps = msc(eps, dl, r0)
    r = mac(mul(r0, F32(1.0)), r0, bf(eps))
    rh, rl = split(r)
    e2 = msc(F32(1.0), dh, rh)
    e2 = msc(e2, dh, rl)
    e2 = msc(e2, dl, rh)
    return mac(r, rh, bf(e2))


def recip1p_i(e):
    return recipa_i(mac(F32(1.0), e, F32(1.0)))


def sigmoid32_i(z):
    return recip1p_i(exp32_i(bf(mul(z, F32(-1.0)))))


def gelu32_i(x_in):
    """nk8::gelu32_i -> fp32 accumulator; the kernels round it to bf16"""
    x_in = np.asarray(x_in, F32)
    x = np.clip(x_in, F32(-8), F32(8))
    c1h = c_bf(0.035677408)
    c1l = c_bf(0.035677408 - float(c1h))
    ta = mul(x, x)
    th, tl = split(ta)
    wa = mac(F32(0.7978845608), th, c1h)
    wa = mac(wa, th, c1l)
    wa = mac(wa, tl, c1h)
    wh, wl = split(wa)
    ua = mul(x, wh)
    ua = mac(ua, x, wl)
    uh, ul = split(ua)
    e = exp32_i(bf(mul(uh, F32(-2.0))), bf(mul(ul, F32(-2.0))))
    s = recip1p_i(e)
    sh, sl = split(s)
    g = mul(x_in, sh)
    return mac(g, x_in, sl)


def gelu_n(x):
    return bf(gelu32_i(x))


# ---- references ---------------------------------------------------------------------------------------------
def gelu_tanh(x):
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from math import erf, sqrt
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))


def all_bf16():
    """every finite bf16 value (as float32)"""
    v = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(bfloat16).astype(F32)
    return v[np.isfinite(v)]


def ulp_err(got, exact):
    e = np.floor(np.log2(np.maximum(np.abs(exact), 1e-38)))
    return np.abs(np.asarray(got, np.float64) - exact) / 2.0 ** (e - 7)
