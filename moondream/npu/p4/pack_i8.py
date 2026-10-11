"""Host packing for design_i8.py (int8 8x8 panels, transposed orientation).

A side = weight rows:  [c][K/k][rho][2 halves][k/8][8 rows][8 k] int8 + sw[16] bias[16] f32  (1152 B per sub-tile)
B side = token tiles:  [j][K/k][n/16][k/8][2][8 k][8 tokens] int8 + sa[96] f32                 (6528 B per tile)
"""

import numpy as np

COLS, ROWS = 8, 4


def pack_w(wq, sw, bias, m, k, rho):
    """wq (R=COLS*m, K) int8, sw/bias (R,) f32 -> words [c][K/k][rho][sub-tile]."""
    R, K = wq.shape
    m_a = m // rho
    assert R == COLS * m and K % k == 0 and m_a == 16
    v = wq.reshape(COLS, rho, 2, 8, K // k, k // 8, 8)  # c g h i kk kb kk8
    v = v.transpose(0, 4, 1, 2, 5, 3, 6)  # c kk g h kb i kk8
    panels = np.ascontiguousarray(v).reshape(COLS, K // k, rho, 1024)
    p = np.stack([sw.reshape(COLS, rho, 16), bias.reshape(COLS, rho, 16)], axis=2)  # c g {sw,bias} 16
    p = np.ascontiguousarray(p.astype(np.float32)).view(np.uint8).reshape(COLS, 1, rho, 128)
    p = np.broadcast_to(p, (COLS, K // k, rho, 128))
    out = np.concatenate([panels.view(np.uint8), p], axis=3)
    return np.ascontiguousarray(out).reshape(-1).view(np.uint32)


def pack_x(xq, sa, n, k, gelu=False):
    """xq (T=ROWS*nt*n tokens, K) int8, sa (T,) f32 -> words [j][K/k][tile]; gelu flag rides after sa."""
    T, K = xq.shape
    assert T % n == 0 and n % 16 == 0
    v = xq.reshape(T // n, n // 16, 2, 8, K // k, k // 8, 8)  # j nb p tok kk kb kk8
    v = v.transpose(0, 4, 1, 5, 2, 6, 3)  # j kk nb kb p kk8 tok
    panels = np.ascontiguousarray(v).reshape(T // n, K // k, (n // 8) * (k // 8) * 64)
    s = sa.astype(np.float32).reshape(T // n, 1, n).view(np.uint8)
    s = np.broadcast_to(s, (T // n, K // k, n * 4))
    flag = np.zeros((T // n, K // k, 128), np.uint8)
    flag[:, :, 0] = 1 if gelu else 0
    out = np.concatenate([panels.view(np.uint8), s, flag], axis=2)
    return np.ascontiguousarray(out).reshape(-1).view(np.uint32)


def quant_rows(x):
    """moondream-opt quant_rows: per-row scale max(amax, 1e-5)/127, round half away from zero."""
    x = np.asarray(x, np.float32)
    amax = np.abs(x).max(axis=1)
    s = (np.maximum(amax, np.float32(1e-5)) / np.float32(127.0)).astype(np.float32)
    v = x / s[:, None]
    q = (v + np.where(v >= 0, np.float32(0.5), np.float32(-0.5))).astype(np.int32)
    return np.clip(q, -127, 127).astype(np.int8), s


def quant_w(w):
    """HI8Lin: per-output-channel scale max|w|/127 (>= 1e-8), round-to-nearest-even, clamp."""
    w = np.asarray(w, np.float32)
    s = (np.maximum(np.abs(w).max(axis=1), np.float32(1e-8)) / np.float32(127.0)).astype(np.float32)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def gelu_tanh(x):
    x = np.asarray(x, np.float32)
    u = np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x * x * x)
    t = np.float32(1.0) - np.float32(2.0) / (np.float32(1.0) + np.exp(np.float32(2.0) * u, dtype=np.float32))
    return (np.float32(0.5) * x * (np.float32(1.0) + t)).astype(np.float32)


def epilogue(acc, sa, sw, bias, gelu):
    """acc (R, T) int64 -> fp32 in the HIP kernel's order: ((acc * sa[t]) * sw[r]) + bias[r]."""
    v = acc.astype(np.float32) * sa[None, :].astype(np.float32)
    v = v * sw[:, None].astype(np.float32) + bias[:, None].astype(np.float32)
    return gelu_tanh(v) if gelu else v
