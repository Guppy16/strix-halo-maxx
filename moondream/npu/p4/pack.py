"""Host packing for design.py: bfp16ebs8 blocks in the exact order the cores pop them."""

import numpy as np
from aie.utils import bfp

COLS, ROWS = 8, 4


def encode_rows(x):
    """float (R, K) -> uint8 (R, K/8, 9), blocks of 8 along K, round to nearest even."""
    R, K = x.shape
    return bfp.encode(np.asarray(x, np.float32), rounding="conv_even").reshape(R, K // 8, 9)


def decode_rows(e):
    R, KB, _ = e.shape
    return bfp.decode(e.reshape(R, KB * 9)).reshape(R, KB * 8)


def pack_a(ea, m, k, rho, half=False):
    """(M=8m, K/8, 9) -> words: [c][K/k][rho][m_a/16][k/8][2][8 rows][9B]
    (half=True, for -DBLK12: [c][K/k][rho][m_a/16][2][k/8][8 rows][9B], row-half major)."""
    M, KB, _ = ea.shape
    m_a = m // rho
    v = ea.reshape(COLS, rho, m_a // 16, 2, 8, KB // (k // 8), k // 8, 9)
    #             c    g    rp         h  i  kk               cb      b
    v = v.transpose(0, 5, 1, 2, 3, 6, 4, 7) if half else v.transpose(0, 5, 1, 2, 6, 3, 4, 7)
    return np.ascontiguousarray(v).reshape(-1).view(np.uint32)


def pack_w(ew, n, k):
    """W^T encoded (N, K/8, 9) -> words: [j][K/k][n/16][k/8][2][8 cols][9B]."""
    N, KB, _ = ew.shape
    v = ew.reshape(N // n, n // 16, 2, 8, KB // (k // 8), k // 8, 9)
    #              j       nb       h  i  kk               cb      b
    v = v.transpose(0, 4, 1, 5, 2, 3, 6)
    return np.ascontiguousarray(v).reshape(-1).view(np.uint32)
