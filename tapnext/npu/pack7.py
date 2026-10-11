"""Host packing for design7: bfp16ebs8 panels in the order the cores pop them, plus the bias objects."""

import numpy as np
from ml_dtypes import bfloat16

from pack import decode_rows, encode_rows, pack_a  # noqa: F401  (pack_a(ea, m, k, rho, half=True) is unchanged)

COLS, ROWS_G = 8, 3


def pack_w7(ew, bias, n, k):
    """W^T encoded (N, K/8, 9) + bias (N,) float -> words [j][K/k tiles + 1 bias object].

    Tile words: [n/16][k/8][2][8 cols][9B] as pack.py; the bias object is the same size (k x n bfp tile) and holds
    8 x 64 bf16: vector cb = bias[j*n + 8 cb : + 8] tiled over 8 rows (mm7.cc acc_store_bias), zero padded."""
    N, KB, _ = ew.shape
    wb = k * n * 9 // 8 // 4
    out = []
    for j in range(N // n):
        v = ew[j * n:(j + 1) * n].reshape(n // 16, 2, 8, KB // (k // 8), k // 8, 9)
        v = v.transpose(3, 0, 4, 1, 2, 5)  # [nb][h][i][kk][cb][b] -> [kk][nb][cb][h][i][b] (pack.py's order per tile)
        out.append(np.ascontiguousarray(v).reshape(-1).view(np.uint32))
        b = np.asarray(bias[j * n:(j + 1) * n], np.float32).astype(bfloat16).reshape(n // 8, 8)
        tiled = np.repeat(b[:, None, :], 8, axis=1).reshape(-1)  # [cb][8 rows][8] -> 8 x 64
        obj = np.zeros(wb * 2, bfloat16)
        obj[: tiled.size] = tiled
        out.append(obj.view(np.uint32))
    return np.concatenate(out)


def unblock(c, m, N, n=64):
    """[COLS*m, N] bf16 in the blocked C layout (per core tile [m/8][n/8][8][8]) -> natural [COLS*m, N]."""
    tiles = c.reshape(COLS, m, N // n, n).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(COLS, N // n, m // 8, n // 8, 8, 8)
    return tiles.transpose(0, 2, 4, 1, 3, 5).reshape(COLS * m, N)


def encode_hw(x):
    """bfp16ebs8 as the AIE2P conversion unit does it (round to nearest even at the block's max exponent; if a
    mantissa rounds to +128 the shared exponent is bumped and every mantissa re-rounded; -128 is kept), unlike
    aie.utils.bfp, which saturates at 127. float (R, K) -> uint8 (R, K/8, 9) = [E][8 x int8], value = m * 2^(E - 133)."""
    R, K = x.shape
    v = np.asarray(x, np.float32).reshape(R, K // 8, 8).astype(np.float64)
    amax = np.abs(v).max(-1)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))) - 6, -133.0)
    scale = np.exp2(e)[..., None]
    m = np.rint(v / scale)  # numpy rint = round half to even
    bump = (m >= 128).any(-1)  # +128 does not fit int8; -128 does, and the device keeps it
    e = e + bump
    scale = np.exp2(e)[..., None]
    m = np.rint(v / scale)
    m = np.clip(m, -128, 127)
    out = np.zeros((R, K // 8, 9), np.uint8)
    out[..., 0] = np.clip(e + 133, 0, 255).astype(np.uint8)
    out[..., 1:] = m.astype(np.int8).view(np.uint8)
    return out
