"""Host packing for design8: weights, parameter objects, the Q16 activation layout, bfp helpers."""

import numpy as np
from ml_dtypes import bfloat16

from pack import decode_rows, encode_rows  # noqa: F401  (the mlir-aie codec, conv_even)
from pack7 import encode_hw  # noqa: F401  (the device's own bfp conversion)
from plan8 import COLS, EPI, KSTEP16, M, OBJ, PAN, PATCH_PARAM_BYTES, RB, SLOT, SSM_PARAM_BYTES, UNIT, XB

SCALE = np.float32(1.0 / 8.0)  # attention: 1/sqrt(64)


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16)


# The asm GEMM call (build kdefs -DMM_ASM, r7/gen_mm.py) pops each pair of 16-column blocks as [q][4 col blocks]:
# step8 / run8 set this from the build (uses_asm_gemm) before packing. GEMM weights only: the EW cores' SSM / patch
# parameter tiles keep the [n/16][k/8][2] order (they pass pair4=False).
B_PAIR4 = False


B_ROT = None  # round 18 (ROT): the k-step's B panel order (stream position -> pair4 panel index), from r18/mm_rot_g2f.h (mm_rot_border)


def rot_order(build_json):
    """the ROT B order if the build's kdefs have -DMM_ROT, else None"""
    import json
    from pathlib import Path
    if "-DMM_ROT" not in json.load(open(build_json)).get("kdefs", "").replace(",", " ").split():
        return None
    import re
    v = json.load(open(build_json)).get("rot_v", "g2f")
    here = Path(__file__).resolve().parent
    h = next(d / f"mm_rot_{v}.h" for d in (here / "r23", here / "r22", here / "r21", here / "r19", here / "r18") if (d / f"mm_rot_{v}.h").exists()).read_text()  # r19: g1a / g0a
    order = [int(v) for v in re.search(r"mm_rot_border\[64\] = \{([^}]*)\}", h).group(1).split(",")]
    assert sorted(order) == list(range(64)), order
    return order


def uses_asm_gemm(build_json):
    import json
    return "-DMM_ASM" in json.load(open(build_json)).get("kdefs", "").replace(",", " ").split()


def w_tile_words(ew_tile, pair4=False):
    """One 64 x 64 weight tile: ew_tile = encoded W^T rows (64 n, 8 k-blocks, 9 B) -> [n/16][k/8][2][8 cols][9B]
    (pair4: [n/32][k/8][2 blocks][2][8 cols][9B], the asm GEMM's order)."""
    v = ew_tile.reshape(4, 2, 8, 8, 9).transpose(0, 3, 1, 2, 4)  # [nb][h][i][cb][b] -> [nb][cb][h][i][b]
    if pair4:
        v = v.reshape(2, 2, 8, 2, 8, 9).transpose(0, 2, 1, 3, 4, 5)  # [p][bb][cb][h][i][b] -> [p][cb][bb][h][i][b]
        if B_ROT is not None:  # round 18 ROT: whole 72 B panels in the rotating k-step's pop order
            v = np.ascontiguousarray(v).reshape(64, 72)[B_ROT]
    return np.ascontiguousarray(v).reshape(-1).view(np.uint32)


def bias_obj(bias64, flag):
    """The object appended to each N-tile's weight stream: 8 x 64 bf16 bias vectors (column block cb repeated for 8
    rows) then the epilogue flag word at byte 1024."""
    b = bf(bias64).reshape(8, 8)
    tiled = np.repeat(b[:, None, :], 8, axis=1).reshape(-1)
    obj = np.zeros(OBJ // 2, bfloat16)
    obj[:512] = tiled
    w = obj.view(np.uint32).copy()
    w[256] = EPI[flag]
    return w


def pack_gemm(op, wt, bias):
    """W^T [N, K] float, bias [N] float -> the op's W words ([row][tile][K/64 tiles + bias object]).
    Gated ops take wt/bias of the whole ffw_up half assigned to the op with columns already interleaved
    ([32 gate | 32 up] per tile): see interleave_gated."""
    N, K = wt.shape
    assert N == op["N"] and K == op["K"], (wt.shape, op["N"], op["K"])
    ew = encode_rows(np.asarray(wt, np.float32))
    order = list(range(N // 64))
    if op["epi"] == "gated":  # the pair's second tile (gated_b) writes its panels first: process (2p+1, 2p)
        order = [t ^ 1 for t in order]
    nt = op["nt"]
    if op.get("rr"):  # round 10 (P23): row r's step j is natural tile 3 j + r; flags are given per natural tile
        perm = op.get("perm") or list(range(nt))  # (round 23 YXR: step j takes natural group perm[j])
        order = [3 * perm[pos % nt] + pos // nt for pos in range(N // 64)]
    out = []
    for pos, t in enumerate(order):  # flags follow the stream position (gated_a = first of a pair)
        for kk in range(K // 64):
            out.append(w_tile_words(ew[t * 64:(t + 1) * 64, kk * 8:(kk + 1) * 8], B_PAIR4))
        out.append(bias_obj(bias[t * 64:(t + 1) * 64] if op["bias_in"] else np.zeros(64), op["flags"][t if op.get("rr") else pos]))
    w = np.concatenate(out)
    assert w.nbytes == 3 * op["nt"] * op["lw"], (w.nbytes, 3 * op["nt"] * op["lw"])
    return w


def interleave_gated(wt_gate, wt_up, b_gate, b_up):
    """ffw_up halves [3072, K] each -> [6144, K] with tiles of [32 gate | 32 up] columns (and the biases)."""
    g = wt_gate.reshape(96, 32, -1)
    u = wt_up.reshape(96, 32, -1)
    wt = np.concatenate([g, u], axis=1).reshape(6144, -1)
    b = np.concatenate([b_gate.reshape(96, 32), b_up.reshape(96, 32)], axis=1).reshape(-1)
    return wt, b


EW_KIND = dict(rms=1, ln=2, ln_out=3, ssm=4)  # header word 3 of an EW op's first params slot (ew8.cc ew_op)


def norm_params(mode, scale, beta=None, n_in=1, eps=1e-6, nres=False):
    """[header slot: word 0 = n_in, word 1 = eps (bf16 bits)][scale + 1 (rms) | gamma][beta (ln)] as 768 bf16
    vectors, one slot each pair."""
    nslots = 3 if mode in ("rms", "none") else 5
    out = np.zeros(nslots * SLOT // 2, bfloat16)
    hdr = out.view(np.uint32)
    hdr[0] = n_in
    out[2] = bf(eps)
    hdr[3] = EW_KIND[mode] | (8 if nres else 0)  # ew_op dispatches on it (round 20: + 8 = NRES, panels to the EW MM2S 1 ring)
    v = bf(np.asarray(scale, np.float32) + 1.0) if mode == "rms" else bf(np.ones(768) if mode == "none" else scale)
    out[SLOT // 2: SLOT // 2 + 768] = v
    if mode in ("ln", "ln_out"):
        out[3 * SLOT // 2: 3 * SLOT // 2 + 768] = bf(beta)
    return out.view(np.uint32)


def ssm_params(head, conv_w, conv_b, a_param, wx, bx, wa, ba, first):
    """Per-head SSM parameter object (SSM_PARAM_BYTES): slots 0-1 = header + bf16 vectors, then Wx, Wa as 64x64
    weight tiles (bfp16, [64 out][64 in] given as W^T rows = out rows). conv_w [4, 64] (taps oldest..current),
    conv_b [64], a_param [64] (softplus folded: we store c = -8 * softplus(a_param) as bf16), wx/wa [64 in, 64 out]
    (torch's BlockDiagonalLinear einsum x[i] w[i][j]), bx/ba [64]."""
    hdr = np.zeros(2 * SLOT // 4, np.uint32)
    hdr[0] = 1 if first else 0
    hdr[1] = head
    vec = np.zeros(2 * SLOT // 2, bfloat16)
    sp = np.log1p(np.exp(np.asarray(a_param, np.float64)))
    vec[64:128] = bf(sp)  # torch: log_a = bf16(bf16(-8 gate_a) * bf16(softplus(a_param)))
    vec[128:192] = bf(conv_b)
    for t in range(4):
        vec[192 + 64 * t:256 + 64 * t] = bf(conv_w[t])
    vec[448:512] = bf(bx)
    vec[512:576] = bf(ba)
    hv = vec.view(np.uint32)
    hv[:2] = hdr[:2]
    hv[3] = EW_KIND["ssm"]
    ewx = encode_rows(np.asarray(wx, np.float32).T)  # W^T [out, in]
    ewa = encode_rows(np.asarray(wa, np.float32).T)
    out = np.concatenate([hv, w_tile_words(ewx), w_tile_words(ewa)])
    assert out.nbytes == SSM_PARAM_BYTES, out.nbytes
    return out


def patch_params(q, conv_w, conv_b):
    """k-step q of the patch embedding: 3 weight tiles (K = 192 = 3 x 64) for output cols [64 q, 64 q + 64) + bias.
    conv_w [768, 3, 8, 8] with the /127.5 - 1 input normalisation folded in by the caller (conv_fold)."""
    wt = np.asarray(conv_w, np.float32).reshape(768, 192)[q * 64:(q + 1) * 64]
    ew = encode_rows(wt)
    tiles = [w_tile_words(ew[:, kk * 8:(kk + 1) * 8]) for kk in range(3)]
    b = np.zeros((PATCH_PARAM_BYTES - 3 * OBJ) // 2, bfloat16)
    b[:64] = bf(conv_b[q * 64:(q + 1) * 64])
    out = np.concatenate(tiles + [b.view(np.uint32)])
    assert out.nbytes == PATCH_PARAM_BYTES
    return out


def conv_fold(conv_w, conv_b):
    """Fold x/127.5 - 1 into the conv so the device gets raw uint8 pixel values as bf16."""
    w = np.asarray(conv_w, np.float32)
    return w / 127.5, np.asarray(conv_b, np.float32) - w.reshape(768, -1).sum(1)


def frame_patches(img_u8):
    """[256, 256, 3] uint8 -> [1024, 192] float (patch-major rows, (c, ky, kx) order like the conv weights)."""
    p = np.asarray(img_u8, np.float32).reshape(32, 8, 32, 8, 3).transpose(0, 2, 4, 1, 3).reshape(1024, 192)
    return p


def to_blocked_units(x, ncols, unit, xoff=0):
    """[COLS*M, ncols] bf16 -> per column [q][rb][unit B] with the 8x8-blocked 1024 B k-step parts at byte xoff."""
    nq = ncols // 64
    out = np.zeros((COLS, nq, RB, unit), np.uint8)
    v = x.reshape(COLS, RB, 8, nq, 8, 8).transpose(0, 3, 1, 4, 2, 5)  # [c][q][rb][cb][8 rows][8 cols]
    out[..., xoff:xoff + XB] = np.ascontiguousarray(v).reshape(COLS, nq, RB, XB // 2).view(np.uint8)
    return out.reshape(-1).view(np.uint32)


def to_q16(x, ncols=768):
    return to_blocked_units(x, ncols, UNIT)


def from_q16(words, ncols=768, unit=UNIT, xoff=0):
    """Inverse of to_q16 (the 1024 B parts) -> [COLS*M, ncols] bf16."""
    nq = ncols // 64
    u = words.view(np.uint8).reshape(COLS, nq, RB, unit)[..., xoff:xoff + XB].view(bfloat16)
    v = u.reshape(COLS, nq, RB, 8, 8, 8).transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(v).reshape(COLS * M, ncols)


def panels_from_q16(words, ncols=768, unit=UNIT, pan_off=XB):
    """The 576 B panel parts -> uint8 [COLS*M, ncols/8, 9] (bfp rows, decodable with pack.decode_rows)."""
    nq = ncols // 64
    u = words.view(np.uint8).reshape(COLS, nq, RB, unit)[..., pan_off:pan_off + PAN]
    v = u.reshape(COLS, nq, RB, 8, 8, 9).transpose(0, 2, 4, 1, 3, 5)  # [c][rb][row][q][cb][9]
    return np.ascontiguousarray(v).reshape(COLS * M, ncols // 8, 9)


def panels_from_apan(words, region, nq):
    """APAN region words (per column [q][kstride]) -> uint8 [COLS*M, 8*nq, 9]."""
    ks = region["kstride"]
    u = words.view(np.uint8).reshape(COLS, region["nq"], ks)[:, :nq, :RB * PAN]
    v = u.reshape(COLS, nq, RB, 8, 8, 9).transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(v).reshape(COLS * M, nq * 8, 9)


def panels_to_q16_words(rows_u8, ncols=768, unit=UNIT, pan_off=XB):
    """uint8 [COLS*M, ncols/8, 9] -> the panel parts of a Q16 region (for building expected buffers)."""
    nq = ncols // 64
    out = np.zeros((COLS, nq, RB, unit), np.uint8)
    v = rows_u8.reshape(COLS, RB, 8, nq, 8, 9).transpose(0, 3, 1, 4, 2, 5)
    out[..., pan_off:pan_off + PAN] = np.ascontiguousarray(v).reshape(COLS, nq, RB, PAN)
    return out.reshape(-1).view(np.uint32)


def rmsnorm_ref(x, sc1):
    """TAPNext RMSNorm with torch's bf16 numerics: x bf16 [R, D], sc1 = (scale + 1) bf16 [D] -> bf16."""
    f = lambda v: np.asarray(v, np.float32).astype(bfloat16).astype(np.float32)
    xf = x.astype(np.float32)
    s = f(xf * xf).sum(axis=1, dtype=np.float32)
    mean = f(s / np.float32(x.shape[1]))
    rstd = f(1.0 / np.sqrt(f(mean + np.float32(1e-6))))
    return f(f(xf * rstd[:, None]) * sc1.astype(np.float32)[None]).astype(bfloat16)


def layernorm_ref(x, gamma, beta):
    """torch's bf16 LayerNorm: fp32 inside, one rounding at the end."""
    xf = x.astype(np.float32)
    mean = xf.mean(axis=1, dtype=np.float32, keepdims=True)
    var = ((xf - mean) ** 2).mean(axis=1, dtype=np.float32, keepdims=True)
    y = (xf - mean) / np.sqrt(var + np.float32(1e-6)) * gamma.astype(np.float32) + beta.astype(np.float32)
    return y.astype(bfloat16)


def gelu_ref(x):
    xf = np.asarray(x, np.float64)
    return 0.5 * xf * (1.0 + np.tanh(0.7978845608028654 * (xf + 0.044715 * xf ** 3)))


def gelu_erf_ref(x):
    """the exact GELU (torch nn.GELU(), the ViT MLP's): the gelu_pan epilogue computes this form since round 6"""
    from math import erf, sqrt
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))


def ssm_ref(y, x, xm1, xm2, xm3, h0, conv_w, conv_b, a_param, wx, bx, wa, ba, first):
    """torch's bf16 numerics of the recurrent block's elementwise part for one head: inputs [R, 64] bf16 (h0 fp32),
    conv_w [4, 64] (oldest..current), wx/wa [64 in, 64 out]. Returns (z bf16, h fp32)."""
    f = lambda v: np.asarray(v, np.float32).astype(bfloat16).astype(np.float32)
    F = lambda v: np.asarray(v, np.float32)
    taps = [xm3, xm2, xm1, x]
    s = sum(f(F(t) * f(conv_w[i])[None]) for i, t in enumerate(taps))
    xc = f(f(s) + f(conv_b)[None])
    gx_pre = f(f(xc.astype(np.float64) @ np.asarray(wx, np.float64)) + f(bx)[None])
    ga_pre = f(f(xc.astype(np.float64) @ np.asarray(wa, np.float64)) + f(ba)[None])
    gate_x = f(1.0 / (1.0 + np.exp(-gx_pre.astype(np.float64))))
    gate_a = f(1.0 / (1.0 + np.exp(-ga_pre.astype(np.float64))))
    sp = f(np.log1p(np.exp(np.asarray(a_param, np.float64))))
    log_a = f(f(-8.0 * gate_a) * sp[None])
    a = f(np.exp(log_a.astype(np.float64)))
    gated = f(xc * gate_x)
    if first:
        h = gated.astype(np.float32)
        hb = gated
    else:
        e2 = f(np.exp(f(2.0 * log_a).astype(np.float64)))
        mult = f(np.sqrt(f(1.0 - e2)))
        nx = f(gated * mult)
        h = (a * F(h0) + nx).astype(np.float32)
        hb = f(h)
    z = f(hb * F(y))
    return z.astype(bfloat16), h


def h_units(h, half):
    """fp32 [COLS*M, 768] -> the X' parts of a Q16 region image holding rows 4 half..4 half+3 of every 8-row block as
    [cb 8][4 rows][8] fp32 (1024 B per unit)."""
    out = np.zeros((COLS, 12, RB, UNIT), np.uint8)
    v = np.asarray(h, np.float32).reshape(COLS, RB, 2, 4, 12, 8, 8)[:, :, half]  # [c][rb][4 rows][q][cb][8]
    v = v.transpose(0, 3, 1, 4, 2, 5)  # [c][q][rb][cb][4][8]
    out[..., :XB] = np.ascontiguousarray(v).reshape(COLS, 12, RB, XB // 4).view(np.uint8)
    return out.reshape(-1).view(np.uint32)


def h_from_units(words_lo, words_hi):
    out = np.zeros((COLS, RB, 2, 4, 12, 8, 8), np.float32)
    for half, words in enumerate((words_lo, words_hi)):
        u = words.view(np.uint8).reshape(COLS, 12, RB, UNIT)[..., :XB].view(np.float32).reshape(COLS, 12, RB, 8, 4, 8)
        out[:, :, half] = u.transpose(0, 2, 4, 1, 3, 5)
    return out.reshape(COLS * M, 768)
