"""Attention exp candidates on real scores (round 7): the deployed expd_i vs piecewise-linear LUT exps.

Scores: build/acts_clip3.npz attn_s_b<block> (CPU bf16 model, clip30 3 frames, 288 sampled query rows x 12 heads x
1072 keys, q k^T / 8). The kernel's lazy max is replayed per 48-key chunk (m starts at -64, moves when a chunk's row
max passes it by TAU = 8), P = exp(s - m) per chunk goes to bfp16 (blocks of 8 keys, encode_hw rounding), and the
softmax weights a = P' / sum P' (P' rescaled to the final m exactly) are compared with exact softmax weights.

Yardstick: ratio = rms(a_approx - a_exact) / rms(a_bfp(exact P) - a_exact), i.e. the error a variant adds on top of
the bfp16 rounding of exact P (1.0 = the bfp floor). Also the row-norm error |a - a_exact|_2 / |a_exact|_2 (what O
sees with i.i.d. V).

usage (npu-iron): python3 r7/expemu.py [blocks]
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r6"))
sys.path.insert(0, str(HERE.parent))
from emu import F32, L_HI, L_LO, M0, bf, bits16, c_bf, from16, mac, msc, mul  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

KC, TAU, NEG = 48, 8.0, -64.0
VARIANTS = ((-16, 8, 1 / 8), (-16, 8, 1 / 4), (-12, 8, 1 / 4), (-10, 8, 1 / 4), (-16, 8, 1 / 2), (-12, 8, 1 / 2))
FUSED = (True, False)


def bfp_hw(x):
    """values after a device bfp16ebs8 conversion along the last axis (encode_hw's rounding), float64"""
    sh = x.shape
    v = np.asarray(x, np.float64).reshape(-1, 8)
    amax = np.abs(v).max(-1)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))) - 6, -133.0)
    m = np.rint(v / np.exp2(e)[:, None])
    e = e + (m >= 128).any(-1)
    m = np.clip(np.rint(v / np.exp2(e)[:, None]), -128, 127)
    return (m * np.exp2(e)[:, None]).reshape(sh)


def expd_i(s, m):
    """nk8::expd_i, bit-level (round 6 deployed)"""
    za = mul(s, L_HI)
    za = mac(za, s, L_LO)
    za = msc(za, m, L_HI)
    za = msc(za, m, L_LO)
    zn = mac(za, F32(1), M0)
    nb = bf(mac(zn, F32(1), -M0))
    f = bf(mac(za, nb, F32(-1)))
    n16 = (zn.view(np.int32) & 0xFFFF).astype(np.int16).astype(np.int32)
    n16 = np.maximum(n16, -120)
    t = bf(mac(F32(0.703125), f, c_bf(0.240234375)))
    p = bf(mac(F32(1), f, t))
    return from16(bits16(p) + (n16 << 7))


def lut_table(lo, hi, h, rel=True):
    """segments of width h on [lo - h, hi): segment 0 is all zero (P = 0 below lo); bf16 slopes, fp32 offsets refit"""
    n = int(round((hi - lo) / h)) + 1
    sl, of = [0.0], [0.0]
    for i in range(1, n):
        x0 = lo + (i - 1) * h
        xs = np.linspace(x0, x0 + h, 257)[:-1]
        y = np.exp(xs)
        w = 1 / y if rel else np.ones_like(y)  # minimise relative error
        a, b = np.polyfit(xs, y, 1, w=w)
        a = float(np.float32(a).astype(bfloat16))
        b = float(np.float32(_best_off(xs, y, a)))
        sl.append(a)
        of.append(b)
    return np.array(sl, F32), np.array(of, F32)


def _best_off(xs, y, a):
    # minimise max |a x + b - y| / y over b: b in [min(y - a x), max(y - a x)], bisection on the two error ends
    lo, hi = float((y - a * xs).min()), float((y - a * xs).max())
    for _ in range(60):
        b = (lo + hi) / 2
        e = (a * xs + b - y) / y
        if e.max() > -e.min():
            hi = b
        else:
            lo = b
    return (lo + hi) / 2


def lut_exp(s, m, tab, lo, hi, h, fused=True):
    """d = s - m in fp32 (exact), hi/lo bf16 split, i = floor((dh - (lo - h)) / h), p = off + slope dh + slope dl"""
    sl, of = tab
    d = msc(mul(s, F32(1)), m, F32(1))
    dh = bf(d)
    dl = bf(msc(d, dh, F32(1)))
    dh = np.minimum(np.maximum(dh, F32(lo - h)), F32(bf(hi - h / 2)))
    i = np.floor((dh.astype(np.float64) - (lo - h)) / h).astype(np.int64)
    p = mac(mac(of[i], sl[i], dh), sl[i], dl)
    return p if fused else bf(p)


def aexp_dev(s, m):
    """the deployed round-7 table (r7/aexp_lut.npz, r7/explut.py) exactly as mm8.cc's expl_i: clamp dh to
    [lo, bf16 below hi], i = floor(dh * 4) + bias (the device's bfloat16_to_int floor), fp32 p"""
    t = np.load(HERE / "aexp_lut.npz")
    sl, of, lo, hi, h = t["slopes"], t["offsets"], float(t["lo"]), float(t["hi"]), 2.0 ** float(t["step"])
    d = mac(np.zeros_like(s, F32) - m, s, F32(1))  # mac(-m, s, 1): fp32 rounding once
    dh = bf(d)
    dl = bf(msc(d, dh, F32(1)))
    dc = np.minimum(np.maximum(dh, F32(lo)), F32(np.nextafter(bfloat16(hi), bfloat16(0))))
    i = np.floor((dc.astype(np.float64) - lo) / h).astype(np.int64)
    return mac(mac(of[i], sl[i], dc), sl[i], dl)


def replay(s, expfn):
    """lazy-max online softmax over 48-key chunks for s [R, K]; returns the final weights a [R, K] (P bfp-rounded)"""
    R, K = s.shape
    m = np.full(R, NEG, F32)
    P = np.zeros((R, K))
    mk = np.zeros((R, K))  # the m each key's P was taken against
    for c0 in range(0, K, KC):
        sc = s[:, c0:c0 + KC]
        mx = sc.max(1)
        mv = np.maximum(mx, m)
        upd = mv > m + TAU
        m = np.where(upd, mv, m).astype(F32)
        mm = np.broadcast_to(m[:, None], sc.shape)
        P[:, c0:c0 + KC] = expfn(sc, mm)
        mk[:, c0:c0 + KC] = mm
    Pb = bfp_hw(P)
    w = Pb * np.exp(mk - m[:, None])  # the exact rescale to the final m (O and sums are rescaled in fp32 hi/lo)
    return w / w.sum(1, keepdims=True)


def main():
    global VARIANTS, FUSED
    blocks = [int(b) for b in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 5, 11]
    if len(sys.argv) > 2:  # "lo:hi:h,lo:hi:h" (fused only)
        VARIANTS = tuple(tuple(float(v) for v in t.split(":")) for t in sys.argv[2].split(","))
        FUSED = (True,)
    z = np.load(HERE.parent / "build" / "acts_clip3.npz")
    variants = {"expd_i (deployed)": lambda s, m: expd_i(s, m)}
    if (HERE / "aexp_lut.npz").exists():
        variants["aexp (r7 device table, fp32->bfp)"] = aexp_dev
    for lo, hi, h in VARIANTS:
        tab = lut_table(lo, hi, h)
        nseg = len(tab[0])
        for fused in FUSED:
            variants[f"lut [{lo},{hi}) h={h} {nseg} seg {nseg * 16} B/copy {'fp32->bfp' if fused else 'bf16->bfp'}"] = (
                lambda s, m, tab=tab, lo=lo, hi=hi, h=h, fused=fused: lut_exp(s, m, tab, lo, hi, h, fused))
    for b in blocks:
        s = bf(z[f"attn_s_b{b}"].astype(F32))  # [rows, heads, keys]
        R, H, K = s.shape
        s2 = s.reshape(R * H, K)
        # exact reference: softmax of the bf16 scores (float64)
        ex = np.exp(s2 - s2.max(1, keepdims=True))
        a_ex = ex / ex.sum(1, keepdims=True)
        a_floor = replay(s2, lambda sc, mm: np.exp(sc.astype(np.float64) - mm))  # exact exp, bfp P
        e_floor = np.sqrt(np.mean((a_floor - a_ex) ** 2))
        print(f"block {b}: rows x heads {R * H}, bfp floor rms {e_floor:.3e}, row-norm err "
              f"{np.mean(np.linalg.norm(a_floor - a_ex, axis=1) / np.linalg.norm(a_ex, axis=1)):.3e}")
        for name, fn in variants.items():
            a = replay(s2, fn)
            e = np.sqrt(np.mean((a - a_ex) ** 2))
            rn = np.mean(np.linalg.norm(a - a_ex, axis=1) / np.linalg.norm(a_ex, axis=1))
            print(f"  {name:62s} ratio {e / e_floor:6.3f}  row-norm {rn:.3e}")


if __name__ == "__main__":
    main()
