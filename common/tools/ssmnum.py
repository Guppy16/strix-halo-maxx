"""ssmnum: bit-level numpy emulation of RG-LRU transcendental designs on real gate inputs (npu-research rev 4).

Reference = torch's bf16 op order (pack8.ssm_ref): sg = bf(sigmoid(x)); t = bf(-8 sg); log_a = bf(t * sp);
a = bf(exp(log_a)); a2 = bf(exp(2 log_a)); om = bf(1 - a2); mult = bf(sqrt(om)). Every bf() is RNE.
Inputs: build/acts_davis3.npz rglru_ga_b* / rglru_gx_b* (288 rows x 768, 12 blocks, real davis_reg frames) and
sp = bf(softplus(a_param)) from the safetensors.

Designs emulated (D-numbers = ANALYSIS rev 4 section 7):
  sigmoid  S0 exact (reference)
           S1 PWL table, segments of w on [-R, 0), sigma(x) = 1 - T(-x) for x > 0, 0 below -R  (one table, both gates)
           S2 PWL table, segments of w on [-R, R), clamped outside (GELU-style)
  a / a2   A0 exact (reference)
           A1 exact table on the bf16 bits of log_a (bit-exact by construction) - size report only
           A2 "split exp": e^x = E[x_hi] * p(x_lo), x_hi = floor(x / h) h, p = 1 + x_lo + x_lo^2/2 + x_lo^3/6 (+ x^4/24)
              with E as fp32 (hi/lo bf16 into the product), p evaluated in fp32 from bf16 x_lo (exact)
           A3 PWL table on [-L, 0) with segments of w (absolute index), fp32 offset + bf16 slope
  mult     M0 exact; M1 = exact sqrt table (mantissa + exponent parity: bit-exact by construction)

usage (read-only container): python3 ssmnum.py [--blocks 0,5,11]
"""

import argparse
import os
import sys

import ml_dtypes
import numpy as np

BF = ml_dtypes.bfloat16
SCRATCH = os.environ.get("SHM_SCRATCH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.scratch")
ACTS = f"{SCRATCH}/npu-tapnext/build/acts_davis3.npz"
W = os.environ.get("TAPNEXT_WEIGHTS", f"{SCRATCH}/models/tapnextpp_256.safetensors")


def bf(x):
    return np.asarray(x, np.float64).astype(np.float32).astype(BF).astype(np.float64)


def bf_bits(x):
    return np.asarray(x, np.float32).astype(BF).view(np.uint16)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ---- PWL table in the AIE2P LUT format: fp32 offset, bf16 slope, value = off + slope * x (one fp32 MAC) ----------
def pwl_table(f, lo, hi, w):
    """minimax-ish PWL: per segment least-max fit on a dense grid, slope rounded to bf16, offset refit (fp32)"""
    n = int(round((hi - lo) / w))
    off = np.zeros(n)
    slope = np.zeros(n)
    for i in range(n):
        xs = np.linspace(lo + i * w, lo + (i + 1) * w, 65)
        ys = f(xs)
        s = bf((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
        r = ys - s * xs
        o = 0.5 * (r.max() + r.min())  # equi-ripple offset for the rounded slope
        slope[i] = s
        off[i] = np.float32(o)
    return off, slope


def pwl_eval(x, off, slope, lo, w):
    i = np.floor((x - lo) / w).astype(np.int64)
    i = np.clip(i, 0, len(off) - 1)
    return off[i].astype(np.float32).astype(np.float64) + slope[i] * x  # fp32 accumulate of an exact product


def sig_s1(x, R=16.0, w=0.125):
    off, slope = pwl_table(sig, -R, 0.0, w)
    ax = -np.abs(x)
    t = pwl_eval(ax, off, slope, -R, w)
    t = np.where(ax < -R, 0.0, t)
    return np.where(x > 0, 1.0 - t, t)  # 1 - T in the accumulator (one msc), then RNE at the use


def sig_s2(x, R=8.0, w=0.125):
    off, slope = pwl_table(sig, -R, R, w)
    t = pwl_eval(x, off, slope, -R, w)
    return np.where(x < -R, 0.0, np.where(x >= R, 1.0, t))


# ---- exp designs for a = exp(log_a), log_a <= 0 bf16 --------------------------------------------------------------
def exp_a2(x, h=0.125, deg=4):
    """split exp: x_hi = floor(x / h) h (exact), x_lo = x - x_hi in [0, h) (exact in bf16 for bf16 x), E = fp32 e^x_hi,
    p(x_lo) by Horner in the accumulator with bf16 x_lo; E split hi/lo bf16 into the product (2 MACs)"""
    xhi = np.floor(x / h) * h
    xlo = x - xhi
    E = np.exp(xhi).astype(np.float32).astype(np.float64)
    Eh = bf(E)
    El = bf(E - Eh)
    c = [1.0, 1.0, 0.5, 1.0 / 6, 1.0 / 24, 1.0 / 120][: deg + 1]
    # Horner with bf16 rounding of each partial (the multiplicand must be bf16): p_k = c_k + x * bf(p_{k+1})
    p = np.full_like(x, c[-1])
    for ck in reversed(c[:-1]):
        p = np.float32(ck) + xlo * bf(p)  # fp32 accumulate
        p = p.astype(np.float32).astype(np.float64)
    ph = bf(p)
    pl = bf(p - ph)
    return Eh * ph + Eh * pl + El * ph  # three MACs, fp32 accumulate (El*pl negligible)


def exp_a3(x, L=19.0, w=1.0 / 64):
    off, slope = pwl_table(np.exp, -L, 0.0, w)
    return pwl_eval(x, off, slope, -L, w)


def mism(ref, got, mask=None):
    r, g = bf_bits(ref), bf_bits(got)
    if mask is not None:
        r, g = r[mask], g[mask]
    d = r.astype(np.int32) - g.astype(np.int32)
    return (d != 0).mean(), np.abs(d).max() if d.size else 0, d.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--only2", action="store_true")
    o = ap.parse_args()
    z = np.load(ACTS)
    from safetensors.numpy import load_file

    sd = load_file(W)
    blocks = [int(b) for b in o.blocks.split(",")]
    GA, GX, SP = [], [], []
    for b in blocks:
        ap_ = sd[f"blocks.{b}.ssm_block.recurrent_block.rg_lru.a_param"].astype(np.float64)
        sp = bf(np.log1p(np.exp(ap_)))
        ga = bf(z[f"rglru_ga_b{b}"].astype(np.float64))
        gx = bf(z[f"rglru_gx_b{b}"].astype(np.float64))
        GA.append(ga)
        GX.append(gx)
        SP.append(np.broadcast_to(sp, ga.shape))
    ga, gx, sp = (np.concatenate([v.reshape(-1) for v in V]) for V in (GA, GX, SP))
    print(f"{ga.size} gate_a / gate_x values, blocks {blocks}")

    # reference chain
    sgr = bf(sig(ga))
    log_a = bf(bf(-8.0 * sgr) * sp)
    a_r = bf(np.exp(log_a))
    a2_r = bf(np.exp(2.0 * log_a))
    om_r = bf(1.0 - a2_r)
    mult_r = bf(np.sqrt(om_r))
    near1 = a_r >= 0.95
    print(f"log_a range [{log_a.min():.3g}, {log_a.max():.3g}];  a == 1: {(a_r == 1).mean():.4f};  a >= 0.95: {near1.mean():.4f};"
          f"  mult == 0: {(mult_r == 0).mean():.4f}")
    nz = np.unique(bf_bits(log_a[log_a < -2.0 ** -9]))
    print(f"distinct bf16 log_a with |log_a| > 2^-9 on this data: {nz.size}; all bf16 in [-9.5, -2^-9]: "
          f"{int(bf_bits(np.float32(-9.5))) - int(bf_bits(np.float32(-(2 ** -9)))) + 1}")
    gxs = bf(sig(gx))
    if o.only2:
        part2(log_a, a_r, a2_r, mult_r, near1)
        part3(log_a, a_r, mult_r, sgr)
        part4(ga, gx, sp, sgr, a_r, mult_r, gxs)
        return 0

    print("\n-- sigmoid designs: bf16 mismatch vs RNE(sigma) [frac, max ulp, mean signed ulp] --")
    for name, f in [("S1 sym  w=1/8  R=16 (128 seg)", lambda x: sig_s1(x, 16, 1 / 8)),
                    ("S1 sym  w=1/16 R=16 (256 seg)", lambda x: sig_s1(x, 16, 1 / 16)),
                    ("S1 sym  w=1/8  R=12 (96 seg)", lambda x: sig_s1(x, 12, 1 / 8)),
                    ("S2 [-8,8) w=1/8 (128 seg)", lambda x: sig_s2(x, 8, 1 / 8))]:
        sa = bf(f(ga))
        sx = bf(f(gx))
        print(f"  {name:32s} gate_a {mism(sgr, sa)}  gate_x {mism(gxs, sx)}")
        # propagate gate_a through the exact rest of the chain
        la = bf(bf(-8.0 * sa) * sp)
        a = bf(np.exp(la))
        mult = bf(np.sqrt(bf(1.0 - bf(np.exp(2.0 * la)))))
        print(f"  {'':32s}  -> a {mism(a_r, a)}  a>=0.95 {mism(a_r, a, near1)}  mult {mism(mult_r, mult)}"
              f"  mean(a - a_ref) {np.mean(a - a_r):+.2e} (a>=0.95: {np.mean((a - a_r)[near1]):+.2e})")
        # the gated input x * sigma_x: relative rms vs bf16 RNE quantisation of the exact product
        ex = gx * sig(gx)
        q = bf(ex) - ex
        print(f"  {'':32s}  gate_x bfp-style ratio rms(err)/rms(bf16 quant) of sigma_x: "
              f"{np.sqrt(np.mean((sx - sig(gx)) ** 2)) / np.sqrt(np.mean((gxs - sig(gx)) ** 2)):.3f}")

    print("\n-- exp designs for a = bf(exp(log_a)) and a2 = bf(exp(2 log_a)) (exact sigmoid upstream) --")
    xs = log_a
    for name, f in [("A2 split h=1/8 deg 3", lambda x: exp_a2(x, 1 / 8, 3)),
                    ("A2 split h=1/8 deg 4", lambda x: exp_a2(x, 1 / 8, 4)),
                    ("A2 split h=1/4 deg 4", lambda x: exp_a2(x, 1 / 4, 4)),
                    ("A2 split h=1/16 deg 3", lambda x: exp_a2(x, 1 / 16, 3)),
                    ("A3 PWL w=1/64 on [-19,0) (1216 seg)", lambda x: exp_a3(x, 19.0, 1 / 64)),
                    ("A3 PWL w=1/16 on [-19,0) (304 seg)", lambda x: exp_a3(x, 19.0, 1 / 16))]:
        a = bf(f(xs))
        a2 = bf(f(2.0 * xs))
        mult = bf(np.sqrt(bf(1.0 - a2)))
        print(f"  {name:36s} a {mism(a_r, a)}  a>=0.95 {mism(a_r, a, near1)}  a2 {mism(a2_r, a2)}  mult {mism(mult_r, mult)}")
    # a2 from a's fp32 value (today's trick) with the split exp
    for name, (h, d) in [("A2 h=1/8 deg 4, a2 = (fp32 a)^2", (1 / 8, 4))]:
        af = exp_a2(xs, h, d)
        ah = bf(af)
        al = bf(af - ah)
        a2 = bf(ah * ah + ah * bf(2 * al))
        print(f"  {name:36s} a2 {mism(a2_r, a2)}  mult {mism(mult_r, bf(np.sqrt(bf(1.0 - a2))))}")




# ---- part 2 (rev 4): grid-aware tables -----------------------------------------------------------------------------
def bf_from_bits(b):
    return np.asarray(b, np.uint16).view(BF).astype(np.float64)


def grid_table(f, b_lo, b_hi, s, exact_fit=True):
    """PWL over the bf16 GRID of |x|: segment i covers the 2^s consecutive bf16 bit patterns b_lo + i 2^s ...;
    fit (bf16 slope, fp32 offset) minimax on the grid points (the only inputs that exist). Returns off, slope."""
    n = (b_hi - b_lo + (1 << s)) >> s
    off = np.zeros(n)
    slope = np.zeros(n)
    for i in range(n):
        xs = bf_from_bits(np.arange(b_lo + (i << s), b_lo + ((i + 1) << s)))
        ys = f(xs)
        sl = bf((ys[-1] - ys[0]) / (xs[-1] - xs[0])) if len(xs) > 1 else 0.0
        r = ys - sl * xs
        off[i] = np.float32(0.5 * (r.max() + r.min()))
        slope[i] = sl
    return off, slope


def grid_eval(x, off, slope, b_lo, s, below):
    """x >= 0 bf16; below b_lo -> `below`"""
    b = bf_bits(x).astype(np.int64)
    i = np.clip((b - b_lo) >> s, 0, len(off) - 1)
    v = off[i].astype(np.float32).astype(np.float64) + slope[i] * x
    return np.where(b < b_lo, below, v)


def exp_grid(u, s):
    """a = exp(-u), u = -log_a >= 0 (bf16): grid table on |x| in [2^-10, 19.0]"""
    b_lo, b_hi = int(bf_bits(np.float32(2.0 ** -10))), int(bf_bits(np.float32(19.0)))
    off, slope = grid_table(lambda t: np.exp(-t), b_lo, b_hi, s)
    return grid_eval(u, off, slope, b_lo, s, 1.0), len(off)


def sqrt_grid(w, s):
    """exact-on-grid sqrt: w = m 2^e (bf16), table on m' = 1.m x 2^(e & 1) in [1, 4), result times 2^((e - 127) >> 1)"""
    b = bf_bits(w).astype(np.int64)
    e = (b >> 7) & 0xFF
    par = e & 1  # bf16 bias 127 is odd: e odd <-> unbiased exponent even
    mbits = np.where(par == 1, 0x3F80, 0x4000) | (b & 0x7F)  # m in [1, 2) (even unbiased exp) or [2, 4)
    m = bf_from_bits(mbits)
    grid = np.concatenate([np.arange(0x3F80, 0x4000), np.arange(0x4000, 0x4080)])  # [1, 2) then [2, 4)
    off = np.zeros((256 >> s))
    slope = np.zeros_like(off)
    for i in range(len(off)):
        xs = bf_from_bits(grid[i << s:(i + 1) << s])
        ys = np.sqrt(xs)
        sl = bf((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
        r = ys - sl * xs
        off[i] = np.float32(0.5 * (r.max() + r.min()))
        slope[i] = sl
    idx = ((mbits - 0x3F80) & 0xFF) >> s
    v = bf(off[idx].astype(np.float32).astype(np.float64) + slope[idx] * m)  # RNE of the PWL value: exact bf16 mantissa
    k = ((e - 127) - (1 - par)) // 2  # 2^(e-127) = 2^(2k) * (1 or 2)
    out = v * np.power(2.0, k)
    return np.where(w > 0, out, 0.0), len(off)


def part2(log_a, a_r, a2_r, mult_r, near1):
    print("\n-- grid-indexed exp tables (index = bf16 bits of -log_a, 2^s grid points per segment) --")
    u = -log_a
    for s in (2, 3, 4, 5):
        ag, n = exp_grid(u, s)
        a2g, _ = exp_grid(2.0 * u, s)
        a = bf(ag)
        a2 = bf(a2g)
        mult = bf(np.sqrt(bf(1.0 - a2)))
        print(f"  s={s}: {n:4d} segments ({16 * n} B per copy): a {mism(a_r, a)}  a>=0.95 {mism(a_r, a, near1)}  "
              f"a2 {mism(a2_r, a2)}  mult {mism(mult_r, mult)}")
        # exhaustive over every bf16 u in [2^-10, 19]
        allb = np.arange(int(bf_bits(np.float32(2.0 ** -10))), int(bf_bits(np.float32(19.0))) + 1)
        ua = bf_from_bits(allb)
        ref = bf(np.exp(-ua))
        got = bf(exp_grid(ua, s)[0])
        bad = ref != got
        print(f"        exhaustive: {bad.mean():.4f} of {ua.size} grid inputs mis-rounded; largest u exact up to "
              f"{ua[np.argmax(bad)] if bad.any() else ua[-1]:.4g}; a>=0.95 region mis-rounded "
              f"{bad[ref >= 0.95].mean():.4f}")
    print("\n-- sqrt by grid table on 1.m x 2^(e&1) (exhaustive over every positive bf16 in [2^-20, 1]) --")
    allw = bf_from_bits(np.arange(int(bf_bits(np.float32(2.0 ** -20))), int(bf_bits(np.float32(1.0))) + 1))
    for s in (1, 2, 3, 4, 5):
        got, n = sqrt_grid(allw, s)
        ref = bf(np.sqrt(allw))
        print(f"  s={s}: {n:3d} segments ({16 * n} B per copy): mis-rounded {np.mean(ref != got):.5f}")
    om = bf(1.0 - a2_r)
    got, _ = sqrt_grid(om, 3)
    print(f"  on the real om (s=3): mult {mism(mult_r, got)}")




def part3(log_a, a_r, mult_r, sg_r, gx_s1=None):
    """absolute-error yardstick: rms(design - exact) / rms(torch_bf16 - exact), the quantisation ratio (1.0 = as
    good as torch bf16 itself) and max |error|, for a and mult; a2 from the fp32 table value squared"""
    ex_a = np.exp(log_a)
    ex_m = np.sqrt(np.maximum(1.0 - np.exp(2.0 * log_a), 0.0))
    qa = np.sqrt(np.mean((a_r - ex_a) ** 2))
    qm = np.sqrt(np.mean((mult_r - ex_m) ** 2))
    print("\n-- absolute-error ratios (1.0 = torch bf16 itself) --")
    for s in (3, 4, 5):
        af, _ = exp_grid(-log_a, s)
        a = bf(af)
        ah = bf(af)
        al = bf(af - ah)
        a2 = bf(ah * ah + ah * bf(2 * al))
        om = bf(1.0 - a2)
        mult, _ = sqrt_grid(om, 3)
        print(f"  grid s={s}, a2=(fp32 a)^2, exact sqrt: a ratio {np.sqrt(np.mean((a - ex_a) ** 2)) / qa:.3f} max|da| "
              f"{np.abs(a - a_r).max():.2e}; a>=0.95 mism {np.mean(a[a_r >= 0.95] != a_r[a_r >= 0.95]):.4f}; "
              f"mult ratio {np.sqrt(np.mean((mult - ex_m) ** 2)) / qm:.3f} max|dm| {np.abs(mult - mult_r).max():.2e} "
              f"mism {np.mean(mult != mult_r):.4f}")


def part4(ga, gx, sp, sgr, a_r, mult_r, gxs):
    """N1 end to end: sigma by the symmetric table (w = 1/8, clamp at -R), log_a = RN(sigma * (-8 sp)) (bit-identical
    fold), a by the grid table (s = 4), a2 = RN((fp32 a)^2), om = 1 - a2, mult by the exact sqrt table (32 segments)"""
    print("\n-- N1 end to end (real gates, all blocks) --")
    near1 = a_r >= 0.95
    for R in (10.0, 12.0):
        sa = bf(sig_s1(ga, R, 1 / 8))
        sx = bf(sig_s1(gx, R, 1 / 8))
        la = bf(sa * bf(-8.0 * sp))
        af, _ = exp_grid(-la, 4)
        a = bf(af)
        ah = bf(af)
        al = bf(af - ah)
        a2 = bf(ah * ah + ah * bf(2 * al))
        mult, _ = sqrt_grid(bf(1.0 - a2), 3)
        print(f"  R={R:4.1f}: sigma_x mism {np.mean(sx != gxs):.4f}; a mism {np.mean(a != a_r):.4f} (a>=0.95: "
              f"{np.mean(a[near1] != a_r[near1]):.4f}, bias {np.mean((a - a_r)[near1]):+.1e}); a==1 agree "
              f"{np.mean((a == 1) == (a_r == 1)):.5f}; mult mism {np.mean(mult != mult_r):.4f} "
              f"(0 vs nonzero {np.mean((mult == 0) != (mult_r == 0)):.4f}); max|da| {np.abs(a - a_r).max():.2e}")


if __name__ == "__main__":
    sys.exit(main())
