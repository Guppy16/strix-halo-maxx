"""Numerics harness on real activations (round 6, npu-research P8 parts 2-3).

usage: python3 r6/numh.py ranges|gelu [--acts build/acts_clip3.npz] [--cands c0,c1,...]

ranges: per site, the real value range against each kernel's limits (margins, counts beyond).
gelu:   GELU candidate formulations (bit-level emulations in r6/emu.py style) on the real GELU inputs of all
        three sites, with two yardsticks:
          per element  vs RNE(exact) of the site's reference: mismatch %, ulp p99 / max
          bfp ratio    rms(approx - exact) / rms(bfp16(exact) - exact), the bfp16 quantisation that every GELU
                       output meets next (gated: on the product with `up`). < ~0.3 means the approximation error
                       hides under the quantisation the step applies anyway.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emu  # noqa: E402
from emu import F32, bf, c_bf, mac, msc, mul, split  # noqa: E402


def bfp16(x):
    """host model of bfp16ebs8 along the last axis (blocks of 8): shared exponent = the block max's, int8 mantissas
    RNE, saturating at 127 (pack.encode semantics; the device's own conversion differs on ~2.5% of panels)"""
    x = np.asarray(x, np.float64)
    b = x.reshape(*x.shape[:-1], -1, 8)
    mx = np.max(np.abs(b), axis=-1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, 1e-38)))
    scale = 2.0 ** (e - 6)
    m = np.clip(np.round(b / scale), -128, 127)
    return (m * scale).reshape(x.shape)


def load(path):
    return np.load(path)


def site_arrays(acts, site):
    keys = sorted((k for k in acts.files if k.startswith(site + "_b")), key=lambda k: int(k.rsplit("_b", 1)[1]))
    return [(int(k.rsplit("_b", 1)[1]), acts[k]) for k in keys]


# ---- ranges ---------------------------------------------------------------------------------------------------
def ranges(o):
    acts = load(o.acts)
    lim = dict(ssm_gelu=("gelu clamp |x|<=8", 8.0), ffw_gate=("gelu clamp |x|<=8", 8.0), vit_gelu=("gelu clamp |x|<=8", 8.0),
               rglru_gx=("exp32 clamp |z|<=60", 60.0), rglru_ga=("exp32 clamp |z|<=60", 60.0))
    for site, (what, lv) in lim.items():
        allv = np.concatenate([a.ravel() for _, a in site_arrays(acts, site)])
        p = np.percentile(np.abs(allv), [50, 99, 99.99])
        print(f"{site:9s} n {allv.size:9d} min {allv.min():8.3f} max {allv.max():8.3f} |x| p50 {p[0]:.3f} p99 {p[1]:.3f} "
              f"p99.99 {p[2]:.3f}  {what}: beyond {np.mean(np.abs(allv) > lv):.2e} (margin {lv / np.abs(allv).max():.2f}x)")
    # attention: s - rowmax (exact max) and the lazy-max headroom
    for b, s in site_arrays(acts, "attn_s")[:: max(1, o.every)]:
        s = s[..., :1072]
        d = s - s.max(-1, keepdims=True)
        print(f"attn b{b:2d}: s range [{s.min():7.2f}, {s.max():7.2f}], row max spread {np.ptp(s.max(-1)):6.2f}, "
              f"s - max p1 {np.percentile(d, 1):7.2f} min {d.min():7.2f} (expd floor n >= -120 <-> -83)")


# ---- GELU candidates --------------------------------------------------------------------------------------------
def gelu_c0(x):
    """the deployed kernel (nk8 gelu32_i -> bf16)"""
    return emu.gelu_n(x)


def _u_split(x):
    """u = x (c0 + c1 x^2) with the deployed hi/lo splits: returns (uh, ul)"""
    c1h = c_bf(0.035677408)
    c1l = c_bf(0.035677408 - float(c1h))
    th, tl = split(mul(x, x))
    wa = mac(F32(0.7978845608), th, c1h)
    wa = mac(wa, th, c1l)
    wa = mac(wa, tl, c1h)
    wh, wl = split(wa)
    ua = mac(mul(x, wh), x, wl)
    return split(ua)


def _u_cheap(x):
    """u with one split fewer: w = bf16(c0 + c1 bf16(x^2)) (c1 hi only), ua = x w"""
    t = bf(mul(x, x))
    w = bf(mac(F32(0.7978845608), t, c_bf(0.035677408)))
    return split(mul(x, w))


def _exp2q(d, dl):
    """exp(d + dl) with a quadratic 2^f (expd_i's polynomial: 1 + f (q1 + q2 f)) and 2^n added to the bits"""
    d = np.clip(d, F32(-60), F32(60))
    za = mul(d, emu.L_HI)
    za = mac(za, d, emu.L_LO)
    za = mac(za, dl, emu.L_HI)
    zn = mac(za, F32(1), emu.M0)
    nb = bf(mac(zn, F32(1), -emu.M0))
    f = bf(mac(za, nb, F32(-1)))
    n16 = (zn.view(np.int32) & 0xFFFF).astype(np.int16).astype(np.int32)
    t = bf(mac(F32(0.703125), f, c_bf(0.240234375)))
    p = bf(mac(F32(1.0), f, t))
    return emu.from16(emu.bits16(p) + (np.maximum(n16, -120) << 7))


def _recip1(d):
    """1/d with the bit-trick seed and ONE hi/lo Newton step"""
    dh, dl = split(d)
    r0 = emu.from16(0x7EF3 - emu.bits16(dh))
    eps = msc(F32(1.0), dh, r0)
    eps = msc(eps, dl, r0)
    return mac(mul(r0, F32(1.0)), r0, bf(eps))


def _gelu_from(x_in, uh, ul, expf, recipf):
    e = expf(bf(mul(uh, F32(-2.0))), bf(mul(ul, F32(-2.0))))
    s = recipf(mac(F32(1.0), e, F32(1.0)))
    sh, sl = split(s)
    return bf(mac(mul(x_in, sh), x_in, sl))


def gelu_c1(x):
    """c0 with one Newton step in the reciprocal"""
    xc = np.clip(x, F32(-8), F32(8))
    uh, ul = _u_split(xc)
    return _gelu_from(x, uh, ul, emu.exp32_i, _recip1)


def gelu_c2(x):
    """c0 with the quadratic 2^f exp"""
    xc = np.clip(x, F32(-8), F32(8))
    uh, ul = _u_split(xc)
    return _gelu_from(x, uh, ul, _exp2q, emu.recipa_i)


def gelu_c3(x):
    """cheap u + quadratic exp + one Newton step"""
    xc = np.clip(x, F32(-8), F32(8))
    uh, ul = _u_cheap(xc)
    return _gelu_from(x, uh, ul, _exp2q, _recip1)


def gelu_c4(x):
    """c3 with s used as bf16 hi only (g = x bf16(s): one mac less)"""
    xc = np.clip(x, F32(-8), F32(8))
    uh, ul = _u_cheap(xc)
    e = _exp2q(bf(mul(uh, F32(-2.0))), bf(mul(ul, F32(-2.0))))
    s = _recip1(mac(F32(1.0), e, F32(1.0)))
    return bf(mul(x, bf(s)))


def make_lut(lo, hi, step_bits, form="tanh"):
    """LUT gelu (kprobe.cc gelu_lut): s = slope[i] x + offset[i] from an aie::linear_approx table on [lo, hi) with
    segments of 2^step_bits, x clamped into the table; gelu = x s with s split hi/lo"""
    import lutgen
    sl, of = lutgen.fit(lutgen.s_gelu if form == "tanh" else lutgen.s_erf, lo, hi, step_bits)
    w = 2.0 ** step_bits

    def fn(x):
        x = np.asarray(x, F32)
        xc = np.clip(x, F32(lo), bf(F32(hi - 1.0 / 64)))
        i = np.clip(np.floor(xc.astype(np.float64) / w).astype(np.int64) + int(round(-lo / w)), 0, len(sl) - 1)
        s = mac(of[i], sl[i], xc)
        sh, sl_ = split(s)
        return bf(mac(mul(x, sh), x, sl_))
    fn.__doc__ = f"LUT s(x) ({form}), {len(sl)} segments of 2^{step_bits} on [{lo}, {hi})"
    return fn


CANDS = dict(c0=gelu_c0, c1=gelu_c1, c2=gelu_c2, c3=gelu_c3, c4=gelu_c4,
             lut8=make_lut(-4.0, 4.0, -3), lut8w=make_lut(-6.0, 4.0, -3), lut16w=make_lut(-6.0, 4.0, -4),
             lut4w=make_lut(-6.0, 4.0, -2), luterf8=make_lut(-4.0, 4.0, -3, "erf"))


def gelu(o):
    acts = load(o.acts)
    cands = o.cands.split(",")
    sites = [("ssm_gelu", emu.gelu_tanh), ("ffw_gate", emu.gelu_tanh), ("vit_gelu", None)]
    print(f"{'cand':5s} {'site':9s} {'mismatch':>9s} {'ulp p99':>8s} {'ulp max':>8s} {'rms rel':>9s} {'bfp ratio':>9s}   (all blocks pooled)")
    for c in cands:
        fn = CANDS[c]
        for site, ref in sites:
            mism, u99, umax, err2, ref2, q2 = [], [], [], 0.0, 0.0, 0.0
            for b, x in site_arrays(acts, site)[:: max(1, o.every)]:
                x = bf(x)
                exact = ref(x) if ref is not None else emu.gelu_erf(x) if o.erf_ok else emu.gelu_tanh(x)
                got = fn(x).astype(np.float64)
                if site == "ffw_gate":  # what the step quantises next: bf16(gelu(gate) * up), then bfp16
                    up = bf(acts[f"ffw_up_b{b}"]).astype(np.float64)
                    exact = exact * up
                    got = bf(got * up).astype(np.float64)
                rne = bf(exact).astype(np.float64)
                mism.append(np.mean(got != rne))
                big = np.abs(exact) > 1e-3
                u = emu.ulp_err(got[big], exact[big])
                u99.append(np.percentile(u, 99))
                umax.append(u.max())
                err2 += np.sum((got - exact) ** 2)
                ref2 += np.sum(exact ** 2)
                q2 += np.sum((bfp16(exact) - exact) ** 2)
            print(f"{c:5s} {site:9s} {np.mean(mism):9.2%} {np.max(u99):8.2f} {np.max(umax):8.1f} {np.sqrt(err2 / ref2):9.2e} "
                  f"{np.sqrt(err2 / q2):9.3f}   {CANDS[c].__doc__.strip().splitlines()[0]}")


def make_sig_lut(step_bits=-3):
    """LUT sigmoid on [0, 8): s = slope[i] |z| + offset[i] (|z| clamped), sigma(z) = z >= 0 ? s : 1 - s, bf16 out"""
    import lutgen
    sl, of = lutgen.fit(lutgen.s_sig, 0.0, 8.0, step_bits)
    w = 2.0 ** step_bits

    def fn(z):
        z = np.asarray(z, F32)
        a = np.minimum(np.abs(z), bf(F32(8.0 - 1.0 / 64)))
        i = np.clip(np.floor(a.astype(np.float64) / w).astype(np.int64), 0, len(sl) - 1)
        s = mac(of[i], sl[i], a)
        neg = msc(F32(1.0), bf(s), F32(1.0))  # 1 - s with s as hi/lo
        neg = msc(neg, bf(msc(s, bf(s), F32(1.0))), F32(1.0))
        return np.where(z >= 0, bf(s), bf(neg))
    return fn


def sigmoid(o):
    """RG-LRU gates: sigma of input_gate / a_gate pre-activations, then a = exp(-8 sigma(ga) softplus(a_param))"""
    import torch
    sys.path.insert(0, "/work/npu-tapnext")
    import step8
    model = step8.load_model(12)
    acts = load(o.acts)
    cands = dict(nk8=lambda z: bf(emu.sigmoid32_i(z)), lut8=make_sig_lut(-3), lut4=make_sig_lut(-2))
    for name, fn in cands.items():
        for site in ("rglru_gx", "rglru_ga"):
            mism, u99, umax, ae = [], [], [], []
            for b, z in site_arrays(acts, site)[:: max(1, o.every)]:
                z = bf(z)
                ex = 1 / (1 + np.exp(-z.astype(np.float64)))
                got = fn(z).astype(np.float64)
                rne = bf(ex).astype(np.float64)
                mism.append(np.mean(got != rne))
                u = emu.ulp_err(got[ex > 1e-4], ex[ex > 1e-4])
                u99.append(np.percentile(u, 99))
                umax.append(u.max())
                if site == "rglru_ga":  # torch: log_a = -8 bf16(sigma) softplus(a_param) in bf16; a = exp(log_a)
                    sp = torch.nn.functional.softplus(model.blocks[b].ssm_block.recurrent_block.rg_lru.a_param.detach().float()).to(torch.bfloat16).float().numpy()
                    def a_of(sg):
                        la = bf(bf(sg * F32(-8.0)) * sp)
                        return np.exp(la.astype(np.float64))
                    a_ref, a_got = a_of(rne.astype(F32)), a_of(got.astype(F32))
                    ae.append(np.max(np.abs(a_got - a_ref) / np.maximum(1 - a_ref, 1e-6)))
            extra = f"  max |da| / (1 - a) {max(ae):.2e}" if ae else ""
            print(f"{name:5s} {site:9s} != RNE {np.mean(mism):6.2%} ulp p99 {max(u99):5.2f} max {max(umax):6.2f}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ranges", "gelu", "sigmoid"])
    ap.add_argument("--acts", default="build/acts_clip3.npz")
    ap.add_argument("--cands", default="c0,c1,c2,c3,c4")
    ap.add_argument("--every", type=int, default=1, help="use every n-th block")
    ap.add_argument("--erf-ok", action="store_true", help="gelu: score vit_gelu against the erf GELU torch uses")
    o = ap.parse_args()
    dict(ranges=ranges, gelu=gelu, sigmoid=sigmoid)[o.cmd](o)


if __name__ == "__main__":
    main()
