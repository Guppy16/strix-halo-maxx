"""RG-LRU transcendentals on real activations (round 9): the deployed kernel against table candidates.

usage (npu-iron-torch): python3 r9/ssmemu.py [--acts build/acts_clip3.npz] [--blocks 0,5,11]

Inputs: the real gate pre-activations of r6/capture.py (rglru_gx_b*, rglru_ga_b*: [rows, 768] bf16) and softplus(a_param)
per channel from the safetensors. Reference = torch's own bf16 ops (the CPU model). For each formulation it prints the
bf16 mismatch rate (!= torch) and ulp p99 / max of gate_x, gate_a, a, mult, and two state metrics:
  dfix  rel. change of the static fixed point h* = mult / (1 - a) (a constant input held forever: worst-case
        compounding of a and mult errors), median / p99 over elements where it is defined
  da1   |a - a_ref| / (1 - a_ref) on the unrounded a (round 6's tail metric), p99 / max where 1 - a_ref > 2^-10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r6"))
import emu  # noqa: E402
import lutgen  # noqa: E402
from emu import F32, M0, L_HI, L_LO, bf, bits16, c_bf, from16, mac, msc, mul, split  # noqa: E402


# ---- the deployed kernel (nk8.h), op by op --------------------------------------------------------------------
def exp32a_i(d):
    d = np.clip(np.asarray(d, F32), F32(-60), F32(60))
    za = mul(d, L_HI)
    za = mac(za, d, L_LO)
    zn = mac(za, F32(1), M0)
    nb = bf(mac(zn, F32(1), -M0))
    f = bf(mac(za, nb, F32(-1)))
    n16 = (zn.view(np.int32) & 0xFFFF).astype(np.int16).astype(np.int32)
    p2n = from16((n16 << 7) + 0x3F80)
    f2 = bf(mul(f, f))
    f3 = bf(mul(f2, f))
    p = mac(F32(1.0), f, c_bf(0.69314718))
    p = mac(p, f2, c_bf(0.24022651))
    p = mac(p, f3, c_bf(0.05550411))
    ph, pl = split(p)
    return mac(mul(ph, p2n), pl, p2n)


def a_om_nk8(la):
    """exp_a2_n: a = bf(exp(la)), om = bf(1 - bf(a^2)) with a^2 from the fp32 a"""
    a = exp32a_i(la)
    ah, al = split(a)
    a2 = bf(mac(mul(ah, ah), ah, bf(mul(al, F32(2.0)))))
    return bf(a), bf(msc(F32(1.0), a2, F32(1.0))), a


def sqrt_nk8(w):
    w = np.asarray(w, F32)
    wc = np.maximum(w, bf(F32(1e-30)))
    r = from16(0x5F37 - (bits16(wc) >> 1))
    hw = bf(mul(wc, F32(0.5)))
    for _ in range(2):
        rr = bf(mul(r, r))
        t = bf(msc(F32(1.5), hw, rr))
        r = bf(mul(r, t))
    return bf(mul(w, r))


def sig_nk8(z):
    return bf(emu.sigmoid32_i(z))


# ---- candidates ------------------------------------------------------------------------------------------------
def pwl(fn, lo, hi, step_bits, rel=False):
    """per-segment line (bf16 slope, fp32 offset) on [lo, hi) with segments of 2^step_bits; x clamped into the table.
    rel: fit the slope by weighted least squares on the relative error (tails of exponentials)"""
    w = 2.0 ** step_bits
    n = int(round((hi - lo) / w))
    sl, of = [], []
    for i in range(n):
        xs = np.linspace(lo + i * w, lo + (i + 1) * w, 513)[:-1]
        y = fn(xs)
        wt = 1 / np.maximum(np.abs(y), 1e-30) if rel else np.ones_like(y)
        a, b = np.polyfit(xs, y, 1, w=wt)
        a = float(np.float32(a).astype(bfloat16))
        # offset: minimise the max relative (or absolute) error for the rounded slope
        r = (y - a * xs) if not rel else (y - a * xs)
        if rel:
            # b minimising max |(a x + b - y) / y|: bisect on the weighted residual
            lo_b, hi_b = float(np.min(r)), float(np.max(r))
            for _ in range(60):
                m = 0.5 * (lo_b + hi_b)
                e = (a * xs + m - y) * wt
                if e.max() + e.min() > 0:
                    hi_b = m
                else:
                    lo_b = m
            b = 0.5 * (lo_b + hi_b)
        else:
            b = 0.5 * (r.min() + r.max())
        sl.append(a)
        of.append(np.float32(b))
    sl = np.array(sl, F32)
    of = np.array(of, F32)
    xhi = F32(bfloat16(hi - w / 2))

    def ev(x):
        xc = np.clip(np.asarray(x, F32), F32(lo), xhi)
        i = np.clip(np.floor((xc.astype(np.float64) - lo) / w).astype(np.int64), 0, n - 1)
        return mac(of[i], sl[i], xc)
    ev.n = n
    ev.sl, ev.of, ev.lo, ev.hi, ev.step = sl, of, lo, hi, step_bits
    return ev


def s_sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def sig_table(lo, hi, step_bits, rel=True):
    t = pwl(s_sig, lo, hi, step_bits, rel)
    f = lambda z: bf(t(z))
    f.n = t.n
    f.tab = t
    return f


def exp2_rr_table(nseg):
    """2^f on f in [-0.5, 0.5) with nseg segments, relative fit"""
    return pwl(lambda f: np.exp2(f), -0.5, 0.5, -int(np.log2(nseg)), rel=True)


def a_om_rr(nseg):
    """a = exp(la) by range reduction: z = la log2e (fp32), n = round(z), f = z - n as hi/lo bf16, p = T(f_hi) with
    the line also applied to f_lo; a = bf(p) 2^n; e2 = bf(p^2) 2^2n from p hi/lo; om = bf(1 - e2)"""
    T = exp2_rr_table(nseg)

    def fn(la):
        la = np.asarray(la, F32)
        za = mul(la, L_HI)
        za = mac(za, la, L_LO)
        zn = mac(za, F32(1), M0)
        nb = bf(mac(zn, F32(1), -M0))
        fa = mac(za, nb, F32(-1))
        fh, fl = split(fa)
        n = (zn.view(np.int32) & 0xFFFF).astype(np.int16).astype(np.int32)
        xc = np.clip(fh, F32(-0.5), F32(bfloat16(0.5 - 1 / nseg / 2)))
        i = np.clip(np.floor((xc.astype(np.float64) + 0.5) * nseg).astype(np.int64), 0, nseg - 1)
        p = mac(T.of[i], T.sl[i], fh)
        p = mac(p, T.sl[i], fl)
        ph, pl = split(p)
        a = from16(bits16(ph) + (n << 7))
        p2 = bf(mac(mul(ph, ph), ph, bf(mul(pl, F32(2.0)))))
        e2 = from16(bits16(p2) + (n << 8))
        om = bf(msc(F32(1.0), e2, F32(1.0)))
        a_full = (ph.astype(np.float64) + pl) * np.exp2(n)
        return a, om, a_full
    fn.n = nseg
    return fn


def a_om_direct(lo, step_bits, rel=True):
    """a = T(la), e2 = T(2 la) from one exp table on [lo, 0) (la = bf16 log_a, 2 la exact); om = bf(1 - bf(e2))"""
    T = pwl(np.exp, lo, 0.0, step_bits, rel)

    def fn(la):
        la = np.asarray(la, F32)
        pa = T(la)
        e2 = bf(T(bf(mul(la, F32(2.0)))))
        return bf(pa), bf(msc(F32(1.0), e2, F32(1.0))), pa
    fn.n = T.n
    return fn


def sqrt_exact(w):
    """exact sqrt by a 256-entry table on (exponent parity, mantissa): RNE(sqrt) for normal w > 0, 0 -> 0"""
    w = np.asarray(w, F32)
    b = bits16(w)
    E = (b >> 7) & 0xFF
    idx = b & 0xFF  # exponent low bit + mantissa
    # entry: bf16 bits of sqrt(2^(1 - (E & 1)) * M) for E's parity, M = 1.mant
    m = 1 + (idx & 0x7F) / 128.0
    par = (idx >> 7) & 1
    tab = bits16(F32(np.sqrt(np.where(par == 1, 1.0, 2.0) * m)))  # E odd: e = E - 127 even
    e = E - 127
    k = np.floor_divide(e, 2)  # floor(e / 2)
    out = from16(tab + (k << 7))
    return np.where(w > 0, out, F32(0.0))


# ---- reference: torch ops, as the CPU model computes them ------------------------------------------------------
def torch_ref(gx, ga, sp):
    import torch
    t = lambda v: torch.from_numpy(np.asarray(v, F32)).to(torch.bfloat16)
    gxt, gat, spt = t(gx), t(ga), t(sp)
    gate_x = torch.sigmoid(gxt)
    gate_a = torch.sigmoid(gat)
    log_a = -8.0 * gate_a * spt
    a = torch.exp(log_a)
    e2 = torch.exp(2 * log_a)
    om = 1 - e2
    mult = torch.sqrt(om)
    f = lambda v: v.float().numpy()
    return dict(gate_x=f(gate_x), gate_a=f(gate_a), log_a=f(log_a), a=f(a), om=f(om), mult=f(mult))


def chain(gx, ga, sp, sig, a_om, sqrt):
    gate_x = sig(gx)
    gate_a = sig(ga)
    la = bf(mul(bf(mul(gate_a, F32(-8.0))), sp))
    a, om, a_full = a_om(la)
    mult = sqrt(om)
    return dict(gate_x=gate_x, gate_a=gate_a, log_a=la, a=a, om=om, mult=mult, a_full=a_full)


def stats(got, ref, name):
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    mism = np.mean(g != r)
    nz = np.abs(r) > 1e-4  # the clamped sigmoid tails (|ulp| huge, values negligible) stay out of the stats
    u = emu.ulp_err(g[nz], r[nz])
    sgn = np.sign(g[nz] - r[nz]) * u  # signed ulps: a bias shows as a mean far from 0
    return f"{name} {mism:6.2%} p99 {np.percentile(u, 99):4.1f} bias {np.mean(sgn):+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="/work/npu-tapnext/build/acts_clip3.npz")
    ap.add_argument("--blocks", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--cands", default="")
    o = ap.parse_args()
    import torch
    from safetensors.numpy import load_file
    sd = load_file("/repo/data/models/tapnextpp_256.safetensors")
    acts = np.load(o.acts)
    blocks = [int(b) for b in o.blocks.split(",")]
    data = []
    for b in blocks:
        ap_ = torch.from_numpy(sd[f"blocks.{b}.ssm_block.recurrent_block.rg_lru.a_param"].astype(F32)).to(torch.bfloat16)
        sp = torch.nn.functional.softplus(ap_).float().numpy()  # bf16 softplus, as the bf16 model
        data.append((bf(acts[f"rglru_gx_b{b}"]), bf(acts[f"rglru_ga_b{b}"]), np.broadcast_to(sp, acts[f"rglru_gx_b{b}"].shape)))
    gx = np.concatenate([d[0].reshape(-1) for d in data])
    ga = np.concatenate([d[1].reshape(-1) for d in data])
    sp = np.concatenate([d[2].reshape(-1) for d in data]).astype(F32)
    print(f"{gx.size} elements; gx [{gx.min():.1f}, {gx.max():.1f}] ga [{ga.min():.1f}, {ga.max():.1f}] sp [{sp.min():.4f}, {sp.max():.4f}]")
    ref = torch_ref(gx, ga, sp)
    a_ref_full = np.exp(ref["log_a"].astype(np.float64))
    cands = {
        "nk8 (deployed)": (sig_nk8, a_om_nk8, sqrt_nk8),
        "nk8 + exact sqrt": (sig_nk8, a_om_nk8, sqrt_exact),
        "sigT[-12,8)/8 + nk8": (sig_table(-12, 8, -3), a_om_nk8, sqrt_exact),
        "sigT[-12,8)/16 + nk8": (sig_table(-12, 8, -4), a_om_nk8, sqrt_exact),
        "sigT[-10,7)/8 + nk8": (sig_table(-10, 7, -3), a_om_nk8, sqrt_exact),
        "sigT[-12,8)/4 + nk8": (sig_table(-12, 8, -2), a_om_nk8, sqrt_exact),
        "nk8 sig + rr32": (sig_nk8, a_om_rr(32), sqrt_exact),
        "nk8 sig + rr64": (sig_nk8, a_om_rr(64), sqrt_exact),
        "nk8 sig + direct[-20,0)/8": (sig_nk8, a_om_direct(-20, -3), sqrt_exact),
        "nk8 sig + direct[-20,0)/16": (sig_nk8, a_om_direct(-20, -4), sqrt_exact),
        "sigT/8 + rr32": (sig_table(-12, 8, -3), a_om_rr(32), sqrt_exact),
        "sigT/8 + rr64": (sig_table(-12, 8, -3), a_om_rr(64), sqrt_exact),
    }
    if o.cands:
        keep = o.cands.split(";")
        cands = {k: v for k, v in cands.items() if any(s in k for s in keep)}
    for name, (sig, a_om, sq) in cands.items():
        r = chain(gx, ga, sp, sig, a_om, sq)
        da = np.abs(r["a_full"] - a_ref_full) / (1 - a_ref_full)
        ok = (1 - a_ref_full) > 2.0 ** -10
        fixr = ref["mult"].astype(np.float64) / np.maximum(1 - ref["a"].astype(np.float64), 1e-30)
        fixg = r["mult"].astype(np.float64) / np.maximum(1 - r["a"].astype(np.float64), 1e-30)
        dok = (ref["a"] < 1) & (r["a"] < 1)
        dfix = np.abs(fixg[dok] - fixr[dok]) / np.maximum(np.abs(fixr[dok]), 1e-30)
        parts = [stats(r[k], ref[k], k) for k in ("gate_x", "gate_a", "a", "mult")]
        print(f"{name:28s} | " + " | ".join(parts) +
              f" | da1 p99 {np.percentile(da[ok], 99):.1e} max {da[ok].max():.1e} | dfix p50 {np.median(dfix):.1e} p99 {np.percentile(dfix, 99):.1e}"
              f" | a==1 flips {np.mean((r['a'] == 1) != (ref['a'] == 1)):.3%}")


if __name__ == "__main__":
    main()
