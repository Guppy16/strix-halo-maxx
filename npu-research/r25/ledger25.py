"""Pass 20: the r23z ledger by segment of the block's critical path, each against its bound.

advance = frontier advance per op (r25/advance.py, 4 r23z d12 timelines); bytes per op from r21/ddrbudget.py (plan8, r23z build);
compute: k-steps per op per core = nt x K / 64 (1,296 per block) at the in-situ KTL k-step; EW ops from tcall / ANALYSIS rev 18.
DDR: aggregate read cap 55-59 GB/s (round 19 DDRCAP, MEASURED: 1 or 2 streams per column cap there); reads + writes reach ~90 GB/s
(norm0's own span: 8.1 MB in 90.4 us). Every EW op reads through ONE shim MM2S channel per column (channel 0, shared with the GEMM
A loads: R15.2), ~3.8 B / cycle (R15.3) = 6.9 GB/s per column: about the same limit as the aggregate cap over 8 columns.
usage: python3 r25/ledger25.py   (numbers are inline: re-run advance.py / ddrbudget.py on new timelines and edit ADV / BYTES)
"""

GHZ = 1.81
KSTEP = (1447, 1411, 1304)  # in-situ k-step now (KTL R 70), with a bank-clean R 68 (-36), at the A-port floor (R17.4)
EPI = dict(plain=1450, gated=3263, gelu=3000, gelu_pan=5000, qkv=2000)  # cycles per tile (TPROF r21t2 / r19, KTL builds)
# us per block (r23z, advance.py over tl_full12r23z_d12*.json)
ADV = dict(norm0=93.0, yx=183.4, ssm=151.0, out=52.0, norm1=78.8, up_a=199.2, up_b=191.6, down_a=80.4, down_b=101.7,
           norm2=72.5, qkv=154.6, attn=493.4, out_proj=75.4, norm3=76.1, mlp0=226.7, mlp3_a=80.3, mlp3_b=96.1)
FINAL = 91.7
# MB per block: (read, write)
BYTES = dict(norm0=(5.333, 2.765), yx=(2.433, 3.539), ssm=(13.566, 5.308), out=(1.714, 1.769), norm1=(3.564, 2.765),
             up_a=(3.870, 3.539), up_b=(2.875, 3.539), down_a=(3.373, 1.769), down_b=(3.373, 1.769), norm2=(5.358, 2.765),
             qkv=(3.152, 5.308), attn=(6.303, 3.539), out_proj=(1.714, 1.769), norm3=(3.588, 2.765), mlp0=(3.870, 7.078),
             mlp3_a=(3.373, 1.769), mlp3_b=(3.373, 1.769))
GEMM = dict(yx=(8, 12, "gelu"), out=(4, 12, "plain"), up_a=(16, 12, "gated"), up_b=(16, 12, "gated"), down_a=(4, 24, "plain"),
            down_b=(4, 24, "plain"), qkv=(12, 12, "qkv"), out_proj=(4, 12, "plain"), mlp0=(16, 12, "gelu_pan"),
            mlp3_a=(4, 24, "plain"), mlp3_b=(4, 24, "plain"))
SSM_UNITS = 216  # per core per block (1 cycle per unit = 1.43 us per step)
CH0 = 3.8  # B / cycle, one shim MM2S channel (R15.3)
SSM_COL = 12 * (12288 + 129024)  # B per column per block on channel 0: per head 12,288 of params + 129,024 of y / x (plan8 r23z)
YX_A_COL = 995328 // 8  # yx's A load per column, the same channel


def us(cyc):
    return cyc / GHZ / 1e3


def gemm_floor(op, ks):
    nt, nk, epi = GEMM[op]
    return us(nt * nk * ks + nt * EPI[epi])


def ddr_floor(ops, rd=(59.0, 55.0), rw=(90.0, 84.0)):
    r = sum(BYTES[o][0] for o in ops)
    w = sum(BYTES[o][1] for o in ops)
    return max(r / rd[0], (r + w) / rw[0]) * 1e3, max(r / rd[1], (r + w) / rw[1]) * 1e3


SEGS = [  # (segment, ops, floor (lo, hi) us per block, why)
    ("norm0 (+ mlp3_b pass 2)", ["norm0"], None, "DDR read cap (= channel 0)"),
    ("yx + SSM (YXSSM / YXR)", ["yx", "ssm"], "ssm", "DDR read cap in the SSM span (channel 0 and compute just under)"),
    ("out (tiles 1-3 + drain after the SSM)", ["out"], (20.0, 30.0), "tile 0 follows the heads (ZCH); tiles 1-3 need all"),
    ("norm1", ["norm1"], None, "DDR read cap (= channel 0)"),
    ("up_a + up_b", ["up_a", "up_b"], "gemm", "k-steps + gated epilogues"),
    ("down_a + down_b + norm2", ["down_a", "down_b", "norm2"], "down", "k-steps + norm2 DDR (MSPLIT-P overlaps ~13)"),
    ("qkv", ["qkv"], "gemm", "k-steps + qkv epilogues"),
    ("attention", ["attn"], (470.0, 490.0), "compute (rev 18 floor ~5.7-5.9 ms per step)"),
    ("out_proj", ["out_proj"], "gemm", "k-steps; A after attention (OPPRE)"),
    ("norm3", ["norm3"], None, "LN compute (STATS64 r24e -0.10) vs DDR"),
    ("mlp0", ["mlp0"], "gemm", "k-steps + gelu_pan epilogues (LUT)"),
    ("mlp3_a + mlp3_b", ["mlp3_a", "mlp3_b"], "gemm", "k-steps + plain epilogues"),
]


def main():
    tot = dict(now=0.0, lo=0.0, hi=0.0)
    print(f"{'segment':42s} {'now':>6s} {'floor lo-hi':>13s}   ms/step now / floor  bound")
    for name, ops, fl, why in SEGS:
        now = sum(ADV[o] for o in ops)
        if fl is None:
            lo, hi = ddr_floor(ops)
        elif fl == "gemm":
            lo = sum(gemm_floor(o, KSTEP[2]) for o in ops)
            hi = sum(gemm_floor(o, KSTEP[1]) for o in ops)
        elif fl == "down":
            d0, d1 = ddr_floor(["norm2"])
            lo = gemm_floor("down_a", KSTEP[2]) + gemm_floor("down_b", KSTEP[2]) + d0 - 13
            hi = gemm_floor("down_a", KSTEP[1]) + gemm_floor("down_b", KSTEP[1]) + d1 - 13
        elif fl == "ssm":
            # the SSM from yx's first group (~20 us at best, 32 now), bound by the DDR read cap over its span: its own 13.57 MB
            # (lo: yx's reads all outside the span) or + yx's ~2.0 MB inside it (hi, today); static compute (2,115 per unit, 252 us)
            # and channel 0 (2,066 per unit) sit just under it
            lo = 20 + max(us(SSM_UNITS * 2115), BYTES["ssm"][0] / 59.0 * 1e3)
            hi = 32 + (BYTES["ssm"][0] + 2.0) / 57.0 * 1e3
        else:
            lo, hi = fl
        lo, hi = min(lo, now), min(hi, now)
        tot["now"] += now
        tot["lo"] += lo
        tot["hi"] += hi
        print(f"{name:42s} {now:6.1f} {lo:6.1f}-{hi:6.1f}   {now * 12e-3:5.2f} / {lo * 12e-3:5.2f}-{hi * 12e-3:5.2f}  {why}")
    f = FINAL * 1e-3
    print(f"{'total (+ final norm once)':42s} {tot['now']:6.1f} {tot['lo']:6.1f}-{tot['hi']:6.1f}   "
          f"{tot['now'] * 12e-3 + f:5.2f} / {tot['lo'] * 12e-3 + f:5.2f}-{tot['hi'] * 12e-3 + f:5.2f}")
    r = sum(b[0] for b in BYTES.values())
    w = sum(b[1] for b in BYTES.values())
    print(f"block bytes: read {r:.1f} MB, write {w:.1f} MB; step DDR floor {r * 12 / 59:.1f}-{r * 12 / 55:.1f} ms (read cap)")
    # the norm0 .. norm1 stretch (EW-heavy): DDR vs the compute chain
    st = ["norm0", "yx", "ssm", "out", "norm1"]
    d_lo, d_hi = ddr_floor(st)
    print(f"SSM per column on channel 0: {SSM_COL} B ({us(SSM_COL / CH0):.0f} us at {CH0} B / cycle; + yx A {us(YX_A_COL / CH0):.0f} us)"
          f"; per unit {SSM_COL / SSM_UNITS / CH0:.0f} cycles vs static compute 2,115, in situ 2,416 (r23z) / 2,336 (r22t)")
    print(f"stretch norm0..norm1: now {sum(ADV[o] for o in st):.0f} us / block; DDR floor {d_lo:.0f}-{d_hi:.0f} us")


if __name__ == "__main__":
    main()
