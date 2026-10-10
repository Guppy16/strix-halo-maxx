"""sim17 (npu-research rev 10): the step model on the deployed full12r14f (47.31 ms at depth 12).

Attention: tools/attsim17.py (attsim's K / V multicast lockstep + head-group barrier, per-row-block costs per chunk class of
the round-14 driver, all one-core MEASURED: batched 261, first-pass rerun 393, exact test 164, rescale 578, exact rerun 393,
chunk 0 1,005, last chunk 358, V 136), on the CPU lazy-max map thinned to kappa = 0.6 (fitted: r14c - F128 2.24 vs 2.08-2.53
measured; per-block r14c / r14f corr 0.95, slope 0.98 / 1.05; independent check: in-situ slow units 8.7 + 1.36 x 21 / 23 =
9.9 % per core, inside the measured 8.6-10.9 %). Error bars = kappa 0.5 .. 0.7.
SSM: static per-chunk counts of the deployed EW elf (identical loops r12e -> r14f; `r15/loops_r14f_0_5.out`), 1 cycle per
chunk = 1.43 us per step (216 chunks x 12 blocks); savings transfer 1:1 (P29a precedent: measured = bench delta).
Rest: rev-8/9 prices (their code is unchanged), re-based on 47.31.
    python3 sim17.py            (numpy; ~3 min, OPENBLAS_NUM_THREADS=1)
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attsim17 as A  # noqa: E402

BASE = 47.31
CHUNK_US = 216 * 12 / 1.81e3 / 1e3  # ms per step per cycle per SSM chunk


def kappa_att(base, kappa, seed):
    rng = np.random.default_rng(seed)
    fl, mv = base.fl.copy(), base.mv.copy()
    keep = rng.random(fl.shape) < kappa
    keep[..., 0] = keep[..., 22] = True
    out = A.Att17.__new__(A.Att17)
    out.S, out.fl, out.mv = base.S, fl & keep, mv & keep
    return out


def g6():
    """P60g: dealt (col, slot position) -> natural (col, tile, half); G = {0,3,6} + {1,4,7}, pair {2, 5}"""
    m = {}
    for T in range(3):
        for e in range(2):
            for j in range(3):
                for H in range(2):
                    m[(3 * T + e, 2 * j + H)] = (3 * j + H, T, e)
    for T in range(3):
        m[(2, T)], m[(2, 3 + T)], m[(5, T)], m[(5, 3 + T)] = (2, T, 0), (5, T, 0), (2, T, 1), (5, T, 1)
    a = np.zeros((8, 3, 6, 3), int)
    for (c, p), (n, T, h) in m.items():
        t, hh = divmod(p, 2)
        for r in range(3):
            a[c, t, 3 * hh + r] = (n, T, 3 * h + r)
    return a


def attention_rows():
    raw = A.load()
    P = replace(A.P17(), VRB=136.0)
    G = g6()
    rows = [("P60g gather deal (DMA only)", True, {}),
            ("+ S1 rescale_rb in asm", True, dict(R=200.0)),
            ("+ S2 running row max in attkb (no first-pass rerun, test on 1 vector)", True, dict(R=200.0, FP=0.0, T=70.0, B=268.0)),
            ("+ S3 test + rescale as one asm call", True, dict(R=170.0, FP=0.0, T=40.0, B=268.0)),
            ("+ V1 row sums folded into attv, glue", True, dict(R=170.0, FP=0.0, T=40.0, B=268.0, VRB=110.0)),
            ("+ K1 batched call prep / flag scan in asm (B 268 -> 237)", True, dict(R=170.0, FP=0.0, T=40.0, B=237.0, VRB=110.0)),
            ("+ K2 chunk 0 as QK-only + test (C0 1,005 -> 600)", True, dict(R=170.0, FP=0.0, T=40.0, B=237.0, VRB=110.0, C0=600.0))]
    out = {}
    for kappa in (0.5, 0.6, 0.7):
        atts = [kappa_att(raw, kappa, s) for s in range(4)]
        b = np.mean([A.ms(a, P) for a in atts])
        out[kappa] = [np.mean([A.ms(A.remap(a, G) if d else a, replace(P, **o)) for a in atts]) - b for _, d, o in rows]
    return [(n, out[0.5][i], out[0.6][i], out[0.7][i]) for i, (n, _, _) in enumerate(rows)]


# SSM per chunk (cycles): deployed static -> generated target (low, high)
SSM = [("P29b log_a + exp chain (1,131: loga_p 23x8, exp_pA-D 20/12/6/4 x16, +275 calls)", 1131, 380, 470),
       ("P29c recurrence (ew_ssm 108 x 8 + 25)", 889, 210, 300),
       ("P29d conv (90 x 8 + 330) + z panels (25 x 8 + 44)", 1294, 600, 720),
       ("P29s sqrt (sqrt_lut_n 54 x 8 + 80)", 512, 180, 345),
       ("P52 out copies (798; outputs written in place)", 798, 0, 300)]
OTHER = [("P55 tile-granular A dependency", -0.36, 0.12), ("P7-E norm passes (port-bound)", -2.3, 0.8),
         ("P35 + P24b", -0.98, 0.3), ("P51 norm writes the next A slot", -0.96, 0.4), ("P57 C object in two halves", -0.5, 0.25)]

if __name__ == "__main__":
    t = BASE
    print(f"sim17 on full12r14f: {BASE:.2f} ms at depth 12\n\nAttention (attsim17, ms per step, cumulative; kappa 0.5 / 0.6 / 0.7):")
    att = attention_rows()
    for n, lo, mid, hi in att:
        print(f"  {lo:+6.2f} / {mid:+6.2f} / {hi:+6.2f}  {n}")
    t += att[-1][2]
    print(f"  -> {t:.2f} ms (attention 8.52 -> {8.52 + att[-1][2]:.2f} ms per step)")
    print("\nSSM (static, 1:1):")
    tot_lo = tot_hi = 0.0
    for n, dep, lo, hi in SSM:
        s_hi, s_lo = (dep - lo) * CHUNK_US, (dep - hi) * CHUNK_US
        tot_lo += s_lo
        tot_hi += s_hi
        print(f"  {-s_lo:+6.2f} .. {-s_hi:+6.2f}  {n}")
    t -= (tot_lo + tot_hi) / 2
    print(f"  SSM total {-tot_lo:+.2f} .. {-tot_hi:+.2f} -> {t:.2f} ms")
    for n, d, e in OTHER:
        t += d
        print(f"  {d:+6.2f} +- {e:.2f}  {n} -> {t:.2f}")
    t2 = t - 4.35
    print(f"  -4.35  k-step 2,055 -> 1,600 (ceiling) -> {t2:.2f} ms; without it {t:.2f}")
    c0 = 2.49  # depth-independent ms (sweep line r14f 47.31 / 39.84 / 31.87 / 24.43: 3.735 ms per block + 2.49)
    blk = (t2 - c0) / 12
    print(f"  ceiling depth 10 ~{10 * blk + c0:.1f} ms, depth 8 ~{8 * blk + c0:.1f} ms")
