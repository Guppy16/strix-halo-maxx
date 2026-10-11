"""sim14 (npu-research rev 7): sim13's block model refitted on round 11.

Inputs (no device):
  - timeline tapnext/npu/build/tl_full12r11b_d12.json (full12r11b, 52.43 ms: GELU_MS, before P43);
  - per-core GEMM composition from the g11-r11gp PROF records (r12/g11comp_r11gp.out);
  - P43 as its measured step delta (r11b 52.43 -> r11c 52.08 ms), spread as a constant per block;
  - the depth line from the full12r11d sweep (52.07 / 43.52 / 34.84 / 26.23 ms).
Lever prices use the loop IIs that msched.py reaches at the round-11 rules (tools/g1sched.py, r12/g1sched.out) and the
per-tile / per-chunk costs in ANALYSIS rev 7.

usage: python3 sim14.py   (reuses sim13's functions with round-11 globals)
"""

import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim13 as S  # noqa: E402

S.TL = S.B / "tl_full12r11b_d12.json"
PREV = S.B / "tl_full12r10r_d12.json"
S.SWEEP = {12: 52.07, 10: 43.52, 8: 34.84, 6: 26.23}
P43_US = -(52.43 - 52.08) / 12 * 1e3
S.COMP = {  # g11-r11gp: nt, ks, steady, gaps, first, coupling, epilogue, handoff, zero (us per core per op)
    "yx": (8, 12, 108.8, 3.3, 0.1, 0.3, 14.5, 10.5, 3.9),
    "out": (4, 12, 54.4, 1.6, 0.1, 0.1, 2.9, 5.3, 2.0),
    "up_a": (16, 12, 217.7, 6.5, 0.1, 0.2, 81.7, 0.5, 7.8),
    "up_b": (16, 12, 217.7, 6.5, 0.1, 0.2, 81.7, 0.5, 7.8),
    "down_a": (4, 24, 109.0, 3.4, 0.1, 0.1, 2.9, 5.3, 2.0),
    "down_b": (4, 24, 109.5, 3.4, 0.2, 0.3, 2.9, 5.3, 2.0),
    "qkv": (12, 12, 163.3, 4.9, 0.1, 0.1, 24.5, 15.8, 5.9),
    "out_proj": (4, 12, 54.4, 1.6, 0.1, 0.1, 2.9, 5.3, 2.0),
    "mlp0": (16, 12, 217.6, 6.5, 0.1, 0.6, 68.8, 21.1, 7.8),
    "mlp3_a": (4, 24, 109.1, 3.4, 4.2, 0.5, 2.9, 5.3, 2.0),
    "mlp3_b": (4, 24, 109.5, 3.4, 0.2, 0.2, 2.9, 5.3, 2.0),
}
F = S.F

# ---- G1 at the measured rules (ANALYSIS rev 7 section 2). Per tile, cycles; bank stalls +5 %.
II_F = 12.5    # fused bias + GELU, 32 lanes (msched: 13 flat-edge, 12 flat-edge + biased LUT base; ResMII 10)
OVH = 150      # call + prologue (S - 1) II + drain per pass
BANK = 1.05


def us(c):
    return c * BANK / F


GELU_TILE = us(288 * II_F + OVH)                        # yx / the pan GELU: 288 vectors in place
GATE_PASS = us(144 * II_F + OVH)                        # the gate half
GUP = us(72 * 6 + 120)                                  # up bias + product + push, II 6 per 64 (msched)
PANPUSH = us(144 * 2 + 120)                             # bf16 -> fp32 -> push, II 2 per 64
STASH = us(400)                                         # gated_b's stash copy (half the tiles carry it: averaged below)
NOW = dict(yx_g=(14.5 - 4 * 0.71) / 4, gated=81.7 / 16, pan=68.8 / 16)
BIAS_PASS, GELU_MS_GATE = 1290 / F, (144 * 14.6 + 2 * 120) / F


def epi_g1(full=True):
    if not full:  # G1a: only the product + push pass replaced (bias pass + gelu_ms stay)
        gated = BIAS_PASS + GELU_MS_GATE + GUP + STASH / 2
        return dict(up_a=16 * gated, up_b=16 * gated)
    gated = GATE_PASS + GUP + STASH / 2
    return dict(yx=4 * GELU_TILE + 4 * 0.71, up_a=16 * gated, up_b=16 * gated, mlp0=16 * (GELU_TILE + PANPUSH))


ATT_UNITS = 432 * 4


def att(cyc):
    return cyc * ATT_UNITS / F


def main():
    m, disp = S.walls(S.TL)
    m0, _ = S.walls(PREV)
    res, const = S.fit(m)
    sl, c, dres = S.depth_line()
    blk = sum(m[k] for k in S.ORDER)
    print(f"sim14 on {S.TL.name}: dispatch {disp:.2f} ms; block (1-11) {blk:.0f} us; depth line {sl:.0f} us/block + {c:.2f} ms"
          f" (residuals " + ", ".join(f"d{k} {v:+.2f}" for k, v in dres.items()) + ")")
    comp = sum(S.comp_us(g) for g in S.GEMMS)
    print(f"  g11-r11gp composition {comp:.0f} us/block x 12 = {12 * comp / 1e3:.2f} ms (r10rp: 2,074 us)")
    print("\nGEMM boundaries (wall - composition), classes, fit:")
    for g in S.GEMMS:
        cl = S.CLASS[g]
        pred = S.comp_us(g) + (const[cl] if cl != "ssm" else res[g])
        print(f"  {g:9s} wall {m[g]:6.1f} comp {S.comp_us(g):6.1f} boundary {res[g]:+6.1f} class {cl:5s} ({const.get(cl, 0):+6.1f}) err {pred - m[g]:+6.1f}")
    errs = [S.comp_us(g) + const[S.CLASS[g]] - m[g] for g in S.GEMMS if S.CLASS[g] != "ssm"]
    t_cls, _ = S.block(m, const, dict(class_bounds=True))
    print(f"  class model rms {np.sqrt(np.mean(np.square(errs))):.1f} us; step (r11b) {12 * t_cls / 1e3 + c - 12 * P43_US / 1e3:.2f}"
          f" vs 52.43; with P43 {12 * t_cls / 1e3 + c:.2f} vs 52.07 (the depth line includes P43)")

    t0, parts = S.block(m, const)
    t0 += P43_US
    print(f"\nBlock breakdown r11b + P43 ({t0:.0f} us):")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:36s} {v:7.1f} us  {12 * v / 1e3:5.2f} ms/step  {100 * v / t0:5.1f}%")
    print("\nRound 10 -> 11 per op (timeline means, us): r10r -> r11b")
    for k in S.ORDER:
        print(f"    {k:9s} {m0[k]:7.1f} -> {m[k]:7.1f}  {m[k] - m0[k]:+7.1f}")

    ref = 12 * t0 / 1e3 + c

    def step(q):
        t, p = S.block(m, const, q)
        return 12 * (t + P43_US + q.get("dblk", 0)) / 1e3 + c, t + P43_US + q.get("dblk", 0), p

    print(f"\nper tile (us): GELU tile {GELU_TILE:.2f} (now yx {NOW['yx_g']:.2f}), gated G1 {GATE_PASS + GUP + STASH / 2:.2f} / G1a "
          f"{BIAS_PASS + GELU_MS_GATE + GUP + STASH / 2:.2f} (now {NOW['gated']:.2f}), pan G1 {GELU_TILE + PANPUSH:.2f} (now {NOW['pan']:.2f})")
    a = m["attn"]
    ssm0 = m["out"] - S.comp_us("out")
    chunk = 216 / F  # us per (cycle per SSM chunk)
    lv = [
        ("G1a: gated product + push pass generated (gup, II 6 / 64)", dict(epi=epi_g1(False))),
        ("G1 : one fused bias+GELU kernel (II 12.5) + gup + panpush", dict(epi=epi_g1(True))),
        ("  G1 at II 11 (if a better allocator reaches it)", None),
        ("P43b: B before the EW op's finish + memtile B one tile deep", dict(dblk=-30)),
        ("P51: norm output straight into the next GEMM's A slot (4 after-EW boundaries 28 -> ~8 us)", dict(dblk=-80)),
        ("P48: attention exp + push generated per row block (330 -> ~250 cycles)", dict(attn=a - att(80))),
        ("P49a: QK^T + store in asm (138 -> ~70)", dict(attn=a - att(68))),
        ("P49b: max / rows8 / control fused (~150 -> ~50)", dict(attn=a - att(100))),
        ("K phase P48 + P49 (571 -> ~320 cycles per (rb, chunk))", dict(attn=a - att(251))),
        ("P50: V phase in asm (213 -> ~90)", dict(attn=a - att(123))),
        ("P29a: sigmoid pass generated (1168 -> 450 cyc/chunk)", dict(ssm=ssm0 - 718 * chunk)),
        ("P29b: loga + exp chain as one generated loop (1170 -> 450)", dict(ssm=ssm0 - 720 * chunk)),
        ("P29c: sqrt + recurrence generated (1381 -> 560)", dict(ssm=ssm0 - 821 * chunk)),
        ("P29d: conv + z panels generated (1302 -> 610)", dict(ssm=ssm0 - 692 * chunk)),
        ("P29 all (x 0.85 realism)", dict(ssm=ssm0 - 0.85 * 2951 * chunk)),
        ("P52: SSM outputs written in the ring, conv cache by rotation (out copies 798 -> ~250)", dict(ssm=ssm0 - 548 * chunk)),
        ("P7-E low / high (pending P44)", dict(norms=dict(norm0=122.0, norm1=87.0, norm2=122.0, norm3=87.0))),
        ("P7-E high", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("P35 + P24b", dict(gap_scale=12 / 67, p24b=True)),
        ("k-step 2,055 -> 1,600 (needs L1, R_A)", dict(ks_scale=(1600 + 67) / (2055 + 67), gap_scale=0)),
    ]
    print(f"\nLevers alone (depth 12, model base {ref:.2f} ms):")
    for n, q in lv:
        if q is None:
            q = dict(epi={k: v * 11 / 12.5 if k in ("yx", "mlp0") else v for k, v in epi_g1(True).items()})
        st, _, _ = step(q)
        print(f"  {st - ref:+6.2f} ms  {st:6.2f}  {n}")

    print("\nCombined (each row adds to the previous):")
    combo = [
        ("G1", dict(epi=epi_g1(True))),
        ("+ P43b", dict(dblk=-30)),
        ("+ attention K phase (P48 + P49)", dict(attn=a - att(251))),
        ("+ P29 (x 0.85) + P52", dict(ssm=ssm0 - (0.85 * 2951 + 548) * chunk)),
        ("+ P7-E high", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("+ P35 + P24b", dict(gap_scale=12 / 67, p24b=True)),
        ("+ P50", dict(attn=a - att(251 + 123))),
        ("+ P51", dict(dblk=-30 - 80)),
        ("+ k-step 1,600 (ceiling)", dict(ks_scale=(1600 + 12) / (2055 + 67), gap_scale=0)),
    ]
    q, rows = {}, []
    for n, d in combo:
        q = {**q, **d}
        st, t, p = step(q)
        rows.append((n, st, t, p, dict(q)))
        print(f"  {st - ref:+6.2f} ms  {st:6.2f}  ({1e3 / st:4.1f} fps)  {n}")
    _, st, t, p, qc = rows[-1]
    print("  ceiling block: " + ", ".join(f"{k} {v:.0f}" for k, v in p.items()) + f" (+{qc.get('dblk', 0) + P43_US:.0f}) = {t:.0f} us")
    for dep in (12, 10, 8):
        x = dep * t / 1e3 + c
        print(f"  ceiling depth {dep}: {x:.1f} ms ({1e3 / x:.1f} fps)")
    _, st7, t7, _, _ = rows[-2]
    print(f"  without the k-step row: depth 12 {st7:.1f} ms, depth 10 {10 * t7 / 1e3 + c:.1f} ms")

    print("\nP14 (two frames in flight; per frame >= busier side + 15 % of the other):")
    for name, qq in (("today", {}), ("after G1 + P43b + attention K", rows[2][4]), ("after + SSM", rows[3][4]),
                     ("after + norms", rows[4][4]), ("ceiling", qc)):
        st_, t_, p_ = step(qq)
        e = p_["norms"] + p_["ssm (+boundary)"]
        g = t_ - e
        per = max(g, e) + 0.15 * min(g, e)
        two = 12 * per / 1e3 + c
        print(f"  {name:32s} serial {st_:5.1f} ms; GEMM side {g:5.0f}, EW side {e:5.0f} us/block; 2 in flight ~{two:5.1f} ms/frame"
              f" ({1e3 / two:4.1f} fps, x{st_ / two:.2f}), latency ~{2 * two:5.1f} ms")


if __name__ == "__main__":
    main()
