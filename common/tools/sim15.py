"""sim15 (npu-research rev 8): sim13's block model refitted on round 12 (full12r12e, 50.05 ms).

Inputs (no device):
  - timeline tapnext/npu/build/tl_full12r12e_d12.json (the deployed build; prices start from it, never from older rounds);
  - per-core GEMM composition from the g11-r12gc PROF records (GUP + CPY32: r13/g11comp_r12gc.out), with the two later
    epilogue changes applied as measured: PAN from the PAN_NOPUSH differential (g11-r12gnp) + the pan bench (161 cycles per
    6 row blocks), BGELU as the r12c -> r12d timeline deltas;
  - the depth line from the r12e sweep (50.05 / 42.02 / 33.51 / 25.20 ms).
Lever prices: ANALYSIS rev 8 (attention K phase from tools/attk.py, SSM passes from the deployed binary's static loop
counts, tools/loops.py -> r13/loops_r12e_0_5.out).

usage: python3 sim15.py
"""

import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim13 as S  # noqa: E402

S.TL = S.B / "tl_full12r12e_d12.json"
PREV = S.B / "tl_full12r11d_d12.json"
S.SWEEP = {12: 50.05, 10: 42.02, 8: 33.51, 6: 25.20}
PAN_US = 161 * 3 / S.F  # asm pan per 18 row blocks (bench), us per tile
S.COMP = {  # g11-r12gc (r13/g11comp_r12gc.out); epilogues adjusted for PAN / BGELU (see the docstring)
    "yx": (8, 12, 108.8, 3.3, 0.1, 0.3, 14.5 - 1.1, 10.5, 3.9),
    "out": (4, 12, 54.4, 1.6, 0.1, 0.1, 2.9, 5.3, 2.0),
    "up_a": (16, 12, 218.5, 6.5, 0.0, 0.1, 33.6 - 3.8, 0.5, 7.8),
    "up_b": (16, 12, 218.6, 6.5, 0.0, 0.0, 33.6 - 3.8, 0.5, 7.8),
    "down_a": (4, 24, 109.0, 3.4, 0.4, 0.3, 2.9, 5.3, 2.0),
    "down_b": (4, 24, 109.6, 3.4, 0.1, 0.2, 2.9, 5.3, 2.0),
    "qkv": (12, 12, 163.2, 4.9, 0.1, 0.1, 9.8 + 8 * PAN_US / 2, 15.8, 5.9),
    "out_proj": (4, 12, 54.4, 1.6, 0.1, 0.1, 2.9, 5.3, 2.0),
    "mlp0": (16, 12, 217.6, 6.5, 0.1, 0.6, 49.2 + 16 * PAN_US - 11.0, 21.1, 7.8),
    "mlp3_a": (4, 24, 109.4, 3.4, 22.5, 0.4, 2.9, 5.3, 2.0),
    "mlp3_b": (4, 24, 109.6, 3.4, 0.1, 0.3, 2.9, 5.3, 2.0),
}
F = S.F
ATT_UNITS = 432 * 4  # (row block, key chunk) units per core per block
CH = 216 / F         # us per block per (cycle per SSM chunk)


def att(cyc):
    return cyc * ATT_UNITS / F


# A-slot bytes per column of each GEMM op (K x 144 rows x 9/8 B): the 'dep' / after-EW boundaries are this load
A_KB = dict(yx=124, out=124, up_a=124, up_b=0, down_a=248, down_b=248, qkv=124, out_proj=124, mlp0=124, mlp3_a=248, mlp3_b=248)


def main():
    m, disp = S.walls(S.TL)
    m0, _ = S.walls(PREV)
    res, const = S.fit(m)
    sl, c, dres = S.depth_line()
    blk = sum(m[k] for k in S.ORDER)
    print(f"sim15 on {S.TL.name}: dispatch {disp:.2f} ms; block (1-11) {blk:.0f} us; depth line {sl:.0f} us/block + {c:.2f} ms"
          f" (residuals " + ", ".join(f"d{k} {v:+.2f}" for k, v in dres.items()) + ")")
    comp = sum(S.comp_us(g) for g in S.GEMMS)
    print(f"  composition {comp:.0f} us/block x 12 = {12 * comp / 1e3:.2f} ms")
    print("\nGEMM boundaries (wall - composition) against the A slot each op loads (KB per column):")
    for g in S.GEMMS:
        cl = S.CLASS[g]
        print(f"  {g:9s} wall {m[g]:6.1f} comp {S.comp_us(g):6.1f} boundary {res[g]:+6.1f} class {cl:5s} A {A_KB[g]:3d} KB"
              f"  (A at 7.3 GB/s per column: {A_KB[g] / 7.3:5.1f} us)")
    t0, parts = S.block(m, const)
    print(f"\nBlock breakdown r12e ({t0:.0f} us):")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:36s} {v:7.1f} us  {12 * v / 1e3:5.2f} ms/step  {100 * v / t0:5.1f}%")
    print("\nRound 11d -> 12e per op (timeline means, us):")
    for k in S.ORDER:
        print(f"    {k:9s} {m0[k]:7.1f} -> {m[k]:7.1f}  {m[k] - m0[k]:+7.1f}")
    ref = 12 * t0 / 1e3 + c

    def step(q):
        t, p = S.block(m, const, q)
        t += q.get("dblk", 0)
        return 12 * t / 1e3 + c, t, p

    a = m["attn"]
    ssm0 = m["out"] - S.comp_us("out")
    K_NOW, K_FUSED, K_SPLIT = 571, float(sys.argv[1]) if len(sys.argv) > 1 else 300, 360
    lv = [
        ("P49a QK^T in asm (II 10 per key block [GEN]: 138 -> ~75 cycles per unit)", dict(attn=a - att(63))),
        ("P48 exp + push generated, separate pass (330 -> ~260)", dict(attn=a - att(70))),
        (f"P56 fused K loop, lazy-fast (571 -> {K_FUSED:.0f} cycles per unit)", dict(attn=a - att(K_NOW - K_FUSED))),
        ("P50 V phase in asm (213 -> ~90)", dict(attn=a - att(123))),
        ("P29b log_a + exp chain generated (1,130 static -> ~470)", dict(ssm=ssm0 - 660 * CH)),
        ("P29c sqrt + recurrence generated (1,398 -> ~440), x 0.85", dict(ssm=ssm0 - 0.85 * 958 * CH)),
        ("P29d conv + z generated (1,302 -> ~610), x 0.85", dict(ssm=ssm0 - 0.85 * 692 * CH)),
        ("P52 SSM outputs in place (798 -> ~250)", dict(ssm=ssm0 - 548 * CH)),
        ("P55 tile-granular A dependency: mlp3_a + out_proj (A load overlaps the producer's tail)", dict(dblk=-30)),
        ("P51 norm -> next GEMM's A slot on chip (4 after-EW boundaries)", dict(dblk=-80)),
        ("P7-E high (norms at the port bound)", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("P35 + P24b", dict(gap_scale=12 / 67, p24b=True)),
        ("P57 C drain overlapped (half the 1.3 us handoff per tile)", dict(dblk=-40)),
    ]
    print(f"\nLevers alone (depth 12, model base {ref:.2f} ms):")
    for n, q in lv:
        st, _, _ = step(q)
        print(f"  {st - ref:+6.2f} ms  {st:6.2f}  {n}")
    print("\nCombined (each row adds to the previous):")
    combo = [
        ("P29b", dict(ssm=ssm0 - 660 * CH)),
        ("+ attention K fused (P56)", dict(attn=a - att(K_NOW - K_FUSED))),
        ("+ P29c + P29d (x 0.85) + P52", dict(ssm=ssm0 - (660 + 0.85 * (958 + 692) + 548) * CH)),
        ("+ P55", dict(dblk=-30)),
        ("+ P7-E high", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("+ P35 + P24b", dict(gap_scale=12 / 67, p24b=True)),
        ("+ P50", dict(attn=a - att(K_NOW - K_FUSED + 123))),
        ("+ P51", dict(dblk=-30 - 80)),
        ("+ P57", dict(dblk=-30 - 80 - 40)),
        ("+ k-step 2,055 -> 1,600 (ceiling)", dict(ks_scale=(1600 + 12) / (2055 + 67), gap_scale=0)),
    ]
    q, rows = {}, []
    for n, d in combo:
        q = {**q, **d}
        st, t, p = step(q)
        rows.append((n, st, t, p, dict(q)))
        print(f"  {st - ref:+6.2f} ms  {st:6.2f}  ({1e3 / st:4.1f} fps)  {n}")
    _, st, t, p, qc = rows[-1]
    print("  ceiling block: " + ", ".join(f"{k} {v:.0f}" for k, v in p.items()) + f" (+{qc.get('dblk', 0):.0f}) = {t:.0f} us")
    for dep in (12, 10, 8):
        print(f"  ceiling depth {dep}: {dep * t / 1e3 + c:.1f} ms")
    _, st9, t9, _, _ = rows[-2]
    print(f"  without the k-step row: depth 12 {st9:.1f} ms, depth 10 {10 * t9 / 1e3 + c:.1f} ms")
    print("\nP14 (two frames in flight; per frame >= busier side + 15 % of the other):")
    for name, qq in (("today", {}), ("after P29b + K fused", rows[1][4]), ("after the SSM", rows[2][4]),
                     ("after norms", rows[4][4]), ("ceiling", qc)):
        st_, t_, p_ = step(qq)
        e = p_["norms"] + p_["ssm (+boundary)"]
        g = t_ - e
        per = max(g, e) + 0.15 * min(g, e)
        two = 12 * per / 1e3 + c
        print(f"  {name:22s} serial {st_:5.1f} ms; GEMM side {g:5.0f}, EW side {e:5.0f} us/block; 2 in flight ~{two:5.1f} ms/frame"
              f" ({1e3 / two:4.1f} fps, x{st_ / two:.2f}), latency ~{2 * two:5.1f} ms")


if __name__ == "__main__":
    main()
