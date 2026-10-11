"""sim16 (npu-research rev 9): sim15's block model on the deployed full12r13f (49.21 ms), with the attention op replaced by
attsim's lockstep simulation (tools/attsim.py), calibrated on the round-13 twins (r14/attcal.py).

Inputs (no device): timeline npu-tapnext/build/tl_full12r13f_d12.json; the r13f depth sweep (49.22 / 41.25 / 33.44 / 25.16);
the GEMM composition of sim15 (g11-r12gc + PAN / BGELU: round 13 left the GEMM ops unchanged, timelines agree within +-5 us);
the flag / move map r14/lazymap.npz (primary: thinned to the in-situ per-core range, `Att.thinned(0.109)`, R 450;
alternative: the raw map, R 300). Attention levers are attsim deltas against the modelled r13f, added to the measured wall.

usage: python3 sim16.py            (numpy only; ~1 min with OPENBLAS_NUM_THREADS=1)
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attsim as A  # noqa: E402
import sim13 as S  # noqa: E402
import sim15 as S15  # noqa: E402,F401  (sets S.COMP to the r12e composition)

S.TL = S.B / "tl_full12r13f_d12.json"
S.SWEEP = {12: 49.22, 10: 41.25, 8: 33.44, 6: 25.16}
F = S.F
CH = 216 / F  # us per block per (cycle per SSM chunk)
R14 = HERE.parent / "r14"


def attn_models():
    pb = replace(A.Prm(**json.load(open(R14 / "attcal.json"))), D=8)
    raw = A.default_att()
    return {"thin": (raw.thinned(0.109), replace(pb, R=450.0)), "raw": (raw, pb)}


def layout(g):
    """query row blocks dealt to the columns in pieces of g row blocks, round robin (r14/attlayout.py)"""
    a = np.zeros((8, 3, 6, 3), int)
    for c in range(8):
        for t in range(3):
            for m in range(6):
                pos = 6 * t + m
                n = c + 8 * (pos // g)
                k = n * g + pos % g
                a[c, t, m] = (k // 18, (k % 18) // 6, k % 6)
    return a


R13F = dict(a=1, test="flag", exp="flag")
C22 = dict(j22=False)  # chunk 22: the loop over its 2 valid key blocks + zero P panels (bit-identical), loop cost below


def att_variant(name):
    """(layout g or None, variant dict, Prm overrides) of the attention levers"""
    L22 = lambda prm: {22: prm.L * 2 / 6 + 40}  # noqa: E731
    V = {
        "P60 query rows interleaved (half tiles)": (3, {}, {}),
        "P60b interleaved by tile (3 pieces)": (6, {}, {}),
        "P61 chunk 22 fast": (None, dict(C22, chunk_L="22"), {}),
        "P62 slow path: rescale_rb in asm (R / 3)": (None, dict(r_scale=1 / 3), {}),
        "P62b slow exp by an attk rerun (aexp_push gone; E = L)": (None, dict(e_scale=1.0), {}),
        "P63 36-iteration K loop (L 400 -> 285)": (None, dict(l_scale=285 / 400), {}),
        "P50 V phase in asm (222 -> 90 per row block)": (None, dict(v_scale=90 / 222), {}),
        "P64 K / V objects of 3,456 B (producer 1600 -> 1200 per object)": (None, {}, dict(P_SHIM=1200.0, X=435.0)),
        "P65 chunk 0 QK-only loop": (None, dict(chunk_L={0: 130.0}), {}),
    }
    return V, L22


def att_ms(models, which, g, v, over):
    at, pb = models[which]
    if g is not None:
        at = at.remap_rows(layout(g))
    prm = replace(pb, **over)
    vv = dict(R13F, **{k: x for k, x in v.items() if k != "chunk_L"})
    if v.get("chunk_L") == "22":
        vv["chunk_L"] = {22: prm.L * 2 / 6 + 40}
    elif "chunk_L" in v:
        vv["chunk_L"] = dict(v["chunk_L"])
    return 12 * at.block_cycles(at.counts(vv), prm, vv) / F / 1e3


def merge(vs):
    out_g, out_v, out_o = None, {}, {}
    for g, v, o in vs:
        out_g = g if g is not None else out_g
        cl = {}
        for d in (out_v, v):
            c = d.get("chunk_L")
            if c == "22":
                cl[22] = "L22"
            elif c:
                cl.update(c)
        out_v.update({k: x for k, x in v.items() if k != "chunk_L"})
        if cl:
            out_v["chunk_L"] = cl
        out_o.update(o)
    return out_g, out_v, out_o


def att_ms_m(models, which, g, v, over):
    at, pb = models[which]
    if g is not None:
        at = at.remap_rows(layout(g))
    prm = replace(pb, **over)
    vv = dict(R13F, **{k: x for k, x in v.items() if k != "chunk_L"})
    if "chunk_L" in v:
        vv["chunk_L"] = {j: (prm.L * 2 / 6 + 40 if x == "L22" else x) for j, x in v["chunk_L"].items()}
    return 12 * at.block_cycles(at.counts(vv), prm, vv) / F / 1e3


def main():
    m, disp = S.walls(S.TL)
    res, const = S.fit(m)
    sl, c, dres = S.depth_line()
    blk = sum(m[k] for k in S.ORDER)
    print(f"sim16 on {S.TL.name}: dispatch {disp:.2f} ms; block (1-11) {blk:.0f} us; depth line {sl:.0f} us/block + {c:.2f} ms"
          f" (residuals " + ", ".join(f"d{k} {v:+.2f}" for k, v in dres.items()) + ")")
    t0, parts = S.block(m, const)
    print(f"\nBlock breakdown r13f ({t0:.0f} us):")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:36s} {v:7.1f} us  {12 * v / 1e3:5.2f} ms/step  {100 * v / t0:5.1f}%")
    ref = 12 * t0 / 1e3 + c

    models = attn_models()
    base = {w: att_ms_m(models, w, None, {}, {}) for w in models}
    print(f"\nattention: timeline {12 * m['attn'] / 1e3:.2f} ms/step; attsim r13f {base['thin']:.2f} (thin) / {base['raw']:.2f} (raw)")

    V, _ = att_variant(None)
    print("\nAttention levers alone (attsim deltas, ms per step; thin = primary, raw = alternative calibration):")
    alone = {}
    for name, (g, v, o) in V.items():
        d = {w: att_ms_m(models, w, g, *merge([(g, v, o)])[1:]) - base[w] for w in models}
        alone[name] = d
        print(f"  {d['thin']:+6.2f} / {d['raw']:+6.2f}  {name}")

    def att_delta(keys):
        g, v, o = merge([V[k] for k in keys])
        return {w: att_ms_m(models, w, g, v, o) - base[w] for w in models}

    print("\nAttention stacks (each row adds to the previous):")
    stack = ["P60 query rows interleaved (half tiles)", "P61 chunk 22 fast", "P62 slow path: rescale_rb in asm (R / 3)",
             "P50 V phase in asm (222 -> 90 per row block)", "P63 36-iteration K loop (L 400 -> 285)",
             "P64 K / V objects of 3,456 B (producer 1600 -> 1200 per object)"]
    att_rows = []
    for i in range(len(stack)):
        d = att_delta(stack[:i + 1])
        att_rows.append(d)
        extra = d["thin"] - (att_rows[-2]["thin"] if i else 0)
        print(f"  {d['thin']:+6.2f} / {d['raw']:+6.2f}  (this row {extra:+5.2f})  + {stack[i]}")
    no64 = att_delta(stack[:-1])
    nolock = {w: 12 * models[w][0].remap_rows(layout(3)).block_cycles(
        models[w][0].counts(R13F), replace(models[w][1], LOCK=False), R13F) / F / 1e3 - base[w] for w in models}
    print(f"  (P60 alone vs every column independent: {alone[stack[0]]['thin']:+.2f} vs {nolock['thin']:+.2f})")

    def step(q):
        t, p = S.block(m, const, q)
        t += q.get("dblk", 0)
        return 12 * t / 1e3 + c, t, p

    a = m["attn"]
    ssm0 = m["out"] - S.comp_us("out")
    att_us = lambda d: a + d * 1e3 / 12  # noqa: E731  (ms per step delta -> us per block)
    print(f"\nCombined (sim16, depth 12, model base {ref:.2f} ms; attention by attsim 'thin'):")
    combo = [
        ("P60 + P61 + P62 (bit-identical, DMA + slow path)", dict(attn=att_us(att_rows[2]["thin"]))),
        ("+ P29b", dict(ssm=ssm0 - 660 * CH)),
        ("+ P50 (V phase asm)", dict(attn=att_us(att_rows[3]["thin"]))),
        ("+ P63 (36-iteration K loop)", dict(attn=att_us(att_rows[4]["thin"]))),
        ("+ P64 (3,456 B K / V objects)", dict(attn=att_us(att_rows[5]["thin"]))),
        ("+ P29c + P29d (x 0.85) + P52", dict(ssm=ssm0 - (660 + 0.85 * (958 + 692) + 548) * CH)),
        ("+ P55", dict(dblk=-30)),
        ("+ P7-E high", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("+ P35 + P24b", dict(gap_scale=12 / 67, p24b=True)),
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
    for name, qq in (("today", {}), ("after the attention stack", rows[4][4]), ("after the SSM", rows[5][4]),
                     ("ceiling", qc)):
        st_, t_, p_ = step(qq)
        e = p_["norms"] + p_["ssm (+boundary)"]
        g = t_ - e
        per = max(g, e) + 0.15 * min(g, e)
        two = 12 * per / 1e3 + c
        print(f"  {name:26s} serial {st_:5.1f} ms; GEMM side {g:5.0f}, EW side {e:5.0f} us/block; 2 in flight ~{two:5.1f} ms/frame"
              f" ({1e3 / two:4.1f} fps), latency ~{2 * two:5.1f} ms")


if __name__ == "__main__":
    main()
