"""sim13: per-block model of design8 at round 10 (full12r10r, 53.89 ms at depth 12), npu-research rev 6.

What changed from sim12:
  - The block is composed from the MEASURED full-step timeline (tapnext/npu/build/tl_full12r10r_d12.json, host-polled,
    ~17 us resolution): per op the mean last-to-last wall over blocks 1..11 (block 0 has missed polls).
  - Every GEMM op's wall is split into the per-core composition of the g11-r10rp PROF records (tools/g11comp.py on
    build/g11-r10rp.npy: steady k-steps, B gaps, first wait, coupling, epilogue, handoff, acc_zero) plus a BOUNDARY term
    (timeline wall - composition). The boundary is fitted per class with 3 constants (after an EW op, after a dependent
    GEMM / attention, independent), so the per-op residuals measure how well the classes describe the step.
  - The EW ops (norms, SSM) and attention are their measured walls; the SSM is the `out` wall minus out's composition.
  - The depth model is the least-squares line through the r10r sweep (53.83 / 45.35 / 35.98 / 27.07 ms at 12 / 10 / 8 / 6).
  - Levers are re-priced on this base; the GELU and attention levers come from instruction counts (ANALYSIS rev 6 sections 2
    and 4), P7 from the dataflow costing of ANALYSIS rev 6 section 3.

usage: python3 sim13.py [timeline.json]   (no device; numpy only)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

F = 1.81e3  # cycles per us
B = Path(__file__).resolve().parents[2] / "tapnext" / "npu" / "build"
TL = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "tl_full12r10r_d12.json"
TL9 = B / "tl_full12r9s_d12.json"
SWEEP = {12: 53.83, 10: 45.35, 8: 35.98, 6: 27.07}  # build/sweep_r10r.out, dispatch medians
G11_NONPROF_MS = 25.21  # g11-r10r (RESULTS 19.1)

# g11-r10rp per-core composition (us per op; tools/g11comp.py output, r11/g11comp_r10rp.out)
#            nt  ks  steady  gaps  first  coup   epi    hand  zero
COMP = {
    "yx": (8, 12, 108.8, 3.3, 0.1, 0.3, 27.1, 10.5, 3.9),
    "out": (4, 12, 54.4, 1.6, 0.1, 0.1, 2.9, 5.3, 2.0),
    "up_a": (16, 12, 217.8, 6.5, 0.1, 0.2, 112.5, 0.5, 7.8),
    "up_b": (16, 12, 217.6, 6.5, 0.1, 0.2, 112.5, 0.5, 7.8),
    "down_a": (4, 24, 109.0, 3.4, 0.1, 0.1, 2.9, 5.3, 2.0),
    "down_b": (4, 24, 109.5, 3.4, 0.2, 0.2, 2.9, 5.3, 2.0),
    "qkv": (12, 12, 163.3, 4.9, 0.1, 0.1, 24.5, 15.8, 5.9),
    "out_proj": (4, 12, 54.4, 1.6, 0.1, 0.1, 2.9, 5.3, 2.0),
    "mlp0": (16, 12, 217.5, 6.5, 0.1, 0.6, 131.0, 21.1, 7.8),
    "mlp3_a": (4, 24, 109.1, 3.4, 4.2, 0.5, 2.9, 5.3, 2.0),
    "mlp3_b": (4, 24, 109.5, 3.4, 0.2, 0.2, 2.9, 5.3, 2.0),
}
# boundary class of each GEMM op in the full step: what it waits for when it starts
CLASS = dict(yx="ew", out="ssm", up_a="ew", up_b="reuse", down_a="ind", down_b="ind", qkv="ew", out_proj="dep",
             mlp0="ew", mlp3_a="dep", mlp3_b="ind")
ORDER = ["norm0", "yx", "out", "norm1", "up_a", "up_b", "down_a", "down_b", "norm2", "qkv", "attn", "out_proj", "norm3",
         "mlp0", "mlp3_a", "mlp3_b"]
GEMMS = [k for k in ORDER if k in COMP]


def walls(path):
    d = json.loads(Path(path).read_text())
    w, prev = defaultdict(list), None
    for o in d["ops"]:
        if o["last"] is None:
            continue
        name = o["name"]
        blk = name.split("_")[0]
        kind = name[len(blk) + 1:] if blk.startswith("b") and blk[1:].isdigit() else name
        if prev is not None and blk != "b0":
            w[kind].append((o["last"] - prev) * 1e3)
        prev = o["last"]
    return {k: float(np.mean(v)) for k, v in w.items()}, d["dispatch_ms"]


def comp_us(name, q=None):
    nt, ks, steady, gaps, first, coup, epi, hand, zero = COMP[name]
    q = q or {}
    tks = q.get("ks_scale", 1.0)
    epi = q.get("epi", {}).get(name, epi)
    gaps = gaps * q.get("gap_scale", 1.0)
    zero = 0.0 if q.get("p24b") else zero
    return steady * tks + gaps + first + coup + epi + hand + zero


def fit(m):
    """Boundary constants per class (mean residual of the class) and the per-op residual after them."""
    res = {g: m[g] - comp_us(g) for g in GEMMS}
    cls = defaultdict(list)
    for g in GEMMS:
        cls[CLASS[g]].append(res[g])
    const = {c: float(np.mean(v)) for c, v in cls.items()}
    return res, const


def block(m, const, q=None):
    """Block time (us) for a lever set q, from the measured walls m and the class constants."""
    q = q or {}
    t = 0.0
    parts = defaultdict(float)
    for op in ORDER:
        if op in COMP:
            c = comp_us(op, q)
            if CLASS[op] == "ssm":
                ssm = q.get("ssm", m["out"] - comp_us("out"))
                parts["ssm (+boundary)"] += ssm
                t += ssm
            else:
                b = q.get("bound", {}).get(op, const[CLASS[op]] if q.get("class_bounds") else m[op] - comp_us(op))
                parts["boundaries"] += b
                t += b
            nt, ks, steady, gaps, first, coup, epi, hand, zero = COMP[op]
            parts["k-steps"] += steady * q.get("ks_scale", 1.0) + gaps * q.get("gap_scale", 1.0)
            parts["epilogues"] += q.get("epi", {}).get(op, epi)
            parts["tile handoff + zero + first + coup"] += hand + (0 if q.get("p24b") else zero) + first + coup
            t += c
        elif op == "attn":
            a = q.get("attn", m["attn"])
            parts["attention"] += a
            t += a
        else:
            n = q.get("norms", {}).get(op, m[op])
            parts["norms"] += n
            t += n
    if q.get("extra"):
        parts["added work (P7)"] += q["extra"]
        t += q["extra"]
    return t, parts


def depth_line():
    d = np.array(sorted(SWEEP))
    t = np.array([SWEEP[k] for k in d])
    s, c = np.linalg.lstsq(np.vstack([d, np.ones_like(d)]).T, t, rcond=None)[0]
    return s * 1e3, c, {int(k): float(v - (s * k + c)) for k, v in zip(d, t)}


# ---- levers (rev 6) ---------------------------------------------------------------------------------------------
GELU_CPP = dict(yx=27.1, up_a=112.5 - 16 * 2.88, up_b=112.5 - 16 * 2.88, mlp0=131.0 - 16 * 1.57)  # G0: calls per tile + gated pass
# G1: fused bias + GELU (+ product + push) modulo-scheduled at II cycles per 32 output lanes (+ ~250 cycles per call)
def gelu_asm(ii):
    per = lambda n, ii_: (n * ii_ + 250) / F  # one call per tile
    yx = 4 * per(288, ii) + 4 * 0.71  # 4 GELU + 4 plain tiles per core
    gated = 16 * per(144, ii + 1)  # st / mv slots one heavier than the plain GELU
    pan = 16 * per(288, ii + 0.5)
    return dict(yx=yx, up_a=gated, up_b=gated, mlp0=pan)


ATT_PER_RBCH = 571.0  # cycles per (row block, key chunk) of the K phase: 136.4 us per head (r8 stamps) / 432
ATT_UNITS = 432 * 4  # (rb, chunk) per head x 4 heads per core per block


def attn_minus(cycles_saved):
    return cycles_saved * ATT_UNITS / F


def main():
    m, disp = walls(TL)
    m9, disp9 = walls(TL9)
    res, const = fit(m)
    s, c, dres = depth_line()
    blk_meas = sum(m[k] for k in ORDER)
    print(f"sim13 on {TL.name}: dispatch {disp:.2f} ms; mean block (blocks 1-11) {blk_meas:.0f} us; depth line {s:.0f} us/block"
          f" + {c:.2f} ms (sweep residuals " + ", ".join(f"d{k} {v:+.2f}" for k, v in dres.items()) + " ms)")
    comp_blk = sum(comp_us(g) for g in GEMMS)
    print(f"  g11 composition {comp_blk:.0f} us/block x 12 = {12 * comp_blk / 1e3:.2f} ms vs g11-r10r measured {G11_NONPROF_MS} ms"
          f" ({100 * (12 * comp_blk / 1e3 / G11_NONPROF_MS - 1):+.1f}%)")

    print("\nGEMM boundaries (timeline wall - g11 composition, us), class constants and per-op fit error:")
    for g in GEMMS:
        cl = CLASS[g]
        pred = comp_us(g) + (const[cl] if cl != "ssm" else res[g])
        print(f"  {g:9s} wall {m[g]:6.1f}  comp {comp_us(g):6.1f}  boundary {res[g]:+6.1f}  class {cl:5s} ({const.get(cl, 0):+6.1f})"
              f"  model {pred:6.1f}  err {pred - m[g]:+6.1f}")
    errs = [comp_us(g) + const[CLASS[g]] - m[g] for g in GEMMS if CLASS[g] not in ("ssm",)]
    t_cls, _ = block(m, const, dict(class_bounds=True))
    print(f"  class model: per-op rms error {np.sqrt(np.mean(np.square(errs))):.1f} us, block {t_cls:.0f} vs {blk_meas:.0f} us"
          f" ({100 * (t_cls / blk_meas - 1):+.2f}%), step {12 * t_cls / 1e3 + c:.2f} vs {SWEEP[12]} ms")

    t0, parts = block(m, const)
    print(f"\nBlock breakdown (critical path; every op is serial today), {t0:.0f} us:")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:36s} {v:7.1f} us  {12 * v / 1e3:5.2f} ms/step  {100 * v / t0:5.1f}%")

    print("\nRound 9 -> 10 per op (timeline means, us):  r9s -> r10r  delta")
    for k in ORDER:
        print(f"    {k:9s} {m9[k]:7.1f} -> {m[k]:7.1f}  {m[k] - m9[k]:+7.1f}")
    print(f"    block {sum(m9[k] for k in ORDER):7.0f} -> {blk_meas:7.0f}  {blk_meas - sum(m9[k] for k in ORDER):+7.0f}")

    ref = 12 * t0 / 1e3 + c

    def show(name, q):
        t, _ = block(m, const, q)
        st = 12 * t / 1e3 + c
        print(f"  {st - ref:+6.2f} ms  {st:6.2f}  {name}")
        return q

    print(f"\nLevers, each alone (depth 12, from the model's {ref:.2f} ms):")
    att = m["attn"]
    norms_now = sum(m[k] for k in ("norm0", "norm1", "norm2", "norm3"))
    lv = [
        ("G0: GELU one call per tile + gated product pass pipelined (C++)", dict(epi=GELU_CPP)),
        ("G1: fused bias+GELU(+product+push) modulo-scheduled, II 10", dict(epi=gelu_asm(10))),
        ("G1 at II 12", dict(epi=gelu_asm(12))),
        ("G1 at II 16", dict(epi=gelu_asm(16))),
        ("A1: attention exp+push modulo-scheduled per row block (330 -> ~150 cyc of the 571)", dict(attn=att - attn_minus(180))),
        ("A2: QK^T + score store in asm (138 -> 68 cyc)", dict(attn=att - attn_minus(70))),
        ("A3: running max fused, rows8 by shuffle (~150 -> ~40 cyc)", dict(attn=att - attn_minus(110))),
        ("A1+A2+A3 (K phase 571 -> ~180 cyc per (rb, chunk))", dict(attn=att - attn_minus(391))),
        ("A4: V phase in asm (213 -> ~70 cyc per (rb, chunk))", dict(attn=att - attn_minus(143))),
        ("P43: B fills hoisted before the dependency wait (+ deeper B forward)",
         dict(bound=dict(mlp3_a=res["mlp3_a"] - 35, yx=res["yx"] - 7, up_a=res["up_a"] - 7, qkv=res["qkv"] - 7,
                         mlp0=res["mlp0"] - 7, out_proj=res["out_proj"] - 5))),
        ("P7-full fold (norm ops gone; producer residual via A path, double drain, stats slot; consumer scale)",
         dict(norms={k: 0.0 for k in ("norm0", "norm1", "norm2", "norm3")},
              extra=161.0)),  # +161 us of added producer / consumer work (ANALYSIS rev 6 section 3)
        ("P7-E low: EW norm kernel at a 4 B/cycle input port (3.5 effective) + 17 us sync", dict(norms=dict(norm0=122.0, norm1=87.0, norm2=122.0, norm3=87.0))),
        ("P7-E high / P7-P: input at ~8 B/cycle or on two channels; output-bound (345 KB) + sync", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("P27: a by table (SSM -58 us)", dict(ssm=m["out"] - comp_us("out") - 58)),
        ("P29: SSM chains modulo-scheduled (SSM -290 us)", dict(ssm=m["out"] - comp_us("out") - 290)),
        ("P35: B locks inside mm_ks (gap 67 -> 12 cycles)", dict(gap_scale=12 / 67)),
        ("P24b: acc_zero folded", dict(p24b=True)),
        ("k-step 2,055 -> 1,600 cycles (needs L1; R_A >= 6.5)", dict(ks_scale=(1600 + 67) / (2055 + 67), gap_scale=0)),
    ]
    for n, q in lv:
        show(n, q)

    print("\nCombined (depth 12), each row adds to the one above:")
    ssm0 = m["out"] - comp_us("out")
    steps = [
        ("G1 (II 12)", dict(epi=gelu_asm(12))),
        ("+ P43", dict(bound=dict(mlp3_a=res["mlp3_a"] - 35, yx=res["yx"] - 7, up_a=res["up_a"] - 7, qkv=res["qkv"] - 7,
                                  mlp0=res["mlp0"] - 7, out_proj=res["out_proj"] - 5))),
        ("+ A1+A2+A3 (attention K phase)", dict(attn=att - attn_minus(391))),
        ("+ P27 + P29 (SSM -348 us)", dict(ssm=ssm0 - 348)),
        ("+ P7-E high / P7-P norms (~270 us)", dict(norms=dict(norm0=65.0, norm1=65.0, norm2=70.0, norm3=70.0))),
        ("+ P35 + P24b", dict(gap_scale=12 / 67, p24b=True)),
        ("+ A4 (V phase)", dict(attn=att - attn_minus(391 + 143))),
        ("+ k-step 1,600 (ceiling; needs L1 and R_A)", dict(ks_scale=(1600 + 12) / (2055 + 67), gap_scale=0)),
    ]
    q = {}
    out = []
    for n, d in steps:
        q = {**q, **d}
        if "bound" in d:
            q["bound"] = {**q.get("bound", {}), **d["bound"]}
        t, p = block(m, const, q)
        st = 12 * t / 1e3 + c
        out.append((n, st, t, p, dict(q)))
        print(f"  {st - ref:+6.2f} ms  {st:6.2f}  ({1e3 / st:4.1f} fps)  {n}")
    _, st, t, p, qc = out[-1]
    print("  ceiling per block: " + ", ".join(f"{k} {v:.0f}" for k, v in p.items()) + f" = {t:.0f} us")
    for dep in (12, 10, 8):
        print(f"  ceiling depth {dep}: {dep * t / 1e3 + c:.1f} ms ({1e3 / (dep * t / 1e3 + c):.1f} fps)")

    print("\nP14 (two frames in flight): EW side (norms + SSM) runs beside the GEMM side (everything else) of the other frame;"
          " list bound per frame >= busier side + 15% of the other (sim9's schedule):")
    for name, qq in (("today", {}), ("after G1 + P43 + A1-3", out[2][4]), ("after + SSM + norms", out[4][4]), ("ceiling", qc)):
        t, p = block(m, const, qq)
        e = p["norms"] + p["ssm (+boundary)"]
        g = t - e
        per = max(g, e) + 0.15 * min(g, e)
        serial = 12 * t / 1e3 + c
        two = 12 * per / 1e3 + c
        print(f"  {name:24s} serial {serial:5.1f} ms; GEMM side {g:5.0f} us, EW side {e:5.0f} us per block;"
              f" 2 in flight ~{two:5.1f} ms per frame ({1e3 / two:4.1f} fps, x{serial / two:.2f}), latency ~{2 * two:5.1f} ms")


if __name__ == "__main__":
    main()
