"""sim12: per-block composition of design8 at round 9 (full12r9s, 58.14 ms at depth 12), npu-research rev 5.

What changed from sim11:
  - The GEMM side is no longer composed from a k-step guess: every GEMM op's wall time is MEASURED from the g11-r9sp
    PROF records (tools/g11wall.py: per column the slowest of the 3 cores) and split into
        L (first-tile A-slot wait) + nt x K/64 x t_ks + nt x t_tile + coupling (row imbalance through the A broadcast)
    with t_ks = the steady in-situ k-step (2,055 cycles inside mm_ks + 67 between calls) and L / t_tile / coupling
    read from the same records (tools/g11comp.py). up_a / up_b have incomplete records: their sum is the g11 total
    (29.47 ms measured) minus the other nine ops.
  - The k-step carries an A-port term: t_ks = max(compute, A bytes / R_A) + gap. R_A (memtile -> core, broadcast)
    is >= 5.03 B/cycle in situ (r9s steady state) and >= 5.5 (hard) / ~7.6 (fit) in the round-6 no-MAC twin
    (tools/twinfit.py). It does not bind today (needed: 10,368 / 2,055 = 5.0 B/cycle).
  - The A-slot load (L ~33 us, DDR -> memtile, all 8 columns at once = 2 MB in ~36 us = ~55 GB/s) is DDR-bound; P5's
    levers are priced against DDR at 55 GB/s.
  - The block (4,800 us) is the depth slope of the r9s sweep (58.10 / 48.53 / 38.85 / 29.21 ms at 12 / 10 / 8 / 6);
    the fixed part (0.5 ms) is patch embedding + final norm + launch. ONE constant is fitted: the SSM + EW-boundary
    sync time (873 us per block; the hand-off's in-situ SSM ~790 leaves ~14 us per EW boundary).

usage: python3 sim12.py
"""

F = 1.81e9
BLOCK_US = 4800.0  # depth slope, r9s sweep
FIXED_MS = 58.10 - 12 * BLOCK_US / 1e3
GAP = 67  # cycles between mm_ks calls (B acquire / release + IRON loop), g11-r9sp
C_KS = 2055  # in-situ steady mm_ks cycles: static 1,830 + bank 24 + ~201 (DMA collisions, locks, record loads)
A_KS = 10368  # bytes of A per k-step per core
B_KS = 4608
R_A = 7.6  # B/cycle, memtile -> core broadcast (twin fit; hard lower bound 5.5, in-situ >= 5.03)
DDR = 55e3  # bytes per us: 8 x 248,832 B per A-slot load in ~33 us from mm_ks entry, ~36 us from the last send (54-60 GB/s)

# name, K, nt, L (us, g11), t_tile (us, avg over rows), coupling (us per op), EW op before it
GEMM = [
    ("yx", 768, 8, 34.3, 5.20, 21.6, True), ("out", 768, 4, 33.5, 2.98, 0.0, True),
    ("up_a", 768, 16, 33.0, 4.18, 0.0, True), ("up_b", 768, 16, 33.0, 4.18, 0.0, False),
    ("down_a", 1536, 4, 28.2, 2.55, 0.0, False), ("down_b", 1536, 4, 34.2, 2.58, 0.0, False),
    ("qkv", 768, 12, 34.3, 4.51, 80.2, True), ("out_proj", 768, 4, 32.8, 2.70, 0.0, False),
    ("mlp0", 768, 16, 33.9, 9.99, 0.0, True), ("mlp3_a", 1536, 4, 34.0, 2.55, 16.7, False),
    ("mlp3_b", 1536, 4, 33.6, 2.58, 0.0, False),
]
HAND, ZERO = 1.32, 0.49  # per tile, inside t_tile
EW = dict(attn=930.0, norm0=124.0, norm1=123.0, norm2=148.0, norm3=146.0)  # attention on the GEMM cores; norms on EW
N_TILES = sum(g[2] for g in GEMM)
KS_BLOCK = sum(g[2] * g[1] // 64 for g in GEMM)


def t_ks_us(q):
    comp = q["c_ks"]
    port = max(A_KS * q.get("a_frac", 1.0) / q["r_a"], B_KS / q.get("r_b", 7.6))
    return (max(comp, port) + q["gap"]) / F * 1e6


def gemm_us(g, q):
    name, K, nt, L, tile, coup, _ = g
    L = q.get("L", {}).get(name, L)
    tile = tile + q.get("tile_add", 0.0)
    coup = 0.0 if q.get("balanced") else coup
    return L + nt * (K // 64) * t_ks_us(q) + nt * tile + coup


def base(**kw):
    q = dict(c_ks=C_KS, gap=GAP, r_a=R_A, ssm_sync=None, ew=dict(EW))
    q.update(kw)
    return q


def parts(q):
    p = dict(kstep=0.0, opstart=0.0, tiles=0.0, coupling=0.0, attn=q["ew"]["attn"],
             norms=sum(q["ew"][f"norm{i}"] for i in range(4)))
    tks = t_ks_us(q)
    for g in GEMM:
        name, K, nt, L, tile, coup, _ = g
        p["kstep"] += nt * (K // 64) * tks
        p["opstart"] += q.get("L", {}).get(name, L)
        p["tiles"] += nt * (tile + q.get("tile_add", 0.0))
        p["coupling"] += 0.0 if q.get("balanced") else coup
    return p


def calib():
    q = base()
    rest = sum(parts(q).values())
    return BLOCK_US - rest  # SSM + EW-boundary syncs


SSM_SYNC = calib()


def block_us(q):
    s = q["ssm_sync"] if q["ssm_sync"] is not None else SSM_SYNC
    return sum(parts(q).values()) + s


def step_ms(q, depth=12):
    return depth * block_us(q) / 1e3 + FIXED_MS


# ---- levers ----------------------------------------------------------------------------------------------------------
def p5a(q):
    """K = 768 slots loaded once (124 KB, DDR-bound: half the wait); up_b reuses up_a's A (same TAP: no load)."""
    L = dict(q.get("L", {}))
    for g in GEMM:
        if g[1] == 768:
            L[g[0]] = g[3] / 2
    L["up_b"] = 2.0
    return dict(q, L=L)


def p5(q):
    """P5a + the slot streamed in k-chunks: the first tile runs at max(compute, DDR) instead of load + compute."""
    q = p5a(q)
    L = dict(q["L"])
    tks = t_ks_us(q)
    for g in GEMM:
        name, K = g[0], g[1]
        if name == "up_b":
            continue
        load = 8 * (K * 144 * 9 // 8) / DDR  # all 8 columns' slots at DDR speed
        first_compute = (K // 64) * tks
        L[name] = 3.0 + max(0.0, load - first_compute)  # first chunk ~3 us, then DDR or compute, whichever is slower
    return dict(q, L=L)


def ssm_delta(q, d):
    s = (q["ssm_sync"] if q["ssm_sync"] is not None else SSM_SYNC) + d
    return dict(q, ssm_sync=s)


def p7_full(q):
    """norms folded: no norm ops, no EW boundary before yx / up_a / qkv / mlp0 (4 x ~14 us), +0.3 us per tile on the
    producer epilogues (stats) and the consumers ((C - mu s) / sigma)."""
    ew = dict(q["ew"], **{f"norm{i}": 0.0 for i in range(4)})
    q = dict(q, ew=ew, tile_add=q.get("tile_add", 0.0) + 0.3)
    return ssm_delta(q, -4 * 14.0)


def p7_cons(q):
    ew = dict(q["ew"], **{f"norm{i}": 40.0 for i in range(4)})
    return dict(q, ew=ew)


def show(name, q, ref):
    s = step_ms(q)
    print(f"  {s - ref:+6.2f} ms  {s:6.2f}  {name}")
    return s


def main():
    q0 = base()
    print(f"sim12 on full12r9s: block {BLOCK_US:.0f} us (+{FIXED_MS:.2f} ms fixed), {KS_BLOCK} k-steps and {N_TILES} tiles per core")
    print(f"  t_ks = max({C_KS} compute, A {A_KS} B / {R_A} B/cyc = {A_KS / R_A:.0f}) + gap {GAP} = {t_ks_us(q0) * F / 1e6:.0f}"
          f" cycles = {t_ks_us(q0):.3f} us   (A-port floor at the hard bound 5.5 B/cyc: {A_KS / 5.5:.0f})")
    p = parts(q0)
    p["ssm+sync (fit)"] = SSM_SYNC
    tot = sum(p.values())
    for k, v in p.items():
        print(f"    {k:15s} {v:7.1f} us  {12 * v / 1e3:5.2f} ms/step  {100 * v / tot:5.1f}%")
    ref = step_ms(q0)
    print(f"  model {ref:.2f} ms (measured 58.14)")

    print("\nLevers, each alone (depth 12):")
    lv = [
        ("P5a: K=768 slot once + up_b reuses up_a's slot", p5a(q0)),
        ("P5: + k-chunk streaming (first tile at max(compute, DDR))", p5(q0)),
        ("P7 full: norms folded into the GEMM epilogues", p7_full(q0)),
        ("P7 conservative: norm ops -> 40 us stats passes", p7_cons(q0)),
        ("P23: epilogue kinds balanced across rows (yx, qkv) + mlp3_a columns", dict(q0, balanced=True)),
        ("P27: a by table (SSM -58 us)", ssm_delta(q0, -58.0)),
        ("SSM hand-scheduled chains (P29, SSM 790 -> ~500)", ssm_delta(q0, -290.0)),
        ("attention 930 -> 600 us (P12 / P19)", dict(q0, ew=dict(q0["ew"], attn=600.0))),
        ("P35: B locks inside mm_ks, one call per tile (gap 67 -> 12)", dict(q0, gap=12)),
        ("P24b: acc_zero folded into the first k-step", dict(q0, tile_add=-ZERO)),
        ("P37: DMA-collision placement (in-situ extra 201 -> ~100 cycles)", dict(q0, c_ks=C_KS - 100)),
        ("k-step compute 1,600 (deep-K visits / 2x2; needs L1), R_A 7.6", dict(q0, c_ks=1600)),
        ("k-step compute 1,600, R_A 5.5 (the hard bound)", dict(q0, c_ks=1600, r_a=5.5)),
        ("k-step compute 1,200, R_A 7.6 (A-port-bound)", dict(q0, c_ks=1200)),
        ("k-step compute 1,200, A split over 2 S2MM (R_A x 2, B shares)", dict(q0, c_ks=1200, r_a=2 * 7.6 * A_KS / (A_KS + B_KS), r_b=1e9)),
    ]
    for n, q in lv:
        show(n, q, ref)

    print("\nCombined (depth 12), each row adds to the one above:")
    q = p5(q0)
    show("P5 (P5a + reuse + chunks)", q, ref)
    q = p7_full(q)
    show("+ P7 full", q, ref)
    q = dict(q, balanced=True)
    show("+ P23", q, ref)
    q = dict(q, gap=12, tile_add=q["tile_add"] - ZERO)
    show("+ P35, P24b", q, ref)
    q = ssm_delta(q, -58.0)
    show("+ P27", q, ref)
    q = dict(q, ew=dict(q["ew"], attn=600.0))
    s4 = show("+ attention 0.6 ms                       (ceiling with today's 4-chain k-step)", q, ref)
    q = ssm_delta(q, -232.0)
    show("+ SSM hand-scheduled (790 -> ~500)", q, ref)
    q1 = dict(q, c_ks=C_KS - 100)
    show("+ P37 DMA-collision placement", q1, ref)
    q2 = p5(dict(q1, c_ks=1600))
    s_all = show("+ k-step compute 1,600 (needs L1; R_A 7.6)   (ceiling)", q2, ref)
    show("  same with R_A = 5.5 (the hard bound)", p5(dict(q2, r_a=5.5)), ref)
    p = parts(q2)
    p["ssm+sync"] = q2["ssm_sync"]
    print("  ceiling per block: " + ", ".join(f"{k} {v:.0f}" for k, v in p.items()) + f" = {block_us(q2):.0f} us")
    for dep in (12, 10, 8):
        print(f"  ceiling depth {dep}: {step_ms(q2, dep):.1f} ms ({1e3 / step_ms(q2, dep):.1f} fps)")

    print("\nP14 (two frames in flight), the EW cores' work hides behind the GEMM cores' (list bound: per frame >= the busier"
          " side; boundaries stay):")
    for name, q in (("today", q0), ("after P5 + P7 + P23", dict(p7_full(p5(q0)), balanced=True)), ("ceiling", q2)):
        g = sum(v for k, v in parts(q).items() if k not in ("norms",))
        e = parts(q)["norms"] + (q["ssm_sync"] if q["ssm_sync"] is not None else SSM_SYNC)
        serial = block_us(q)
        per = max(g, e) + 0.15 * min(g, e)  # 15% of the hidden side stays exposed (sync points), sim9's list schedule ~1.31x
        print(f"  {name:22s} serial {12 * serial / 1e3 + FIXED_MS:5.1f} ms; GEMM side {g:6.0f} us, EW side {e:5.0f} us per block;"
              f" 2 in flight ~{12 * per / 1e3 + FIXED_MS:5.1f} ms per frame ({1e3 / (12 * per / 1e3 + FIXED_MS):4.1f} fps),"
              f" latency ~{2 * (12 * per / 1e3 + FIXED_MS):5.1f} ms")


if __name__ == "__main__":
    main()
