"""sim11: per-block composition model of design8 at round 8 (full12r8c), built from the round-8 stamps (npu-research rev 4).

Why not a span fit (sim10): no r8c timeline exists, and sim10's free t_tile (7.6 us) and t_ks were confounded: the P24
stamps measured the tile boundary at 2.4 us, so sim10 under-counted the k-step and over-counted the tile. sim11 fixes every
term the stamps measured and fits ONE constant: the dynamic k-step extra d, from the measured block time.

    block = sum over GEMM ops [ L_op + nt x K/64 x t_ks + tiles (handoff + acc_zero + tail + epilogue + ring coupling) ]
            + SSM + attention + norms                                         (everything serial: finish_all per EW op)
    t_ks  = (9 x (CALL + s_bank + s_dma) + IRON_KS + d) / F

Sources: RESULTS 17.1-17.6 (g11 stamps, mmb, SSM / attention stamps, step times); tools/bankmodel.py (s_bank of the r8c
placement); round-7 r7p stamps (ring coupling yx / qkv).
usage: python3 sim11.py
"""

F = 1.81e9
BLOCK_US = 67.89e3 / 12  # full12r8c, step8 time median of 20, depth 12 (the depth sweep slope gives 5.66-5.73)
TAIL_US = 0.0

# --- the k-step, from the binary (DOC) and the bank model (MEAS mmb + bankmodel.py on the r8c placement) ---------------
CALL = 34 + 225  # mm_acc wrapper + mm_core straight line
IRON_KS = 89
S_BANK = 306 / 9  # bankmodel: stripes 33,36,36,33,33,33,36,33,33 (C load+store in one bank, A slots 1,2 share C's bank 2)
S_DMA = 12  # in situ (g11-r8pb, ~283) minus mmb (271 for in-place C): DMA writes into the core's banks

# --- per tile, MEASURED (g11-r8pd stamps, us) -------------------------------------------------------------------------
HAND, ZERO, TAILC = 1.31, 0.49, 69 / F * 1e6
EPI = dict(plain=0.64, qkv=2.41, yx=3.32, gated=7.04, gelu_pan=7.75)
RINGX = dict(yx=5.04, qkv=1.23)  # r7p: the A broadcast couples the 3 rows (P23)

# name, kind, K, nt (tiles per core), epilogue kind, start-wait class
# start waits (us): 'g' = after a GEMM op (g11 stamps 27-34; mlp3_a 50), 'e' = after an EW op (stamps ~180-204 minus the
# norm itself ~124-148: ~55), 'a' = after attention (out_proj: g11 33)
BLOCK = [
    ("norm0", "norm", 124.0), ("yx", "gemm", 768, 8, "yx", 55.0), ("ssm", "ssm", 1115.0), ("out", "gemm", 768, 4, "plain", 55.0),
    ("norm1", "norm", 123.0), ("up_a", "gemm", 768, 16, "gated", 55.0), ("up_b", "gemm", 768, 16, "gated", 33.0),
    ("down_a", "gemm", 1536, 4, "plain", 27.0), ("down_b", "gemm", 1536, 4, "plain", 33.4), ("norm2", "norm", 148.0),
    ("qkv", "gemm", 768, 12, "qkv", 55.0), ("attn", "attn", 930.0), ("out_proj", "gemm", 768, 4, "plain", 33.0),
    ("norm3", "norm", 146.0), ("mlp0", "gemm", 768, 16, "gelu_pan", 55.0), ("mlp3_a", "gemm", 1536, 4, "plain", 50.0),
    ("mlp3_b", "gemm", 1536, 4, "plain", 33.0),
]
KS_BLOCK = sum(o[3] * o[2] // 64 for o in BLOCK if o[1] == "gemm")


def tiles_us(o, q):
    _, _, K, nt, epi, _ = o
    if epi == "gated":  # pairs: one epilogue per pair, handoff 0.03 inside a pair
        pairs = nt // 2
        return pairs * (q["hand"] + q["zero"] + q["tail"]) + pairs * (q["hand_pair"] + q["zero"] + q["tail"]) + pairs * q["epi"]["gated"]
    return nt * (q["hand"] + q["zero"] + q["tail"] + q["epi"][epi] + q["ringx"].get(o[0], 0.0))


def op_us(o, q):
    if o[1] == "gemm":
        _, _, K, nt, epi, L = o
        lw = q["L"].get(o[0], L) if isinstance(q.get("L"), dict) else L
        if q.get("L_scale") is not None:
            lw = q["L_scale"](o, lw)
        return lw + nt * (K // 64) * q["ks_cyc"] / F * 1e6 + tiles_us(o, q)
    return q["ew"].get(o[0], o[2])


def base_q(d=0.0):
    q = dict(hand=HAND, hand_pair=0.03, zero=ZERO, tail=TAILC, epi=dict(EPI), ringx=dict(RINGX), ew={}, L={})
    q["ks_cyc"] = 9 * (CALL + S_BANK + S_DMA) + IRON_KS + d
    return q


def block_us(q):
    return sum(op_us(o, q) for o in BLOCK)


def fit_d():
    q0 = base_q(0.0)
    rest = block_us(q0) - KS_BLOCK * q0["ks_cyc"] / F * 1e6
    ks_us = (BLOCK_US - rest) / KS_BLOCK
    return ks_us * F / 1e6 - q0["ks_cyc"], ks_us


def parts(q):
    p = dict(kstep=0.0, opstart=0.0, tilebnd=0.0, epilogue=0.0, ring=0.0, ssm=0.0, attn=0.0, norms=0.0)
    for o in BLOCK:
        if o[1] == "gemm":
            _, _, K, nt, epi, L = o
            full = op_us(o, q)
            ks = nt * (K // 64) * q["ks_cyc"] / F * 1e6
            tl = tiles_us(o, q)
            if epi == "gated":
                ep = (nt // 2) * q["epi"]["gated"]
                rg = 0.0
            else:
                ep = nt * q["epi"][epi]
                rg = nt * q["ringx"].get(o[0], 0.0)
            p["kstep"] += ks
            p["epilogue"] += ep
            p["ring"] += rg
            p["tilebnd"] += tl - ep - rg
            p["opstart"] += full - ks - tl
        else:
            p[{"ssm": "ssm", "attn": "attn", "norm": "norms"}[o[1]]] += op_us(o, q)
    return p


def step_ms(q, depth=12):
    return depth * block_us(q) / 1e3


def show(name, q, base=None):
    s = step_ms(q)
    print(f"  {name:58s} {s:6.1f} ms" + (f"  ({s - base:+5.1f})" if base else ""))
    return s


def main():
    d, ks_us = fit_d()
    q0 = base_q(d)
    print(f"sim11 on full12r8c: block {BLOCK_US:.0f} us, {KS_BLOCK} k-steps per core per block")
    print(f"  k-step static {9 * CALL + IRON_KS} + bank {9 * S_BANK:.0f} + DMA {9 * S_DMA} + fitted d {d:+.0f} = "
          f"{q0['ks_cyc']:.0f} cycles = {ks_us:.3f} us  (g11 PROF stamps: 3352 cycles incl. ~16-40 cycles per call of stamps)")
    p = parts(q0)
    tot = sum(p.values())
    print("  per block:")
    for k, v in p.items():
        print(f"    {k:9s} {v:7.1f} us  {12 * v / 1e3:5.2f} ms/step  {100 * v / tot:5.1f}%")
    base = show("today (fit)", q0)

    print("\nLevers, each alone (depth 12):")
    def ks(c):
        q = dict(q0)
        q["ks_cyc"] = c
        return q
    P20 = 9 * 192 - 13 + 22 + 22 + 6 + 6 + 25 + 18 + 108 + 19  # static call 1771 + IRON 25 + locks 18 + DMA 108 + B gap 19
    lv = []
    lv.append(("P20 + conflict-free pieces: k-step %d -> %d cycles" % (q0["ks_cyc"], P20), ks(P20)))
    lv.append(("  P20 staged: layout only (in-place C removed: -bank)", ks(q0["ks_cyc"] - 9 * S_BANK)))
    lv.append(("  P20 staged: + one call per k-step, stripes not pipelined", ks(q0["ks_cyc"] - 9 * S_BANK - 9 * 34 - 72 + 25)))
    q = dict(q0); q["ks_cyc"] = 1500
    lv.append(("P20 + 2x2 blocking (k-step ~1500; A port ~1364 below)", q))
    q = dict(q0); q["ew"] = dict(ssm=1115 - 350)
    lv.append(("SSM N1 tables in C++ (5140 -> ~2210 cycles/chunk)", q))
    q = dict(q0); q["ew"] = dict(ssm=1115 - 500)
    lv.append(("SSM N2 / N0 hand-scheduled (-> ~1000-1600 cycles/chunk)", q))
    q = dict(q0); q["ew"] = {f"norm{i}": 40.0 for i in range(4)}
    lv.append(("P7 norms folded (40 us each)", q))
    q = dict(q0); q["ew"] = {f"norm{i}": 0.0 for i in range(4)}; q["epi"] = {k: v + 0.3 for k, v in EPI.items()}
    lv.append(("P7 full (no norm ops; +0.3 us per tile of epilogue)", q))
    q = dict(q0); q["L_scale"] = lambda o, L: 17.0 if o[2] == 768 and L <= 34 else (L - 16.0 if o[2] == 768 else L)
    lv.append(("P5a K=768 slot loaded once (124 KB, -16 us)", q))
    # P5 full: the slot streamed in k-chunks (first k-step after ~5 us, unless the first tile outruns the load), the
    # sync after an EW op (~22 us = 55 - 33) stays; up_b / down_b / mlp3_b (A ready before the previous op ends) are
    # loaded into the other half while the previous op runs (ping-pong): ~2 us
    def p5(o, L, tks=None):
        if o[0] in ("up_b", "down_b", "mlp3_b"):
            return 2.0
        t = (tks or q0["ks_cyc"]) / F * 1e6
        load = 16.5 if o[2] == 768 else 33.0
        first = (o[2] // 64) * t
        return 5.0 + max(0.0, load - first) + max(0.0, L - 33.0 - (17.0 if o[0] == "mlp3_a" else 0.0))
    q = dict(q0); q["L_scale"] = p5
    lv.append(("P5 full: chunk-streamed + ping-pong for up_b/down_b/mlp3_b", q))
    q = dict(q0); q["ew"] = dict(attn=600.0)
    lv.append(("attention 0.93 -> 0.6 ms (P12/P19)", q))
    q = dict(q0); q["zero"] = 0.0
    lv.append(("acc_zero folded into the first k-step", q))
    q = dict(q0); q["hand"] = 0.3
    lv.append(("C handoff 1.31 -> 0.3 us (C object double-buffered)", q))
    q = dict(q0); q["ringx"] = {}
    lv.append(("P23 row balance (yx, qkv)", q))
    rows = [(step_ms(q) - base, n) for n, q in lv]
    for dl, n in rows:
        print(f"  {dl:+6.1f} ms  {n}")

    print("\nCombined:")
    q = dict(q0); q["ks_cyc"] = P20
    s1 = show("P20", q, base)
    q["ew"] = dict(ssm=765.0)
    show("+ SSM N1", q, base)
    q["L_scale"] = lambda o, L: p5(o, L, P20)
    show("+ P5", q, base)
    q["ew"] = dict(ssm=765.0, **{f"norm{i}": 40.0 for i in range(4)})
    show("+ P7 (40 us norms)", q, base)
    q["ew"]["attn"] = 600.0
    show("+ attention 0.6 ms", q, base)
    q["zero"], q["hand"], q["ringx"] = 0.0, 0.3, {}
    s_all = show("+ acc_zero fold, C double-buffer, P23   (ceiling, 4-chain GEMM)", q, base)
    q["ks_cyc"] = 1500
    q["ew"]["ssm"] = 615.0
    s_all2 = show("+ 2x2 GEMM, SSM hand-scheduled          (ceiling)", q, base)
    p = parts(q)
    tot = sum(p.values())
    print("  ceiling per block: " + ", ".join(f"{k} {v:.0f}" for k, v in p.items()) + f"  = {tot:.0f} us")
    for dep in (12, 10, 8):
        print(f"  ceiling depth {dep}: {step_ms(q, dep):.1f} ms ({1e3 / step_ms(q, dep):.1f} fps)")


if __name__ == "__main__":
    main()
