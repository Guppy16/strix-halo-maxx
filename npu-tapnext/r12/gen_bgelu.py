"""Round 12 (npu-research G1b, kernel 1): bias + GELU fused into one modulo-scheduled loop over the fp32 pieces.

Today an in-place GELU epilogue (gated's gate half, mlp0) runs bias_pcs (x = bf16(acc + bias), written to the piece's
start) and then gelu_ms over those bf16 vectors. Here, per 32 lanes:
    AC0 / AC1  vlda.2d bm x 2         the fp32 accumulator (128 B, contiguous over the run's pieces)
    BL         vldb.2d                 the bias vector (the pair's 8 vectors, re-walked per piece)
    BM         vmac.f y = acc + bias * 1.0
    CX         vconv x = bf16(y)       -> the clamp / index chain (gen_gelu's ops with flat-edge tables)
    SX         vst.2d.conv bf16(y)     -> x at its final place (piece start + 64 i, as bias_pcs writes it)
    R*         x reloaded from there for the s MAC and the products (no x register held across the chain)
    ST         the GELU result over it
Bit-identical to bias_pcs + gelu_lut_i: the same MAC, the same roundings; flat-edge tables (r12/lutflat.py) return the
clamped edge value exactly.

    extern "C" void bgelu(const float *acc /*p0*/, bfloat16 *out /*p1*/, int trips /*r0*/, const float *ab /*p2*/,
                          const float *cd /*p3*/, const bfloat16 *bias /*p4, moved to p5*/)
`out` = the run's first piece (the bf16 lands at each piece's start: 8 vectors of 64 B, then +576); acc = the same pieces.

    python3 r12/gen_bgelu.py OUT.s [--rel S:b,PQ:b] [--ii N] [--cp F.json | --times F.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE))
import gen_gelu as G  # noqa: E402
from msched import MEM_R, MEM_W, Infeasible, Loop, Op  # noqa: E402

LDA_BM = "VLDA_2D_dmx_lda_bm"
LDB_X = "VLDB_2D_dmx_ldb_x"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
STCONV = "VST_2D_CONV_bf16_fp32_dmx_sts_srs_bf"
C_ONE = 11  # bf16 1.0, in gen_gelu's -1.0 register: D becomes vmsc s - sh * 1.0 (exact either way: bit-identical)
X_FREE = [0, 1, 2, 3, 4, 5, 6, 7]
RING = 8  # x slots (64 B) on the stack: more than the iterations between an SX and its last reload
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"


def ops(rel):
    g = G.ops(None, flat=True, rel=rel)
    g = [o for o in g if o.name != "L"]  # x now comes from the bias MAC
    head = [
        Op("AC0", "lda", "vlda.2d {acc.bmll}, [p0], d0", defs=[("acc", "bmll", 7)], mem=("ld", "acc"), instr=LDA_BM),
        Op("AC1", "lda", "vlda.2d {acc.bmlh}, [p0], d0", defs=[("acc", "bmlh", 7)], mem=("ld", "acc"), instr=LDA_BM, after=[("AC0", 1)]),
        Op("BL", "ldb", "vldb.2d {b.x}, [p5], d5", defs=[("b", "x", 7)], mem=("ld", "bias"), instr=LDB_X),
        Op("BM", "vec", f"vmac.f {{y.dm}}, {{acc.dm}}, {{b.x}}, x{C_ONE}, r1", defs=[("y", "dm", 6)],
           uses=[("acc", "cml", 4), ("b", "x", 1)], instr=VMAC),
        Op("CX", "st", "vconv.bf16.fp32 {x.x}, {y.cml}", defs=[("x", "x", 2)], uses=[("y", "cml", 1)], instr=CONV),
        # x through a ring of RING slots on the stack, NOT the output: SX runs in an early stage, so the drain's garbage
        # iterations execute it too (round 12: writing x to its final place corrupted the 4 vectors past each run's end)
        Op("SX", "st", "vst.2d.conv.bf16.fp32 {y.cml}, [p1], d1", uses=[("y", "cml", 1)], mem=("st", "ring"), spill=("x", "x"),
           instr=STCONV),
    ]
    out = head + g
    by = {o.name: o for o in out}
    d = by["D"]
    d.fmt = d.fmt.replace("vmac.f", "vmsc.f").replace(f"x{G.C_M1}", f"x{C_ONE}")
    d.instr, d.ports = VMSC, __import__("msched").ports(VMSC)
    for o in out:
        if o.name.startswith("R") and o.name[1:].isdigit():  # x reloads read SX's store: through memory, checked
            k = int(o.name[1:])
            o.fill = ("x", "x")
            o.after.append(("SX", MEM_W - MEM_R + 1))  # the stored bytes are in memory before the load reads
    for o in out:
        if o.name.startswith("R") and o.name[1:].isdigit():
            o.mem = ("ld", f"ring{o.name[1:]}")
    p, d = G.RELOAD_PTRS[len(G.parse_rel(rel))]  # the result's own pointer: the first reload pointer not used (p6 / p7 are callee-saved)
    by["ST"].fmt = by["ST"].fmt.replace("[p1], d1", f"[{p}], {d}")
    return out


def mk(rel):
    lp = Loop(ops(rel), consts={"x": [G.C_HI, G.C_LO, G.C_BASE, C_ONE]}, classes={"x": X_FREE, "dm": [0, 1, 2, 3, 4]})
    lp.flat, lp.rel = True, G.parse_rel(rel)
    return lp


def maxdist(II, rel):
    # a ring slot is rewritten RING iterations later: every reload of x within RING - 1 IIs of its SX
    return [("AC0", "AC1", II - 1)] + [("SX", f"R{k}", (RING - 1) * II) for k in range(len(G.parse_rel(rel)))]


def side_effects_last(lp):
    """every store outside the ring in the last stage (the drain's garbage iterations run the earlier stages)"""
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.mem[1] != "ring" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def build(rel, II=None, times=None, cp=None, max_ii=16):
    if times is not None:
        lp = mk(rel)
        lp.place(times["II"], times["t"])
        lp.allocate()
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 10, (II or max_ii) + 1):
        lp = mk(rel)
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii, rel), last_stage=["ST"], **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        side_effects_last(lp)
        lp.times = {"II": ii, "t": res[0]}
        return lp
    raise Infeasible("none")


def check(lp, n=None):
    n = n or 8 * 3
    n -= n % lp.U
    out = lambda i: (i // 8) * 1024 + (i % 8) * 64  # noqa: E731
    ring = lambda i: (i % RING) * 64  # noqa: E731
    rp = dict(base=0x40000, dn=RING - 1, dj=64, m=-64 * (RING - 1))
    st = {"acc": (lambda i, k: i * 128 + 64 * k, dict(base=0x10000, dn=15, dj=64, m=64)),
          "bias": (lambda i: (i % 8) * 64, dict(base=0x30000, dn=7, dj=64, m=-448)),
          "ring": (ring, dict(rp)),
          "out": (out, dict(base=0x10000, dn=7, dj=64, m=576))}
    for k in range(len(lp.rel)):
        st[f"ring{k}"] = (ring, dict(rp))
    return lp.check(n, st)


def function(lp, name="bgelu", drain=6):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        "mov\tp5, p4",  # bias: the 5th pointer argument arrives in p4, which the first reload stream takes below
        "movxm\tr16, #52428",
        "mov\tr1, p2", "vbcst.32\tx0, r1", "mov\tr1, p3", "vbcst.32\tx1, r1",
        f"mova\tr1, #{G.FLAT_BIAS_B}", "vbcst.32\tx2, r1", "vsel.32\tx3, x0, x1, r16", "nop", "nop", f"vadd.32\tx{G.C_BASE}, x3, x2",
        f"movxm\tr1, #{G.IDX_HI}", f"vbcst.16\tx{G.C_HI}, r1", f"movxm\tr1, #{G.IDX_LO}", f"vbcst.16\tx{G.C_LO}, r1",
        "movxm\tr1, #16256", f"vbcst.16\tx{C_ONE}, r1",
        "mova\tr1, #828", "mova\tr2, #24", "mova\tr3, #5", "mov\ts0, #7",
        # acc: contiguous fp32; bias: 8 vectors per piece; x / reloads / result: 8 x 64 B per piece, then +576
        "mova\tdn0, #15", "mova\tdj0, #64", "mova\tm0, #64", "mova\tdc0, #0",
        "mova\tdn5, #7", "mova\tdj5, #64", "movxm\tm5, #-448", "mova\tdc5, #0",
    ]
    nr = len(lp.rel)
    p, d = G.RELOAD_PTRS[nr]  # the result: 8 x 64 B per piece, then +576, from the `out` argument
    pre += [f"mov\t{p}, p1", f"mova\tdn{d[1]}, #7", f"mova\tdj{d[1]}, #64", f"movxm\tm{d[1]}, #576", f"mova\tdc{d[1]}, #0"]
    # the x ring: RING x 64 B on the stack (grows up; the space is at the old sp), SX on p1 / d1, each reload its own pointer
    pre += [f"paddxm\t[sp], #{64 * RING}", "mov\tp1, sp", f"padda\t[p1], #-{64 * RING}",
            f"mova\tdn1, #{RING - 1}", "mova\tdj1, #64", f"movxm\tm1, #{-64 * (RING - 1)}", "mova\tdc1, #0"]
    for k in range(nr):
        p, d = G.RELOAD_PTRS[k]
        pre += [f"mov\t{p}, p1", f"mova\tdn{d[1]}, #{RING - 1}", f"mova\tdj{d[1]}, #64", f"movxm\tm{d[1]}, #{-64 * (RING - 1)}",
                f"mova\tdc{d[1]}, #0"]
    pre += [f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r0, #0", "nop", "nop", "nop", "nop", "nop", "nop"]
    # p2 / p3 hold the table bases until x0 / x1 are formed, then serve as reload pointers: the vbcst above reads them first
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", f"\tpaddxm\t[sp], #-{64 * RING}", "\tnop", f".Lfunc_end_{name}:",
          f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rel", default="S:b,PQ:b", help="at most 2 reloads (the third pointer takes the result)")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--func", default="bgelu")
    ap.add_argument("--fit", action="store_true", help="round 13: compact the loop body (r12/asmfit.py)")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    rel = times.get("rel", o.rel) if times else o.rel  # a saved schedule carries its reload variant
    lp = build(rel, o.ii, times, cp=dict(time_limit=o.time))
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, 24 vectors)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(dict(lp.times, rel=rel), indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r12/gen_bgelu.py\n#define BGELU_U {lp.U}\n#define BGELU_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
