"""Round 15 (npu-research P29c, the recurrence): the RG-LRU state update as one modulo-scheduled asm loop (EW cores, ew8.cc ew_ssm).

Today (ew8.cc ew_ssm, ~900 cycles per chunk in situ, r15 PROF), per 32 lanes i (= 2 cb + half):
    gated = bf(xc gx); nx = bf(gated mult)
    h0h = bf(h0); h0l = bf(h0 - h0h 1)                      (h0: fp32 state from the input ring)
    h = a h0h + a h0l + nx 1;  hh = bf(h); hl = bf(h - hh 1)
    hsel = hh rest + hl rest + gated first                  (first / rest: 1 / 0 per head and frame)
    hout = fp32(hsel) (128 B), hb = bf(hsel)
The same MACs and roundings here (vmac x, -1 -> vmsc x, 1.0: exact), one vector per iteration. The h state is kept in VECTOR
order ([cb][half][32 fp32] = 128 B per vector) instead of [half][cb]: only the SSM reads its own state (the DMA moves the bytes
opaquely), so every value is unchanged (bit-identical tokens).
Streams: one pointer p0 at g_e2 with dj offsets for mult / xc / gx / a (dj0..dj3) and one padda per iteration; h0 by p1 (2D: the
input ring's two 1 KB parts, a jump after 16 loads); hb by p3, hout by p4.

    extern "C" void reck(const bfloat16 *e2 /*p0: g_e2; xc, gx, a at fixed offsets*/, const float *h5 /*p1*/, const float *h6 /*p2*/,
                         bfloat16 *hb /*p3*/, float *hout /*p4*/, int first_bits /*r0: bf16 of first*/, int rest_bits /*r1*/)

    python3 r15/gen_rec.py OUT.s [--ii N] [--cp F.json | --times F.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

NV = 16
C_ONE, C_REST, C_FIRST = 11, 10, 9
X_FREE = list(range(9))
DM_FREE = [0, 1, 2, 3, 4]
OFF = dict(mu=0, xc=1088, gx=2112, a=3136)  # from g_e2 (arena + 10240): g_xc 11328, g_gx 12352, g_ga 13376
DJ = dict(mu=0, xc=1, gx=2, a=3)
BANK = dict(v=(1,), h0=(3,), hb=(1,), ho=(2,))  # in situ: arena 10240..14400 bank 1, the input ring bank 3, g_hout bank 2
R_MAC = "r2"  # 828 (r0 / r1 carry the arguments)
# round 18 (HRING): h straight into the EW output ring (no out_copy of g_hout). The 2 KB of h may cross the 9 KB ring's end (one
# chunk in three under XROT: chunks of 3 KB start at 4608 + 3072 k), so the second store of each vector post-modifies by the 2D
# modifier d5 (+64; at the wrap 64 - OUT_B), its counter started by the caller (r2: 16 - vectors before the end, 0 = no wrap)
RING = False
OUT_B = 9216
VST_BM2 = "VST_2D_dmx_sts_bm"

VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
LDA_IDX = "VLDA_dmx_lda_x_idx"
LDB_IDX = "VLDB_dmx_ldb_x_idx"
LDA_BM2 = "VLDA_2D_dmx_lda_bm"
ST_BM = "VST_dmx_sts_bm_pstm_nrm_imm"
ST_CONV = "VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm"
PADDA = "PADDA_pstm_nrm_imm"


def ops(vslots="abab"):
    o = []
    for k, nm in enumerate(("mu", "xc", "gx", "a")):
        s = vslots[k]
        o.append(Op(f"L{nm}", "lda" if s == "a" else "ldb", f"vld{s} {{{nm}.x}}, [p0, dj{DJ[nm]}]", defs=[(nm, "x", 7)],
                    mem=("ld", "v"), banks=BANK["v"], instr=LDA_IDX if s == "a" else LDB_IDX))
    o.append(Op("PA", "lda", "padda [p0], #64", mem=("pa", "v"), after=[(f"L{nm}", 1) for nm in ("mu", "xc", "gx", "a")], instr=PADDA))
    o.append(Op("LH0", "lda", "vlda.2d {h0.bmll}, [p1], d4", defs=[("h0", "bmll", 7)], mem=("ld", "h0"), banks=BANK["h0"], instr=LDA_BM2))
    o.append(Op("LH1", "lda", "vlda.2d {h0.bmlh}, [p1], d4", defs=[("h0", "bmlh", 7)], mem=("ld", "h0"), banks=BANK["h0"], instr=LDA_BM2,
                after=[("LH0", 1)]))

    def mul(name, d, a, b, op="vmul.f", acc=None):
        ins = {"vmul.f": VMUL, "vmac.f": VMAC, "vmsc.f": VMSC}[op]
        A = lambda v: f"x{v}" if isinstance(v, int) else f"{{{v}.x}}"  # noqa: E731
        uses = [(v, "x", 1) for v in (a, b) if not isinstance(v, int)]
        if acc:
            o.append(Op(name, "vec", f"{op} {{{d}.dm}}, {{{acc}.dm}}, {A(a)}, {A(b)}, {R_MAC}", defs=[(d, "dm", 6)],
                        uses=[(acc, "cml", 4)] + uses, instr=ins))
        else:
            o.append(Op(name, "vec", f"{op} {{{d}.dm}}, {A(a)}, {A(b)}, {R_MAC}", defs=[(d, "dm", 6)], uses=uses, instr=ins))

    def conv(name, d, a):
        o.append(Op(name, "st", f"vconv.bf16.fp32 {{{d}.x}}, {{{a}.cml}}", defs=[(d, "x", 2)], uses=[(a, "cml", 1)], instr=CONV))

    mul("G", "ga", "xc", "gx")
    conv("CG", "gt", "ga")
    mul("N", "na", "gt", "mu")
    conv("CN", "nx", "na")
    conv("CH", "h0h", "h0")
    mul("MS", "dl", "h0h", C_ONE, op="vmsc.f", acc="h0")
    conv("CL", "h0l", "dl")
    mul("H1", "e1", "a", "h0h")
    mul("H2", "e2", "a", "h0l", op="vmac.f", acc="e1")
    mul("H3", "e3", "nx", C_ONE, op="vmac.f", acc="e2")
    conv("CHH", "hh", "e3")
    mul("MS2", "fl", "hh", C_ONE, op="vmsc.f", acc="e3")
    conv("CHL", "hl", "fl")
    mul("Y1", "y1", "hh", C_REST)
    mul("Y2", "y2", "hl", C_REST, op="vmac.f", acc="y1")
    mul("Y3", "y", "gt", C_FIRST, op="vmac.f", acc="y2")
    o.append(Op("SO0", "st", "vst {y.bmll}, [p4], #64", uses=[("y", "bmll", 1)], mem=("st", "ho"), banks=BANK["ho"], instr=ST_BM))
    if RING:
        o.append(Op("SO1", "st", "vst.2d {y.bmlh}, [p4], d5", uses=[("y", "bmlh", 1)], mem=("st", "ho"), banks=BANK["ho"], instr=VST_BM2,
                    after=[("SO0", 1)]))
    else:
        o.append(Op("SO1", "st", "vst {y.bmlh}, [p4], #64", uses=[("y", "bmlh", 1)], mem=("st", "ho"), banks=BANK["ho"], instr=ST_BM,
                    after=[("SO0", 1)]))
    o.append(Op("SB", "st", "vst.conv.bf16.fp32 {y.cml}, [p3], #64", uses=[("y", "cml", 1)], mem=("st", "hb"), banks=BANK["hb"], instr=ST_CONV))
    return topo(o)


def topo(o):
    defined, out, rest = set(), [], list(o)
    while rest:
        for x in rest:
            if all(v in defined for v, *_ in x.uses):
                out.append(x)
                defined |= {v for v, *_ in x.defs}
                rest.remove(x)
                break
        else:
            raise ValueError("cycle")
    return out


def mk(vslots="abab"):
    return Loop(ops(vslots), consts={"x": [C_ONE, C_REST, C_FIRST]}, classes={"x": X_FREE, "dm": DM_FREE})


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def maxdist(II):
    return [(f"L{nm}", "PA", II - 1) for nm in ("mu", "xc", "gx", "a")] + [("LH0", "LH1", II - 1), ("SO0", "SO1", II - 1)]


def build(II=None, times=None, cp=None, max_ii=16, vslots="abab", U=4):
    if times is not None:
        lp = mk(times.get("vslots", vslots))
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U", U))
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 10, (II or max_ii) + 1):
        lp = mk(vslots)
        res = cpsched.solve(lp, ii, last_stage=["SO0", "SO1", "SB"], max_life=U * ii, H=8 * ii, maxdist=maxdist(ii), **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(U)
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        side_effects_last(lp)
        lp.times = {"II": ii, "t": res[0], "vslots": vslots, "U": U}
        return lp
    raise Infeasible("none")


JUMP = 0x3000  # checker: P[6] 12 KB after P[5]


def check(lp, n=NV, wrap=None):
    """wrap (RING): the h stores cross the ring end after `wrap` vectors"""
    noop = dict(dn=0, dj=0, m=0)
    vops = {f"L{nm}": noop for nm in ("mu", "xc", "gx", "a")}
    vops["PA"] = dict(dn=0, dj=0, m=64)
    h0 = lambda i, k: (128 * i + 64 * k) if i < 8 else (JUMP + 128 * (i - 8) + 64 * k)  # noqa: E731
    st = {"v": (lambda i: 64 * i, dict(base=0x10000, dn=0, dj=0, m=0, ops=vops)),
          "h0": (h0, dict(base=0x20000, dn=15, dj=64, m=JUMP - 15 * 64)),
          "hb": (lambda i: 64 * i, dict(base=0x30000, dn=0, dj=64, m=64)),
          "ho": (lambda i, k: 128 * i + 64 * k, dict(base=0x40000, dn=0, dj=64, m=64))}
    if RING:
        w = NV if wrap is None else wrap
        st["ho"] = (lambda i, k: 128 * i + 64 * k - (OUT_B if i >= w else 0),
                    dict(base=0x40000, dn=0, dj=64, m=64, ops={"SO1": dict(dn=15, dj=64, m=64 - OUT_B, dc0=(16 - w) % 16)}))
    return lp.check(n, st)


def function(lp, name="reck"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = (["mov\tdc5, r2", "mova\tdn5, #15", "mova\tdj5, #64", f"movxm\tm5, #{64 - OUT_B}"] if RING else []) + [
        f"vbcst.16\tx{C_FIRST}, r0", f"vbcst.16\tx{C_REST}, r1", "movxm\tr0, #16256", f"vbcst.16\tx{C_ONE}, r0", f"mova\t{R_MAC}, #828",
        # h0: 16 x 64 B in P[5], then P[6]: m4 = h6 - h5 - 15 x 64
        "mov\tr0, p2", "mov\tr1, p1", "sub\tr0, r0, r1", "movxm\tr1, #-960", "add\tr0, r0, r1", "mov\tm4, r0",
        "mova\tdn4, #15", "mova\tdj4, #64", "mova\tdc4, #0",
    ] + [f"movxm\tr0, #{OFF[nm]}" if OFF[nm] > 511 else f"mova\tr0, #{OFF[nm]}" for nm in ()] + []
    for nm in ("mu", "xc", "gx", "a"):
        pre += [f"movxm\tr0, #{OFF[nm]}", f"mov\tdj{DJ[nm]}, r0"]
    pre += [f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NV // lp.U}", "add.nc\tlc, r0, #0"]
    # (no nops: the prologue's (S - 1) II bundles give the zero-overhead loop its setup distance)
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--vslots", default="abab")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--ring", action="store_true", help="round 18 HRING: h into the output ring, 2D wrap on the second store")
    o = ap.parse_args()
    global RING
    RING = o.ring
    if RING:
        BANK["ho"] = (0,)  # the EW output ring (bank 0; no other reck stream uses it)
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), vslots=o.vslots)
    assert lp.stages() >= 2, "the prologue must give the loop setup distance"
    cyc, land = check(lp)
    if RING:
        for w in (12, 4, 1, 15):
            check(lp, wrap=w)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NV} vectors)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "reck", o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r15/gen_rec.py\n#define RECK_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
