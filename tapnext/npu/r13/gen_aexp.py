"""Round 13 (npu-research P48): the attention's exp + bfp push of one row block as a modulo-scheduled asm loop
(mm8.cc aexp_push, bit-identical: the same MACs, conversions, clamps and table reads).

Per 64 scores s (one key block cb, two 32-lane halves h), with the row max m (64 lanes, 8 copies per row) and -m as an
fp32 accumulator:
    y  = -m + s * 1.0              (vmac.f on the pinned negm accumulator)       dh = bf16(y)        (vconv)
    c  = max(min(dh, HI), LO)       (vmin_ge / vmax_lt, constant first: NaN clamps to HI)
    i  = floor(c * 2^3)  (vfloor, s0 = 3: 16 B per 2^-2 segment at SU 8)       j = i + table bases + 44 * 16
    (off fp32, slope bf16) = LUT(j)  (4 x vldb.4x64 per 16 lanes, both table copies; vshuffle 5 / 24, vmov)
    q  = off - slope * m   (vmsc.f)        p = q + slope * s  (vmac.f, s reloaded on ldb)
    push(p0 | p1) as one bfp16ebs8 block (p1 moved into the high accumulator half by vmov cmh, cml); flush.

    extern "C" void aexpk(const bfloat16 *s /*p0*/, bfp16ebs8 *dst /*p1*/, const float *ab /*p2*/, const float *cd /*p3*/,
                          const bfloat16 *mrow /*p4*/, int trips /*r0*/)
s = the row block's 6 x 64 scores (768 B); dst = its 6 P panels (contiguous, 432 B); mrow = rows8(m) (128 B).
trips * U = 6. The push and flush sit in the last stage (the drain's garbage iterations never push); garbage loads read
up to (S - 1) x 128 B behind the scores (g_maxb / g_mbuf: data, harmless).

    python3 r13/gen_aexp.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

LDB_X = "VLDB_dmx_ldb_x_pstm_nrm_imm"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
MN, MX = "VMIN_GE_bf16", "VMAX_LT_bf16"
FLOOR = "VFLOOR_s32_bf16_mv_float_to_int_w"
VADD32 = "VADD_32"
LUT_LO, LUT_HI = "VLDB_4x64_lo", "VLDB_4x64_hi"
SHUF_BM, SHUF_X = "VSHUFFLE_vec_shuffle_bm", "VSHUFFLE_vec_shuffle_x"
VMOV_W = "VMOV_alu_mv_mv_w"
VMOV_CM = "VMOV_alu_mv_mv_cm"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"
FLUSH = "VST_FLUSH_512_CONV_normal_flush"
# constants (mm8.cc at::): clamp [AEXP_LO * SU, 8.9375 * SU] = [-88, 71.5]; index shift 3; bias 44 * 16; bf16 1.0
C_HI, C_LO, C_BASE, C_ONE, C_M0, C_M1 = 6, 7, 8, 9, 10, 11
X_FREE = [0, 1, 2, 3, 4, 5]
NEGM = [3, 4]  # dm registers holding -m (half 0 / 1) as fp32 accumulators (cml)
DM_FREE = [0, 1, 2]
HI_BITS, LO_BITS, ONE_BITS, SHIFT, BIAS_B = 17039, -15696, 16256, 3, 44 * 16  # bf16 0x428F = 71.5, 0xC2B0 = -88
CFG = "r1"  # 828


def ops():
    o = []
    for h in (0, 1):
        s, y, dh, t, c = f"s{h}", f"y{h}", f"dh{h}", f"t{h}", f"c{h}"
        o += [
            Op(f"L{h}", "ldb", f"vldb {{{s}.x}}, [p0], #64", defs=[(s, "x", 7)], mem=("ld", "s"), instr=LDB_X),
            Op(f"D{h}", "vec", f"vmac.f {{{y}.dm}}, dm{NEGM[h]}, {{{s}.x}}, x{C_ONE}, {CFG}", defs=[(y, "dm", 6)], uses=[(s, "x", 1)], instr=VMAC),
            Op(f"CX{h}", "st", f"vconv.bf16.fp32 {{{dh}.x}}, {{{y}.cml}}", defs=[(dh, "x", 2)], uses=[(y, "cml", 1)], instr=CONV),
            Op(f"MN{h}", "mv", f"vmin_ge.bf16 {{{t}.x}}, r16, x{C_HI}, {{{dh}.x}}", defs=[(t, "x", 2)], uses=[(dh, "x", 1)], instr=MN),
            Op(f"MX{h}", "mv", f"vmax_lt.bf16 {{{c}.x}}, r16, x{C_LO}, {{{t}.x}}", defs=[(c, "x", 2)], uses=[(t, "x", 1)], instr=MX),
        ]
        for q in (0, 1):  # 16-lane quarters of the half
            i, j, c0, c1 = f"i{h}{q}", f"j{h}{q}", f"c0{h}{q}", f"c1{h}{q}"
            w = "wl" if q == 0 else "wh"
            o += [
                Op(f"F{h}{q}", "st", f"vfloor.s32.bf16 {{{i}.x}}, {{{c}.{w}}}, s0", defs=[(i, "x", 2)], uses=[(c, w, 1)], instr=FLOOR),
                Op(f"A{h}{q}", "mv", f"vadd.32 {{{j}.x}}, {{{i}.x}}, x{C_BASE}", defs=[(j, "x", 2)], uses=[(i, "x", 1)], instr=VADD32),
                Op(f"LA{h}{q}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)], instr=LUT_LO),
                Op(f"LB{h}{q}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)], instr=LUT_HI),
                Op(f"LC{h}{q}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)], instr=LUT_LO),
                Op(f"LD{h}{q}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)], instr=LUT_HI),
                Op(f"SO{h}{q}", "mv", f"vshuffle {{off{h}.{'bmll' if q == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r3",
                   defs=[(f"off{h}", "bmll" if q == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=SHUF_BM),
                Op(f"SS{h}{q}", "mv", f"vshuffle {{{'sl' + str(h) if q == 0 else 'slope' + str(h)}.x}}, {{{c0}.x}}, {{{c1}.x}}, r2",
                   defs=[("sl" + str(h) if q == 0 else "slope" + str(h), "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=SHUF_X),
            ]
        sl, slope, off, qq, sr = f"sl{h}", f"slope{h}", f"off{h}", f"q{h}", f"sr{h}"
        p = "w" if h == 0 else "p1"
        o += [
            Op(f"MV{h}", "mv", f"vmov {{{slope}.wl}}, {{{sl}.wh}}", defs=[(slope, "wl", 2)], uses=[(sl, "wh", 1)], instr=VMOV_W),
            Op(f"Q{h}", "vec", f"vmsc.f {{{qq}.dm}}, {{{off}.dm}}, {{{slope}.x}}, x{C_M0 if h == 0 else C_M1}, {CFG}", defs=[(qq, "dm", 6)],
               uses=[(off, "cml", 4), (slope, "x", 1)], instr=VMSC),
            Op(f"R{h}", "ldb", f"vldb {{{sr}.x}}, [p5], #64", defs=[(sr, "x", 7)], mem=("ld", "sr"), instr=LDB_X),
            Op(f"P{h}", "vec", f"vmac.f {{{p}.dm}}, {{{qq}.dm}}, {{{slope}.x}}, {{{sr}.x}}, {CFG}", defs=[(p, "cml", 6)],
               uses=[(qq, "cml", 4), (slope, "x", 1), (sr, "x", 1)], instr=VMAC),
        ]
    o += [
        Op("MVH", "mv", "vmov {w.cmh}, {p1.cml}", defs=[("w", "cmh", 2)], uses=[("p1", "cml", 1)], instr=VMOV_CM),
        Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p1, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH),
        Op("FL", "st", "vst.flush.512.conv [p1, sf, r26]", after=[("PU", 1)], instr=FLUSH),
    ]
    return o


STREAMS = [("L0", "L1"), ("R0", "R1")]


def mk():
    o = ops()
    by = {x.name: x for x in o}
    for grp in STREAMS:
        for a, b in zip(grp, grp[1:]):
            by[b].after.append((a, 1))
    return Loop(o, consts={"x": [C_HI, C_LO, C_BASE, C_ONE, C_M0, C_M1], "dm": NEGM}, classes={"x": X_FREE, "dm": DM_FREE})


def maxdist(II):
    return [(g[0], g[-1], II - 1) for g in STREAMS] + [("PU", "FL", II - 1)]


def stream_ok(lp):
    S = lp.stages()
    return all(lp.by_name[n].t // lp.II == S - 1 for n in ("PU", "FL"))


def build(II=None, times=None, cp=None, max_ii=30):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        assert stream_ok(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 20, (II or max_ii) + 1):
        lp = mk()
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=["PU", "FL"], max_life=2 * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        assert stream_ok(lp)
        lp.times = {"II": ii, "t": res[0]}
        return lp
    raise Infeasible("none")


def check(lp, n=6):
    n -= n % lp.U
    st = {"s": (lambda i, k: i * 128 + 64 * k, dict(base=0x73C00, dn=1 << 20, dj=64, m=64)),
          "sr": (lambda i, k: i * 128 + 64 * k, dict(base=0x73C00, dn=1 << 20, dj=64, m=64))}
    return lp.check(n, st)


def function(lp, name="aexpk", drain=8):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        "movxm\tr16, #52428", "mov\tr1, p2", "vbcst.32\tx0, r1", "mov\tr1, p3", "vbcst.32\tx1, r1",
        f"movxm\tr1, #{BIAS_B}", "vbcst.32\tx2, r1", "vsel.32\tx3, x0, x1, r16", "nop", "nop", f"vadd.32\tx{C_BASE}, x3, x2",
        f"movxm\tr1, #{HI_BITS}", f"vbcst.16\tx{C_HI}, r1", f"movxm\tr1, #{LO_BITS}", f"vbcst.16\tx{C_LO}, r1",
        f"movxm\tr1, #{ONE_BITS}", f"vbcst.16\tx{C_ONE}, r1",
        "movxm\tr1, #-16512", "vbcst.16\tx4, r1",  # bf16 -1.0
        f"vlda\tx{C_M0}, [p4, #0]", f"vlda\tx{C_M1}, [p4, #64]",
        f"mova\t{CFG}, #828", "mova\tr2, #24", "mova\tr3, #5", f"mov\ts0, #{SHIFT}", "mova\tr26, #0",
        "mov\tp5, p0",  # the reload stream starts at the scores too
        "nop", "nop", "nop", "nop", "nop", "nop",
        f"vmul.f\tdm{NEGM[0]}, x{C_M0}, x4, {CFG}", f"vmul.f\tdm{NEGM[1]}, x{C_M1}, x4, {CFG}",
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r0, #0", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def report(lp):
    print(f"II {lp.II} (resource bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:8])}), U {lp.U}, "
          f"stages {lp.stages()}, maxlive {lp.maxlive()}")
    for o in sorted(lp.ops, key=lambda o: o.t):
        print(f"  t {o.t:3d} (stage {o.t // lp.II}, slot {o.t % lp.II:2d}) {o.slot} {o.name:5s} {lp.text(o, 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--func", default="aexpk")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time))
    report(lp)
    cyc, land = check(lp)
    print(f"check OK: {cyc} cycles for 6 key blocks, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r13/gen_aexp.py\n#define AEXPK_U {lp.U}\n#define AEXPK_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
