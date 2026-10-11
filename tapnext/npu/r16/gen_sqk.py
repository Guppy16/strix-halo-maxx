"""Round 16: the SSM's mult = sqrt(1 - a^2) as one modulo-scheduled asm loop (EW cores, ew8.cc ew_ssm; flag -DSQK).

Replaces ssm9::sqrt_lut_n (r9/ssm9.h sqrt_lut_i: Peano's loop runs ~58 bundles per 2 vectors unpipelined, 493 cycles per
chunk in situ). Exact RNE bf16 sqrt of w >= 0, bit-identical to sqrt_lut_i on every non-negative bf16 (the bench sweeps all
32,768 patterns 0x0000-0x7FFF). Per 32 lanes, b = bits(w), e = b >> 7 = 2 j + p:
    idx = b & 0xFF (exponent parity p, 7 mantissa bits); t' = LUT(idx) (int16 table, Peano's load_lut_2x_int8 sequence:
          ups by 2 -> int32 byte offsets, two vmov out of the accumulator, + the (ab, cd) lane bases, 4 vldb.4x16 per
          16 lanes, three shuffles)
    k7  = (b & 0xFF00) >> 1 = j << 7            (vband, vups, vsrs: exact in any rounding mode, the low bit is 0)
    r   = t' + k7                               (vadd.16)
    out = min(r, bf16(w 2^60))                  (w = 0 -> 0; 2^60 w >= 1 >= r otherwise)
The table folds the exponent bias in: t'[idx] = t[idx] - ((64 - p) << 7), t[idx] = bf16 bits of sqrt(2^(1-p) M) (r9/ssmlut.py),
so t' + (j << 7) = t + (floor((e - 127) / 2) << 7), sqrt_lut_i's ((b >> 7) - 127) >> 1 exponent without its 3 ups / srs
pairs. The tables replace ssqrt_ab / ssqrt_cd (same size and banks; r9/ssm_lut.h skips its own under SSQRT_NONE).

    extern "C" void sqk(bfloat16 *v /*p0: in place, 16 vectors*/, const int16_t *ab /*p1*/, const int16_t *cd /*p2*/)

    python3 r16/gen_sqk.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]      (OUT.h gets SQK_TAB)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

NV = 16
C_FF, C_HI, C_BASE, C_2E60 = 8, 9, 10, 11  # x constants: 0x00FF, 0xFF00, the (ab, cd) lane bases, bf16 2^60
X_FREE = list(range(8))
DM_FREE = [0, 1, 2, 3, 4]
BANK = dict(v=(1,), tab=(0, 3))  # in situ: g_e2 (arena + 10240 = 0x77200) bank 1; ssqrt_ab bank 0, ssqrt_cd bank 3

VLDA = "VLDA_dmx_lda_x_pstm_nrm_imm"
VST = "VST_dmx_sts_x_pstm_nrm_imm"
VBAND = "VBAND"
UPS = "VUPS_2x_mv_ups_x2c_upsSign1"
SRS = "VSRS_2x_mv_x_srs_cm_srsSign1"
VADD32 = "VADD_32"
VADD16 = "VADD_16"
SHUF = "VSHUFFLE_vec_shuffle_x"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
MIN = "VMIN_GE_bf16"
LUT = {"lo": "VLDB_4x16_lo", "hi": "VLDB_4x16_hi"}


def bf16_bits(v):
    """float32 -> bf16 bits, round to nearest even (finite inputs)"""
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def table():
    """t' in r9/ssmlut.py's layout (runs of 16 entries stored twice): 512 int16"""
    idx = np.arange(256)
    m = 1 + (idx & 0x7F) / 128.0
    par = (idx >> 7) & 1
    t = bf16_bits(np.sqrt(np.where(par == 1, 1.0, 2.0) * m).astype(np.float32)).astype(np.int32)
    tp = (t - ((64 - par) << 7)).astype(np.int16)
    g = tp.reshape(-1, 1, 16)
    return t, np.concatenate([g, g], 1).reshape(-1)


def sqrt_ops(o, w, out, bk, tag="", k=None, sh=("s0", "s1", "s2"), rm=("r3", "r4")):
    """append the ops computing vreg `out` = sqrt(w) (w: a bf16 x vreg). Scalar config: r1 828, rm = the shuffle mode
    registers (24, 10), sh = the shift registers (2, 0, 1). k: constant -> vreg name (a reloaded constant, fusion) for
    'ff', 'hi', 'base', 'b60'; missing ones are the physical x constants C_FF, C_HI, C_BASE, C_2E60. tag: suffix for
    op / vreg names"""
    T = lambda s: s + tag  # noqa: E731
    k = k or {}
    phys = dict(ff=C_FF, hi=C_HI, base=C_BASE, b60=C_2E60)

    def K(name):  # (operand text, uses)
        return (f"{{{k[name]}.x}}", [(k[name], "x", 1)]) if name in k else (f"x{phys[name]}", [])

    ix, iu, a0, a1, j0, j1 = (T(v) for v in ("ix", "iu", "a0", "a1", "j0", "j1"))
    kt, ku = K("ff")
    o.append(Op(T("B1"), "mv", f"vband {{{ix}.x}}, {{{w}.x}}, {kt}", defs=[(ix, "x", 2)], uses=[(w, "x", 1)] + ku, instr=VBAND))
    o.append(Op(T("U1"), "mv", f"vups.2x {{{iu}.cml}}, {{{ix}.x}}, {sh[0]}, upssign1", defs=[(iu, "cml", 3)], uses=[(ix, "x", 1)], instr=UPS))
    for nm, v, q in (("M1", a0, "bmll"), ("M2", a1, "bmlh")):  # vmov x <- bm: its own itinerary (DM_RM_L0_PORT, round 15)
        o.append(Op(T(nm), "mv", f"vmov {{{v}.x}}, {{{iu}.{q}}}", defs=[(v, "x", 2)], uses=[(iu, q, 1)]))
        o[-1].ports = [(0, "DM_RM_L0_PORT")]
    kt, ku = K("base")
    o.append(Op(T("A0"), "mv", f"vadd.32 {{{j0}.x}}, {{{a0}.x}}, {kt}", defs=[(j0, "x", 2)], uses=[(a0, "x", 1)] + ku, instr=VADD32))
    o.append(Op(T("A1"), "mv", f"vadd.32 {{{j1}.x}}, {{{a1}.x}}, {kt}", defs=[(j1, "x", 2)], uses=[(a1, "x", 1)] + ku, instr=VADD32))
    cs = []
    for h, j in ((0, j0), (1, j1)):
        for q, half in ((0, "wl"), (1, "wh")):
            c = T(f"c{h}{q}")
            cs.append(c)
            for lh, part in (("lo", "wl"), ("hi", "wh")):
                o.append(Op(T(f"L{h}{q}{lh}"), "ldb", f"vldb.4x16.{lh} {{{c}.{part}}}, {{{j}.{half}}}", defs=[(c, part, 7)],
                            uses=[(j, half, 1)], banks=bk("tab"), instr=LUT[lh]))
    s0, s1, t, hb, hu, k7, r, zz, zb = (T(v) for v in ("s0", "s1", "t", "hb", "hu", "k7", "r", "zz", "zb"))
    o.append(Op(T("S0"), "mv", f"vshuffle {{{s0}.x}}, {{{cs[0]}.x}}, {{{cs[1]}.x}}, {rm[0]}", defs=[(s0, "x", 2)],
                uses=[(cs[0], "x", 1), (cs[1], "x", 1)], instr=SHUF))
    o.append(Op(T("S1"), "mv", f"vshuffle {{{s1}.x}}, {{{cs[2]}.x}}, {{{cs[3]}.x}}, {rm[0]}", defs=[(s1, "x", 2)],
                uses=[(cs[2], "x", 1), (cs[3], "x", 1)], instr=SHUF))
    o.append(Op(T("S2"), "mv", f"vshuffle {{{t}.x}}, {{{s0}.x}}, {{{s1}.x}}, {rm[1]}", defs=[(t, "x", 2)],
                uses=[(s0, "x", 1), (s1, "x", 1)], instr=SHUF))
    kt, ku = K("hi")
    o.append(Op(T("B2"), "mv", f"vband {{{hb}.x}}, {{{w}.x}}, {kt}", defs=[(hb, "x", 2)], uses=[(w, "x", 1)] + ku, instr=VBAND))
    o.append(Op(T("U2"), "mv", f"vups.2x {{{hu}.cml}}, {{{hb}.x}}, {sh[1]}, upssign1", defs=[(hu, "cml", 3)], uses=[(hb, "x", 1)], instr=UPS))
    o.append(Op(T("R2"), "st", f"vsrs.2x {{{k7}.x}}, {{{hu}.cml}}, {sh[2]}, srssign1", defs=[(k7, "x", 4)], uses=[(hu, "cml", 1)], instr=SRS))
    o.append(Op(T("AD"), "mv", f"vadd.16 {{{r}.x}}, {{{t}.x}}, {{{k7}.x}}", defs=[(r, "x", 2)], uses=[(t, "x", 1), (k7, "x", 1)], instr=VADD16))
    kt, ku = K("b60")
    o.append(Op(T("VM"), "vec", f"vmul.f {{{zz}.dm}}, {{{w}.x}}, {kt}, r1", defs=[(zz, "dm", 6)], uses=[(w, "x", 1)] + ku, instr=VMUL))
    o.append(Op(T("CV"), "st", f"vconv.bf16.fp32 {{{zb}.x}}, {{{zz}.cml}}", defs=[(zb, "x", 2)], uses=[(zz, "cml", 1)], instr=CONV))
    o.append(Op(T("MN"), "mv", f"vmin_ge.bf16 {{{out}.x}}, r16, {{{r}.x}}, {{{zb}.x}}", defs=[(out, "x", 2)],
                uses=[(r, "x", 1), (zb, "x", 1)], instr=MIN))


def ops(banks=True):
    bk = (lambda n: BANK[n]) if banks else (lambda n: ())  # noqa: E731
    o = [Op("L", "lda", "vlda {w.x}, [p0], #64", defs=[("w", "x", 7)], mem=("ld", "v"), banks=bk("v"), instr=VLDA)]
    sqrt_ops(o, "w", "y", bk)
    o.append(Op("ST", "st", "vst {y.x}, [p3], #64", uses=[("y", "x", 1)], mem=("st", "o"), banks=bk("v"), instr=VST))
    return o


def mk(banks=True):
    return Loop(ops(banks), consts={"x": [C_FF, C_HI, C_BASE, C_2E60]}, classes={"x": X_FREE, "dm": DM_FREE})


def build(II=None, times=None, cp=None, max_ii=20, banks=True, U=None):
    if times is not None:
        lp = mk(banks)
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U"))
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 13, (II or max_ii) + 1):
        lp = mk(banks)
        res = cpsched.solve(lp, ii, last_stage=["ST"], max_life=(U or 2) * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(U)
            check(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0], "U": lp.U}
        return lp
    raise Infeasible("none")


def check(lp, n=NV):
    st = {"v": (lambda i: 64 * i, dict(base=0x10000, dn=0, dj=64, m=64)), "o": (lambda i: 64 * i, dict(base=0x10000, dn=0, dj=64, m=64))}
    return lp.check(n - n % lp.U, st)


def function(lp, name="sqk"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        "movxm\tr16, #52428", "mov\tr0, p1", "vbcst.32\tx0, r0", "mov\tr0, p2", "vbcst.32\tx1, r0",
        "mova\tr0, #255", f"vbcst.16\tx{C_FF}, r0", "movxm\tr0, #65280", f"vbcst.16\tx{C_HI}, r0",
        f"vsel.32\tx{C_BASE}, x0, x1, r16", "movxm\tr0, #23936", f"vbcst.16\tx{C_2E60}, r0",
        "mova\tr1, #828", "mova\tr3, #24", "mova\tr4, #10", "mov\ts0, #2", "mov\ts1, #0", "mov\ts2, #1",
        "mov\tcrupsmode, #0", "mov\tcrsrsmode, #0", "mov\tp3, p0",
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NV // lp.U}", "add.nc\tlc, r0, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def header(lp, path, gen="r16/gen_sqk.py"):
    _, tp = table()
    rows = ",\n  ".join(", ".join(str(int(v)) for v in tp[i:i + 16]) for i in range(0, len(tp), 16))
    Path(path).write_text(f"// generated by {gen}\n#define SQK_U {lp.U}\n#define SQK_II {lp.II}\n#define SQK_TAB {{\\\n  "
                          + rows.replace("\n", " \\\n") + "}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--u", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--func", default="sqk")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), banks=not o.nobanks, U=o.u)
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NV} vectors)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    header(lp, Path(o.out).with_suffix(".h"))
    print("wrote", o.out)


if __name__ == "__main__":
    main()
