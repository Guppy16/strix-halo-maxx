"""rev 7: G1 (fused bias + GELU (+ gate product) + output) and the other LUT / chain loops, scheduled with the main agent's
modulo scheduler (tapnext/npu/r11/msched.py, imported read-only) at the round-11 rules (LLVM visibility, mv bypass,
itinerary ports). No device, no build: it answers "what II does the generator reach for this op list, with which binding
resource, how many registers and stages, and how much text".

Variants (one iteration = 32 lanes for yx, 64 lanes = one bfp block for the pushing variants):
  yx      fp32 piece + bias -> x = bf16 -> GELU (tanh table) -> bf16 out (vst.conv)
  pan     same chain (erf table) x 2 halves -> bf16 -> fp32 (ups) -> vst.push.576.conv (one bfp block)
  gated   gate chain x 2 halves + up (acc + bias -> bf16) -> bf16(g u) -> ups -> push
  gpush   G0.5: the gated product + push pass only, over today's bf16 v / u (no GELU)
  sig     SSM sigmoid table pass (P29a), 32 lanes, bf16 in / out
  aexp    attention exp + push (P48), 64 scores per iteration, row max as a constant register
  rec     SSM recurrence (P29c part), 32 lanes
Options: --flat (flat-edge tables: the s MAC takes x, no xc ring), --bias-ptr (biased LUT base: no index vadd),
         --bias-ldb (bias / x reloads on ldb instead of lda)

    python3 g1sched.py [variant ...] [--flat] [--bias-ptr] [--max-ii 30]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "tapnext" / "npu" / "r11"))
from msched import Infeasible, Loop  # noqa: E402
from msched import Op  # noqa: E402

# constant x registers (never written by the loop)
C_HI, C_LO, C_BASE, C_ONE, C_M1 = 8, 9, 10, 11, 7

LDA_BM = "VLDA_2D_dmx_lda_bm"
LDA_X = "VLDA_2D_dmx_lda_x"
LDB_X = "VLDB_2D_dmx_ldb_x"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
STCONV = "VST_2D_CONV_bf16_fp32_dmx_sts_srs_bf"
STX = "VST_2D_dmx_sts_x"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"


def ld(name, slot, fmt, v, part, stream, banks=(), **k):
    instr = {"lda": LDA_X if part == "x" else LDA_BM, "ldb": LDB_X}[slot]
    return Op(name, slot, fmt, defs=[(v, part, 7)], mem=("ld", stream), banks=banks, instr=instr, **k)


def lut_chain(p, x, xs, flat, bias_ptr, out_acc, ld_x="ldb"):
    """GELU-by-table on bf16 x (vreg name x) -> fp32 result in out_acc (vreg, dm). xs: a spill of x's source (name of the
    store op and of the ring) for the reload that feeds the s MAC (flat) and the products. Prefix p keeps names unique."""
    o = [
        Op(f"{p}MN", "mv", f"vmin_ge.bf16 {{{p}t.x}}, r16, x{C_HI}, {{{x}.x}}", defs=[(f"{p}t", "x", 2)], uses=[(x, "x", 1)], instr="VMIN_GE_bf16"),
        Op(f"{p}MX", "mv", f"vmax_lt.bf16 {{{p}xc.x}}, r16, x{C_LO}, {{{p}t.x}}", defs=[(f"{p}xc", "x", 2)], uses=[(f"{p}t", "x", 1)], instr="VMAX_LT_bf16"),
        Op(f"{p}F0", "st", f"vfloor.s32.bf16 {{{p}i0.x}}, {{{p}xc.wl}}, s0", defs=[(f"{p}i0", "x", 2)], uses=[(f"{p}xc", "wl", 1)], instr="VFLOOR_s32_bf16_mv_float_to_int_w"),
        Op(f"{p}F1", "st", f"vfloor.s32.bf16 {{{p}i1.x}}, {{{p}xc.wh}}, s0", defs=[(f"{p}i1", "x", 2)], uses=[(f"{p}xc", "wh", 1)], instr="VFLOOR_s32_bf16_mv_float_to_int_w"),
    ]
    j0, j1 = f"{p}i0", f"{p}i1"
    if not bias_ptr:
        o += [Op(f"{p}A0", "mv", f"vadd.32 {{{p}j0.x}}, {{{p}i0.x}}, x{C_BASE}", defs=[(f"{p}j0", "x", 2)], uses=[(f"{p}i0", "x", 1)], instr="VADD_32"),
              Op(f"{p}A1", "mv", f"vadd.32 {{{p}j1.x}}, {{{p}i1.x}}, x{C_BASE}", defs=[(f"{p}j1", "x", 2)], uses=[(f"{p}i1", "x", 1)], instr="VADD_32")]
        j0, j1 = f"{p}j0", f"{p}j1"
    for h, j in ((0, j0), (1, j1)):
        c0, c1 = f"{p}c0{h}", f"{p}c1{h}"
        o += [
            Op(f"{p}LA{h}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)], instr="VLDB_4x64_lo"),
            Op(f"{p}LB{h}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)], instr="VLDB_4x64_hi"),
            Op(f"{p}LC{h}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)], instr="VLDB_4x64_lo"),
            Op(f"{p}LD{h}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)], instr="VLDB_4x64_hi"),
            Op(f"{p}SO{h}", "mv", f"vshuffle {{{p}sacc.{'bmll' if h == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r3",
               defs=[(f"{p}sacc", "bmll" if h == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr="VSHUFFLE_vec_shuffle_bm"),
            Op(f"{p}SS{h}", "mv", f"vshuffle {{{p + ('sl0' if h == 0 else 'slope')}.x}}, {{{c0}.x}}, {{{c1}.x}}, r2",
               defs=[(p + ("sl0" if h == 0 else "slope"), "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr="VSHUFFLE_vec_shuffle_x"),
        ]
    o.append(Op(f"{p}MV", "mv", f"vmov {{{p}slope.wl}}, {{{p}sl0.wh}}", defs=[(f"{p}slope", "wl", 2)], uses=[(f"{p}sl0", "wh", 1)], instr="VMOV_alu_mv_mv_w"))
    # the x the s MAC and the products read: reloaded from the spill (xs = (store op, value vreg, part, ring))
    st_op, sv, spart, ring = xs
    xr = f"{p}xr"
    o.append(ld(f"{p}XL", ld_x, f"{'vldb' if ld_x == 'ldb' else 'vlda'}.2d {{{xr}.x}}, [p3], d4", xr, "x", ring + "r",
                fill=(sv, spart), after=[(st_op, 8)]))
    sx = xr if flat else f"{p}xc"  # without the flat-edge table the s MAC needs the clamped x (kept in a register here)
    o += [
        Op(f"{p}S", "vec", f"vmac.f {{{p}s.dm}}, {{{p}sacc.dm}}, {{{p}slope.x}}, {{{sx}.x}}, r1", defs=[(f"{p}s", "dm", 6)],
           uses=[(f"{p}sacc", "cml", 4), (f"{p}slope", "x", 1), (sx, "x", 1)], instr=VMAC),
        Op(f"{p}SH", "st", f"vconv.bf16.fp32 {{{p}sh.x}}, {{{p}s.cml}}", defs=[(f"{p}sh", "x", 2)], uses=[(f"{p}s", "cml", 1)], instr=CONV),
        Op(f"{p}D", "vec", f"vmsc.f {{{p}d.dm}}, {{{p}s.dm}}, {{{p}sh.x}}, x{C_ONE}, r1", defs=[(f"{p}d", "dm", 6)],
           uses=[(f"{p}s", "cml", 4), (f"{p}sh", "x", 1)], instr=VMSC),
        Op(f"{p}SL", "st", f"vconv.bf16.fp32 {{{p}sl.x}}, {{{p}d.cml}}", defs=[(f"{p}sl", "x", 2)], uses=[(f"{p}d", "cml", 1)], instr=CONV),
        Op(f"{p}P", "vec", f"vmul.f {{{p}p.dm}}, {{{xr}.x}}, {{{p}sh.x}}, r1", defs=[(f"{p}p", "dm", 6)], uses=[(xr, "x", 1), (f"{p}sh", "x", 1)], instr=VMUL),
        Op(f"{p}Q", "vec", f"vmac.f {{{out_acc}.dm}}, {{{p}p.dm}}, {{{xr}.x}}, {{{p}sl.x}}, r1", defs=[(out_acc, "dm", 6)],
           uses=[(f"{p}p", "cml", 4), (xr, "x", 1), (f"{p}sl", "x", 1)], instr=VMAC),
    ]
    return o


def bias_x(p, acc_stream, bias_stream, ring, bias_slot):
    """fp32 piece (2 x 16 lanes into bm quarters) + bf16 bias -> y (fp32), x = bf16(y), y spilled as bf16 to the ring"""
    return [
        ld(f"{p}L0", "lda", f"vlda.2d {{{p}a.bmll}}, [p0], d0", f"{p}a", "bmll", acc_stream + "0"),
        ld(f"{p}L1", "lda", f"vlda.2d {{{p}a.bmlh}}, [p6], d6", f"{p}a", "bmlh", acc_stream + "1"),
        ld(f"{p}LB", bias_slot, f"{'vldb' if bias_slot == 'ldb' else 'vlda'}.2d {{{p}b.x}}, [p5], d5", f"{p}b", "x", bias_stream),
        Op(f"{p}BM", "vec", f"vmac.f {{{p}y.dm}}, {{{p}a.dm}}, {{{p}b.x}}, x{C_ONE}, r1", defs=[(f"{p}y", "dm", 6)],
           uses=[(f"{p}a", "cml", 4), (f"{p}b", "x", 1)], instr=VMAC),
        Op(f"{p}CX", "st", f"vconv.bf16.fp32 {{{p}x.x}}, {{{p}y.cml}}", defs=[(f"{p}x", "x", 2)], uses=[(f"{p}y", "cml", 1)], instr=CONV),
        Op(f"{p}XS", "st", f"vst.2d.conv.bf16.fp32 {{{p}y.cml}}, [p2], d3", uses=[(f"{p}y", "cml", 1)], mem=("st", ring),
           spill=(f"{p}y", "cml"), instr=STCONV),
    ]


def v_yx(o):
    ops = bias_x("", "acc", "bias", "ring", o.bias_slot)
    ops += lut_chain("", "x", ("XS", "y", "cml", "ring"), o.flat, o.bias_ptr, "q", o.ld_x)
    ops.append(Op("ST", "st", "vst.2d.conv.bf16.fp32 {q.cml}, [p1], d1", uses=[("q", "cml", 1)], mem=("st", "out"), instr=STCONV))
    return ops, 32


def v_pan(o):
    ops = []
    for h in (0, 1):
        p = f"h{h}"
        ops += bias_x(p, f"acc{h}", f"bias{h}", f"ring{h}", o.bias_slot)
        ops += lut_chain(p, f"{p}x", (f"{p}XS", f"{p}y", "cml", f"ring{h}"), o.flat, o.bias_ptr, f"{p}q", o.ld_x)
        ops.append(Op(f"{p}CO", "st", f"vconv.bf16.fp32 {{{p}o.x}}, {{{p}q.cml}}", defs=[(f"{p}o", "x", 2)], uses=[(f"{p}q", "cml", 1)], instr=CONV))
        ops.append(Op(f"{p}UP", "mv", f"vconv.fp32.bf16 {{w.{'cml' if h == 0 else 'cmh'}}}, {{{p}o.x}}", defs=[("w", "cml" if h == 0 else "cmh", 2)],
                      uses=[(f"{p}o", "x", 1)], instr=UPS))
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops, 64


def v_gated(o):
    ops = []
    for h in (0, 1):
        p = f"g{h}"
        ops += bias_x(p, f"gacc{h}", f"gbias{h}", f"ring{h}", o.bias_slot)
        ops += lut_chain(p, f"{p}x", (f"{p}XS", f"{p}y", "cml", f"ring{h}"), o.flat, o.bias_ptr, f"{p}q", o.ld_x)
        u = f"u{h}"
        ops += [
            ld(f"{u}L0", "lda", f"vlda.2d {{{u}a.bmll}}, [p0], d0", f"{u}a", "bmll", f"uacc{h}0"),
            ld(f"{u}L1", "lda", f"vlda.2d {{{u}a.bmlh}}, [p6], d6", f"{u}a", "bmlh", f"uacc{h}1"),
            ld(f"{u}LB", o.bias_slot, f"vldb.2d {{{u}b.x}}, [p5], d5", f"{u}b", "x", f"ubias{h}"),
            Op(f"{u}BM", "vec", f"vmac.f {{{u}y.dm}}, {{{u}a.dm}}, {{{u}b.x}}, x{C_ONE}, r1", defs=[(f"{u}y", "dm", 6)],
               uses=[(f"{u}a", "cml", 4), (f"{u}b", "x", 1)], instr=VMAC),
            Op(f"{u}CU", "st", f"vconv.bf16.fp32 {{{u}u.x}}, {{{u}y.cml}}", defs=[(f"{u}u", "x", 2)], uses=[(f"{u}y", "cml", 1)], instr=CONV),
            Op(f"{p}CG", "st", f"vconv.bf16.fp32 {{{p}g.x}}, {{{p}q.cml}}", defs=[(f"{p}g", "x", 2)], uses=[(f"{p}q", "cml", 1)], instr=CONV),
            Op(f"{p}PR", "vec", f"vmul.f {{{p}pr.dm}}, {{{p}g.x}}, {{{u}u.x}}, r1", defs=[(f"{p}pr", "dm", 6)],
               uses=[(f"{p}g", "x", 1), (f"{u}u", "x", 1)], instr=VMUL),
            Op(f"{p}CP", "st", f"vconv.bf16.fp32 {{{p}pv.x}}, {{{p}pr.cml}}", defs=[(f"{p}pv", "x", 2)], uses=[(f"{p}pr", "cml", 1)], instr=CONV),
            Op(f"{p}UP", "mv", f"vconv.fp32.bf16 {{w.{'cml' if h == 0 else 'cmh'}}}, {{{p}pv.x}}", defs=[("w", "cml" if h == 0 else "cmh", 2)],
               uses=[(f"{p}pv", "x", 1)], instr=UPS),
        ]
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops, 64


def v_gpush(o):
    """G0.5: today's bf16 v (GELU'd gate) and u in L1 -> bf16(v u) -> bfp push"""
    ops = []
    for h in (0, 1):
        ops += [
            ld(f"V{h}", "lda", f"vlda.2d {{v{h}.x}}, [p0], d0", f"v{h}", "x", f"v{h}"),
            ld(f"U{h}", "ldb", f"vldb.2d {{u{h}.x}}, [p1], d1", f"u{h}", "x", f"u{h}"),
            Op(f"PR{h}", "vec", f"vmul.f {{pr{h}.dm}}, {{v{h}.x}}, {{u{h}.x}}, r1", defs=[(f"pr{h}", "dm", 6)], uses=[(f"v{h}", "x", 1), (f"u{h}", "x", 1)], instr=VMUL),
            Op(f"CP{h}", "st", f"vconv.bf16.fp32 {{pv{h}.x}}, {{pr{h}.cml}}", defs=[(f"pv{h}", "x", 2)], uses=[(f"pr{h}", "cml", 1)], instr=CONV),
            Op(f"UP{h}", "mv", f"vconv.fp32.bf16 {{w.{'cml' if h == 0 else 'cmh'}}}, {{pv{h}.x}}", defs=[("w", "cml" if h == 0 else "cmh", 2)],
               uses=[(f"pv{h}", "x", 1)], instr=UPS),
        ]
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops, 64


def v_sig(o):
    """SSM sigmoid by table: s = off + slope xc (no product), bf16 out"""
    ops = [ld("L", "lda", "vlda.2d {x.x}, [p0], d0", "x", "x", "in")]
    full = lut_chain("", "x", ("L", "x", "x", "in"), False, o.bias_ptr, "q")
    keep = [op for op in full if op.name in ("MN", "MX", "F0", "F1", "A0", "A1", "MV", "S") or op.name[:2] in ("LA", "LB", "LC", "LD", "SO", "SS")]
    ops += keep
    ops.append(Op("ST", "st", "vst.2d.conv.bf16.fp32 {s.cml}, [p1], d1", uses=[("s", "cml", 1)], mem=("st", "out"), instr=STCONV))
    return ops, 32


def v_aexp(o):
    """attention exp + push: per 32 scores d = bf16(s - m) clamped, line off - slope m + slope s (fp32), 2 halves -> push"""
    ops = []
    for h in (0, 1):
        p = f"e{h}"
        ops += [ld(f"{p}L", "lda", f"vlda.2d {{{p}s.x}}, [p0], d0", f"{p}s", "x", f"s{h}"),
                Op(f"{p}AR", "vec", f"vmac.f {{{p}da.dm}}, dm4, {{{p}s.x}}, x{C_ONE}, r1", defs=[(f"{p}da", "dm", 6)], uses=[(f"{p}s", "x", 1)], instr=VMAC),
                Op(f"{p}DH", "st", f"vconv.bf16.fp32 {{{p}dh.x}}, {{{p}da.cml}}", defs=[(f"{p}dh", "x", 2)], uses=[(f"{p}da", "cml", 1)], instr=CONV)]
        full = lut_chain(p, f"{p}dh", (f"{p}L", f"{p}s", "x", f"s{h}"), False, o.bias_ptr, f"{p}q")
        keep = [op for op in full if op.name[len(p):] in ("MN", "MX", "F0", "F1", "A0", "A1", "MV")
                or op.name[len(p):len(p) + 2] in ("LA", "LB", "LC", "LD", "SO", "SS")]
        ops += keep
        # p = off - slope m + slope s: m is a pinned constant register (x7 here), s re-read from its load (same vreg)
        ops += [Op(f"{p}M", "vec", f"vmsc.f {{{p}t2.dm}}, {{{p}sacc.dm}}, {{{p}slope.x}}, x{C_M1}, r1", defs=[(f"{p}t2", "dm", 6)],
                   uses=[(f"{p}sacc", "cml", 4), (f"{p}slope", "x", 1)], instr=VMSC),
                Op(f"{p}V", "vec", f"vmac.f {{w.{'cml' if h == 0 else 'cmh'}}}, {{{p}t2.dm}}, {{{p}slope.x}}, {{{p}s.x}}, r1",
                   defs=[("w", "cml" if h == 0 else "cmh", 6)], uses=[(f"{p}t2", "cml", 4), (f"{p}slope", "x", 1), (f"{p}s", "x", 1)], instr=VMAC)]
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops, 64


def v_rec(o):
    """SSM recurrence per 32 lanes (ew8.cc): gated, nx, h0 split, h, h split, select, stores"""
    L = lambda n, slot, v, part, s: ld(n, slot, f"{'vlda' if slot == 'lda' else 'vldb'}.2d {{{v}.{part}}}, [p0], d0", v, part, s)  # noqa: E731
    ops = [L("Lxc", "lda", "xc", "x", "xc"), L("Lgx", "ldb", "gx", "x", "gx"), L("La", "lda", "a", "x", "a"), L("Lm", "ldb", "mu", "x", "mu"),
           L("Lh0", "lda", "h0", "bmll", "h00"), L("Lh1", "lda", "h0", "bmlh", "h01")]
    V = lambda n, kind, d, uses, txt: Op(n, "vec", txt, defs=[(d, "dm", 6)], uses=uses, instr={"mul": VMUL, "mac": VMAC, "msc": VMSC}[kind])  # noqa: E731
    C = lambda n, d, s: Op(n, "st", f"vconv.bf16.fp32 {{{d}.x}}, {{{s}.cml}}", defs=[(d, "x", 2)], uses=[(s, "cml", 1)], instr=CONV)  # noqa: E731
    ops += [
        V("G", "mul", "ga", [("xc", "x", 1), ("gx", "x", 1)], "vmul.f {ga.dm}, {xc.x}, {gx.x}, r1"), C("Cg", "g", "ga"),
        V("N", "mul", "na", [("g", "x", 1), ("mu", "x", 1)], "vmul.f {na.dm}, {g.x}, {mu.x}, r1"), C("Cn", "nx", "na"),
        C("Ch0", "h0h", "h0"), V("Dh0", "msc", "h0d", [("h0", "cml", 4), ("h0h", "x", 1)], f"vmsc.f {{h0d.dm}}, {{h0.dm}}, {{h0h.x}}, x{C_ONE}, r1"),
        C("Cl0", "h0l", "h0d"),
        V("H1", "mul", "h1", [("a", "x", 1), ("h0h", "x", 1)], "vmul.f {h1.dm}, {a.x}, {h0h.x}, r1"),
        V("H2", "mac", "h2", [("h1", "cml", 4), ("a", "x", 1), ("h0l", "x", 1)], "vmac.f {h2.dm}, {h1.dm}, {a.x}, {h0l.x}, r1"),
        V("H3", "mac", "h", [("h2", "cml", 4), ("nx", "x", 1)], f"vmac.f {{h.dm}}, {{h2.dm}}, {{nx.x}}, x{C_ONE}, r1"),
        C("Chh", "hh", "h"), V("Dh", "msc", "hd", [("h", "cml", 4), ("hh", "x", 1)], f"vmsc.f {{hd.dm}}, {{h.dm}}, {{hh.x}}, x{C_ONE}, r1"),
        C("Chl", "hl", "hd"),
        V("S1", "mul", "s1", [("hh", "x", 1)], f"vmul.f {{s1.dm}}, {{hh.x}}, x{C_HI}, r1"),
        V("S2", "mac", "s2", [("s1", "cml", 4), ("hl", "x", 1)], f"vmac.f {{s2.dm}}, {{s1.dm}}, {{hl.x}}, x{C_HI}, r1"),
        V("S3", "mac", "hs", [("s2", "cml", 4), ("g", "x", 1)], f"vmac.f {{hs.dm}}, {{s2.dm}}, {{g.x}}, x{C_LO}, r1"),
        Op("Wf0", "st", "vst.2d {hs.bmll}, [p1], d1", uses=[("hs", "bmll", 1)], mem=("st", "hout0"), instr="VST_2D_dmx_sts_bm"),
        Op("Wf1", "st", "vst.2d {hs.bmlh}, [p1], d1", uses=[("hs", "bmlh", 1)], mem=("st", "hout1"), instr="VST_2D_dmx_sts_bm"),
        Op("Wb", "st", "vst.2d.conv.bf16.fp32 {hs.cml}, [p2], d2", uses=[("hs", "cml", 1)], mem=("st", "hb"), instr=STCONV),
    ]
    return ops, 32


def v_gelubf(o):
    """gen_gelu's loop with the flat-edge table: bf16 x in L1 -> GELU -> bf16 (pass 1 of a two-pass gated epilogue)"""
    ops = [ld("L", "lda", "vlda.2d {x.x}, [p0], d0", "x", "x", "in")]
    ops += lut_chain("", "x", ("L", "x", "x", "in"), o.flat, o.bias_ptr, "q", o.ld_x)
    ops.append(Op("ST", "st", "vst.2d.conv.bf16.fp32 {q.cml}, [p1], d1", uses=[("q", "cml", 1)], mem=("st", "out"), instr=STCONV))
    return ops, 32


def v_gup(o):
    """pass 2 of a two-pass gated epilogue: up fp32 + bias -> u = bf16, v (GELU'd gate, bf16 in L1), bf16(v u) -> push"""
    ops = []
    for h in (0, 1):
        u = f"u{h}"
        ops += [
            ld(f"V{h}", "ldb", f"vldb.2d {{v{h}.x}}, [p1], d1", f"v{h}", "x", f"v{h}"),
            ld(f"{u}L0", "lda", f"vlda.2d {{{u}a.bmll}}, [p0], d0", f"{u}a", "bmll", f"uacc{h}0"),
            ld(f"{u}L1", "lda", f"vlda.2d {{{u}a.bmlh}}, [p6], d6", f"{u}a", "bmlh", f"uacc{h}1"),
            ld(f"{u}LB", "ldb", f"vldb.2d {{{u}b.x}}, [p5], d5", f"{u}b", "x", f"ubias{h}"),
            Op(f"{u}BM", "vec", f"vmac.f {{{u}y.dm}}, {{{u}a.dm}}, {{{u}b.x}}, x{C_ONE}, r1", defs=[(f"{u}y", "dm", 6)],
               uses=[(f"{u}a", "cml", 4), (f"{u}b", "x", 1)], instr=VMAC),
            Op(f"{u}CU", "st", f"vconv.bf16.fp32 {{{u}u.x}}, {{{u}y.cml}}", defs=[(f"{u}u", "x", 2)], uses=[(f"{u}y", "cml", 1)], instr=CONV),
            Op(f"PR{h}", "vec", f"vmul.f {{pr{h}.dm}}, {{v{h}.x}}, {{{u}u.x}}, r1", defs=[(f"pr{h}", "dm", 6)], uses=[(f"v{h}", "x", 1), (f"{u}u", "x", 1)], instr=VMUL),
            Op(f"CP{h}", "st", f"vconv.bf16.fp32 {{pv{h}.x}}, {{pr{h}.cml}}", defs=[(f"pv{h}", "x", 2)], uses=[(f"pr{h}", "cml", 1)], instr=CONV),
            Op(f"UP{h}", "mv", f"vconv.fp32.bf16 {{w.{'cml' if h == 0 else 'cmh'}}}, {{pv{h}.x}}", defs=[("w", "cml" if h == 0 else "cmh", 2)],
               uses=[(f"pv{h}", "x", 1)], instr=UPS),
        ]
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops, 64


def v_panpush(o):
    """pan pass 2 if the GELU stays a separate in-place bf16 pass: bf16 -> fp32 (vlda.ups-free form: vlda + ups) -> push"""
    ops = []
    for h in (0, 1):
        ops += [ld(f"V{h}", "lda" if h == 0 else "ldb", f"{'vlda' if h == 0 else 'vldb'}.2d {{v{h}.x}}, [p0], d0", f"v{h}", "x", f"v{h}"),
                Op(f"UP{h}", "mv", f"vconv.fp32.bf16 {{w.{'cml' if h == 0 else 'cmh'}}}, {{v{h}.x}}", defs=[("w", "cml" if h == 0 else "cmh", 2)],
                   uses=[(f"v{h}", "x", 1)], instr=UPS)]
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops, 64


VARIANTS = dict(yx=v_yx, pan=v_pan, gated=v_gated, gpush=v_gpush, sig=v_sig, aexp=v_aexp, rec=v_rec, gelubf=v_gelubf, gup=v_gup, panpush=v_panpush)


def try_ii(ops_fn, o, consts, xfree, ii):
    ops, lanes = ops_fn(o)
    lp = Loop(ops, consts={"x": consts}, classes={"x": xfree, "dm": [0, 1, 2, 3] if o.pin_dm4 else [0, 1, 2, 3, 4]})
    try:
        lp.schedule(ii)
        lp.allocate()
    except Infeasible:
        ops, lanes = ops_fn(o)
        lp = Loop(ops, consts={"x": consts}, classes={"x": xfree, "dm": [0, 1, 2, 3] if o.pin_dm4 else [0, 1, 2, 3, 4]})
        lp.schedule_ims(ii)
        lp.allocate()
    return lp, lanes


def run(name, o):
    consts = [C_HI, C_LO, C_ONE] + ([] if o.bias_ptr else [C_BASE]) + ([C_M1] if name == "aexp" else [])
    xfree = [r for r in range(12) if r not in consts]
    o.pin_dm4 = name == "aexp"  # -m held in dm4 for the argument MAC
    ops, lanes = VARIANTS[name](o)
    lp0 = Loop(ops, consts={"x": consts}, classes={"x": xfree, "dm": [0, 1, 2, 3, 4]})
    res = lp0.res_mii(detail=True)
    mii = max(res.values())
    top = ", ".join(f"{k if isinstance(k, str) else k[1]} {v}" for k, v in list(res.items())[:6])
    fails = []
    for ii in range(mii, o.max_ii + 1):
        try:
            lp, lanes = try_ii(VARIANTS[name], o, consts, xfree, ii)
        except Infeasible as e:
            fails.append(f"II {ii}: {str(e)[:90]}")
            continue
        S = lp.stages()
        body_b = lp.U * lp.II * 16
        pro_b = (S - 1) * lp.II * 10  # prologue bundles in the short form, ~10 B average (gelu_ms: 718 B at II 13, 5 stages)
        text = body_b + pro_b + 260   # + preamble / ret (~260 B in gelu_ms)
        print(f"{name:6s} lanes {lanes:2d}  ops {len(lp.ops):3d}  ResMII {mii:2d} [{top}]  ->  II {lp.II:2d} ({32 * lp.II / lanes:5.2f} cyc/32 lanes)"
              f"  U {lp.U}  stages {S}  maxlive {lp.maxlive()}  text ~{text} B (kernel-only ~{body_b + 260} B)")
        if o.verbose:
            for f in fails:
                print("    " + f)
            for op in sorted(lp.ops, key=lambda x: x.t):
                print(f"    t {op.t:3d} st {op.t // lp.II} slot {op.t % lp.II:2d} {op.slot} {op.name}")
        return lp
    print(f"{name:6s} ResMII {mii} [{top}]: no schedule up to II {o.max_ii}; " + "; ".join(fails[-2:]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--bias-ptr", action="store_true")
    ap.add_argument("--bias-slot", default="ldb")
    ap.add_argument("--ld-x", default="ldb")
    ap.add_argument("--max-ii", type=int, default=32)
    ap.add_argument("-v", "--verbose", action="store_true")
    o = ap.parse_args()
    print(f"options: flat {o.flat}, biased LUT pointer {o.bias_ptr}, bias loads on {o.bias_slot}, x reload on {o.ld_x}")
    for v in o.variants:
        run(v, o)


if __name__ == "__main__":
    main()
