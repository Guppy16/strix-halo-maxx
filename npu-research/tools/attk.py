"""rev 8: the attention K phase as modulo-scheduled loops (P48 / P49 / the fused "lazy-fast" K loop), scheduled with the
main agent's msched.py (list / IMS) and, on the laptop, cpsched.py (CP-SAT, exact register pressure).
No device, no build. Answers: which II per key block (64 scores = 8 query rows x 8 keys) each op list reaches under the
round-12 rules, with which binding resource and register pressure.

One iteration = one key block of one row block:
  QK  : 8 Q pops (lda) + 8 K pops (ldb), two mmul chains (even / odd dim panels, today's order: bit-identical),
        vadd.f of the chains, 2 vconv to bf16 (the scores s), 2 vst of s (the fallback path's copy: today's g_srb)
  EXP : per 32 lanes: d = bf16(s - m) (arg), clamp (vmin_ge / vmax_lt; the vmin_ge's r16 mask OR-ed into a scalar: the
        lazy-update test, see ANALYSIS rev 8 3), vfloor, index add, 4 LUT loads, 2 shuffles, the line off - slope m +
        slope s; 64 lanes -> one bfp push
  fused = QK + EXP in one body (the stored max m is known before the scores: the lazy max makes the exp independent of
        the current block's max; rows whose max passes the threshold go to the fallback, today's C++ path)
Options:
  --arg dm4      d = vmac(-m held in dm4, s, 1.0) (today's form; pins one dm)
  --arg mulmsc   d = vmsc(vmul(s, 1.0), m, 1.0) (no pinned dm; exact: one fp32 rounding of s - m either way)
  --bias-ptr     LUT base folded into the pointer (no index vadd; P54, unprobed)
  --qhold        Q's 8 panels held in x registers instead of popped per key block (no Q pops; 8 x pinned)

    python3 attk.py qk|exp|fused [--arg mulmsc] [--cp] [--ii N] [--max-ii N] [--time 120]
CP-SAT needs OR-tools: uv run --no-project --with ortools --with numpy python attk.py fused --cp
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for p in (HERE.parent, HERE.parents[1] / "npu-tapnext" / "r11", HERE.parents[1] / "npu-tapnext" / "r12",
          HERE.parent / "r11", HERE.parent / "r12", HERE.parents[1] / "r11", HERE.parents[1] / "r12"):
    sys.path.insert(0, str(p))
from msched import Infeasible, Loop, Op  # noqa: E402

LDA_X = "VLDA_2D_dmx_lda_x"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
BMAC = "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"
BMUL = "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX"
VADDF = "VADD_f_vmac_cm2_add_reg"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
STX = "VST_2D_dmx_sts_x"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"
LDCONV = "VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf"
POPA, POPB = "VLDA_POP_576_normal_pop", "VLDB_POP_576_normal_pop"


def qk(o, p=""):
    ops = []
    for j in range(8):
        if not o.qhold:
            ops.append(Op(f"{p}QP{j}", "lda", f"vlda.pop.576 {{{p}q{j}.x}}, [p0, lf0, r24]", defs=[(f"{p}q{j}", "x", 8)], instr=POPA,
                          after=[(f"{p}QP{j - 1}", 1)] if j else []))
        ops.append(Op(f"{p}KP{j}", "ldb", f"vldb.pop.576 {{{p}k{j}.x}}, [p1, lf1, r25]", defs=[(f"{p}k{j}", "x", 8)], instr=POPB,
                      after=[(f"{p}KP{j - 1}", 1)] if j else []))
    for j in range(8):
        ch = "e" if j % 2 == 0 else "o"
        q = f"x{j}" if o.qhold else f"{{{p}q{j}.x}}"
        uses = [] if o.qhold else [(f"{p}q{j}", "x", 1)]
        uses.append((f"{p}k{j}", "x", 1))
        if j < 2:
            ops.append(Op(f"{p}M{j}", "vec", f"vmul.f {{{p}{ch}{j}.dm}}, {q}, {{{p}k{j}.x}}, r3", defs=[(f"{p}{ch}{j}", "dm", 6)],
                          uses=uses, instr=BMUL))
        else:
            ops.append(Op(f"{p}M{j}", "vec", f"vmac.f {{{p}{ch}{j}.dm}}, {{{p}{ch}{j - 2}.dm}}, {q}, {{{p}k{j}.x}}, r3",
                          defs=[(f"{p}{ch}{j}", "dm", 6)], uses=[(f"{p}{ch}{j - 2}", "dm", 4)] + uses, instr=BMAC))
    ops += [
        Op(f"{p}SUM", "vec", f"vadd.f {{{p}t.dm}}, {{{p}e6.dm}}, {{{p}o7.dm}}, r6", defs=[(f"{p}t", "dm", 6)],
           uses=[(f"{p}e6", "dm", 4), (f"{p}o7", "dm", 4)], instr=VADDF),
        Op(f"{p}C0", "st", f"vconv.bf16.fp32 {{{p}s0.x}}, {{{p}t.cml}}", defs=[(f"{p}s0", "x", 2)], uses=[(f"{p}t", "cml", 1)], instr=CONV),
        Op(f"{p}C1", "st", f"vconv.bf16.fp32 {{{p}s1.x}}, {{{p}t.cmh}}", defs=[(f"{p}s1", "x", 2)], uses=[(f"{p}t", "cmh", 1)], instr=CONV),
    ]
    if o.store:
        ops += [Op(f"{p}W0", "st", f"vst.2d {{{p}s0.x}}, [p3], d3", uses=[(f"{p}s0", "x", 1)], mem=("st", "srb0"), instr=STX),
                Op(f"{p}W1", "st", f"vst.2d {{{p}s1.x}}, [p3], d3", uses=[(f"{p}s1", "x", 1)], mem=("st", "srb1"), instr=STX)]
    return ops


def exp(o, s_of, consts, load=False):
    """the exp + push on scores s_of(h) (vregs); consts: dict of pinned x numbers"""
    C = consts
    ops = []
    for h in (0, 1):
        p = f"e{h}"
        s = s_of(h)
        if load:
            ops.append(Op(f"{p}L", "lda", f"vlda.2d {{{s}.x}}, [p0], d0", defs=[(s, "x", 7)], mem=("ld", f"s{h}"), instr=LDA_X))
        if o.mload:
            ops.append(Op(f"{p}LM", o.mload, f"{'vlda' if o.mload == 'lda' else 'vldb'}.2d {{{p}m.x}}, [p5], d5", defs=[(f"{p}m", "x", 7)],
                          mem=("ld", f"m{h}"), instr=LDA_X if o.mload == "lda" else "VLDB_2D_dmx_ldb_x"))
        if o.mload and o.arg == "upsadd":  # -m per row block: loaded per iteration straight into an fp32 half (lda only)
            ops.append(Op(f"{p}LN", "lda", f"vlda.2d.conv.fp32.bf16 {{{p}nm.cml}}, [p6], d6", defs=[(f"{p}nm", "cml", 7)],
                          mem=("ld", f"nm{h}"), instr=LDCONV))
        if o.arg == "upsadd":  # ups s to fp32 (exact) then + (-m) held in dm4: no x constant
            ops += [Op(f"{p}U", "mv", f"vconv.fp32.bf16 {{{p}su.cml}}, {{{s}.x}}", defs=[(f"{p}su", "cml", 2)], uses=[(s, "x", 1)],
                       instr="VCONV_fp32_bf16_mv_ups_xbf"),
                    (Op(f"{p}AR", "vec", f"vadd.f {{{p}da.dm}}, {{{p}su.dm}}, {{{p}nm.dm}}, r6", defs=[(f"{p}da", "dm", 6)],
                        uses=[(f"{p}su", "cml", 4), (f"{p}nm", "cml", 4)], instr=VADDF) if o.mload else
                     Op(f"{p}AR", "vec", f"vadd.f {{{p}da.dm}}, {{{p}su.dm}}, dm4, r6", defs=[(f"{p}da", "dm", 6)],
                       uses=[(f"{p}su", "cml", 4)], instr=VADDF))]
        elif o.arg == "dm4":
            ops.append(Op(f"{p}AR", "vec", f"vmac.f {{{p}da.dm}}, dm4, {{{s}.x}}, x{C['one']}, r1", defs=[(f"{p}da", "dm", 6)],
                          uses=[(s, "x", 1)], instr=VMAC))
        else:
            ops += [Op(f"{p}A1", "vec", f"vmul.f {{{p}d1.dm}}, {{{s}.x}}, x{C['one']}, r1", defs=[(f"{p}d1", "dm", 6)],
                       uses=[(s, "x", 1)], instr=VMUL),
                    (Op(f"{p}AR", "vec", f"vmsc.f {{{p}da.dm}}, {{{p}d1.dm}}, {{{p}m.x}}, x{C['one']}, r1",
                        defs=[(f"{p}da", "dm", 6)], uses=[(f"{p}d1", "cml", 4), (f"{p}m", "x", 1)], instr=VMSC) if o.mload else
                     Op(f"{p}AR", "vec", f"vmsc.f {{{p}da.dm}}, {{{p}d1.dm}}, x{C['m' + str(h)]}, x{C['one']}, r1",
                        defs=[(f"{p}da", "dm", 6)], uses=[(f"{p}d1", "cml", 4)], instr=VMSC))]
        ops += [
            Op(f"{p}DH", "st", f"vconv.bf16.fp32 {{{p}dh.x}}, {{{p}da.cml}}", defs=[(f"{p}dh", "x", 2)], uses=[(f"{p}da", "cml", 1)], instr=CONV),
            Op(f"{p}MN", "mv", f"vmin_ge.bf16 {{{p}t.x}}, r16, x{C['hi']}, {{{p}dh.x}}", defs=[(f"{p}t", "x", 2)], uses=[(f"{p}dh", "x", 1)], instr="VMIN_GE_bf16"),
            Op(f"{p}OR", "alu", "or r20, r20, r16", instr="OR", after=[(f"{p}MN", 2)]),
            Op(f"{p}MX", "mv", f"vmax_lt.bf16 {{{p}xc.x}}, r16, x{C['lo']}, {{{p}t.x}}", defs=[(f"{p}xc", "x", 2)], uses=[(f"{p}t", "x", 1)], instr="VMAX_LT_bf16"),
            Op(f"{p}F0", "st", f"vfloor.s32.bf16 {{{p}i0.x}}, {{{p}xc.wl}}, s0", defs=[(f"{p}i0", "x", 2)], uses=[(f"{p}xc", "wl", 1)], instr="VFLOOR_s32_bf16_mv_float_to_int_w"),
            Op(f"{p}F1", "st", f"vfloor.s32.bf16 {{{p}i1.x}}, {{{p}xc.wh}}, s0", defs=[(f"{p}i1", "x", 2)], uses=[(f"{p}xc", "wh", 1)], instr="VFLOOR_s32_bf16_mv_float_to_int_w"),
        ]
        ops[-3].after.append((f"{p}OR", 1))  # the vmax_lt rewrites r16: after the OR read it
        j0, j1 = f"{p}i0", f"{p}i1"
        if not o.bias_ptr:
            ops += [Op(f"{p}A0", "mv", f"vadd.32 {{{p}j0.x}}, {{{p}i0.x}}, x{C['base']}", defs=[(f"{p}j0", "x", 2)], uses=[(f"{p}i0", "x", 1)], instr="VADD_32"),
                    Op(f"{p}A1x", "mv", f"vadd.32 {{{p}j1.x}}, {{{p}i1.x}}, x{C['base']}", defs=[(f"{p}j1", "x", 2)], uses=[(f"{p}i1", "x", 1)], instr="VADD_32")]
            j0, j1 = f"{p}j0", f"{p}j1"
        for g, j in ((0, j0), (1, j1)):
            c0, c1 = f"{p}c0{g}", f"{p}c1{g}"
            ops += [
                Op(f"{p}LA{g}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)], instr="VLDB_4x64_lo"),
                Op(f"{p}LB{g}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)], instr="VLDB_4x64_hi"),
                Op(f"{p}LC{g}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)], instr="VLDB_4x64_lo"),
                Op(f"{p}LD{g}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)], instr="VLDB_4x64_hi"),
                Op(f"{p}SO{g}", "mv", f"vshuffle {{{p}sacc.{'bmll' if g == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r3",
                   defs=[(f"{p}sacc", "bmll" if g == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr="VSHUFFLE_vec_shuffle_bm"),
                Op(f"{p}SS{g}", "mv", f"vshuffle {{{p + ('sl0' if g == 0 else 'slope')}.x}}, {{{c0}.x}}, {{{c1}.x}}, r2",
                   defs=[(p + ("sl0" if g == 0 else "slope"), "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr="VSHUFFLE_vec_shuffle_x"),
            ]
        ops += [
            Op(f"{p}MV", "mv", f"vmov {{{p}slope.wl}}, {{{p}sl0.wh}}", defs=[(f"{p}slope", "wl", 2)], uses=[(f"{p}sl0", "wh", 1)], instr="VMOV_alu_mv_mv_w"),
            (Op(f"{p}M", "vec", f"vmsc.f {{{p}t2.dm}}, {{{p}sacc.dm}}, {{{p}slope.x}}, {{{p}m.x}}, r1", defs=[(f"{p}t2", "dm", 6)],
                uses=[(f"{p}sacc", "cml", 4), (f"{p}slope", "x", 1), (f"{p}m", "x", 1)], instr=VMSC) if o.mload else
             Op(f"{p}M", "vec", f"vmsc.f {{{p}t2.dm}}, {{{p}sacc.dm}}, {{{p}slope.x}}, x{C['m' + str(h)]}, r1", defs=[(f"{p}t2", "dm", 6)],
               uses=[(f"{p}sacc", "cml", 4), (f"{p}slope", "x", 1)], instr=VMSC)),
        ]
        sl = s
        if o.sreload:
            sl = f"{p}sr"
            ops.append(Op(f"{p}RL", o.sreload, f"{'vlda' if o.sreload == 'lda' else 'vldb'}.2d {{{sl}.x}}, [p4], d4", defs=[(sl, "x", 7)],
                          mem=("ld", f"srb{h}r"), instr=LDA_X if o.sreload == "lda" else "VLDB_2D_dmx_ldb_x",
                          after=[(f"W{h}", 4)] if o.store else []))
        ops.append(Op(f"{p}V", "vec", f"vmac.f {{w.{'cml' if h == 0 else 'cmh'}}}, {{{p}t2.dm}}, {{{p}slope.x}}, {{{sl}.x}}, r1",
               defs=[("w", "cml" if h == 0 else "cmh", 6)], uses=[(f"{p}t2", "cml", 4), (f"{p}slope", "x", 1), (sl, "x", 1)], instr=VMAC))
    ops.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH))
    return ops


def build(variant, o):
    pinned = ["hi", "lo"] + ([] if o.mload else ["m0", "m1"]) + ([] if o.arg == "upsadd" else ["one"]) + ([] if o.bias_ptr else ["base"])
    if variant == "qk":
        pinned = []
    first = 8 if o.qhold else 0  # x0..x7 hold Q when --qhold
    C = {n: first + i for i, n in enumerate(pinned)}
    used = set(C.values()) | (set(range(8)) if o.qhold else set())
    xfree = [r for r in range(12) if r not in used]
    if variant == "qk":
        ops = qk(o)
    elif variant == "exp":
        ops = exp(o, lambda h: f"s{h}", C, load=True)
    else:
        ops = qk(o) + exp(o, lambda h: f"s{h}", C)
    dm = [0, 1, 2, 3] if ((o.arg == "dm4" or (o.arg == "upsadd" and not o.mload)) and variant != "qk") else [0, 1, 2, 3, 4]
    return Loop(ops, consts={"x": sorted(used)}, classes={"x": xfree, "dm": dm}), xfree, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["qk", "exp", "fused"])
    ap.add_argument("--arg", default="mulmsc", choices=["dm4", "mulmsc", "upsadd"])
    ap.add_argument("--sreload", default="", help="lda|ldb: the line reads s from a reload of the stored scores")
    ap.add_argument("--bias-ptr", action="store_true")
    ap.add_argument("--qhold", action="store_true")
    ap.add_argument("--mload", default="", help="lda|ldb: m rows loaded per iteration (one loop over all 6 row blocks) instead of pinned")
    ap.add_argument("--no-store", dest="store", action="store_false")
    ap.add_argument("--cp", action="store_true")
    ap.add_argument("--ii", type=int)
    ap.add_argument("--max-ii", type=int, default=40)
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--H", type=int)
    o = ap.parse_args()
    lp0, xfree, dm = build(o.variant, o)
    res = lp0.res_mii(detail=True)
    mii = max(res.values())
    top = ", ".join(f"{k if isinstance(k, str) else k[1]} {v}" for k, v in list(res.items())[:8])
    print(f"{o.variant}: {len(lp0.ops)} ops, x free {len(xfree)}, dm {len(dm)}; ResMII {mii} [{top}]")
    for ii in range(o.ii or mii, (o.ii or o.max_ii) + 1):
        lp, _, _ = build(o.variant, o)
        try:
            if o.cp:
                import cpsched
                r = cpsched.solve(lp, ii, H=o.H or 6 * ii, time_limit=o.time, last_stage=["PU"] if o.variant != "qk" else (),
                                  maxdist=[])
                if r is None:
                    print(f"  II {ii}: CP-SAT none in {o.time:.0f} s")
                    continue
                lp.place(ii, r[0])
                info = r[1]
            else:
                try:
                    lp.schedule(ii)
                    lp.allocate()
                except Infeasible:
                    lp, _, _ = build(o.variant, o)
                    lp.schedule_ims(ii)
                info = {}
            lp.allocate()
        except Infeasible as e:
            print(f"  II {ii}: {str(e)[:100]}")
            continue
        print(f"  II {ii}: U {lp.U} stages {lp.stages()} maxlive {lp.maxlive()} {info}")
        return
    print("  none")


if __name__ == "__main__":
    main()
