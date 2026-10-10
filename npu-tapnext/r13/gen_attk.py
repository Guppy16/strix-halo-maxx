"""Round 13 (npu-research P56, per-row-block form): the attention K phase of one row block as ONE modulo-scheduled asm loop:
QK^T, the scores to L1, exp(s - m_old) with the STORED max, and the bfp push, per key block (iteration = 64 scores).

The lazy max (mm8.cc chunk_k) moves the stored max m only when a row's max passes m + TAU, else P = exp(s - m): so P can
be formed with m_old as soon as the scores exist. The C++ driver (chunk_k_fused) then reads the stored scores, runs today's
exact test and, for the row blocks whose max moves (~4 % of units after chunk 0, r13/lazycount.py), reruns today's
rescale_rb + aexp_push over the P slot: bit-identical by construction. Chunks with invalid key blocks take the slow path too.

    extern "C" int32_t attk(const bfp16ebs8 *q /*p0*/, const bfp16ebs8 *kt /*p1*/, bfp16ebs8 *dst /*p2: the push stream's
                         pointer must be p2*/, const bfloat16 *mrow /*p3*/, bfloat16 *srb /*p4*/, const float *cd /*p5*/, int trips /*r0*/)
Returns a flag: nonzero when some lane of the row block had s - m_old >= 63.75 SU (a superset of "the max passed the stored max
by TAU": the driver runs today's exact test and slow path only then; the clamp also sits at 63.75, so an unflagged block's P is
exactly today's).
q: the row block's 8 Q panels (576 B; popped into ex0-ex7 in the preamble, held). kt: the K object (one ldb pop stream over
the 6 key blocks; the drain's garbage pops read (S - 1) x 576 B into the B object's junk tail: S <= 3 asserted). srb: the 6
score blocks (768 B) + ONE block of slack behind it (the garbage store of iteration 6; the stores sit in stage 1). mrow: rows8
of -m_old (128 B, read per iteration; the line's vmac takes -m). dst: the 6 P panels. cd: the exp table's second copy (ab = the global aexp_ab).
Per 64 scores, bit-identical to chunk_k + aexp_push: s = bf16(ae + ao); d = fp32(s * 1.0 - m) (one rounding, as
mac(-m, s, 1.0)); dh = bf16(d); clamp; floor; LUT; off - slope m + slope s (s reloaded from srb).

    python3 r13/gen_attk.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import MEM_R, MEM_W, Infeasible, Loop, Op  # noqa: E402

FILL = "VLDB_FILL_512"
POP = "VLDB_POP_576_normal_pop"
BMUL = "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX"
BMAC = "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"
VADDF = "VADD_f_vmac_cm2_add_reg"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
STX = "VST_dmx_sts_x_pstm_nrm_imm"
LDA_IDX = "VLDA_dmx_lda_x_idx_imm"
LDB_X = "VLDB_dmx_ldb_x_pstm_nrm_imm"
MN, MX = "VMIN_GE_bf16", "VMAX_LT_bf16"
FLOOR = "VFLOOR_s32_bf16_mv_float_to_int_w"
VADD32 = "VADD_32"
LUT_LO, LUT_HI = "VLDB_4x64_lo", "VLDB_4x64_hi"
SHUF_BM, SHUF_X = "VSHUFFLE_vec_shuffle_bm", "VSHUFFLE_vec_shuffle_x"
VMOV_W, VMOV_CM = "VMOV_alu_mv_mv_w", "VMOV_alu_mv_mv_cm"
PUSH, FLUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32", "VST_FLUSH_512_CONV_normal_flush"
C_HI, C_LO, C_BASE = 9, 10, 11
X_FREE = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # pops (ex_n = x_n) and the exp's temporaries share these
DM_FREE = [0, 1, 2, 3, 4]
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
R_MM, R_ADD, CFG = "r5", "r6", "r1"  # mmul config 780, accumulator add 60, bf16 mac 828 (r2 / r3 = shuffle modes)
HI_BITS, LO_BITS, SHIFT, BIAS_B = 17023, -15696, 3, 44 * 16  # bf16 63.75 (= TAU - 0.25 SU: the flag bound, see FLAG), -88; 2^3 B per 1/8
VLT = "VLT_bf16"
FLAG = True  # the row block's flag: lanes with d >= 63.75 (vlt.bf16 into r18 / r19, OR-ed into r17 in the last stage, returned in r0)
JUNK_B = 4608 - 3456
QKONLY = False  # debug: the QK^T + score stores only (same preamble)
KLDA = False  # the K pop stream on load unit A (lf0 / r24) instead of B: tests whether the LUT loads disturb a live B-unit FIFO


def ops():
    """one key block: Q (8 panels) and K (8 panels) popped, two mmul chains, the scores, exp(s + (-m)) and the push.
    ex_n aliases x_n (msched PARTS): the pops take x registers; the Q stream restarts every iteration (its pointer reloaded from
    the stack, the FIFO position reset) so the same 576 B are popped per key block."""
    o = [
        # the Q object's address (saved by the preamble); usable SCL_LAT later. Its ports: a scalar load shares DM_ADA with the
        # LUT loads (round 13: unmodelled, it co-issued with a vldb.4x64 and loaded garbage into p0 from iteration 2 on)
        Op("LP", "lda", "lda p0, [sp, #-64]", instr="LDA_dmv_lda_q_spill"),
        Op("RS", "mv", "mov r24, #0", after=[("LP", 1)]),  # the Q FIFO's position: a fresh stream per key block
        Op("QF", "lda", "vlda.fill.512 [p0, lf0, r24]", after=[("LP", 9), ("RS", 1)], instr="VLDA_FILL_512"),
    ]
    for j in range(8):
        o.append(Op(f"Q{j}", "lda", f"vlda.pop.576 {{q{j}.ex}}, [p0, lf0, r24]", defs=[(f"q{j}", "ex", 8)], mem=("ld", "q"),
                    after=[(f"Q{j - 1}" if j else "QF", 1)], instr="VLDA_POP_576_normal_pop"))
    o.append(Op("F", "ldb", "vldb.fill.512 [p1, lf1, r25]", instr=FILL))
    for j in range(8):
        o.append(Op(f"K{j}", "ldb", f"vldb.pop.576 {{k{j}.ex}}, [p1, lf1, r25]", defs=[(f"k{j}", "ex", 8)], mem=("ld", "k"),
                    after=[(f"K{j - 1}" if j else "F", 1)], instr=POP))
    for j in range(8):
        acc = "ae" if j % 2 == 0 else "ao"
        if j < 2:
            o.append(Op(f"M{j}", "vec", f"vmul.f {{{acc}.dm}}, {{q{j}.ex}}, {{k{j}.ex}}, {R_MM}", defs=[(acc, "dm", 6)],
                        uses=[(f"q{j}", "ex", 1), (f"k{j}", "ex", 1)], instr=BMUL))
        else:
            o.append(Op(f"M{j}", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{q{j}.ex}}, {{k{j}.ex}}, {R_MM}", defs=[(acc, "dm", 6)],
                        uses=[(acc, "dm", 4), (f"q{j}", "ex", 1), (f"k{j}", "ex", 1)], instr=BMAC))
    o.append(Op("AD", "vec", f"vadd.f {{t.dm}}, {{ae.dm}}, {{ao.dm}}, {R_ADD}", defs=[("t", "dm", 6)], uses=[("ae", "dm", 4), ("ao", "dm", 4)], instr=VADDF))
    for h in (0, 1):
        o += [Op(f"C{h}", "st", f"vconv.bf16.fp32 {{s{h}.x}}, {{t.{'cml' if h == 0 else 'cmh'}}}", defs=[(f"s{h}", "x", 2)],
                 uses=[("t", "cml" if h == 0 else "cmh", 1)], instr=CONV),
              Op(f"W{h}", "st", f"vst {{s{h}.x}}, [p4], #64", uses=[(f"s{h}", "x", 1)], mem=("st", "srb"), spill=(f"s{h}", "x"), instr=STX,
                 after=[("W0", 1)] if h else [])]
    if QKONLY:
        return o
    for h in (0, 1):
        s, su, nm, da, dh, t, c = f"s{h}", f"su{h}", f"nm{h}", f"da{h}", f"dh{h}", f"t{h}", f"c{h}"
        o += [
            # d = fp32(s + (-m)): s and -m (rows8 of the negated stored max, bf16 in L1) widened exactly, one fp32 add
            Op(f"MA{h}", "lda", f"vlda {{ma{h}.x}}, [p3, #{64 * h}]", defs=[(f"ma{h}", "x", 7)], instr=LDA_IDX),
            Op(f"NU{h}", "mv", f"vconv.fp32.bf16 {{{nm}.cml}}, {{ma{h}.x}}", defs=[(nm, "cml", 2)], uses=[(f"ma{h}", "x", 1)], instr=UPS),
            Op(f"SU{h}", "mv", f"vconv.fp32.bf16 {{{su}.cml}}, {{{s}.x}}", defs=[(su, "cml", 2)], uses=[(s, "x", 1)], instr=UPS),
            Op(f"AR{h}", "vec", f"vadd.f {{{da}.dm}}, {{{su}.dm}}, {{{nm}.dm}}, {R_ADD}", defs=[(da, "dm", 6)],
               uses=[(su, "cml", 4), (nm, "cml", 4)], instr=VADDF),
            Op(f"DH{h}", "st", f"vconv.bf16.fp32 {{{dh}.x}}, {{{da}.cml}}", defs=[(dh, "x", 2)], uses=[(da, "cml", 1)], instr=CONV),
            Op(f"MN{h}", "mv", f"vmin_ge.bf16 {{{t}.x}}, r16, x{C_HI}, {{{dh}.x}}", defs=[(t, "x", 2)], uses=[(dh, "x", 1)], instr=MN),
            Op(f"MX{h}", "mv", f"vmax_lt.bf16 {{{c}.x}}, r16, x{C_LO}, {{{t}.x}}", defs=[(c, "x", 2)], uses=[(t, "x", 1)], instr=MX),
        ]
        if FLAG:  # lanes where HI < dh -> r18 / r19 (own mask registers), OR-ed into r17 in the LAST stage (the drain's garbage
            # iterations run the compare but never the OR; the OR must read before the next iteration's compare lands: maxdist)
            # r18 / r19 / r17: r20+ may be implicit stream-state registers (with r20-r22 the loop's last push never landed)
            o += [Op(f"CM{h}", "mv", f"vlt.bf16 r1{8 + h}, x{C_HI}, {{{dh}.x}}", uses=[(dh, "x", 1)], instr=VLT),
                  Op(f"OR{h}", "alu", f"or r17, r17, r1{8 + h}", after=[(f"CM{h}", 2)])]
        for q in (0, 1):
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
        sl, slope, off, qq, sr = f"sl{h}", f"slope{h}", f"off{h}", f"ql{h}", f"sr{h}"
        pw = "w" if h == 0 else "p1"
        o += [
            Op(f"MV{h}", "mv", f"vmov {{{slope}.wl}}, {{{sl}.wh}}", defs=[(slope, "wl", 2)], uses=[(sl, "wh", 1)], instr=VMOV_W),
            Op(f"MB{h}", "lda", f"vlda {{mb{h}.x}}, [p3, #{64 * h}]", defs=[(f"mb{h}", "x", 7)], instr=LDA_IDX),
            # off + slope (-m): the same product and rounding as today's msc(off, slope, m)
            Op(f"Q{h}L", "vec", f"vmac.f {{{qq}.dm}}, {{{off}.dm}}, {{{slope}.x}}, {{mb{h}.x}}, {CFG}", defs=[(qq, "dm", 6)],
               uses=[(off, "cml", 4), (slope, "x", 1), (f"mb{h}", "x", 1)], instr=VMAC),
            Op(f"R{h}", "ldb", f"vldb {{{sr}.x}}, [p5], #64", defs=[(sr, "x", 7)], mem=("ld", "sr"), fill=(s, "x"),
               after=[(f"W{h}", MEM_W - MEM_R + 1)] + ([("R0", 1)] if h else []), instr=LDB_X),
            Op(f"P{h}", "vec", f"vmac.f {{{pw}.dm}}, {{{qq}.dm}}, {{{slope}.x}}, {{{sr}.x}}, {CFG}", defs=[(pw, "cml", 6)],
               uses=[(qq, "cml", 4), (slope, "x", 1), (sr, "x", 1)], instr=VMAC),
        ]
    o += [
        Op("MVH", "mv", "vmov {w.cmh}, {p1.cml}", defs=[("w", "cmh", 2)], uses=[("p1", "cml", 1)], instr=VMOV_CM),
        Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH),
        Op("FL", "st", "vst.flush.512.conv [p2, sf, r26]", after=[("PU", 1)], instr=FLUSH),
    ]
    return o


def mk():
    return Loop(ops(), consts={"x": [C_HI, C_LO, C_BASE]}, classes={"x": X_FREE, "dm": DM_FREE})


def maxdist(II):
    # the Q stream: the next iteration's pointer reload and FIFO reset come after this iteration's last pop
    # the line's -m loads (MB) of iteration 5 must read g_mbuf before iteration 6's garbage score store lands on it (the last
    # trip runs both): MB no later than one II after W0 (round 13: panel 5 of every row block was wrong without this)
    return [("F", "K7", II - 1), ("QF", "Q7", II - 1), ("LP", "Q7", II - 1), ("RS", "Q7", II - 2), ("W0", "W1", II - 1)] + \
        ([] if QKONLY else [("R0", "R1", II - 1), ("PU", "FL", II - 1), ("W0", "MB0", II - 1), ("W0", "MB1", II - 1)]) + \
        ([("CM0", "OR0", II + 1), ("CM1", "OR1", II + 1)] if FLAG and not QKONLY else [])


def last_ops():
    return ["W0", "W1"] if QKONLY else ["PU", "FL"] + (["OR0", "OR1"] if FLAG else [])


def stages_ok(lp):
    S = lp.stages()
    assert (S - 1) * 576 <= JUNK_B, f"{S} stages: garbage pops read {(S - 1) * 576} B past the K object (> {JUNK_B})"
    for n in last_ops():
        assert lp.by_name[n].t // lp.II == S - 1, f"{n} not in the last stage"
    for n in (() if QKONLY else ("W0", "W1")):  # the stores: only iteration 6's garbage store (one block of slack) may run
        assert 1 <= lp.by_name[n].t // lp.II <= S - 2, f"{n} at stage {lp.by_name[n].t // lp.II}: needs 1 .. {S - 2}"


def build(II=None, times=None, cp=None, max_ii=40):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        stages_ok(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 27, max(max_ii, II or 0) + 1):
        lp = mk()
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=last_ops(), max_len=3 * ii - 1,
                            min_t={} if QKONLY else {"W0": ii, "W1": ii}, max_life=2 * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
            stages_ok(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0]}
        print(f"II {ii}: {res[1]}", file=sys.stderr)
        return lp
    raise Infeasible("none")


def check(lp, n=6):
    n -= n % lp.U
    st = {"k": (lambda i, k: i * 576 + 72 * k, dict(base=0x7C000, dn=1 << 20, dj=72, m=72)),
          "q": (lambda i, k: 72 * k, dict(base=0x7A000, dn=7, dj=72, m=-504)),  # the same 8 panels every key block
          "srb": (lambda i, k: i * 128 + 64 * k, dict(base=0x73C00, dn=1 << 20, dj=64, m=64)),
          "sr": (lambda i, k: i * 128 + 64 * k, dict(base=0x73C00, dn=1 << 20, dj=64, m=64))}
    if QKONLY:
        del st["sr"]
    return lp.check(n, st)


DRAIN = 8


def function(lp, name="attk", drain=None):
    drain = DRAIN if drain is None else drain
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        # Peano's ABI: a function returning a value has r0 reserved for it and its first scalar argument in r1 (round 13: with
        # `int32_t attk(...)` the trip count arrived in r1 while the loop read r0; r1 is also the mac config below: copy first)
        "mov\tr7, r1" if FLAG else "mov\tr7, r0",
        f"mova\t{R_MM}, #780", f"mova\t{R_ADD}, #60", "mova\tr24, #0", "mova\tr25, #0",
        "paddxm\t[sp], #64", "st\tp0, [sp, #-64]",  # the Q object's address, reloaded per key block (the stream restarts)
        "movxm\tr16, #52428", "movxm\tr2, #aexp_ab", "vbcst.32\tx0, r2", "mov\tr2, p5", "vbcst.32\tx1, r2",
        f"movxm\tr2, #{BIAS_B}", "vbcst.32\tx2, r2", "vsel.32\tx3, x0, x1, r16", "nop", "nop", f"vadd.32\tx{C_BASE}, x3, x2",
        f"movxm\tr2, #{HI_BITS}", f"vbcst.16\tx{C_HI}, r2", f"movxm\tr2, #{LO_BITS}", f"vbcst.16\tx{C_LO}, r2",
        f"mova\t{CFG}, #828", "mova\tr2, #24", "mova\tr3, #5", f"mov\ts0, #{SHIFT}", "mova\tr26, #0", "mova\tr17, #0",
        "mov\tp5, p4",  # the score reloads start where the stores do
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r7, #0", "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tmov\tr0, r17",  # the flag (returned)
          "\tret\tlr", "\tnop", "\tnop", "\tnop", "\tpaddxm\t[sp], #-64", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
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
    ap.add_argument("--max-ii", type=int, default=40)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=200)
    ap.add_argument("--func", default="attk")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--qkonly", action="store_true")
    ap.add_argument("--klda", action="store_true")
    ap.add_argument("--drain", type=int, default=8)
    ap.add_argument("--noflag", action="store_true", help="debug: no flag ops (the schedule's extra times are ignored)")
    ap.add_argument("--hi71", action="store_true", help="debug: the clamp at 71.5 (no flag bound)")
    o = ap.parse_args()
    global QKONLY, KLDA, DRAIN, FLAG, HI_BITS
    QKONLY, KLDA, DRAIN = o.qkonly, o.klda, o.drain
    if o.noflag:
        FLAG = False
    if o.hi71:
        HI_BITS = 17039
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), max_ii=o.max_ii if not o.qkonly else max(o.max_ii, 14))
    report(lp)
    cyc, land = check(lp)
    print(f"check OK: {cyc} cycles for 6 key blocks, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r13/gen_attk.py\n#define ATTK_U {lp.U}\n#define ATTK_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
