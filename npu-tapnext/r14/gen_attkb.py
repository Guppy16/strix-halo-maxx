"""Round 14 (npu-research P56, the batched form): the attention K phase of up to 6 row blocks of one (tile, chunk) as ONE
modulo-scheduled asm loop over (row block, key block) iterations, row-block major. Per iteration: QK^T of one key block,
the scores to a 6-slot ring, exp(s - m_old) with the row block's STORED max and the bfp push, and the iteration's flag
(some lane has s - m_old >= TAU - 0.25 SU) stored to a flag array. Same arithmetic as r13/gen_attk.py (bit-identical P).

What changes against round 13's per-row-block loop (r13/gen_attk.py):
- the streams step by 2D modifiers instead of being reloaded: Q returns to its row block's start after each key block
  (padda.2d: -576 five times, then 0 = the next row block), K returns to the object's start after each row block
  (padds.2d: 0 five times, then -3456), the -m rows advance 128 B per row block (the second load of each pair is 2D),
  the score ring has 7 slots (896 B: a 6-trip call's scores stay in slots 0-5 for the slow path, the drain's garbage
  store lands in slot 6), the push stream skips the 16 B pad after each row block (flush.2d);
- -m reaches the fp32 add by vlda.conv (bf16 -> fp32 in the load: no ups op, no x register);
- the flag of every iteration is stored (st r4, [p6], #4) instead of returned: the driver ORs a row block's six;
- ops carry the L1 banks of the in-situ layout (full12r14a: Q / P bank 1, K + the -m rows in the K object's tail bank 3,
  ring + exp table copy bank 2, aexp_ab + the stack bank 0): no two accesses to one bank in a bundle.

    extern "C" void attkb(const bfp16ebs8 *q /*p0*/, const bfp16ebs8 *kt /*p1*/, bfp16ebs8 *dst /*p2*/,
                          const bfloat16 *mrows /*p3*/, bfloat16 *srb /*p4*/, int32_t *flags /*p5*/, int32_t trips /*r0*/,
                          int32_t hi /*r1*/)
hi: the bf16 bits of the clamp's upper bound, which is also the flag bound: 63.75 (0x427F) for the fast pass; 71.5 (0x428F, aexp_push's
clamp) for the slow path's exact rerun, whose flags nobody reads (round 14: this replaced aexp_push, -464 B of text).
q: the first row block's 8 Q panels (row blocks 576 B apart); kt: the K object (6 key blocks of 576 B); dst: the first row
block's P slot (slots 448 B apart); mrows: rows8 of -m_old per row block (128 B apart); srb: the 896 B score ring (the exp
table's second copy is at srb + 896: mm8.cc attn_begin's layout); flags: one word per iteration, then their OR (flags[trips]). trips: 6 x row blocks
(full chunks) or the chunk's valid key blocks for one row block (2 in the last chunk).

    python3 r14/gen_attkb.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit] [--nobanks] [--s2]

Round 17 (--s2, research rev 11 S2): the running row-block max. Every iteration loads its row block's max slot (2 x 64 B,
[p6, #0] / [p6, #64]), takes vmax_lt(slot, s) (the C++ exact test's operand order: running max first) and stores it back
(vst [p6], #64 then vst.2d d7: the slot pointer advances 128 B per row block like the -m rows). The per-iteration flag stores
are gone: the call returns the OR of the flags in r0, and a flagged row block's own flag is recomputed by the driver from its
slot (bf16(max s - m) > bound for some lane <=> some s of the row block flagged: the rounding is monotone). So a flagged row
block needs no first-pass rerun to get its maximum back (k_rb's 393 cycles).
    extern "C" int32_t attkb(q p0, kt p1, dst p2, mrows p3, srb p4, bfloat16 *slots /*p5: 6 x 128 B*/, int32_t trips /*r1*/,
                             int32_t hi /*r2*/)      (r0 is the return value's: the scalar arguments move up by one)
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import MEM_R, MEM_W, Infeasible, Loop, Op  # noqa: E402

FILL_A, FILL_B = "VLDA_FILL_512", "VLDB_FILL_512"
POP_A, POP_B = "VLDA_POP_576_normal_pop", "VLDB_POP_576_normal_pop"
BMUL = "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX"
BMAC = "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"
VADDF = "VADD_f_vmac_cm2_add_reg"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
ST_X, ST_X2 = "VST_dmx_sts_x_pstm_nrm_imm", "VST_2D_dmx_sts_x"
LDA_X, LDA_X2 = "VLDA_dmx_lda_x_pstm_nrm_imm", "VLDA_2D_dmx_lda_x"
LDA_C, LDA_C2 = "VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm_imm", "VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf"
LDB_X, LDB_X2 = "VLDB_dmx_ldb_x_pstm_nrm_imm", "VLDB_2D_dmx_ldb_x"
PADDA2, PADDS2 = "PADDA_2D", "PADDS_2D"
MN, MX = "VMIN_GE_bf16", "VMAX_LT_bf16"
FLOOR = "VFLOOR_s32_bf16_mv_float_to_int_w"
VADD32 = "VADD_32"
LUT_LO, LUT_HI = "VLDB_4x64_lo", "VLDB_4x64_hi"
SHUF_BM, SHUF_X = "VSHUFFLE_vec_shuffle_bm", "VSHUFFLE_vec_shuffle_x"
VMOV_W, VMOV_CM = "VMOV_alu_mv_mv_w", "VMOV_alu_mv_mv_cm"
PUSH, FLUSH2 = "VST_PUSH_576_CONV_bfp16ebs8_fp32", "VST_FLUSH_512_CONV_2D"
VLT = "VLT_bf16"
ST_FLAG = ST_X  # the scalar st has no itinerary in the .td: reserve a vector store's ports (conservative)
C_HI, C_LO, C_BASE = 9, 10, 11
X_FREE = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # pops (ex_n = x_n) and the exp's temporaries share these
DM_FREE = [0, 1, 2, 3, 4]
R_MM, R_ADD, CFG = "r5", "r6", "r1"  # mmul config 780, accumulator add 60, bf16 mac 828 (r2 / r3 = shuffle modes)
HI_BITS, LO_BITS, SHIFT, BIAS_B = 17023, -15696, 3, 44 * 16  # bf16 63.75 (= TAU - 0.25 SU, the flag bound), -88; 2^3 B per 1/8
# modifier register sets: (dn, dj, m) per op that steps its pointer by a 2D modifier
MODS = {"PQ": (0, 5, -576, 0), "PK": (1, 5, 0, -3456), "MA1": (2, 5, -64, 64), "MB1": (3, 5, -64, 64),
        "W1": (4, 6, 64, -832), "R1": (5, 6, 64, -832), "FL": (6, 5, 0, 16)}
MODS_S2 = {"XS1": (7, 5, -64, 64)}  # round 17: the max slots (0 five times, then the next row block's 128 B)
# L1 banks of the in-situ layout (full12r14a core 0_2: acc_hi 0x718c0 -> Q copies + P buffers in bank 1; B objects at 53504 /
# 58112 -> bank 3 incl. their tails; the C object at 35072 -> ring + exp table copy in bank 2; aexp_ab 0x71300 + stack: bank 0)
BANKS = {"q": (1,), "k": (3,), "m": (3,), "lut": (0, 2), "srb": (2,), "p": (1,), "fl": (0,), "mx": (0,)}
MAXSRC = "s"  # round 17: the max takes the scores as computed ("s") or as reloaded from the ring ("sr")
USE_BANKS = True
FORBID_TRIES = [6]
S2 = False  # round 17: the running max (see the module doc)
STAGES = 3  # schedule length bound in IIs (round 14: 2 = a 34-bundle-shorter prologue, for text)


def bk(name):
    return BANKS[name] if USE_BANKS else ()


def ops():
    o = [Op("QF", "lda", "vlda.fill.512 [p0, lf0, r24]", banks=bk("q"), instr=FILL_A)]
    for j in range(8):
        o.append(Op(f"Q{j}", "lda", f"vlda.pop.576 {{q{j}.ex}}, [p0, lf0, r24]", defs=[(f"q{j}", "ex", 8)], mem=("ld", "q"),
                    after=[(f"Q{j - 1}" if j else "QF", 1)], banks=bk("q"), instr=POP_A))
    # back to the row block's start for the next key block (after the 6th: on to the next row block)
    o.append(Op("PQ", "lda", "padda.2d [p0], d0", mem=("pa", "q"), after=[("Q7", 1)], instr=PADDA2))
    o.append(Op("F", "ldb", "vldb.fill.512 [p1, lf1, r25]", banks=bk("k"), instr=FILL_B))
    for j in range(8):
        o.append(Op(f"K{j}", "ldb", f"vldb.pop.576 {{k{j}.ex}}, [p1, lf1, r25]", defs=[(f"k{j}", "ex", 8)], mem=("ld", "k"),
                    after=[(f"K{j - 1}" if j else "F", 1)], banks=bk("k"), instr=POP_B))
    o.append(Op("PK", "st", "padds.2d [p1], d1", mem=("pa", "k"), after=[("K7", 1)], instr=PADDS2))
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
                 uses=[("t", "cml" if h == 0 else "cmh", 1)], instr=CONV)]
    o += [Op("W0", "st", "vst {s0.x}, [p4], #64", uses=[("s0", "x", 1)], mem=("st", "srb"), spill=("s0", "x"), banks=bk("srb"), instr=ST_X),
          Op("W1", "st", "vst.2d {s1.x}, [p4], d4", uses=[("s1", "x", 1)], mem=("st", "srb"), spill=("s1", "x"), banks=bk("srb"),
             instr=ST_X2, after=[("W0", 1)])]
    for h in (0, 1):
        s, su, nm, da, dh, t, c = f"s{h}", f"su{h}", f"nm{h}", f"da{h}", f"dh{h}", f"t{h}", f"c{h}"
        o += [
            # d = fp32(s + (-m)): -m (rows8 of the negated stored max, bf16 in L1) widened by the load itself, s by ups
            Op(f"MA{h}", "lda", f"vlda{'.2d' if h else ''}.conv.fp32.bf16 {{{nm}.cml}}, [p3], {'d2' if h else '#64'}", defs=[(nm, "cml", 7)],
               mem=("ld", "ma"), banks=bk("m"), instr=LDA_C2 if h else LDA_C, after=[("MA0", 1)] if h else []),
            Op(f"SU{h}", "mv", f"vconv.fp32.bf16 {{{su}.cml}}, {{{s}.x}}", defs=[(su, "cml", 2)], uses=[(s, "x", 1)], instr=UPS),
            Op(f"AR{h}", "vec", f"vadd.f {{{da}.dm}}, {{{su}.dm}}, {{{nm}.dm}}, {R_ADD}", defs=[(da, "dm", 6)],
               uses=[(su, "cml", 4), (nm, "cml", 4)], instr=VADDF),
            Op(f"DH{h}", "st", f"vconv.bf16.fp32 {{{dh}.x}}, {{{da}.cml}}", defs=[(dh, "x", 2)], uses=[(da, "cml", 1)], instr=CONV),
            Op(f"MN{h}", "mv", f"vmin_ge.bf16 {{{t}.x}}, r16, x{C_HI}, {{{dh}.x}}", defs=[(t, "x", 2)], uses=[(dh, "x", 1)], instr=MN),
            Op(f"MX{h}", "mv", f"vmax_lt.bf16 {{{c}.x}}, r16, x{C_LO}, {{{t}.x}}", defs=[(c, "x", 2)], uses=[(t, "x", 1)], instr=MX),
            # lanes where HI < dh -> r18 / r19 (own mask registers); r20+ may be implicit stream-state registers (round 13)
            Op(f"CM{h}", "mv", f"vlt.bf16 r1{8 + h}, x{C_HI}, {{{dh}.x}}", uses=[(dh, "x", 1)], instr=VLT),
        ]
        for q in (0, 1):
            i, j, c0, c1 = f"i{h}{q}", f"j{h}{q}", f"c0{h}{q}", f"c1{h}{q}"
            w = "wl" if q == 0 else "wh"
            o += [
                Op(f"F{h}{q}", "st", f"vfloor.s32.bf16 {{{i}.x}}, {{{c}.{w}}}, s0", defs=[(i, "x", 2)], uses=[(c, w, 1)], instr=FLOOR),
                Op(f"A{h}{q}", "mv", f"vadd.32 {{{j}.x}}, {{{i}.x}}, x{C_BASE}", defs=[(j, "x", 2)], uses=[(i, "x", 1)], instr=VADD32),
                Op(f"LA{h}{q}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)], banks=bk("lut"), instr=LUT_LO),
                Op(f"LB{h}{q}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)], banks=bk("lut"), instr=LUT_HI),
                Op(f"LC{h}{q}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)], banks=bk("lut"), instr=LUT_LO),
                Op(f"LD{h}{q}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)], banks=bk("lut"), instr=LUT_HI),
                Op(f"SO{h}{q}", "mv", f"vshuffle {{off{h}.{'bmll' if q == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r3",
                   defs=[(f"off{h}", "bmll" if q == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=SHUF_BM),
                Op(f"SS{h}{q}", "mv", f"vshuffle {{{'sl' + str(h) if q == 0 else 'slope' + str(h)}.x}}, {{{c0}.x}}, {{{c1}.x}}, r2",
                   defs=[("sl" + str(h) if q == 0 else "slope" + str(h), "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=SHUF_X),
            ]
        sl, slope, off, qq, sr = f"sl{h}", f"slope{h}", f"off{h}", f"ql{h}", f"sr{h}"
        pw = "w" if h == 0 else "p1"
        o += [
            Op(f"MV{h}", "mv", f"vmov {{{slope}.wl}}, {{{sl}.wh}}", defs=[(slope, "wl", 2)], uses=[(sl, "wh", 1)], instr=VMOV_W),
            Op(f"MB{h}", "lda", f"vlda{'.2d' if h else ''} {{mb{h}.x}}, [p7], {'d3' if h else '#64'}", defs=[(f"mb{h}", "x", 7)],
               mem=("ld", "mb"), banks=bk("m"), instr=LDA_X2 if h else LDA_X, after=[("MB0", 1)] if h else []),
            # off + slope (-m): the same product and rounding as today's msc(off, slope, m)
            Op(f"Q{h}L", "vec", f"vmac.f {{{qq}.dm}}, {{{off}.dm}}, {{{slope}.x}}, {{mb{h}.x}}, {CFG}", defs=[(qq, "dm", 6)],
               uses=[(off, "cml", 4), (slope, "x", 1), (f"mb{h}", "x", 1)], instr=VMAC),
            Op(f"R{h}", "ldb", f"vldb{'.2d' if h else ''} {{{sr}.x}}, [p5], {'d5' if h else '#64'}", defs=[(sr, "x", 7)], mem=("ld", "sr"),
               fill=(s, "x"), banks=bk("srb"), after=[(f"W{h}", MEM_W - MEM_R + 1)] + ([("R0", 1)] if h else []), instr=LDB_X2 if h else LDB_X),
            Op(f"P{h}", "vec", f"vmac.f {{{pw}.dm}}, {{{qq}.dm}}, {{{slope}.x}}, {{{sr}.x}}, {CFG}", defs=[(pw, "cml", 6)],
               uses=[(qq, "cml", 4), (slope, "x", 1), (sr, "x", 1)], instr=VMAC),
        ]
    o += [
        Op("MVH", "mv", "vmov {w.cmh}, {p1.cml}", defs=[("w", "cmh", 2)], uses=[("p1", "cml", 1)], instr=VMOV_CM),
        Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], banks=bk("p"), instr=PUSH),
        Op("FL", "st", "vst.flush.512.conv.2d [p2, sf, r26, d6]", after=[("PU", 1)], banks=bk("p"), instr=FLUSH2),
        # the iteration's flag: both halves' compare masks OR-ed, stored (last stage: the drain's garbage iterations never store)
        Op("FO", "alu", "or r4, r18, r19", after=[("CM0", 2), ("CM1", 2)]),
    ] + ([] if S2 else [Op("FS", "st", "st r4, [p6], #4", after=[("FO", 2)], mem=("st", "fl"), banks=bk("fl"), instr=ST_FLAG)]) + [
        Op("FA", "alu", "or r17, r17, r4", after=[("FO", 2)]),  # a scalar result is read 2 cycles on (round 14: at +1 FA saw the previous iteration's r4)  # and OR-ed over the call: stored after the loop as flags[trips]
    ]
    if S2:  # round 17: the running max per row block through its stack slot (bank 0); loads by offset, stores step the pointer
        src = MAXSRC
        for h in (0, 1):
            o += [Op(f"XL{h}", "lda", f"vlda {{mo{h}.x}}, [p6, #{64 * h}]", defs=[(f"mo{h}", "x", 7)], mem=("ld", "mx"), banks=bk("mx"),
                     instr=LDA_X),
                  Op(f"XM{h}", "mv", f"vmax_lt.bf16 {{mn{h}.x}}, r16, {{mo{h}.x}}, {{{src}{h}.x}}", defs=[(f"mn{h}", "x", 2)],
                     uses=[(f"mo{h}", "x", 1), (f"{src}{h}", "x", 1)], instr=MX)]
        o += [Op("XS0", "st", "vst {mn0.x}, [p6], #64", uses=[("mn0", "x", 1)], mem=("st", "mx"), banks=bk("mx"), instr=ST_X,
                 after=[("XL0", 1), ("XL1", 1)]),
              Op("XS1", "st", "vst.2d {mn1.x}, [p6], d7", uses=[("mn1", "x", 1)], mem=("st", "mx"), banks=bk("mx"), instr=ST_X2,
                 after=[("XS0", 1)])]
    return o


def mk():
    return Loop(ops(), consts={"x": [C_HI, C_LO, C_BASE]}, classes={"x": X_FREE, "dm": DM_FREE})


def maxdist(II):
    """t_b - t_a <= d. The streams of one iteration stay inside one II, in order: the next iteration's fill comes after this
    one's pointer step (a pointer write needs ~3 cycles before a load uses it); the 2D loads / stores step after the plain ones.
    The flag OR reads r18 / r19 before the next iteration's compares land."""
    return [("QF", "PQ", II - 3), ("F", "PK", II - 3), ("QF", "Q7", II - 1), ("F", "K7", II - 1),
            ("W0", "W1", II - 1), ("R0", "R1", II - 1), ("MA0", "MA1", II - 1), ("MB0", "MB1", II - 1), ("PU", "FL", II - 1),
            ("CM0", "FO", II + 1), ("CM1", "FO", II + 1), ("FO", "FA", II - 1)] + (
        # round 17: the next iteration's slot loads read what this one stored (data land 4 cycles after the store, the post-
        # modified pointer needs ~3 cycles: round 14's fill-after-step margin)
        [("XL0", "XS1", II - 4), ("XL1", "XS1", II - 4)] if S2 else [])


def last_ops():
    return ["PU", "FL", "FO", "FA"] + (["XS0", "XS1"] if S2 else ["FS"])


def stages_ok(lp):
    S = lp.stages()
    for n in last_ops():
        assert lp.by_name[n].t // lp.II == S - 1, f"{n} not in the last stage"
    # the ring: iteration i + 7's store must not land before iteration i's reload (7 slots)
    for h in (0, 1):
        assert lp.by_name[f"R{h}"].t + MEM_R < lp.by_name[f"W{h}"].t + 7 * lp.II, "ring too short"


def build(II=None, times=None, cp=None, max_ii=40):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        stages_ok(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 29, max(max_ii, II or 0) + 1):
        lp = mk()
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=last_ops(), max_len=STAGES * ii - 1, max_life=2 * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
            stages_ok(lp)
            for n in (36, 6, 2):  # round 17: the pad replays inside the search (a schedule can fail only with the pad)
                check(lp, n)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            import re
            m = re.search(r"cycle \d+ (\w+) it \d+: port \S+ at \d+ taken by (\w+) it", str(e))
            if m and FORBID_TRIES[0] > 0:  # round 17: forbid the colliding op's time and re-solve at the same II
                FORBID_TRIES[0] -= 1
                cp = dict(cp or {})
                cp["forbid"] = list(cp.get("forbid", [])) + [(m.group(2), res[0][m.group(2)])]
                cp["hint"] = res[0]
                print(f"II {ii}: forbid {cp['forbid'][-1]}, re-solve", file=sys.stderr)
                return build(ii, None, cp, max_ii)
            continue
        lp.times = {"II": ii, "t": res[0]}
        print(f"II {ii}: {res[1]}", file=sys.stderr)
        return lp
    raise Infeasible("none")


def streams():
    def mod(name):
        _, dn, dj, m = (MODS | MODS_S2)[name]
        return dict(dn=dn, dj=dj, m=m)

    inc = lambda b: dict(dn=1 << 20, dj=b, m=b)  # noqa: E731  (a post-increment immediate)
    return {
        "q": (lambda i, k: (i // 6) * 576 + 72 * k, dict(base=0x74000, dn=1 << 20, dj=72, m=72, ops={"PQ": mod("PQ")})),
        "k": (lambda i, k: (i % 6) * 576 + 72 * k, dict(base=0x7D000, dn=1 << 20, dj=72, m=72, ops={"PK": mod("PK")})),
        "srb": (lambda i, k: (i % 7) * 128 + 64 * k, dict(base=0x78900, dn=1 << 20, dj=0, m=0, ops={"W0": inc(64), "W1": mod("W1")})),
        "sr": (lambda i, k: (i % 7) * 128 + 64 * k, dict(base=0x78900, dn=1 << 20, dj=0, m=0, ops={"R0": inc(64), "R1": mod("R1")})),
        "ma": (lambda i, k: (i // 6) * 128 + 64 * k, dict(base=0x7DD80, dn=1 << 20, dj=0, m=0, ops={"MA0": inc(64), "MA1": mod("MA1")})),
        "mb": (lambda i, k: (i // 6) * 128 + 64 * k, dict(base=0x7DD80, dn=1 << 20, dj=0, m=0, ops={"MB0": inc(64), "MB1": mod("MB1")})),
    } | ({"mx": (lambda i, k: (i // 6) * 128 + (64 if k == 3 else 0), dict(base=0x70100, dn=1 << 20, dj=0, m=0, ops={
        "XL0": dict(dn=1 << 20, dj=0, m=0), "XL1": dict(dn=1 << 20, dj=0, m=0), "XS0": inc(64), "XS1": mod("XS1")}))} if S2 else
        {"fl": (lambda i: 4 * i, dict(base=0x70100, dn=1 << 20, dj=4, m=4))})


def check(lp, n=36):
    n -= n % lp.U
    if S2:  # round 17: the slot round trip (not in msched's memory model: a load reads the PREVIOUS iteration's store)
        t = lp.by_name
        for h in (0, 1):
            assert t[f"XL{h}"].t + lp.II + MEM_R >= t[f"XS{h}"].t + MEM_W, "slot load before the previous store landed"
            assert t[f"XL{h}"].t + lp.II >= t["XS1"].t + 3, "slot load before the pointer step"
    return lp.check(n, streams())


DRAIN = 8


def function(lp, name="attkb", drain=None):
    drain = DRAIN if drain is None else drain
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    mods = []
    for op, (d, dn, dj, m) in (MODS | (MODS_S2 if S2 else {})).items():
        mods += [f"mova\tdn{d}, #{dn}", f"movxm\tdj{d}, #{dj}", f"movxm\tm{d}, #{m}", f"mova\tdc{d}, #0"]
    pre = [
        # void function: the trip count arrives in r0 (copied: r0 is free for the loop's use of nothing, but keep the habit)
        # (S2: int32_t function: r0 is the result's, trips / hi arrive in r1 / r2)
        "mov\tr7, r1" if S2 else "mov\tr7, r0", f"vbcst.16\tx{C_HI}, r2" if S2 else f"vbcst.16\tx{C_HI}, r1",  # the clamp / flag bound arrives in r1 (bf16 bits; read before r1 = 828)
        "paddxm\t[sp], #64", "st\tp6, [sp, #-64]", "st\tp7, [sp, #-60]",  # callee-saved pointers this loop uses
        "mov\tp6, p5", "mov\tp7, p3", "mov\tp5, p4",  # flags; the second -m pointer (line); the ring reload pointer
        f"mova\t{R_MM}, #780", f"mova\t{R_ADD}, #60", "mova\tr24, #0", "mova\tr25, #0",
        "mov\tr2, p4", "movxm\tr3, #896", "add\tr2, r2, r3",  # the exp table's second copy: srb + 896 (attn_begin's layout)
        "movxm\tr16, #52428", "vbcst.32\tx1, r2", "movxm\tr2, #aexp_ab", "vbcst.32\tx0, r2",
        f"movxm\tr2, #{BIAS_B}", "vbcst.32\tx2, r2", "vsel.32\tx3, x0, x1, r16", "nop", "nop", f"vadd.32\tx{C_BASE}, x3, x2",
        f"movxm\tr2, #{LO_BITS}", f"vbcst.16\tx{C_LO}, r2",
        f"mova\t{CFG}, #828", "mova\tr2, #24", "mova\tr3, #5", f"mov\ts0, #{SHIFT}", "mova\tr26, #0", "mova\tr17, #0",
    ] + mods + [
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r7, #0", "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    # the callee-saved pointers come back during the drain (the stores / pushes of the last iterations land meanwhile)
    L += ["\tmov\tr0, r17" if S2 else "\tst\tr17, [p6, #0]", "\tlda\tp6, [sp, #-64]", "\tlda\tp7, [sp, #-60]"] + ["\tnop"] * max(drain - 3, 6) + [
          "\tret\tlr", "\tnop", "\tnop", "\tnop", "\tpaddxm\t[sp], #-64", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def report(lp):
    print(f"II {lp.II} (resource bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:10])}), U {lp.U}, "
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
    ap.add_argument("--func", default="attkb")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--drain", type=int, default=8)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--s2", action="store_true", help="round 17: the running max per row block (module doc)")
    ap.add_argument("--maxsrc", default="s", choices=["s", "sr"])
    ap.add_argument("--workers", type=int, default=4, help="round 17: CP-SAT workers (the laptop stays usable)")
    ap.add_argument("--reserve-x", type=int, default=0, help="round 17: x registers kept free in the CP pressure bound (allocation slack)")
    o = ap.parse_args()
    global USE_BANKS, DRAIN, STAGES, S2, MAXSRC
    S2, MAXSRC = o.s2, o.maxsrc
    STAGES = o.stages
    USE_BANKS, DRAIN = not o.nobanks, o.drain
    times = json.loads(Path(o.times).read_text()) if o.times else None
    if times and "maxsrc" in times:  # round 17: the S2 schedule records which score copy its max reads
        MAXSRC = times["maxsrc"]
    cp = dict(time_limit=o.time, workers=o.workers) | ({"reserve": {"x": o.reserve_x}} if o.reserve_x else {})
    lp = build(o.ii, times, cp=cp, max_ii=o.max_ii)
    report(lp)
    for n in (36, 6, 2):
        cyc, land = check(lp, n)
        print(f"check OK ({n} iterations): {cyc} cycles, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times | ({"maxsrc": MAXSRC} if S2 else {}), indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r14/gen_attkb.py\n#define ATTKB_U {lp.U}\n#define ATTKB_II {lp.II}\n"
                                       + ("#define ATTKB_S2 1\n" if S2 else ""))
    print("wrote", o.out)


if __name__ == "__main__":
    main()
