"""Round 16: the SSM's two gate GEMMs (gate_x, gate_a pre-activations) as ONE modulo-scheduled asm loop (EW cores, ew8.cc
ew_ssm; flag -DGGK). Replaces two ew8.cc gate_gemm calls (~560 cycles of the 651-cycle in-situ gates phase).

Per iteration = one np (two 8-column blocks) of one gate, 8 iterations (gate_x np 0-3, then gate_a np 0-3), bit-identical to
gate_gemm: acc_c = sum_kk A_kk B_kk,c (bfp mmul, kk = 0..7, c = 0, 1); out_c = bf16(bf16(acc_c) 1 + bias_c), the bias added
in fp32 as gate_gemm's mul(BF(acc), ones) + bias * ones (one rounding of the same exact operands: bias loaded into the
accumulator by vlda.conv, then vmac BF(acc) * 1.0). Two 32-lane halves per column block.
- A (xc's 8 bfp panels, 576 B = 9 lines) is popped on load unit A, restarted every iteration (pointer copy, position reset,
  fill); its buffer is a bank-2 static (ew8.cc g_xcpb) so its pops do not share bank 1 with B's;
- B (Wx then Wa: 128 panels, 9,216 B contiguous in the arena) is ONE pop stream on unit B with a fill before every 8th pop
  (a dense 576-bit pop stream underflows at its 9th pop without it: r16/popprobe.s probes 6 / 7);
- the biases (tbx, then tba: 16 x 64 B each) by vlda.conv with a 2D modifier that jumps from tbx's end to tba;
- the outputs (gx, then ga: contiguous) by vst.conv, linear.

    extern "C" void ggk(const bfp16ebs8 *xcp /*p0*/, const bfp16ebs8 *wt /*p1*/, const void *tbx /*p2*/, bfloat16 *dst /*p3*/,
                         const void *tba /*p4*/)

    python3 r16/gen_ggk.py OUT.s [--ii N] [--u U] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

NIT = 8
C_ONE = 11
X_FREE = list(range(11))
DM_FREE = [0, 1, 2, 3, 4]
FILL_A = "VLDA_FILL_512"
POP_A, POP_B = "VLDA_POP_576_normal_pop", "VLDB_POP_576_normal_pop"
BMUL = "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX"
BMAC = "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
LDA_C2 = "VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf"
LDA_CP = "VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm_imm"
# debug variants used to find the B-stream underflow (round 16; outputs differ from gate_gemm unless the bench matches):
BIAS_1D = False  # plain +64 bias loads (tba must follow tbx)
NO_BIAS = False  # no bias loads; out = bf16(bf16(acc) 1) (bench with zero biases)
POP_EXCL = False  # no A pop / fill in the same cycle as a B pop (pseudo port)
NO_VM = False  # debug (round 16): out = bf16(acc) stored from the conv's x register (bench with zero biases)
ST_C = "VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm"
R_MM, R_MAC = "r5", "r1"
# in situ: wx / wa at the arena's start (bank 1), the A copy in bank 2, the biases tbx (0x79A40) / tba (0x78A40) bank 2,
# gx / ga (0x77A40 - 0x78240) mostly bank 1
BANKS = {"a": (2,), "b": (1,), "bias": (2,), "out": (1,)}


def ops(banks=True):
    bk = (lambda n: BANKS[n]) if banks else (lambda n: ())  # noqa: E731
    o = [Op("MA", "mv", "mov p0, p6"), Op("RA", "mv", "mov r24, #0"),
         Op("FA", "lda", "vlda.fill.512 [p0, lf0, r24]", banks=bk("a"), after=[("RA", 1), ("MA", 3)], instr=FILL_A)]
    for kk in range(8):
        o.append(Op(f"A{kk}", "lda", f"vlda.pop.576 {{a{kk}.ex}}, [p0, lf0, r24]", defs=[(f"a{kk}", "ex", 8)],
                    after=[(f"A{kk - 1}" if kk else "FA", 1)], banks=bk("a"), instr=POP_A))
    # round 16 (r16/popprobe probes 6 / 7): a 576-bit pop brings in one 64 B line but consumes 72 B, so a dense stream underflows
    # at its 9th pop (pops 8-12 read zeros) unless a fill at the same position follows every 8 pops: FB0 before B00, FB1 before B40
    prev = None
    for kk in range(8):
        if kk in (0, 4):
            f = f"FB{kk // 4}"
            o.append(Op(f, "ldb", "vldb.fill.512 [p1, lf1, r25]", after=[(prev, 1)] if prev else [], banks=bk("b"), instr="VLDB_FILL_512"))
            prev = f
        for c in range(2):
            n = f"B{kk}{c}"
            o.append(Op(n, "ldb", f"vldb.pop.576 {{b{kk}{c}.ex}}, [p1, lf1, r25]", defs=[(f"b{kk}{c}", "ex", 8)],
                        after=[(prev, 1)], banks=bk("b"), instr=POP_B))
            prev = n
    for kk in range(8):
        for c in range(2):
            acc, a, b = f"acc{c}", f"a{kk}", f"b{kk}{c}"
            if kk == 0:
                o.append(Op(f"M{kk}{c}", "vec", f"vmul.f {{{acc}.dm}}, {{{a}.ex}}, {{{b}.ex}}, {R_MM}", defs=[(acc, "dm", 6)],
                            uses=[(a, "ex", 1), (b, "ex", 1)], instr=BMUL))
            else:
                o.append(Op(f"M{kk}{c}", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{{a}.ex}}, {{{b}.ex}}, {R_MM}", defs=[(acc, "dm", 6)],
                            uses=[(acc, "dm", 4), (a, "ex", 1), (b, "ex", 1)], instr=BMAC))
    prev_l = prev_s = None
    for c in range(2):
        for h, q in ((0, "cml"), (1, "cmh")):
            r, z = f"r{c}{h}", f"z{c}{h}"
            o.append(Op(f"CV{c}{h}", "st", f"vconv.bf16.fp32 {{{r}.x}}, {{acc{c}.{q}}}", defs=[(r, "x", 2)], uses=[(f"acc{c}", q, 1)], instr=CONV))
            if NO_VM:
                o.append(Op(f"ST{c}{h}", "st", f"vst {{{r}.x}}, [p3], #64", uses=[(r, "x", 1)], mem=("st", "out"),
                            banks=bk("out"), after=[(prev_s, 1)] if prev_s else [], instr="VST_dmx_sts_x_pstm_nrm_imm"))
                prev_s = f"ST{c}{h}"
                continue
            if NO_BIAS:
                o.append(Op(f"VM{c}{h}", "vec", f"vmul.f {{{z}.dm}}, {{{r}.x}}, x{C_ONE}, {R_MAC}", defs=[(z, "dm", 6)],
                            uses=[(r, "x", 1)], instr="VMUL_f_vmul_bf_vmul_bf_core_X_X"))
                o.append(Op(f"ST{c}{h}", "st", f"vst.conv.bf16.fp32 {{{z}.cml}}, [p3], #64", uses=[(z, "cml", 1)], mem=("st", "out"),
                            banks=bk("out"), after=[(prev_s, 1)] if prev_s else [], instr=ST_C))
                prev_s = f"ST{c}{h}"
                continue
            o.append(Op(f"LB{c}{h}", "lda", f"vlda.conv.fp32.bf16 {{{z}.cml}}, [p2], #64" if BIAS_1D else
                        f"vlda.2d.conv.fp32.bf16 {{{z}.cml}}, [p2], d2", defs=[(z, "cml", 7)], mem=("ld", "bias"),
                        banks=bk("bias"), after=[(prev_l, 1)] if prev_l else [], instr=LDA_CP if BIAS_1D else LDA_C2))
            prev_l = f"LB{c}{h}"
            o.append(Op(f"VM{c}{h}", "vec", f"vmac.f {{{z}.dm}}, {{{z}.dm}}, {{{r}.x}}, x{C_ONE}, {R_MAC}", defs=[(z, "dm", 6)],
                        uses=[(z, "cml", 4), (r, "x", 1)], instr=VMAC))
            o.append(Op(f"ST{c}{h}", "st", f"vst.conv.bf16.fp32 {{{z}.cml}}, [p3], #64", uses=[(z, "cml", 1)], mem=("st", "out"),
                        banks=bk("out"), after=[(prev_s, 1)] if prev_s else [], instr=ST_C))
            prev_s = f"ST{c}{h}"
    if POP_EXCL:
        for x in o:
            if x.name in ("FA",) or x.name[0] in "AB" and x.name[1:].isdigit():
                x.ports = x.ports + [(0, "POPX")]
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


def maxdist(II):
    if NO_BIAS:
        return [("MA", "A7", II - 1), ("RA", "A7", II - 2), ("B00", "B71", II - 1), ("ST00", "ST11", II - 1)]
    """a FIFO's ops of one iteration inside one II, in order: the next iteration's A restart after this iteration's last A pop;
    the B stream's 16 pops inside one II (the next iteration's first pop comes after this one's last)"""
    return [("MA", "A7", II - 1), ("RA", "A7", II - 2), ("FB0", "B71", II - 1), ("LB00", "LB11", II - 1), ("ST00", "ST11", II - 1)]


SLACK = 0  # round 16: cycles a register stays reserved after its last read (pops landing a cycle early: --slack 1)


def mk(banks=True):
    lp = Loop(ops(banks), consts={"x": [C_ONE]}, classes={"x": X_FREE, "dm": DM_FREE})
    lp.slack = SLACK
    return lp


LAST = [f"ST{c}{h}" for c in range(2) for h in range(2)]


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def build(II=None, times=None, cp=None, max_ii=28, banks=True, U=1):
    global SLACK
    global POP_EXCL
    if times is not None:
        SLACK = times.get("slack", 0)
        POP_EXCL = times.get("popexcl", POP_EXCL)
        lp = mk(banks)
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U", U))
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 20, (II or max_ii) + 1):
        lp = mk(banks)
        res = cpsched.solve(lp, ii, last_stage=LAST, max_life=U * ii, H=8 * ii, maxdist=maxdist(ii), slack=SLACK, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(U)
            side_effects_last(lp)
            check(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0], "U": lp.U, "slack": SLACK, "popexcl": POP_EXCL}
        return lp
    raise Infeasible("none")


def check(lp, n=NIT):
    if NO_BIAS:
        return lp.check(n - n % lp.U, {"out": (lambda i, k: 256 * i + 64 * k, dict(base=0x20000, dn=0, dj=64, m=64))})
    if BIAS_1D:
        return lp.check(n - n % lp.U, {"bias": (lambda i, k: 256 * i + 64 * k, dict(base=0x10000, dn=0, dj=64, m=64)),
                                       "out": (lambda i, k: 256 * i + 64 * k, dict(base=0x20000, dn=0, dj=64, m=64))})
    # bias: 16 loads +64, then the jump to tba (m), 16 more (dn 15: the 16th load of a gate takes m)
    st = {"bias": (lambda i, k: (64 * (4 * i + k)) if i < 4 else 0x4000 + 64 * (4 * (i - 4) + k), dict(base=0x10000, dn=15, dj=64, m=0x4000 - 15 * 64)),
          "out": (lambda i, k: 256 * i + 64 * k, dict(base=0x20000, dn=0, dj=64, m=64))}
    return lp.check(n - n % lp.U, st)


def function(lp, name="ggk"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        # callee-saved p6 holds the A restart pointer
        "paddxm\t[sp], #64", "st\tp6, [sp, #-64]", "mov\tp6, p0",
        # the bias modifier: 15 steps of +64, then m = tba - (tbx + 960)
        "mov\tr0, p4", "mov\tr2, p2", "sub\tr0, r0, r2", "movxm\tr3, #-960", "add\tr0, r0, r3", "mov\tm2, r0", "mova\tdn2, #15", "mova\tdj2, #64", "mova\tdc2, #0",
        f"movxm\tr0, #16256", f"vbcst.16\tx{C_ONE}, r0",
        "mova\tr1, #828", "mova\tr5, #780", "mova\tr25, #0",  # B's fills are in the loop (FB0 / FB1)
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NIT // lp.U}", "add.nc\tlc, r0, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tlda\tp6, [sp, #-64]", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop",
          "\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tpaddxm\t[sp], #-64", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--u", type=int, default=1)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=240)
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--func", default="ggk")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--slack", type=int, default=0)
    ap.add_argument("--bias1d", action="store_true")
    ap.add_argument("--nobias", action="store_true")
    ap.add_argument("--popexcl", action="store_true")
    ap.add_argument("--novm", action="store_true")
    o = ap.parse_args()
    global SLACK, BIAS_1D
    SLACK = o.slack
    BIAS_1D = o.bias1d
    global NO_BIAS, POP_EXCL
    NO_BIAS = o.nobias
    POP_EXCL = o.popexcl
    global NO_VM
    NO_VM = o.novm
    NO_BIAS = NO_BIAS or NO_VM
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), banks=not o.nobanks, U=o.u)
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:8])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NIT} iterations)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r16/gen_ggk.py\n#define GGK_U {lp.U}\n#define GGK_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
