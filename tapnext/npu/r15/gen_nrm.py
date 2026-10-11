"""Round 15 (research P7-E, first step): the norm ops' pass-2 math as modulo-scheduled asm loops (EW cores, ew8.cc norm_rows).

Per k-step q of a row block (8 column blocks cb, 2 halves h of 32 lanes), op for op today's C++ (bit-identical):
  RMS:  sc = rep4(p0 + 8 cb)                              (8 bf16 repeated 4 times: vlda.128 + 4 vshift, as Peano does)
        z_h = bf(bf(x_h r_h) sc)
  LN:   ga = rep4(p0 + 8 cb), be = rep4(p1 + 8 cb)
        t = nb_h + x_h r_h + x_h rl_h;  th = bf(t), tl = bf(t - th 1)
        z_h = bf(be 1 + th ga + tl ga)                    (z also back over x in the stage, for to_x)
  then  push bfp(fp32(z_0 | z_1)) (64 lanes) into the panel stream; one flush after the loop.
The X' / panel copies into the output ring stay in C++ (copy_run).

    extern "C" void nrm_rms(const bfloat16 *x /*p0*/, const bfloat16 *p0 /*p1*/, bfp16ebs8 *panels /*p2*/, const bfloat16 *rr /*p3*/,
                            bfloat16 *xdst /*p4: X' out*/)
        rr: r_0 | r_1 (2 x 64 B)
    extern "C" void nrm_ln(bfloat16 *x /*p0*/, const bfloat16 *p0 /*p1*/, bfp16ebs8 *panels /*p2*/, const void *rr /*p3*/,
                           const bfloat16 *p1 /*p4*/, bfloat16 *xdst /*p5: X' out*/)
        rr: r_0 | r_1 | rl_0 | rl_1 (4 x 64 B) | nb_0 | nb_1 (2 x 128 B fp32)

    python3 r15/gen_nrm.py OUT.s --mode rms|ln [--ii N] [--cp F.json | --times F.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

NCB = 8
C_ONE = 11
R_MAC = "r1"
R_SH = "r3"  # vshift by 16 bytes
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
LDA_X = "VLDA_dmx_lda_x_pstm_nrm_imm"
LDB_128 = "VLDB_128_pstm_nrm_imm"
LDA_BMI = "VLDA_dmx_lda_bm_idx_imm"
ST_X = "VST_dmx_sts_x_pstm_nrm_imm"
VSHIFT = "VSHIFT"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"


def consts(mode):
    # x constants: 1.0, r_0, r_1 (+ rl_0, rl_1 for LN)
    return [C_ONE, 10, 9] + ([8, 7] if mode == "ln" else [])


def ops(mode):
    R0, R1 = 10, 9
    o = []

    def rep4(tag, ptr):
        w = f"w{tag}"
        o.append(Op(f"LS{tag}", "ldb", f"vldb.128 {{{w}.wl}}, [{ptr}], #16", defs=[(w, "x", 7)], mem=("ld", f"p{tag}"), instr=LDB_128))
        prev = w
        for k in range(4):
            d = f"s{tag}{k}" if k < 3 else f"sc{tag}"
            o.append(Op(f"V{tag}{k}", "mv", f"vshift {{{d}.x}}, {{{prev}.x}}, {{{w}.x}}, {R_SH}", defs=[(d, "x", 2)],
                        uses=[(prev, "x", 1), (w, "x", 1)], instr=VSHIFT))
            prev = d
        return f"sc{tag}"

    o.append(Op("LX0", "lda", "vlda {x0.x}, [p0], #64", defs=[("x0", "x", 7)], mem=("ld", "x"), instr=LDA_X))
    o.append(Op("LX1", "lda", "vlda {x1.x}, [p0], #64", defs=[("x1", "x", 7)], mem=("ld", "x"), instr=LDA_X, after=[("LX0", 1)]))
    if mode == "rms":
        sc = rep4("a", "p1")
        for h, R in ((0, R0), (1, R1)):
            o += [
                Op(f"M1{h}", "vec", f"vmul.f {{a{h}.dm}}, {{x{h}.x}}, x{R}, {R_MAC}", defs=[(f"a{h}", "dm", 6)], uses=[(f"x{h}", "x", 1)], instr=VMUL),
                Op(f"C1{h}", "st", f"vconv.bf16.fp32 {{b{h}.x}}, {{a{h}.cml}}", defs=[(f"b{h}", "x", 2)], uses=[(f"a{h}", "cml", 1)], instr=CONV),
                Op(f"M2{h}", "vec", f"vmul.f {{c{h}.dm}}, {{b{h}.x}}, {{{sc}.x}}, {R_MAC}", defs=[(f"c{h}", "dm", 6)],
                   uses=[(f"b{h}", "x", 1), (sc, "x", 1)], instr=VMUL),
                Op(f"C2{h}", "st", f"vconv.bf16.fp32 {{z{h}.x}}, {{c{h}.cml}}", defs=[(f"z{h}", "x", 2)], uses=[(f"c{h}", "cml", 1)], instr=CONV),
            ]
    else:
        ga = rep4("a", "p1")
        be = rep4("b", "p4")
        RL = {0: 8, 1: 7}
        for h, R in ((0, R0), (1, R1)):
            o += [
                # nb_h (fp32, 128 B) reloaded into an accumulator per use: no accumulator held across the loop
                Op(f"NB{h}0", "lda", f"vlda {{n{h}.bmll}}, [p3, #{256 + 128 * h}]", defs=[(f"n{h}", "bmll", 7)], instr=LDA_BMI),
                Op(f"NB{h}1", "lda", f"vlda {{n{h}.bmlh}}, [p3, #{256 + 128 * h + 64}]", defs=[(f"n{h}", "bmlh", 7)], instr=LDA_BMI),
                Op(f"T1{h}", "vec", f"vmac.f {{t1{h}.dm}}, {{n{h}.dm}}, {{x{h}.x}}, x{R}, {R_MAC}", defs=[(f"t1{h}", "dm", 6)],
                   uses=[(f"n{h}", "cml", 4), (f"x{h}", "x", 1)], instr=VMAC),
                Op(f"T2{h}", "vec", f"vmac.f {{t{h}.dm}}, {{t1{h}.dm}}, {{x{h}.x}}, x{RL[h]}, {R_MAC}", defs=[(f"t{h}", "dm", 6)],
                   uses=[(f"t1{h}", "cml", 4), (f"x{h}", "x", 1)], instr=VMAC),
                Op(f"CH{h}", "st", f"vconv.bf16.fp32 {{th{h}.x}}, {{t{h}.cml}}", defs=[(f"th{h}", "x", 2)], uses=[(f"t{h}", "cml", 1)], instr=CONV),
                Op(f"MS{h}", "vec", f"vmsc.f {{d{h}.dm}}, {{t{h}.dm}}, {{th{h}.x}}, x{C_ONE}, {R_MAC}", defs=[(f"d{h}", "dm", 6)],
                   uses=[(f"t{h}", "cml", 4), (f"th{h}", "x", 1)], instr=VMSC),
                Op(f"CL{h}", "st", f"vconv.bf16.fp32 {{tl{h}.x}}, {{d{h}.cml}}", defs=[(f"tl{h}", "x", 2)], uses=[(f"d{h}", "cml", 1)], instr=CONV),
                Op(f"Y0{h}", "vec", f"vmul.f {{y0{h}.dm}}, {{{be}.x}}, x{C_ONE}, {R_MAC}", defs=[(f"y0{h}", "dm", 6)], uses=[(be, "x", 1)], instr=VMUL),
                Op(f"Y1{h}", "vec", f"vmac.f {{y1{h}.dm}}, {{y0{h}.dm}}, {{th{h}.x}}, {{{ga}.x}}, {R_MAC}", defs=[(f"y1{h}", "dm", 6)],
                   uses=[(f"y0{h}", "cml", 4), (f"th{h}", "x", 1), (ga, "x", 1)], instr=VMAC),
                Op(f"Y2{h}", "vec", f"vmac.f {{y{h}.dm}}, {{y1{h}.dm}}, {{tl{h}.x}}, {{{ga}.x}}, {R_MAC}", defs=[(f"y{h}", "dm", 6)],
                   uses=[(f"y1{h}", "cml", 4), (f"tl{h}", "x", 1), (ga, "x", 1)], instr=VMAC),
                Op(f"C2{h}", "st", f"vconv.bf16.fp32 {{z{h}.x}}, {{y{h}.cml}}", defs=[(f"z{h}", "x", 2)], uses=[(f"y{h}", "cml", 1)], instr=CONV),
                Op(f"SZ{h}", "st", f"vst {{z{h}.x}}, [p6], #64", uses=[(f"z{h}", "x", 1)], mem=("st", "z"), instr=ST_X,
                   after=([("SZ0", 1)] if h else [])),
            ]
    px = "p5" if mode == "ln" else "p4"
    for h in (0, 1):  # round 15: X' straight into the output ring (or the caller's scratch when the unit's X' wraps)
        xv = f"x{h}"
        if mode == "ln":  # LN: x reloaded (p7 walks the stage like p0) before this iteration's z overwrites it: short lifetimes
            xv = f"xr{h}"
            o.append(Op(f"RX{h}", "lda", f"vlda {{{xv}.x}}, [p7], #64", defs=[(xv, "x", 7)], mem=("ld", "xre"), instr=LDA_X,
                        after=[("RX0", 1)] if h else []))
        if mode == "ln":
            next(x for x in o if x.name == f"SZ{h}").after.append((f"RX{h}", 4))  # the reload reads before z is stored over x
        o.append(Op(f"SX{h}", "st", f"vst {{{xv}.x}}, [{px}], #64", uses=[(xv, "x", 1)], mem=("st", "xr"), instr=ST_X,
                    after=[("SX0", 1)] if h else []))
    o += [
        Op("U0", "mv", "vconv.fp32.bf16 {zz.cml}, {z0.x}", defs=[("zz", "cml", 2)], uses=[("z0", "x", 1)], instr=UPS),
        Op("U1", "mv", "vconv.fp32.bf16 {zz.cmh}, {z1.x}", defs=[("zz", "cmh", 2)], uses=[("z1", "x", 1)], instr=UPS),
        Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {zz.dm}, [p2, sf, r26]", uses=[("zz", "dm", 1)], instr=PUSH),
    ]
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


def mk(mode):
    cs = consts(mode)
    return Loop(ops(mode), consts={"x": cs}, classes={"x": [r for r in range(12) if r not in cs], "dm": [0, 1, 2, 3, 4]})


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if (o.name == "PU" or (o.mem and o.mem[0] == "st")) and o.t // lp.II != S - 1]
    assert not bad, f"side effects before the last stage: {bad}"


def maxdist(mode, II):
    md = [("LX0", "LX1", II - 1), ("SX0", "SX1", II - 1)]
    if mode == "ln":
        md += [("SZ0", "SZ1", II - 1), ("RX0", "RX1", II - 1)]
    return md


def build(mode, II=None, times=None, cp=None, max_ii=20, U=1):
    side = ["PU", "SX0", "SX1"] + (["SZ0", "SZ1"] if mode == "ln" else [])
    if times is not None:
        lp = mk(mode)
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U", U))
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 6, (II or max_ii) + 1):
        lp = mk(mode)
        res = cpsched.solve(lp, ii, last_stage=side, max_life=U * ii, H=8 * ii, maxdist=maxdist(mode, ii), **(cp or {}))
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
        try:
            check(lp, mode)
        except AssertionError as e:  # (e.g. a port collision only with the .p2align pad after the prologue)
            print(f"II {ii}: check failed: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0], "U": U, "mode": mode}
        return lp
    raise Infeasible("none")


def check(lp, mode, n=NCB):
    st = {"x": (lambda i, k: 128 * i + 64 * k, dict(base=0x10000, dn=0, dj=64, m=64)),
          "pa": (lambda i: 16 * i, dict(base=0x20000, dn=0, dj=16, m=16)),
          "xr": (lambda i, k: 128 * i + 64 * k, dict(base=0x50000, dn=0, dj=64, m=64))}
    if mode == "ln":
        st["pb"] = (lambda i: 16 * i, dict(base=0x30000, dn=0, dj=16, m=16))
        st["z"] = (lambda i, k: 128 * i + 64 * k, dict(base=0x40000, dn=0, dj=64, m=64))
        st["xre"] = (lambda i, k: 128 * i + 64 * k, dict(base=0x10000, dn=0, dj=64, m=64))
    return lp.check(n, st)


def function(lp, mode, name):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [f"vlda\tx10, [p3, #0]", f"vlda\tx9, [p3, #64]"]
    if mode == "ln":  # z back over x through p6 (callee-saved: kept on the stack)
        pre += [f"vlda\tx8, [p3, #128]", f"vlda\tx7, [p3, #192]", "paddxm\t[sp], #64", "st\tp6, [sp, #-64]", "st\tp7, [sp, #-60]", "mov\tp6, p0", "mov\tp7, p0"]
    pre += ["movxm\tr0, #16256", f"vbcst.16\tx{C_ONE}, r0", f"mova\t{R_MAC}, #828", f"mova\t{R_SH}, #16", "mova\tr26, #0",
            f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NCB // lp.U}", "add.nc\tlc, r0, #0"]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    # the panel stream's last bytes: one flush after the loop (round 12: flush per push / at the end give the same bytes)
    post = ["\tlda\tp6, [sp, #-64]", "\tlda\tp7, [sp, #-60]", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", "\tpaddxm\t[sp], #-64"] if mode == "ln" else []
    L += ["\tvst.flush.512.conv\t[p2, sf, r26]"] + post + ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop",
          f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["rms", "ln"], required=True)
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--u", type=int, default=1)
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.mode, o.ii, times, cp=dict(time_limit=o.time), U=o.u)
    assert lp.stages() >= 2, "the prologue gives the loop setup distance"
    cyc, land = check(lp, o.mode)
    print(f"{o.mode}: II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages "
          f"{lp.stages()}, maxlive {lp.maxlive()}; check OK ({cyc} cycles)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    name = f"nrm_{o.mode}"
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.mode, name)) + "\n", name, o.out, o.fit)
    print("wrote", o.out)


if __name__ == "__main__":
    main()
