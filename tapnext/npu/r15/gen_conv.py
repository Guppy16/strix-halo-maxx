"""Round 15 (npu-research P29d, the conv half): the SSM's causal conv1d as one modulo-scheduled asm loop (EW cores, ew8.cc ew_ssm).

Today (ew8.cc ew_ssm, round 8): per 32 lanes (4 rows x 8 columns of a 1 KB k-step piece)
    s  = bf(x3 w0) 1;  s += bf(x2 w1) 1;  s += bf(x1 w2) 1;  s += bf(x0 w3) 1        (fp32, one rounding per add)
    xc = bf(bf(s) 1 + b 1)
with x3 .. x0 = the input parts x-3, x-2, x-1, x (P[4] .. P[1]), w_t / b the head's tap / bias vectors per column block
(g_tcw [t][cb], g_tcb [cb]: rep4 of 8 values). ~1,125 cycles per chunk in situ (r15 PROF, incl. the 7 in_part calls).
Here: one iteration = one column block = 2 vectors (h = 0, 1 share w / b), the same MACs and roundings (bit-identical):
10 vec ops per vector (vec bound II 20 per iteration). The tap table is re-laid out by ssm_tables as [cb][w0 w1 w2 w3 b]
(320 B per cb, the same 2,560 B region): one pointer, immediate offsets, one padda per iteration.

    extern "C" void convk(const bfloat16 *x3 /*p0*/, const bfloat16 *x2 /*p1*/, const bfloat16 *x1 /*p2*/,
                          const bfloat16 *x0 /*p3*/, const void *tab /*p4*/, bfloat16 *xc /*p5*/)
over 8 column blocks (16 vectors of each part). The drain's garbage iterations only load (past the parts / the table).

    python3 r15/gen_conv.py OUT.s [--ii N] [--cp F.json | --times F.json] [--time S] [--fit]
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
X_FREE = list(range(11))
DM_FREE = [0, 1, 2, 3, 4]
ROWB = 320  # table bytes per column block
BANK = dict(x=(3,), tab=(2,), out=(1,))  # in situ: the input ring (bank 3), the arena's table (2), g_xc (1)
R_MAC = "r1"
# round 18 (CGF, research rev 13): convk also pushes xc as the gate GEMMs' bfp panels (the C++ push loop + its glue go). The push
# pointer must be p2, so part x1 moves to p6 (callee-saved: kept on the stack) and p2 = g_xcpb_k (a symbol); xc is rounded to bf16
# once (vconv), stored, and widened back (vconv.fp32.bf16, mv slot) into one accumulator for the push: the bf16 values' panels,
# as the C++ A64(load(xc)).to_vector<bfp16ebs8>()
PUSH = False
PART_PTR = {0: "p0", 1: "p1", 2: "p2", 3: "p3"}
VST_X = "VST_dmx_sts_x_pstm_nrm_imm"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
PUSH_I = "VST_PUSH_576_CONV_bfp16ebs8_fp32"

VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
LDA_X = "VLDA_dmx_lda_x_pstm_nrm_imm"
LDB_X = "VLDB_dmx_ldb_x_pstm_nrm_imm"
LDB_I = "VLDB_dmx_ldb_x_idx_imm"
LDA_I = "VLDA_dmx_lda_x_idx_imm"
ST_CONV = "VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm"
PADDA = "PADDA_pstm_nrm_imm"


def ops(xslots="abab", wslots="bbbbb", banks=True):
    """xslots: load slot per input part (x3 x2 x1 x0); wslots: per table row (w0..w3, b)"""
    bk = (lambda n: BANK[n]) if banks else (lambda n: ())  # noqa: E731
    o = []
    for r in range(5):
        s = wslots[r]
        o.append(Op(f"W{r}", "lda" if s == "a" else "ldb", f"vld{s} {{w{r}.x}}, [p4, #{64 * r}]", defs=[(f"w{r}", "x", 7)],
                    mem=("ld", "tab"), banks=bk("tab"), instr=LDA_I if s == "a" else LDB_I))
    o.append(Op("PA", "lda", f"padda [p4], #{ROWB}", mem=("pa", "tab"), after=[(f"W{r}", 1) for r in range(5)], instr=PADDA))
    for h in (0, 1):
        for t in range(4):
            s = xslots[t]
            o.append(Op(f"X{t}{h}", "lda" if s == "a" else "ldb", f"vld{s} {{x{t}{h}.x}}, [{PART_PTR[t]}], #64", defs=[(f"x{t}{h}", "x", 7)],
                        mem=("ld", f"x{t}"), banks=bk("x"), instr=LDA_X if s == "a" else LDB_X,
                        after=[(f"X{t}0", 1)] if h else []))
        for t in range(4):
            o.append(Op(f"M{t}{h}", "vec", f"vmul.f {{p{t}{h}.dm}}, {{x{t}{h}.x}}, {{w{t}.x}}, {R_MAC}", defs=[(f"p{t}{h}", "dm", 6)],
                        uses=[(f"x{t}{h}", "x", 1), (f"w{t}", "x", 1)], instr=VMUL))
            o.append(Op(f"C{t}{h}", "st", f"vconv.bf16.fp32 {{b{t}{h}.x}}, {{p{t}{h}.cml}}", defs=[(f"b{t}{h}", "x", 2)],
                        uses=[(f"p{t}{h}", "cml", 1)], instr=CONV))
        o.append(Op(f"S0{h}", "vec", f"vmul.f {{s0{h}.dm}}, {{b0{h}.x}}, x{C_ONE}, {R_MAC}", defs=[(f"s0{h}", "dm", 6)],
                    uses=[(f"b0{h}", "x", 1)], instr=VMUL))
        for t in (1, 2, 3):
            o.append(Op(f"S{t}{h}", "vec", f"vmac.f {{s{t}{h}.dm}}, {{s{t - 1}{h}.dm}}, {{b{t}{h}.x}}, x{C_ONE}, {R_MAC}",
                        defs=[(f"s{t}{h}", "dm", 6)], uses=[(f"s{t - 1}{h}", "cml", 4), (f"b{t}{h}", "x", 1)], instr=VMAC))
        o.append(Op(f"CS{h}", "st", f"vconv.bf16.fp32 {{bs{h}.x}}, {{s3{h}.cml}}", defs=[(f"bs{h}", "x", 2)], uses=[(f"s3{h}", "cml", 1)], instr=CONV))
        o.append(Op(f"MS{h}", "vec", f"vmul.f {{m{h}.dm}}, {{bs{h}.x}}, x{C_ONE}, {R_MAC}", defs=[(f"m{h}", "dm", 6)],
                    uses=[(f"bs{h}", "x", 1)], instr=VMUL))
        o.append(Op(f"AB{h}", "vec", f"vmac.f {{y{h}.dm}}, {{m{h}.dm}}, {{w4.x}}, x{C_ONE}, {R_MAC}", defs=[(f"y{h}", "dm", 6)],
                    uses=[(f"m{h}", "cml", 4), ("w4", "x", 1)], instr=VMAC))
        if PUSH:
            o.append(Op(f"CX{h}", "st", f"vconv.bf16.fp32 {{xb{h}.x}}, {{y{h}.cml}}", defs=[(f"xb{h}", "x", 2)], uses=[(f"y{h}", "cml", 1)],
                        instr=CONV))
            o.append(Op(f"ST{h}", "st", f"vst {{xb{h}.x}}, [p5], #64", uses=[(f"xb{h}", "x", 1)], mem=("st", "out"), banks=bk("out"),
                        instr=VST_X, after=[("ST0", 1)] if h else []))
            o.append(Op(f"U{h}", "mv", f"vconv.fp32.bf16 {{zz.{'cml' if h == 0 else 'cmh'}}}, {{xb{h}.x}}",
                        defs=[("zz", "cml" if h == 0 else "cmh", 2)], uses=[(f"xb{h}", "x", 1)], instr=UPS))
            continue
        o.append(Op(f"ST{h}", "st", f"vst.conv.bf16.fp32 {{y{h}.cml}}, [p5], #64", uses=[(f"y{h}", "cml", 1)], mem=("st", "out"),
                    banks=bk("out"), instr=ST_CONV, after=[("ST0", 1)] if h else []))
    if PUSH:
        o.append(Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {zz.dm}, [p2, sf, r26]", uses=[("zz", "dm", 1)], instr=PUSH_I))
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


def mk(xslots="abab", wslots="bbbbb", banks=True):
    return Loop(ops(xslots, wslots, banks), consts={"x": [C_ONE]}, classes={"x": X_FREE, "dm": DM_FREE})


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def maxdist(II):
    # the table pointer moves once per iteration: its 5 loads of iteration i before its padda, the padda before
    # iteration i + 1's loads (II later): all inside one II
    return [(f"W{r}", "PA", II - 1) for r in range(5)] + [(f"X{t}0", f"X{t}1", II - 1) for t in range(4)] + [("ST0", "ST1", II - 1)]


def build(II=None, times=None, cp=None, max_ii=26, U=1, xslots="abab", wslots="bbbbb", banks=True):
    if times is not None:
        lp = mk(times.get("xslots", xslots), times.get("wslots", wslots), banks)
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U", U))
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 20, (II or max_ii) + 1):
        lp = mk(xslots, wslots, banks)
        res = cpsched.solve(lp, ii, last_stage=["ST0", "ST1"] + (["PU"] if PUSH else []), max_life=U * ii, H=6 * ii, maxdist=maxdist(ii),
                            **(cp or {}))
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
        lp.times = {"II": ii, "t": res[0], "xslots": xslots, "wslots": wslots, "U": U}
        return lp
    raise Infeasible("none")


def check(lp, n=NCB):
    noop = dict(dn=0, dj=0, m=0)
    st = {f"x{t}": (lambda i, k: 128 * i + 64 * k, dict(base=0x10000 * (t + 1), dn=0, dj=64, m=64)) for t in range(4)}
    st["out"] = (lambda i, k: 128 * i + 64 * k, dict(base=0x50000, dn=0, dj=64, m=64))
    # the table: W0..W4 read [p4, #64 r] (the pointer itself does not move), PA steps it by a row
    st["tab"] = (lambda i, k: ROWB * i, dict(base=0x60000, dn=0, dj=0, m=0,
                                             ops={**{f"W{r}": noop for r in range(5)}, "PA": dict(dn=0, dj=0, m=ROWB)}))
    return lp.check(n, st)


def function(lp, name="convk"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    push = ["paddxm\t[sp], #64", "st\tp6, [sp, #-64]", "mov\tp6, p2", "movxm\tp2, #g_xcpb_k", "mova\tr26, #0"] if PUSH else []
    pre = push + [
        f"movxm\tr0, #16256", f"vbcst.16\tx{C_ONE}, r0", "mova\tr1, #828",
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NCB // lp.U}", "add.nc\tlc, r0, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    if PUSH:
        L += ["\tvst.flush.512.conv\t[p2, sf, r26]", "\tlda\tp6, [sp, #-64]", "\tpaddxm\t[sp], #-64"]
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--u", type=int, default=1)
    ap.add_argument("--xslots", default="abab")
    ap.add_argument("--wslots", default="bbbbb")
    ap.add_argument("--func", default="convk")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--push", action="store_true", help="round 18 CGF: also push xc's bfp panels to g_xcpb_k (part x1 on p6)")
    ap.add_argument("--workers", type=int, default=4)
    o = ap.parse_args()
    global PUSH
    PUSH = o.push
    if PUSH:
        PART_PTR[2] = "p6"
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time, workers=o.workers), U=o.u, xslots=o.xslots, wslots=o.wslots)
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NCB} column blocks)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r15/gen_conv.py\n#define CONVK_U {lp.U}\n#define CONVK_II {lp.II}\n#define CONVK_ROWB {ROWB}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
