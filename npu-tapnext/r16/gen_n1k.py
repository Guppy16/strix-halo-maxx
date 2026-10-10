"""Round 16 (research N1): the norm's pass 1 AND its statistics for one unit (8 column blocks of one k-step) as one modulo-scheduled
asm loop (EW cores, ew8.cc norm_rows_k; flag -DN1K). Bit-identical to norm_rows_k's pass-1 loop + statistics loop: per cb
    pass 1  y_h = bf16(pa_h 1 + pb_h w1 + pc_h w2)   (h = 0, 1: the two 32-lane halves), stored to the stage
    LN      s += y 1;  q += y y                      (fp32, 64 lanes: y = y_0 | y_1 in one register pair)
    RMS     q += bf16(y y) 1
in the C++'s order (k = q-major, cb-minor per lane), so the per-lane fp32 sums are the same. The statistics accumulators are
LOOP-CARRIED (msched cuses, U = 1): the preamble loads them from `acc` (s: 256 B, q: 256 B; zeroed by the caller per row
block), the epilogue stores them back. 64-lane bf16 MACs read register PAIRS y_n = x_2n : x_2n+1 (Peano's vmul.f dm, y3, y2):
a "y" register class owns x8:x9 (y4: the pass-1 result) and, for RMS, x6:x7 (y3: the squares); 1.0 lives in y5 = x10:x11.
The ring wrap per input stream as r16/gen_n1p.py (2D modifier, counter start 8 - n1).

    extern "C" void n1k_ln_begin / _step / _end (and n1k_rms_*): see functions() - constants, modifiers and the two
    accumulators stay in registers across a row block's 12 unit calls

    python3 r16/gen_n1k.py OUT.s --mode ln|rms [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
import msched  # noqa: E402
from msched import Infeasible, Loop, Op  # noqa: E402

# the y class: register pairs x_2n : x_2n+1 (generator-level, as r14/gen_attv.py adds bm quarters)
msched.PARTS.setdefault("y", ("y", (0, 1, 2, 3)))
msched.PARTS.setdefault("ylo", ("y", (0, 1)))
msched.PARTS.setdefault("yhi", ("y", (2, 3)))
msched.PHYS.setdefault("y", lambda n, p: f"y{n}" if p == "y" else f"x{2 * n + (p == 'yhi')}")

NIT = 8
MODE = "ln"
Y_ONE = 5  # x10 : x11 = 1.0
VLDA = "VLDA_dmx_lda_x_pstm_nrm_imm"
VLDB = "VLDB_dmx_ldb_x_pstm_nrm_imm"
VLDA2 = "VLDA_2D_dmx_lda_x"
VLDB2 = "VLDB_2D_dmx_ldb_x"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
VST = "VST_dmx_sts_x_pstm_nrm_imm"
BANKS = {"in": (3,), "st": (1,)}
SLOT = {"a": "lda", "b": "ldb", "c": "lda"}


def regs():
    """(x allocatable, W1, W2, y allocatable)"""
    if MODE == "ln":
        return list(range(8)), "x6", "x7", [4]
    return list(range(4)), "x4", "x5", [3, 4]


def ops(banks=True):
    bk = (lambda n: BANKS[n]) if banks else (lambda n: ())  # noqa: E731
    _, W1, W2, _ = regs()
    o = []
    for s, p, d in (("a", "p0", "d0"), ("b", "p1", "d1"), ("c", "p2", "d2")):
        sl = SLOT[s]
        L = "vlda" if sl == "lda" else "vldb"
        o.append(Op(f"L{s}0", sl, f"{L} {{{s}0.x}}, [{p}], #64", defs=[(f"{s}0", "x", 7)], mem=("ld", s), banks=bk("in"),
                    instr=VLDA if sl == "lda" else VLDB))
        o.append(Op(f"L{s}1", sl, f"{L}.2d {{{s}1.x}}, [{p}], {d}", defs=[(f"{s}1", "x", 7)], mem=("ld", s), banks=bk("in"),
                    after=[(f"L{s}0", 1)], instr=VLDA2 if sl == "lda" else VLDB2))
    for h in (0, 1):
        acc = f"p{h}"
        o.append(Op(f"M{h}a", "vec", f"vmul.f {{{acc}.dm}}, {{a{h}.x}}, x11, r6", defs=[(acc, "dm", 6)], uses=[(f"a{h}", "x", 1)], instr=VMUL))
        o.append(Op(f"M{h}b", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{b{h}.x}}, {W1}, r6", defs=[(acc, "dm", 6)],
                    uses=[(acc, "cml", 4), (f"b{h}", "x", 1)], instr=VMAC))
        o.append(Op(f"M{h}c", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{c{h}.x}}, {W2}, r6", defs=[(acc, "dm", 6)],
                    uses=[(acc, "cml", 4), (f"c{h}", "x", 1)], instr=VMAC))
        q = "ylo" if h == 0 else "yhi"
        o.append(Op(f"CV{h}", "st", f"vconv.bf16.fp32 {{y.{q}}}, {{{acc}.cml}}", defs=[("y", q, 2)], uses=[(acc, "cml", 1)], instr=CONV))
    o.append(Op("ST0", "st", "vst {y.ylo}, [p3], #64", uses=[("y", "ylo", 1)], mem=("st", "st"), banks=bk("st"), instr=VST))
    o.append(Op("ST1", "st", "vst {y.yhi}, [p3], #64", uses=[("y", "yhi", 1)], mem=("st", "st"), banks=bk("st"), after=[("ST0", 1)], instr=VST))
    if MODE == "ln":
        o.append(Op("SS", "vec", f"vmac.f {{ss.dm}}, {{ss.dm}}, {{y.y}}, y{Y_ONE}, r6", defs=[("ss", "dm", 6)], uses=[("y", "y", 1)],
                    cuses=[("ss", "dm", 4)], instr=VMAC))
        o.append(Op("SQ", "vec", "vmac.f {sq.dm}, {sq.dm}, {y.y}, {y.y}, r6", defs=[("sq", "dm", 6)], uses=[("y", "y", 1)],
                    cuses=[("sq", "dm", 4)], instr=VMAC))
    else:
        o.append(Op("QM", "vec", "vmul.f {t.dm}, {y.y}, {y.y}, r6", defs=[("t", "dm", 6)], uses=[("y", "y", 1)], instr=VMUL))
        o.append(Op("QC0", "st", "vconv.bf16.fp32 {u.ylo}, {t.cml}", defs=[("u", "ylo", 2)], uses=[("t", "cml", 1)], instr=CONV))
        o.append(Op("QC1", "st", "vconv.bf16.fp32 {u.yhi}, {t.cmh}", defs=[("u", "yhi", 2)], uses=[("t", "cmh", 1)], instr=CONV))
        o.append(Op("SQ", "vec", f"vmac.f {{sq.dm}}, {{sq.dm}}, {{u.y}}, y{Y_ONE}, r6", defs=[("sq", "dm", 6)], uses=[("u", "y", 1)],
                    cuses=[("sq", "dm", 4)], instr=VMAC))
    return o


def mk(banks=True):
    X, W1, W2, Y = regs()
    consts = {"x": [int(W1[1:]), int(W2[1:])], "y": [Y_ONE]}
    return Loop(ops(banks), consts=consts, classes={"x": X, "dm": [0, 1, 2, 3, 4], "y": Y})




def last_ops():
    return ["ST0", "ST1", "SQ"] + (["SS"] if MODE == "ln" else [])


def maxdist(II):
    return [("La0", "La1", II - 1), ("Lb0", "Lb1", II - 1), ("Lc0", "Lc1", II - 1), ("ST0", "ST1", II - 1)]


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if (o.mem and o.mem[0] == "st" or o.cuses) and o.t // lp.II != S - 1]
    assert not bad, f"stores / carried accumulations before the last stage: {bad}"


def build(II=None, times=None, cp=None, max_ii=16, banks=True):
    if times is not None:
        lp = mk(banks)
        lp.place(times["II"], times["t"])
        lp.allocate(1)
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 10, (II or max_ii) + 1):
        lp = mk(banks)
        res = cpsched.solve(lp, ii, last_stage=last_ops(), max_life=ii, maxdist=maxdist(ii), **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(1)
            side_effects_last(lp)
            check(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0], "U": 1, "mode": MODE}
        return lp
    raise Infeasible("none")


def check(lp, n=NIT):
    st = {s: (lambda i, k: 128 * i + 64 * k, dict(base=0x10000 * (j + 1), dn=0, dj=64, m=64,
                                                   ops={f"L{s}1": dict(dn=1000, dj=64, m=64)})) for j, s in enumerate("abc")}
    st["st"] = (lambda i, k: 128 * i + 64 * k, dict(base=0x40000, dn=0, dj=64, m=64))
    return lp.check(n, st)


def acc_regs(lp):
    """physical dm registers of the carried accumulators: [(name, dm number)] in the acc layout order (s, then q)"""
    names = (["ss"] if MODE == "ln" else []) + ["sq"]
    return [(v, lp.alloc[v, 0]) for v in names]


def _fn(name, body):
    return ([f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
            + ["\t" + x for x in body] + [f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"])


def acc_moves(lp, op, ptr):
    """vlda / vst of the carried accumulators (s at +0, q at +256) through ptr (p0); LN: s then q"""
    out = []
    for v, r in acc_regs(lp):
        out += ["mov\tp1, p0"] + (["padda\t[p1], #256"] if v == "sq" else [])
        out += [f"{op}\tbm{q}{r}, [p1], #64" for q in ("ll", "lh", "hl", "hh")]
    return out


def functions(lp, name):
    """round 16: three entries so the per-unit call carries only the loop. Everything set in _begin stays in registers across
    the row block's 12 _step calls: the C++ between them (in_need / in_done / pointer math) is scalar (checked in the elf);
    m0-m2 and r6 are scalar-side registers and are set per step.
        _begin(float *acc /*p0*/, int w1 /*r0*/, int w2 /*r1*/): constants, dn / dj, the accumulators from acc
        _step(pa, pb, pc, st /*p0-p3*/, int dca, int dcb, int dcc /*r0-r2*/, int m /*r3: 64 - IN_B*/): one unit
        _end(float *acc /*p0*/): the accumulators to acc"""
    _, W1, W2, _ = regs()
    begin = ["mova\tdn0, #7", "mova\tdn1, #7", "mova\tdn2, #7", "mova\tdj0, #64", "mova\tdj1, #64", "mova\tdj2, #64",
             f"vbcst.16\t{W1}, r0", f"vbcst.16\t{W2}, r1", "movxm\tr7, #16256", "vbcst.16\tx11, r7", "vbcst.16\tx10, r7"]
    begin += acc_moves(lp, "vlda", "p0") + ["ret\tlr"] + ["nop"] * 5
    step = ["mov\tm0, r3", "mov\tm1, r3", "mov\tm2, r3", "mov\tdc0, r0", "mov\tdc1, r1", "mov\tdc2, r2", "mova\tr6, #828",
            f"movxm\tls, #{name}_step_start", f"movxm\tle, #{name}_step_last", f"mova\tr7, #{NIT}", "add.nc\tlc, r7, #0",
            "nop", "nop", "nop", "nop", "nop", "nop"]
    loop = lp.emit_loop(f"{name}_step", drain=0)
    step_l = _fn(f"{name}_step", step)
    step_l = step_l[:-2] + loop + ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop"] + step_l[-2:]
    end = ["nop"] * 7 + acc_moves(lp, "vst", "p0") + ["ret\tlr"] + ["nop"] * 5
    return _fn(f"{name}_begin", begin) + step_l + _fn(f"{name}_end", end)


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["ln", "rms"], required=True)
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    MODE = o.mode
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), banks=not o.nobanks)
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}, acc {acc_regs(lp)}; check OK ({cyc} cycles, {NIT} iterations)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    fn = f"n1k_{MODE}"
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(functions(lp, fn)) + "\n", f"{fn}_step", o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r16/gen_n1k.py --mode {MODE}\n#define N1K_{MODE.upper()}_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
