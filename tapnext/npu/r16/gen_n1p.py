"""Round 16 (research N1, step 1): the norm's pass 1 for one unit (8 rows x 64 columns of one k-step) as one modulo-scheduled asm
loop (EW cores, ew8.cc norm_rows_k; flag -DN1P). Bit-identical to the C++ loop it replaces: per column block cb (two 32-lane
halves h)
    s_h = bf16(pa_h 1 + pb_h w1 + pc_h w2)   (vmul, then two vmacs in that order; fp32 accumulation, one rounding)
stored to the stage (128 B per cb, contiguous). The three input streams are 1 KB parts of the input ring in 128 B pieces; a part
may wrap the ring end between pieces, so each stream's second load per piece uses a 2D modifier: +64 for dn pieces, then
m = 64 - IN_B (back to the ring start). Missing streams (n_in < 3) re-read stream a with weight 0, as the C++ does.

    extern "C" void n1p(const bfloat16 *pa /*p0*/, const bfloat16 *pb /*p1*/, const bfloat16 *pc /*p2*/, bfloat16 *st /*p3*/,
                        int dca /*r0*/, int dcb /*r1*/, int dcc /*r2*/, int w1 /*r3: bf16 bits*/, int w2 /*r4*/, int m /*r5: 64 - IN_B*/)
dcX = max(0, 8 - n1X), n1X = the 128 B pieces of stream X before the ring end (the 2D counter's start: one jump at piece n1 - 1).

    python3 r16/gen_n1p.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
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
C_ONE, C_W1, C_W2 = 9, 10, 11
X_FREE = list(range(9))
DM_FREE = [0, 1, 2, 3, 4]
VLDA = "VLDA_dmx_lda_x_pstm_nrm_imm"
VLDB = "VLDB_dmx_ldb_x_pstm_nrm_imm"
VLDA2 = "VLDA_2D_dmx_lda_x"
VLDB2 = "VLDB_2D_dmx_ldb_x"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
VST = "VST_dmx_sts_x_pstm_nrm_imm"
# in situ: the input ring ewi (bank 3 per the round-15 notes: "ring bank 3 -> arena"), the stage g_stage = arena (bank 1)
BANKS = {"in": (3,), "st": (1,)}
SLOT = {"a": "lda", "b": "ldb", "c": "lda"}


def ops(banks=True):
    bk = (lambda n: BANKS[n]) if banks else (lambda n: ())  # noqa: E731
    o = []
    for s, p, d in (("a", "p0", "d0"), ("b", "p1", "d1"), ("c", "p2", "d2")):
        sl = SLOT[s]
        L = "vlda" if sl == "lda" else "vldb"
        o.append(Op(f"L{s}0", sl, f"{L} {{{s}0.x}}, [{p}], #64", defs=[(f"{s}0", "x", 7)], mem=("ld", s), banks=bk("in"),
                    instr=VLDA if sl == "lda" else VLDB))
        o.append(Op(f"L{s}1", sl, f"{L}.2d {{{s}1.x}}, [{p}], {d}", defs=[(f"{s}1", "x", 7)], mem=("ld", s), banks=bk("in"),
                    after=[(f"L{s}0", 1)], instr=VLDA2 if sl == "lda" else VLDB2))
    for h in (0, 1):
        acc = f"s{h}"
        o.append(Op(f"M{h}a", "vec", f"vmul.f {{{acc}.dm}}, {{a{h}.x}}, x{C_ONE}, r6", defs=[(acc, "dm", 6)], uses=[(f"a{h}", "x", 1)], instr=VMUL))
        o.append(Op(f"M{h}b", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{b{h}.x}}, x{C_W1}, r6", defs=[(acc, "dm", 6)],
                    uses=[(acc, "cml", 4), (f"b{h}", "x", 1)], instr=VMAC))
        o.append(Op(f"M{h}c", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{c{h}.x}}, x{C_W2}, r6", defs=[(acc, "dm", 6)],
                    uses=[(acc, "cml", 4), (f"c{h}", "x", 1)], instr=VMAC))
        o.append(Op(f"CV{h}", "st", f"vconv.bf16.fp32 {{y{h}.x}}, {{{acc}.cml}}", defs=[(f"y{h}", "x", 2)], uses=[(acc, "cml", 1)], instr=CONV))
    o.append(Op("ST0", "st", "vst {y0.x}, [p3], #64", uses=[("y0", "x", 1)], mem=("st", "st"), banks=bk("st"), instr=VST))
    o.append(Op("ST1", "st", "vst {y1.x}, [p3], #64", uses=[("y1", "x", 1)], mem=("st", "st"), banks=bk("st"), after=[("ST0", 1)], instr=VST))
    return o


def mk(banks=True):
    return Loop(ops(banks), consts={"x": [C_ONE, C_W1, C_W2]}, classes={"x": X_FREE, "dm": DM_FREE})


def maxdist(II):
    return [("La0", "La1", II - 1), ("Lb0", "Lb1", II - 1), ("Lc0", "Lc1", II - 1), ("ST0", "ST1", II - 1)]


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def build(II=None, times=None, cp=None, max_ii=12, banks=True, U=None):
    if times is not None:
        lp = mk(banks)
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U"))
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 6, (II or max_ii) + 1):
        lp = mk(banks)
        res = cpsched.solve(lp, ii, last_stage=["ST0", "ST1"], max_life=(U or 2) * ii, maxdist=maxdist(ii), **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(U)
            if NIT % lp.U:
                raise Infeasible(f"U {lp.U} does not divide {NIT}")
            side_effects_last(lp)
            check(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0], "U": lp.U}
        return lp
    raise Infeasible("none")


def check(lp, n=NIT):
    # stream pointers: two loads per piece, 128 B per piece; no wrap in the model (dn large)
    st = {s: (lambda i, k: 128 * i + 64 * k, dict(base=0x10000 * (j + 1), dn=0, dj=64, m=64,
                                                   ops={f"L{s}1": dict(dn=1000, dj=64, m=64)})) for j, s in enumerate("abc")}
    st["st"] = (lambda i, k: 128 * i + 64 * k, dict(base=0x40000, dn=0, dj=64, m=64))
    return lp.check(n - n % lp.U, st)


def function(lp, name="n1p"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        # a 2D modifier is PERIODIC (dn steps of dj, then m, again): dn = 7 and the counter starts at 8 - n1 so the one ring-end
        # jump comes after piece n1 - 1 and the next one only past the unit's 8 pieces
        "mova\tdn0, #7", "mova\tdn1, #7", "mova\tdn2, #7", "mova\tdj0, #64", "mova\tdj1, #64", "mova\tdj2, #64",
        "mov\tm0, r5", "mov\tm1, r5", "mov\tm2, r5", "mov\tdc0, r0", "mov\tdc1, r1", "mov\tdc2, r2",
        f"vbcst.16\tx{C_W1}, r3", f"vbcst.16\tx{C_W2}, r4", "movxm\tr7, #16256", f"vbcst.16\tx{C_ONE}, r7",
        "mova\tr6, #828",
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr7, #{NIT // lp.U}", "add.nc\tlc, r7, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--u", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--func", default="n1p")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), banks=not o.nobanks, U=o.u)
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NIT} iterations)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r16/gen_n1p.py\n#define N1P_U {lp.U}\n#define N1P_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
