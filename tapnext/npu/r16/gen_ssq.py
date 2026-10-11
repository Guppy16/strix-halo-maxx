"""Round 16: ssx + sqk as ONE modulo-scheduled loop (EW cores, ew8.cc ew_ssm; flag -DSSQ): log_a, a = exp(log_a),
om = 1 - a^2 AND mult = sqrt(om) per vector, so the sqrt's 13 mv ops ride in the ssx loop's free mv slots (ssx is vec-bound
at II 20 with 7 mv ops) instead of a second loop with its own preamble / prologue.

Ops: r15/gen_ssx.py's chain op for op (bit-identical), its om store replaced by vconv om -> x + r16/gen_sqk.py's sqrt ops
(bit-identical: the same tables, SQK_TAB in ssqrt_ab / ssqrt_cd) + the mult store. Every constant except bf16 1.0 (x11) is
reloaded per use from the ssx table, which grows by four rows: 0x00FF, 0xFF00, bf16 2^60 and the (ab, cd) lane bases (16
int32, written by the preamble from the table pointers). Scalars: r1 828, r2 2 (ssx), r3 24, r4 10; s0 7, s1 0 (ssx and the
sqrt's exponent ups), s2 2 (index ups), s3 1 (exponent srs).

    extern "C" void ssq(bfloat16 *g /*p0: gate_a in, a out*/, const B32 *tsp /*p1*/, bfloat16 *mult /*p2*/,
                         const bfloat16 *tab /*p3: table + 512*/, const int16_t *ab /*p4*/, const int16_t *cd /*p5*/)

    python3 r16/gen_ssq.py OUT.s [--ii N] [--u U] [--cp F.json | --times F.json] [--rel a,b] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
sys.path.insert(0, str(HERE.parent / "r15"))
sys.path.insert(0, str(HERE))
import gen_sqk  # noqa: E402
import gen_ssx as X  # noqa: E402
from msched import Infeasible, Loop, Op  # noqa: E402

NV = 16
CONSTS = dict(X.CONSTS, ff=0x00FF, hi=0xFF00, b60=0x5D80, base=0)  # base: written by the preamble
ROW = {k: i for i, k in enumerate(CONSTS)}
OFF = {k: 64 * i - X.TAB for k, i in ROW.items()}
assert max(OFF.values()) <= 448
BANK = dict(X.BANK, mult=(1,), lut=(0, 3))
SQ_REL = ("ff", "hi", "base", "b60")


def ops(rel_slot=None, banks=True):
    rs = rel_slot or {}
    bk = (lambda n: BANK[n]) if banks else (lambda n: ())  # noqa: E731
    X.OFF.update(OFF)  # the ssx rows keep their offsets (CONSTS extends X.CONSTS in order)
    o = [op for op in X.ops(rs, banks) if op.name != "STO"]
    o.append(Op("C13", "st", "vconv.bf16.fp32 {omx.x}, {oma.cml}", defs=[("omx", "x", 2)], uses=[("oma", "cml", 1)], instr=X.CONV))
    k = {}
    for nm in SQ_REL:
        s = rs.get(nm, "b")
        o.append(Op(f"K{nm}", "lda" if s == "a" else "ldb", f"vld{s} {{k{nm}.x}}, [p3, #{OFF[nm]}]", defs=[(f"k{nm}", "x", 7)],
                    banks=bk("tab"), instr=X.LDA_I if s == "a" else X.LDB_I))
        k[nm] = f"k{nm}"
    sq_bk = lambda n: bk("lut") if n == "tab" else bk(n)  # noqa: E731
    gen_sqk.sqrt_ops(o, "omx", "mult", sq_bk, tag="q", k=k, sh=("s2", "s1", "s3"), rm=("r3", "r4"))
    o.append(Op("STM", "st", "vst {mult.x}, [p2], #64", uses=[("mult", "x", 1)], mem=("st", "om"), banks=bk("mult"), instr=X.ST_X))
    return X.topo(o)


def mk(rel_slot=None, banks=True):
    return Loop(ops(rel_slot, banks), consts={"x": [X.C_ONE]}, classes={"x": X.X_FREE, "dm": X.DM_FREE})


def build(II=None, times=None, cp=None, max_ii=26, rel_slot=None, banks=True, U=1):
    if times is not None:
        lp = mk(times.get("rel", rel_slot), banks)
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U", U))
        X.side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 21, (II or max_ii) + 1):
        lp = mk(rel_slot, banks)
        res = cpsched.solve(lp, ii, last_stage=["STA", "STM"], max_life=U * ii, H=9 * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(U)
            X.side_effects_last(lp)
            X.check(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0], "rel": rel_slot, "U": lp.U}
        return lp
    raise Infeasible("none")


def function(lp, name="ssq"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        # the lane-base row: (ab, cd) addresses interleaved as Peano's load_lut_2x_int8 does (vsel mask 0xCCCC)
        "movxm\tr16, #52428", "mov\tr0, p4", "vbcst.32\tx0, r0", "mov\tr0, p5", "vbcst.32\tx1, r0", "nop", "nop",
        "vsel.32\tx2, x0, x1, r16", "nop", "nop", f"vst\tx2, [p3, #{OFF['base']}]",
        f"movxm\tr0, #{X.CONSTS['one']}", f"vbcst.16\tx{X.C_ONE}, r0",
        "mova\tr1, #828", "mova\tr2, #2", "mova\tr3, #24", "mova\tr4, #10", "mov\ts0, #7", "mov\ts1, #0", "mov\ts2, #2", "mov\ts3, #1",
        "mov\tcrupsmode, #0", "mov\tcrsrsmode, #0",
        "mov\tp4, p0",  # a: stored over g
        "mova\tdn1, #1", "mova\tdj1, #0", "mova\tm1, #64", "mova\tdc1, #0",  # sp: each row twice
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NV // lp.U}", "add.nc\tlc, r0, #0",
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
    ap.add_argument("--u", type=int, default=1)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=300)
    ap.add_argument("--rel", default="", help="constants reloaded on lda, e.g. m8,c1 (default all on ldb)")
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--func", default="ssq")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    rel_slot = {k: "a" for k in o.rel.split(",") if k}
    lp = build(o.ii, times, cp=dict(time_limit=o.time), rel_slot=rel_slot, banks=not o.nobanks, U=o.u)
    cyc, land = X.check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:8])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NV} vectors)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    rows = ", ".join(f"SSX_ROW({b & 0xFFFF:#06x})" for b in CONSTS.values())
    Path(o.out).with_suffix(".h").write_text(f"// generated by r16/gen_ssq.py\n#define SSQ_U {lp.U}\n#define SSQ_II {lp.II}\n#define SSQ_TAB_OFF {X.TAB}\n"
                                             f"#define SSQ_NROWS {len(CONSTS)}\n#define SSQ_ROWS {rows}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
