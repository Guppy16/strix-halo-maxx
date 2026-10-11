"""Pass 20 (FSX): gate_a's sigmoid folded into ssq's loop (r16/gen_ssq.py), so sigk runs gate_x only (16 vectors).

Per 32 lanes: ga = vlda [p0] (the pre-activation, in place of ssq's gate_a load), then r12/gen_sig.py's ops op for op
(clamp, floor, the (ab, cd) LUT gathers, offset + slope xc) with its three resident constants reloaded per use from a second
table pointer (p5 = sigmoid rows: hi, lo, base), and the result converted into the x register ssq's chain reads as gate_a
(vconv on the st slot instead of sigk's vst.conv). Everything after is ssq unchanged (a -> [p4] over ga, mult -> [p2]).
Scalars added to ssq's: r5 = 5 (sigk's SO shuffle mode); SS uses ssq's r3 = 24; s0 = 7 is shared.

The question (E16): is there a schedule whose x pressure fits x0-x10 (x11 = bf16 1.0) at some II, and at what II?
    uv run --with ortools --with numpy python r25/gen_fsx.py --ii-from 30 --ii-to 40 --u 2 --time 600 --workers 4 --json r25/fsx.json
"""

import argparse
import json
import sys
from pathlib import Path

NT = Path.home() / "git/app/.scratch/npu-tapnext"
for d in ("r11", "r12", "r15", "r16"):
    sys.path.insert(0, str(NT / d))
import gen_ssq as Q  # noqa: E402
import gen_ssx as X  # noqa: E402
from gen_gelu import INSTR  # noqa: E402
from msched import Infeasible, Loop, Op  # noqa: E402

SIG_ROWS = dict(shi=0, slo=64, sbase=128)  # [p5, #off]
SIG_BANK = (1, 2)  # in situ: ssig_ab in .data (bank 1), ssig_cd .aie.bank2


def I(n):  # noqa: E743
    return INSTR[n if n in INSTR else n.rstrip("0123456789")]


def sig_ops(src, dst, bk):
    """sigk's chain on vreg `src` (bf16 pre-activation) -> vreg `dst` (bf16 sigmoid) in an x register"""
    o = []
    for nm in SIG_ROWS:
        o.append(Op(f"K{nm}", "ldb", f"vldb {{k{nm}.x}}, [p5, #{SIG_ROWS[nm]}]", defs=[(f"k{nm}", "x", 7)], banks=bk("tab"),
                    instr=X.LDB_I))
    o += [
        Op("gMN", "mv", f"vmin_ge.bf16 {{gt.x}}, r16, {{kshi.x}}, {{{src}.x}}", defs=[("gt", "x", 2)],
           uses=[(src, "x", 1), ("kshi", "x", 1)], instr=I("MN")),
        Op("gMX", "mv", "vmax_lt.bf16 {gxc.x}, r16, {kslo.x}, {gt.x}", defs=[("gxc", "x", 2)],
           uses=[("gt", "x", 1), ("kslo", "x", 1)], instr=I("MX")),
        Op("gF0", "st", "vfloor.s32.bf16 {gi0.x}, {gxc.wl}, s0", defs=[("gi0", "x", 2)], uses=[("gxc", "wl", 1)], instr=I("F0")),
        Op("gF1", "st", "vfloor.s32.bf16 {gi1.x}, {gxc.wh}, s0", defs=[("gi1", "x", 2)], uses=[("gxc", "wh", 1)], instr=I("F1")),
        Op("gA0", "mv", "vadd.32 {gj0.x}, {gi0.x}, {ksbase.x}", defs=[("gj0", "x", 2)], uses=[("gi0", "x", 1), ("ksbase", "x", 1)],
           instr=I("A0")),
        Op("gA1", "mv", "vadd.32 {gj1.x}, {gi1.x}, {ksbase.x}", defs=[("gj1", "x", 2)], uses=[("gi1", "x", 1), ("ksbase", "x", 1)],
           instr=I("A1")),
    ]
    for h in (0, 1):
        j, c0, c1 = f"gj{h}", f"gc0{h}", f"gc1{h}"
        o += [
            Op(f"gLA{h}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)],
               banks=bk("sig"), instr=I("LA")),
            Op(f"gLB{h}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)],
               banks=bk("sig"), instr=I("LB")),
            Op(f"gLC{h}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)],
               banks=bk("sig"), instr=I("LC")),
            Op(f"gLD{h}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)],
               banks=bk("sig"), instr=I("LD")),
            Op(f"gSO{h}", "mv", f"vshuffle {{gsacc.{'bmll' if h == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r5",
               defs=[("gsacc", "bmll" if h == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=I("SO")),
            Op(f"gSS{h}", "mv", f"vshuffle {{{'gsl0' if h == 0 else 'gslope'}.x}}, {{{c0}.x}}, {{{c1}.x}}, r3",
               defs=[("gsl0" if h == 0 else "gslope", "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=I("SS")),
        ]
    o += [
        Op("gMV", "mv", "vmov {gslope.wl}, {gsl0.wh}", defs=[("gslope", "wl", 2)], uses=[("gsl0", "wh", 1)], instr=I("MV")),
        Op("gS", "vec", "vmac.f {gs.dm}, {gsacc.dm}, {gslope.x}, {gxc.x}, r1", defs=[("gs", "dm", 6)],
           uses=[("gsacc", "cml", 4), ("gslope", "x", 1), ("gxc", "x", 1)], instr=I("S")),
        Op("gCV", "st", f"vconv.bf16.fp32 {{{dst}.x}}, {{gs.cml}}", defs=[(dst, "x", 2)], uses=[("gs", "cml", 1)], instr=X.CONV),
    ]
    return o


def ops(rel_slot=None, banks=True):
    bank = dict(Q.BANK, sig=SIG_BANK)
    bk = (lambda n: bank[n]) if banks else (lambda n: ())  # noqa: E731
    o = Q.ops(rel_slot, banks)
    lx = next(x for x in o if x.name == "LX")
    lx.fmt = lx.fmt.replace("{g.x}", "{ga.x}")
    lx.defs = [("ga", "x", 7)]
    return X.topo(o + sig_ops("ga", "g", bk))


def mk(rel_slot=None, banks=True):
    return Loop(ops(rel_slot, banks), consts={"x": [X.C_ONE]}, classes={"x": X.X_FREE, "dm": X.DM_FREE})


def function(lp, name="fsx"):
    """ssq's function with two preamble additions: r5 = 5 (sigk's SO mode) and p5 = the sigmoid rows (ssq_tab rows 24-26, i.e.
    p3 + 1024: shi, slo, sbase; the C++ fills them once, sbase as sigk's preamble builds it: (ssig_ab, ssig_cd) lane bases
    interleaved by vsel 0xCCCC + 96 x 16)"""
    L = Q.function(lp, name)
    i = L.index("\tmov\tp4, p0")
    return L[: i + 1] + ["\tmova\tr5, #5", "\tmov\tp5, p3", "\tpadda\t[p5], #1024"] + L[i + 1 :]


def emit(times, out, fit=False):
    import asmfit
    lp = mk(times.get("rel"))
    lp.place(times["II"], times["t"])
    lp.allocate(times.get("U", 2))
    X.side_effects_last(lp)
    cyc, _ = X.check(lp)
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "fsx", out, fit)
    Path(out).with_suffix(".h").write_text(f"// generated by npu-research r25/gen_fsx.py\n#define FSX_U {lp.U}\n#define FSX_II {lp.II}\n"
                                           f"#define FSX_SIG_ROW0 24\n")
    print(f"emitted {out}: II {lp.II}, U {lp.U}, stages {lp.stages()}, check {cyc} cycles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ii-from", type=int, default=None)
    ap.add_argument("--ii-to", type=int, default=40)
    ap.add_argument("--u", type=int, default=2)
    ap.add_argument("--time", type=float, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rel", default="ff,hi", help="ssq constants reloaded on lda (ssq_u2_30: ff,hi)")
    ap.add_argument("--hmul", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--mii", action="store_true", help="print the resource bound only")
    ap.add_argument("--emit", help="OUT.s from --times")
    ap.add_argument("--times")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    if o.emit:
        return emit(json.loads(Path(o.times).read_text()), o.emit, o.fit)
    rel_slot = {k: "a" for k in o.rel.split(",") if k}
    lp = mk(rel_slot)
    det = lp.res_mii(detail=True)
    print(f"ops {len(lp.ops)}; ResMII {lp.res_mii()}: {dict(list(det.items())[:8])}", flush=True)
    if o.mii:
        return
    import cpsched
    for ii in range(o.ii_from or lp.res_mii(), o.ii_to + 1):
        lp = mk(rel_slot)
        res = cpsched.solve(lp, ii, last_stage=["STA", "STM"], max_life=o.u * ii, H=o.hmul * ii, time_limit=o.time,
                            workers=o.workers)
        if res is None:
            print(f"II {ii}: CP-SAT found none in {o.time:.0f} s", flush=True)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", flush=True)
        lp.place(ii, res[0])
        try:
            lp.allocate(o.u)
            X.side_effects_last(lp)
            cyc, _ = X.check(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", flush=True)
            continue
        print(f"II {ii}: allocated U {lp.U}, stages {lp.stages()}, maxlive {lp.maxlive()}; check OK ({cyc} cycles, 16 vectors)")
        if o.json:
            Path(o.json).write_text(json.dumps({"II": ii, "t": res[0], "rel": rel_slot, "U": lp.U}, indent=1, sort_keys=True) + "\n")
        return
    print("none up to II", o.ii_to)


if __name__ == "__main__":
    main()
