"""Round 12 (npu-research P29a): the SSM gate sigmoid as a kernel-only modulo-scheduled loop (EW cores, ew8.cc ew_ssm).

Same arithmetic as r9/ssm9.h sig_lut_i (bit-identical): per 32 lanes
    xc = clamp(x, -12, 7.9375); idx = floor(xc * 2^7) (byte offsets, 16 B per segment) + table bases + 96 * 16
    (offset fp32, slope bf16) = LUT(idx); out = bf16(offset + slope * xc)
The EW core has ~144 B of text free, so the loop is KERNEL-ONLY (no prologue): the body runs S - 1 extra trips first, over
negative (garbage) iterations. It walks the vectors BACKWARDS from the last one, so those trips' garbage stores land
just past the end of the buffer (ew_ssm: g_hout, scratch at that point), and the preamble sets every allocatable x
register to the table base so the garbage LUT loads read inside the table.

    extern "C" void sigk(bfloat16 *v_last /*p0*/, int trips /*r0*/, const float *ab /*p1*/, const float *cd /*p2*/)
in place over trips * U vectors ending at v_last (v_last = the last 32-lane vector); writes (S - 1) * U garbage vectors
behind v_last + 64 B and reads SIGK_OVERREAD vectors there first.

    python3 r12/gen_sig.py OUT.s [--ii N] [--cp F.json | --times F.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
from msched import Infeasible, Loop, Op  # noqa: E402

# plain post-modify by -64 (backwards): the 2D forms with dj = m = -64 read past the END on the device (round 12 sigb)
LDA_X = "VLDA_dmx_lda_x_pstm_nrm_imm"
STCONV = "VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm"
C_HI, C_LO, C_BASE = 8, 9, 10
X_FREE = [0, 1, 2, 3, 4, 5, 6, 7, 11]
HI, LO, BIAS_B = 16638, -16064, 96 * 16  # bf16 0x40FE = 7.9375 (SSIG_HI - 1/16), 0xC140 = -12; SSIG_BIAS * 16


def ops():
    from gen_gelu import INSTR
    I = lambda n: INSTR[n if n in INSTR else n.rstrip("0123456789")]  # noqa: E731, E741
    o = [
        Op("L", "lda", "vlda {x.x}, [p0], #-64", defs=[("x", "x", 7)], mem=("ld", "in"), instr=LDA_X),
        Op("MN", "mv", f"vmin_ge.bf16 {{t.x}}, r16, x{C_HI}, {{x.x}}", defs=[("t", "x", 2)], uses=[("x", "x", 1)], instr=I("MN")),
        Op("MX", "mv", f"vmax_lt.bf16 {{xc.x}}, r16, x{C_LO}, {{t.x}}", defs=[("xc", "x", 2)], uses=[("t", "x", 1)], instr=I("MX")),
        Op("F0", "st", "vfloor.s32.bf16 {i0.x}, {xc.wl}, s0", defs=[("i0", "x", 2)], uses=[("xc", "wl", 1)], instr=I("F0")),
        Op("F1", "st", "vfloor.s32.bf16 {i1.x}, {xc.wh}, s0", defs=[("i1", "x", 2)], uses=[("xc", "wh", 1)], instr=I("F1")),
        Op("A0", "mv", f"vadd.32 {{j0.x}}, {{i0.x}}, x{C_BASE}", defs=[("j0", "x", 2)], uses=[("i0", "x", 1)], instr=I("A0")),
        Op("A1", "mv", f"vadd.32 {{j1.x}}, {{i1.x}}, x{C_BASE}", defs=[("j1", "x", 2)], uses=[("i1", "x", 1)], instr=I("A1")),
    ]
    for h in (0, 1):
        j, c0, c1 = f"j{h}", f"c0{h}", f"c1{h}"
        o += [
            Op(f"LA{h}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)], instr=I("LA")),
            Op(f"LB{h}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)], instr=I("LB")),
            Op(f"LC{h}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)], instr=I("LC")),
            Op(f"LD{h}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)], instr=I("LD")),
            Op(f"SO{h}", "mv", f"vshuffle {{sacc.{'bmll' if h == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r3",
               defs=[("sacc", "bmll" if h == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=I("SO")),
            Op(f"SS{h}", "mv", f"vshuffle {{{'sl0' if h == 0 else 'slope'}.x}}, {{{c0}.x}}, {{{c1}.x}}, r2",
               defs=[("sl0" if h == 0 else "slope", "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)], instr=I("SS")),
        ]
    o += [
        Op("MV", "mv", "vmov {slope.wl}, {sl0.wh}", defs=[("slope", "wl", 2)], uses=[("sl0", "wh", 1)], instr=I("MV")),
        Op("S", "vec", "vmac.f {s.dm}, {sacc.dm}, {slope.x}, {xc.x}, r1", defs=[("s", "dm", 6)],
           uses=[("sacc", "cml", 4), ("slope", "x", 1), ("xc", "x", 1)], instr=I("S")),
        Op("ST", "st", "vst.conv.bf16.fp32 {s.cml}, [p3], #-64", uses=[("s", "cml", 1)], mem=("st", "out"), instr=STCONV),
    ]
    return o


KERNEL_ONLY = True
UMAX = 2  # round 16: --umax (32 / U trips must stay integral: U in 1, 2, 4)


def mk():
    lp = Loop(ops(), consts={"x": [C_HI, C_LO, C_BASE]}, classes={"x": X_FREE, "dm": [0, 1, 2, 3, 4]})
    lp.kernel_only = KERNEL_ONLY
    return lp


def build(II=None, times=None, cp=None, max_ii=16):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U"))
        return lp
    sys.path.insert(0, str(HERE))
    import cpsched
    for ii in range(II or 10, (II or max_ii) + 1):
        lp = mk()
        res = cpsched.solve(lp, ii, last_stage=["ST"], max_life=UMAX * ii, **(cp or {}))  # U <= UMAX (32 vectors per call, text)
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            for u in [x for x in (1, 2, 4) if x <= UMAX]:  # round 16: UMAX (default 2) from --umax; U divides 32
                try:
                    lp.allocate(u)
                    break
                except Infeasible:
                    if u == max(x for x in (1, 2, 4) if x <= UMAX):
                        raise
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0]}
        return lp
    raise Infeasible("none")


def check(lp, n=None):
    n = n or 32
    n -= n % lp.U
    back = dict(base=0x8000, dn=7, dj=-64, m=-64)  # backwards from the last vector
    return lp.check(n, {"in": (lambda i: -64 * i, dict(back)), "out": (lambda i: -64 * i, dict(back))})


def function(lp, name="sigk"):
    E = lp.extra_trips()
    neg_in, neg = lp.neg_steps.get("in", 0), lp.neg_steps.get("out", 0)
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        "movxm\tr16, #52428", "mov\tr1, p1", "vbcst.32\tx0, r1", "mov\tr1, p2", "vbcst.32\tx1, r1",
        f"movxm\tr1, #{BIAS_B}", "vbcst.32\tx2, r1", "vsel.32\tx3, x0, x1, r16", "nop", "nop", f"vadd.32\tx{C_BASE}, x3, x2",
        f"movxm\tr1, #{HI}", f"vbcst.16\tx{C_HI}, r1", f"movxm\tr1, #{LO}", f"vbcst.16\tx{C_LO}, r1",
        "mova\tr1, #828", "mova\tr2, #24", "mova\tr3, #5", "mov\ts0, #7",
        # both pointers start as many vectors past v_last as the extra trips access for negative iterations
        "mov\tp3, p0", f"padda\t[p3], #{64 * neg}", f"padda\t[p0], #{64 * neg_in}", "nop",
        # the index registers hold the table base: the extra trips' LUT loads (garbage iterations) stay inside the table
    ] + [f"vmov\tx{r}, x{C_BASE}" for r in sorted({lp.alloc[v, k] for v in ("j0", "j1") for k in range(lp.U)})] + [
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"add.nc\tlc, r0, #{E if lp.kernel_only else 0}", "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)  # the last store is issued inside the body; nothing after the loop reads its results
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--func", default="sigk")
    ap.add_argument("--prologue", action="store_true", help="debug: an explicit prologue instead of kernel-only")
    ap.add_argument("--umax", type=int, default=2)
    ap.add_argument("--time", type=float, default=120)
    o = ap.parse_args()
    global KERNEL_ONLY, UMAX
    UMAX = o.umax
    KERNEL_ONLY = not o.prologue
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time))
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}), U {lp.U}, stages {lp.stages()}, maxlive {lp.maxlive()}, garbage stores "
          f"{lp.neg_steps.get('out', 0)}; check OK ({cyc} cycles, 32 vectors)")
    if o.cp:
        Path(o.cp).write_text(json.dumps({"II": lp.II, "U": lp.U, "t": {x.name: x.t for x in lp.ops}}, indent=1, sort_keys=True) + "\n")
    import asmfit
    render = lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n"  # noqa: E731
    if asmfit.available():  # compact body, last bundle still aligned (round 12: EW text)
        text, added, saved = asmfit.fit(lp, render, o.func, Path(o.out).parent)
        print(f"asmfit: body compacted, {saved} B saved ({added} B of expansions for the alignment)")
    else:
        text = render()
    Path(o.out).write_text(text)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r12/gen_sig.py\n#define SIGK_U {lp.U}\n#define SIGK_GARBAGE {lp.neg_steps.get('out', 0)}\n"
                                             f"#define SIGK_OVERREAD {lp.neg_steps.get('in', 0)}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
