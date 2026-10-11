"""Round 14 (npu-research T2): the qkv epilogue's V^T objects (mm8.cc acc_epi flag 7) as a modulo-scheduled asm loop.

One call per (object, column pair): 4 dim groups j x 6 key blocks cb = 24 iterations. Per iteration, bit-identical to the C++:
    a0, a1 = the 64 fp32 accumulators of (row block 6 obj + cb, column block 4 pair + j) (8 keys x 8 dims, two halves of 32)
    a_h += bias_h(j) x 1.0  (bf16 MACs: the same product and rounding as aie::mac(a, bias, ones))
    t = transpose8x8(bf16(a0) | bf16(a1))  (vshuffle modes 52 / 53: Peano's aie::transpose(v64, 8, 8))
    push(fp32(t) -> bfp16): 6 panels per dim group into the stash (448 B per group: a 2D flush skips the 16 B pad)
Pieces are 1 KB per row block (MM_PIECES), column block j at +256 B: the load pointer steps +1024 per key block and
256 - 5120 per dim group (2D post-modify).

    extern "C" void vtk(const float *src /*p0*/, const bfloat16 *bias /*p1*/, bfp16ebs8 *dst /*p2*/, int32_t trips /*r0*/)

    python3 r14/gen_vt.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

LDA_BM, LDA_BM2 = "VLDA_dmx_lda_bm_pstm_nrm_imm", "VLDA_2D_dmx_lda_bm"
LDA_X, LDA_X2 = "VLDA_dmx_lda_x_pstm_nrm_imm", "VLDA_2D_dmx_lda_x"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
SHUF_X = "VSHUFFLE_vec_shuffle_x"
PUSH, FLUSH2 = "VST_PUSH_576_CONV_bfp16ebs8_fp32", "VST_FLUSH_512_CONV_2D"
C_ONES = 11  # bf16 1.0 broadcast
CFG = "r1"  # bf16 mac config 828
MODS = {"LB1": (0, 5, 1024 - 192, 256 - 5 * 1024 - 192), "BL1": (1, 5, -64, 64), "FL": (2, 5, 0, 16)}


def ops():
    o = []
    # a0 -> A.cml, a1 -> B.cml (each a 32-lane half, where the bf16 MAC works)
    for n, (acc, q) in enumerate((("A", "bmll"), ("A", "bmlh"), ("B", "bmll"), ("B", "bmlh"))):
        name = ["LA0", "LA1", "LB0", "LB1"][n]
        two = name == "LB1"
        o.append(Op(name, "lda", f"vlda{'.2d' if two else ''} {{{acc}.{q}}}, [p0], {'d0' if two else '#64'}", defs=[(acc, q, 7)],
                    mem=("ld", "src"), instr=LDA_BM2 if two else LDA_BM, after=[(["LA0", "LA1", "LB0"][n - 1], 1)] if n else []))
    o += [Op("BL0", "lda", "vlda {b0.x}, [p1], #64", defs=[("b0", "x", 7)], mem=("ld", "bias"), instr=LDA_X),
          Op("BL1", "lda", "vlda.2d {b1.x}, [p1], d1", defs=[("b1", "x", 7)], mem=("ld", "bias"), instr=LDA_X2, after=[("BL0", 1)])]
    for h, (acc, b) in enumerate((("A", "b0"), ("B", "b1"))):
        o += [Op(f"M{h}", "vec", f"vmac.f {{{acc}2.dm}}, {{{acc}.dm}}, {{{b}.x}}, x{C_ONES}, {CFG}", defs=[(f"{acc}2", "cml", 6)],
                 uses=[(acc, "cml", 4), (b, "x", 1)], instr=VMAC),
              Op(f"C{h}", "st", f"vconv.bf16.fp32 {{t{h}.x}}, {{{acc}2.cml}}", defs=[(f"t{h}", "x", 2)], uses=[(f"{acc}2", "cml", 1)], instr=CONV)]
    o += [Op("S0", "mv", "vshuffle {u0.x}, {t0.x}, {t1.x}, r2", defs=[("u0", "x", 2)], uses=[("t0", "x", 1), ("t1", "x", 1)], instr=SHUF_X),
          Op("S1", "mv", "vshuffle {u1.x}, {t0.x}, {t1.x}, r3", defs=[("u1", "x", 2)], uses=[("t0", "x", 1), ("t1", "x", 1)], instr=SHUF_X),
          Op("U0", "mv", "vconv.fp32.bf16 {w.cml}, {u0.x}", defs=[("w", "cml", 2)], uses=[("u0", "x", 1)], instr=UPS),
          Op("U1", "mv", "vconv.fp32.bf16 {w.cmh}, {u1.x}", defs=[("w", "cmh", 2)], uses=[("u1", "x", 1)], instr=UPS),
          Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH),
          Op("FL", "st", "vst.flush.512.conv.2d [p2, sf, r26, d2]", after=[("PU", 1)], instr=FLUSH2)]
    return o


def mk():
    return Loop(ops(), consts={"x": [C_ONES]}, classes={"x": list(range(11)), "dm": [0, 1, 2, 3, 4]})


def maxdist(II):
    return [("LA0", "LB1", II - 1), ("BL0", "BL1", II - 1), ("PU", "FL", II - 1)]


def last_ops():
    return ["PU", "FL"]


def build(II=None, times=None, cp=None, max_ii=16, stages=2):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 6, max(max_ii, II or 0) + 1):
        lp = mk()
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=last_ops(), max_len=stages * ii - 1, max_life=ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0]}
        print(f"II {ii}: {res[1]}", file=sys.stderr)
        return lp
    raise Infeasible("none")


def streams():
    def mod(name):
        _, dn, dj, m = MODS[name]
        return dict(dn=dn, dj=dj, m=m)

    inc = lambda b: dict(dn=1 << 20, dj=b, m=b)  # noqa: E731
    return {"src": (lambda i, k: (i % 6) * 1024 + (i // 6) * 256 + 64 * k,
                    dict(base=0x78000, dn=1 << 20, dj=0, m=0, ops={"LA0": inc(64), "LA1": inc(64), "LB0": inc(64), "LB1": mod("LB1")})),
            "bias": (lambda i, k: (i // 6) * 128 + 64 * k, dict(base=0x7C000, dn=1 << 20, dj=0, m=0, ops={"BL0": inc(64), "BL1": mod("BL1")}))}


def check(lp, n=24):
    n -= n % lp.U
    return lp.check(n, streams())


def function(lp, name="vtk"):
    mods = []
    for op, (d, dn, dj, m) in MODS.items():
        mods += [f"mova\tdn{d}, #{dn}", f"movxm\tdj{d}, #{dj}", f"movxm\tm{d}, #{m}", f"mova\tdc{d}, #0"]
    pre = ["mov\tr7, r0", f"mova\t{CFG}, #828", "mova\tr2, #52", "mova\tr3, #53", "mova\tr26, #0",
           "movxm\tr4, #16256", f"vbcst.16\tx{C_ONES}, r4"] + mods + [  # bf16 1.0 = 0x3F80
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r7, #0", "nop", "nop", "nop", "nop", "nop", "nop"]
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tnop"] * 4 + ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
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
    ap.add_argument("--max-ii", type=int, default=16)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time), max_ii=o.max_ii, stages=o.stages)
    report(lp)
    cyc, land = check(lp)
    print(f"check OK (24 iterations): {cyc} cycles, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "vtk", o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r14/gen_vt.py\n#define VTK_U {lp.U}\n#define VTK_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
