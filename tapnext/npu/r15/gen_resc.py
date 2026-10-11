"""Round 15 (research rev 10 S1): the lazy-max rescale's O update as a modulo-scheduled asm loop (GEMM cores, mm8.cc rescale_rb).

rescale_rb (minsize C++, ~578 cycles per flagged row block in the k_rb bench) multiplies one row block's O (8 dim blocks x 64
fp32 lanes) and its row sums (64 fp32) by the 8 per-row scales srow (bf16 rows8 of exp(m_old - m_new)):
    hi = bf(a); lo = bf(a - hi 1); a' = hi s + lo s          (per 64 lanes; fp32 stored back)
Here one iteration = 64 lanes as two 32-lane halves (lanes 0-31 with srow[0:32], 32-63 with srow[32:64]); 9 iterations: the
8 dim blocks of O (contiguous, 256 B each), then a 2D-modifier jump to the row sums. The same products and roundings
(vmac a, hi, -1.0 -> vmsc a, hi, 1.0 is exact): bit-identical. exp32 and the srow transpose stay in C++ (srow passed in memory).

    extern "C" void resc(float *o /*p0*/, float *osum /*p1*/, const bfloat16 *srow /*p2: 128 B*/)

    python3 r15/gen_resc.py OUT.s [--ii N] [--cp F.json | --times F.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

NIT = 9
C_ONE, C_S0, C_S1 = 11, 9, 10
X_FREE = list(range(9))
DM_FREE = [0, 1, 2, 3, 4]
R_MAC = "r1"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
LDA_BM = "VLDA_2D_dmx_lda_bm"
ST_BM = "VST_2D_dmx_sts_bm"


def ops():
    o = []
    loads = [("a0", "bmll"), ("a0", "bmlh"), ("a1", "bmll"), ("a1", "bmlh")]
    for k, (v, q) in enumerate(loads):
        o.append(Op(f"L{k}", "lda", f"vlda.2d {{{v}.{q}}}, [p0], d0", defs=[(v, q, 7)], mem=("ld", "o"), banks=("o",),
                    instr=LDA_BM, after=[(f"L{k - 1}", 1)] if k else []))
    for h in (0, 1):
        a = f"a{h}"
        o += [
            Op(f"CH{h}", "st", f"vconv.bf16.fp32 {{hi{h}.x}}, {{{a}.cml}}", defs=[(f"hi{h}", "x", 2)], uses=[(a, "cml", 1)], instr=CONV),
            Op(f"MS{h}", "vec", f"vmsc.f {{d{h}.dm}}, {{{a}.dm}}, {{hi{h}.x}}, x{C_ONE}, {R_MAC}", defs=[(f"d{h}", "dm", 6)],
               uses=[(a, "cml", 4), (f"hi{h}", "x", 1)], instr=VMSC),
            Op(f"CL{h}", "st", f"vconv.bf16.fp32 {{lo{h}.x}}, {{d{h}.cml}}", defs=[(f"lo{h}", "x", 2)], uses=[(f"d{h}", "cml", 1)], instr=CONV),
            Op(f"MU{h}", "vec", f"vmul.f {{p{h}.dm}}, {{hi{h}.x}}, x{C_S0 + h}, {R_MAC}", defs=[(f"p{h}", "dm", 6)],
               uses=[(f"hi{h}", "x", 1)], instr=VMUL),
            Op(f"MA{h}", "vec", f"vmac.f {{r{h}.dm}}, {{p{h}.dm}}, {{lo{h}.x}}, x{C_S0 + h}, {R_MAC}", defs=[(f"r{h}", "dm", 6)],
               uses=[(f"p{h}", "cml", 4), (f"lo{h}", "x", 1)], instr=VMAC),
        ]
    stores = [("r0", "bmll"), ("r0", "bmlh"), ("r1", "bmll"), ("r1", "bmlh")]
    for k, (v, q) in enumerate(stores):
        o.append(Op(f"S{k}", "st", f"vst.2d {{{v}.{q}}}, [p3], d1", uses=[(v, q, 1)], mem=("st", "os"), banks=("o",), instr=ST_BM,
                    after=[(f"S{k - 1}", 1)] if k else []))
    return o


def mk():
    return Loop(ops(), consts={"x": [C_ONE, C_S0, C_S1]}, classes={"x": X_FREE, "dm": DM_FREE})


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def build(II=None, times=None, cp=None, max_ii=14):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate(1)
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 8, (II or max_ii) + 1):
        lp = mk()
        md = [("L0", "L3", ii - 1), ("S0", "S3", ii - 1)]
        res = cpsched.solve(lp, ii, last_stage=["S0", "S1", "S2", "S3"], max_life=ii, H=8 * ii, maxdist=md, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(1)
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        side_effects_last(lp)
        lp.times = {"II": ii, "t": res[0]}
        return lp
    raise Infeasible("none")


JUMP = 0x4000  # checker: the row sums 16 KB after O (any distance: the preamble computes the real one)


def check(lp, n=NIT):
    addr = lambda i, k: (256 * i + 64 * k) if i < 8 else (2048 + JUMP + 64 * k)  # noqa: E731
    m = 2048 + JUMP - 31 * 64
    return lp.check(n, {"o": (addr, dict(base=0x10000, dn=31, dj=64, m=m)), "os": (addr, dict(base=0x10000, dn=31, dj=64, m=m))})


def function(lp, name="resc"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [
        f"vlda\tx{C_S0}, [p2, #0]", f"vlda\tx{C_S1}, [p2, #64]",
        "movxm\tr0, #16256", f"vbcst.16\tx{C_ONE}, r0", "mova\tr1, #828",
        # the jump from O's last 64 B (o + 1984) to the row sums: m = osum - o - 1984 (32nd access of each stream)
        "mov\tr2, p1", "mov\tr3, p0", "sub\tr2, r2, r3", "movxm\tr4, #-1984", "add\tr2, r2, r4", "mov\tm0, r2", "mov\tm1, r2",
        "mova\tdn0, #31", "mova\tdj0, #64", "mova\tdc0, #0", "mova\tdn1, #31", "mova\tdj1, #64", "mova\tdc1, #0", "mov\tp3, p0",
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NIT}", "add.nc\tlc, r0, #0",
    ]  # (no nops: the prologue's (S - 1) II bundles give the zero-overhead loop its setup distance)
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time))
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "resc", o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r15/gen_resc.py\n#define RESC_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
