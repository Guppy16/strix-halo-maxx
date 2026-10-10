"""Round 15 (research P29d, the z half): the SSM's z = bf16(hb y) panels as one modulo-scheduled loop pushing straight into the
output ring's Z part (EW cores, ew8.cc ew_ssm).

Today: z = BF(mul(hb, y)) per 64 lanes into g_panels by an OutS push (237 cycles per chunk, r15 PROF), then g_panels' 1 KB
(576 B panels + a zero pad) copied into the output ring. Here the same product, rounding and bfp push (bit-identical panels),
into the ring directly; the Z part's 448 B pad is left as it is (only the 576 B panels are ever read: out GEMM A gather).

    extern "C" void zk(const bfloat16 *hb /*p0*/, const bfloat16 *y /*p1*/, bfp16ebs8 *dst /*p2: 64 B aligned*/)
over 8 column blocks (2 x 32 lanes each).

    python3 r15/gen_zk.py OUT.s [--cp F.json | --times F.json]
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
R_MAC = "r1"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
LDA_X = "VLDA_dmx_lda_x_pstm_nrm_imm"
LDB_X = "VLDB_dmx_ldb_x_pstm_nrm_imm"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"


def ops():
    o = []
    for h in (0, 1):
        o += [Op(f"LH{h}", "lda", f"vlda {{h{h}.x}}, [p0], #64", defs=[(f"h{h}", "x", 7)], mem=("ld", "hb"), instr=LDA_X,
                 after=[("LH0", 1)] if h else []),
              Op(f"LY{h}", "ldb", f"vldb {{y{h}.x}}, [p1], #64", defs=[(f"y{h}", "x", 7)], mem=("ld", "y"), instr=LDB_X,
                 after=[("LY0", 1)] if h else [])]
    for h in (0, 1):
        o += [Op(f"M{h}", "vec", f"vmul.f {{a{h}.dm}}, {{h{h}.x}}, {{y{h}.x}}, {R_MAC}", defs=[(f"a{h}", "dm", 6)],
                 uses=[(f"h{h}", "x", 1), (f"y{h}", "x", 1)], instr=VMUL),
              Op(f"C{h}", "st", f"vconv.bf16.fp32 {{z{h}.x}}, {{a{h}.cml}}", defs=[(f"z{h}", "x", 2)], uses=[(f"a{h}", "cml", 1)], instr=CONV)]
    o += [Op("U0", "mv", "vconv.fp32.bf16 {zz.cml}, {z0.x}", defs=[("zz", "cml", 2)], uses=[("z0", "x", 1)], instr=UPS),
          Op("U1", "mv", "vconv.fp32.bf16 {zz.cmh}, {z1.x}", defs=[("zz", "cmh", 2)], uses=[("z1", "x", 1)], instr=UPS),
          Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {zz.dm}, [p2, sf, r26]", uses=[("zz", "dm", 1)], instr=PUSH)]
    return o


def mk():
    return Loop(ops(), consts={}, classes={"x": list(range(12)), "dm": [0, 1, 2, 3, 4]})


def build(II=None, times=None, cp=None, max_ii=10):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate(times.get("U"))
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 3, (II or max_ii) + 1):
        for U in (1, 2, 4):
            lp = mk()
            res = cpsched.solve(lp, ii, last_stage=["PU"], max_life=U * ii, H=10 * ii,
                                maxdist=[("LH0", "LH1", ii - 1), ("LY0", "LY1", ii - 1)], **(cp or {}))
            if res is None:
                continue
            lp.place(ii, res[0])
            try:
                lp.allocate(U)
                check(lp)
            except (Infeasible, AssertionError) as e:
                print(f"II {ii} U {U}: {e}", file=sys.stderr)
                continue
            assert lp.ops[-1].t // ii == lp.stages() - 1
            lp.times = {"II": ii, "t": res[0], "U": U}
            return lp
    raise Infeasible("none")


def check(lp, n=NCB):
    return lp.check(n, {"hb": (lambda i, k: 128 * i + 64 * k, dict(base=0x10000, dn=0, dj=64, m=64)),
                        "y": (lambda i, k: 128 * i + 64 * k, dict(base=0x20000, dn=0, dj=64, m=64))})


def function(lp, name="zk"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [f"mova\t{R_MAC}, #828", "mova\tr26, #0", f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last",
           f"mova\tr0, #{NCB // lp.U}", "add.nc\tlc, r0, #0"]
    if lp.stages() < 2 or (lp.stages() - 1) * lp.II < 6:  # the zero-overhead loop setup distance: full-form nops
        pre += ["nopa;\t\tnopb;\t\tnops;\t\tnopx;\t\tnopm;\t\tnopv"] * 6
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tvst.flush.512.conv\t[p2, sf, r26]", "\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop",
          f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(times=times, cp=dict(time_limit=o.time))
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}), U {lp.U}, stages {lp.stages()}, maxlive {lp.maxlive()}; check OK ({cyc} cycles)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "zk", o.out, o.fit)
    print("wrote", o.out)


if __name__ == "__main__":
    main()
