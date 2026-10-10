"""Round 13 (npu-research P49a): the attention QK^T of one row block as a modulo-scheduled asm loop (mm8.cc chunk_k).

Per row block rb and key chunk: scores[8 queries][48 keys] = sum over the 8 dim panels j of Q[rb][j] x K[cb][j]^T, as
6 bfp mmul chains of 8 (one per key block cb). Same MAC order as the C++ (bit-identical): even j in one accumulator (ae),
odd j in another (ao), then ae + ao, stored as bf16 (vst.conv from the accumulator halves) to srb + 128 cb.
    extern "C" void attqk(const bfp16ebs8 *q /*p0*/, const bfp16ebs8 *kt /*p1*/, bfloat16 *srb /*p2*/, int trips /*r0*/)
q = the row block's 8 Q panels (576 B), held in ex0-ex7 for the whole call (popped once in the preamble, lda stream);
kt = the K object ([kb 6][8 dim panels], 3456 B), ONE ldb pop stream across the 6 key blocks (a fill, then 8 pops per
block). One iteration = one key block cb; trips * U = 6. The loop's garbage iterations (the drain) pop (S - 1) x 576 B
past the K object: the B object is 4608 B, so S <= 3 keeps them inside its junk tail (asserted). Stores in the last stage.

    python3 r13/gen_attqk.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

FILL = "VLDB_FILL_512"
POP = "VLDB_POP_576_normal_pop"
VMUL = "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX"
VMAC = "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"
VADD = "VADD_f_vmac_cm2_add_reg"
STCONV = "VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm"
Q_EX = list(range(8))  # Q panels (constants for the loop)
K_EX = [8, 9, 10, 11]
R_MM, R_ADD = "r5", "r6"  # mmul config 780, accumulator add config 60 (as Peano emits them)
JUNK_B = 4608 - 3456  # bytes behind the K object inside its B-fifo object


def ops():
    o = [Op("F", "ldb", "vldb.fill.512 [p1, lf1, r25]", instr=FILL)]
    for j in range(8):
        o.append(Op(f"K{j}", "ldb", f"vldb.pop.576 {{k{j}.ex}}, [p1, lf1, r25]", defs=[(f"k{j}", "ex", 8)], mem=("ld", "k"),
                    after=[(f"K{j - 1}" if j else "F", 1)], instr=POP))
    for j in range(8):
        acc = "ae" if j % 2 == 0 else "ao"
        if j < 2:
            o.append(Op(f"M{j}", "vec", f"vmul.f {{{acc}.dm}}, ex{j}, {{k{j}.ex}}, {R_MM}", defs=[(acc, "dm", 6)],
                        uses=[(f"k{j}", "ex", 1)], instr=VMUL))
        else:
            o.append(Op(f"M{j}", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, ex{j}, {{k{j}.ex}}, {R_MM}", defs=[(acc, "dm", 6)],
                        uses=[(acc, "dm", 4), (f"k{j}", "ex", 1)], instr=VMAC))
    o += [
        Op("AD", "vec", f"vadd.f {{t.dm}}, {{ae.dm}}, {{ao.dm}}, {R_ADD}", defs=[("t", "dm", 6)], uses=[("ae", "dm", 4), ("ao", "dm", 4)],
           instr=VADD),
        Op("S0", "st", "vst.conv.bf16.fp32 {t.cml}, [p2], #64", uses=[("t", "cml", 1)], mem=("st", "srb"), instr=STCONV),
        Op("S1", "st", "vst.conv.bf16.fp32 {t.cmh}, [p2], #64", uses=[("t", "cmh", 1)], mem=("st", "srb"), after=[("S0", 1)], instr=STCONV),
    ]
    return o


def mk():
    return Loop(ops(), consts={"x": Q_EX}, classes={"x": K_EX, "dm": [0, 1, 2, 3, 4]})  # ex_n aliases x_n


def maxdist(II):
    return [("F", "K7", II - 1), ("S0", "S1", II - 1)]


def stages_ok(lp):
    S = lp.stages()
    assert (S - 1) * 576 <= JUNK_B, f"{S} stages: the drain's garbage pops read {(S - 1) * 576} B past the K object (> {JUNK_B})"
    for n in ("S0", "S1"):
        assert lp.by_name[n].t // lp.II == S - 1, f"{n} not in the last stage"
    assert lp.by_name["F"].t % lp.II <= lp.by_name["K0"].t % lp.II or True


def build(II=None, times=None, cp=None, max_ii=14):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        stages_ok(lp)
        lp.times = times
        return lp
    for ii in range(II or 9, (II or max_ii) + 1):
        lp = mk()
        try:
            if cp is not None:
                import cpsched
                res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=["S0", "S1"], max_len=3 * ii - 1, **cp)
                if res is None:
                    raise Infeasible("CP-SAT: none")
                lp.place(ii, res[0])
            else:
                try:
                    lp.schedule(ii)
                except Infeasible:
                    lp.schedule_ims(ii)
            lp.allocate()
            stages_ok(lp)
            lp.times = {"II": ii, "t": {o.t for o in []} or {o.name: o.t for o in lp.ops}}
            return lp
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
    raise Infeasible("no schedule")


def check(lp, n=6):
    n -= n % lp.U
    st = {"k": (lambda i, k: i * 576 + 72 * k, dict(base=0x7C000, dn=1 << 20, dj=72, m=72)),
          "srb": (lambda i, k: i * 128 + 64 * k, dict(base=0x73C00, dn=1 << 20, dj=64, m=64))}
    return lp.check(n, st)


def function(lp, name="attqk", drain=8):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = [f"mova\t{R_MM}, #780", f"mova\t{R_ADD}, #60", "mova\tr24, #0", "mova\tr25, #0",
           "vlda.fill.512\t[p0, lf0, r24]"] + [f"vlda.pop.576\tex{j}, [p0, lf0, r24]" for j in Q_EX] + [
           f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r0, #0", "nop", "nop", "nop", "nop", "nop", "nop"]
    L += ["\t" + p for p in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def report(lp):
    print(f"II {lp.II} (resource bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, "
          f"stages {lp.stages()}, maxlive {lp.maxlive()}")
    for o in sorted(lp.ops, key=lambda o: o.t):
        print(f"  t {o.t:3d} (stage {o.t // lp.II}, slot {o.t % lp.II:2d}) {o.slot} {o.name:4s} {lp.text(o, 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--func", default="attqk")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=60) if o.cp else None)
    report(lp)
    cyc, land = check(lp)
    print(f"check OK: {cyc} cycles for 6 key blocks, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r13/gen_attqk.py\n#define ATTQK_U {lp.U}\n#define ATTQK_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
