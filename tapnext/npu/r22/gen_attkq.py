"""Round 22 K2 (research rev 10/17): the attention's chunk 0 needs only the row blocks' running maxima before its exact path
(every row block of chunk 0 flags: m_old is the initial -inf stand-in), so its batched fast pass (r14/gen_attkb.py --s2: QK^T, the
scores, exp(s - m_old), the bfp push, the flags, the max; II 36 per (row block, key block)) is replaced by a QK-only loop with the
same running max: QK^T of one key block, the scores to bf16 (the same vadd.f + vconv as attkb), vmax_lt(slot, s) into the row
block's slot. The max operands and order are attkb's (slot first, s second): the slots come out bit-identical, so the exact path
that follows (k_rb, force) sees the same maxima.

    extern "C" void attkq(const bfp16ebs8 *q /*p0*/, const bfp16ebs8 *kt /*p1*/, bfloat16 *slots /*p2: 6 x 128 B*/,
                          int32_t trips /*r0*/)

    python3 r22/gen_attkq.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r14"))
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
import gen_attkb as G  # noqa: E402
from msched import MEM_R, MEM_W, Infeasible, Loop, Op  # noqa: E402

# modifiers: Q back to the row block's start after each key block, K back to the object's start after each row block, the slot
# pointer advancing 128 B per row block (0 five times, then 64 after the second store's own 64)
MODS = {"PQ": (0, 5, -576, 0), "PK": (1, 5, 0, -3456), "XS1": (2, 5, -64, 64)}
X_FREE = list(range(12))
BANKS = {"q": (1,), "k": (3,), "mx": (0,)}
USE_BANKS = True


def bk(n):
    return BANKS[n] if USE_BANKS else ()


def ops():
    o = [Op("QF", "lda", "vlda.fill.512 [p0, lf0, r24]", banks=bk("q"), instr=G.FILL_A)]
    for j in range(8):
        o.append(Op(f"Q{j}", "lda", f"vlda.pop.576 {{q{j}.ex}}, [p0, lf0, r24]", defs=[(f"q{j}", "ex", 8)], mem=("ld", "q"),
                    after=[(f"Q{j - 1}" if j else "QF", 1)], banks=bk("q"), instr=G.POP_A))
    o.append(Op("PQ", "lda", "padda.2d [p0], d0", mem=("pa", "q"), after=[("Q7", 1)], instr=G.PADDA2))
    o.append(Op("F", "ldb", "vldb.fill.512 [p1, lf1, r25]", banks=bk("k"), instr=G.FILL_B))
    for j in range(8):
        o.append(Op(f"K{j}", "ldb", f"vldb.pop.576 {{k{j}.ex}}, [p1, lf1, r25]", defs=[(f"k{j}", "ex", 8)], mem=("ld", "k"),
                    after=[(f"K{j - 1}" if j else "F", 1)], banks=bk("k"), instr=G.POP_B))
    o.append(Op("PK", "st", "padds.2d [p1], d1", mem=("pa", "k"), after=[("K7", 1)], instr=G.PADDS2))
    for j in range(8):
        acc = "ae" if j % 2 == 0 else "ao"
        if j < 2:
            o.append(Op(f"M{j}", "vec", f"vmul.f {{{acc}.dm}}, {{q{j}.ex}}, {{k{j}.ex}}, {G.R_MM}", defs=[(acc, "dm", 6)],
                        uses=[(f"q{j}", "ex", 1), (f"k{j}", "ex", 1)], instr=G.BMUL))
        else:
            o.append(Op(f"M{j}", "vec", f"vmac.f {{{acc}.dm}}, {{{acc}.dm}}, {{q{j}.ex}}, {{k{j}.ex}}, {G.R_MM}", defs=[(acc, "dm", 6)],
                        uses=[(acc, "dm", 4), (f"q{j}", "ex", 1), (f"k{j}", "ex", 1)], instr=G.BMAC))
    o.append(Op("AD", "vec", f"vadd.f {{t.dm}}, {{ae.dm}}, {{ao.dm}}, {G.R_ADD}", defs=[("t", "dm", 6)],
                uses=[("ae", "dm", 4), ("ao", "dm", 4)], instr=G.VADDF))
    for h in (0, 1):
        o.append(Op(f"C{h}", "st", f"vconv.bf16.fp32 {{s{h}.x}}, {{t.{'cml' if h == 0 else 'cmh'}}}", defs=[(f"s{h}", "x", 2)],
                    uses=[("t", "cml" if h == 0 else "cmh", 1)], instr=G.CONV))
        # the slot loads go on ldb (lda carries Q's 10 stream ops): vldb x by offset
        o += [Op(f"XL{h}", "ldb", f"vldb {{mo{h}.x}}, [p2, #{64 * h}]", defs=[(f"mo{h}", "x", 7)], mem=("ld", "mx"), banks=bk("mx"),
                 instr=G.LDB_X),
              Op(f"XM{h}", "mv", f"vmax_lt.bf16 {{mn{h}.x}}, r16, {{mo{h}.x}}, {{s{h}.x}}", defs=[(f"mn{h}", "x", 2)],
                 uses=[(f"mo{h}", "x", 1), (f"s{h}", "x", 1)], instr=G.MX)]
    o += [Op("XS0", "st", "vst {mn0.x}, [p2], #64", uses=[("mn0", "x", 1)], mem=("st", "mx"), banks=bk("mx"), instr=G.ST_X,
             after=[("XL0", 1), ("XL1", 1)]),
          Op("XS1", "st", "vst.2d {mn1.x}, [p2], d2", uses=[("mn1", "x", 1)], mem=("st", "mx"), banks=bk("mx"), instr=G.ST_X2,
             after=[("XS0", 1)])]
    return o


def mk():
    return Loop(ops(), consts={}, classes={"x": X_FREE, "dm": G.DM_FREE})


def maxdist(II):
    return [("QF", "PQ", II - 3), ("F", "PK", II - 3), ("QF", "Q7", II - 1), ("F", "K7", II - 1),
            ("XL0", "XS1", II - 4), ("XL1", "XS1", II - 4)]


LAST = ["XS0", "XS1"]


def streams():
    inc = lambda b: dict(dn=1 << 20, dj=b, m=b)  # noqa: E731
    mod = lambda n: dict(dn=MODS[n][1], dj=MODS[n][2], m=MODS[n][3])  # noqa: E731
    return {
        "q": (lambda i, k: (i // 6) * 576 + 72 * k, dict(base=0x74000, dn=1 << 20, dj=72, m=72, ops={"PQ": mod("PQ")})),
        "k": (lambda i, k: (i % 6) * 576 + 72 * k, dict(base=0x7D000, dn=1 << 20, dj=72, m=72, ops={"PK": mod("PK")})),
        "mx": (lambda i, k: (i // 6) * 128 + (64 if k == 3 else 0), dict(base=0x70100, dn=1 << 20, dj=0, m=0, ops={
            "XL0": dict(dn=1 << 20, dj=0, m=0), "XL1": dict(dn=1 << 20, dj=0, m=0), "XS0": inc(64), "XS1": mod("XS1")})),
    }


def check(lp, n=36):
    n -= n % lp.U
    t = lp.by_name
    for h in (0, 1):
        assert t[f"XL{h}"].t + lp.II + MEM_R >= t[f"XS{h}"].t + MEM_W, "slot load before the previous store landed"
        assert t[f"XL{h}"].t + lp.II >= t["XS1"].t + 3, "slot load before the pointer step"
    return lp.check(n, streams())


def stages_ok(lp):
    S = lp.stages()
    for n in LAST:
        assert lp.by_name[n].t // lp.II == S - 1, f"{n} not in the last stage"


def build(II=None, times=None, cp=None, max_ii=30, stages=3, life=2):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        stages_ok(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 10, max(max_ii, II or 0) + 1):
        lp = mk()
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=LAST, max_len=stages * ii - 1, max_life=life * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
            stages_ok(lp)
            for n in (36, 6):
                check(lp, n)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0]}
        print(f"II {ii}: {res[1]}", file=sys.stderr)
        return lp
    raise Infeasible("none")


def function(lp, name="attkq"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    mods = []
    for op, (d, dn, dj, m) in MODS.items():
        mods += [f"mova\tdn{d}, #{dn}", f"movxm\tdj{d}, #{dj}", f"movxm\tm{d}, #{m}", f"mova\tdc{d}, #0"]
    pre = ["mov\tr7, r0", f"mova\t{G.R_MM}, #780", f"mova\t{G.R_ADD}, #60", "mova\tr24, #0", "mova\tr25, #0",
           "movxm\tr16, #52428"] + mods + [
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r7, #0", "nop", "nop", "nop", "nop", "nop", "nop"]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tnop"] * 6 + ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:",
                          f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--max-ii", type=int, default=30)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--life", type=int, default=2, help="max register life in IIs (1 forces U = 1: half the body text)")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    o = ap.parse_args()
    global USE_BANKS
    USE_BANKS = not o.nobanks
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time, workers=o.workers), max_ii=o.max_ii, stages=o.stages, life=o.life)
    G.report(lp)
    for n in (36, 6):
        cyc, land = check(lp, n)
        print(f"check OK ({n} iterations): {cyc} cycles, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "attkq", o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r22/gen_attkq.py\n#define ATTKQ_U {lp.U}\n#define ATTKQ_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
