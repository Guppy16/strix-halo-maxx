"""Round 14 (npu-research P50): the attention V phase (O += P V for 8 dims of one row block, two dim blocks per iteration) as
one modulo-scheduled asm loop over all (row block, dim pair) iterations of a (tile, chunk): 6 x 4 = 24 iterations per call.

Per iteration (row block rb = i / 4, dim pair k = i % 4), bit-identical to mm8.cc chunk_v's pair loop:
    a_j = O[rb][2k + j] + P0 V_j0 + P2 V_j2 + P4 V_j4;  b_j = P1 V_j1 + P3 V_j3 + P5 V_j5;  O[rb][2k + j] = a_j + b_j  (j = 0, 1)
- P (the row block's 6 bfp panels, 448 B slots) is popped once per iteration on load unit A and held for both dim blocks; its
  stream returns to the slot's start after each pair (padds.2d: -448 three times, then 0 = the next row block);
- V^T (8 dim groups of 6 panels, 448 B apart) is popped on unit B: group 2k, a refill, group 2k + 1; after the 4th pair the
  stream returns to the object's start (padds.2d: 0 three times, then -3584);
- O: 8 x 64 B loads straight into the accumulators' quarters (vlda bmXX) and 8 stores of the sums, both streams linear
  (512 B per iteration: the O rows of a tile are [rb 6][8 dims][64 fp32] contiguous). The row sums (P times the ones panels)
  stay in C++.

    extern "C" void attv(const bfp16ebs8 *p /*p0*/, const bfp16ebs8 *v /*p1*/, float *o /*p2*/, int32_t trips /*r0*/)

    python3 r14/gen_attv.py OUT.s [--ii N] [--cp F.json | --times F.json] [--fit]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import PARTS, Infeasible, Loop, Op  # noqa: E402

PARTS.setdefault("bmhl", ("dm", (2,)))  # round 14: the accumulator's upper quarters (O loads / stores per 64 B)
PARTS.setdefault("bmhh", ("dm", (3,)))
FILL_A, FILL_B = "VLDA_FILL_512", "VLDB_FILL_512"
POP_A, POP_B = "VLDA_POP_576_normal_pop", "VLDB_POP_576_normal_pop"
BMUL = "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX"
BMAC = "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"
VADDF = "VADD_f_vmac_cm2_add_reg"
LDA_BM = "VLDA_dmx_lda_bm_pstm_nrm_imm"
ST_BM = "VST_dmx_sts_bm_pstm_nrm_imm"
PADDS2 = "PADDS_2D"
R_MM, R_ADD = "r5", "r6"
QUARTERS = ("bmll", "bmlh", "bmhl", "bmhh")
# walker steps (modifier set, dn, dj, m): P +0 three times then +448; V^T after the second group of a pair +448, after the 4th -3136
MODS = {"PW": (0, 3, 0, 448), "VW1": (1, 3, 448, 448 - 4 * 896)}
# --sums (attvs): the row sums, O += P x ones per row block: one dim block per iteration, P +448 every iteration, the ones restart
SUMS = False
DIMS = (0, 1)
# in-situ banks (full12r14cz core 0_2): P buffers in acc_hi (bank 1), V objects bank 3, O tile 0 in the C object (banks 2-3),
# tile 1 at acc_hi's start (banks 0-1): O is left untagged (it moves between tiles)
BANKS = {"p": (1,), "v": (3,)}
USE_BANKS = True
# round 18 (V2T): one call per chunk for BOTH tiles: the O rows of tile 1 (acc_hi) are not contiguous with tile 0's (the C object),
# so the last O load and the last O store of each iteration post-modify by a 2D modifier (d2 / d3: +64, after every 24th
# iteration + 64 + jump, jump = o1 - (o0 + 12,288) from r1). P buffers (erb) and the V walker continue across the tiles unchanged.
TWO = False
NIT_T = 24


def bk(name):
    return BANKS[name] if USE_BANKS else ()


def ops():
    # Each fill starts from an explicit pointer: walker p4 (P slot of the row block: +0 three times, then +448) and walker p5 (V^T
    # dim groups: +448, then +448 three times / -3136 after the 4th pair) are copied into the FIFO pointers p0 / p1, and the FIFO
    # positions reset (round 14: continuing a FIFO pointer after 6 pops of 72 B = 432 B, not a multiple of 64, gave wrong data
    # with +448 and with +512 assumed; the 576 B restarts of round 13 end on a line)
    o = [Op("MP", "mv", "mov p0, p4"),
         Op("PW", "st", "padds.2d [p4], d0", after=[("MP", 1)], instr=PADDS2),
         Op("RP", "mv", "mov r24, #0"),
         Op("PF", "lda", "vlda.fill.512 [p0, lf0, r24]", banks=bk("p"), after=[("RP", 1), ("MP", 3)], instr=FILL_A)]
    for j in range(6):
        o.append(Op(f"P{j}", "lda", f"vlda.pop.576 {{p{j}.ex}}, [p0, lf0, r24]", defs=[(f"p{j}", "ex", 8)],
                    after=[(f"P{j - 1}" if j else "PF", 1)], banks=bk("p"), instr=POP_A))
    o += [Op("MV0", "mv", "mov p1, p5")] + ([] if SUMS else [
          Op("VW0", "st", "padds [p5], #448", after=[("MV0", 1)], instr="PADDS_pstm_nrm_imm")]) + [
          Op("RV0", "mv", "mov r25, #0"),
          Op("VF0", "ldb", "vldb.fill.512 [p1, lf1, r25]", banks=bk("v"), after=[("RV0", 1), ("MV0", 3)], instr=FILL_B)]
    for j in range(6 * len(DIMS)):
        prev = "VF1" if j == 6 else (f"V{j - 1}" if j else "VF0")
        o.append(Op(f"V{j}", "ldb", f"vldb.pop.576 {{v{j}.ex}}, [p1, lf1, r25]", defs=[(f"v{j}", "ex", 8)],
                    after=[(prev, 1)], banks=bk("v"), instr=POP_B))
        if j == 5 and not SUMS:
            o += [Op("MV1", "mv", "mov p1, p5", after=[("V5", 1), ("VW0", 2)]),
                  Op("VW1", "st", "padds.2d [p5], d1", after=[("MV1", 1)], instr=PADDS2),
                  Op("RV1", "mv", "mov r25, #0", after=[("V5", 1)]),
                  Op("VF1", "ldb", "vldb.fill.512 [p1, lf1, r25]", after=[("MV1", 3), ("RV1", 1)], banks=bk("v"), instr=FILL_B)]
    for d in DIMS:  # the dim blocks
        a, b = f"a{d}", f"b{d}"
        for q in QUARTERS:
            if TWO and d == DIMS[-1] and q == QUARTERS[-1]:
                o.append(Op(f"L{d}{q}", "lda", f"vlda.2d {{{a}.{q}}}, [p2], d2", defs=[(a, q, 7)], mem=("ld", "o"), instr="VLDA_2D_dmx_lda_bm"))
                continue
            o.append(Op(f"L{d}{q}", "lda", f"vlda {{{a}.{q}}}, [p2], #64", defs=[(a, q, 7)], mem=("ld", "o"), instr=LDA_BM))
        for k in range(6):
            v = f"v{6 * d + k}"
            if k % 2 == 0:  # the loaded O accumulates the even key blocks
                o.append(Op(f"M{d}{k}", "vec", f"vmac.f {{{a}.dm}}, {{{a}.dm}}, {{p{k}.ex}}, {{{v}.ex}}, {R_MM}", defs=[(a, "dm", 6)],
                            uses=[(a, "dm", 4), (f"p{k}", "ex", 1), (v, "ex", 1)], instr=BMAC))
            elif k == 1:
                o.append(Op(f"M{d}{k}", "vec", f"vmul.f {{{b}.dm}}, {{p{k}.ex}}, {{{v}.ex}}, {R_MM}", defs=[(b, "dm", 6)],
                            uses=[(f"p{k}", "ex", 1), (v, "ex", 1)], instr=BMUL))
            else:
                o.append(Op(f"M{d}{k}", "vec", f"vmac.f {{{b}.dm}}, {{{b}.dm}}, {{p{k}.ex}}, {{{v}.ex}}, {R_MM}", defs=[(b, "dm", 6)],
                            uses=[(b, "dm", 4), (f"p{k}", "ex", 1), (v, "ex", 1)], instr=BMAC))
        o.append(Op(f"AD{d}", "vec", f"vadd.f {{c{d}.dm}}, {{{a}.dm}}, {{{b}.dm}}, {R_ADD}", defs=[(f"c{d}", "dm", 6)],
                    uses=[(a, "dm", 4), (b, "dm", 4)], instr=VADDF))
        for q in QUARTERS:
            if TWO and d == DIMS[-1] and q == QUARTERS[-1]:
                o.append(Op(f"S{d}{q}", "st", f"vst.2d {{c{d}.{q}}}, [p3], d3", uses=[(f"c{d}", q, 1)], mem=("st", "os"), instr="VST_2D_dmx_sts_bm"))
                continue
            o.append(Op(f"S{d}{q}", "st", f"vst {{c{d}.{q}}}, [p3], #64", uses=[(f"c{d}", q, 1)], mem=("st", "os"), instr=ST_BM))
    return o


def mk():
    return Loop(ops(), classes={"x": list(range(12)), "dm": [0, 1, 2, 3, 4]})


def maxdist(II):
    """the streams of one iteration inside one II, in order: the next iteration's pointer copy / position reset after this
    iteration's last pop of that FIFO"""
    vl = f"V{6 * len(DIMS) - 1}"
    md = [("MP", "P5", II - 1), ("RP", "P5", II - 2), ("MV0", vl, II - 1), ("RV0", vl, II - 2),
          ("MP", "PW", II - 2)]  # the next copy reads the walker after this step landed
    if not SUMS:
        md.append(("MV0", "VW1", II - 2))
    names = [f"L{d}{q}" for d in DIMS for q in QUARTERS]
    md += [(names[0], names[-1], II - 1)] + [(x, y, II) for x, y in zip(names, names[1:])]
    sn = [f"S{d}{q}" for d in DIMS for q in QUARTERS]
    md += [(sn[0], sn[-1], II - 1)]
    return md


def order_edges(o):
    """stream ops in program order (a post-modified pointer walks in iteration order)"""
    by = {x.name: x for x in o}
    names = [f"L{d}{q}" for d in DIMS for q in QUARTERS]
    for x, y in zip(names, names[1:]):
        by[y].after.append((x, 1))
    sn = [f"S{d}{q}" for d in DIMS for q in QUARTERS]
    for x, y in zip(sn, sn[1:]):
        by[y].after.append((x, 1))
    return o


def last_ops():
    return [f"S{d}{q}" for d in DIMS for q in QUARTERS]


def stages_ok(lp):
    S = lp.stages()
    for n in last_ops():
        assert lp.by_name[n].t // lp.II == S - 1, f"{n} not in the last stage"


KERNEL_ONLY = False  # tried: the negative iterations run their last-stage O stores before the buffer (g_lcd)
STAGES = 2
LIFE = 2  # register lifetimes <= LIFE x II (1: no modulo variable expansion, U = 1: half the body text)


def mkl():
    lp = Loop(order_edges(ops()), classes={"x": list(range(12)), "dm": [0, 1, 2, 3, 4]})
    lp.kernel_only = KERNEL_ONLY
    return lp


def build(II=None, times=None, cp=None, max_ii=24):
    if times is not None:
        lp = mkl()
        lp.place(times["II"], times["t"])
        lp.allocate()
        stages_ok(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 14, max(max_ii, II or 0) + 1):
        lp = mkl()
        res = cpsched.solve(lp, ii, maxdist=maxdist(ii), last_stage=last_ops(), max_len=STAGES * ii - 1, max_life=LIFE * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        lp.place(ii, res[0])
        try:
            lp.allocate()
            stages_ok(lp)
        except (Infeasible, AssertionError) as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        lp.times = {"II": ii, "t": res[0]}
        print(f"II {ii}: {res[1]}", file=sys.stderr)
        return lp
    raise Infeasible("none")


JUMP = 0x3000  # checker: tile 1's rows 12 KB past tile 0's end


def streams():
    if TWO:
        f = lambda i, k: 256 * len(DIMS) * i + 64 * k + (JUMP if i >= NIT_T else 0)  # noqa: E731
        lo, so = (f"L{DIMS[-1]}{QUARTERS[-1]}", f"S{DIMS[-1]}{QUARTERS[-1]}")
        mod = dict(dn=NIT_T - 1, dj=64, m=64 + JUMP)
        return {"o": (f, dict(base=0x79000, dn=1 << 20, dj=64, m=64, ops={lo: mod})),
                "os": (f, dict(base=0x79000, dn=1 << 20, dj=64, m=64, ops={so: mod}))}
    return {
        "o": (lambda i, k: 256 * len(DIMS) * i + 64 * k, dict(base=0x79000, dn=1 << 20, dj=64, m=64)),
        "os": (lambda i, k: 256 * len(DIMS) * i + 64 * k, dict(base=0x79000, dn=1 << 20, dj=64, m=64)),
    }


def check(lp, n=24):
    n -= n % lp.U
    return lp.check(n, streams())


def function(lp, name="attv"):
    """kernel-only (round 14, text): the loop runs E extra trips first over iterations -E..-1 (no prologue). Their early-stage ops
    run: the walkers start as if E pairs of a previous row block had passed (P: one slot back, counter at 4 - E; V: so that the
    wrap of the virtual pair 3 lands on group 0), the O loads start E x 512 B early (reads only; the stores are in the last stage)."""
    E = lp.extra_trips() if lp.kernel_only else 0
    assert E <= 3
    dcs = {"PW": (4 - E) % 4, "VW1": (4 - E) % 4}
    p_back = 448 if E else 0
    v_start = (3584 - 896 * E) if E else 0  # E pairs: E x (448 + 448), the last of them wrapping by -3584
    mods = []
    for op, (d, dn, dj, m) in MODS.items():
        mods += [f"mova\tdn{d}, #{dn}", f"movxm\tdj{d}, #{dj}", f"movxm\tm{d}, #{m}", f"mova\tdc{d}, #{dcs[op]}"]
    neg = lp.neg_steps.get("o", 0)
    two = []
    if TWO:  # r1 = 64 + jump = m2 = m3 (the caller adds the 64: add takes a 7-bit signed immediate and #64 assembled as -64)
        assert neg == 0
        two = ["mov\tm2, r1", "mov\tm3, r1", f"mova\tdn2, #{NIT_T - 1}", f"mova\tdn3, #{NIT_T - 1}", "mova\tdj2, #64",
               "mova\tdj3, #64", "mova\tdc2, #0", "mova\tdc3, #0"]
    pre = two + ["mov\tr7, r0", f"mova\t{R_MM}, #780", f"mova\t{R_ADD}, #60", "mova\tr24, #0", "mova\tr25, #0", "mov\tp3, p2",
           "mov\tp4, p0", "mov\tp5, p1", f"padda\t[p4], #{-p_back}", f"padda\t[p5], #{v_start}", f"padda\t[p2], #{-64 * neg}"] + mods + [
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"add.nc\tlc, r7, #{E}", "nop", "nop", "nop", "nop", "nop", "nop"]
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tnop"] * 4 + ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def report(lp):
    print(f"II {lp.II} (resource bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:10])}), U {lp.U}, "
          f"stages {lp.stages()}, maxlive {lp.maxlive()}")
    for o in sorted(lp.ops, key=lambda o: o.t):
        print(f"  t {o.t:3d} (stage {o.t // lp.II}, slot {o.t % lp.II:2d}) {o.slot} {o.name:6s} {lp.text(o, 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--max-ii", type=int, default=24)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--func", default="attv")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--sums", action="store_true", help="attvs: the row sums (P x ones), one row block per iteration")
    ap.add_argument("--two", action="store_true", help="round 18 V2T: both tiles in one call (O jump by d2 / d3, r1 = jump)")
    ap.add_argument("--stages", type=int, default=None, help="round 22: schedule length bound in IIs (default 2: text)")
    ap.add_argument("--life", type=int, default=None, help="round 22: register life bound in IIs")
    ap.add_argument("--workers", type=int, default=8)
    o = ap.parse_args()
    global TWO, STAGES, LIFE
    TWO = o.two and not o.sums
    global USE_BANKS, SUMS, DIMS, MODS
    USE_BANKS = not o.nobanks
    if o.sums:
        SUMS, DIMS, MODS, STAGES, LIFE = True, (0,), {"PW": (0, 0, 448, 448)}, 2, 1
        if o.func == "attv":
            o.func = "attvs"
    STAGES = o.stages or STAGES
    LIFE = o.life or LIFE
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=o.time, workers=o.workers), max_ii=o.max_ii)
    report(lp)
    for n in ((6, 1) if SUMS else ((48, 24, 4) if TWO else (24, 4))):
        cyc, land = check(lp, n)
        print(f"check OK ({n} iterations): {cyc} cycles, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    tag = "ATTVS" if SUMS else "ATTV"
    Path(o.out).with_suffix(".h").write_text(f"// generated by r14/gen_attv.py\n#define {tag}_U {lp.U}\n#define {tag}_II {lp.II}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
