"""Round 11: the GELU-by-table epilogue loop as a hand modulo schedule (r11/msched.py).

Same arithmetic as mm8.cc gelu_lut_i (bit-identical): per 32 lanes
    xc = clamp(x, LO, HI); idx = floor(xc * 2^7) (byte offsets, 16 B per segment) + table bases + bias
    (offset fp32, slope bf16) = LUT(idx) (4 x vldb.4x64 per 16 lanes, both table copies)
    s = offset + slope * xc; sh = bf16(s); sl = bf16(s - sh); out = bf16(x * sh + x * sl)
Slots per 32 lanes: mv 9 (vmin, vmax, 2 vadd.32, 4 vshuffle, 1 vmov), ldb 8 (LUT), st 5 (2 vfloor, 2 vconv, vst.conv),
vec 4, lda 2 (x, and x reloaded for the final product so it need not live in a register across the chain).
The clamp takes the constant FIRST (vmin_ge HI, x): a NaN then clamps to HI, so garbage over-reads past the end of the
data (the drain iterations) always index inside the table.

    extern "C" void gelu_ms(const bfloat16 *in /*p0*/, bfloat16 *out /*p1*/, int trips /*r0*/, const float *ab /*p2*/,
                            const float *cd /*p3*/, int jump /*r1*/)
processes 32 lanes x U x trips: 8 vectors at a 64 B stride, then `jump` bytes (64 = contiguous, 576 = the first half of
consecutive 1 KB pieces); in place allowed (out = in).

    python3 r11/gen_gelu.py OUT.s [--ii N] [--probe K] [--func NAME] [--banks]   (prints the schedule report)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msched import Infeasible, Loop  # noqa: E402
from msched import Op as _Op  # noqa: E402

INSTR = {  # op name (prefix) -> the .td instruction (its itinerary's ports are reserved by the scheduler)
    "L": "VLDA_2D_dmx_lda_x", "R": "VLDA_2D_dmx_lda_x", "XL": "VLDA_2D_dmx_lda_x", "XS": "VST_2D_dmx_sts_x",
    "MN": "VMIN_GE_bf16", "MX": "VMAX_LT_bf16", "F0": "VFLOOR_s32_bf16_mv_float_to_int_w", "F1": "VFLOOR_s32_bf16_mv_float_to_int_w",
    "A0": "VADD_32", "A1": "VADD_32", "LA": "VLDB_4x64_lo", "LB": "VLDB_4x64_hi", "LC": "VLDB_4x64_lo", "LD": "VLDB_4x64_hi",
    "SO": "VSHUFFLE_vec_shuffle_bm", "SS": "VSHUFFLE_vec_shuffle_x", "MV": "VMOV_alu_mv_mv_w",
    "S": "VMAC_f_vmac_bf_vmul_bf_core_X_X", "D": "VMAC_f_vmac_bf_vmul_bf_core_X_X", "Q": "VMAC_f_vmac_bf_vmul_bf_core_X_X",
    "P": "VMUL_f_vmul_bf_vmul_bf_core_X_X", "SH": "VCONV_bf16_fp32_mv_x_srs_bf", "SL": "VCONV_bf16_fp32_mv_x_srs_bf",
    "ST": "VST_2D_CONV_bf16_fp32_dmx_sts_srs_bf",
    "Ra": "VLDA_2D_dmx_lda_x", "Rb": "VLDB_2D_dmx_ldb_x",  # round 12: x reloads on either load slot (plain vldb: QEY_WB only)
}


def Op(name, *a, **k):
    if "instr" not in k and name in RELOAD_INSTR:
        k["instr"] = RELOAD_INSTR[name]
    if "instr" not in k:
        base = name if name in INSTR else name.rstrip("0123456789")
        k["instr"] = INSTR[base]
    return _Op(name, *a, **k)

C_HI, C_LO, C_BASE, C_M1 = 8, 9, 10, 11  # constant x registers
X_FREE = [0, 1, 2, 3, 4, 5, 6, 7]
DM_FREE = [0, 1, 2, 3, 4]
SCALE = {"in": 0, "out": 1, "re": 2}  # 2D modifier register per stream: d0 / d1 / d2
RELOAD_INSTR = {}  # reload op name -> .td instruction (set by ops())
# round 12, flat-edge tables (r12/lutflat.py): the index clamp [-4.125, 4.09375] lands out-of-range x in flat segments
# (slope 0), so the s MAC takes the unclamped x and the xc ring goes; LUT_BIAS_B 36 * 16
IDX_HI, IDX_LO, FLAT_BIAS_B = 16515, -16252, 576  # bf16 0x4083 = 4.09375, 0xC084 = -4.125
RELOAD_PTRS = [("p4", "d2"), ("p2", "d3"), ("p3", "d4")]  # flat mode: (pointer, 2D modifier) per x reload stream


def parse_rel(spec):
    """'SPQ:a' = one x reload on lda read by S, P and Q; 'S:b,PQ:a' = two reloads (S's on ldb, P / Q's on lda)"""
    out = []
    for grp in spec.split(","):
        users, slot = grp.split(":")
        assert set(users) <= set("SPQ") and slot in ("a", "b"), spec
        out.append((users, slot))
    assert sorted("".join(u for u, _ in out)) == ["P", "Q", "S"], f"{spec}: S, P and Q each need exactly one reload"
    return out


def ops(banks=None, flat=False, rel=None):
    """one iteration (32 lanes). banks: dict stream / table -> bank number (None: no bank constraints).
    flat: flat-edge tables (no xc ring; S reads a reloaded x); rel: parse_rel spec of the x reloads (flat only)"""
    b = banks or {}
    rel = parse_rel(rel or "SPQ:a") if flat else None
    RELOAD_INSTR.clear()
    xr = {}  # user op -> the reload vreg it reads
    if flat:
        for k, (users, slot) in enumerate(rel):
            RELOAD_INSTR[f"R{k}"] = INSTR["R" + slot]
            for u in users:
                xr[u] = f"xr{k}"
    bk = lambda *names: tuple(b[n] for n in names if n in b)  # noqa: E731
    o = [
        Op("L", "lda", "vlda.2d {x.x}, [p0], d0", defs=[("x", "x", 7)], mem=("ld", "in"), banks=bk("in")),
        Op("MN", "mv", f"vmin_ge.bf16 {{t.x}}, r16, x{C_HI}, {{x.x}}", defs=[("t", "x", 2)], uses=[("x", "x", 1)]),
        Op("MX", "mv", f"vmax_lt.bf16 {{xc.x}}, r16, x{C_LO}, {{t.x}}", defs=[("xc", "x", 2)], uses=[("t", "x", 1)]),
    ]
    if not flat:  # xc is needed again by the s MAC ~20 cycles later: through a 4-slot ring on the stack (st + lda slots, no mv op)
        o += [
            Op("XS", "st", "vst.2d {xc.x}, [p2], d3", uses=[("xc", "x", 1)], mem=("st", "ring"), spill=("xc", "x"), banks=bk("ring")),
            Op("XL", "lda", "vlda.2d {xc2.x}, [p3], d4", defs=[("xc2", "x", 7)], mem=("ld", "ringr"), fill=("xc", "x"),
               after=[("XS", 8)], banks=bk("ring")),
        ]
    o += [
        Op("F0", "st", "vfloor.s32.bf16 {i0.x}, {xc.wl}, s0", defs=[("i0", "x", 2)], uses=[("xc", "wl", 1)]),
        Op("F1", "st", "vfloor.s32.bf16 {i1.x}, {xc.wh}, s0", defs=[("i1", "x", 2)], uses=[("xc", "wh", 1)]),
        Op("A0", "mv", f"vadd.32 {{j0.x}}, {{i0.x}}, x{C_BASE}", defs=[("j0", "x", 2)], uses=[("i0", "x", 1)]),
        Op("A1", "mv", f"vadd.32 {{j1.x}}, {{i1.x}}, x{C_BASE}", defs=[("j1", "x", 2)], uses=[("i1", "x", 1)]),
    ]
    for h in (0, 1):
        j, c0, c1 = f"j{h}", f"c0{h}", f"c1{h}"
        tb = bk("ab", "cd")
        o += [
            Op(f"LA{h}", "ldb", f"vldb.4x64.lo {{{c0}.wl}}, {{{j}.wl}}", defs=[(c0, "wl", 7)], uses=[(j, "wl", 1)], banks=tb),
            Op(f"LB{h}", "ldb", f"vldb.4x64.hi {{{c0}.wh}}, {{{j}.wl}}", defs=[(c0, "wh", 7)], uses=[(j, "wl", 1)], banks=tb),
            Op(f"LC{h}", "ldb", f"vldb.4x64.lo {{{c1}.wl}}, {{{j}.wh}}", defs=[(c1, "wl", 7)], uses=[(j, "wh", 1)], banks=tb),
            Op(f"LD{h}", "ldb", f"vldb.4x64.hi {{{c1}.wh}}, {{{j}.wh}}", defs=[(c1, "wh", 7)], uses=[(j, "wh", 1)], banks=tb),
            Op(f"SO{h}", "mv", f"vshuffle {{sacc.{'bmll' if h == 0 else 'bmlh'}}}, {{{c0}.x}}, {{{c1}.x}}, r3",
               defs=[("sacc", "bmll" if h == 0 else "bmlh", 2)], uses=[(c0, "x", 1), (c1, "x", 1)]),
            Op(f"SS{h}", "mv", f"vshuffle {{{'sl0' if h == 0 else 'slope'}.x}}, {{{c0}.x}}, {{{c1}.x}}, r2",
               defs=[("sl0" if h == 0 else "slope", "x", 2)], uses=[(c0, "x", 1), (c1, "x", 1)]),
        ]
    xs = xr["S"] if flat else "xc2"
    if flat:
        for k, (users, slot) in enumerate(rel):
            p, d = RELOAD_PTRS[k]
            o.append(Op(f"R{k}", "lda" if slot == "a" else "ldb", f"vld{slot}.2d {{xr{k}.x}}, [{p}], {d}", defs=[(f"xr{k}", "x", 7)],
                        mem=("ld", f"re{k}"), banks=bk("in")))
    o += [
        Op("MV", "mv", "vmov {slope.wl}, {sl0.wh}", defs=[("slope", "wl", 2)], uses=[("sl0", "wh", 1)]),
        Op("S", "vec", f"vmac.f {{s.dm}}, {{sacc.dm}}, {{slope.x}}, {{{xs}.x}}, r1", defs=[("s", "dm", 6)],
           uses=[("sacc", "cml", 4), ("slope", "x", 1), (xs, "x", 1)]),
        Op("SH", "st", "vconv.bf16.fp32 {sh.x}, {s.cml}", defs=[("sh", "x", 2)], uses=[("s", "cml", 1)]),
        Op("D", "vec", f"vmac.f {{d.dm}}, {{s.dm}}, {{sh.x}}, x{C_M1}, r1", defs=[("d", "dm", 6)],
           uses=[("s", "cml", 4), ("sh", "x", 1)]),
        Op("SL", "st", "vconv.bf16.fp32 {sl.x}, {d.cml}", defs=[("sl", "x", 2)], uses=[("d", "cml", 1)]),
    ]
    if not flat:
        o.append(Op("R", "lda", "vlda.2d {xr.x}, [p4], d2", defs=[("xr", "x", 7)], mem=("ld", "re"), banks=bk("in")))
    xp, xq = (xr["P"], xr["Q"]) if flat else ("xr", "xr")
    o += [
        Op("P", "vec", f"vmul.f {{p.dm}}, {{{xp}.x}}, {{sh.x}}, r1", defs=[("p", "dm", 6)], uses=[(xp, "x", 1), ("sh", "x", 1)]),
        Op("Q", "vec", f"vmac.f {{q.dm}}, {{p.dm}}, {{{xq}.x}}, {{sl.x}}, r1", defs=[("q", "dm", 6)],
           uses=[("p", "cml", 4), (xq, "x", 1), ("sl", "x", 1)]),
        Op("ST", "st", "vst.2d.conv.bf16.fp32 {q.cml}, [p1], d1", uses=[("q", "cml", 1)], mem=("st", "out"), banks=bk("out")),
    ]
    return o


def tapped(o, tap):
    """debug: the final store writes x register `tap` (e.g. xc, xc2, slope, sh, sl, xr) instead of the result"""
    if not tap:
        return o
    o = [x for x in o if x.name not in ("Q", "ST")] if tap not in ("p", "q") else [x for x in o if x.name != "ST"]
    o.append(Op("ST", "st", f"vst.2d {{{tap}.x}}, [p1], d1", uses=[(tap, "x", 1)], mem=("st", "out"), instr="VST_2D_dmx_sts_x"))
    used = {v for x in o for v, *_ in x.uses}
    return [x for x in o if not x.defs or any(v in used for v, *_ in x.defs) or x.spill or x.mem]


def build(II=None, banks=None, max_ii=16, tap=None, slack=0, flat=False, rel=None, seeds=1, times=None, cp=None):
    """times: a saved schedule {"II": n, "t": {op: cycle}} (replayed, no search); cp: dict of r12/cpsched.solve options
    (needs OR-tools: the laptop) - the solved schedule is returned in lp.times for saving"""
    lo = II or 9
    last = None
    mk = lambda: Loop(tapped(ops(banks, flat, rel), tap), consts={"x": [C_HI, C_LO, C_BASE, C_M1]},  # noqa: E731
                      classes={"x": X_FREE, "dm": DM_FREE})
    if times is not None:
        lp = mk()
        lp.slack = slack
        lp.flat, lp.rel = flat, parse_rel(rel or "SPQ:a") if flat else None
        lp.place(times["II"], times["t"])
        lp.allocate()
        lp.times = times
        return lp
    if cp is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r12"))
        import cpsched
        for ii in range(lo, (II or max_ii) + 1):
            lp = mk()
            lp.slack = slack
            lp.flat, lp.rel = flat, parse_rel(rel or "SPQ:a") if flat else None
            res = cpsched.solve(lp, ii, slack=slack, **cp)
            if res is None:
                print(f"II {ii}: CP-SAT found no schedule", file=sys.stderr)
                continue
            t, info = res
            print(f"II {ii}: CP-SAT {info}", file=sys.stderr)
            lp.place(ii, t)
            try:
                lp.allocate()
            except Infeasible as e:
                print(f"II {ii}: {e}", file=sys.stderr)
                continue
            lp.times = {"II": ii, "t": t}
            return lp
        raise Infeasible("CP-SAT: no schedule")
    for ii in range(lo, (II or max_ii) + 1):
        lp = mk()
        lp.slack = slack
        lp.flat, lp.rel = flat, parse_rel(rel or "SPQ:a") if flat else None
        try:
            try:
                lp.schedule(ii)
                lp.allocate()
            except Infeasible:
                for seed in range(seeds):  # round 12: IMS from several seeds (the eviction order is randomised)
                    try:
                        lp.schedule_ims(ii, seed=seed)
                        lp.allocate()
                        break
                    except Infeasible as e:
                        if seed == seeds - 1:
                            raise e
            return lp
        except Infeasible as e:
            last = e
            print(f"II {ii}: {e}", file=sys.stderr)
    raise last


def piece_off(i, jump):
    """byte offset of vector i: 8 vectors at 64 B, then `jump` (64 = contiguous)"""
    return (i // 8) * (7 * 64 + jump) + (i % 8) * 64


def check(lp, jump, n=None):
    n = n or 8 * lp.U * 3
    names = ("in", "out") + (tuple(f"re{k}" for k in range(len(lp.rel))) if getattr(lp, "flat", False) else ("re",))
    st = {s: (lambda i, j=jump: piece_off(i, j), dict(base=0x1000 * (k + 1), dn=7, dj=64, m=jump)) for k, s in enumerate(names)}
    if not getattr(lp, "flat", False):
        for s in ("ring", "ringr"):
            st[s] = (lambda i: (i % 4) * 64, dict(base=0x8000, dn=3, dj=64, m=-192))
    # in place is a separate question (the out stream shares the in addresses); the checker only verifies each stream's walk
    return lp.check(n, st)


def function(lp, name, drain=6):
    """asm text of one function: preamble (constants, pointers, 2D registers, loop registers), the schedule, return"""
    L = []
    a = L.append
    a(f"\t.section\t.text.{name},\"ax\",@progbits")
    a(f"\t.globl\t{name}")
    a("\t.p2align\t4")
    a(f"\t.type\t{name},@function")
    a(f"{name}:")
    flat = getattr(lp, "flat", False)
    pre = [
        "movxm\tr16, #52428",  # lane mask 0xCCCC: lanes 2,3, 6,7, ... index the cd copy (aie::lut layout)
        "mov\tm0, r1", "mov\tm1, r1", "mov\tm2, r1",
        "mov\tr1, p2", "vbcst.32\tx0, r1", "mov\tr1, p3", "vbcst.32\tx1, r1",
        f"mova\tr1, #{FLAT_BIAS_B if flat else 512}", "vbcst.32\tx2, r1",  # LUT_BIAS_B: segment 32 (flat: 36) at x = 0
        "vsel.32\tx3, x0, x1, r16", "nop", "nop",
        f"vadd.32\tx{C_BASE}, x3, x2",
    ]
    if flat:  # the index clamp of the flat-edge tables
        pre += [f"movxm\tr1, #{IDX_HI}", f"vbcst.16\tx{C_HI}, r1", f"movxm\tr1, #{IDX_LO}", f"vbcst.16\tx{C_LO}, r1"]
    else:
        pre += ["movxm\tr1, #16511", f"vbcst.16\tx{C_HI}, r1",  # 0x407F = 3.984375 = SGELU_HI - 1/64
                "movxm\tr1, #-16256", f"vbcst.16\tx{C_LO}, r1"]  # 0xC080 = -4.0
    pre += [
        "movxm\tr1, #-16512", f"vbcst.16\tx{C_M1}, r1",  # 0xBF80 = -1.0
        "mova\tr1, #828", "mova\tr2, #24", "mova\tr3, #5", "mov\ts0, #7",  # mac config, shuffle modes, LUT shift
    ]
    if flat:  # every x reload stream walks the input like p0 (its own pointer and 2D modifier)
        for k in range(len(lp.rel)):
            p, d = RELOAD_PTRS[k]
            if d != "d2":
                pre.append(f"mov\tm{d[1]}, m0")
            pre += [f"mov\t{p}, p0", f"mova\tdn{d[1]}, #7", f"mova\tdj{d[1]}, #64", f"mova\tdc{d[1]}, #0"]
    else:
        pre += [
            "mov\tp4, p0",
            "paddxm\t[sp], #256", "mov\tp2, sp", "padda\t[p2], #-256", "mov\tp3, p2",  # the xc ring: 4 x 64 B on the stack
            "mova\tdn3, #3", "mova\tdj3, #64", "mova\tm3, #-192", "mova\tdc3, #0",
            "mova\tdn4, #3", "mova\tdj4, #64", "mova\tm4, #-192", "mova\tdc4, #0",
        ]
    pre += [
        "mova\tdn0, #7", "mova\tdj0, #64", "mova\tdc0, #0",
        "mova\tdn1, #7", "mova\tdj1, #64", "mova\tdc1, #0",
    ] + ([] if flat else ["mova\tdn2, #7", "mova\tdj2, #64", "mova\tdc2, #0"]) + [
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r0, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + p for p in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop" if flat else "\tpaddxm\t[sp], #-256", "\tnop"]
    a(f".Lfunc_end_{name}:")
    a(f"\t.size\t{name}, .Lfunc_end_{name}-{name}")
    return L


def report(lp):
    lt = lp.lifetimes()
    print(f"II {lp.II} (resource bound {lp.res_mii()}), U {lp.U}, stages {lp.stages()}, pressure "
          + ", ".join(f"{c} {v:.1f}" for c, v in lp.pressure().items()))
    for o in sorted(lp.ops, key=lambda o: o.t):
        print(f"  t {o.t:3d} (stage {o.t // lp.II}, slot {o.t % lp.II:2d}) {o.slot} {o.name:4s} {o.fmt}")
    print("  lifetimes: " + ", ".join(f"{v} {h - l + 1}" for v, (l, h) in sorted(lt.items(), key=lambda kv: kv[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--func", default="gelu_ms")
    ap.add_argument("--banks", default=None, help="e.g. in=1,out=2,ab=0,cd=3")
    ap.add_argument("--tap", default=None, help="debug: store this x register instead of the result")
    ap.add_argument("--slack", type=int, default=0, help="keep registers reserved this many cycles after their last read")
    ap.add_argument("--flat", action="store_true", help="round 12: flat-edge tables (r12/lutflat.py), no xc ring")
    ap.add_argument("--rel", default=None, help="flat: x reloads, e.g. SPQ:a or S:b,PQ:a (users S / P / Q : load slot a / b)")
    ap.add_argument("--seeds", type=int, default=1, help="IMS attempts from different seeds per II")
    ap.add_argument("--times", default=None, help="round 12: replay a saved schedule (JSON from --cp)")
    ap.add_argument("--cp", default=None, help="round 12: solve with r12/cpsched.py (OR-tools) and save the schedule to this JSON; "
                    "e.g. reserve=x:1,time=60")
    ap.add_argument("--cp-opts", default="", help="cpsched options: reserve_x=1,time=60,w_life=1")
    o = ap.parse_args()
    banks = dict((k, int(v)) for k, v in (kv.split("=") for kv in o.banks.split(","))) if o.banks else None
    import json
    times = json.loads(Path(o.times).read_text()) if o.times else None
    cp = None
    if o.cp:
        kv = dict(a.split("=") for a in o.cp_opts.split(",") if a)
        cp = dict(reserve={"x": int(kv.get("reserve_x", 0)), "dm": int(kv.get("reserve_dm", 0))}, time_limit=float(kv.get("time", 60)),
                  w_life=int(kv.get("w_life", 0)), w_len=int(kv.get("w_len", 1)))
    lp = build(o.ii, banks, tap=o.tap, slack=o.slack, flat=o.flat, rel=o.rel, seeds=o.seeds, times=times, cp=cp)
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
        print("saved schedule", o.cp)
    report(lp)
    for jump in (64, 576):
        cyc, land = check(lp, jump)
        print(f"check OK (jump {jump}): {cyc} cycles issued, last write lands at {land}")
    text = function(lp, o.func)
    Path(o.out).write_text("\t.text\n" + "\n".join(text) + "\n")
    Path(o.out).with_suffix(".h").write_text(f"// generated by r11/gen_gelu.py\n#define GELU_MS_U {lp.U}\n#define GELU_MS_II {lp.II}\n")
    print(f"wrote {o.out} ({len(text)} lines), U {lp.U}")


if __name__ == "__main__":
    main()
