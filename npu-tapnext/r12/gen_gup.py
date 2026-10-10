"""Round 12 (npu-research G1a, "gup"): the gated epilogue's up half, product and bfp push as one modulo-scheduled asm loop.

Today (mm8.cc acc_epi, flags 2 / 3) the up half goes through bias_pcs (fp32 + bias -> bf16 in place), then a C++ loop
multiplies gate x up per row block and pushes 4 bfp blocks. Per 64-lane block (column block cb of row block rb) this
kernel does the same ops in one pass:
    u = bf16(acc_up + bias_up * 1.0)              (vmac + vconv: bias_pcs's arithmetic)
    p = bf16(v * u)                               (v = the gate, already bias + GELU'd bf16 in L1; vmul + vconv)
    push(fp32(p)) as one bfp16ebs8 block           (vconv.fp32.bf16 + vst.push.576.conv)
    vst.flush.512.conv.2d: after every 4th block the 2D modifier adds `jump` to the stream pointer (32: the stash's 320 B
    slots; 288: gated_b's [own 4 | stash 4] 576 B rows). r12/gen_pushprobe.py measured: the stream pointer sits at the
    exact byte after a flush, and the 2D post-modify adds to it.

    extern "C" void gup(const bfloat16 *v /*p0*/, const float *u /*p1*/, bfp16ebs8 *dst /*p2*/, int trips /*r0*/,
                        const bfloat16 *bias /*p3*/, int jump /*r1*/)
v walks 8 vectors (one row block's 4 blocks) at 64 B then +576 to the next 1 KB piece; u is contiguous fp32 (1 KB per row
block); bias = the up half's 4 blocks (512 B), re-walked for every row block. Blocks per call = trips * U.

    python3 r12/gen_gup.py OUT.s [--ii N] [--times F.json] [--cp F.json]     (prints the schedule report)
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE))
from msched import Infeasible, Loop, Op  # noqa: E402

LDB_X = "VLDB_2D_dmx_ldb_x"
LDA_BM = "VLDA_2D_dmx_lda_bm"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
UPS = "VCONV_fp32_bf16_mv_ups_xbf"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"
FLUSH = "VST_FLUSH_512_CONV_2D"
C_ONE = 11  # bf16 1.0 in every lane
X_FREE = list(range(11))
DM_FREE = [0, 1, 2, 3, 4]
CFG = "r2"  # vmac / vmul config (828: bf16 x bf16 -> fp32, as gen_gelu)


def ops():
    o = []
    for h in (0, 1):  # the block's two 32-lane halves
        u, ua = f"u{h}", f"ua{h}"
        o += [
            Op(f"V{h}", "ldb", f"vldb.2d {{v{h}.x}}, [p0], d0", defs=[(f"v{h}", "x", 7)], mem=("ld", "v"), instr=LDB_X),
            Op(f"B{h}", "ldb", f"vldb.2d {{b{h}.x}}, [p3], d3", defs=[(f"b{h}", "x", 7)], mem=("ld", "bias"), instr=LDB_X),
            Op(f"A{h}0", "lda", f"vlda.2d {{{ua}.bmll}}, [p1], d1", defs=[(ua, "bmll", 7)], mem=("ld", "u"), instr=LDA_BM),
            Op(f"A{h}1", "lda", f"vlda.2d {{{ua}.bmlh}}, [p1], d1", defs=[(ua, "bmlh", 7)], mem=("ld", "u"), instr=LDA_BM),
            Op(f"M{h}", "vec", f"vmac.f {{y{h}.dm}}, {{{ua}.dm}}, {{b{h}.x}}, x{C_ONE}, {CFG}", defs=[(f"y{h}", "dm", 6)],
               uses=[(ua, "cml", 4), (f"b{h}", "x", 1)], instr=VMAC),
            Op(f"CU{h}", "st", f"vconv.bf16.fp32 {{{u}.x}}, {{y{h}.cml}}", defs=[(u, "x", 2)], uses=[(f"y{h}", "cml", 1)], instr=CONV),
            Op(f"P{h}", "vec", f"vmul.f {{p{h}.dm}}, {{v{h}.x}}, {{{u}.x}}, {CFG}", defs=[(f"p{h}", "dm", 6)],
               uses=[(f"v{h}", "x", 1), (u, "x", 1)], instr=VMUL),
            Op(f"CP{h}", "st", f"vconv.bf16.fp32 {{q{h}.x}}, {{p{h}.cml}}", defs=[(f"q{h}", "x", 2)], uses=[(f"p{h}", "cml", 1)], instr=CONV),
            Op(f"UP{h}", "mv", f"vconv.fp32.bf16 {{w.{'cml' if h == 0 else 'cmh'}}}, {{q{h}.x}}", defs=[("w", "cml" if h == 0 else "cmh", 2)],
               uses=[(f"q{h}", "x", 1)], instr=UPS),
        ]
    # the stream state (p2, sf, r26) is outside the register model: PU and FL stay in the last stage, FL right after PU,
    # and the next iteration's PU comes after this FL (asserted in build())
    o += [
        Op("PU", "st", "vst.push.576.conv.bfp16ebs8.fp32 {w.dm}, [p2, sf, r26]", uses=[("w", "dm", 1)], instr=PUSH),
        Op("FL", "st", "vst.flush.512.conv.2d [p2, sf, r26, d4]", after=[("PU", 1)], instr=FLUSH),
    ]
    return o


def mk():
    o = ops()
    order_edges(o)
    return Loop(o, consts={"x": [C_ONE]}, classes={"x": X_FREE, "dm": DM_FREE})


def stream_ok(lp):
    pu, fl = lp.by_name["PU"], lp.by_name["FL"]
    S = lp.stages()
    return pu.t // lp.II == S - 1 and fl.t // lp.II == S - 1 and fl.t - pu.t <= lp.II - 1


def build(II=None, max_ii=12, times=None, cp=None):
    if times is not None:
        lp = mk()
        lp.place(times["II"], times["t"])
        lp.allocate()
        assert stream_ok(lp)
        lp.times = times
        return lp
    for ii in range(II or 6, (II or max_ii) + 1):
        lp = mk()
        try:
            if cp is not None:
                sys.path.insert(0, str(HERE))
                import cpsched
                res = cpsched.solve(lp, ii, maxdist=maxdist(ii) + [("PU", "FL", ii - 1)], last_stage=["PU", "FL"], **cp)
                if res is None:
                    raise Infeasible("CP-SAT: none")
                lp.place(ii, res[0])
            else:
                try:
                    lp.schedule(ii)
                except Infeasible:
                    lp.schedule_ims(ii)
            lp.allocate()
            if not stream_ok(lp):
                raise Infeasible(f"PU / FL not together in the last stage: {lp.by_name['PU'].t}, {lp.by_name['FL'].t}, stages {lp.stages()}")
            lp.times = {"II": ii, "t": {o.name: o.t for o in lp.ops}}
            return lp
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
    raise Infeasible("no schedule")


def check(lp, nrb=3):
    """value-level replay (msched.check) of nrb row blocks: every load at its stream address, every read its own value"""
    n = 4 * nrb
    st = {
        "v": (lambda i, k: (i // 4) * 1024 + (i % 4) * 128 + 64 * k, dict(base=0x1000, dn=7, dj=64, m=576)),
        "u": (lambda i, k: i * 256 + 64 * k, dict(base=0x9000, dn=3, dj=64, m=64)),
        "bias": (lambda i, k: (i % 4) * 128 + 64 * k, dict(base=0xE000, dn=7, dj=64, m=-448)),
    }
    return lp.check(n, st)


STREAM_ORDER = [("V0", "V1"), ("B0", "B1"), ("A00", "A01", "A10", "A11")]


def order_edges(o):
    """each stream's ops in op order, one iteration inside one II (the post-modified pointer walks in iteration order)"""
    by = {x.name: x for x in o}
    for grp in STREAM_ORDER:
        for a, b in zip(grp, grp[1:]):
            by[b].after.append((a, 1))


def maxdist(II):
    return [(g[0], g[-1], II - 1) for g in STREAM_ORDER]


def function(lp, name="gup", drain=6):
    L = []
    a = L.append
    a(f"\t.section\t.text.{name},\"ax\",@progbits")
    a(f"\t.globl\t{name}")
    a("\t.p2align\t4")
    a(f"\t.type\t{name},@function")
    a(f"{name}:")
    pre = [
        "mov\tm4, r1", "mova\tdn4, #3", "mova\tdj4, #0", "mova\tdc4, #0",  # flush: +jump after every 4th block
        "mova\tdn0, #7", "mova\tdj0, #64", "movxm\tm0, #576", "mova\tdc0, #0",  # v: 8 vectors per row block, then the next 1 KB piece
        "mova\tdn1, #3", "mova\tdj1, #64", "mova\tm1, #64", "mova\tdc1, #0",  # u: contiguous
        "mova\tdn3, #7", "mova\tdj3, #64", "movxm\tm3, #-448", "mova\tdc3, #0",  # bias: the up half's 512 B, again per row block
        "movxm\tr3, #16256", f"vbcst.16\tx{C_ONE}, r3",  # bf16 1.0
        f"mova\t{CFG}, #828", "mova\tr26, #0",
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r0, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + p for p in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop"]
    a(f".Lfunc_end_{name}:")
    a(f"\t.size\t{name}, .Lfunc_end_{name}-{name}")
    return L


def report(lp):
    print(f"II {lp.II} (resource bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:5])}), U {lp.U}, "
          f"stages {lp.stages()}, maxlive {lp.maxlive()}")
    for o in sorted(lp.ops, key=lambda o: o.t):
        print(f"  t {o.t:3d} (stage {o.t // lp.II}, slot {o.t % lp.II:2d}) {o.slot} {o.name:4s} {o.fmt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--func", default="gup")
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None, help="solve with r12/cpsched.py (OR-tools) and save the schedule here")
    ap.add_argument("--fit", action="store_true", help="round 13: compact the loop body (r12/asmfit.py)")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times=times, cp=dict(time_limit=60) if o.cp else None)
    report(lp)
    cyc, land = check(lp)
    print(f"check OK: {cyc} cycles issued for 3 row blocks, last write lands at {land}")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    text = function(lp, o.func)
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r12/gen_gup.py\n#define GUP_U {lp.U}\n#define GUP_II {lp.II}\n")
    print(f"wrote {o.out} ({len(text)} lines), U {lp.U}")


if __name__ == "__main__":
    main()
