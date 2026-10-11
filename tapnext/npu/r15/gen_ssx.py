"""Round 15 (npu-research P29b): the SSM's log_a + exp + 1 - a^2 chain as ONE modulo-scheduled asm loop (EW cores, ew8.cc ew_ssm).

Replaces ssm9::loga_exp_a2 (r9/ssm9.h: loga_p + exp_pA..pD, five Peano loops through four 1 KB scratch slots, ~1,130
static cycles per 16 vectors; the deployed sequence is r14/ssm_exp_deployed.s). Per 32 lanes, op for op the deployed passes
(bit-identical by construction; vmac x, -1 -> vmsc x, 1.0 and vmac ones, -M0 -> vmsc ones, M0 are exact):
    t1 = bf(g * -8); la = bf(t1 * sp)                                    (loga_p)
    d = min(max(la, -60), 60); za = d L_HI + d L_LO; zn = za + 1 M0       (exp_pA)
    nb = bf(zn - 1 M0); f = bf(za - nb 1); n16 = even halves of zn's bits
    f2 = bf(f f); f3 = bf(f2 f); p = 1 + f c1 + f2 c2 + f3 c3             (exp_pB)
    ph = bf(p); pl = bf(p - ph 1)
    p2n = (n16 << 7 by vups / vsrs) + 0x3F80; r = ph p2n + pl p2n         (exp_pC)
    ah = bf(r) -> g (in place: a); al = bf(r - ah 1)
    t = bf(al 2); a2 = bf(ah ah + ah t); om = bf(1 - a2 1) -> om          (exp_pD)
Only bf16 1.0 lives in a register (x11); the other constants are reloaded per use from a static table on the
static table (bank 2), 1.0 fp32 (the c0 accumulator) by vlda.conv from the 1.0 row. Stores in the last stage (the drain's garbage
iterations only load: past g's end is g_hout, past tsp's g_tbx, both readable).

    extern "C" void ssx(bfloat16 *g /*p0: gate_a in, a out*/, const B32 *tsp /*p1: sp per 2 vectors*/, bfloat16 *om /*p2*/,
                         const bfloat16 *tab /*p3*/)
over 16 vectors of 32 lanes, with p3 = ssx_tab + 512 B (the constant table, CONSTS order, 64 B rows: ssx_tab_init()).

    python3 r15/gen_ssx.py OUT.s [--ii N] [--cp F.json | --times F.json] [--time S]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE.parent / "r12"))
from msched import Infeasible, Loop, Op  # noqa: E402

NV = 16
C_ONE = 11  # bf16 1.0 (also the 0x3F80 of 2^n's exponent bias)
X_FREE = list(range(11))
DM_FREE = [0, 1, 2, 3, 4]
# constant table rows (64 B each) on the stack: bf16 bit patterns
CONSTS = dict(one=16256, m8=-16128, n60=-15760, p60=17008, lhi=16313, llo=-17620, m0=19264, c1=16177, c2=15990, c3=15715, two=16384)
ROW = {k: i for i, k in enumerate(CONSTS)}
TAB = 512  # p3 = table + 512: load / store immediates are 4 bits x 64 B (-512 .. 448), so rows 0 .. 15 are reachable
OFF = {k: 64 * i - TAB for k, i in ROW.items()}
R_MAC = "r1"  # 828: the bf16 vmac config
BANK = dict(g=(1, 2), sp=(2,), om=(1,), tab=(2,))  # in situ: g (arena+13376) straddles banks 1 / 2, tsp bank 2, g_e2 bank 1,
# the constant table (ssx_tab, .aie.bank2: bank 2 has 2.5 KB free)

VMUL = "VMUL_f_vmul_bf_vmul_bf_core_X_X"
VMAC = "VMAC_f_vmac_bf_vmul_bf_core_X_X"
VMSC = "VMSC_f_vmac_bf_vmul_bf_core_X_X"
CONV = "VCONV_bf16_fp32_mv_x_srs_bf"
LDA_X = "VLDA_dmx_lda_x_pstm_nrm_imm"
LDB_2D = "VLDB_2D_dmx_ldb_x"
LDB_I = "VLDB_dmx_ldb_x_idx_imm"
LDA_I = "VLDA_dmx_lda_x_idx_imm"
LDA_CI = "VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx_imm"
ST_X = "VST_dmx_sts_x_pstm_nrm_imm"
ST_CONV = "VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm"
MAX = "VMAX_LT_bf16"
MIN = "VMIN_GE_bf16"
SHUF = "VSHUFFLE_vec_shuffle_x"
UPS = "VUPS_2x_mv_ups_x2c_upsSign1"
SRS = "VSRS_2x_mv_x_srs_cm_srsSign1"
VADD16 = "VADD_16"


def ops(rel_slot=None, banks=True):
    """one iteration (32 lanes). rel_slot: constant name -> 'a' / 'b' (load slot of its reload; default ldb)"""
    rs = rel_slot or {}
    bk = (lambda n: BANK[n]) if banks else (lambda n: ())  # noqa: E731
    o = []

    def rel(name, c, users):
        """reload constant c for `users` (one vreg named k_<name>)"""
        s = rs.get(name, "b")
        o.append(Op(f"K{name}", "lda" if s == "a" else "ldb", f"vld{s} {{k{name}.x}}, [p3, #{OFF[c]}]",
                    defs=[(f"k{name}", "x", 7)], banks=bk("tab"), instr=LDA_I if s == "a" else LDB_I))
        return f"k{name}"

    def mul(name, d, a, b, op="vmul.f", acc=None):
        ins = {"vmul.f": VMUL, "vmac.f": VMAC, "vmsc.f": VMSC}[op]
        A = lambda v: f"x{C_ONE}" if v == "ONE" else f"{{{v}.x}}"  # noqa: E731
        uses = [(v, "x", 1) for v in (a, b) if v != "ONE"]
        if acc:
            o.append(Op(name, "vec", f"{op} {{{d}.dm}}, {{{acc}.dm}}, {A(a)}, {A(b)}, {R_MAC}", defs=[(d, "dm", 6)],
                        uses=[(acc, "cml", 4)] + uses, instr=ins))
        else:
            o.append(Op(name, "vec", f"{op} {{{d}.dm}}, {A(a)}, {A(b)}, {R_MAC}", defs=[(d, "dm", 6)], uses=uses, instr=ins))

    def conv(name, d, a):
        o.append(Op(name, "st", f"vconv.bf16.fp32 {{{d}.x}}, {{{a}.cml}}", defs=[(d, "x", 2)], uses=[(a, "cml", 1)], instr=CONV))

    # loga_p
    o.append(Op("LX", "lda", "vlda {g.x}, [p0], #64", defs=[("g", "x", 7)], mem=("ld", "g"), banks=bk("g"), instr=LDA_X))
    o.append(Op("LS", "ldb", "vldb.2d {sp.x}, [p1], d1", defs=[("sp", "x", 7)], mem=("ld", "sp"), banks=bk("sp"), instr=LDB_2D))
    mul("M8", "t1a", "g", rel("m8", "m8", ["M8"]))
    conv("C1", "t1", "t1a")
    mul("M2", "laa", "t1", "sp")
    conv("C2", "la", "laa")
    # exp_pA
    kn = rel("n60", "n60", ["MX"])
    o.append(Op("MX", "mv", f"vmax_lt.bf16 {{d0.x}}, r16, {{la.x}}, {{{kn}.x}}", defs=[("d0", "x", 2)], uses=[("la", "x", 1), (kn, "x", 1)], instr=MAX))
    kp = rel("p60", "p60", ["MN"])
    o.append(Op("MN", "mv", f"vmin_ge.bf16 {{d.x}}, r16, {{d0.x}}, {{{kp}.x}}", defs=[("d", "x", 2)], uses=[("d0", "x", 1), (kp, "x", 1)], instr=MIN))
    mul("Z1", "za0", "d", rel("lhi", "lhi", ["Z1"]))
    mul("Z2", "za", "d", rel("llo", "llo", ["Z2"]), op="vmac.f", acc="za0")
    km = rel("m0", "m0", ["Z3", "Z4"])
    mul("Z3", "zn", "ONE", km, op="vmac.f", acc="za")
    mul("Z4", "nbm", "ONE", km, op="vmsc.f", acc="zn")
    conv("C3", "nb", "nbm")
    mul("Z5", "fa", "nb", "ONE", op="vmsc.f", acc="za")
    conv("C4", "f", "fa")
    for nm, v, q in (("N1", "nlo", "bmll"), ("N2", "nhi", "bmlh")):  # vmov x <- bm: itinerary II_VMOV_alu_mv_mv_x_eXe_eBMLx
        o.append(Op(nm, "mv", f"vmov {{{v}.x}}, {{zn.{q}}}", defs=[(v, "x", 2)], uses=[("zn", q, 1)]))
        o[-1].ports = [(0, "DM_RM_L0_PORT")]  # (not in aie2p_isa_timing.json: the generic VMOV_alu_mv_mv_x has per-class itineraries)
    o.append(Op("N3", "mv", "vshuffle {n16.x}, {nlo.x}, {nhi.x}, r2", defs=[("n16", "x", 2)], uses=[("nlo", "x", 1), ("nhi", "x", 1)], instr=SHUF))
    o.append(Op("U1", "mv", "vups.2x {nu.cml}, {n16.x}, s0, upssign1", defs=[("nu", "cml", 3)], uses=[("n16", "x", 1)], instr=UPS))
    o.append(Op("S1", "st", "vsrs.2x {n7.x}, {nu.cml}, s1, srssign1", defs=[("n7", "x", 4)], uses=[("nu", "cml", 1)], instr=SRS))
    o.append(Op("A1", "mv", f"vadd.16 {{p2n.x}}, {{n7.x}}, x{C_ONE}", defs=[("p2n", "x", 2)], uses=[("n7", "x", 1)], instr=VADD16))
    # exp_pB
    mul("F2", "f2a", "f", "f")
    conv("C5", "f2", "f2a")
    mul("F3", "f3a", "f2", "f")
    conv("C6", "f3", "f3a")
    o.append(Op("KC0", "lda", f"vlda.conv.fp32.bf16 {{c0.cml}}, [p3, #{OFF['one']}]", defs=[("c0", "cml", 7)], banks=bk("tab"), instr=LDA_CI))
    mul("P1", "pa1", "f", rel("c1", "c1", ["P1"]), op="vmac.f", acc="c0")
    mul("P2", "pa2", "f2", rel("c2", "c2", ["P2"]), op="vmac.f", acc="pa1")
    mul("P3", "pa", "f3", rel("c3", "c3", ["P3"]), op="vmac.f", acc="pa2")
    conv("C7", "ph", "pa")
    mul("P4", "pla", "ph", "ONE", op="vmsc.f", acc="pa")
    conv("C8", "pl", "pla")
    # exp_pC
    mul("R1", "ra1", "ph", "p2n")
    mul("R2", "ra", "pl", "p2n", op="vmac.f", acc="ra1")
    conv("C9", "ah", "ra")
    o.append(Op("STA", "st", "vst {ah.x}, [p4], #64", uses=[("ah", "x", 1)], mem=("st", "a"), banks=bk("g"), instr=ST_X))
    mul("R3", "ala", "ah", "ONE", op="vmsc.f", acc="ra")
    conv("C10", "al", "ala")
    # exp_pD
    mul("T1", "ta", "al", rel("two", "two", ["T1"]))
    conv("C11", "t", "ta")
    mul("Q1", "qa1", "ah", "ah")
    mul("Q2", "qa", "ah", "t", op="vmac.f", acc="qa1")
    conv("C12", "a2", "qa")
    o.append(Op("KC1", "lda", f"vlda.conv.fp32.bf16 {{c1a.cml}}, [p3, #{OFF['one']}]", defs=[("c1a", "cml", 7)], banks=bk("tab"), instr=LDA_CI))
    mul("O1", "oma", "a2", "ONE", op="vmsc.f", acc="c1a")
    o.append(Op("STO", "st", "vst.conv.bf16.fp32 {oma.cml}, [p2], #64", uses=[("oma", "cml", 1)], mem=("st", "om"), banks=bk("om"), instr=ST_CONV))
    return topo(o)


def topo(o):
    """msched wants defs before uses in list order: the reloads were appended next to their users; keep stable order"""
    defined, out, rest = set(), [], list(o)
    while rest:
        for x in rest:
            if all(v in defined for v, *_ in x.uses):
                out.append(x)
                defined |= {v for v, *_ in x.defs}
                rest.remove(x)
                break
        else:
            raise ValueError("cycle")
    return out


def mk(rel_slot=None, banks=True):
    return Loop(ops(rel_slot, banks), consts={"x": [C_ONE]}, classes={"x": X_FREE, "dm": DM_FREE})


def side_effects_last(lp):
    S = lp.stages()
    bad = [o.name for o in lp.ops if o.mem and o.mem[0] == "st" and o.t // lp.II != S - 1]
    assert not bad, f"stores before the last stage: {bad}"


def build(II=None, times=None, cp=None, max_ii=26, rel_slot=None, banks=True, U=1):
    if times is not None:
        lp = mk(times.get("rel", rel_slot), banks)
        lp.place(times["II"], times["t"])
        lp.allocate(U)
        side_effects_last(lp)
        lp.times = times
        return lp
    import cpsched
    for ii in range(II or 20, (II or max_ii) + 1):
        lp = mk(rel_slot, banks)
        res = cpsched.solve(lp, ii, last_stage=["STA", "STO"], max_life=U * ii, H=7 * ii, **(cp or {}))
        if res is None:
            print(f"II {ii}: CP-SAT found none", file=sys.stderr)
            continue
        print(f"II {ii}: CP-SAT {res[1]}", file=sys.stderr)
        lp.place(ii, res[0])
        try:
            lp.allocate(U)
        except Infeasible as e:
            print(f"II {ii}: {e}", file=sys.stderr)
            continue
        side_effects_last(lp)
        lp.times = {"II": ii, "t": res[0], "rel": rel_slot}
        return lp
    raise Infeasible("none")


def check(lp, n=NV):
    st = {"g": (lambda i: 64 * i, dict(base=0x10000, dn=0, dj=64, m=64)),
          "a": (lambda i: 64 * i, dict(base=0x10000, dn=0, dj=64, m=64)),
          "sp": (lambda i: 64 * (i // 2), dict(base=0x20000, dn=1, dj=0, m=64)),
          "om": (lambda i: 64 * i, dict(base=0x30000, dn=0, dj=64, m=64))}
    return lp.check(n, st)


def function(lp, name="ssx"):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = []
    pre += [
        f"movxm\tr0, #{CONSTS['one']}", f"vbcst.16\tx{C_ONE}, r0",
        "mova\tr1, #828", "mova\tr2, #2", "mov\ts0, #7", "mov\ts1, #0", "mov\tcrupsmode, #0", "mov\tcrsrsmode, #0",
        "mov\tp4, p0",  # a: stored over g
        "mova\tdn1, #1", "mova\tdj1, #0", "mova\tm1, #64", "mova\tdc1, #0",  # sp: each row twice
        f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", f"mova\tr0, #{NV // lp.U}", "add.nc\tlc, r0, #0",
        "nop", "nop", "nop", "nop", "nop", "nop",
    ]
    L += ["\t" + x for x in pre]
    L += lp.emit_loop(name, drain=0)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:",
          f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=None)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--u", type=int, default=1)
    ap.add_argument("--rel", default="", help="constants reloaded on lda, e.g. m8,c1 (default all on ldb)")
    ap.add_argument("--nobanks", action="store_true")
    ap.add_argument("--func", default="ssx")
    ap.add_argument("--fit", action="store_true")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    rel_slot = {k: "a" for k in o.rel.split(",") if k}
    lp = build(o.ii, times, cp=dict(time_limit=o.time), rel_slot=rel_slot, banks=not o.nobanks, U=o.u)
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}: {dict(list(lp.res_mii(detail=True).items())[:6])}), U {lp.U}, stages {lp.stages()}, "
          f"maxlive {lp.maxlive()}; check OK ({cyc} cycles, {NV} vectors)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp, o.func)) + "\n", o.func, o.out, o.fit)
    rows = ", ".join(f"SSX_ROW({b & 0xFFFF:#06x})" for b in CONSTS.values())
    Path(o.out).with_suffix(".h").write_text(f"// generated by r15/gen_ssx.py\n#define SSX_U {lp.U}\n#define SSX_II {lp.II}\n#define SSX_TAB_OFF {TAB}\n"
                                             f"#define SSX_NROWS {len(CONSTS)}\n#define SSX_ROWS {rows}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
