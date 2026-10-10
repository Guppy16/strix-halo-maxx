"""Round 12: bf16 row blocks -> bfp panels as one modulo-scheduled asm loop (mm8.cc flags 4 / 5 / 6, the panel push).

Today, per row block rb: 8 x OutS::push(A64(load_v<64>(...)).to_vector<bfp16ebs8>()) - column blocks 0-3 from pair 0's
compact bf16 (rb_bf2(rb, 0)), 4-7 from pair 1's - to dst + 576 rb (flag 4: gelu_pan, contiguous k-step chunk) or into the
3 Q / K objects of 6 row blocks (3,456 B each at a 4,608 B stride, flags 5 / 6). One iteration here = one row block:
16 x vlda.conv.fp32.bf16 (bf16 -> fp32 into cml / cmh, exact), 8 x vst.push.576.conv.bfp16ebs8.fp32 and one
vst.flush.512.conv.2d whose 2D modifier adds `jump` after every `group`-th row block (flags 5 / 6: 1,152 B; flag 4: 0).
8 pushes = 576 B = 9 whole 64 B words, so every flush is at a word boundary (r12/gen_pushprobe.py: flushes there are
harmless; the pointer sits at the exact byte).

    extern "C" void pan(const bfloat16 *v /*p0*/, const bfloat16 *u /*p1*/, bfp16ebs8 *dst /*p2*/, int trips /*r0*/,
                        int jump /*r1*/, int group_m1 /*r2*/)
v and u walk 8 vectors (512 B) per row block, then +576 to the next 1 KB piece. trips * U row blocks per call.
The panels land behind the reads (mm8.cc: block j of rb ends at or below what rb has already loaded), and every push of
an iteration is ordered after all its loads (memory edges below).

    python3 r12/gen_pan.py OUT.s [--times F.json | --cp F.json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
sys.path.insert(0, str(HERE))
from msched import Infeasible, Loop, Op  # noqa: E402

LDC = "VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf"
PUSH = "VST_PUSH_576_CONV_bfp16ebs8_fp32"
FLUSH = "VST_FLUSH_512_CONV_2D"


def ops():
    o = []
    for j in range(8):  # column block j: 0-3 from v (p0), 4-7 from u (p1)
        p, d, s = ("p0", "d0", "v") if j < 4 else ("p1", "d1", "u")
        o += [Op(f"L{j}{h}", "lda", f"vlda.2d.conv.fp32.bf16 {{b{j}.{'cml' if h == 0 else 'cmh'}}}, [{p}], {d}",
                 defs=[(f"b{j}", "cml" if h == 0 else "cmh", 7)], mem=("ld", s), instr=LDC) for h in (0, 1)]
    for j in range(8):
        o.append(Op(f"P{j}", "st", f"vst.push.576.conv.bfp16ebs8.fp32 {{b{j}.dm}}, [p2, sf, r26]", uses=[(f"b{j}", "dm", 1)],
                    after=[(f"P{j - 1}", 1)] if j else [], instr=PUSH))
    o.append(Op("FL", "st", "vst.flush.512.conv.2d [p2, sf, r26, d4]", after=[("P7", 1)], instr=FLUSH))
    # memory: rb 0's panels land on its own bf16 in the C object: panel j (bytes 72 j .. 72 j + 71) overlaps v blocks <= j
    # (128 B each), panels 4-7 (bytes 288-575) v blocks 2-3; rb 1's panels 6-7 reach v(1) block 0. So panel j comes
    # after the v loads of blocks <= min(j, 3) (u is in acc_hi, never overwritten). Every push after all loads would
    # keep 8 blocks (8 dm) live: infeasible with 5.
    for x in o:
        if x.name.startswith("P"):
            j = int(x.name[1])
            x.after += [(f"L{c}{h}", 1) for c in range(min(j, 3) + 1) for h in (0, 1)]
    return o


def mk():
    o = ops()
    for grp in (["L00", "L01", "L10", "L11", "L20", "L21", "L30", "L31"], ["L40", "L41", "L50", "L51", "L60", "L61", "L70", "L71"]):
        by = {x.name: x for x in o}
        for a, b in zip(grp, grp[1:]):
            by[b].after.append((a, 1))
    return Loop(o, classes={"x": list(range(12)), "dm": [0, 1, 2, 3, 4]})


def maxdist(II):
    return [("L00", "L31", II - 1), ("L40", "L71", II - 1), ("P0", "FL", II - 1)]


def stream_ok(lp):
    S = lp.stages()
    names = [f"P{j}" for j in range(8)] + ["FL"]
    return all(lp.by_name[n].t // lp.II == S - 1 for n in names) and lp.by_name["FL"].t - lp.by_name["P0"].t <= lp.II - 1


def build(II=16, times=None, cp=None):
    lp = mk()
    if times is not None:
        lp.place(times["II"], times["t"])
    else:
        sys.path.insert(0, str(HERE))
        import cpsched
        res = cpsched.solve(lp, II, maxdist=maxdist(II), last_stage=[f"P{j}" for j in range(8)] + ["FL"], **(cp or {}))
        if res is None:
            raise Infeasible(f"CP-SAT: no schedule at II {II}")
        lp.place(II, res[0])
    lp.allocate()
    assert stream_ok(lp), "pushes / flush not in the last stage within one II"
    lp.times = {"II": lp.II, "t": {o.name: o.t for o in lp.ops}}
    return lp


def check(lp, nrb=3):
    st = {s: (lambda i, k: i * 1024 + 64 * k, dict(base=0x1000 * (n + 1), dn=7, dj=64, m=576)) for n, s in enumerate(("v", "u"))}
    n = nrb - nrb % lp.U
    return lp.check(n, st)


def function(lp, name="pan", drain=6):
    L = [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    pre = ["mov\tm4, r1", "mov\tdn4, r2", "mova\tdj4, #0", "mova\tdc4, #0",
           "mova\tdn0, #7", "mova\tdj0, #64", "movxm\tm0, #576", "mova\tdc0, #0",
           "mova\tdn1, #7", "mova\tdj1, #64", "movxm\tm1, #576", "mova\tdc1, #0", "mova\tr26, #0",
           f"movxm\tls, #{name}_start", f"movxm\tle, #{name}_last", "add.nc\tlc, r0, #0", "nop", "nop", "nop", "nop", "nop", "nop"]
    L += ["\t" + p for p in pre]
    L += lp.emit_loop(name, drain=drain)
    L += ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ii", type=int, default=16)
    ap.add_argument("--times", default=None)
    ap.add_argument("--cp", default=None)
    ap.add_argument("--fit", action="store_true", help="round 13: compact the loop body (r12/asmfit.py)")
    o = ap.parse_args()
    times = json.loads(Path(o.times).read_text()) if o.times else None
    lp = build(o.ii, times, cp=dict(time_limit=60))
    cyc, land = check(lp)
    print(f"II {lp.II} (bound {lp.res_mii()}), U {lp.U}, stages {lp.stages()}, maxlive {lp.maxlive()}; check OK ({cyc} cycles, 3 row blocks)")
    if o.cp:
        Path(o.cp).write_text(json.dumps(lp.times, indent=1, sort_keys=True) + "\n")
    import asmfit
    asmfit.emit(lp, lambda: "\t.text\n" + "\n".join(function(lp)) + "\n", "pan", o.out, o.fit)
    Path(o.out).with_suffix(".h").write_text(f"// generated by r12/gen_pan.py\n#define PAN_U {lp.U}\n")
    print("wrote", o.out)


if __name__ == "__main__":
    main()
