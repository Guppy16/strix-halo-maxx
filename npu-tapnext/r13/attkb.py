"""Round 13: one-core bench of the attention K phase, C++ chunk_k vs the asm chunk_k_asm (r13/attkb.cc).

    python3 r13/attkb.py build --out build/r13/attkb [--qk r13/attqk.s --exp r13/aexp.s]     (npu-iron)
    python3 r13/attkb.py run --out build/r13/attkb [--seeds 0,1,2] [--scales 4,8,16] [--nkb 6,4]   (npu-iron, NPU)
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r11"))
import apb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--qk", default=str(HERE / "attqk.s"))
    ap.add_argument("--exp", default=str(HERE / "aexp.s"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scales", default="4,8,16", help="x 0.25 = the bf16 value range of Q and K")
    ap.add_argument("--nkb", default="6,4")
    ap.add_argument("--warm", default="0,2")
    ap.add_argument("--skip", default="0", help="asm skip masks (timing only): 1 attqk, 2 glue, 4 aexpk")
    ap.add_argument("--detail", action="store_true", help="run: per-block equality of scores / P / maxima")
    ap.add_argument("--dump", type=int, default=0, help="run: dump score block N (1-based) and P panel --panel of both paths")
    ap.add_argument("--panel", type=int, default=0)
    ap.add_argument("--cppexp", action="store_true", help="build: the C++ exp with the asm QK (aexp_stub.s)")
    ap.add_argument("--fused", default=None, help="build: the fused K loop (r13/gen_attk.py's .s) against chunk_k")
    ap.add_argument("--extra-asm", default="", help="build: more .s files to link (round 14: r14/zero2.s with ZASM)")
    ap.add_argument("--vphase", action="store_true", help="run: the V phase (chunk_v) hashes + cycles (compare a build with ATTV)")
    ap.add_argument("--krb", action="store_true", help="run: k_rb's phase stamps (build with --defs ...,KRB_STAMP)")
    ap.add_argument("--zero", action="store_true", help="run: test acc_zero (x[0] = 1) instead of the K phase")
    ap.add_argument("--hname", default="attk.h", help="build: the header name mm8.cc includes for --fused (round 14: attkb.h)")
    ap.add_argument("--defs", default="", help="build: extra macros for mm8.cc, e.g. ATTKE,X=1 (round 14)")
    o = ap.parse_args()
    o.nw, o.data_size, o.stack, o.stack_override = 128, 52 * 1024, 0xA00, 0x900  # L1: acc_hi 24.5 KB + the C object 18 KB + K 4.6 KB
    apb.NW = o.nw
    apb.GC = True
    apb.FLAGS = apb.FLAGS + ["-DMM_NOCOMPUTE"]  # the bench never runs mm_acc: its MAC loop would not fit beside both K paths
    if o.cmd == "build":
        bd = HERE.parent / "build"
        defs = ""
        if o.fused:
            shutil.copy(Path(o.fused).with_suffix(".h"), bd / o.hname)
            allp = Path(o.out + "_all.s")
            allp.write_text(Path(o.fused).read_text() + "".join("\n" + Path(x).read_text() for x in o.extra_asm.split(",") if x))
            defs = "#define ATTKF_BENCH\n" + "".join(f"#define {d.replace('=', ' ')}\n" for d in o.defs.split(",") if d)
        else:
            for src, name in ((o.qk, "attqk.h"), (o.exp, "aexp.h")):
                shutil.copy(Path(src).with_suffix(".h"), bd / name)
            allp = Path(o.out + "_all.s")
            allp.write_text(Path(o.qk).read_text() + "\n" + Path(o.exp).read_text())
            if o.cppexp:
                defs = "#define ATTK_CPPEXP\n"
        o.asm, o.cc = str(allp), str(HERE / "attkb.cc")
        if defs:
            cc = Path(o.out + "_v.cc")
            cc.write_text(defs + (HERE / "attkb.cc").read_text().replace('"../mm8.cc"', '"../../mm8.cc"'))
            o.cc = str(cc)
        apb.build(o)
        return
    d = apb.Dev(o.out)
    if o.vphase:
        for seed in range(3):
            y = d.call([3, seed])
            np_ = __import__("numpy"); np_.save(o.out + f"_v{seed}.npy", y[8:8 + 48].copy())
            print(f"V phase seed {seed}: {y[0]} cycles ({y[0] / 6:.0f} per row block), O hash {int(y[1]) & 0xFFFFFFFF:08x}, sums {int(y[2]) & 0xFFFFFFFF:08x}")
        return
    if o.krb:
        for moved in (1, 0):
            y = d.call([2, moved])
            print(f"k_rb ({'warm, the max moves: rescale' if moved else 'chunk 0: no rescale'}): first pass {y[0]}, exact test {y[1]}, "
                  f"rescale + -m rows {y[2]}, exact rerun {y[3]} cycles")
        return
    if o.zero:
        y = d.call([1])
        print(f"acc_zero: {y[0]} cycles, nonzero words C {y[1]} acc_hi {y[2]}, guards changed {y[3]}")
        return
    for skip in (int(k) for k in o.skip.split(",")):
      for warm in (int(w) for w in o.warm.split(",")):
        for nkb in (int(n) for n in o.nkb.split(",")):
            for sc in (int(s) for s in o.scales.split(",")):
                for seed in (int(s) for s in o.seeds.split(",")):
                    y = d.call([0, seed, sc, nkb, warm, skip, o.dump, o.panel])
                    if o.dump:
                        import numpy as np
                        h = lambda a: " ".join(f"{v:04x}" for v in np.array(a, np.int32).view(np.uint16))  # noqa: E731
                        print("   C++ s:", h(y[16:48])); print("   asm s:", h(y[48:80])); print("   C++ P:", h(y[80:98])); print("   asm P:", h(y[98:116]))
                    ok = "OK" if y[2] == 0 and y[6] == 0 and y[7] == 0 and y[8] == 0 and y[9] == 0 and y[10] == 0 else \
                        f"BAD P {y[2]} maxima {y[6]} scores {y[7]} O {y[8]} sums {y[9]} tail {y[10]}"
                    if o.detail:
                        print("   scores C++/asm equal per block:", "".join("=" if y[16 + c] == y[24 + c] else "X" for c in range(6)),
                              " P per row block:", "".join("=" if y[32 + r] == y[40 + r] else "X" for r in range(6)),
                              " maxima:", "".join("=" if y[48 + r] == y[56 + r] else "X" for r in range(6)))
                    print(f"skip {skip} warm {warm} nkb {nkb} scale {sc / 4:.2f} seed {seed}: C++ {y[0]} cycles ({y[0] / 6:.0f} per row block), "
                          f"asm {y[1]} ({y[1] / 6:.0f})  {ok}", flush=True)


if __name__ == "__main__":
    main()
