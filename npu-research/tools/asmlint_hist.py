#!/usr/bin/env python3
"""asmlint_hist.py - asmlint against the bugs that taught its rules (npu-research rev 11).

Each case is a historic artefact (copied into npu-research/r16/hist/; two are reconstructions, marked) with the rule that
must fire, or a fixed / deployed version on which it must NOT fire. The deployed build must be clean (no error, no warning).
Run in npu-iron (Peano):
    docker run --rm --network none -v $HOME/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron python3 /work/npu-research/tools/asmlint_hist.py
        [--prj /work/npu-tapnext/build/full12r15m.prj]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asmlint  # noqa: E402

HIST = Path(os.path.abspath(__file__)).parent.parent / "r16" / "hist"

# (file, must fire, must not fire, what happened)
CASES = [
    ("r11_g7.s", ["ZOL-END-ALIGN"], [], "r11 GELU probe before the end-alignment rule (bodies ran ~once)"),
    ("r11_p2.s", ["ZOL-END-ALIGN"], [], "r11 GELU probe, same"),
    ("r11_t1.s", ["LAT-UNLANDED", "LAT-MV-BYPASS"], [], "r11 first GELU_MS attempts: c + r >= c' + w timing (1 cycle optimistic), then the mv bypass"),
    ("cpy32_u2.s", ["ZOL-MIN-BODY"], [], "r12 gen_copy U 2: a 2-bundle body ran once"),
    ("bgelu_bg11_pad.s", ["PORT"], [], "r12 bgelu II 11: the .p2align pad shifted a prologue vlda's QEY_WA onto a body vfloor"),
    ("bgelu_bg13.s", ["STAGE-SIDE-EFFECT"], [], "r12 bgelu: x stored to its final place in an early stage (corrupted 4 vectors after every run)"),
    ("bgelu_bg12_good.s", [], ["STAGE-SIDE-EFFECT", "PORT"], "r12 bgelu fixed: x through a stack ring"),
    ("attk_b_p4push.s", ["PUSH-P2"], [], "r13 fused K loop pushing on p4 (does not assemble)"),
    ("attk_klda_exq.s", ["EX-ALIAS"], [], "r13 fused K loop holding Q in ex0-7 beside x constants (garbage)"),
    ("attk_d_dmada.s", ["PORT"], [], "r13: lda p0, [sp] co-issued with vldb.4x64 (DM_ADA): Q pointer garbage from iteration 2"),
    ("attk_e_good.s", [], ["PORT"], "r13: the same loop with the scalar load moved"),
    ("attk_i_r0.s", ["ABI-R0"], [], "r13: int32_t attk(...) read its trip count from r0 (5 trips, or a hang)"),
    ("attk_r13f_good.s", [], ["ABI-R0", "PORT", "EX-ALIAS"], "r13 deployed attk (trip count copied from r1)"),
    ("attkb_b3_fa1.s", ["SCALAR-PLUS1"], [], "r14 attkb: the flag OR read r4 one cycle after it was written (missed the last iteration)"),
    ("attkb_r2_good.s", [], ["SCALAR-PLUS1"], "r14 attkb fixed (FA at FO + 2)"),
    ("zero2_v1.s", ["ZOL-SETUP"], [], "r14 zero2 with six 2-byte nops after add.nc lc: ran once (RECONSTRUCTED)"),
    ("zero2_v2_good.s", [], ["ZOL-SETUP", "ZOL-SETUP-MARGIN"], "r14 zero2 deployed (six full-form nop bundles)"),
    ("ggk_r16_nofill.s", ["FIFO-POP9"], [], "r16 first ggk: the B stream popped 16 x per np after one fill (kk 4-6 terms lost)"),
    ("ggk2_r16_good.s", [], ["FIFO-POP9"], "r16 deployed ggk: a fill before every 8th B pop"),
    ("nrm_ln_nosave.s", ["ABI-CALLEE-SAVED"], [], "r15 rule: p6 / p7 written without save / restore (RECONSTRUCTED from nrm_ln.s)"),
]


def run(paths, model, tools):
    items = []
    for p in paths:
        rep = asmlint.Report()
        funcs, labels, _ = asmlint.load_input(p, tools, model, rep)
        asmlint.lint(funcs, labels, rep)
        items += rep.items
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prj", default=None, help="a deployed build that must lint clean (default: full12r15m if found)")
    o = ap.parse_args()
    model, tools = asmlint.Model(), asmlint.Tools()
    ok = True
    for fn, must, mustnot, what in CASES:
        items = [it for it in run([str(HIST / fn)], model, tools) if it["severity"] in ("error", "warning")]
        fired = {it["rule"] for it in items}
        miss, extra = [r for r in must if r not in fired], [r for r in mustnot if r in fired]
        good = not miss and not extra
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {fn:22s} fires {sorted(fired) or '-'}" + (f"  MISSING {miss}" if miss else "")
              + (f"  UNEXPECTED {extra}" if extra else "") + f"  | {what}")
    prj = o.prj or asmlint.find_file("npu-tapnext/build/full12r15m.prj")
    if prj:
        paths = [str(Path(prj) / f"elfs_main_core_{c}" / f"elfs_main_core_{c}.elf") for c in ("0_2", "0_5")]
        items = run(paths, model, tools)
        bad = [it for it in items if it["severity"] in ("error", "warning")]
        ok &= not bad
        print(f"{'PASS' if not bad else 'FAIL'}  {Path(prj).name}: {len(bad)} error / warning findings (deployed: must be 0); "
              f"{sum(it['severity'] == 'calib' for it in items)} calib findings in Peano functions")
        for it in bad[:10]:
            print("      ", it["rule"], it["func"], hex(it["addr"]), it["msg"][:120])
    print("asmlint_hist:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
