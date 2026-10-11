"""pass 19 KTLP-NL driver (r11/apb.py harness). Lock-free instrumented kernels (r24/mkvar.py): nothing can block.
build (npu-iron, CPU): python3 r24/ktlpb.py build --out /work/npu-research/r24/build/ktlp
run   (npu-iron, NPU): python3 r24/ktlpb.py run --out /work/npu-research/r24/build/ktlp"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "npu" / "r11"))
import apb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "run"])
ap.add_argument("--out", required=True)
ap.add_argument("--k", default=str(HERE / "mm_e7k_nl.s"))
ap.add_argument("--kh", help="the generator header of the schedule --k came from (its record order); copied to r24/e7k.h")
o = ap.parse_args()
apb.NW = 128
if o.cmd == "build":
    import shutil
    assert o.kh, "--kh: the record order must come from the kernel's own schedule (pass 19: a stale header timed out)"
    shutil.copy(o.kh, HERE / "e7k.h")
    asm = HERE / "build" / "ktlp_kernels.s"
    asm.write_text(Path(o.k).read_text() + "\n" + (HERE / "mm_e7_nl.s").read_text())
    o.asm, o.cc, o.nw, o.data_size, o.stack, o.stack_override = str(asm), str(HERE / "ktlpb.cc"), 128, 56 * 1024, 0x800, 0x600
    apb.ROOT = HERE  # the bench includes e7.h / e7k.h from r24/
    apb.build(o)
    sys.exit()
d = apb.Dev(o.out)
FIELDS = ["A acq", "A rel", "B acq", "B rel", "Y", "L", "N", "X", "r27", "r6", "p2"]
res = {}
for mode, n in ((0, 3), (1, 1), (1, 3), (2, 1), (2, 3)):
    y = [int(v) for v in d.call([mode, n])]
    res[mode, n] = y[:36]
    print(f"mode {mode} n {n}: B buffers {y[37]:#x} / {y[38]:#x}, rec {y[39]:#x}", flush=True)
    for c in range(3 if mode == 1 else 1):
        lg = y[40 + 16 * c:40 + 16 * c + 11]
        if lg[0] != -7:
            print("   call", c, ", ".join(f"{f} {v if f not in ('Y', 'L', 'N', 'X', 'p2') else hex(v & 0xFFFFF)}" for f, v in zip(FIELDS, lg)))
for key in res:
    if key[1] == 3 or key == (1, 1) or key == (2, 1):
        ref = res[(0, 3)] if key[1] == 3 else None
        if ref is not None:
            same = sum(a == b for a, b in zip(res[key], ref))
            print(f"mode {key[0]} n {key[1]}: {same}/36 piece hashes equal to E7 (mode 0, n 3)")
same11 = sum(a == b for a, b in zip(res[(1, 1)], res[(2, 1)]))
print(f"KTL1 n 1 vs KTL n 1: {same11}/36 equal")
import json  # noqa: E402
(HERE / "build" / "ktlp_ref.json").write_text(json.dumps({f"{m},{n}": v for (m, n), v in res.items()}))
