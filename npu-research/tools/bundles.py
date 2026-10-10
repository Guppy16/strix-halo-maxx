"""Static bundle counts from `llvm-objdump -d -z --no-show-raw-insn` output of an AIE2P elf (npu-research rev 3).

Every line with an address is one bundle (one issue cycle when nothing stalls), so straight-line code costs its bundle
count. Loops (hardware loops `.L_LEnd*`, branch loops) are counted once: multiply by trips yourself.

usage:
  python3 bundles.py X.dis --funcs mm_acc mm_core        # bundles and bytes per function
  python3 bundles.py X.dis --range 0x170 0x360           # bundles in [lo, hi) (e.g. IRON's k-step loop)
  python3 bundles.py X.dis --calls                       # jl targets (by symbol) and how often each is called
  python3 bundles.py X.dis --sizes                       # every function's bytes, largest first (program memory)

Make the .dis in a read-only container:
  docker run --rm --network none -v <ws>:/work/ws:ro -v <out>:/work/out npu-iron bash -c \
    "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin/llvm-objdump -d -z --no-show-raw-insn <elf> > /work/out/X.dis"
"""

import argparse
import re
from collections import Counter


def parse(path):
    txt = open(path).read()
    syms = [(int(m.group(1), 16), m.group(2)) for m in re.finditer(r"^([0-9a-f]+) <([^>]+)>:", txt, re.M)]
    bundles = [(int(m.group(1), 16), m.group(2)) for m in re.finditer(r"^\s+([0-9a-f]+):\s+(.*)$", txt, re.M)]
    funcs = [(a, n) for a, n in syms if not n.startswith(".")]
    return syms, funcs, bundles


def func_ranges(funcs, bundles):
    end = max(a for a, _ in bundles) + 2
    out = {}
    for (a, n), nxt in zip(funcs, funcs[1:] + [(end, None)]):
        out[n] = (a, nxt[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dis")
    ap.add_argument("--funcs", nargs="*")
    ap.add_argument("--range", nargs=2)
    ap.add_argument("--calls", action="store_true")
    ap.add_argument("--sizes", action="store_true")
    o = ap.parse_args()
    syms, funcs, bundles = parse(o.dis)
    fr = func_ranges(funcs, bundles)
    for f in o.funcs or []:
        lo, hi = next(r for n, r in fr.items() if n == f or n.startswith(f))
        n = sum(1 for a, _ in bundles if lo <= a < hi)
        print(f"{f:24s} {n:5d} bundles  {hi - lo:6d} B  @0x{lo:x}")
    if o.range:
        lo, hi = (int(v, 16) for v in o.range)
        print(f"[0x{lo:x}, 0x{hi:x}): {sum(1 for a, _ in bundles if lo <= a < hi)} bundles")
    if o.calls:
        name = {a: n for a, n in funcs}
        c = Counter()
        for _, t in bundles:
            m = re.search(r"\bjl\s+#0x([0-9a-f]+)", t)
            if m:
                c[name.get(int(m.group(1), 16), m.group(1))] += 1
        for n, k in c.most_common():
            print(f"{k:4d} x jl {n}")
    if o.sizes:
        for n, (lo, hi) in sorted(fr.items(), key=lambda t: t[1][0] - t[1][1]):
            print(f"{hi - lo:6d} B  {n}")


if __name__ == "__main__":
    main()
