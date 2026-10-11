"""Round 21: memtile buffer map of a lowered design (input_with_addresses.mlir): per memtile, the buffers, bytes used and free holes.
usage: python3 r21/mtmap.py X.mlir [col ...]"""
import re
import sys

src = open(sys.argv[1]).read()
cols = [int(c) for c in sys.argv[2:]] or None
W = {"bf16": 2, "ui32": 4, "i32": 4, "i8": 1, "ui8": 1, "f32": 4, "i16": 2}
tot = {}
for m in re.finditer(r"%(\w+) = aie\.buffer\(%mem_tile_(\d)_1\) \{address = (\d+) : i32,[^}]*\} : memref<([\dx]+)x(\w+)>", src):
    name, c, addr, dims, t = m.groups()
    n = 1
    for d in dims.split("x"):
        n *= int(d)
    tot.setdefault(int(c), []).append((int(addr), n * W[t], name))
for c in sorted(tot):
    L = sorted(tot[c])
    used = sum(s for _, s, _ in L)
    holes, pos = [], 0
    for a, s, _ in L:
        if a > pos:
            holes.append((pos, a - pos))
        pos = max(pos, a + s)
    if pos < 524288:
        holes.append((pos, 524288 - pos))
    print(f"memtile {c}: {len(L)} buffers, used {used}, free {524288 - used}; holes " + ", ".join(f"{a}+{s}" for a, s in holes if s >= 1024))
    if cols is None or c in cols:
        for a, s, n in L:
            print(f"   {a:7d} {s:7d} {n}")
