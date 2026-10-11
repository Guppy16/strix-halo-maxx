s = open("NOTES.md").read()
old = s[s.index("- **The GEMM k-step is A-port-bound now**"):s.index("- LUT lookups (load_lut_2x_float")]
new = """- ~~The GEMM k-step is A-port-bound now~~ **CORRECTED in round 10**: the 2,588 (down_a) / 3,300 (out) cycles per k-step were
  MEANS over tiles. Per-tile medians (r10/gdist.py on build/g11sp.npy) are 2,050-2,065 for every op: steady k-steps run at
  their compute (static 1,830 + banks 24 + call/stamp overhead). Each op's first tile carries the A-slot load wait (28-34 us)
  inside its first in-kernel acquire; K = 768 ops with few tiles show it most. A arrives at >= 5 B/cycle per core. The
  ~4 B/cycle that is real: the DDR -> memtile slot load per column (8 x 248 KB in ~34 us = DDR read peak).
"""
s = s.replace(old, new)
open("NOTES.md", "w").write(s)
