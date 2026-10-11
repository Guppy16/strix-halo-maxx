"""npu-research rev 9: statistics of the lazy-max fallback over the NPU's own layout (r14/lazymap.npz from lazymap.py).

Where do the slow units fall (chunk, column, head, block), and what does the K / V multicast make of them: a core row's 8
columns take each K object together, so per (block, head group, core row, pass, chunk) the wall pays the MAX over columns
of that chunk's slow row blocks, not the mean.
    python3 lazystat.py
"""
from pathlib import Path

import numpy as np

z = np.load(Path(__file__).resolve().parent / "lazymap.npz")
FL, MV = z["flag"].astype(bool), z["move"].astype(bool)  # [fr, blk, head, col, tile, rb, chunk 23]
nfr = FL.shape[0]
CH = np.arange(23)
J22 = np.zeros(23, bool)
J22[22] = True
SLOW = FL | J22  # the deployed driver's slow units (flag or the invalid-block chunk)

print(f"{nfr} frames x 12 blocks x 12 heads x 8 cols x 18 rb x 23 chunks")
print(f"slow (flag | chunk 22) {100 * SLOW.mean():.2f} %; flag {100 * FL.mean():.2f} %; move {100 * MV.mean():.2f} %; "
      f"flag-not-move {100 * (FL & ~MV).mean():.3f} %")
print(f"after chunk 0, excl. chunk 22: flag {100 * FL[..., 1:22].mean():.2f} %, move {100 * MV[..., 1:22].mean():.2f} %")

print("\nper chunk (flag %, all heads / cols):")
pc = 100 * FL.mean(axis=(0, 1, 2, 3, 4, 5))
print("  " + " ".join(f"{j}:{v:.1f}" for j, v in enumerate(pc)))
print("\nper column (flag % chunks 1-21):")
print("  " + " ".join(f"c{c}:{100 * FL[:, :, :, c, ..., 1:22].mean():.2f}" for c in range(8)))
print("per block (flag % chunks 1-21):")
print("  " + " ".join(f"b{b}:{100 * FL[:, b, ..., 1:22].mean():.2f}" for b in range(12)))
print("per head (flag % chunks 1-21):")
print("  " + " ".join(f"h{h}:{100 * FL[:, :, h, ..., 1:22].mean():.2f}" for h in range(12)))

# per core: core (row r, col c) takes heads r, r+3, r+6, r+9 over all 3 tiles
print("\nper core (slow % incl. chunks 0 / 22; in situ 8.6-10.9 %): rows = columns, cols = core rows")
for c in range(8):
    print(f"  col {c}: " + " ".join(f"{100 * SLOW[:, :, r::3, c].mean():5.1f}" for r in range(3)))

# lockstep: per (fr, blk, hg, row, pass, chunk) count slow rb per column; pass 0 = tiles 0,1 (12 rb), pass 1 = tile 2 (6 rb)
print("\nmulticast lockstep: slow row blocks per K object, mean over columns vs max over columns (chunks 1-21):")
for p, tiles in ((0, slice(0, 2)), (1, slice(2, 3))):
    n = FL[:, :, :, :, tiles, :, 1:22].sum(axis=(4, 5))  # [fr, blk, head, col, chunk]
    mean, mx = n.mean(3), n.max(3)
    nrb = 12 if p == 0 else 6
    print(f"  pass {p} ({nrb} rb): mean {mean.mean():.3f}, max {mx.mean():.3f} slow rb per object -> amplification "
          f"{mx.mean() / max(mean.mean(), 1e-9):.2f}x; P(any column slow) {100 * (mx > 0).mean():.1f} %")
    # by chunk
    print("    max by chunk: " + " ".join(f"{j + 1}:{v:.2f}" for j, v in enumerate(mx.mean(axis=(0, 1, 2)))))
    print("    mean by chunk:" + " ".join(f"{j + 1}:{v:.2f}" for j, v in enumerate(mean.mean(axis=(0, 1, 2)))))

# independence check: if flags were iid at the per-(chunk, pass) rate, what would the max be?
rng = np.random.default_rng(0)
for p, nrb in ((0, 12), (1, 6)):
    tiles = slice(0, 2) if p == 0 else slice(2, 3)
    rate = FL[:, :, :, :, tiles, :, 1:22].mean(axis=(0, 1, 2, 3, 4, 5))  # per chunk
    sim = rng.binomial(nrb, rate[None, :, None], size=(20000, 21, 8))
    print(f"  iid at the per-chunk rate, pass {p}: max {sim.max(-1).mean():.3f} vs mean {sim.mean():.3f}")

# which columns are the max: the slowest column per object (ties split)
n0 = FL[:, :, :, :, 0:2, :, 1:22].sum(axis=(4, 5))
argm = (n0 == n0.max(3, keepdims=True)) & (n0.max(3, keepdims=True) > 0)
w = argm / np.maximum(argm.sum(3, keepdims=True), 1)
print("\nshare of 'slowest column' per object, pass 0 (chunks 1-21): " +
      " ".join(f"c{c}:{100 * w[:, :, :, c].sum() / max(w.sum(), 1):.0f}%" for c in range(8)))
# column 7 holds 16 image rows + 48 queries + 80 zero rows; which rows flag there?
f7 = FL[:, :, :, 7, :, :, 1:22].mean(axis=(0, 1, 2, 5))  # [tile, rb]
print("column 7 flag % by (tile, rb), chunks 1-21:\n" + "\n".join("  t%d " % t + " ".join(f"{100 * v:5.2f}" for v in f7[t])
                                                              for t in range(3)))
