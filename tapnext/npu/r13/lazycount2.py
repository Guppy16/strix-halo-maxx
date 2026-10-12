"""Round 13: lazy-max slow-path rate per head and per row (the attention op's wall is the slowest core's): r13/lazycount.py split."""
from pathlib import Path
import numpy as np
SU, TAU, NEG, KC = 8.0, 8.0, -64.0, 48
z = np.load(Path(__file__).resolve().parent.parent / "build" / "acts_davis3.npz")
keys = sorted(k for k in z.files if k.startswith("attn_s_b"))
rows_meta = [k for k in z.files if "row" in k or "idx" in k]
print("other keys:", [k for k in z.files if not k.startswith("attn_s")][:12], rows_meta)
per_head = np.zeros(12); per_row = None; n = 0
for k in keys:
    s = z[k].astype(np.float32) * SU
    rows, heads, nk = s.shape
    nch = -(-nk // KC)
    s = np.pad(s, ((0, 0), (0, 0), (0, nch * KC - nk)), constant_values=-1024 * SU)
    nrb = rows // 8
    s = s[:nrb * 8].reshape(nrb, 8, heads, nch, KC)
    m = np.full((nrb, 8, heads), NEG * SU, np.float32)
    if per_row is None: per_row = np.zeros((nrb, 8))
    for c in range(1, nch - 1):
        rowmax = s[:, :, :, c, :].max(-1)
        move = rowmax > m + TAU * SU
        m = np.where(move, rowmax, m)
        per_head += move.any(1).mean(0); per_row += move.any(2); n += 1
print("moved rate per head (%):", " ".join(f"{100 * v / n:.1f}" for v in per_head))
pr = 100 * per_row / n / 12
print("moved rate per sampled row (% of (head, chunk) units), rows 0..:", " ".join(f"{v:.0f}" for v in pr.reshape(-1)[:72]))
print("rows with rate > 20%:", int((pr > 20).sum()), "of", pr.size, "; mean", pr.mean())
