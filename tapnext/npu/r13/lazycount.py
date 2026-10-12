"""Round 13 (P56 step 0): how often would a row block need the slow path under the lazy-fast K loop?

Replays the kernel's lazy max on real scores (build/acts_davis3.npz attn_s_b<block>: 288 sampled query rows x 12 heads x
1072 keys, q k^T / 8; the kernel's scores are 8x that, TAU = 8 x 8, NEG = -64 x 8): per 48-key chunk, in the kernel's
units, a row block (8 rows) is flagged when any lane has s - m_old >= TAU - 0.25 SU (the lowered clamp bound), and the
stored max moves (today's rule) when a row's max passes m_old + TAU. Chunk 0 always takes the slow path (m_old = NEG).
Sampled rows are grouped 8 at a time (pessimistic: unrelated queries share a block).

    python3 r13/lazycount.py [bound_su]
"""

import sys
from pathlib import Path

import numpy as np

SU, TAU, NEG, KC = 8.0, 8.0, -64.0, 48
bound = float(sys.argv[1]) if len(sys.argv) > 1 else TAU - 0.25
z = np.load(Path(__file__).resolve().parent.parent / "build" / "acts_davis3.npz")
keys = sorted(k for k in z.files if k.startswith("attn_s_b"))
tot = flagged = moved = flag_after0 = units_after0 = 0
per_chunk = np.zeros(23)
for k in keys:
    s = z[k].astype(np.float32) * SU  # [rows, heads, keys] in kernel units
    rows, heads, nk = s.shape
    nch = -(-nk // KC)
    s = np.pad(s, ((0, 0), (0, 0), (0, nch * KC - nk)), constant_values=-1024 * SU)
    nrb = rows // 8
    s = s[:nrb * 8].reshape(nrb, 8, heads, nch, KC)
    m = np.full((nrb, 8, heads), NEG * SU, np.float32)
    for c in range(nch):
        sc = s[:, :, :, c, :]
        rowmax = sc.max(-1)
        flag = (rowmax - m >= bound * SU).any(1)  # per (rb, head)
        move = (rowmax > m + TAU * SU)
        m = np.where(move, rowmax, m)
        n = flag.size
        tot += n
        flagged += flag.sum()
        moved += move.any(1).sum()
        if c > 0:
            flag_after0 += flag.sum()
            units_after0 += n
            per_chunk[c - 1] += flag.mean()
print(f"blocks {len(keys)}: units {tot}, flagged {flagged} ({100 * flagged / tot:.1f}%), of which after chunk 0: "
      f"{flag_after0} / {units_after0} ({100 * flag_after0 / units_after0:.2f}%); row blocks whose max moved {moved} "
      f"({100 * moved / tot:.1f}%); bound {bound} SU")
print("flag rate per chunk 1..23 (%):", " ".join(f"{100 * v / len(keys):.1f}" for v in per_chunk))
