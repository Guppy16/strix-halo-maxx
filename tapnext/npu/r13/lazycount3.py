"""Round 13: the fused K loop's slow-path rate IN THE NPU'S OWN TERMS, from the CPU bf16 model on the NPU's 1152 rows (1024 image +
48 queries + 80 zero rows), per core (column c, core row r takes heads r, r+3, r+6, r+9; 3 tiles x 6 row blocks; 23 key chunks;
keys 0..1071). A unit is slow when: chunk 0 (stored max = NEG), chunk 22 (invalid key blocks), or a row's max passes the stored
max by TAU (today's rescale). The attention op's wall time is the slowest core's.
    python3 r13/lazycount3.py [frames]   (npu-iron-torch, CPU)
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/work/npu-tapnext")
import step8 as S8  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

SU, TAU, NEG, KC, NK = 8.0, 8.0 * 8, -64.0 * 8, 48, 1072
torch.set_num_threads(16)
model = S8.load_model(12)
frames = np.load("/work/tapnet/data/davis_reg.npy")[: int(sys.argv[1]) if len(sys.argv) > 1 else 1]
video = S8.preprocess(frames)
q = S8.demo_queries()
cap = []
orig = F.scaled_dot_product_attention
def sdpa(qq, kk, vv, *a, **kw):
    cap.append((qq[0].detach().float().numpy(), kk[0].detach().float().numpy()))
    return orig(qq, kk, vv, *a, **kw)
F.scaled_dot_product_attention = sdpa
caches = None
for it in range(len(video)):
    rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
    s = S8.Step8.__new__(S8.Step8); s.model = model; s._rows = None
    S8.HOST_FAST = 0
    rows = s.x0_rows(video[it], rel)  # [1152, 768] bf16 as the NPU sees them
    cap.clear()
    with torch.no_grad():
        x = rows
        for blk in model.blocks:
            x, c = blk(x, None if caches is None else caches[len(cap)])
    # (the conv / recurrence caches are not replayed across frames here: frame 0 numbers only, as the NPU's first frame)
    per_core = np.zeros((8, 3)); n_units = 0
    for b, (qb, kb) in enumerate(cap):  # [H, 1152, 64]
        s = (qb @ kb[:, :NK].transpose(0, 2, 1)).astype(bfloat16).astype(np.float32)  # [H, 1152 rows, 1072 keys], kernel units
        nch = -(-NK // KC)
        s = np.pad(s, ((0, 0), (0, 0), (0, nch * KC - NK)), constant_values=-1024 * SU)
        s = s.reshape(12, 8, 3, 6, 8, nch, KC)  # heads, columns, tiles, rb, rows, chunks, keys
        rowmax = s.max(-1)  # [H, col, tile, rb, row, chunk]
        for col in range(8):
            for r in range(3):
                heads = [r, r + 3, r + 6, r + 9]
                m = np.full((4, 3, 6, 8), NEG, np.float32)
                slow = 0
                for c in range(nch):
                    rm = rowmax[heads, col, :, :, :, c]
                    move = rm > m + TAU
                    unit_slow = move.any(-1) | (c == 0) | (c == nch - 1)
                    slow += unit_slow.sum()
                    m = np.where(move, rm, m)
                per_core[col, r] += slow
        n_units += 4 * 3 * 6 * nch
    pc = 100 * per_core / n_units
    print(f"frame {it}: slow units per core (%), rows = columns 0..7, cols = core rows:")
    for col in range(8):
        print(f"  col {col}: " + " ".join(f"{v:5.1f}" for v in pc[col]))
    print(f"  mean {pc.mean():.1f}%, max {pc.max():.1f}% (chunk 0 + chunk 22 alone: {100 * 2 / nch:.1f}%)")
