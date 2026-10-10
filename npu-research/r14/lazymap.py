"""npu-research rev 9: the fused K loop's flag / move map per (block, head, column, tile, row block, chunk), saved for sim16.

lazycount3.py (npu-tapnext r13) counted slow units per core; sim16 needs WHERE they fall, because the K / V multicast makes the
8 columns of a core row advance together per object: a chunk costs the slowest column's units. This replays the kernel's lazy
max on the CPU bf16 model's scores for the NPU's 1152 rows (column c = rows 144c .. 144c + 143, tile t = 48 rows, row block =
8 rows; 23 key chunks of 48 over keys 0..1071), in kernel units (s = q k^T, SU 8, TAU 64, NEG -512):
  flag[rb]  = some lane of the row block had s - m_old >= 63.75 (the asm loop's vlt bound: today's driver then runs the exact
              test + aexp_push, + rescale_rb when the max moves);
  move[rb]  = some row's max passed m_old + TAU (today's rescale; m moves to that row max).
Chunk 22 (16 valid keys) always takes the slow path in the deployed driver (nkb < CB); chunk 23 has no keys (no K work).

    docker run --rm -v ~/git/app/.scratch:/work -v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx npu-iron-torch /opt/ironenv/bin/python /work/npu-research/r14/lazymap.py 3
writes r14/lazymap.npz: flag, move (today) and pflag / pmove (policies x ..) [frames, 12 blocks, 12 heads, 8 cols, 3 tiles, 6 rb, 23 chunks] (uint8). CPU only, 4 threads.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/npu-tapnext")
import step8 as S8  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

TAU, FLAGB, NEG, KC, NK = 64.0, 63.75, -512.0, 48, 1072
NCH = -(-NK // KC)  # 23
torch.set_num_threads(4)
nfr = int(sys.argv[1]) if len(sys.argv) > 1 else 1
model = S8.load_model(12)
frames = np.load("/work/tapnet/data/clip30.npy")[:nfr]
video = S8.preprocess(frames)
q = S8.demo_queries()
cap = []
orig = F.scaled_dot_product_attention


def sdpa(qq, kk, vv, *a, **kw):
    cap.append((qq[0].detach().float().numpy(), kk[0].detach().float().numpy()))
    return orig(qq, kk, vv, *a, **kw)


F.scaled_dot_product_attention = sdpa
# policies: (name, TAU, key order, seed): "today" is the deployed kernel; the others change numerics (not bit-identical)
NKB = NK // 8  # 134 key blocks
perm_kb = np.concatenate([np.arange(c, NCH * 6, NCH) for c in range(NCH)])  # key-block interleave: chunk c = blocks c, c+23, ..
perm = (perm_kb[:, None] * 8 + np.arange(8)).ravel()  # key index per slot (>= 1072 = padding)
POL = [("today", 64.0, None, False), ("tau96", 96.0, None, False), ("tau128", 128.0, None, False),
       ("kbinter", 64.0, perm, False), ("selfseed", 64.0, None, True)]
FL = np.zeros((len(POL), nfr, 12, 12, 8, 3, 6, NCH), np.uint8)
MV = np.zeros_like(FL)
for it in range(nfr):
    rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
    s8 = S8.Step8.__new__(S8.Step8)
    s8.model = model
    s8._rows = None
    S8.HOST_FAST = 0
    rows = s8.x0_rows(video[it], rel)
    cap.clear()
    with torch.no_grad():
        x = rows
        for blk in model.blocks:
            x, _ = blk(x, None)  # caches not replayed (lazycount3's caveat: every frame as a first frame)
    for b, (qb, kb) in enumerate(cap):  # [H, 1152, 64]
        s0 = (qb @ kb[:, :NK].transpose(0, 2, 1)).astype(bfloat16).astype(np.float32)
        s0 = np.pad(s0, ((0, 0), (0, 0), (0, NCH * KC - NK)), constant_values=-1024 * 8.0)
        diag = np.einsum("hrd,hrd->hr", qb[:, :NK], kb[:, :NK])  # q_i . k_i (rows < 1072 have their own key)
        diag = np.pad(diag, ((0, 0), (0, 1152 - NK)), constant_values=NEG).astype(bfloat16).astype(np.float32)
        for pi, (name, tau, pm, seed) in enumerate(POL):
            s = s0 if pm is None else s0[:, :, pm]
            rmax = s.reshape(12, 8, 3, 6, 8, NCH, KC).max(-1)  # [H, col, tile, rb, row, chunk]
            m = diag.reshape(12, 8, 3, 6, 8) if seed else np.full(rmax.shape[:-1], NEG, np.float32)
            for c in range(NCH):
                rm = rmax[..., c]
                FL[pi, it, b, ..., c] = ((rm - m) >= tau - 0.25).any(-1)
                mv = rm > m + tau
                MV[pi, it, b, ..., c] = mv.any(-1)
                m = np.where(mv, rm, m)
    for pi, (name, *_r) in enumerate(POL):
        print(f"frame {it} {name:9s}: flag {100 * FL[pi, it].mean():.2f} %, move {100 * MV[pi, it].mean():.2f} %, "
              f"chunk 0 {100 * FL[pi, it][..., 0].mean():.1f} %, chunks 1-21 {100 * FL[pi, it][..., 1:22].mean():.2f} %", flush=True)
out = Path(__file__).resolve().parent / "lazymap.npz"
np.savez_compressed(out, flag=FL[0], move=MV[0], pflag=FL, pmove=MV, pnames=np.array([p[0] for p in POL]))
print("wrote", out)
