"""Round 11: cheaper host packing of X0 (to_q16) — variants must be byte-identical to pack8.to_q16.
    python3 r11/q16bench.py"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/npu-tapnext")
import pack8 as P  # noqa: E402
from plan8 import COLS, M, RB, UNIT, XB  # noqa: E402

N_IMG, NQ = 1024, 48
rng = np.random.default_rng(0)
x = torch.from_numpy(rng.standard_normal((N_IMG, 768), np.float32)).to(torch.bfloat16)
tok = torch.from_numpy(rng.standard_normal((NQ, 768), np.float32)).to(torch.bfloat16)


def ref():
    rows = torch.zeros(COLS * M, 768, dtype=torch.bfloat16)
    rows[:N_IMG] = x
    rows[N_IMG:N_IMG + NQ] = tok
    return P.to_q16(rows.view(torch.int16).numpy().view(P.bfloat16))


OUT = np.zeros((COLS, 12, RB, UNIT), np.uint8)  # persistent: the panel parts stay zero
ROWS = torch.zeros(COLS * M, 768, dtype=torch.bfloat16)


def v_persist():
    ROWS[:N_IMG] = x
    ROWS[N_IMG:N_IMG + NQ] = tok
    v = ROWS.view(torch.int16).numpy().reshape(COLS, RB, 8, 12, 8, 8).transpose(0, 3, 1, 4, 2, 5)
    dst = OUT[..., :XB].view(np.int16).reshape(COLS, 12, RB, 8, 8, 8)
    np.copyto(dst, v)
    return OUT.reshape(-1).view(np.uint32)


OUT_T = torch.zeros((COLS, 12, RB, UNIT // 2), dtype=torch.int16)


def v_torch():
    ROWS[:N_IMG] = x
    ROWS[N_IMG:N_IMG + NQ] = tok
    v = ROWS.view(torch.int16).reshape(COLS, RB, 8, 12, 8, 8).permute(0, 3, 1, 4, 2, 5)
    OUT_T[..., :XB // 2].view(COLS, 12, RB, 8, 8, 8).copy_(v)
    return OUT_T.numpy().reshape(-1).view(np.uint32)


def bench(f, n=50):
    for _ in range(5):
        f()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        t.append((time.perf_counter() - t0) * 1e3)
    return np.median(t), np.percentile(t, 90)


r = ref()
for name, f in (("ref", ref), ("persist", v_persist), ("torch", v_torch)):
    same = np.array_equal(f(), r)
    for th in (8, 1):
        torch.set_num_threads(th)
        md, p90 = bench(f)
        print(f"{name:8s} threads {th}: median {md:.3f} p90 {p90:.3f} ms, identical {same}")
