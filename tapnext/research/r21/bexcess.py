"""Rev 16: per-op excess over steady tiles = (gap + span) - nt * T_tile, from timelines (blocks 1-11, mean) and TPROF tile
parameters. T_tile = first + (K/64 - 1) ks + mid + epi + hand (TPROF r19tp medians; hand / ks per build below).
usage: python3 bexcess.py r18m|r19d tl1.json [...]"""
import json
import sys
from collections import defaultdict

import numpy as np

F = 1810.0  # cycles per us
BUILD = sys.argv[1]
# per build: steady k-step, first k-step, hand (C-acquire tiles), hand of a gated pair's second tile
P = dict(r18m=dict(ks=1600, first=1512, hand=3010, hand2=682), r19d=dict(ks=1576, first=1487, hand=1980, hand2=682))[BUILD]
# op: K, nt, epilogue cycles per tile (yx: 4 gelu 4,510 + 4 plain 1,313 per core), gated (pairs share a C object)
OPS = dict(yx=(768, 8, (4 * 4510 + 4 * 1313) / 8, False), out=(768, 4, 1310, False), up_a=(768, 16, 3263, True),
           up_b=(768, 16, 3263, True), down_a=(1536, 4, 1311, False), down_b=(1536, 4, 1308, False), qkv=(768, 12, 1941, False),
           out_proj=(768, 4, 1308, False), mlp0=(768, 16, 4918, False), mlp3_a=(1536, 4, 1311, False), mlp3_b=(1536, 4, 1308, False))
runs = [json.load(open(f))["ops"] for f in sys.argv[2:]]
gap, span = defaultdict(list), defaultdict(list)
for r in runs:
    last = 0.0
    for op in r:
        k = op["name"].split("_", 1)[1] if op["name"] != "final" else "final"
        if op["name"].split("_")[0] != "b0" and op["first"] is not None and op["last"] is not None:
            gap[k].append((op["first"] - last) * 1e3)
            span[k].append((op["last"] - op["first"]) * 1e3)
        if op["last"] is not None:
            last = op["last"]
tot = 0.0
print(f"{BUILD}: us per op (blocks 1-11 mean of {len(runs)} runs); T = steady tile")
print("op         K    nt  T_tile  nt*T    gap    span   share  excess")
for k, (K, nt, epi, gated) in OPS.items():
    hand = (P["hand"] + P["hand2"]) / 2 if gated else P["hand"]
    T = (P["first"] + (K // 64 - 1) * P["ks"] + 42 + epi + hand) / F
    g, s = np.mean(gap[k]), np.mean(span[k])
    ex = g + s - nt * T
    tot += ex
    print(f"{k:9s} {K:5d} {nt:3d} {T:7.2f} {nt * T:6.1f} {g:6.1f} {s:7.1f} {g + s:7.1f} {ex:7.1f}")
print(f"GEMM excess per block {tot:.1f} us -> {12 * tot / 1e3:.2f} ms per step")
for k in ("attn", "ssm", "norm0", "norm1", "norm2", "norm3"):
    print(f"{k:9s} gap {np.mean(gap[k]):6.1f} span {np.mean(span[k]):6.1f}")
