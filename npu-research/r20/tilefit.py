#!/usr/bin/env python3
"""tilefit.py - per-tile period of every GEMM op kind from host-polled timelines (npu-research pass 15).

A GEMM op's span (first output word .. last output word) covers nt - 1 tile periods per core (the first word lands when the
first tile's drain does). period = span / (nt - 1); compute = K / 64 k-steps x KSTEP cycles; the rest is the per-tile overhead
(acc_zero, epilogue, C-object drain wait, tile setup). The op's gap + first tile is its start-up (A slot load, first tile).

    python3 r20/tilefit.py npu-tapnext/build/tl_r18m_run{1,2,3}.json [--kstep 1652]
"""

import argparse
import json
from collections import defaultdict

import numpy as np

# per block (plan8.full_step): K, nt per core, epilogue, A = output of the IMMEDIATELY preceding op?
GEMM = dict(yx=(768, 8, "gelu x12 / plain", True), out=(768, 4, "plain", True), up_a=(768, 16, "gated", True),
            up_b=(768, 16, "gated", False), down_a=(1536, 4, "plain", False), down_b=(1536, 4, "plain", False),
            qkv=(768, 12, "qkv (V^T)", True), out_proj=(768, 4, "plain", True), mlp0=(768, 16, "gelu_pan", True),
            mlp3_a=(1536, 4, "plain", True), mlp3_b=(1536, 4, "plain", False))
GHZ = 1.81e3  # cycles per us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--kstep", type=float, default=1652.0, help="in-situ cycles per k-step incl. the gap between calls")
    o = ap.parse_args()
    span, gap = defaultdict(list), defaultdict(list)
    for f in o.files:
        ops = json.load(open(f))["ops"]
        last = None
        for op in ops:
            k = op["name"].split("_", 1)[1] if op["kind"] == "gemm" else op["kind"]
            blk = op["name"].split("_")[0]
            if op["first"] is not None and op["last"] is not None and blk not in ("b0",):
                span[k].append((op["last"] - op["first"]) * 1e3)
                if last is not None:
                    gap[k].append((op["first"] - last) * 1e3)
            if op["last"] is not None:
                last = op["last"]
    ks_us = o.kstep / GHZ
    print(f"k-step {o.kstep:.0f} cycles = {ks_us:.3f} us; medians over {len(o.files)} runs x blocks 1-11 (us)")
    print(f"{'op':9s} {'K':>5s} {'nt':>3s} {'epilogue':16s} {'dep':>4s} {'span':>7s} {'period':>7s} {'compute':>8s} {'tile ovh':>8s} "
          f"{'gap':>6s} {'ovh/op ms x12':>13s}")
    tot = 0.0
    for k, (K, nt, epi, dep) in GEMM.items():
        sp = float(np.median(span[k]))
        per = sp / (nt - 1)
        comp = K / 64 * ks_us
        ovh = per - comp
        g = float(np.median(gap[k]))
        tot += ovh * nt * 12 / 1e3
        print(f"{k:9s} {K:5d} {nt:3d} {epi:16s} {'yes' if dep else 'no':>4s} {sp:7.1f} {per:7.2f} {comp:8.2f} {ovh:8.2f} {g:6.1f} "
              f"{ovh * nt * 12 / 1e3:13.2f}")
    print(f"per-tile overhead summed over the step (12 blocks, all tiles of a core): {tot:.2f} ms")


if __name__ == "__main__":
    main()
