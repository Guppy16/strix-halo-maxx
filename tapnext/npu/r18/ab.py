"""Round 18: A/B timing of two builds in ONE process, interleaved in blocks (the background CPU load and the box's state drift hit both
alike). Each block discards its first dispatches (a hw-context switch between the two xclbins precedes it).

    python3 r18/ab.py --a full12r17d --b full12r18a [--depth 12] [--rounds 10] [--block 22] [--skip 2] [--warmup 20]

Prints SPREAD lines per build (p10 / median / mean over rounds x (block - skip) dispatches) and the deltas b - a."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step8  # noqa: E402


def run_block(s, video, q, it0, n):
    out = []
    for it in range(it0, it0 + n):
        rel = torch.cat([q[..., :1] - it, q[..., 1:]], -1)
        s.step(video[it % 2], rel)
        out.append(s.last_us / 1e3)
    return out


def spread(name, tk):
    p = np.percentile(tk, [10, 25, 75, 90])
    print(f"SPREAD {name}: n {len(tk)} p10 {p[0]:.3f} p25 {p[1]:.3f} median {np.median(tk):.3f} p75 {p[2]:.3f} p90 {p[3]:.3f} "
          f"mean {np.mean(tk):.3f} ms", flush=True)
    return p[0], float(np.median(tk)), float(np.mean(tk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True, nargs="+")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--block", type=int, default=22)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    o = ap.parse_args()
    frames = np.load("/work/tapnet/data/davis_reg.npy")[:2]
    video = step8.preprocess(frames)
    q = step8.demo_queries()
    names = [o.a] + o.b
    ss = []
    for n in names:
        s = step8.Step8(n, o.depth)
        s.reset()
        ss.append(s)
    its = [0] * len(ss)
    for k, s in enumerate(ss):
        run_block(s, video, q, 0, o.warmup)
        its[k] = o.warmup
    res = {n: [] for n in names}
    t0 = time.perf_counter()
    for r in range(o.rounds):
        order = list(range(len(ss))) if r % 2 == 0 else list(reversed(range(len(ss))))
        for k in order:
            tk = run_block(ss[k], video, q, its[k], o.block)
            its[k] += o.block
            res[names[k]] += tk[o.skip:]
    print(f"AB {o.rounds} rounds x {o.block - o.skip} kept in {time.perf_counter() - t0:.0f} s", flush=True)
    base = spread(names[0], res[names[0]])
    for n in names[1:]:
        v = spread(n, res[n])
        print(f"DELTA {n} - {names[0]}: p10 {v[0] - base[0]:+.3f} median {v[1] - base[1]:+.3f} mean {v[2] - base[2]:+.3f} ms", flush=True)


if __name__ == "__main__":
    main()
