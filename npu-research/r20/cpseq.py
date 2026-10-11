#!/usr/bin/env python3
"""cpseq.py - command-processor work on the critical path of every op boundary (npu-research pass 15, SEQPRE).

The runtime sequence (.bin) is one instruction stream for the NPU's command processor (CP): BD block-writes, address patches,
queue pushes (write32), queue-space polls (maskpoll32), maskwrites, and token waits (tct). After a token-wait run (a sync: the
CP waits for a producer's drains) the CP executes the next op's instructions one by one; a DMA starts only when its push has
been executed. If the CP costs t_op per instruction, an op whose A path (shim A fill -> memtile slot load -> memtile -> core
send) is armed only after N post-sync instructions starts N x t_op late.

Per sync segment this tool reports
  now   = CP instructions from the sync to the last push that arms the A path of the segment's first op (GEMM / attention: the
          last memtile push before the first drain push; norm: the last shim MM2S push = its input fills)
  pre   = the same after SEQPRE (everything that does not read the producer's output issued BEFORE the sync: memtile slot loads
          and sends, B fills, drains, EW params fills; after it only the shim fills that read the producer's data)
and sums now - pre over the step. t_op is NOT measured (CPRATE probe, PROPOSALS rev 15); the timeline fits 0.13-0.22 us.

    python3 r20/cpseq.py /work/npu-tapnext/build/full12r19b.bin [--block 1]     (npu-iron: aie.utils.txn_trace)
"""

import argparse
import collections

from aie.utils import txn_trace as T

NAMES = ["yx", "ssm0"] + [f"ssm{h}" for h in range(1, 10)] + ["out", "norm1", "up_a+b", "down_a", "down_b", "norm2", "qkv", "attn",
                                                             "out_proj", "norm3", "mlp0", "mlp3_a+b", "norm0"]


def segments(path):
    w = T.load(path)
    ops = T.decode(w)[1:]
    pushes = [e for e in T.trace(w) if e.kind == "push"]
    pidx = [i for i, o in enumerate(ops) if o.kind == "write32"]
    assert len(pidx) == len(pushes)
    tag = {}
    for i, e in zip(pidx, pushes):
        tag[i] = ({0: "shim", 1: "mem"}.get(e.row, "core"), e.direction[0], e.channel, e.col, e.bd.length if e.bd else 0)
    runs, i = [], 0
    while i < len(ops):
        if ops[i].kind == "tct":
            j = i
            while j < len(ops) and ops[j].kind == "tct":
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    segs = []
    for k in range(len(runs) - 1):
        a, b = runs[k][1], runs[k + 1][0]
        segs.append([(x - a, ops[x].kind, tag.get(x)) for x in range(a, b)])
    return segs


def analyse(seg):
    n = len(seg)
    first_drain = next((i for i, _, t in seg if t and t[0] == "shim" and t[1] == "S"), n)
    mem = [i for i, _, t in seg if t and t[0] == "mem" and i < first_drain]
    shim_m = [i for i, _, t in seg if t and t[0] == "shim" and t[1] == "M"]
    if mem:  # GEMM / attention: the A path is armed at the last memtile push before the drains
        now = mem[-1] + 1
        # after SEQPRE: only the shim MM2S pushes on the A channel (0) before that point, with their BD writes / patches / polls
        a_fills = [i for i, _, t in seg if t and t[0] == "shim" and t[1] == "M" and t[2] == 0 and i < now]
        per_push = 4.5  # blockwrite + patch + maskpoll + write32 (+ maskwrite on some): measured mix of these segments
        pre = int(round(len(a_fills) * per_push))
        kind = "gemm/attn"
    elif shim_m:  # EW: input fills (shim MM2S ch 0) interleaved with drains; SEQPRE keeps only the input fills (not params)
        now = shim_m[-1] + 1
        cols = len({t[3] for _, _, t in seg if t and t[0] == "shim" and t[1] == "M"})
        fills = len(shim_m)
        inputs = max(0, fills - cols)  # one params fill per column first (plan8 norm: params, then the n_in inputs)
        pre = int(round(inputs * 5.0))
        kind = "ew"
    else:
        now, pre, kind = 0, 0, "-"
    return kind, n, now, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--block", type=int, default=1, help="which block's 24 segments to print")
    ap.add_argument("--tops", default="0.13,0.17,0.22", help="t_op values (us) to price with")
    o = ap.parse_args()
    segs = segments(o.bin)
    per = 24
    start = per * o.block
    tot_now = tot_pre = 0
    print(f"{'segment':10s} {'kind':9s} {'ops':>5s} {'now':>5s} {'pre':>5s} {'saved':>6s}")
    for k, seg in enumerate(segs):
        kind, n, now, pre = analyse(seg)
        tot_now += now
        tot_pre += pre
        if start <= k < start + per:
            nm = NAMES[(k - start) % len(NAMES)]
            print(f"{nm:10s} {kind:9s} {n:5d} {now:5d} {pre:5d} {now - pre:6d}")
    saved = tot_now - tot_pre
    print(f"step: {len(segs)} segments, post-sync CP instructions on the A path now {tot_now}, after SEQPRE {tot_pre}: {saved} saved")
    for t in (float(x) for x in o.tops.split(",")):
        print(f"  t_op {t:.2f} us -> {saved * t / 1e3:.2f} ms per step (upper bound: the SSM per-head segments are paced by "
              f"compute, not by the CP; subtract them below)")
    # the critical boundaries: an op whose data comes from the op just before it (dependent GEMMs, attention, the SSM's first
    # segment, the norms); down_a / down_b and the SSM's per-head segments are issued while the previous op computes
    crit = {"yx", "ssm0", "out", "norm1", "up_a+b", "norm2", "qkv", "attn", "out_proj", "norm3", "mlp0", "mlp3_a+b", "norm0"}
    cs = 0
    for k, seg in enumerate(segs):
        if NAMES[(k % per)] in crit and k // per < 12:
            kind, n, now, pre = analyse(seg)
            cs += now - pre
    print(f"  critical boundaries only (dependent ops, 12 blocks): {cs} CP instructions -> "
          + ", ".join(f"{cs * float(t) / 1e3:.2f}" for t in o.tops.split(",")) + f" ms at t_op {o.tops} us")

if __name__ == "__main__":
    main()
