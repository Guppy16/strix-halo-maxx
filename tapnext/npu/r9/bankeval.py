"""Bank-conflict cost of mm_core_p for an accumulator piece layout (round 9; rule from npu-research tools/bankmodel.py:
a bundle costs max over banks of its memory accesses in that bank).

Parses the generated schedule (r7/gen_mm.py --pieces), walks the k-step's 9 stripes with their 4 piece addresses, A slot
and B bank, and prints the extra cycles per stripe and per k-step for the layouts below (addresses as linked in the step:
acc_hi 0x71900, C 0x78900, A slots 0x77B80 / 0x78000 / 0x78480, B fifo bank 3). Per-call ramps / drains and DMA are not
modelled: only same-bundle bank collisions.

usage: python3 r9/bankeval.py build/r9/mm_core_p.s [--acc-hi 0x71900]
"""

import argparse
import re
from collections import Counter


def bank(a):
    return (a - 0x70000) >> 14


def parse(path):
    """per bundle: list of (kind, index) with kind A (pop/fill), B, L (acc load #i), S (acc store #i)"""
    out, nl, ns = [], 0, 0
    for line in open(path):
        s = line.strip()
        if not s or s.startswith(".") or s.endswith(":") or s == "nop":
            if s == "nop":
                out.append([])
            continue
        ops = []
        for o in s.split(";"):
            o = o.strip()
            if o.startswith(("vlda.pop", "vlda.fill")):
                ops.append(("A", 0))
            elif o.startswith(("vldb.pop", "vldb.fill")):
                ops.append(("B", 0))
            elif o.startswith("vlda bm"):
                ops.append(("L", nl))
                nl += 1
            elif o.startswith("vst bm"):
                ops.append(("S", ns))
                ns += 1
        out.append(ops)
    return out


def stripe_cost(bs, pieces, a_addr, b_bank=3):
    extra = 0
    for b in bs:
        n = Counter()
        for k, i in b:
            if k == "A":
                n[bank(a_addr)] += 1
            elif k == "B":
                n[b_bank] += 1
            else:
                n[bank(pieces[i // 16] + 64 * (i % 16))] += 1
        extra += max([1] + list(n.values())) - 1
    return extra


def layout_affine(c0, h0):
    def piece(rb, pair):
        return h0 + 1024 * (6 + rb) if pair else (c0 + 1024 * rb if rb < 12 else h0 + 1024 * (rb - 12))
    return piece


def layout_old(c0, h0):  # rows < 48 in C, the rest in acc_hi, row block rb's 2 KB contiguous
    def piece(rb, pair):
        base = c0 + 2048 * rb if rb < 6 else h0 + 2048 * (rb - 6)
        return base + 1024 * pair
    return piece


def evaluate(bs, piece, slots):
    per = []
    for j in range(9):
        ps = [piece(2 * j, 0), piece(2 * j, 1), piece(2 * j + 1, 0), piece(2 * j + 1, 1)]
        per.append(stripe_cost(bs, ps, slots[j % 3]))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sched")
    ap.add_argument("--acc-hi", default="0x71900")
    ap.add_argument("--c", default="0x78900")
    ap.add_argument("--slots", default="0x77B80,0x78000,0x78480")
    o = ap.parse_args()
    bs = parse(o.sched)
    h0, c0 = int(o.acc_hi, 16), int(o.c, 16)
    slots = [int(v, 16) for v in o.slots.split(",")]
    for name, lay in (("old (contiguous rows)", layout_old(c0, h0)), ("affine pieces", layout_affine(c0, h0))):
        per = evaluate(bs, lay, slots)
        print(f"{name:24s} extra cycles per stripe {per}  per k-step {sum(per)}")


if __name__ == "__main__":
    main()
