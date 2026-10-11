#!/usr/bin/env python3
"""ksplan.py - the GEMM k-step budget and its levers (npu-research rev 13).

The deployed k-step (mm_ks, r9/gen_ks.py; r17d) per core: C tile 144 x 64 fp32 (18 row blocks x 8 column blocks of 8x8),
K = 64 per k-step, 1,152 bfp16 8x8x8 MACs. A arrives as 9 objects of 1,152 B (2 row blocks x 8 k-panels) through a 3-slot
ring; B (4,608 B = 8 k-panels x 8 column blocks) is held for the k-step and popped once per MAC (1 x 4 register blocking:
4 accumulators = 4 column blocks of one row block; A's 8 panels held for 8 column blocks). Per group of 32 MACs the 4
accumulators are stored and reloaded (16 vlda + 16 vst of bm quarters), and the MACs stop for that swap.

This tool does arithmetic only (no scheduling): static bounds per variant, the in-situ DMA-collision model, and prices.

    ksplan.py                # the table: deployed decomposition, variants, in-situ model, ms per step
    ksplan.py --dis r18/r17d_0_2.dis   # also re-count the deployed body from the disassembly

Model facts used (MACHINE / SKILL): one memory access per bank per cycle (bank rule); a 576-bit pop and a fill each read
64 B (one access); FIFO pops issue on lda or ldb; vmac 1 per cycle; dependent vmac distance >= 3, or 2 through the one-deep
bypass (round 7); 5 accumulators (dm0-4), 12 ex registers; memtile -> core stream 7.95 B/cycle (r10/rtm.py).
UNVERIFIED (marked in the output): DMA writes are 32 B per access and the core yields one cycle when a DMA write hits a bank
the core accesses in that cycle (fits the measured 231-cycle in-situ excess within ~10 %).
"""

import argparse
import re

GHZ = 1.81
KSTEPS = 15552          # k-steps per GEMM core per depth-12 step (12 blocks x 1,296; MACHINE section 1)
MACS = 1152             # vmac per k-step
A_B, B_B = 10368, 4608  # bytes per k-step per core
R_MT = 7.95             # memtile -> core B/cycle (measured, idle consumer)
GAP_STATIC, GAP_MEAS = 51, 67   # between mm_ks calls: mm_kstep wrapper 26 + main's k-loop 25 (static) / measured
MS_PER_CYCLE = KSTEPS / GHZ / 1e6


def deployed(dis=None):
    """the r17d mm_ks: 9 trips x 4 groups x (32 MAC + swap + ldb surplus) + glue"""
    d = dict(trips=9, groups=4, mac=32, swap=12.75, ldb_surplus=3.0, glue=96)
    if dis:
        txt = open(dis).read().splitlines()
        body = [ln for ln in txt if re.match(r"^\s+[0-9a-f]+:", ln)]
        # the ZOL body of mm_ks: between .Lks_body and .Lks_end labels
        s = next(i for i, ln in enumerate(txt) if "<.Lks_body>:" in ln)
        e = next(i for i, ln in enumerate(txt) if "<.Lks_end>:" in ln)
        b = [ln for ln in txt[s:e] if re.match(r"^\s+[0-9a-f]+:", ln)]
        mac = sum("vmac" in ln for ln in b)
        swap = sum(("vmac" not in ln) and ("vlda\t bm" in ln or "vst\t bm" in ln) for ln in b)
        rest = len(b) - mac - swap
        d.update(mac=mac / 4, swap=swap / 4, ldb_surplus=rest / 4, body=len(b))
        del body
    per_group = d["mac"] + d["swap"] + d["ldb_surplus"]
    d["static"] = d["trips"] * d["groups"] * per_group + d["glue"]
    return d


def collisions(cycles, bank3_busy, bank12_busy, b_in_bank3=True, a_free_bank=False):
    """UNVERIFIED model: DMA writes (32 B each) cost the core a cycle when they hit a bank it accesses that cycle.
    bank3_busy / bank12_busy = core accesses per cycle in the B bank / in the A ring's banks."""
    b_writes, a_writes = B_B / 32, A_B / 32
    cb = b_writes * min(1.0, bank3_busy) if b_in_bank3 else b_writes * 0.3
    ca = a_writes * (0.1 if a_free_bank else min(1.0, bank12_busy))
    return cb + ca


VARIANTS = [
    # name, cycles per 8 MACs (ResMII), glue per call, bank-3 accesses per cycle, banks-1/2 accesses per cycle, note
    ("deployed (4 chains, swap per 32 MACs)", None, 96, None, None, "r17d"),
    ("ROT: 5th accumulator rotates, 2-3 chains, II 9 per 8 MACs (bank-3 bound)", 9, 60, 1.0, 0.47,
     "B 8 pops + 1 fill per 8 MACs in one bank"),
    ("ROT + B split over two banks (II 8: MAC bound)", 8, 60, 0.56, 0.53, "needs L1 moves"),
    ("2x2 blocking + ROT (B and A 0.56 access per MAC each)", 8, 70, 0.56, 0.6, "A re-popped 4x from L1; dm 4+1"),
    ("K-pair (two k-steps per C swap, groups as today)", None, 96, 0.75, 0.35, "needs a 3rd B buffer (+4.6 KB L1)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dis", default=None)
    o = ap.parse_args()
    d = deployed(o.dis)
    per_group = d["mac"] + d["swap"] + d["ldb_surplus"]
    print(f"deployed mm_ks per group of 32 MACs: {d['mac']:.0f} MAC bundles + {d['swap']:.2f} swap (C store / reload, MACs stopped)"
          f" + {d['ldb_surplus']:.2f} ldb-only (36 B ops per 32 MACs) = {per_group:.2f}; static mm_ks {d['static']:.0f} (tcall 1,824)")
    meas = 2055 + GAP_MEAS
    excess = 2055 - 1824
    c_now = collisions(None, 144 / 192, 0.35)
    print(f"measured k-step 2,055 in mm_ks + {GAP_MEAS} between calls = {meas}; in-situ excess {excess} "
          f"(bank rule static 24; DMA-collision model {c_now:.0f} [UNVERIFIED])")
    print(f"A port floor {A_B / R_MT:.0f} cycles (memtile -> core {R_MT} B/cycle, one channel); B {B_B / R_MT:.0f} (other channel)")
    print(f"1 cycle per k-step = {MS_PER_CYCLE * 1e3:.2f} us per step ({KSTEPS} k-steps per core)\n")
    print(f"{'variant':72s} {'static':>6s} {'coll':>5s} {'gap':>4s} {'k-step':>6s} {'d ms':>6s}")
    base = meas
    for name, ii, glue, b3, b12, note in VARIANTS:
        if ii is None and name.startswith("deployed"):
            st, coll = 1824, excess
        elif ii is None:  # K-pair: swap amortised over 64 MACs
            st = d["trips"] * d["groups"] * (d["mac"] + d["swap"] / 2 + d["ldb_surplus"]) + glue
            coll = collisions(None, b3, b12)
        else:
            st = MACS / 8 * ii + glue
            coll = collisions(None, b3, b12)
        for gapname, gap in (("", GAP_MEAS), (" + P35 (B lock + k-loop in mm_ks)", 16)):
            ks = max(st + coll, A_B / R_MT) + gap
            print(f"{name + gapname:72s} {st:6.0f} {coll:5.0f} {gap:4d} {ks:6.0f} {(ks - base) * MS_PER_CYCLE:+6.2f}")
        print(f"    {note}")
    print("\nlayout levers on top of ROT (UNVERIFIED collision model): B prefetch buffer outside the bank the k-step pops "
          f"(-{collisions(None, 1.0, 0) - collisions(None, 1.0, 0, b_in_bank3=False):.0f}), A ring in a bank without C pieces "
          f"(-{collisions(None, 0, 0.47) - collisions(None, 0, 0.47, a_free_bank=True):.0f}) cycles per k-step")


if __name__ == "__main__":
    main()
