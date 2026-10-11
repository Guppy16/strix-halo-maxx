"""Hand-scheduled AIE2P assembly for the GEMM microkernel's register-blocked inner loop (phase 11).

  extern "C" void mm_tile_asm(bfp16ebs8 *pA /*p0: A sub-tile, [2 row halves][8 k-subs] panels (pack half=True)*/,
                              bfp16ebs8 *pB /*p1: B tile, [n/16][k/8][2] panels*/,
                              float *pC /*p2: fp32 accumulator rows of this call, [2 row blocks][12 col blocks][64]*/);

Same arithmetic as the compiled BLK12 loop (DIM_M 96, DIM_N 96, DIM_K 64, RHO 6 -> m_a 16, n 96, k 64): for each of
the 2 row blocks and each of the 6 (8 x 16) column blocks, acc0 / acc1 += sum over the 8 k-subs of A_q x B_{2q}^T /
B_{2q+1}^T, sequential in q (one chain per accumulator: bit-identical to Peano's kernel).
Schedule (latencies from aie2p_isa_timing.json): the B panels stream through the ldb-slot FIFO at one pop per cycle -
the ldb slot is the binding resource (16 pops + 2 fills per 16 MACs -> II 18 per block = 89% of the MAC slot) -
into a 4-register conveyor (ex8-11) read exactly when the pop lands (+8); the row block's 8 A panels are popped once
through the lda-slot FIFO into ex0-7 (the second row block's panels are popped into the registers as the first
row block's last column block finishes with them); the two accumulator chains alternate (a chain accepts a mac
every 2 cycles). The fp32 accumulators of column block n (8 x 64 B quarter loads, lda slot) are loaded 5 cycles
before its first pop into the dm pair n % 2 - after block n-2's stores have read that pair - and stored (8 x 64 B,
st slot) 30 cycles after the block's first pop, overlapping the next block.
Registers: ex0-7 A, ex8-11 B conveyor, dm0/dm1 + dm2/dm3 accumulator pairs, r3 = 780 (bfp mmul config),
r24/r25 FIFO positions, p0/p1 FIFO pointers (1-bit field), p3 load cursor, p4 store cursor, p5 = pB.
"""
RB, NBLK, KS, II = 2, 6, 8, 26   # II 26: same-accumulator macs 3 cycles apart (at 2 the core loses every other mac)
GAP = 1                           # the pair (acc0, acc1) of macs, then one idle vec cycle
AREG = [0, 1, 2, 3, 4, 6, 8, 10]   # A panel registers (Peano's own choice for the lf0 pops)
BREG = [5, 7, 9, 11]               # B conveyor (Peano's lf1 pop targets)
Q = ["bmll", "bmlh", "bmhl", "bmhh"]


def main(out):
    sched = {}

    def put(c, slot, text):
        sched.setdefault(c, [])
        assert all(s != slot for s, _ in sched[c]), f"slot {slot} double-booked at cycle {c}: {text} vs {sched[c]}"
        sched[c].append((slot, text))

    def acc_loads(c, pair):
        for i in range(8):
            put(c + i, "lda", f"vlda {Q[i % 4]}{2 * pair + i // 4}, [p3], #64")

    def acc_loads_window(lo, hi_acc0, hi_acc1, pair):
        """8 quarter loads in memory order in free lda cycles >= lo, the latest placement with quarter i landing (c + 7)
        no later than the first read of its accumulator (hi_acc0 for i < 4, hi_acc1 for i >= 4)."""
        hi = [hi_acc0 - 7] * 4 + [hi_acc1 - 7] * 4
        chosen = []
        c = hi[7]
        for i in range(7, -1, -1):   # place from the last quarter backwards, as late as allowed
            c = min(c, hi[i])
            while any(sl == "lda" for sl, _ in sched.get(c, [])):
                c -= 1
            assert c >= lo, f"no lda slot for quarter {i} of pair {pair} in [{lo}, {hi[i]}]"
            chosen.append(c); c -= 1
        for i, c in enumerate(reversed(chosen)):
            put(c, "lda", f"vlda {Q[i % 4]}{2 * pair + i // 4}, [p3], #64")

    def acc_stores(c, pair):
        for i in range(8):
            put(c + i, "st", f"vst {Q[i % 4]}{2 * pair + i // 4}, [p4], #64")

    # prologue: config, cursors, block 0's accumulators, row block 0's A panels, block 1's accumulators
    put(0, "lda", "mova r3, #780")
    put(0, "st", "movs p3, p2")
    put(0, "mv", "mov r24, #0")
    put(1, "mv", "mov r25, #0")
    put(1, "st", "movs p4, p2")
    put(2, "st", "movs p5, p1")
    acc_loads(3, 0)                                     # p3 was written at cycle 0 (pointer moves: ~2 cycles); lands 10..17
    put(11, "lda", "vlda.fill.512 [p0, lf0, r24]")
    for q in range(KS):
        put(12 + q, "lda", f"vlda.pop.576 ex{AREG[q]}, [p0, lf0, r24]")   # lands 20 + q
    acc_loads(20, 1)                                    # block 1 (first mac at s1 + 8 = 46), lands 27..34
    starts = [12 + II * b for b in range(NBLK)]         # row block 0: first B pop of each column block
    s5 = starts[-1]
    s6 = s5 + 16 + 8 * GAP + 4                          # row block 1: B stream re-init (r25, p1) after row block 0's last pop, then its fill
    starts += [s6 + II * b for b in range(NBLK)]
    put(11, "ldb", "vldb.fill.512 [p1, lf1, r25]")
    last_mac = {}
    for n, s in enumerate(starts):
        pair = n % 2
        for j in range(16):
            jp = j + (1 if j >= 8 else 0)
            tp = s + jp + jp // 2 * GAP                 # pops follow the mac pattern so the conveyor is read at landing
            if j == 8:
                put(tp - 1, "ldb", "vldb.fill.512 [p1, lf1, r25]")
            put(tp, "ldb", f"vldb.pop.576 ex{BREG[j % 4]}, [p1, lf1, r25]")
            acc = f"dm{2 * pair + j % 2}"
            tm = tp + 8                                 # pairs of macs with an idle vec cycle between pairs (chain distance 3)
            put(tm, "vec", f"vmac.f {acc}, {acc}, ex{AREG[j // 2]}, ex{BREG[j % 4]}, r3")
            last_mac[n] = tm
        if n not in (NBLK - 1, 2 * NBLK - 1):
            put(s + II - 1, "ldb", "vldb.fill.512 [p1, lf1, r25]")  # next column block's first fill
        acc_stores(last_mac[n] + 6, pair)
        if n >= 1:  # the next block's accumulators: after this block (same pair as n+1 = n-1) ... see below
            pass
    # row block 1: B stream back to pB, A panels popped as row block 0's last block releases the registers:
    # A_q of row block 1 goes to ex_q (q < 8); ex_q is last read by row block 0 at s5 + 9 + 2q (+1 for q >= 4),
    # so its pop (landing +8) must issue at or after s5 + 2q + 2 (+1); lda cycles 95..102 hold block 5's loads
    for n, s in enumerate(starts):
        if n < 2:
            continue
        lo = last_mac[n - 2] + 6 + 1 + 1 - 7          # quarter 0: lands >= store_0 read + 1 (later quarters are later still)
        first0 = s + 8 + 3                              # acc0's first mac reads the accumulator at stage 3 (measured; the tables say 4)
        first1 = s + 9 + 3
        acc_loads_window(lo, first0, first1, n % 2)
    # row block 1's A panels: pop ex_q after row block 0's last read of it (mac j = 2q+1 of block 5) and before
    # row block 1's first use (mac j = 2q of block 6), in a free lda cycle
    def last_read(q):
        j = 2 * q + 1; jp = j + (1 if j >= 8 else 0)
        return s5 + jp + 8 + jp // 2 * GAP
    def first_use(q):
        j = 2 * q; jp = j + (1 if j >= 8 else 0)
        return s6 + jp + 8 + jp // 2 * GAP
    fill_c = last_read(0) - 7 - 1
    while any(sl == "lda" for sl, _ in sched.get(fill_c, [])):
        fill_c += 1
    put(fill_c - 1, "mv", "mov r24, #0")
    put(fill_c, "lda", "vlda.fill.512 [p0, lf0, r24]")
    c = fill_c + 1
    for q in range(KS):
        c = max(c, last_read(q) - 7)
        while any(sl == "lda" for sl, _ in sched.get(c, [])):
            c += 1
        assert c <= first_use(q) - 8, f"A_{q} of row block 1 cannot be popped in time ({c} > {first_use(q) - 8})"
        put(c, "lda", f"vlda.pop.576 ex{AREG[q]}, [p0, lf0, r24]")
        c += 1
    last_pop = s5 + 16 + 8 * GAP                         # block 5's pop j = 15 (jp = 16)
    put(last_pop, "mv", "mov r25, #0")                   # written at +2: after that pop read r25 at stage 1
    put(last_pop + 1, "mv", "mov p1, p5")
    assert s6 - 1 >= last_pop + 3
    put(s6 - 1, "ldb", "vldb.fill.512 [p1, lf1, r25]")   # row block 1's first pop is at s6
    # intent check for the conveyors: a vmac at cycle c reading exN must see the pop of exN issued at c - 8 and no
    # pop of exN issued in (c - 8, c - 8 + 8) i.e. landing at or before c
    pops = {}
    for c, ops in sched.items():
        for sl, t in ops:
            if ".pop.576" in t:
                pops.setdefault(t.split()[1].rstrip(","), []).append(c)
    for c, ops in sched.items():
        for sl, t in ops:
            if t.startswith("vmac.f"):
                for r in t.split(",")[2:4]:
                    r = r.strip()
                    if r.startswith("ex") and int(r[2:]) in BREG:
                        assert c - 8 in pops[r], f"{t} at {c}: no pop of {r} at {c - 8}"
                        assert not any(c - 8 < pc <= c - 8 for pc in pops[r]), "conveyor overrun"
                        assert not any(pc + 8 <= c and pc > c - 8 for pc in pops[r]), f"{t} at {c}: a newer pop of {r} lands first"
    last = max(sched)
    put(last + 1, "alu", "ret lr")
    lines = ["\t.text", "\t.globl\tmm_tile_asm", "\t.p2align\t4", "\t.type\tmm_tile_asm,@function", "mm_tile_asm:"]
    for c in range(0, last + 2):
        ops = sched.get(c, [])
        lines.append("\tnop" if not ops else "\t" + ";\t\t".join(t for _, t in ops))
    for _ in range(5):
        lines.append("\tnop")
    lines.append(".Lfunc_end_mm:")
    lines.append("\t.size\tmm_tile_asm, .Lfunc_end_mm-mm_tile_asm")
    open(out, "w").write("\n".join(lines) + "\n")
    macs = RB * NBLK * 16
    print(f"{out}: {last + 7} bundles for {macs} MACs = {(last + 7) / macs:.2f} bundles/MAC (Peano BLK12 unroll 2: 1.59)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "gemm_blk.s")
