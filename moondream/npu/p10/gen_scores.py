"""Hand-scheduled AIE2P assembly for the attention score block of one row block (phase 10).

  extern "C" void attn_scores_asm(bfp16ebs8 *q /*p0: the Q row-block group, 9 panels, 704 B*/,
                                  bfp16ebs8 *kt /*p1: 6 K groups of 704 B*/,
                                  bfloat16 *s_out /*p2: 6 x 64 bf16*/,
                                  bfloat16 *scratch /*p3: [0..64) scale (bf16 x 64), [64..128) running max in/out*/);

For each key block cb: s = bf(bf(Q K_cb^T) * scale) with the 8x8x8 bfp16 mmul accumulating sequentially over the
9 dim panels (one chain: identical arithmetic to the C++ ONE_CHAIN build), stored to s_out + cb * 64, and the
running max (scratch[64..128)) updated with aie::max.
Schedule: the Q and K panels stream through the two FIFO load units (Q: lda-slot pops on lf0/r24, K: ldb-slot pops
on lf1/r25) at one panel pair every 2 cycles - the mmul chain can accept a mac every 2 cycles (accumulator read at
stage 4, result at 6) - into 4 rotating ex registers per stream (pop latency 8 at 1 pop / 2 cycles = 4 in flight).
A key block takes 20 cycles of issue; its tail (conversions, scale multiply, stores, max) overlaps the next block's
pops. Register use: ex0-3 Q conveyor, ex4-7 K conveyor, x8:x9 (= y4) the score pair, x10:x11 (= y5) scale / max,
dm0/dm1 alternating accumulators, dm2/dm3 the scaled products, r3 = 780 (bfp mmul config), r4 = 828 (bf16 mul
config), r16 the vmax flag register, r24/r25 FIFO positions, p0/p1 the stream cursors (the pop's pointer field is 1 bit), p4 the Q base.
Latencies from AIE2PGenSchedule.td (pop dst 8, mmul dst 6 / acc 4, conv 2, vlda 7, vmax 2).
"""
NB, CB = 9, 6
PERIOD = 20


def main(out):
    sched = {}  # cycle -> list of (slot, text)

    def put(c, slot, text):
        sched.setdefault(c, [])
        assert all(s != slot for s, _ in sched[c]), f"slot {slot} double-booked at cycle {c}: {text} vs {sched[c]}"
        sched[c].append((slot, text))

    put(0, "lda", "mova r3, #780")      # bfp mmul config (the compiler's r13)
    put(0, "mv", "mov r4, #828")        # bf16 mul config (the compiler's r15)
    put(1, "lda", "mova r16, #0")       # vmax flag register (the instruction names r16)
    put(2, "st", "movs p5, p2")         # p5 = s_out + 448: store immediates are 10-bit
    put(3, "lda", "padda [p5], #448")   # padda immediates are 10-bit too
    put(1, "st", "movs p4, p0")         # keep the Q group base: the pop pointer field is 1 bit (p0 / p1 only)
    base = 6
    for cb in range(CB):
        t0 = base + cb * PERIOD
        acc = f"dm{cb % 2}"
        prod = f"dm{2 + cb % 2}"
        put(t0 - 2, "st", "movs p0, p4")          # Q cursor (p0) back to the group start; K cursor p1 advances by itself
        put(t0 - 2, "mv", "mov r24, #0")
        put(t0 - 1, "alu", "mov r25, #0")
        put(t0, "lda", "vlda.fill.512 [p0, lf0, r24]")
        put(t0, "ldb", "vldb.fill.512 [p1, lf1, r25]")
        for j in range(NB):
            tp = t0 + 2 + 2 * j
            if j == 8:  # a group is 9 panels: refill after the 8th pop
                put(tp - 1, "lda", "vlda.fill.512 [p0, lf0, r24]")
                put(tp - 1, "ldb", "vldb.fill.512 [p1, lf1, r25]")
            put(tp, "lda", f"vlda.pop.576 ex{j % 2}, [p0, lf0, r24]")       # 2-register conveyor: the value is read
            put(tp, "ldb", f"vldb.pop.576 ex{4 + j % 2}, [p1, lf1, r25]")   # exactly when it lands (pop + 8)
            tm = tp + 8
            if j == 0:
                put(tm, "vec", f"vmul.f {acc}, ex{j % 2}, ex{4 + j % 2}, r3")
            else:
                put(tm, "vec", f"vmac.f {acc}, {acc}, ex{j % 2}, ex{4 + j % 2}, r3")
        tend = t0 + 2 + 16 + 8 + 6                     # = t0 + 32: the chain's result lands
        put(tend, "st", f"vconv.bf16.fp32 x8, cml{cb % 2}")
        put(tend + 1, "st", f"vconv.bf16.fp32 x9, cmh{cb % 2}")
        put(t0 + 25, "lda", "vlda x6, [p3, #0]")       # scale pair = y3 (lands 32 / 34)
        put(t0 + 27, "lda", "vlda x7, [p3, #64]")
        put(tend + 3, "vec", f"vmul.f {prod}, y4, y3, r4")
        put(tend + 9, "st", f"vconv.bf16.fp32 x8, cml{2 + cb % 2}")
        put(tend + 10, "st", f"vconv.bf16.fp32 x9, cmh{2 + cb % 2}")
        put(t0 + 39, "lda", "vlda x10, [p3, #128]")    # running max pair (lands 46 / 48)
        put(t0 + 41, "lda", "vlda x11, [p3, #192]")
        sp_, so = ("p2", cb * 128) if cb < 4 else ("p5", cb * 128 - 448)
        put(tend + 11, "st", f"vst x8, [{sp_}, #{so}]")
        put(tend + 12, "st", f"vst x9, [{sp_}, #{so + 64}]")
        put(tend + 14, "mv", "vmax_lt.bf16 x10, r16, x10, x8")
        put(tend + 15, "mv", "vmax_lt.bf16 x11, r16, x11, x9")
        put(tend + 16, "st", "vst x10, [p3, #128]")
        put(tend + 17, "st", "vst x11, [p3, #192]")
    last = max(sched)
    put(last + 1, "alu", "ret lr")
    lines = ["\t.text", "\t.globl\tattn_scores_asm", "\t.p2align\t4", "\t.type\tattn_scores_asm,@function", "attn_scores_asm:"]
    for c in range(0, last + 2):
        ops = sched.get(c, [])
        lines.append("\tnop" if not ops else "\t" + ";\t\t".join(t for _, t in ops))
    for _ in range(5):
        lines.append("\tnop")  # ret delay slots
    lines.append(".Lfunc_end_scores:")
    lines.append("\t.size\tattn_scores_asm, .Lfunc_end_scores-attn_scores_asm")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{out}: {last + 7} bundles for {CB} key blocks ({(last + 7) / CB:.1f} per block)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "scores.s")
