# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 4, 2026-10-09 night. State analysed: round 8's `full12r8c` (67.89 ms per depth-12 dispatch, Turbo, gpu_busy 0)
with the round-8 stamps (`g11-r8p{a,b,c,d}`, the SSM and attention stamps, the mmb bank sweep). Revision 3 is in `rev3/`.
Tags as in MACHINE.md: [DOC] vendor source or the deployed binary, [MEAS] measured by us, [INF] model or arithmetic.
Cycles at **1.81 GHz** [MEAS].

New tools this revision (all in `tools/`, all run without the device):
- `bankmodel.py`: the L1 bank rule; predicts an asm schedule's stall cycles for any bank placement (exact on mmb).
- `sim11.py` (+ `r9/sim11_r8c.out`): per-block composition model on the stamps; one fitted constant.
- `p20map.py`: the conflict-free accumulator piece map for P20, checked against the rule.
- `ssmnum.py`: bit-level numpy emulation of the RG-LRU transcendental designs on the real gate inputs
  (`npu-tapnext/build/acts_clip3.npz`, all 12 blocks, 2.65 M values per gate) and the real `a_param`.

## 0. Scorecard: revision 3 against round 8

| rev-3 prediction | outcome | verdict |
|---|---|---|
| P21 rules R1 / R2: 225-232 bundles, 295 +- 8 cycles per call, 68.2 ms | R1 273 bundles, A-off-B 255; neither kept | **refuted** (§1) |
| P21 placement (A, B, C in separate banks) | A ring pinned: -1.5 ms, 337 -> 323.5 cycles per call | confirmed in kind |
| P24: C handoff 3-6 us per tile | 1.31 us + acc_zero 0.49 | **refuted**: sim10's 7.6 us t_tile held op-start waits and k-step time |
| P13a: SSM compute-bound, input wait < 0.2 ms | waits 1.3% of the op | confirmed |
| sim10: k-step 1.532 us (r7g) | sim11 from the stamps: 1.60 us in r8c, so ~1.67 in r7g | sim10 under-counted the k-step by ~9% and paid for it with t_tile |

Two method rules from these:
- A fitted per-tile or per-op constant absorbs whatever the stamps have not separated. Fix every term that has a stamp and
  fit at most one (sim11 does).
- Before predicting what a scheduling rule removes, count the slot occupancy inside the window where the conflicts sit
  (§1.2). A stall and a serialisation cost the same cycle.

## 1. P21 post-mortem

### 1.1 The rule, exactly
`tools/bankmodel.py` classifies every memory op of `mm_core_split` by its pointer (A = p0 FIFO, B = p1 FIFO, Cl = p3
loads, Cs = p4 stores) and charges each bundle max(1, max over banks of the accesses in that bank). With one constant
(13 cycles of loop + call) it reproduces **all 207 measured placements of the mmb sweep with zero error** [MEAS].

| pair in one bank | co-issued bundles in mm_core (225) | measured extra cycles per call |
|---|---|---|
| B pop / fill + C store | 42 | +42 |
| B pop / fill + C load | 36 | +36 |
| C load + C store (in-place accumulators) | 33 | +33 |
| A pop + B pop | 14 | +14 |
| A pop + C store | 3 | +3 |
| A pop + C load | 0 (both lda: never co-issued) | 0 |
| B + Cl + Cs (triple, 27 bundles) | | 84 = 42 + 36 + 33 - 27: a triple costs 2 |

So: same bundle only; no adjacent-bundle effect; loads conflict with loads as much as with stores; 16 KB granularity.
MACHINE §2.2 now states it.

### 1.2 Why the rules lost
At each of the 3 group boundaries of `mm_core` (e.g. bundles 60-80 of `mmb_split.s`), the next group's 16 accumulator
loads must issue at or before (next first MAC - 4), and the previous group's 16 stores at or after (its last MAC + 6).
The lda slot already carries all 16 loads in 16 consecutive bundles and the st slot all 16 stores; 11 bundles hold
both. R1 forbids those 11 co-issues, so the window has to grow by ~16 cycles (the scheduler cannot move them anywhere
else), and the next group's MACs move with it: 3 x 16 = 48 -> 273 bundles. The hardware charges 11 per boundary for the
same conflicts (33 per call). **A bank conflict costs exactly one cycle per extra access: the same as the serialisation
a scheduler would insert.** A rule only wins where the displaced access finds an empty slot of its unit inside its
dependency window, and in this kernel the conflicts sit exactly where the slots are full.

### 1.3 What rev 3 got wrong
- It assumed the conflicting accesses had slack ("225-232 bundles") without counting slot occupancy in the boundary
  window.
- It weighted the pairs by placement only. R2 targeted A pop + C store, which co-issues 3 times per call (3 cycles).
  The pairs that cost were C load + C store (33, every stripe) and A + B (14, on the 6 stripes whose A slot sat in B's
  bank). Placement could fix only the second, and round 8 did (-1.5 ms).
- The total it explained (44 cycles per call) was right; the remedy was wrong.

### 1.4 Consequence
In-place accumulators cost 33 cycles per call, and no schedule of this group structure avoids it. The fix is a
layout in which consecutive groups' accumulators live in different banks (NOTES 13.4). P20 needs that layout anyway
because its stripe boundaries also co-issue: §3 specifies both together.

## 2. sim11: the r8c step and its critical path

No r8c timeline exists, so sim11 composes the block from the stamps and fits one constant (`tools/sim11.py`):

    block = sum over GEMM ops [L_op + nt * K/64 * t_ks + tiles] + SSM + attention + norms   (all serial)
    t_ks  = (9 * (259 + 34 bank + 12 DMA) + 89 + d) / F;  tiles = handoff 1.31 + acc_zero 0.49 + tail 0.04 + epilogue + ring

Fixed from stamps: epilogues per kind (plain 0.64, qkv 2.41, yx 3.32, gated 7.04 per pair, gelu_pan 7.75 us), op-start
waits (27-34 us after a GEMM op, 50 for mlp3_a, ~55 after an EW op = the 180-204 us measured minus the norm itself),
ring coupling (yx 5.04, qkv 1.23 us per tile), SSM 1,115 us, attention 930 us, norms 124 / 123 / 148 / 146 us.
Fitted: **d = +65 cycles per k-step -> t_ks = 2,899 cycles = 1.60 us** (block 5,658 us = 67.89 / 12).

| part (r8c, per block) | us | ms per step | share |
|---|---|---|---|
| GEMM k-steps (1,296 x 1.60 us) | 2,076 | 24.9 | 36.7% |
| SSM (EW cores; GEMM cores idle) | 1,115 | 13.4 | 19.7% |
| attention (GEMM cores; EW cores idle) | 930 | 11.2 | 16.4% |
| norms x 4 (EW cores) | 541 | 6.5 | 9.6% |
| GEMM op starts (A slot load + sync) | 484 | 5.8 | 8.6% |
| GEMM epilogues | 308 | 3.7 | 5.4% |
| tile boundaries (handoff, acc_zero) | 149 | 1.8 | 2.6% |
| yx / qkv row coupling | 55 | 0.7 | 1.0% |

The critical path is still the whole op chain: every op waits for its predecessor. The GEMM cores idle 1.66 ms per block
(19.9 ms per step) while the SSM and the norms run. The k-step inside: static 2,420 (9 x 259 + 89), bank 306, DMA ~108,
other ~65. **Uncertainty:** d absorbs the errors of the EW figures: +-100 us per block of EW time moves t_ks by +-140
cycles. One `step8.py timeline` of r8c would let sim11 fit spans again.

## 3. P20 + conflict-free accumulator pieces: the `mm_kstep` spec

One asm entry per k-step: `mm_kstep(const void *ptab /*p2*/, const void *pB /*p1*/)`. It takes the A-ring locks itself,
walks the 9 stripes in a zero-overhead hardware loop with the stripe boundary software-pipelined, and keeps every
accumulator piece in a bank that differs from its neighbours'. IRON keeps the B acquire / release per k-step.

### 3.1 Pieces and banks (`tools/p20map.py`, checked against the rule)
A piece is one group's accumulators: (row block rb, column pair) = 8 rows x 32 columns fp32 = 1 KB = 16 quarters. A
k-step walks 36 pieces in the order stripe j = 0..8, groups (2j, 0), (2j, 1), (2j+1, 0), (2j+1, 1). At every boundary
piece t's 16 stores co-issue with piece t+1's 16 loads, and both with B pops.

Constraints: consecutive pieces in different banks; no piece in B's bank 3; the C object may hold, in its slot s
(bytes [s KB, s+1 KB)), only a piece of row block <= s, because the epilogue writes output rb over C bytes
[rb KB, +1 KB). Today's C object holds 12 KB of fp32 rows, acc_hi 24 KB.

| stripe | group (2j,0) | (2j,1) | (2j+1,0) | (2j+1,1) |
|---|---|---|---|---|
| 0 | C0 b2 | H6 b0 | C1 b2 | H20 b1 |
| 1 | C2 b2 | H16 b1 | C3 b2 | H21 b1 |
| 2 | C4 b2 | H17 b1 | C5 b2 | H9 b0/1 |
| 3 | C6 b2 | H7 b0 | C7 b2 | H22 b1 |
| 4 | C8 b2 | H18 b1 | C9 b2 | H23 b1 |
| 5 | C10 b2 | H19 b1 | C11 b2 | H8 b0 |
| 6 | H10 b1 | H0 b0 | H11 b1 | H1 b0 |
| 7 | H12 b1 | H2 b0 | H13 b1 | H3 b0 |
| 8 | H14 b1 | H4 b0 | H15 b1 | H5 b0 |

Cs = C object + s KB (0x78900 + 1024 s, all bank 2); Hh = acc_hi + h KB (0x71900 + 1024 h; H0-H8 bank 0, H9 straddles
0x74000, H10-H23 bank 1). A ring unchanged (A_BANKS=1,2,2; slots contiguous at 0x77B80 / 0x78000 / 0x78480). The A pops
of a stripe co-issue with the stores of the piece before (rb 1's pops at the (2j,1) -> (2j+1,0) boundary, the next
stripe's at the (2j+1,1) -> next boundary), so those pieces avoid the A slot's bank: X_0, X_3, X_6, Y_2, Y_5 sit in bank
0 (slot 0 is bank 1). **Residual: <= 4 cycles per k-step** (H9's last 4 quarters in bank 1 against stripe 3's A pops).
`p20map.py` prints the 36 addresses in k-step order (the table `mm_kstep` walks) and `acc_piece(rb, pair)` for the
epilogues.

### 3.2 Registers

| register | use |
|---|---|
| p0, lf0, r24 | A FIFO (lda pops / fill); r24 reset to 0 before each row block's fill |
| p1, lf1, r25 | B FIFO (ldb); p1 rewound to p5 (`mov p1, p5`; `mov r25, #0`) before each row block |
| p5 | B object base (copied from the argument) |
| p2 | table cursor over `ptab`: per stripe [A slot address, 4 piece addresses] = 5 words, 45 + 5 dummy words (a dummy last record makes the 9th iteration's next-stripe prefetch harmless) |
| p3 | accumulator load cursor (`vlda bmXX, [p3], #64`), set per group by `lda p3, [p2], #4` >= 10 cycles before its first load (mid-group: lda is free there) |
| p4 | accumulator store cursor; `mov p4, r0` just before the boundary's first store |
| r0 | the current piece's base, saved by `mov r0, p3` right after p3 lands (before it post-increments) |
| r1 | next stripe's A slot address (`lda r1, [p2], #4`); `mov p0, r1` >= 3 cycles before the next stripe's A fill |
| r3 | 780 (mmul config), set once |
| r4 = -1, r5 = 1 | lock request values |
| ex0-4, ex6, ex8, ex10 | A panels of the current row block (as gen_mm.py) |
| ex5, 7, 9, 11 | B conveyor (as gen_mm.py) |
| dm0-dm3 | the 4 accumulator chains (dm4 free: see P31) |

No callee-saved register is touched, so there is no save / restore.

### 3.3 Loop structure
- Prologue (~30 bundles): r3, r4, r5, p5, ls / le / lc = 9; stripe 0's record (A slot, first piece); `acq #53, r4`
  (aU = lock 5); stripe 0's A fill + 8 pops, B fill + pops, the first piece's 16 loads. The first MAC comes at ~cycle 30
  (today's call: 22).
- Body = one stripe, modulo-scheduled, **192 bundles = 4 x (35 MAC span + 13 boundary)**. The 4th boundary is the stripe
  boundary: stores of (2j+1, 1), loads of stripe j+1's first piece, `acq`, the next A slot's fill and rb-0 pops, and the B
  rewind. Inside: A rb-1 re-pops at the (2j,1) -> (2j+1,0) boundary (as today); `rel #52, r5` (aF = lock 4) >= 6
  cycles after the stripe's last A pop (pop issue + 5 = its memory cycle); the next record's `lda`s in mid-group.
- The 9th iteration's prefetch reads the dummy record: piece loads from a harmless scratch KB, A pops from slot memory
  that is never used. Its `acq` must not run: use `acq.cond` with r26 = (iterations left > 1), or peel the last stripe
  (+192 bundles, ~1.3 KB: avoid).
- Epilogue: the last piece's stores (last MAC + 6, 16 stores) and `ret` + 5 delay slots: ~28 bundles.
- Hardware-loop caveat: copy Peano's `movxm ls / le` + `lc` idiom from an existing .s, and check that the assembler
  accepts `acq` / `rel` / `lda` in the body's last bundles; otherwise rotate the body so they sit at its top.

### 3.4 Lock protocol
- design8: give the A ring's locks fixed ids, `Lock(tile, init=3, name="aF..", lock_id=4)` and `Lock(tile, init=0,
  name="aU..", lock_id=5)`. These are the ids IRON allocates today (input_with_addresses.mlir: B 0/1, C 2/3, aF 4, aU 5),
  so the attention path, which still acquires the ring through IRON in threes, is unchanged.
- The worker becomes: `b = in_b.acquire(1); mm_kstep(ptab, b); in_b.release(1)`.
- Per stripe the kernel does `acq #53, r4` before the stripe's first A access and `rel #52, r5` after the stripe's last
  A pop has read memory. It holds at most 2 slots of 3 (the current stripe and the next one, during the stripe boundary),
  so it cannot deadlock against the DMA.
- 9 stripes per k-step is a multiple of 3, so every GEMM op and every attention triple starts at slot 0.

### 3.5 Program memory
mm_kstep ~ 30 + 192 + 28 = 250 bundles x 6.8 B = **~1.7 KB**. It replaces mm_core (1,524 B), mm_acc (176 B) and IRON's 9
unrolled call sequences (~0.45 KB, now 1 call): GEMM text 14,848 -> ~14,450 B. The PROF build keeps fitting.

### 3.6 Epilogues
Every epilogue that reads accumulators (`acc_rows`, `rb_bf`, the qkv packing at mm8.cc:272, bias_pass's two calls,
the gated stash) goes through `acc_piece(rb, pair)`, a 36-entry L1 table, with column block cb at piece
(rb, cb / 4) + (cb % 4) x 256 B. The in-place output stays safe by the slot rule. acc_zero zeroes the same 36 KB as today.

### 3.7 Prediction (pre-registered; sim11)

| quantity | today | P20 + pieces |
|---|---|---|
| static k-step (cycles) | 2,420 (9 x 259 + 89) | 1,771 (30 + 9 x 192 - 13 + 28 + 6) + 25 of IRON |
| bank stalls per k-step | 306 | <= 4 |
| DMA, locks, B gap | ~108 + ~65 | ~108 + ~18 + ~19 |
| k-step | 2,899 = 1.60 us | **~1,940 = 1.07 us (range 1,850-2,100)** |
| step, depth 12 | 67.9 ms | **59.7 +- 1.5 ms (-8.2)** |
| g11 -DPROF, cycles per k-step (stamp around the call) | 3,352 | 1,950-2,150 |

Staged, so each stage is measurable alone:
- (a) the piece layout inside today's per-stripe call: in-situ call 283 -> ~250 (-33), **-2.6 ms**;
- (b) + one call per k-step with the locks inside, stripes still separate (no wrapper, no IRON per-call code): **-5.7 ms
  cumulative**;
- (c) + the stripe boundary pipelined: **-8.2 ms cumulative**.

If (c) lands above 2,100 cycles per k-step, the DMA term grew with the denser schedule: measure with P32.

## 4. SSM transcendentals: numerics designs

### 4.1 The chain and its floor
Per chunk (8 rows x 64 channels = 16 vectors of 32 lanes per gate): sigmoid on gate_x and gate_a, 2,541 cycles;
log_a + exp, 1,506; sqrt, 1,093. That is 5,140 of the chunk's 9,347 cycles. Per 32-lane channel vector, today's ops by
slot (MACHINE §2.3):

| function (today, nk8) | vec (MACs) | st (conversions + stores) | mv | Peano measured | floor |
|---|---|---|---|---|---|
| sigmoid32 (exp32 + recip, 2 Newton) | 22 | 14 | 6 | 79 | 22 |
| log_a + exp_a2 (exp32a, a2 from fp32 a) | 21 | 18 | 5 | 94 | 21 |
| sqrt32 (seed + 2 Newton) | 8 | 9 | 2 | 68 | 9 |
| **per channel vector (x2 sigmoid)** | **73** | **55** | **19** | **321** | **73** |

Peano runs these long dependent chains at 3-7x their slot floor.

### 4.2 Which errors matter
- **a near 1.** h = a h + mult x, h in fp32. With a = 1 - k 2^-8 (bf16 near 1), one ulp of a changes 1 - a by 1/k: the
  channel's time constant and its gain mult / (1 - a). On the real data 7.6% of values have a >= 0.95.
  - Required there: no systematic bias, and no more 1-ulp flips than today.
- **a = 1 exactly** occurs on 0.82% of values. torch-bf16 makes those channels pure integrators for the frame. A design
  must give a = 1 exactly where torch does: the threshold is |log_a| <= 2^-9.
- **mult near a2 = 1.** torch's mult = sqrt(1 - bf(exp(2 log_a))) is a coarse staircase (0, 0.0625, 0.088, ...) for
  small |log_a|: mult = 0 iff |log_a| <= 2^-10. Matching it matters for bit-level regress; against the true value both
  neighbours are equally wrong.
- **Errors that do not matter:**
  - relative errors of small a: a < 0.5 is a memory of under 2 frames, and the absolute error is what reaches h;
  - sigmoid tails. gate_a < -9.18 gives (a, mult) = (1, 0) exactly for every channel, because sp <= 1.18 (measured
    0.006-1.18 over all 12 blocks). gate_x < -10 contributes |xc| sigma <= 4.5e-5 |xc|.
- **Facts that make tables exact:**
  - log_a lies in [-9.06, 0] on the data and [-9.44, 0] in principle;
  - |log_a| > 2^-9 spans only 1,561 bf16 values;
  - log_a = RN(sigma * (-8 sp)) is bit-identical to torch's RN(RN(-8 sigma) * sp), because -8 sigma and -8 sp are exact
    in bf16. That saves one MAC and one conversion per vector.

### 4.3 Designs (errors emulated by `tools/ssmnum.py` on all 12 blocks' real gates; cycles per 32 lanes)

| id | design | floor (slot) | Peano est. | error (MEASURED by emulation) |
|---|---|---|---|---|
| S1 | sigmoid by one symmetric PWL table: T on [-10, 0) in 80 segments of 1/8 (fp32 offset, bf16 slope, minimax offsets for the rounded slope); sigma = T(-\|x\|), 1 - T for x > 0; 0 below -10. 1.25 KB per copy | 12 (mv) | ~32 | 4.5% of values 1 ulp off RNE; rms error vs exact = bf16 rounding's own (ratio 1.002); tails as §4.2 |
| S1' | as S1, 1/16 segments (160 seg, 2.5 KB per copy) | 12 | ~32 | 2.1% 1-ulp |
| A1 | a = grid table indexed by the bf16 bits of -log_a: segment = 2^s consecutive bf16 values (s = 4: 106 segments to 9.5, 1.7 KB per copy; s = 3: 3.4 KB). Fused with the log_a fold; a2 = RN((fp32 a)^2) (2 MACs on a hi/lo split), om = 1 - a2 | 11 (mv) | ~38 | a >= 0.95: **0 mismatches** (exhaustive over every bf16 log_a). s = 4: abs-error ratio 1.008 vs torch's own rounding, a2 / mult 1.4% 1-ulp; s = 3: 1.001, mult 0.18%. a = 1 agrees 100% |
| Q1 | sqrt by a table on 1.m x 2^(e & 1) in [1, 4): 32 segments (512 B per copy); exponent (e - 127) >> 1 added in int16; om = 0 -> 0 | 14 (mv) | ~32 | **bit-exact on every positive bf16** (exhaustive); today's Newton sqrt is 1 ulp off in narrow bands |
| L | log_a = RN(sigma * c), c = -8 sp per channel precomputed (g_tsp) | 1 vec + 1 st | -4 | bit-identical |
| rejected | split exp e^(x_hi) p(x_lo) with bf16 Horner partials | | | 2.9% a mismatch (the partials' bf16 rounding); fixing it costs as much as exp32a |
| rejected | uniform PWL exp in log_a | | | 1,216 segments for 0.5% (19 KB) |
| rejected | mult as one smooth table of log_a | | | cannot reproduce torch's staircase (0 vs 0.044 at \|log_a\| = 2^-10) |

**N1 end to end** (S1 on both gates, L, A1 s = 4, Q1; `ssmnum.py --only2`):
- sigma_x 4.6% 1-ulp; gated-input ratio 1.00;
- a: 9.3% mismatched overall, all at a < 0.95 (max |da| 5.9e-3); **a >= 0.95: 0.40% 1-ulp flips, bias -4e-6**
  (from S1's sigma_a, not from A1); a = 1 agreement 100%;
- mult: 2.3% mismatched, no 0-vs-nonzero flips.

**Composite designs**, per 32-lane channel vector (2 sigmoids + a + sqrt):

| id | what | cycles | per chunk | SSM op | step |
|---|---|---|---|---|---|
| today | Peano, arithmetic | 321 | 5,140 | 1,115 us | |
| **N1** | tables S1 + L + A1 + Q1 in C++ (Peano), one fused pass for a, a2, om, mult | ~138 | ~2,210 | ~765 us | **-4.2 ms** |
| N1-x | N1 but sigma_a kept arithmetic (a >= 0.95 flips stay at today's rate) | ~183 | ~2,930 | ~850 us | -3.1 ms |
| N0 | today's math, bit-identical, hand-scheduled asm (modulo-scheduled, 4-6 chains in flight) | ~95 (1.3 x floor 73) | ~1,520 | ~680 us | -5.2 ms |
| N2 | N1's tables hand-scheduled (mv-slot floor ~55) | ~70 | ~1,120 | ~600 us | -5.8 ms |

Memory: S1 1.25 + A1 1.7 + Q1 0.5 = 3.45 KB per copy. With two copies (each LUT instruction reads 2 of its 4 lanes from
each copy) that is 6.9 KB; EW has ~6 KB free. If it does not fit, single-copy one table (both pointers equal) and
expect +8 cycles per 32 lanes from that table's own bank conflicts (MACHINE §2.2).

Recommended order:
1. Q1 (bit-exact, -0.8 ms alone).
2. L + A1 fused with om (-1.3 ms, exact where it matters).
3. S1 on gate_x (-1.0 ms).
4. S1 on gate_a, last and only if DAVIS holds (-1.1 ms).
5. N0 or N2 only if the SSM is still on the path after P14.

## 5. Op-start waits: P5 and P7 with the measured waits

Measured: every GEMM op's first tile waits for its memtile A slot.
- 27-34 us after a GEMM op: the 248 KB slot at ~7.5 GB/s; 50 us for mlp3_a.
- ~55 us after an EW op: the 180-204 us measured includes the norm (124-148) and ~22 us of finish_all / reissue.
- Total: 11 ops = **484 us per block = 5.8 ms per step**.

P5, the A slot:
- **P5a (K = 768 ops load their 124 KB once, not twice into both halves):** the wait drops ~16 us on 7 ops, **-1.3 ms**,
  and saves 107 MB of DDR per step.
- **P5 full:**
  - the slot streamed in k-chunks with a lock release per chunk, so the first k-step waits ~5 us;
  - up_b, down_b and mlp3_b, whose A is ready before the previous op ends, loaded into the other half during the
    previous op (ping-pong);
  - the post-EW sync (~22 us) stays.

  **-3.9 ms.**
- After P20 the first tile outruns the load: 12 k-steps x 1.07 = 12.9 us against 16.5 us of K = 768 load, 25.7 against
  33 for K = 1536. So P5's gain shrinks by ~0.3 ms when combined.

P7, the norms (541 us per block on the path):
- rev 3's form (norm op -> 40 us) gives -4.6 ms.
- The full fold gives **-6.2 ms**, with +0.3 us of epilogue per tile:
  - the producing GEMM's epilogue (out, down_b, out_proj, mlp3_b, which write the residual stream) also writes the A
    panels of the un-normalised x and per-row partial sums;
  - the consumer computes x W' with gamma folded into W' on the host, then applies (C - mu s) / sigma + b' per row in
    its epilogue (s = column sums of W').
- **Numerics risk:** for LayerNorm sites with |mu| >> sigma, bfp16 of the un-normalised rows plus the mu s cancellation
  can lose the deviations. RMSNorm sites (no mean) only see a per-row scale, which leaves the bfp relative error
  statistics unchanged. Emulate on `acts_clip3` before building (the main agent's numh / emu tools).

Ranking by measured-input predictions (each alone):

| rank | lever | ms | cost / risk |
|---|---|---|---|
| 1 | P20 + pieces | **-8.2** | M-L, medium |
| 2 | P7 full fold | -6.2 (-4.6 conservative) | L, numerics risk on LN sites |
| 3 | SSM N1 | -4.2 | M, low (emulated) |
| 4 | P5 full | -3.9 (-3.6 after P20) | M, medium (memtile BDs / locks) |
| 5 | attention 0.93 -> 0.6 ms | -4.0 | M |
| 6 | P5a alone | -1.3 | S-M |

## 6. Ceiling (sim11)

| sequence | depth 12 |
|---|---|
| today r8c | 67.9 ms |
| + P20 | 59.7 |
| + SSM N1 | 55.5 |
| + P5 | 52.0 |
| + P7 (40 us norms) | 47.4 |
| + attention 0.6 ms | 43.4 |
| + acc_zero folded, C double-buffered, P23 | 41.3 (4-chain GEMM ceiling) |
| + 2x2 GEMM (k-step ~1,500; the A port floor ~1,364 cycles) + SSM hand-scheduled (N2) | **35.7 ms (28 fps); depth 10 29.8 ms (34 fps), depth 8 23.8** |

- Critical path at that ceiling, per block: k-steps 1,074 us, SSM 615, attention 600, epilogues 307, op starts 192,
  norms 160, tile boundaries 27. It is still a serial chain.
- Two frames in flight (P14) hide the EW side behind the GEMM side: rev 3 priced it at 1.30x fps today, ~1.15x after the
  SSM work.
- rev 3's ceiling (34.1) was close to this, but for the wrong reasons: it under-counted the k-step and over-counted
  t_tile.

## 7. What still stands from revision 3
- Compute, not DDR, bounds the step: ~22 GB/s of ~55.
- The op chain is serial: finish_all around every EW op, and a single-buffered A slot.
- The method: compose from the binary, fit only what it cannot know, pre-register, and confirm with a perturbation or a
  stamp.
