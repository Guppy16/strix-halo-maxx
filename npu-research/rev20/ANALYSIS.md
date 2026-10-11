# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 20, 2026-10-11, research pass 20. State: full12r23z (28.89 ms sweep at d12, quiet, MEASURED); main's round 24 has kept STATS64 (-0.10) and OPPRE2 v2 (-0.15) since: full12r24g ~28.64 (EXTRAPOLATED). Data: 4 r23z d12 timelines (r25/advance.py), r21/ddrbudget.py on the r23z build, plan8's SSM fills, main's TL_ONLY boundary timelines (3-4 us polls). Rev 19 in rev19/.

## 0. Scorecard
- Rev 19: KTL r70b hit above its range (-0.82 vs -0.50..-0.70). MSPLIT-P -0.25 in total, under its -0.35..-0.6. SSM glue: its gate failed (tcall ~1,930 > 1,880); this revision shows the gate no longer matters. E7'' + KTL R 66-68: not built (main's R 68 had 35 modelled conflicts). Main's own YXSSM (-0.41) and YXR (-0.38) were the round's other big items.
- This pass: no NPU use; CP-SAT at 2-4 workers. One mistake was made and corrected in the same pass: a 15:05 note to main called the SSM span channel-bound with DDR headroom (DEAD_ENDS L3).

## 1. The ledger (us per block, r23z; r25/ledger25.py)
| segment | now | floor | bound |
|---|---|---|---|
| norm0 (+ mlp3_b pass 2) | 93 | 90-93 | DDR read cap (= channel 0) |
| yx + SSM | 334 | 272-305 | DDR read cap in the SSM span; channel 0 and compute just under |
| out (tiles 1-3 + drain) | 52 | 20-30 | tile 0 follows the heads; tiles 1-3 need all |
| norm1 | 79 | 70-75 | DDR read cap |
| up_a + up_b | 391 | 334-357 | k-steps + gated epilogues |
| down_a + down_b + norm2 | 255 | 223-241 | k-steps + norm2 |
| qkv | 155 | 117-126 | k-steps + qkv epilogues |
| attention | 493 | 470-490 | compute |
| out_proj | 75 | 38-41 | k-steps; A after attention |
| norm3 | 76 | 71-76 | LN compute vs DDR |
| mlp0 | 227 | 183-194 | k-steps + gelu_pan epilogues |
| mlp3_a + mlp3_b | 176 | 145-156 | k-steps + plain epilogues |
| step | 28.97 ms | 24.5-26.3 ms | floor ~25.4 ± 0.9 (EXTRAPOLATED) |
- The per-op number is the frontier advance: how far the op moves the latest last-word time. It sums to the wall and gives an overlapped op only what it extends. yx's wall grew 94 -> 169 us but is hidden. The SSM's span grew 279 -> 288 us and starts 32 us after yx's first word.
- GEMM floors: 1,296 k-steps per block per core at 1,411 (R 68) to 1,304 (A-port) cycles, now 1,447; epilogues per tile from TPROF.
- Now minus floor:
  - k-step DMA collisions ~1.2 ms per step (capacity: no idle bank);
  - EW -> GEMM boundaries ~0.8;
  - the SSM span 0.4-0.7 (read bytes);
  - out's tail 0.3;
  - attention 0-0.3.

## 2. The SSM span is at the DDR read cap
- In its 288 us span the SSM reads 13.57 MB and the overlapping yx ~2.0 MB: 54 GB/s against the 55-59 GB/s aggregate read cap (round 19 DDRCAP). Writes add ~28 GB/s.
- Channel 0 (one shim MM2S per column, shared by the EW input and the GEMM A loads) carries 1,695,744 B per column for the SSM: 2,066 cycles per unit at 3.8 B / cycle. Static compute is 2,115 per unit (~252 us per block). In situ 2,416 per unit (2,336 before YXSSM).
- r24d is the check: the SSM started 15 us earlier and its span grew 19 us, because more of yx's reads landed in it.
- Re-priced: SSM glue alone and FSX ~0 (-0..-0.05). The lever is read bytes in the span.
  - The SSM params are read by all 8 columns from one address: 12 heads x 12,288 B x 8 = 1.18 MB per block, 7.6 % of the span's reads.
  - Read once and broadcast to the EW cores' unused S2MM 1: -0.10..-0.20 ms.
  - Each params fill also reads 1,536 B of kb_pad (147 KB per block): -0.02..-0.03.
  - After that the span's read floor (~254 us) falls to compute (~252 us), and the glue is worth -0.05..-0.12 again.
- Main's ZDROP v2 tests whether writes take read bandwidth here (DRAM turnaround).

## 3. Producer -> consumer pairs with a finer consumer unit
- Already overlapped:
  - yx -> SSM (YXSSM / YXR);
  - SSM -> out (ZCH / OUTPRE);
  - down_b / mlp3_b / out_proj -> norms (MSPLIT-P);
  - attention -> out_proj (OPPRE2 v2);
  - the same-core GEMM pairs (DAPRE, 3A, AQPRE, AKVPRE).
- Dead:
  - finer yx groups (YXH: more reads in the read-capped span);
  - out's tiles 1-3 on the heads (4 C tiles in L1);
  - out -> norm1 row passes (neutral).
- Left: the EW -> GEMM first tiles.
  - TL_ONLY gaps: norm1 -> up_a 27.9 us, norm3 -> mlp0 17.6, norm0 -> yx 16.1, norm2 -> qkv 14.2.
  - MSPLIT-C up_a -0.05..-0.10; qkv and mlp0 -0.03..-0.08 each; yx <= -0.035 (main).
- Rule: the overlap pays when producer and consumer run on different cores and the consumer's first unit needs a small part of the producer's output, unless it moves reads into a span that already sits at the read cap.

## 4. KTL R 68-69 and FSX
- KTL bank-clean at R 68-69: not found with <= 4 CP workers.
  - R 69 hard banks: 1,800 s at 4 workers, and 1,500 s at 2 workers with a scaled hint: no solution.
  - R 69 soft, 1,200 s at 2 workers: no solution at all.
  - R 68 soft repaired from main's 35-conflict solution: still 35, a plateau.
  - Bank 3 carries 63 accesses per row block (Y 27, L 18, N 18), 91-93 % of the bundles. The A pops' bank and the accumulator pieces' banks both move with the row block.
  - Worth -0.2..-0.3 (R 68) if found.
- FSX (gate_a's sigmoid inside ssq): ResMII 30 (bank 2 30, mv 29); no schedule at II 30-33 in 600-700 s per II. Worth <= -0.05 in a read-capped span: parked.

## 5. Prices (ms, from r24g ~28.64)
| rank | item | price |
|---|---|---|
| 1 | SSPM-B (params read once, broadcast) | -0.10..-0.20 |
| 2 | SSM glue after 1 | -0.05..-0.12 |
| 3 | KPAD (no params padding) | -0.02..-0.03 |
| 4 | MSPLIT-C up_a | -0.05..-0.10 |
| 5 | MSPLIT-C qkv, mlp0 | -0.03..-0.08 each |
| 6 | KTL R 68 bank-clean (if a schedule is found) | -0.2..-0.3 |
| 7 | OPPRE-W / out_proj first tile | -0.03..-0.06 |
Ceiling ~27.9 (EXTRAPOLATED); floor ~25.4 ± 0.9.

## 6. Method
- Price a bound over the binding op's own span, with the cap of the same kind (read cap for reads); window averages hide it.
- Once ops overlap, the ledger unit is the frontier advance per op (r25/advance.py); it sums to the wall.
- A compute cut in a span at a bandwidth cap is worth ~0, and a byte cut in a compute-bound span is worth ~0. Find which bound holds. When the two are close, price the pair together (SSPM-B + glue).
