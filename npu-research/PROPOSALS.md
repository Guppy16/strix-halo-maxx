# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 19 is archived in `rev19/`. This is revision 20, 2026-10-11, research pass 20, written during main's round 24.

- **Base:** full12r23z, 28.89 ms at d12 (quiet sweep, MEASURED). Main's round 24 has since kept STATS64 (-0.10) and OPPRE2 v2
  (-0.15): full12r24g, ~28.64 (EXTRAPOLATED).
- **This pass (no NPU use):**
  - the r23z ledger and floor, with the YXSSM / YXR overlap counted (`r25/advance.py`, `r25/ledger25.py`);
  - the SSM span is DDR-read-capped (main's reading, confirmed by bytes): the SSM glue and FSX are re-priced to ~0, and the
    lever is DDR read bytes inside the span, mainly the SSM params, which every column reads (8 x 1.18 MB per block);
  - FSX: a generator (`r25/gen_fsx.py`), ResMII 30, no schedule found at II 30-33; parked;
  - KTL at R 68-69 without bank conflicts: not found within 4 CP workers (four runs);
  - every producer -> consumer pair with a finer consumer unit, priced;
  - CLKLOAD noted, with a cheap separating probe (`r25/clkload2.sh`, not run).
- **CP-SAT:** 13:50-15:30, 2-4 workers, nice 10 (logged in RESULTS_FROM_MAIN).
- **Correction during the pass:** my 15:05 note to main said "channel-bound, not DDR". It compared reads + writes over the whole
  320 us window with norm0's r+w; reads inside the SSM's own span are at the measured read cap. Corrected in place at 15:20.

## Headlines (rev 20)

1. **The SSM span is at the DDR read cap** (INFERRED from plan8 bytes and the r23z timelines; main's r24d says the same).
   - In its 288 us span the SSM reads 13.57 MB and yx, which overlaps 137 of its 169 us, ~2.0 MB: 15.5 MB = 54 GB/s, against the
     aggregate read cap of 55-59 GB/s (round 19 DDRCAP). Writes add ~8.2 MB (28 GB/s).
   - Channel 0 (the SSM's 1,695,744 B per column = 2,066 cycles per unit at 3.8 B / cycle) and static compute (2,115 per unit, ~252
     us per block) sit just under the cap. In situ is 2,416 per unit (2,336 before the overlap).
   - r24d fits: starting the SSM earlier put more of yx's reads into its span, and the SSM slowed by 19 us.
2. **The SSM glue and FSX are ~0 now** (-0..-0.05 ms each): a compute cut does not move a read-capped span. Main's tcall gate
   (1,930 vs 1,880) no longer matters; the glue revives after the read bytes drop (rank 1).
3. **The SSM params are read 8 times.** Each column's fill reads the same `params_w + q x ps` (no column term): 12 heads x 12,288 B
   x 8 columns = 1.18 MB per block, 7.6 % of the span's reads. Read once and broadcast to the 8 EW cores (their S2MM 1 is unused):
   -1.03 MB, ~-19 us per block raw. Each fill is also padded: 12,288 B for 10,752 B of data (147 KB per block).
4. **Floor 25.4 ± 0.9 ms** (EXTRAPOLATED, `r25/ledger25.py`): k-steps at R 68 or the A-port floor, epilogues as now, EW ops at the
   read cap, no boundary residue, attention 5.6-5.9. The biggest unpriced part is the k-step's DMA collisions (~140 cycles per
   k-step, ~1.2 ms per step), blocked by L1 capacity.
5. **KTL R 68-69 bank-clean: not found** within 4 CP workers (R 69: no schedule in three runs; R 68 repaired from main's
   solution: still 35 conflicts, a plateau). Bank 3 carries 63 accesses per row block (Y 27, L 18, N 18): 91-93 %
   of the bundles at R 68-69. The A pops' bank (1 / 2) and the accumulator pieces' banks both move with the row block.
6. **Overlap pairs:** after YXSSM / YXR / MSPLIT-P / OPPRE2 the remaining finer-consumer pairs are the three EW -> GEMM first tiles
   (MSPLIT-C up_a / qkv / mlp0, -0.03..-0.10 each). Out's tail after the SSM (tiles 1-3) has no cheap form.

## Ranking (rev 20; base full12r24g ~28.64)

| rank | id | what | cost | price | gate / check |
|---|---|---|---|---|---|
| 1 | **SSPM-B** | the SSM params read once per head and broadcast to the 8 EW cores: one shim MM2S (a free MM2S 1 on columns 3-7) -> a circuit flow with 8 destinations, each EW core's S2MM 1 into a 2-slot params buffer (12,288 -> 10,752 B per head); the input ring then carries only the 7 activation streams | L (IRON flow + EW params ring + ew8 SSM fill path) | **-0.10..-0.20** | TL: the SSM span shrinks by ~1.03 MB / 57 GB/s per block; routing: aiecc places the broadcast flow (UNVERIFIED) |
| 2 | SSM glue after 1 | ssq2 + reck2 + ggk2 (static ~1,930 per unit) once the span's read floor (~254 us) drops under compute (~252) | M | -0.05..-0.12 on top of 1 (alone ~0) | after 1; tcall <= 1,950 |
| 3 | KPAD | the params fill without kb_pad (10,752 B, 14 slots, instead of 12,288): 147 KB per block fewer reads | S if AL1 allows | -0.02..-0.03 | AL1's alignment reason first (round 17) |
| 4 | MSPLIT-C up_a | up_a's first tile on rows 0-5 while norm1 writes 6-17 (a gated row-pass epilogue) | M | -0.05..-0.10 | TL_ONLY norm1 -> up_a 27.9 us (the largest EW -> GEMM gap) |
| 5 | MSPLIT-C qkv / mlp0 | as 4 (qkv epilogue / gelu_pan pass) | M | -0.03..-0.08 each | norm2 -> qkv 14.2, norm3 -> mlp0 17.6 us |
| 6 | KTL R 68 (bank-clean) | more CP workers in a quiet window (main's 12 found 35 conflicts in 1,800 s), or a placement whose A slots and accumulator pieces never share a bank | S if found | -0.2..-0.3 | replay 0 stalls; R 69 = half |
| 7 | OPPRE-W / out_proj first tile | after OPPRE2 v2: attn -> out_proj first 23.7 us; out_proj's first tile paced at 6.7 us | S | -0.03..-0.06 | TL_ONLY |
| — | SSM glue alone, FSX, out tiles on heads, MSPLIT out, YXH, MSPLIT-C yx, NIN2 / EW2CH | — | — | ~0 / dead | DEAD_ENDS L2-L6; main's yx price <= -0.035; a second input channel per column cannot beat the aggregate read cap (round 19 DDRCAP) |

ZDROP (main, in flight) is the right test of whether writes take read bandwidth (DRAM turnaround) inside the span: if v2 hits, the
span is r+w-bound too, and write cuts join rank 1's class.

**Ceiling:** 28.64 -> SSPM-B 28.49 -> glue 28.41 -> KPAD 28.39 -> MSPLIT-C x3 28.2 -> KTL R 68 27.95 (if found) -> OPPRE-W
**~27.9** (EXTRAPOLATED).

## 1. The r23z ledger (us per block, `r25/advance.py` over 4 r23z timelines; floors from `r25/ledger25.py`)

| segment | now | floor | bound |
|---|---|---|---|
| norm0 (+ mlp3_b pass 2) | 93.0 | 90-93 | DDR read cap (= channel 0) |
| yx + SSM | 334.4 | 272-305 | DDR read cap in the SSM span; channel 0 and compute just under |
| out (tiles 1-3 + drain) | 52.0 | 20-30 | tile 0 follows the heads; tiles 1-3 need all |
| norm1 | 78.8 | 70-75 | DDR read cap |
| up_a + up_b | 390.8 | 334-357 | k-steps + gated epilogues |
| down_a + down_b + norm2 | 254.6 | 223-241 | k-steps + norm2 |
| qkv | 154.6 | 117-126 | k-steps + qkv epilogues |
| attention | 493.4 | 470-490 | compute |
| out_proj | 75.4 | 38-41 | k-steps (A after attention: OPPRE2 v2 since) |
| norm3 | 76.1 | 71-76 | LN compute (STATS64 since) vs DDR |
| mlp0 | 226.7 | 183-194 | k-steps + gelu_pan epilogues |
| mlp3_a + mlp3_b | 176.4 | 145-156 | k-steps + plain epilogues |
| step (x 12 + final) | 28.97 ms | 24.5-26.3 ms | |

- advance = how far an op moves the frontier (the maximum last-word time so far); the advances sum to the wall. yx's wall
  (94 -> 169 us) is hidden: the SSM ends after it. The SSM's own span grew 279 -> 288 us; it starts 32 us after yx's first word.
- GEMM rows: 1,296 k-steps per block per core at 1,447 (in situ, KTL R 70) for "now", 1,411 (R 68) and 1,304 (A-port) for the
  floor; epilogues per tile from TPROF.
- What separates now from the floor: k-step DMA collisions (~1.2 ms per step; capacity), the GEMM side of the EW -> GEMM
  boundaries (~0.8 ms), the SSM span (0.4-0.7 ms: bytes), out's tail (0.3 ms), attention (0-0.3 ms).

## 2. Producer -> consumer pairs where the consumer's unit is finer than the producer's op

| pair | consumer unit | now | exposed (us / block) | lever | price (ms) |
|---|---|---|---|---|---|
| mlp3_b -> norm0 | row pass | MSPLIT-P | ~0 (norm0 at the read cap) | — | done |
| norm0 -> yx | yx's first tile, k-step 0 = every row's chunk 0 | sequential | 16 (TL_ONLY) | MSPLIT-C yx | <= -0.035 (main: pass 1 B-bound) |
| yx -> SSM | SSM head (its y, x) | YXSSM + YXR | 32 to the SSM start | finer groups (YXH) | dead: more of yx's reads in the read-capped span (r24d) |
| SSM -> out | out k-step = one head | ZCH + OUTPRE | 52 total; tiles 1-3 ~29 | tiles 1-3 on heads too | dead: 4 C tiles per core in L1 (L4) |
| out -> norm1 | norm row block | — | out's drain | MSPLIT out (MSDEFER) | neutral (main r24a) |
| norm1 -> up_a | up_a's first tile | sequential | 27.9 | MSPLIT-C up_a (gated pass) | -0.05..-0.10 |
| up_a / up_b -> down_a / down_b | down k-step k = an up tile | DAPRE / 3A (same cores) | ~0 | — | done |
| down_b -> norm2 | row pass | MSPLIT-P | ~13 left | — | done |
| norm2 -> qkv | qkv's first tile | sequential | 14.2 | MSPLIT-C qkv | -0.03..-0.06 |
| qkv -> attention | head group | AQPRE / AKVPRE (same cores) | pass 0 of group 0 | — | ~0 (compute) |
| attention -> out_proj | out_proj k-step = one head | OPPRE2 v2 (main r24g) | 23.7 | OPPRE-W, first tile pace | -0.03..-0.06 |
| out_proj -> norm3 | row pass | MSPLIT-P | ~0 | — | done |
| norm3 -> mlp0 | mlp0's first tile | sequential | 17.6 | MSPLIT-C mlp0 (gelu_pan pass) | -0.03..-0.08 |
| mlp0 -> mlp3_a / mlp3_b | k-step = an mlp0 tile | 3A / DAPRE (same cores) | ~0 | — | done |
The pattern pays where producer and consumer run on different cores and the consumer's first unit needs a small part of the
producer's output, as long as the overlap does not push more reads into a span already at the read cap (YXH's lesson). Same-core
pairs (GEMM -> GEMM, qkv -> attention) gain only DMA prefetch, which exists.

## 3. Probes

- **SSMIN** (S, gate for rank 2): EW TPROF stamps around in_need per SSM unit on r24g. Input waits that stay ~250-300 per unit at
  every head say the span is input-bound (today's reading); waits near 0 would say compute-bound and revive the glue alone.
- **CLKLOAD2** (`r25/clkload2.sh`, ~3 min, only in a window main announces): 12 threads each of vec (round 23's load), vec_l1
  (64 x 64 matmul, AVX-512 in L1), avx2 (OPENBLAS_CORETYPE=Haswell), mem (np.copyto on 64 MB arrays); logs package power, Tctl,
  mean CPU MHz and the NPU clock. mem drops the clock and vec_l1 does not: DRAM / fabric. vec_l1 drops it and mem does not: CPU
  vector current (an EDC-type limit the SMU settles by lowering the NPU). avx2 in between: width-dependent current.

## Tools (rev 20, `r25/`)

| tool | what |
|---|---|
| `advance.py` | per-op frontier advance from step8 timelines (sums to the wall; overlapped ops count only what they extend) |
| `ledger25.py` (+ `.out`) | the segment ledger: advance vs k-step / epilogue / DDR floors, the SSM span's bytes |
| `gen_fsx.py` | FSX: ssq + gate_a's sigmoid in one loop (sigmoid constants reloaded from ssq_tab rows 24-26 via p5), CP search per II, emit |
| `clkload2.sh` | the CLKLOAD2 probe above (not run) |
| `hint_r6{8,9}_scaled.json` | r70b's times scaled to R 68 / 69 (CP hints) |
| `solve_r69h.log`, `solve_r69hs.log`, `solve_r69s.log`, `solve_r68s.log`, `fsx_u2*.log` | this pass's CP runs |
