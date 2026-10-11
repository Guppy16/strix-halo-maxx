# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 21, 2026-10-11, research pass 21: the state at the pause of the NPU work. State: full12r24z, 28.43 ms at d12 (quiet sweep, MEASURED; r25z rebuilt byte-identical, 28.42); DAVIS d12 AJ 70.84 (OA 92.57), bit-identical. Earlier revisions: `git log --follow -p tapnext/research/ANALYSIS.md`, or `git show bbff6a1:npu-research/revN/<file>`.

## 0. Scorecard
- Rev 20 ranked SSPM-B first and priced the SSM glue at ~0 ("span read-capped"). Main's SSMIN and this pass's SSMCUT refuted that pricing (DEAD_ENDS M1).
- Round 24 kept STATS64 (−0.10), OPPRE2 v2 (−0.15), SSMPRE (−0.07) and CPPRE (−0.09): −0.48 vs r23z.

## 1. The SSM: compute-bound above an input floor (MEASURED)
- SSMCUT, per chunk (mean of 12 blocks × 8 columns):
  - r25s (r24z + SSMIN): chunk 2,289, in_need 71.
  - Twin r25g (sigk + ssq removed, ~1,000 cycles): chunk 2,034, in_need 529.
- Today's SSM is compute-bound by ~255 cycles per chunk. Its input floor is ~2,030 cycles per chunk: 7,851 B per column per chunk at 3.86 B/cycle, which is one shim MM2S channel and also ~56 GB/s over 8 columns (the read cap). The two limits coincide.
- Prices:
  - SSM glue: −0.15..−0.25 ms.
  - Further compute (FSX): ≤ 70 cycles per chunk left, until bytes drop.
  - SSPM-B (params read once instead of by all 8 columns, 683 of 7,851 B per chunk) + KPAD: they lower the floor to ~1,870. Worth −0.05..−0.15, only together with compute cuts.
- Twin end spread 0.5-0.9 µs vs 1.6-2.4: input-paced columns finish together.

## 2. The ledger (rev 20's, less round 24)
- Floor ~25.2 ± 0.9 ms (EXTRAPOLATED).
- Now minus floor: k-step DMA collisions ~1.2 ms per step (L1 capacity), EW → GEMM first tiles ~0.6, the SSM ~0.3, out's tail 0.3, attention 0-0.3.

## 3. EW L1 (r24z elf)
- Layout:
  - stack 0x70000 + 4,864 B (deepest path 1,792 B; asm frames: 64 B in ggk / convk / nrm_ln only);
  - ewp 1,152 and ewo 9,216;
  - .aie.bank0 0x73B80;
  - data 0x74000-0x7AC00 (arena 22,016 B);
  - .aie.bank2 0x7AC00;
  - input ring 0x7C000 (20 × 768);
  - .aie.bank3 0x7FC00.
- Space for the glue:
  - stack → 0x800: +2,816 B;
  - free section tails: 1,472 + 960 + 128 B;
  - sharing the norm-only buffers g_panels + g_nacc: +1,536 B;
  - total 5.4-6.9 KB; a 16-slot input ring would give +3 KB.

## 4. Command processor and BDs
- A post-wait shim fill costs ~0.55 µs (main), so 8 columns skew by ~4.4 µs per wait. C8 is revived; CPPRE / SSMPRE used this.
- Shim BD pressure at out_proj came from fills held by long-lived groups.
  - P0FREE (main): peak 20 → 16.
  - P1FREE frees one more per part wait.
  - Rule: a fill belongs in the earliest drain group that proves it was consumed.

## 5. The CPU-load clock drop (MEASURED)
- 12 threads per load type:
  - DRAM copy: 1.808 GHz;
  - AVX-512 matmul: 1.408;
  - AVX2 matmul: 1.408;
  - L1 matmul: 1.584;
  - idle: 1.809.
- Round 23: 12 spin threads at 99 W and 6 matmul threads keep 1.81.
- Many FP-compute threads with the package at ~100 W slow the NPU ~22 %; DRAM traffic, vector width and scalar load do not. Cap BLAS / OpenMP threads (≤ 6) beside the NPU.

## 6. Queue at pause (ms, from 28.43)
| rank | item | price |
|---|---|---|
| 1 | SSM glue (+ EWSTACK for L1) | −0.15..−0.25 |
| 2 | MSPLIT-C up_a / qkv / mlp0 | −0.05..−0.10 / −0.03..−0.08 each |
| 3 | SSPM-B + KPAD + FSX after 1 | −0.05..−0.15 together |
| 4 | P1FREE | ~0 (BD room) |
| 5 | KTL R 68 bank-clean (needs a new placement) | −0.2..−0.3 |
Ceiling ~27.7 (EXTRAPOLATED); floor ~25.2 ± 0.9.

## 7. Method
- Near a cap is not on it. A removal twin with input-wait stamps measures the distance to a floor directly; pre-register the in-between outcome too.
- A post-wait CP action costs ~0.55 µs per column; configure before the wait, push after.
