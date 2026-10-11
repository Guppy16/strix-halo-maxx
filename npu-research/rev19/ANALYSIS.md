# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 19, 2026-10-11, research pass 19. State: full12r22z (30.78 ms sweep at d12, quiet, MEASURED); main round 23 on MSPLIT-P
(down_b / mlp3_b in two row passes: -0.16..-0.21) and the SSM glue. Rev 18's ledger stands; this revision is about KTL. Rev 18 in rev18/.

## 0. Scorecard
- Rev 18: TXT18 hit (EW text 16,272 -> 15,824, and -0.07 ms); NTWIN passed (norms start 30-50 us earlier, walls unchanged; upper
  bound -0.92); 3A hit (low). KTL (rev 17 rank 1) hung in situ, also as one call per k-step: explained here.

## 1. KTL: why it hung (MEASURED)
- acq.cond / rel.cond take their condition from r26 and test bit 0 only (RELC: r26 = 2 no release, 3 release).
- KTL's masks were flag x counter: an A acquire at an even counter was skipped, every release ran; with an even trip count (18 per
  k-step) all nine acquires went. Lock counts drifted until a DMA or the core stalled.
- The lock-free twin (each lock op an add of its condition) shows the whole lock traffic of a call without blocking: the old kernel
  "acquired" a sum of 90 per call (the counter values), the fixed one exactly 9 / 9 A and nks - 1 / nks B per tile.
- Fixed with boolean masks (nez + and), re-solved at R 70 (same II, 0 stalls), bit-identical against E7 R 68 in the lock-free
  bench (1 and 3 k-steps, per call and per tile) and in KTLP-D with real locks and DMA-fed A / B.
- Not the cause: scalar mul / add latency in a ZOL body (MULLAT: as the .td), lda.2d (main's probe), records, caller registers.

## 2. Prices after the fix (ms, from 30.78)
| rank | item | price |
|---|---|---|
| 1 | KTL r70b | -0.50..-0.70 |
| 2 | MSPLIT-P (main) | -0.35..-0.6 (NTWIN bound -0.92) |
| 3 | SSM glue | -0.25..-0.45 |
| 4 | E7'' + KTL R 66-68 | -0.15..-0.3 |
| 5 | MSPLIT-C | -0.15..-0.3 |
| 6 | norm3 LN glue | -0.1..-0.18 |
| 7 | OPPRE-W -> lag | -0.04..-0.10 |
Not worth: attvs II 18 -> 11 (<= -0.02: 6 iterations per call), S3 (-0.01..-0.04).
Ceiling ~28.8 (EXTRAPOLATED); floor ~27.0 +- 0.7 (rev 18 ledger).

## 3. Method
- A modulo kernel's lock behaviour can be tested without risk: in-place adds instead of lock ops (same slot, same bundle), logged
  after the loop. A generated kernel and its record header must come from the same schedule.
- Replays must model the hardware's lock-condition semantics (bit 0), not the intent (non-zero).
