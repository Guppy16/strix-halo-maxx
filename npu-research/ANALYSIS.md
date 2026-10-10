# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 12, 2026-10-10 (night). State analysed: round 16's deployed `full12r16z` (39.94 ms per depth-12 dispatch, Turbo,
gpu_busy 0, BIT-IDENTICAL to full12r9a), its elfs disassembled with `-z` (`r17/r16z_0_2.dis`, `r17/r16z_0_5.dis`), the EW elfs of
r15h / r15i / r15k3 / r16f / r16h (`r17/*_0_5.dis`), round 15–16 timelines (`npu-tapnext/build/tl_full12r15{d,h,iN1}_d12.json`),
NPROF r16hN and RESULTS 24/25. Revision 11 is in `rev11/`. New tools: `tools/tcall.py`, `tools/ewio.py`; asmlint rule FIFO-POP9.
Written by the coordinating session from research pass 12's hand-back.

## 0. Scorecard (rev 11 vs round 16)
- SQK −0.18..−0.25 vs −0.31..−0.35: miss (preamble + call unpriced). P-IN −0.29 vs −0.25: hit. X1b −0.28 vs −0.18: stale base
  (432, not 289). G1 −0.41 vs −0.50: II 24, not 20.
- **N1 −0.16..−0.28 vs −0.6: a rev-11 pricing error, not I/O.** The C++ pass 1 was 273 cycles per unit whole-call and n1k is 256
  (the 116–130 cycles of per-unit glue stayed); the stats loop went (LN 480 / RMS 384 per row block). Static −0.32, through the
  I/O model −0.24: hit.
- Twins (NORM_TWIN=2 / 4) saved 0.17 / 0.5 ms more than their own static compute (§3).

### 0.1 Method changes
- Every price is a tcall whole-call count of the deployed elf, put through ewio for EW ops. Phases are priced with `--seg`; the
  census lists every call site per step.

## 1. tcall calibration

| item | tcall static | measured |
|---|---|---|
| ggk / ssq / convk / reck / n1k | 282 / 674 / 220 / 263 / 134 | bench − ~20 stamps: 290 / 681 / 231 / 272 / 139 |
| nrm_rms / nrm_ln | 117 / 177 | 136 / 210 (19 / 33 over: pushes or bank stalls, open) |
| r15h norm row block LN / RMS | 13,070 / 10,470 | timeline 13,760 / 10,760 |
| r15h → r15i (out_begin / out_end div768) | −0.54 ms | −0.53 |
| r15i → NRMK | −1.23 static | −0.37..−0.42 (§2) |
| r16f → r16h (N1K) | −0.32 static | −0.16..−0.28 |

## 2. The EW I/O model (ewio)
Rings: input ewi 20 × 768 B (bank 3, shim packet flow), output ewo 12 × 768 B (bank 0, core MM2S → shim). A norm op is 18 row
blocks × 12 units; a unit is n_in KB in and 1,600 B out (X' 1 KB + panels 576). Pass 1 consumes the input (12 n_in KB per row
block), pass 2 produces the output (19.2 KB), in turn.

Fit over (R_in, R_out) against r15i, NRMK, N1K, r15h n_in invariance and the NPROF in_need wait: **R_in 6.0 (5.75–6.25),
R_out 2.35 (2.3–2.45) B/cycle per column** (`r17/ewio_fit.out`). Twin-1 at r15i (RMS 81 µs first→last) requires R_out ≥ 2.3.

| check | measured | model |
|---|---|---|
| r15h → r15i | −0.53 | −0.50 |
| NRMK (static −1.23) | −0.37..−0.42 | −0.44 |
| N1K | −0.16..−0.28 | −0.24 |
| r15h LN n_in 3 vs 2 | equal | 1 µs apart |
| NPROF in_need wait per unit | ~45–65 | 51 |

Per op at r16z (model; + ~8 µs fixed):

| op kind | wall | compute | in_need wait / unit | out_begin wait / unit |
|---|---|---|---|---|
| RMS, n_in 3 | 99 µs | 68 µs | 97 | 147 |
| RMS, n_in 2 | 89 µs | 68 µs | 7 | 147 |
| LN, n_in 3 | 105 µs | 81 µs | 98 | 90 |
| LN, n_in 2 | 94 µs | 81 µs | 8 | 90 |

Floors per op: output 216 × 1,600 B / 2.35 = 81 µs, input (n_in 3) 61 µs. Norms per step ≈ 3.7 static compute + 0.4 fixed +
1.0 waits (≈ 0.7 out_begin, 0.3 in_need) ≈ 5.1 ms. The norms have been output-port bound since NRMK, which is why NRMK realised
only a third of its static gain. The out_begin waits are a pre-registered prediction (NPROF never stamped out_begin; NIO).

Consequence: compute cuts in the norms are worth little now (NG1 −0.20, NG2 −0.09, NS −0.31 alone, all ~0 after NPIPE). The
lever is overlap: **NPIPE (pass 2 of rb interleaved unit by unit with pass 1 of rb+1) −0.55**, after which every op is within
~5 µs of the output floor. Beyond that only the output rate itself (EWO bench → OSLOT) or fewer output bytes help.

SSM: chunk 7 KB in / 6 KB out; input floor 1,195 cycles, output floor 2,614 vs ~2,950 compute. Not port-bound now (model waits
in_need 67, out_begin 0), but output-bound after about −350 cycles of compute unless X1 halves the output.

## 3. Twins
Twin-4 (n1k skipped) removed 176 cycles per unit (n1k 140 + pointer setup 36) = 1.03 ms of static compute and saved 1.53–1.61;
twin-2 removed 1.81 ms and saved 1.98. ewio has no mechanism by which removing compute saves more than the compute. Inferred
(unverified): wrong norm outputs change downstream data-dependent work, the attention's lazy-max fallback (~0.9 ms at full
rate in sim18). Rule: price twins from their per-op timeline walls, never the step delta.

## 4. The SSM chunk, phase by phase (tcall `--seg` on r16z vs r16eP PROF; PROF adds ~30 per stamped phase)

| phase | static | PROF | note |
|---|---|---|---|
| in_need | 28–41 | | |
| conv | 292–493 | 440 | P[] 58 via the stack + straddle in_copy 6–207 (~47 % of chunks) + convk 220 |
| gates | 327 | 365 | |
| sigk | 365 | ~440 | not stamped |
| ssq | 685 | 715 | |
| recur | 281 | 313 | |
| z | 228 | 264 | |
| x copies | 147 one call / ~365 three calls (~1 chunk in 3) | 269 | |
| h copy | 81–189 | 192 | |
| rest | 97 + 59 + 28–41 | 213 + 80 | |

**The conv gap (440 vs convk 251) is the pointer array 58 + the straddle copy + the stamp.** Cause of the straddles: head params
of 10,752 B leave g_in_b at 512 mod 1 KB, so a 1 KB part crosses the 15 KB ring end whenever the end falls inside the 7 KB chunk
(7 of 15 positions). **AL1** (pad every op's params to a 1 KB multiple: SSM 10,752 → 11,264, RMS 2,304 → 3,072, LN 3,840 →
4,096) removes them: −0.21 ± 0.05; predicted conv 440 → 310–330, cp_x 269 → 180–200.

## 5. X1 full rotation (designed)
The mod-4 x history R0..R3 sits where x / C0 / C1 / C2 are now. The yx GEMM writes x to R[t mod 4]. The fill stays one TAP
[RB, 7, 1, XB] / [UNIT, S, 0, 1]; the kernel maps taps by t mod 4 (header word: 4 W copies of the params, or a rot prefix). The
drain is one TAP [12, RB, 3, XB] / [KSTEP16, UNIT, S, 1] at base + 5S ([Hlo, Hhi, Z]). Same 8S footprint; 4 runtime sequences per
depth; output 6 → 3 KB per chunk; cp_x → 0. −0.34 alone, −0.26 after AL1; the SSM output floor drops to 1,307.

## 6. Attention after S2 (unchanged prices, sim18)
S2 −0.38 (−0.29..−0.45); then S3 −0.06, V1 −0.21, K1 −0.27, K2 −0.19. GEMM bank 0: stack (0x70000–0x709FF; attkb's frame and the
768 B slot ring) + sgelu_ab / serf_ab / aexp_ab (0x71300, 1,280 B) / ks_tab; acc_hi from 0x718C0; Q in bank 1
(q_buf(t) = 0x748C0 + 3,456 t). S2 adds a bank-0 slot load + store per iteration: a CP pairing constraint (≤ 1 bank-0 access per
bundle, counting the aexp_ab LUT load, the slot vlda and the slot vst: 18 of 34 bundles), not a resource bound. If the CP fails,
allow 2 per bundle at +1 cycle each (II 35–36 stays inside the price). Unverified: whether a vldb.4x64 LUT load counts as one
bank access.

## 7. Program memory (r16z)
GEMM 16,192 / 16,384. EW r16h/z 15,600 / 16,384 (r16f 15,088). NPIPE ~+150 B C++; AL1 and X1 ~0 (X1 removes the XONE path).

## 8. Ranking and ceiling (deltas applied to the measured 39.94)

| rank | lever | ms |
|---|---|---|
| 1 | NPIPE | −0.55 (−0.54..−0.56) |
| 2 | S2 | −0.38 (−0.29..−0.45) |
| 3 | AL1 | −0.21 ± 0.05 |
| 4 | X1 full, mod-4 rotation | −0.26 ± 0.04 after AL1 (−0.34 alone) |
| 5 | S3 + V1 + K1 + K2 | −0.73 |

Also: EWO (output rate vs slot size bench; OSLOT ≈ −0.3 after NPIPE if per-slot overhead), NIO (NPROF with out_begin stamped).
Parked: NG1 (−0.20 alone, ~0 after NPIPE), NG2 (−0.09), NS (−0.31 alone, ~−0.02 after NPIPE), FSX (after X1).

| added | ms |
|---|---|
| NPIPE | 39.39 |
| S2 | 39.01 |
| AL1, X1 | 38.54 |
| S3, V1, K1, K2 | 37.81 |
| OSLOT (if EWO says per-slot overhead), SSM compute after X1 | 37.3 |
| P55, P35 + P24b, P51, P57 | 34.5 |
| k-step 2,055 → 1,600 | **~30.1** |

Depth 10 ≈ 25.3 and depth 8 ≈ 20.1 (r16z sweep ratios 33.84 / 40.30, 26.86 / 40.30).
