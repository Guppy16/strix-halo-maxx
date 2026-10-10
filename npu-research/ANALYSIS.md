# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 10, 2026-10-10 (night). State analysed: round 14's deployed `full12r14f` (47.31 ms per depth-12 dispatch, Turbo,
gpu_busy 0, BIT-IDENTICAL to full12r9a), the round-14 twins (incl. `full12r14cF128`, 46.15 ms) and one-core measurements
(RESULTS_FROM_MAIN round 14). Revision 9 is in `rev9/`. Tools: `tools/attsim17.py`, `tools/sim17.py` (output
`r15/sim17_r14f.out`), `r15/deal4.py`. Written by the coordinating session from research pass 10's hand-back (the harness
refused the subagent's own write of this file).

## 0. Scorecard (rev 9 predictions vs round 14)
- **P61**: measured −0.63 vs −0.33 predicted: hit; the extra is the chunk-0 rescale skip, which rev 9 did not price.
- **P63**: miss. The loop was right; sim16 carried r13's slow-path costs. The batched driver's slow path costs 1,587 cycles
  per flagged row block: first-pass rerun 393 + exact test 164 + rescale 578 + −m rows 59 + exact rerun 393.
- **P50**: the step hit (−0.9). The bench pre-registration missed by scope, not by loop: loop 87 + row-sums asm pass 21 +
  calls/glue 27 = 135 cycles per row block vs 136 measured.
- **P60**: single-TAP round robin impossible as designed (main's deal.py / deal2.py). Replaced by P60g (§3).
- **"Lockstep ×6"**: a re-read. sim16's map was too hot, and its fitted T 66 / R 300 compensated (T + 0.1R = 96 vs 222 measured).

### 0.1 Method changes
- Every per-row-block cost is now a one-core measurement; the only fitted constant is κ (share of the CPU map's flags kept
  in situ).
- Error bars: κ 0.5–0.7 for flag-driven prices; for code rewrites only the target II carries a range. No ±100 % defaults.
- Tooling trap: `llvm-objdump` without `-z` elides runs of 2-byte nops (zero bytes) as `...`; loga_p then shows 11 bundles
  instead of 23. Always disassemble with `-z`.

## 1. κ calibration (map thinned uniformly in chunks 1–21, 4 seeds)

| check | κ 0.5 | κ 0.6 | κ 0.7 | raw (1.0) | measured |
|---|---|---|---|---|---|
| r14c − F128 (ms) | +1.86 | +2.24 | +2.62 | +3.77 | +2.08 .. +2.53 |
| per-block slope r14c (corr 0.95) | 1.19 | 0.98 | 0.84 | 0.59 | 1 |
| per-block slope r14f (corr 0.95) | 1.26 | 1.05 | 0.90 | 0.64 | 1 |
| in-situ slow units per core | 9.7 % | 9.9 % | 10.1 % | 10.8 % | 8.6–10.9 % |

- r13 twins rms (T free, R 578): 0.16 at κ 0.6, 0.30 raw.
- In situ about 1.36 % of row blocks in chunks 1–21 move the max (the CPU map says 2.26 %).
- r14a's per-block slope is 2.6: its in-situ slow path was ~1,000–1,100 cycles, not the bench's 426; this explains
  F128 − r14a (model −2.23, measured −2.56).
- The model sits a constant 38 µs per block below the timeline (663 vs 701).

## 2. Lockstep decomposition (κ 0.6, r14f)
- Own cycles of the flagged row blocks: 0.22 ms. Removing all of them: −2.30 ms.
- Columns served independently: −1.29 ⇒ multicast lockstep ≈ 1.3 ms, head-group barrier ≈ 0.8 ms.
- Hence dealing beats decoupling:

| lever | ms |
|---|---|
| round robin, 1 / 2 / 3 / 6-rb pieces | −1.41 / −1.29 / −1.13 / −0.66 |
| decoupled columns, D 8 / D 24 | −1.29 / −1.35 |
| deal + decoupled | −1.66 |
| floor (no slow path) | −2.30 |

- Memtile depth still has no effect in the model: Q1 (the l2-depth 4→8 gain) stays open, low priority.

## 3. Feasibility
- The A slot is loaded in per-head 10,368 B chunks (A_RT BdIteration, one lock per chunk), so each gather must be one
  head-major 4-dim TAP: [12 heads][n1][n2][1,728 B]. Multi-TAP or BD chains break the chunk locks.
- Memtiles 0–2 use all 6 S2MM channels: per-column K/V queues are infeasible.

| lever | κ 0.5 / 0.6 / 0.7 (ms) | note |
|---|---|---|
| P60g | −0.87 / −1.08 / −1.30 | −1.11 with the clustering moved |
| other P60g variants (pair {0,7}, {1,6}, {3,4}) | −1.05 / −1.04 / −0.94 | κ 0.6 |
| main's tile permutation (P60t) | −0.53 / −0.66 / −0.79 | fallback |
| P60t + P60g composed | −0.76 | worse than P60g alone |
| 3 / 4 L1 slots | −0.75 / −0.89 | no L1 |

### 3.1 P60g algebra
- Offsets in units of 1,728 B: natural piece (n, T, H) sits at 32n + 8T + H in QKV; dealt piece (c, p) at c·col + p·1728 in
  APAN, with p = 2t + h.
- Obstruction (main's sum-of-squares argument): if each slot position is a column permutation, the sums of c and c² force
  the column coefficient of any size-3 dimension to zero.
- Escape: G = {x + jd + Hs}. Dealt column (T, e) at x + Td + es takes pieces (x + jd + Hs, T, e) at position 2j + H.
  - Attention side: offset = base + 32d·j + 32s·H, a [3][2] family with linear destination.
  - out_proj side: offset = base + d·col·T + s·col·half, a [3][2] family with destination k = 2T + half.
- Pair {a, b}: dealt a takes half 0 of every tile of a and b; dealt b takes the halves 1.
- Valid (x, d, s) / pair: (0,3,1)+{2,5}; (1,1,3)+{0,7}; (1,2,1)+{0,7}; (0,1,5)+{3,4}; (0,2,3)+{1,6}. All verified
  single-TAP on both sides by deal4.py. deal3.py found no "position = tile" deal that works on the out_proj side.

### 3.2 TAPs (bytes; plan8 only; chosen variant (0,3,1)+{2,5})
- Attention fill:
  - c in G, (T, e) = divmod(c, 3): sizes `[12,3,2,1728]`, strides `[QKV_PER_H, 3·QKV_PER_HC, QKV_PER_HC, 1]`,
    base qkv + 3T·OBJ + e·1728.
  - c = 2 / 5: sizes `[12,2,3,1728]`, strides `[QKV_PER_H, 3·QKV_PER_HC, 3·OBJ, 1]`, base qkv + 2·QKV_PER_HC + (c==5)·1728.
- out_proj fill:
  - n in G, (j, H) = divmod(n, 3): sizes `[12,3,2,1728]`, strides `[ks, 3·col, col, 1]`, base O + (2j+H)·1728.
  - n = 2 / 5: strides `[ks, 1728, 3·col, 1]`, base O + 2·col + 3(n==5)·1728.
- Strides stay under 2^20 words; still one BD per column; expected BIT-IDENTICAL.

## 4. Slow path, statically
- Per flagged row block: FP 393 = the 6-trip rerun that rebuilds the scores the 7-slot ring overwrote; T 164 = minsize C++;
  R 578 = minsize branch loop + exp32 61; XR 393 unavoidable.
- S2 keeps the elementwise max of the 6 score blocks in 768 B in bank 3. Bank 3 has 2,688 B free but also holds the K
  objects: the CP schedule must model the bank.
- Prices after P60g (κ 0.5 / 0.6 / 0.7): S1 −0.31 / −0.36 / −0.40; S2 −0.32 / −0.38 / −0.43; S3 −0.05.
  Without P60g: S1 −0.60, S1 + S2 −1.32.
- Pre-registered attention wall per block, blocks 1–11 (µs):
  - r14f measured: 717 663 853 750 662 649 799 736 838 531 514
  - P60g: 572 589 727 637 599 574 676 629 712 525 514 (mean 614 ± 18)
  - P60g + S1 + S2: 499 550 628 556 547 527 584 551 632 520 518 (mean 556 ± 25)

## 5. FlashAttention-2/3/4 (arXiv 2307.08691, FA3, 2603.05451)
- What the papers do:
  - FA4: exp split between MUFU and a degree-3 polynomial on the FMA units for 10–25 % of a row (floor by adding
    2^23 + 2^22 in round-down mode); rescale only when the max grows by > τ = log2 256 = 8, warp-wide; a correction
    warpgroup does the O rescale off the critical path; ping-pong of two tiles between softmax and MMA; LPT scheduling.
  - FA2: normalise once at the end, split Q across workers, skip masked blocks.
  - FA3: warp-specialised producer/consumer, ping-pong, GEMM i+1 overlapped with softmax i.
- Transfer to AIE2P:
  - Conditional rescale: our lazy max is the same idea with a looser bound (8 nats = ×2,981 vs ×256), per row block.
  - Correction warpgroup ↔ P60g + S1–S3. Ping-pong ↔ attkb's 3-stage modulo schedule. FA2's three points: deployed.
  - F2 (attention exp split): attkb's resource bound is 29 (DM_ADA 29) but II 34 is register-bound; a polynomial needs
    ~6 x-register constants for ≤ −0.1 ms and is not bit-identical. Parked.
  - F1 (SSM exp split): the mirror case (vec-bound, LUT port idle); a 2^f int16 table for half the vectors gives
    −0.05 .. −0.15 ms after P29b, not bit-identical. Parked.
  - LPT as an adaptive deal (P60h): ≤ −0.33 beyond P60g. Parked.

## 6. SSM (r14f EW loops identical to r12e; per-slot op counts from slots.py; savings transfer 1:1)

| pass | deployed → target (cycles per chunk) | ms |
|---|---|---|
| P29b log_a + exp | 1,131 → 380–470 | −0.95 .. −1.08 |
| P29c recurrence | 889 → 210–300 | −0.84 .. −0.97 |
| P29d conv + z | 1,294 → 600–720 | −0.82 .. −0.99 |
| sqrt | 512 → 180–345 | −0.24 .. −0.48 |
| P52 outputs in place | 798 → 0–300 | −0.71 .. −1.14 |

- The SSM wall runs ~1,100 cycles per chunk above these static passes; EW PROF stamps on r14f (`EW_DEFS=-DPROF`,
  ewprof.py) decide whether that is ring waits or code.
- Input-port floor 1,792 cycles per chunk (7 KB at 4 B/cycle), still below compute after all five.

## 7. Program memory
- GEMM: 64 B free. S1 frees ~360 B, T3f (mm_kstep) ~200 B, S3 (k_rb exact path in asm) ~400 B; P60g needs 0 B.
- EW: 160 B free; P29b frees ~1.4 KB.

## 8. Ceiling (sim17 deltas applied to the measured 47.31)

| added | ms |
|---|---|
| attention stack (P60g, S1–S3, V1, K1, K2) | 44.77 |
| + SSM | 40.66 |
| + P55, P7-E, P35/P24b, P51, P57 | 35.56 |
| + k-step to 1,600 | **31.21** |

Depth 10 ≈ 26.4 ms, depth 8 ≈ 21.6 ms.
