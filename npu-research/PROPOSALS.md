# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 12, 2026-10-10 (night), after round 16 (`full12r16z`, 39.94 ms at depth 12; RESULTS_FROM_MAIN round 16). Revision 11
is in `rev11/`, older ones in `rev10/` .. `rev1/`. The analysis behind every number is ANALYSIS rev 12 (the coordinator writes it
from research pass 12's hand-back). Prices are deltas against r16z at depth 12, Turbo, 1.81 GHz.

**Method (rev 12).** Two tools now price everything:
- `tools/tcall.py` (T-CALL): whole-call static cycles from the deployed elf's `-z` disassembly: CFG natural loops (rotated loops
  included), ZOL trip counts, call / ret delay slots, callees recursively, per-mode branch pinning, segment costs, and a call
  census per step. Calibrated: ggk 282 / bench 290, ssq 674 / 681, convk 220 / 231, reck 263 / 272, n1k 134 / 139 (bench minus
  ~20 of stamps); r15h norm row blocks 13,070 (LN) / 10,470 (RMS) vs the r15h timeline 13,760 / 10,760; r15h -> r15i -0.54
  static vs -0.53 measured; N1K -0.32 static (-0.24 through ewio) vs -0.16..-0.28.
- `tools/ewio.py`: the EW core's I/O model (slot rings, lock protocol, DMA rates; event-driven per 768 B slot) on top of the
  tcall compute. Fitted on four measurements at once: **R_in = 6.0 B/cycle (5.75-6.25), R_out = 2.35 B/cycle (2.3-2.45)** per
  column (score table `r17/ewio_fit.out`). It reproduces r15i -0.50 (meas -0.53), NRMK -0.44 (meas -0.37..-0.42, static -1.23),
  N1K -0.24 (meas -0.16..-0.28), r15h's n_in invariance (LN n3 - n2 = 1 us) and NPROF's in_need wait (51 per unit vs 45-65).
Bars: the fit range of R_in / R_out plus the static min..max. Every price is a whole call through both tools.

**Headlines.**
- **The norms are output-port bound now, not input-bound.** R_out ~2.35 B/cycle (4.3 GB/s per column, core MM2S -> shim) is the
  binding rate: a norm op writes 216 x 1,600 B = 81 us at that rate; RMS ops take ~89-99 us for 68 us of compute, LN 94-105 for
  81. Waits per unit at r16z (model): in_need 97 on n_in = 3 ops / 7 on n_in = 2; out_begin 147 (RMS) / 90 (LN) - pre-registered
  for an NPROF build with out_begin stamped. NRMK's missing 0.8 ms and N1K's "miss" are both this (N1K was mis-priced by rev 11
  from the loop body; its whole-call static price -0.26..-0.32 is a hit).
- **Norm passes are serialised against small rings**: pass 1 consumes 12-36 KB through a 15 KB input ring while the output DMA
  idles; pass 2 bursts 19.2 KB into a 9 KB output ring while the input DMA idles. **NPIPE** (pass 2 of row block rb interleaved
  unit by unit with pass 1 of rb + 1) keeps both DMAs busy: **-0.55 ms (-0.54..-0.56)**. After it every norm op sits within ~5 us
  of the output floor, so further norm compute levers are worth ~0 (NG1 / NG2 / NS retired to parked).
- **SSM conv gap explained**: 440 in situ = P[] pointers 58 (through the stack) + the straddle copy of a part that crosses the
  15 KB ring end (in_copy, ~250 when taken, taken ~47 % of chunks because 10,752 B of head params leave g_in_b at 512 mod 1 KB)
  + convk 220 + PH stamp ~30. **AL1** (pad every op's params to a 1 KB multiple) removes the straddles and most of cp_x's
  3-call fallback: **-0.21 +- 0.05 ms**.
- **X1 full rotation designed** (mod 4, single-TAP fills, no memory growth): **-0.26 ms after AL1** (-0.34 alone), SSM output
  6 KB -> 3 KB per chunk. It also lifts the SSM's output floor: at R_out 2.35 a 6 KB chunk needs 2,614 cycles of drain against
  ~2,950 of compute today, so **any SSM compute lever beyond ~350 cycles per chunk needs X1 first**.
- **Twins with wrong outputs overstate (new rule, inferred)**: twin-4 removed 176 cycles per unit of compute (1.03 ms) but saved
  1.53-1.61; twin-2 removed 1.81 ms of static compute and saved 1.98. The excess is not in the norm op (ewio has no mechanism for
  it); the downstream attention's lazy-max fallback is data dependent (~0.9 ms at full rate, sim18) and garbage norm outputs
  change it. Price from per-op timeline walls of a twin, never from the step.
- **asmlint FIFO-POP9** (new rule): >8 dense 576-bit pops on one lf since its last fill = error. Fires on the first ggk and
  popprobe 6, not on probe 7 / ggk2 / r16z (asmlint_hist ALL PASS, 20 cases).
- Ceiling: **~30.0 ms** at depth 12 (34.4 without the k-step row).

## Ranking (revision 12; vs full12r16z 39.94 ms)

| rank | id | what | cost | risk | expected (depth 12) | pre-registered whole-call price |
|---|---|---|---|---|---|---|
| 1 | **NPIPE** (new) | norm_rows_k: pass 1 of rb 0 + stats; then for rb = 0..17 and q = 0..11: nrm(rb, q) [reads stage q, writes ring] then n1k(rb + 1, q) [writes stage q, same slot: no second stage]; stats of rb + 1 after its 12th step; nrm of rb 17 alone. The n1k accumulators live across 12 steps in registers today: around each nrm call save / restore them (g_nacc, 4 x 128 B: ~16 cycles per unit, priced) or keep them in registers nrm does not touch | M (C++ restructure + n1k _save / _load entries) | low (bit-identical: per-unit math and accumulation order unchanged) | **-0.55 (-0.54..-0.56)** | per op (us, model): RMS n3 99 -> 86, RMS n2 89 -> 84, LN n3 105 -> 87, LN n2 94 -> 86; NPROF in_need per unit 51 -> ~12, out_begin (new stamp) RMS 147 -> ~96, LN 90 -> ~5 |
| 2 | **S2** (rev 11) | running row-block max on attkb's stack frame (bank 0) | M | low | **-0.38 (-0.29..-0.45)** | as rev 11: attkb II 34 -> 34-36; flagged rb 1,129 -> ~650. CP constraint below |
| 3 | **AL1** (new) | pad every EW op's params fill to a 1 KB multiple (SSM 10,752 -> 11,264 B per head; RMS 2,304 -> 3,072; LN 3,840 -> 4,096; plan8 + pack8 + ew8's SSM_PARAM_B / header): g_in_b stays 1 KB aligned, no 1 KB part ever straddles the 15 KB ring end; drop the straddle path (keep an assert) | S | low (bit-identical; +98 KB of input per column per step) | **-0.21 +- 0.05** | conv phase 440 -> 310-330 (PROF); cp_x 269 -> 180-200 (parts 1..3 non-consecutive in 2 of 15 chunks instead of ~1 in 3) |
| 4 | **X1** full (mod 4, new design) | conv state by rotation: the yx GEMM writes x into R[t mod 4] (four x-history regions where x / C0 / C1 / C2 sit now); the SSM fill stays ONE TAP over [y, R0, R1, R2, R3, Hlo, Hhi] (7 streams S apart, unchanged); the kernel maps taps by rot = t mod 4 from a header word (4 copies of the SSM params in W, +4.8 MB, or a 64 B rot prefix fill); the drain becomes ONE TAP of 3 streams [Hlo, Hhi, Z] at base + 5S (strides [KSTEP16, UNIT, S, 1], sizes [12, RB, 3, XB]); out_copy of x and XONE removed; 4 slots of output per chunk; 4 runtime sequences per depth (step8 picks t mod 4, swapping the instruction bo as set_depth does) | M (plan8 / step8 / ew8; no core text growth) | medium (reset: zero R0..R3; the "first" flag as now) | **-0.26 +- 0.04 after AL1** (-0.34 alone) | cp_x phase -> 0; SSM output 6 -> 3 KB per chunk; SSM op ~426 -> ~395 us (PROF) |
| 5 | S3, V1, K1, K2 (rev 10) | attention glue after S2 | M | low | -0.73 (-0.06 / -0.21 / -0.27 / -0.19) | as rev 10 / sim18 |
| 6 | **EWO** (new, microbench first) | find why R_out is 2.35 B/cycle: one-column bench of the EW output path (core MM2S -> shim S2MM -> DDR) draining N slots of 768 B vs 1,536 B vs one 9 KB BD, and the same with the norm drain TAP (1,600 B pieces 28,800 B apart) vs contiguous | S (bench) | none | decides OSLOT | if per-slot overhead (~135 cycles per slot fits 2.35): OSLOT (1,600 B output slots) -> R_out ~3.0: -0.3 more after NPIPE (ewio at R_out 3.0: NPIPE + NG1 + NG2 -0.96 vs -0.55) |
| 7 | **NIO** (measurement) | NPROF + stamps around out_begin, per op kind (mode, n_in) | S | none | validates ewio | pre-registered above (rank 1 column, "now" values) |
| 8 | SSM compute (after X1): FSX inside SSQ, ssq's 194-cycle glue (6 stages x II 30 prologue), the P[] stack round trip (58) | M | medium (x pressure) | -0.1..-0.25 | only after X1 (output floor 2,614 -> 1,307) |
| 9 | P55, P35 + P24b, P51, P57, k-step | as rev 10 | | | -0.36, -0.98, -0.96, -0.5, -4.35 (ceiling row) | |
| parked | NG1 (pass-1 glue inline, -0.20 alone, ~0 after NPIPE), NG2 (pass-2 glue, -0.09: output-bound), NS (stats in asm, -0.31 alone, ~-0.02 after NPIPE), P60t, F1 / F2, P60h, P66 / P67, H1 / H2, P14, P59, H4 | | | | | |

Combined (cumulative from 39.94): NPIPE -0.55 (39.39) -> S2 -0.38 (39.01) -> AL1 -0.21 (38.80) -> X1 -0.26 (38.54) -> S3 + V1 +
K1 + K2 -0.73 (37.81) -> EWO / OSLOT -0.3 if the bench says per-slot overhead (37.5) -> SSM compute after X1 -0.2 (37.3) -> P55,
P35 + P24b, P51, P57 -2.8 (34.5) -> k-step 2,055 -> 1,600 -4.35: **~30.1 ms** (depth 10 ~25.3, depth 8 ~20.1 scaled by the r16z sweep
ratios 33.84 / 40.30 and 26.86 / 40.30).

## Status of the revision-11 proposals (round 16 verdicts)

| id | predicted | measured | verdict / lesson (rev 12) |
|---|---|---|---|
| SQK | -0.31..-0.35 | -0.18..-0.25 | miss: preamble + call unpriced (whole-call rule; tcall now counts it) |
| P-IN | -0.25 +- 0.06 | -0.29 | hit |
| X1b | -0.18 +- 0.03 | -0.28 | miss high: stale base (r15b's 289, the phase was 432 in r15m) |
| G1 | -0.50 +- 0.07 | -0.41 | slight miss: II 24 not 20 (CP) |
| N1 | -0.6 +- 0.25 | -0.16..-0.28 | **pricing error, not I/O**: rev 11 priced the 23-bundle C++ body against a 10-cycle asm cb; the per-unit glue (in_need / in_done / pointers / loop, 116-130 cycles) stayed and the C++ pass 1 was 273 per unit, n1k's 256. Whole-call static -0.32, through ewio -0.24: hit |
| twin NORM_TWIN=2 / 4 | (used as N1's ceiling) | 1.98 / 1.53-1.61 | overstated by 0.17 / 0.5 ms beyond their own static compute: wrong outputs change downstream data-dependent work (inferred: attention fallback) |
| T-CALL | tool | built | tools/tcall.py |
| FSX | -0.24 | not built | x pressure inside SSQ; and the SSM output floor now caps it: after X1 |
| S2, X1 full | | not built | ranks 2 / 4 |

## Answers to main's round-16 requests
1. **S2 bank-0 sharing in CP terms.** GEMM bank 0 (r16z nm): stack 0x70000-0x709FF (attkb's frame and the 768 B slot ring live
   here), sgelu_ab 0x70A00, serf_ab 0x70E80, aexp_ab 0x71300 (1,280 B), ks_tab 0x71800, acc_hi from 0x718C0. Q is NOT in bank 0
   (q_buf(t) = acc_hi + 12,288 + 3,456 t = 0x748C0.. : bank 1). The bank rule (MACHINE 2.2: a bundle costs max over banks of its
   accesses in that bank) makes this a pairing constraint, not a resource: in cpsched add per bundle `b0(bundle) <= 1` with b0 =
   [the aexp_ab LUT load (vldb.4x64 on ab)] + [slot vlda] + [slot vst]. Per II 34-35: 16 LUT bundles + 2 slot accesses = 18 of
   34 bundles; the slot load (lda) and store (st) need 2 of the 18 bundles without an ab LUT load. If the CP cannot place them
   (their dependency windows are fixed by the max chain), allow b0 = 2 at cost +1 cycle and let II absorb it (II 35-36: still
   inside rev 11's -0.29..-0.45). UNVERIFIED: whether a 4x64 LUT load counts as one bank access (the ab / cd table split implies
   the two halves go to different banks).
2. **SSM phases per chunk (static, tcall segments of r16z vs r16eP PROF; PROF adds ~30 per stamp interval)**: conv 292 (no
   straddle) .. 503 (straddle, wrapped copy) static, mean ~400 + PH ~30 = 430 vs 440: P[] 58 + straddle in_copy (~47 % of chunks)
   + convk 220 + call glue. gates 327 vs 365; sigk 365 (U 4, 8 trips) vs ~440 (not stamped, bench 431 incl. stamps); ssq 685 vs
   715; recur 281 vs 313; z 228 vs 264; rest 114 + in_done / loop 59 + in_need 28-41 vs 213 + 80; cp_h 98..199 vs 192; cp_x 147
   (one call) / ~365 (three calls when parts 1..3 are not consecutive, ~1 chunk in 3) vs 269. Nothing in the SSM is unexplained
   by more than one stamp: the chunk is compute (code) bound, ~2,950 static.
3. **Norms' I/O-bound share**: see the headline and ANALYSIS rev 12 section 2. Of r16z's ~5.1 ms of norms: static compute 3.7,
   per-op fixed ~0.4 (8 us per op: params + first-row-block latency + drain tail), I/O waits ~1.0 (out_begin ~0.7, in_need ~0.3).

## The new tools — use them for every price

### T-CALL (`tools/tcall.py`)
Host-only (python 3) on a `llvm-objdump -d -z --no-show-raw-insn` disassembly (make it in npu-iron; r17/r16z_0_{2,5}.dis are
the deployed ones):
- `--all`: every function's whole-call min / max and glue (cycles outside ZOL bodies).
- `--func F [--spec S]`: loops (natural loops incl. rotated ones, trip counts inferred or '?'), ZOLs, every call site with the
  loop nest around it and the callee's whole-call cost.
- `--seg F:0xA:0xB`: cost of the path A -> B (inside the innermost loop holding both) - phase prices.
- `--spec S --census`: roots x calls per step -> top call sites by static ms per step (glue split out). Specs:
  `{"trips": {func: {"0xHEAD": n}}, "branches": {func: {"0xBRANCH": true|false}}, "roots": {func: n}}`; `branches` pins a mode
  (LN vs RMS paths of norm_rows_k). r16z specs: `tools/tcall_r16z_{ln,rms}.json`; output `r17/tcall_r16z_census.out`.
- Limits: no stalls (locks, banks, DMA: that is ewio's job and the stamps'); conditional call sites count as always taken in the
  census (use `branches` / `--seg` for the common path); trip counts of argument-driven loops come from the spec.
Use: before pricing any change, `--func` the caller and read the glue column: the per-call cost (preamble, prologue, delay
slots, helpers) is what four rev-11 prices missed.

### ewio (`tools/ewio.py`)
`fit` (the calibration grid), `norm` / `ssm` (one op at given per-unit costs and rates), `levers` (the norm levers through the
model). Feed it tcall's per-unit costs; it says which of compute / input port / output port binds and the waits per unit.

### asmlint (rule added) — FIFO-POP9
Counts dense `.576` pops per lf between fills along asmlint's replay streams (ZOL bodies 3x). Error at the 9th. History:
`r16/hist/ggk_r16_nofill.s` fires, `ggk2_r16_good.s` does not. It also fires on popprobe 2 / 3 (16 B-stream pops after one
fill), which round 16 reported correct: **please confirm which pops probes 2 / 3 checked** (if all 16 were right, the rule needs
the restart condition those probes had).

## NPIPE — the norm row-block pipeline (rank 1)
Order per op: in params; pass 1 (rb 0) + stats(0); for rb in 0..16: for q in 0..11: { out_begin; nrm(rb, q) -> ring; out_end;
in_need; n1k(rb + 1, q) -> stage[q]; in_done }; stats(rb + 1); finally pass 2 of rb 17. The stage slot q is read by nrm(rb, q)
before n1k(rb + 1, q) overwrites it, so one 12 KB stage suffices. rr (r / rl / nb) of rb is used by all 12 nrm calls of rb and
must not be overwritten by stats(rb + 1) until they are done (it is: stats run after the 12th unit). Accumulators: n1k keeps
y4 / y5 / dm0-3 across steps; nrm_ln / nrm_rms clobber dm and x registers, so either add n1k `_save` / `_load` (4 x 128 B to
g_nacc, ~16 cycles per unit: priced) or regenerate nrm with those registers reserved. Text: ~+150 B C++. Bit-identical.
Why it works (ewio): today the output DMA idles through pass 1 + stats (~4,300 cycles per rb) and the input DMA through pass 2,
so the op takes compute + the non-overlapped part of both port floors; interleaved, the op is max(compute, 216 x 1,600 / R_out,
216 n_in KB / R_in) + ~5 us.

## AL1 — 1 KB-aligned params (rank 3)
Every op's input is params + n x 1 KB pieces; the ring is 15 x 1 KB. Today the SSM's 10,752 B (= 10.5 KB) per head and the
norms' 2,304 / 3,840 B leave g_in_b at 256 / 512 / 768 mod 1 KB, so a 1 KB SSM part straddles the ring end whenever the end falls
inside the chunk (7 of 15 positions): in_copy + cpy64 ~250 cycles. Pad the params so every fill is a 1 KB multiple. Checks: the
W layout must hold the pad (alloc_w(12 x 11,264)), ew8's SSM_PARAM_B and the header reads move with it; INPART_INLINE's
straddle test then never fires (keep it as an assert in debug builds).

## X1 full rotation (rank 4): the TAPs
Per block b, column c, area base B = areas[b] + c x 8S (S = 345,600): y at B, R0..R3 at B + S .. B + 4S, Hlo B + 5S, Hhi B + 6S,
Z B + 7S - the same 8S as today. Frame t: yx GEMM: y half as now (B), x half's C objects to B + (1 + t mod 4) S (per-q tasks:
only their base constants differ by frame); SSM fill per head q: unchanged single TAP [RB, 7, 1, XB] strides [UNIT, S, 0, 1] at
B + q KSTEP16; kernel: P[1 + ((rot - k) mod 4)] for x-k (k = 0..3, rot = t mod 4); SSM drain: [12, RB, 3, XB] strides [KSTEP16,
UNIT, S, 1] at B + 5S (Hlo, Hhi, Z); the out GEMM reads Z as now. Four .bin per depth (16 files); step8 swaps the instruction
bo by t mod 4 (DAVIS loop and the live path both). The pad / record area of the 6-slot output (PROF) moves to the 4-slot one.

## Microbenches / measurements named
1. **NIO** (no new kernel): NPROF + out_begin stamps per op kind: pre-registered waits per unit (ewio, R 6.0 / 2.35): in_need RMS
   n3 97, n2 7, LN n3 98, n2 8; out_begin RMS 147, LN 90. And a `step8.py timeline` of r16z: per-op walls RMS n3 99, n2 89, LN n3
   105, n2 94 us (+-5, + ~8 fixed).
2. **EWO**: the output-path bench (rank 6) - R_out vs slot size and drain TAP shape.
3. **Twin check**: a timeline of a twin (e.g. NORM_TWIN=4): per-op norm walls should drop by ~21 us (its static compute) and the
   attention walls should drop too (the inferred mechanism of the 0.5 ms excess).
4. Rev 11's ZOL-setup and SCALAR-PLUS1 probes: still open.
