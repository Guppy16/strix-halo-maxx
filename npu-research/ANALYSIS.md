# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 12, 2026-10-10 (night). State analysed: round 16's deployed `full12r16z` (39.94 ms per depth-12 dispatch, Turbo, gpu_busy 0,
BIT-IDENTICAL to full12r9a). Disassemblies (`-z`) of r15h, r15i, r15k3, r16f, r16h, r16z in `r17/`; round 15/16 timelines
(tl_full12r15{d,h,iN1}), NPROF r16hN, PROF r16eP, round-16 twins. Revision 11 is in `rev11/`. Tools: `tools/tcall.py` (new, T-CALL),
`tools/ewio.py` (new, EW I/O model), `tools/asmlint.py` (+FIFO-POP9), `tools/asmlint_hist.py` (20 cases, ALL PASS).
Text from research pass 12's hand-back, written by the coordinating session.

## 0. Scorecard (rev 11 vs round 16)
- P-IN -0.29 vs -0.25 +- 0.06: hit. G1 -0.41 vs -0.50: slight miss (II 24). X1b -0.28 vs -0.18: stale base (432, not 289).
- SQK -0.18..-0.25 vs -0.31..-0.35: miss, preamble + call unpriced.
- N1 -0.16..-0.28 vs -0.6 +- 0.25: **pricing error**. Rev 11 priced the 23-bundle C++ body against a 10-cycle asm cb and assumed
  the per-unit glue would go; it stayed (in_need 28-41, pointer setup 36, in_done 39, loop 13). Whole-call static: pass 1 273 ->
  256 per unit, the stats loop (LN 480 / RMS 384 per row block) -> begin / end ~95: -0.32 ms static, -0.24 through ewio. Hit.
- Twins: NORM_TWIN=2 removed 1.81 ms of static compute (cb loop + stats + stats calls) and saved 1.98; NORM_TWIN=4 removed 176
  cycles per unit (n1k 140 + pointer setup 36 = 1.03 ms) and saved 1.53-1.61. The excess (0.17 / 0.5 ms) is outside the norm op's
  model; inferred (UNVERIFIED): wrong norm outputs change the data-dependent attention fallback downstream. Twins are priced from
  per-op timeline walls only.

## 1. T-CALL (tools/tcall.py) and its calibration
Whole-call static counts from the deployed elf: natural loops by dominators (rotated loops entered mid-body: the norm rb loop, the
pass-1 loop), ZOL trips from `add.nc lc` (Peano writes n - 1 and peels one; generated loops write n), call / ret delay slots,
callees recursively, branch pinning per mode, segments, a census per step. Calibration (static vs bench minus ~20 of stamps):
ggk 282 / 290, ssq 674 / 681, convk 220 / 231, reck 263 / 272, n1k 134 / 139; nrm_rms 117 / 136 and nrm_ln 177 / 210 are the
exceptions (pushes or bank stalls: 19 / 33 cycles unexplained). In situ: r15h norm row blocks LN 13,070 / RMS 10,470 static vs
13,760 / 10,760 from the r15h timeline (first -> last / 17.5 row blocks); r15h -> r15i (out_begin / out_end by div768) -0.54 static
vs -0.53 measured. Census of r16z (`r17/tcall_r16z_census.out`, static ms per step): ssq 0.97, nrm_ln 0.53, sigk 0.52, ggk 0.40,
n1k_ln 0.40, n1k_rms 0.38, reck 0.38, nrm_rms 0.34, convk 0.32; glue (outside ZOL bodies) is 40-70 % of every norm / SSM root.

## 2. The EW I/O model (tools/ewio.py)
Rings: ewi 20 x 768 B (bank 3) filled by shim -> core packet flow; ewo 12 x 768 B (bank 0) drained by core MM2S -> shim S2MM. The
protocol is ew8's (in_need waits for whole filled slots, in_done frees consumed slots, out_begin waits for free slots, out_end
hands whole slots to the DMA). Compute per step from tcall, mode-pinned. Fit over an R_in x R_out grid (r17/ewio_fit.out):

| evidence | measured | model at R_in 6.0, R_out 2.35 |
|---|---|---|
| r15h -> r15i | -0.53 ms | -0.48..-0.50 |
| r15i -> NRMK (static -1.23) | -0.37..-0.42 | -0.44 |
| r16f -> r16h, N1K (static -0.32) | -0.16..-0.28 | -0.24 |
| r15h LN n3 - n2 | ~0 us | 1 us |
| NPROF in_need wait per unit | ~45-65 | 51 |

Per op at r16z (us; compute in brackets; waits per unit): RMS n3 99 (68), in 97 / out 147; RMS n2 89 (68), 7 / 147; LN n3 105 (81),
98 / 90; LN n2 94 (81), 8 / 90; + ~8 us fixed per op (params, first row block, drain tail; from r15h's timeline). Norms ~5.1 ms
per step = compute 3.7 + fixed 0.4 + waits ~1.0. Floors per op: output 216 x 1,600 B / 2.35 = 81 us; input n3 61, n2 41 us.
Mechanism: pass 1 (all input, no output) and pass 2 (all output, no input) alternate per row block; the 15 KB input ring cannot
hold an n3 row block's 36 KB and the 9 KB output ring cannot absorb pass 2's 19.2 KB burst, so each DMA idles half the time.
NRMK made pass 2 fast enough to hit the output side: that is where 0.8 ms of its static gain went.
Levers through the model (r17/ewio_levers.out): NPIPE -0.55 (-0.54..-0.56); NG1 (pass-1 glue -80 per unit) -0.20 alone; NG2 (pass-2
glue -100) -0.09; NS (stats in asm) -0.31 alone; after NPIPE the three together add only ~-0.03 at R_out 2.35 (-0.4 at R_out 3.0):
the next norm lever is the output rate itself (EWO bench, OSLOT).
Open: why R_out is 2.35 B/cycle (= 768 B per ~327 cycles: a ~135-cycle per-slot overhead on a 4 B/cycle stream fits, as does the
drain TAP's 1,600 B pieces 28,800 B apart).

## 3. The SSM after round 16
Chunk phases (tcall segments of r16z vs PROF r16eP; PROF adds ~30 per interval): conv 292..493 static (P[] 58 + convk 220 + call
glue; + the straddle in_copy ~200-250 when taken) vs 440; gates 327 / 365; sigk 365 / ~440; ssq 685 / 715; reck 281 / 313; z 228 /
264; cp_h 81..189 / 192; cp_x 147 one call, ~365 three calls / 269; rest + in_done + in_need ~200 / 293. Static chunk ~2,950;
compute-bound with in_wait 80.
The straddle: a head's params are 10,752 B = 10.5 KB, so within a head g_in_b sits at 512 mod 1 KB and a 1 KB part straddles the
15 KB ring end whenever the end falls inside the chunk's 7 KB (7 of 15 positions): ~47 % of chunks. AL1 (all params 1 KB
multiples) removes it: conv -> ~310-330, cp_x's three-call fallback from ~1 chunk in 3 to 2 in 15: -0.21 +- 0.05 ms.
Output floor: 6 KB per chunk / 2.35 = 2,614 cycles against ~2,950 of compute; ewio puts out_begin waits at 49 per chunk at a
2,500-cycle chunk and 237 at 2,300. X1 full (3 KB per chunk) moves the floor to 1,307: it gates every SSM compute lever > ~350.

## 4. X1 full rotation (mod 4)
Layout per block / column unchanged (8S): y at B, x-history R0..R3 at B + S .. B + 4S (where x / C0 / C1 / C2 sit now), Hlo,
Hhi, Z at B + 5S .. 7S. Frame t: the yx GEMM writes x into R[t mod 4] (only its per-q drain base constants change); the SSM fill
is the same single TAP over 7 streams S apart; the kernel maps tap x-k to R[(t - k) mod 4] from a header word; the drain is one TAP
of 3 streams [Hlo, Hhi, Z] at B + 5S. No copies of x, 4 output slots per chunk, 4 .bin per depth (step8 swaps the instruction bo by
t mod 4). Price: cp_x ~240 non-PROF (~180 after AL1) + XONE glue -> 0: -0.34 alone, -0.26 after AL1, +- 0.04. DDR writes of the
SSM halve.

## 5. Attention (unchanged from rev 11 numbers)
S2 on attkb's stack: -0.38 (-0.29..-0.45); then S3 -0.06, V1 -0.21, K1 -0.27, K2 -0.19. Bank 0 holds the stack and aexp_ab
(sgelu_ab, serf_ab, ks_tab) but not Q (q_buf in bank 1): S2's slot load and store must avoid bundles that carry an ab LUT load
(bank rule), 18 of 34 bundles; a pairing constraint for cpsched, not a resource bound.

## 6. Program memory
EW r16z 15,600 / 16,384 (non-PROF; NPROF +460). NPIPE ~+150 B C++ (+ n1k save / load ~60 B); AL1 removes the straddle path (~-150);
X1 removes XONE's copy path (~-100). GEMM 16,192 / 16,384 (S2 ~40 B).

## 7. Ceiling (deltas applied to the measured 39.94)

| added | ms |
|---|---|
| NPIPE | 39.39 |
| S2 | 39.01 |
| AL1, X1 | 38.54 |
| S3, V1, K1, K2 | 37.81 |
| OSLOT (if EWO says per-slot overhead), SSM compute after X1 | ~37.3 |
| P55, P35 + P24b, P51, P57 (rev 8/9 prices) | ~34.5 |
| k-step 2,055 -> 1,600 | **~30.1** |

Depth 10 ~25.3, depth 8 ~20.1 (r16z sweep ratios).
