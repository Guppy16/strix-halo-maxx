# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 13, 2026-10-11, after round 17 (`full12r17d`, 39.16 ms at depth 12; RESULTS_FROM_MAIN round 17) and main's first
round-18 builds (`full12r18a` HRING + PTRG, A/B −0.34 vs r17d; `full12r18b` + ZK, −0.44 vs r17d: **~38.72**). Revision 12 is in `rev12/`, older ones in
`rev11/` .. `rev1/`. The analysis behind the numbers is ANALYSIS rev 13 (the coordinator writes it from research pass 13's
hand-back). Prices are deltas at depth 12, Turbo, 1.81 GHz, whole-call (tcall static + ewio / ksplan for waits and stalls).

**Method (rev 13).**
- `tools/ewio.py fit13 / levers13` (new): the EW I/O model refit on round 17's stamps. It adds in-situ per-unit costs, NPIPE
  groups of G, the static cost inside each stamp interval, and the timeline's per-kind norm walls. The rev-12 `fit` stays for history.
- `tools/ksplan.py` (new): the GEMM k-step budget. It decomposes the deployed mm_ks from the disassembly, gives the resource
  bounds of each kernel variant, and adds an in-situ DMA-collision model (UNVERIFIED) and the A-port floor.
- `tools/tcall.py`: unchanged. Used on r17d (`r18/r17d_0_{2,5}.dis`) and on main's r18a (`r18/r18a_0_5.dis`).

## Headlines
- **The norms are not output-bound. The rev-12 ewio fit was degenerate.** With the out_begin stamp the refit gives:
  - R_in **5.75 B/cycle** (5.5–6.0) and R_out **≥ 3.5** (not binding).
  - In-situ pass-2 cost **+76 (RMS) / +132 (LN) cycles per unit** above static (s2 = 4, range 3–5).
  - Net waits are the stamped intervals minus the call's own ~40–44 cycles. That gives in_need 57.5 → 17 per unit, and
    out_begin ≈ 5 both times.

  Why rev 12 got it wrong: r15i, NRMK and N1K are all compute cuts that realised less than their static price. "Pass 2 waits
  for the port" (R_out 2.35, s2 = 1) and "pass 2 runs slower in situ than static" (R_out ≥ 3.5, s2 ≈ 4) fit those three
  deltas equally well:

  | point | r15i | NRMK | N1K |
  |---|---|---|---|
  | R_out 2.35, s2 1 (rev 12) | −0.48 | −0.42 | −0.18 |
  | R_out 4.0, s2 4 (rev 13) | −0.54 | −0.45 | −0.22 |

  Rev 12 fixed s2 at the bench value by assumption, so the missing pass-2 time landed on R_out. The fifth check, NPROF
  in_need, cannot see R_out. Only a stamp on the hypothesised wait could separate the two, and round 17 added it.
- **After NPIPE (G 6) the norms are compute-bound.** In the model the input waits are ~12 per unit. FOLD (n_in 3 → 2 by
  summing D1 + D2 / D3 + D4 upstream) is worth **−0.03 ms**, so it is retired. The rev-12-parked compute levers come back:
  pass-1 glue (NG1) −0.44, pass-2 glue (NG2) −0.56, stats (NS) −0.34.
- **GEMM text: about 280 B is recoverable at once.** mm_kstep is 320 B, and ~280 B of it is the once-per-run k-step record
  builder plus its per-call cache check. Every GEMM core has the same placement (C object 0x78900, A slots 0x77B80 / 0x78000 /
  0x78480: checked on all 24 cores in input_with_addresses.mlir), so the 58 records can be data computed at build time.
  Second source: IRON lowers every worker loop counter to i64 (12 `phi i64` in opted_main_core_0_2.ll), so each loop carries
  ~6 extra scalar ops (−200..−300 B with i32 counters).
- **The k-step plan (section K) has one big lever, ROT.** Each group of 32 MACs stops for 12.75 bundles while all 4
  accumulators are stored and reloaded together: 459 cycles per k-step, the largest static item. A 5th accumulator rotating
  through the swap (dm4 is unused) removes the stop.
  - Static floor: the B bank. B gets 9 accesses per 8 MACs (8 pops + 1 fill), so II = 9 per 8 MACs and mm_ks ~1,356 static (from 1,824).
  - Price: **−3.5 ms**, or −3.9 with P35, through the UNVERIFIED collision model.
  - The A port floor is 1,304 cycles, so beyond ~1,450 the k-step needs A on two channels.
- **One hypothesis explains every in-situ excess we have** (UNVERIFIED, untested since MACHINE open question 6): a DMA
  access to a bank costs the core a cycle when the core touches that bank in the same cycle.

  | where | measured above static | DMA traffic in the bank | model |
  |---|---|---|---|
  | GEMM k-step | +231 | A + B writes, 468 accesses | 221 |
  | norm n1k step | +17 | input ring, 96 writes per unit | 20–30 |
  | norm pass 2 | fitted +76 / +132 | output ring, 50 reads per unit | 50+ |
  | SSM chunk | ~+250–365 | 224 writes + 96 reads | ~115 |

  The one-core DMAB bench decides it, and with it every layout lever.

## Ranking (revision 13; base full12r18b ≈ 38.72 ms = r17d 39.16 − 0.44 A/B)

| rank | id | what | cost | risk | expected (depth 12) | pre-registered whole-call check |
|---|---|---|---|---|---|---|
| 1 | **KSC** (new) + S3, V1, K1, K2 | k-step records as build-time data: design8 computes the 58 `g_ks_tab` words from the placement (C object, A slots from input_with_addresses.mlir; `acc_hi` from llvm-nm of the link) and emits them as an initialised table; mm_kstep becomes `mm_ks(tab, pB, piece0)` (tail jump). A post-link check (nm + mlir addresses) rebuilds once if anything moved. Then S3 / V1 / K1 / K2 fit | S (+ M for S3..K2) | low | text **−270..−290 B** (mm_kstep 320 → 32-48); 0 ms itself; S3 + V1 + K1 + K2 **−0.73** (rev 10/11 prices on r15m; ±30 %: re-price with sim18 on r17d before building) | `llvm-size` GEMM .text 16,336 → ≤ 16,064; regress bit-identical |
| 2 | **P35** (rev 8, re-priced) | B lock + k-loop inside mm_ks: the B ring with pinned locks (like the A ring's 4 / 5), one mm_ks call per tile running its k-steps; main's k-loop (0x310-0x39e, 144 B of i64 counters) and mm_kstep's per-call frame go | M | medium (hand-built B ring, A precedent) | **−0.44 ± 0.1** (gap 67 → ~16 per k-step x 15,552) ; text −80..−120 B | PROF: gap between mm_ks calls 67 → ≤ 20 (records `gkb`); k-step in mm_ks unchanged 2,055 |
| 3 | **N-IO** (new; = NG1 + NG2 done in asm) | the norm unit's I/O and pointer glue into the asm steps: n1k_step takes the lock acquire / release and the ring pointers (2D-modifier wraps as in n1p) (N-IO1); nrm_* take out_begin / out_end and the ring-wrap logic (N-IO2). Per unit today: pass 1 ~117 cycles of glue around a 140-cycle n1k call, pass 2 ~170 around nrm | M each | low-medium | **N-IO1 −0.45..−0.55, N-IO2 −0.5..−0.7** (ewio levers13 at the fit, compute-bound) | NIO2 stamps first (below): pass-2 unit RMS ~325 / LN ~449 in situ; after N-IO1 a stamp around the whole pass-1 unit reads ~170 (today 256 static + 17 in situ + waits) |
| done | **ZK** (main r18b: −0.17 A/B, hit; listed for the record) | the SSM z panels pipelined: Peano compiles the in-situ z loop as a serial 27-bundle chain per 64 lanes (zol @0x1610 in r18a, x 8: ~216 + setup), the one-core bench of the same C++ ran 97 for all 8 (it pipelined there). Use r15/gen_zk.py's kernel (bench 100) or the C++ as a noinline function with restrict parameters (the bench's form) | S | low (bit-identical; zkb bench exists) | **−0.18..−0.21** (−130..−146 per chunk x 2,592) | ew_ssm chunk static 1,979 → ~1,840; CPROF rest 2,562 → ~2,420 |
| 5 | **ROT** (new; the k-step program, section K) | mm_ks with a 5th accumulator rotating through the C swap: jobs of 8 MACs (one column block of one row block), 2-3 interleaved chains, II 9 per job (bank-3 bound: 8 B pops + 1 fill), C load of job j+2 / store of job j-2 in the free lda / st slots (always opposite pieces = different banks), A refreshed in place over 16 cycles at each row-block change | L | high | **−3.5 (−2.5..−4.0)**, with P35 −3.9; static mm_ks 1,824 → ~1,356 | E1 (CP offline) first: II 9 allocates with ex ≤ 12, dm ≤ 5 and the bank rule on the deployed placement; one-core ksb.py bit-identical, ~1,360 + collisions |
| m1 | **DMAB** (measurement, new) | one core, a loop with a known bank access per cycle (bank X), while a DMA streams into / out of bank X vs bank Y; 32 B vs 64 B DMA access granularity | S | none | decides every layout lever (GEMM ~−200 per k-step = −1.7 ms ceiling; EW pass 2 / SSM ~−0.3..−0.6) | pre-registered: slowdown ≈ (DMA bytes / 32) x core bank-busy fraction; 0 if the DMA yields |
| m2 | **KS-P38** (rev 5 P38, still open) | core MEMORY_STALL / LOCK_STALL counters around mm_ks in the PROF build | S | none | splits the 231 in-situ cycles: A waits (LOCK) vs collisions (MEMORY) | pre-registered: LOCK ≤ 30, MEMORY 150-230 per k-step |
| m3 | **NIO2** (measurement, new) | NPROF stamps around the nrm call, the whole pass-2 unit and the whole pass-1 unit (per mode) | S | none | locates the fitted pass-2 excess (+76 / +132 per unit) | pass-2 unit RMS 325 ± 40, LN 449 ± 60; nrm call RMS 136-193, LN 210-309 (bench 136 / 210) |
| 4 | NS | norm statistics per row block in asm (ln_stats 317 x 2 + rowsum8 x 4 / rms_rstd 199 x 2 per rb) | M | low | −0.34 (levers13) | |
| 6 | **CGF** (new) | convk pushes its xc output as bfp panels itself (the xc → bfp push loop + one call glue go) | M | low | −0.10..−0.13 (−70..−90 per chunk) | |
| 8 | T2 / IV32 (new) | i32 loop counters in design8's worker loops (`scf.for` on i32 bounds instead of `index`) | S-M | low (control code only) | text −200..−300 B (GEMM), EW ~−40 B; ~0 ms | `grep -c "phi i64" opted_main_core_0_2.ll` 12 → ≤ 1 |
| 9 | P55, P51, P57, P24b | as rev 8-10 (not re-derived this pass) | | | −0.36, −0.96, −0.5, −0.12 | |
| parked | FSX (−0.07 now: sigk's glue is 45), FOLD (−0.03), NKEEP (~0), NPIPE G 2 (+0.08), OSLOT / EWO (R_out not binding), AL1 (done), K-pair (needs +4.6 KB L1; ROT dominates) | | | | | |

Combined (cumulative from ~38.72, r18b; ZK done):

| step | change | ms |
|---|---|---|
| KSC + S3/V1/K1/K2 | −0.73 | 37.99 |
| P35 | −0.44 | 37.55 |
| N-IO1 + N-IO2 | −1.0 | 36.55 |
| NS | −0.34 | 36.2 |
| CGF | −0.12 | 36.1 |
| DMAB layout, EW only | −0.3 | ~35.8 |
| P55 + P51 + P57 | −1.8 | ~34.0 |
| ROT (on top of P35) | −3.5 | **~30.5** |
| B split over two banks | −1.6 | ~28.9 |

The A port floor (1,304 cycles per k-step) caps the k-step at ~1,450 without two-channel A (rev 5 §2.1).

## Status of the revision-12 proposals (round 17 verdicts)

| id | predicted | measured | verdict / lesson (rev 13) |
|---|---|---|---|
| S2 | −0.38 (−0.29..−0.45) | −0.31 | hit |
| X1 (mod 4) | −0.34 alone / −0.26 after AL1 | −0.33 | hit |
| AL1 | −0.21 ± 0.05 | −0.05..−0.15; 1 KB padding hung | miss: CPROF confirms the straddle path is gone (conv 440 → convk 280 + pointers 101), but the realised gain is half the static price (47 % straddled chunks x ~250). Unexplained; candidates: fewer straddles in situ than the ring arithmetic says, or the larger params fills (+98 KB per column per step) costing back ~0.05-0.1. A fill group must be a 768 B slot multiple (in the skill) |
| NPIPE | −0.55 (−0.54..−0.56) | G2 +0.07, G4-6 −0.05..−0.15 | **miss from a wrong bound**: the refit (above) puts the norms at compute, not the output port; the rev-13 model gives G6 −0.21 / G2 −0.13 (still ~0.1 optimistic: NPIPE's own glue) |
| NIO (stamp request) | out_begin 147 / 90 per unit | 45.0 gross ≈ 5 net | the measurement that broke the degeneracy |
| FIFO-POP9 on popprobes 2 / 3 | true positive asked | true positive (round 16 checked only lf0) | rule stands |
| EWO bench | decide OSLOT | not built | retired: R_out is not binding |

## Answers to main's round-17 requests
1. **ewio refit**: `tools/ewio.py fit13` (`r18/ewio_fit13.out`):
   - Rates: R_in 5.75 (5.5–6.0); R_out ≥ 3.5 (any value up to 8 fits equally).
   - In-situ excess: pass 2 +76 / +132 per unit (RMS / LN; s2 = 4, 3–5); pass 1 +17 (the NPROF n1k stamp).
   - Per-op fixed ~5 µs; wall rms 3.9 µs over the four kinds.
   - Reproduced:

     | quantity | model | measured |
     |---|---|---|
     | in_need (net, per unit) | 51 | ~57 |
     | out_begin (net, per unit) | 0 | ~5 |
     | NPIPE_G=2 in_need | 11 | ~17 |
     | r15i | −0.54 | −0.53 |
     | NRMK | −0.45 | −0.37..−0.42 |
     | N1K | −0.22 | −0.16..−0.28 |
     | G6 | −0.21 | −0.05..−0.15 |
     | G2 | −0.13 | +0.07 |

   - n3 − n2 walls: model 10.7 µs, timeline 15–19 µs (low by ~5–8 µs). Why the old fit mis-bound: the headline above.
   - Input-side levers (`r18/ewio_levers13.out`):

     | order | FOLD |
     |---|---|
     | at G 6 | −0.03 |
     | on the serial order | −0.05 |

     The n_in 3 ops wait ~100 cycles per unit only in the serial order, and NPIPE already hides that.
2. **GEMM program memory** (r17d elf, `tools/textmap.py`, `r18/r17d_0_2.dis`): byte-level candidates in section T below.
   KSC alone gives 270-290 B.
3. **SSM after X1 (3 KB per chunk)**: section S. Output floor 3 KB / ≥ 3.5 B/cycle ≤ 880 cycles and input 7 KB / 5.75 =
   1,247, both far under the chunk: every compute cycle counts 1:1 (main's HRING + PTRG measured −0.34 for a static −0.17..−0.32).

## T — GEMM text candidates (r17d: 16,336 / 16,384 B; textmap + tcall)

| id | bytes | where | how | risk |
|---|---|---|---|---|
| **KSC** | **−270..−290** | mm_kstep 0xfc0-0x10ff (320 B): frame + `g_ks_for` compare (0xfc0-0x101a, 90 B), the inlined ks_records loop (0x101a-0x10d0, 182 B), store / tail (48 B) | records as build-time data (all 24 cores share the placement); mm_kstep = a tail jump with two constants | low (post-link check) |
| P35 | −80..−120 | main's k-loop 0x310-0x39e (144 B, 25 bundles, half of them i64 counter ops) | the loop moves into mm_ks (+~60 B there) | medium |
| IV32 | −200..−300 | every main loop (12 `phi i64` in opted_main_core_0_2.ll; each increment is `add / ltu / add` and each test `ltu / lt / eq / sel`, plus hi-word spills) | i32 bounds for design8's `range_` loops (scf.for takes any integer type) | low |
| END_MIN | ? (measure) | attn_end 752 B at -O2 | the existing `-DEND_MIN` flag (minsize); attn_end runs ~96 times per step, so even 2x slower costs ≤ 0.02 ms | none: one build, read llvm-size |
| EXIT | −64..−120 | main's register restore at 0x800-0x83e (64 B) and its saves: the endless loop exits only at i64 max | a true `while (1)` worker body (IRON emits `range_(sys.maxsize)`) | medium (IRON internals) |

Not candidates: nop bundles in main (204, 416 B) are delay slots and branch-target alignment; the asm kernels' `_last`
epilogues (~48 B each) are ret + delay slots + the 16 B aligned end.

## S — the SSM after X1 and main's HRING + PTRG (r18a, tcall)
Chunk static 1,979..1,993 + sigk's 8 trips (+280) ≈ 2,265; in situ ~2,550-2,650 (r17d timeline 3.99 ms per step minus
HRING / PTRG). Per chunk (static): ssq 680 (glue 194), sigk 371 (glue 45), ggk 288 (glue 90) + the xc → bfp push, reck 274
(glue 92), z ~246 (serial), convk 226 (glue 60), in_need + in_done 81, rest ~80. 1 cycle per chunk = 1.43 µs per step
(2,592 chunks).

| rank | lever | per chunk | ms |
|---|---|---|---|
| done | ZK (main r18b: chunk static 1,979 → 1,854; −0.17 A/B) | −125 | −0.17 |
| 2 | CGF (convk writes bfp panels; one call and the push loop go) | −70..−90 | −0.10..−0.13 |
| 3 | ssq two chunks per call (prologue / drain 150 once per two chunks; needs the g_ga / g_tsp double buffers) | −75 | −0.11 |
| 4 | FSX (sigk inside ssq) | −50 | −0.07 |
| 5 | per-head params in_copy x 3 (93-194 each, /18 chunks) by DMA into the arena | −25 | −0.04 |
"The 101-cycle pointer glue" is main's PTRG (done, in the −0.34).

## K — the GEMM k-step: 2,122 → ~1,450-1,700 (plan)

**What binds now** (r17d mm_ks, `tools/ksplan.py --dis r18/r17d_0_2.dis`):
- Per group of 32 MACs: 32 MAC bundles + 12.75 swap bundles + 3 ldb-only bundles = 47.75. The swap is the C store /
  reload; the MACs stop because all 4 accumulators end together. The ldb-only bundles come from 36 B ops (32 pops + 4 fills)
  on ldb.
- Per k-step: 9 trips x 4 groups = 1,719, plus glue 96 = 1,815, against tcall's 1,824.
- In situ: 2,055 in mm_ks (+231) and 67 between calls (static 51).

The 231 are not the bank rule: by the r8 placement, consecutive groups alternate pieces in different banks, and the static
model gives 24. The A port is not binding either: it needs 5.0 B/cycle and has 7.95. That leaves DMA writes colliding with
core accesses. The UNVERIFIED model gives 221:
- B prefetch: 144 writes x the bank-3 busy fraction 0.75.
- A slots: 324 writes x ~0.35 in banks 1 / 2.

**The bounds that remain once the swap is gone:**

| limit | per 8 MACs | per k-step |
|---|---|---|
| vec (MAC) | 8 | 1,152 |
| bank 3 (B pops + fills) | 9 | 1,296 |
| load slots (C 4 + B 9 + A 1.1) / 2 | 7.1 | |
| st | 4 | |
| A port | | 10,368 B / 7.95 = 1,304 |
| B port | | 580 |

Registers for ROT:
- ex: 8 A + 4 B = 12, all of them; A is refreshed in place at the row-block change, as today.
- dm: 2-3 chains + 1 loading + 1 storing ≤ 5. The C load of job j + 2 and the store of job j − 2 are 4 jobs apart, so they
  always sit in opposite pieces: bank 2 vs 0 / 1 for rb < 12, and bank 0 vs 1 for rb ≥ 12.

**Experiments, in order** (prices from ksplan):

| # | experiment | where | cost | decides / price |
|---|---|---|---|---|
| E1 | ROT CP model (gen_ksr.py op graph for cpsched: II 9 per job, ex 12, dm 5, the bank rule on the deployed addresses, pops 8 / loads 7 / acc distance 2-3 / old-value windows for the swaps and A refresh) | research pass 14, offline | 1 pass | II 9 feasible? (II 10 with bank stalls still gives ~1,500 static: ~−2.4 ms) |
| E2 | DMAB one-core bench | main, device | S | the collision model: if the DMA yields (0 cycles), ROT's in-situ price is the static −3.9 ms and layout levers are void; if the core yields, layout levers are worth ~−200 per k-step |
| E3 | KS-P38 counters (or the A-only twin: mm_ks with MACs and C traffic removed) | main, device | S | R_A in situ: if the A path delivers < 7 B/cycle with the core busy, ROT bottoms out at A (~1,500) and two-channel A (rev 5 §2.1) joins the plan |
| E4 | P35 (B lock + k-loop in mm_ks) | main | M | −0.44; also the structure ROT needs (one call per tile, both locks in asm) |
| E5 | ROT from E1: one-core ksb.py bit-identical, then in situ | main | L | −3.5 (−2.5..−4.0) on top of E4 |
| E6 | layout from E2: B prefetch buffer outside the bank the k-step pops; A ring away from C pieces (L1 is ~0.9 KB free: needs a GELU table / stack move) | main | M | −100 + −120 per k-step if the core yields (~−1.9 ms) |
| E7 | B split over two banks (bank-3 bound 9 → 8 per 8 MACs) | main | M-L | −144 per k-step (−1.2 ms); then the A port (1,304) binds |

Text: ROT's body is one job loop. U = 16 jobs covers an A object, with the A refresh and acquire inside, so it is
~144 bundles, about today's 1.5 KB. After KSC and P35 there is room.

## Microbenches / measurements named
1. **DMAB** (above). One core, ~30 lines over apb.py: a ZOL of `vlda` (or vst) into bank X at one access per cycle, with a
   tile DMA S2MM filling a 4 KB buffer in bank X (then bank Y) from the memtile at full rate. Read the cycles of N trips.
   - Pre-registered: (1) slowdown = DMA writes / 32 B x core bank-busy; or (2) 0, if the DMA yields.
   - Also 64 B DMA granularity (half the slowdown).
2. **KS-P38**: Performance_Control0 0x31500 start / stop on MEMORY_STALL (23) and LOCK_STALL (26), Counter0 / 1 0x31520 / 4,
   read by the core with `read_tm` (bus enabled as for stamps) around mm_ks in the PROF build.
3. **NIO2**: three more NPROF stamps (nrm call, whole pass-2 unit, whole pass-1 unit; per mode), pre-registered above.
4. Twin check, ZOL-setup and SCALAR-PLUS1 probes: still open (rev 11/12).

## The tools
- `tools/ewio.py fit13 [--rins .. --routs .. --s2s .. --ddrs ..]`, `levers13 --rin 5.75 --rout 4.0 --s2 4`. Constants at
  the top of the rev-13 block:
  - C_IN / C_OUT: static cycles inside the stamp intervals.
  - D1 / D2: in-situ per-unit excess.
  - GCOST: NPIPE group cost.
- `tools/ksplan.py [--dis X.dis]`: the k-step table. Its collision model is UNVERIFIED until DMAB.
- `tools/tcall.py`: unchanged. Note sigk's ZOL shows 1 trip unless the spec gives 8.
