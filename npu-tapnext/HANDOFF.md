# ROUND 21 DONE (2026-10-11) - read this block first; the round-20 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r21k1**, A/B in one process vs r20z **-0.70 ms** (30 x 20: -0.686 / -0.705 / -0.721), BIT-IDENTICAL vs full12r9a
| depth | r21k1 sweep median (min) | whole step | r20z same session | DAVIS AJ (OA) |
|---|---|---|---|---|
| 12 | 31.30 (31.23) | 34.06 (DAVIS loop 34.4) | 32.33 (31.94) | 70.84 (92.57) |
| 10 | 26.22 (26.17) | 29.02 (DAVIS loop 29.5) | 26.89 (26.79) | 69.90 (91.02) |
| 8 | 21.06 (21.00) | 24.11 | 21.53 (21.47) | not run |
| 6 | 16.16 (15.76) | 19.44 | 16.23 (16.17) | not run |
Quiet box for the sweep (crop_study finished; research CP-SAT 4 workers + the background job; 1.81 GHz).
Build: `r21/try21.sh NAME [--sweep]` = r20/try20.sh + CSUB + CSUB-QK + ZLOAD + E7'' (E7_V=e7r68b) + K0B (each with an env =0 switch;
IV32 is design8's default), then csplit20 with CSUB sub-parts + NRESFIX + E7PATCH. Never dispatch *.unpatched.xclbin. A new E7 kernel:
`python3 r21/mke7.py ../npu-research/r22/mm_e7_<x>.s r21/mm_rot_<v>.s` (FITTED .s only) then `E7_V=<v> r21/try21.sh NAME`.
Text: GEMM 15,760 / 16,384 (624 B free), EW 16,272 (112 free). Timeouts this round: 1 (LATEC; recovered).
## What changed (NOTES 26, RESULTS 30): the C drain overlaps the epilogue (CSUB) and the first k-step needs no zeroing (ZLOAD);
B is popped from both FIFOs out of two banks (E7'', research's generator, main's layout + BD split); i32 loop counters (IV32).
## Where the time goes: GEMM k-steps (E7 R 68: 1,338 static per call), epilogues ~1.6 ms (+ the drain collisions CSUB moved there),
attention ~6.5, norms ~4.6 (DDR read cap), SSM ~3.6 (DDR), first-tile paces ~0.25 ms, GEMM -> norm tails ~0.58 ms.
**The NPU clock falls to 1.405 GHz under ~20 busy CPU threads** (r7/clk.py): label absolute numbers; A/B pairs stay valid.
## Next concrete actions (round 22)
1. KTL (research rev 17 rank 1, -0.40..-0.52): one modulo loop per tile on the E7 kernel (research gen_e7 --aligned in flight); main:
   mm_ktile's call sequence (B(0) acquire, mm_rot(.., 781), release, the multi-k-step call), KTLP one-core probe if research asks.
2. E7'' at R 64-66 with near-zero replay bank stalls (R 64 + 129 stalls was neutral).
3. Attention S3 / K2 (text is there now: 624 B): K2 on the K0B batched form (QK-only + running max for chunk 0).
4. 3A (-0.10..-0.14; memtile repack or l2 depth 7), OPPRE-W (the round-20 OPPRE patch is lost: rebuild from NOTES 25.4).
5. SSM ssq2 / FSX once EW text is found (EW 112 B free; norm_rows_k is 3,504 B of C++).
## Rules that bit us this round (also in the skill)
- A BD block needs one acquire and one release or none: release-by-0 works; acquire-by-0 fails at CDO generation (use a hand-off lock).
- An unfitted modulo kernel (prologue not ending 16 B aligned) runs and loses exactly its prologue's jobs: always fit; e7dbg finds it.
- Never make a core wait on its own C drain while it holds multicast inputs (LATEC hang).

# ROUND 20 DONE (2026-10-11) - read this block first; the round-19 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r20z**, A/B in one process vs r19z **-2.07 ms** (40 x 20: -1.99 / -2.07 / -2.06), BIT-IDENTICAL vs full12r9a
Every number after ~03:50 was taken under a second foreign CPU job (crop_study, ~7.6 cores) besides the usual worker: the sweep below
is inflated for BOTH builds (r19z quiet 33.88 vs 35.65 here); quiet-box estimate for r20z ~31.8 ms at depth 12.
| depth | r20z sweep median (min) | whole step | r19z same session | DAVIS AJ (OA) |
|---|---|---|---|---|
| 12 | 33.71 (32.15) | 36.87 (DAVIS loop 36.9) | 35.65 (34.03) | 70.84 (92.57) |
| 10 | 28.30 (27.13) | 31.55 (DAVIS loop 31.7) | 29.95 (28.51) | 69.90 (91.02) |
| 8 | 22.35 (21.51) | 25.58 | 23.91 (22.96) | not run |
| 6 | 16.76 (16.20) | 19.82 | 17.89 (17.14) | not run |
Build: `r20/try20.sh NAME [--sweep] [--k-add ..] [--env ..]` = r19z + IL DAPRE NSEQ NRES(norm0-3) SRES AQPRE AKVPRE (env defaults;
NRES=, SRES=0 etc. override), then r20/csplit20.sh (CSPLIT patch + **NRESFIX**: the EW MM2S 1 panel route; never dispatch any
*.unpatched.xclbin or a depth build's own xclbin). Text: GEMM 15,984, **EW 16,272 (112 B free)**. Timeouts this round: 0.
Commits 80202f6 .. b83c7e4 + round end.
## What changed (NOTES 25, RESULTS 29): the op boundaries = DDR bytes (drain tail at the write cap + CP + first tile at the read cap);
A kept on chip from the EW producers (NRES, SRES), the next A loaded into the free slot half (IL, DAPRE, AQPRE, AKVPRE), NSEQ.
## Where the time goes (timeline r20z1, ms, loaded box): attention 6.6, norms 4.7 (DDR read cap), SSM 3.6, GEMMs ~17 (k-step model
~14.8); remaining first-tile pace per block: out_proj 6.8, down_b 6.0, mlp3_b 5.3 us; C waits ~0.48 ms per step (hand 1,355 plain /
1,807 gelu_pan, qkv, gated-second vs 676 floor).
## Next concrete actions (round 21)
1. E7'' (B on both FIFOs, research -0.9 +- 0.3): POPLF1 passed; needs research's generator design (gen_rot variant).
2. C waits: CSPLIT4-lite (gelu_pan restructure: bgelu3 + pan split at rb 12, 1a = [0, 6,912)), LATE-C / zero-load first k-step, or a
   runtime-driven C path with per-op object sizes (also drops 74 MB of junk writes).
3. down_b / mlp3_b pace: a third A region needs ~124 KB on memtiles 0-2 (B fifos there: move B_r's forward to memtiles 3-5?).
4. out_proj boundary (OPPRE was +0.18: stamp it with ATPROF + TPROF first); attention S3 / K2; SSM ssq2 / FSX (EW text is tight).
## Rules that bit us this round (also in the skill)
- IRON's pathfinder: no route for two packet flows from two DMA channels of one tile: declare from one, patch the switchbox rule.
- One full lock for both slot halves breaks any half-1-first load: per-half F locks.
- A scattered slot layout must stay chunk-major (the last tile's chunk-wise release).

# ROUND 19 DONE (2026-10-11) - read this block first; the round-18 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r19z 33.88 ms** at depth 12 (sweep median; A/B vs r18m -0.88 in one process), BIT-IDENTICAL vs full12r9a
| depth | dispatch ms (sweep median) | whole step ms | fps | DAVIS AJ (OA) (CPU bf16) |
|---|---|---|---|---|
| 12 | 33.88 | 35.95 (DAVIS loop 36.8) | 27.8 | 70.84 (92.57) (70.98) |
| 10 | 28.37 | 30.25 (DAVIS loop 30.7) | 33.1 | 69.90 (91.02) (70.07) |
| 8 | 22.74 | 24.96 | 40.1 | not run (68.22) |
| 6 | 17.09 | 18.99 | 52.7 | not run (64.77) |
Build: `r19/try19.sh NAME [--sweep] [--k-add ..] [--env ..]` = try18 + ROT_V=g0a + SEQPRE=1 + ZHI + CSPLIT + EPLR + KPREP, then
**r19/csplit.sh** (patches the lowered MLIR: two-BD C drain + lock 6, and re-runs aiecc; an UNPATCHED -DCSPLIT xclbin HANGS: never
dispatch full12NAME.unpatched.xclbin or a depth build's own xclbin), then regress / sweep. KS_ACC_HI default 0x719C0 (ksc_check names
a new one when .data moves; g0a was solved conflict-free for 0x71940..0x71A00). Timing: r18/ab.py. Text: GEMM 15,984, EW 15,808.
Timeouts this round: 0. Commits 86857ca .. 0b85f5e + round end.
## Where the time goes (timeline r19d + TPROF): GEMM k-steps ~13 ms (1,576 per k-step), GEMM epilogues ~1.65, C part-1 drain waits
~0.78, op-boundary bubbles ~5 (GEMM ~3.3, attention 1.2, norms 0.8), attention span 5.5, norms 4.1 (DDR-read-bound), SSM 3.35.
## Next concrete actions (round 20)
1. **CSPLIT4-lite** (~-0.2..-0.3): a static sub-part boundary inside part 1 (e.g. 9,216 B) released early by gated (gup + stash copy
   rows 0-15 = bytes 0..9,216) and gelu_pan (pan's rb 0-11) epilogues; needs an acquire-only first BD in csplit.py (probe that the
   lowering accepts it and the drain stays in order) and the epilogue passes split at the boundary (in-place order checked).
2. **Op boundaries**: SEQPRE showed the CP instruction count is not the boundary; stamp the dependent ops' first tiles (TPROF
   `first` with ~400 B of text: drop something for the timing build) and decide DMA latency vs A-slot pacing. Research rev 16.
3. LATE-C (research R3, -0.3), DAPRE (-0.2..-0.3), attention S3 / K2, SSM ssq2 / FSX; N-IO / NS only for LN 2-input norms.
4. E7'' (B on both FIFOs) is open research; norms need fewer DDR bytes (on-chip residency) to move.
## Rules that bit us this round (also in the skill)
- `vldb.pop.576.2d` with dn = 0 post-modifies by line step + m (not m alone); padda must precede the fill by a bundle.
- Pushing memtile MM2S tasks before the fills of the same op can fill the 4-deep queue and block the CP on unissued work (+1.1 ms).
- `#pragma unroll` over a loop with a conditional vector reference stayed rolled (selects): unroll by macro.

# ROUND 18 DONE (2026-10-11) - read this block first; the round-17 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r18m 35.20 ms** at depth 12 (sweep median, min 34.80), 4 phase sequences per depth, BIT-IDENTICAL vs full12r9a
| depth | dispatch ms (sweep median) | whole step ms | fps | DAVIS AJ (OA) (CPU bf16) |
|---|---|---|---|---|
| 12 | 35.20 | 38.44 (DAVIS loop 37.0) | 26.0 | 70.84 (92.57) (70.98) |
| 10 | 29.11 | 30.99 (DAVIS loop 31.7) | 32.3 | 69.90 (91.02) (70.07) |
| 8 | 23.30 | 25.31 | 39.5 | not run (68.22) |
| 6 | 17.53 | 19.61 | 51.0 | not run (64.77) |
Build: `r18/try18.sh NAME [--sweep] [--ew-add ..] [--k-add ..]` = r17d + EW HRING PTRG ZK NIOC CGF + k V2T KSC (KS_* addresses,
acc_hi 0x71980) P35 MM_ROT (ROT_V=g2a). ksc_check (design8) fails a build whose layout moved and names the new acc_hi (KS_ACC_HI=..).
Timing: `r16/nrun.sh LOG python3 r18/ab.py --a BASE --b NEW --rounds 20 --block 22` (one process, interleaved). Commits 52ffad1 ..
1ffff40 + round end. Timeouts: 1 (V2T, `add #64` wrapped; recovered). Text: GEMM 15,536, EW 15,808 (NIO2 stamps need ZK off).
## Where the time goes now: GEMM k-step (mm_rot) 1,603 cycles in situ per k-step incl. 168 memory stalls (KS-P38); norms ~4.3 ms;
SSM ~3.3 ms; attention ~6.6 ms. (A per-op timeline of r18m was not taken this round: take it first in round 19.)
## Next concrete actions (round 19)
1. ROT variants g0f (-0.28) / g1f (-0.14): run research's probes PWRAP / PADDA first (PROPOSALS rev 14); re-solve with
   ROT_ACQ_GAP=6 ROT_REL_GAP=8 (r18/gen_rot.py) - the delivered g2f raced in situ with acq.cond one bundle before the A fill.
2. GEMM layout vs DMA collisions (DMAB: ~1 cycle per 32 B into the bank in use; 168 stall cycles per k-step left): a hand-built B
   ring (P35 already takes the B locks by id) to put the two B objects in different banks; A slots 1 / 2 share bank 2 with C.
3. Norms: N-IO in asm (nrm / n1k over a group per call: the pass-2 unit is ~155 cycles of glue around nrm, NIO2), NS.
4. Attention: S3 / K1 / K2 (GEMM text has 848 B free now); V1's remaining part (attvs folded into attv).
5. SSM: ssq two chunks per call (-0.11), FSX (-0.07), params in_copy by DMA.
## Rules that bit us this round (also in the skill)
- `add rX, rY, #imm` takes a 7-bit signed immediate: #64 assembles silently as #-64. Read every new preamble's disassembly.
- A lock acquire must sit >= 6 bundles before the load that depends on it (ROT g2f: 1 bundle raced in situ; one-core benches with
  pre-released locks cannot see it). Benches of lock-taking kernels need a DMA-fed ring to test the acquire timing.
- One-core benches that #include build/<k>.h pick up whatever the last design build left there: copy the variant's header first.

# ROUND 17 DONE (2026-10-11) - read this block first; the round-16 block below is the base it builds on
## Working rules from the user (keep for every later round)
- **All compute runs on testbox, never the laptop**: CP-SAT / generators / sims / python tools from ~/git/app/.scratch/npu-tapnext
  with `nice -n 10` and at most 12 solver workers (ortools: `~/.local/bin/uv run --no-project --with ortools --with numpy python ...`,
  9.15.6755 as on the laptop; r17/cp_attks.sh is the pattern; cpsched defaults to 4 workers, pass --workers). Edit files on testbox
  directly (source of truth); the laptop is only ssh + reading; rsync the laptop mirror at round end.
- Dispatch times are bimodal (~0.17 ms apart) under the foreign CPU worker: compare SPREAD lines (p10 / median / mean of 200
  iterations interleaved with the baseline), not one 60-iteration median.
## State: **full12r17d 39.16 ms** at depth 12 (sweep; time run 39.19), 4 phase sequences per depth, BIT-IDENTICAL vs full12r9a
| depth | dispatch ms (sweep median) | whole step ms | fps | DAVIS AJ (OA) (CPU bf16) |
|---|---|---|---|---|
| 12 | 39.16 | 41.01 (DAVIS loop 42.2) | 24.4 | 70.84 (92.57) (70.98) |
| 10 | 32.78 | 34.86 (DAVIS loop 35.0) | 28.7 | 69.90 (91.02) (70.07) |
| 8 | 26.29 | 28.37 | 35.2 | not run (68.22) |
| 6 | 19.79 | 21.87 | 45.7 | not run (64.77) |
Build: `r17/xtry.sh NAME 4 [--sweep] --k-add -DATTKS2MAX --env AL1=1 --ew-add -DAL1 --ew-add -DNPIPE --ew-add -DNPIPE_G=6` (try17.sh
= r16z's flags; xtry adds XROT=4 / -DXROT and the phase builds full<d>NAME_p1..3, then regress / time / sweep). Commits: 16998cd (S2),
26ccb0a (AL1, NPIPE, sliding X1, stamps), d3c671d (X1 mod 4), round end. Timeouts: 1 (1 KB AL1 padding; recovered). Text: GEMM
16,336 / 16,384 (48 free), EW 15,808 (NPROF + CPROF fit only without N1K).
## Where the time goes (timeline r17d, ms per step): attn 6.78, norms 4.87, ssm 3.99, mlp0 3.82, up_a 3.53, up_b 3.29, qkv 2.76,
yx 1.98, mlp3_a 1.92, down_a 1.45, mlp3_b 1.32, down_b 1.27, out 1.17, out_proj 1.04. SSM chunk (CPROF): convk 280, pointers 101,
rest 2,562. Norm units (NPROF r16z / NPIPE 2): in_need 101.5 / 61, out_begin 45 / 47, n1k step 161.
## Next concrete actions
1. GEMM text (48 B free) before S3 / V1 / K1 / K2 (research -0.73 together): find 200-400 B (csize.sh; k_rb's slow path is smaller
   now; KFOLD-style folding; the attkb prologue with 3 stages at II 36).
2. Norms: in_need still 61 per unit after NPIPE: price the input side (n_in 3 ops read 3 KB per unit); fold D1 + D2 / D3 + D4 partials
   upstream (P35 / P24b) so norm0 / norm2 read 2 streams; the EWO bench (research) for the output rate.
3. SSM after X1 (3 KB output per chunk): FSX inside SSQ, the 101-cycle part pointer + mapping glue, ssq's prologue.
4. Research rev 13 (RESULTS_FROM_MAIN round 17 holds the verdicts and requests).
## Rules that bit us this round (also in the skill)
- An EW input fill group must total a multiple of the 768 B ring slot (1 KB-padded params hung the step): pad to 3 KB multiples.
- Memory-carried modulo state: load by offset / store with post-modify, maxdist(load, store) <= II - 4, stores in the last stage.
- CP schedules failing only the pad replay: re-solve with the colliding op's time forbidden; allocation at maxlive = budget: reserve 1.
- r13/attkb.py's C++-reference compare reports BAD P for the deployed attkb too: compare builds with --hashes.

# ROUND 16 DONE (2026-10-11) - read this block first; the round-15 block below is the base it builds on
Read the skill (round 16 added: the 576-bit pop-stream underflow and its fill rule, periodic 2D modifiers, #undef-ed flags, 64-lane
y-pair MACs, msched loop-carried uses, register-resident state across C++ calls, per-call overhead, twins vs I/O bound) and NOTES 21 /
RESULTS 25 (every number of this round). Research rev 11 tools are in use (try.sh via r16/try16.sh, asmlint, the build lock).
## State: **full12r16z 39.94 ms** at depth 12 (min 39.83; r15m re-timed 41.50), any depth from one xclbin, regress BIT-IDENTICAL vs full12r9a
| depth | dispatch ms (sweep median) | ms per frame (sweep whole step) | fps | DAVIS AJ (OA) (CPU bf16) |
|---|---|---|---|---|
| 12 | 39.94 (time) / 40.30 (sweep) | 43.6 (DAVIS loop 42.3) | 23.0 | 70.84 (92.57) (70.98) |
| 10 | 33.84 | 37.1 (DAVIS loop 36.3) | 27.0 | 69.90 (91.02) (70.07) |
| 8 | 26.86 | 29.1 | 34.3 | not run (68.22) |
| 6 | 20.18 | 22.5 | 44.4 | not run (64.77) |
Build: `r16/try16.sh NAME [--ew-add ...] [--sweep]` = r15m's flags with EW_DEFS - -DSSX + -DSQK -DINPART_INLINE -DXONE -DGGK -DSSQ
-DSIG10 -DN1K (kdefs and design args = r15m's). Commits this round (local repo holding npu-tapnext/): SQK, P-IN, XONE, GGK, SSQ, SIG10,
N1K, NPROF, round end. Timeouts this round: 0. Text: GEMM 16,192 / 16,384; EW 15,600 / 16,384 (NPROF adds ~460; PROF no longer fits).
## Where the time goes now (EW, cycles per SSM chunk, r16eP + SIG10): conv 440, gates 365, sigk ~440, ssq 715, recur 313, z 264,
rest 213, par 151, cp_h_pan 192, cp_x 269 (SSM ~420 us per op x 12). Norms: n1k step 161.5 per unit (in situ = bench), in_need 105
per unit (partly input-bound now).
## Next concrete actions
1. **S2** (research rank 2: the attention running max on attkb's stack, -0.38): the biggest priced item left; then S3 / V1 / K1 / K2.
2. **X1 full rotation** (cp_x 269 -> 0 and 3 KB less SSM output per chunk, ~-0.4): slide the SSM area's base by one region per frame
   (x, x-1, x-2 then already sit where the next frame reads them); per-frame runtime sequences + a copy frame at the wrap. NOTES 21.4.
3. Norms: they are partly input-bound (in_need 105 per unit): price pass 2 + glue vs the port before more asm; N1K at II 8 (vec
   bound) if the y pairs allow.
4. SSM: conv 440 vs convk 251 on one core (where are ~190?), sigk / FSX inside SSQ (x pressure), recur + z.
5. Research rev 12 (requested in RESULTS_FROM_MAIN round 16).
## Rules that bit us this round (also in the skill)
- A long 576-bit pop stream needs a fill every 8 pops (it silently underflows at the 9th pop).
- A 2D modifier repeats: one jump inside k steps = dn k - 1 with the counter started at k - n1.
- -DPIN did nothing (a header #undef-s PIN): check the elf changed after adding a flag.
- Timing twins overstate once the op is near its I/O bound: stamp the waits first.

# ROUND 15 DONE (2026-10-11) - read this block first; the round-14 hand-off below is the base it builds on
Read the skill (round 15 added: immediate ranges, non-const bank tables, vmov x <- bm ports, output-ring store stalls, direct
pushes, rep4 by vshift, div768, reload streams, the jit cache key) and NOTES 20 / RESULTS 24 (every number of this round).
## State: **full12r15m 41.52 ms** at depth 12 (r14f re-timed 47.26), any depth from one xclbin, regress BIT-IDENTICAL vs full12r9a
| depth | dispatch ms | ms per frame | fps | DAVIS AJ (CPU bf16) |
|---|---|---|---|---|
| 12 | 41.52 | 43.9 | 22.8 | 70.84 (70.98) |
| 10 | 35.03 | 38.1 (37.0 r15k3) | 26.2 | 69.90 (70.07) |
| 8 | 27.86 | 30.1 | 33.2 | not run (68.22) |
| 6 | 20.96 | 23.3 | 43.0 | not run (64.77) |
`P60G=1 P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3 -DSIGK -DSSX -DCPY64 -DCONVK -DRECK
-DNRMK -DZRING" AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r15m --lookahead 2 --l2-depth 8 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,
-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU,-DATTKF,-DATTKE,-DATTKB,-DKFOLD,-DZASM,-DATTV,-DVTK,-DRESCK`
(`EWD="<EW_DEFS>" r15/endrun.sh r15m` builds full{10,8,6}r15m and runs the sweep + DAVIS d12 / d10.) Timeouts this round: 0.
Text: GEMM 16,192 / 16,384 (192 free); EW 14,960 / 16,384 (+ZRING: same) - the EW PROF variant fits again.
## What exists (r15/)
- Generators (CP-SAT schedules saved as JSON next to them; solve on the laptop with `uv run --no-project --with ortools`):
  gen_ssx (SSM log_a + exp, ssx.json), gen_cpy64, gen_conv (convk.json), gen_rec (reck4.json: U 4), gen_resc (resc.json),
  gen_nrm (nrm_rms.json / nrm_ln.json; --mode), gen_zk (unused). One-core benches: ssxb, cpyb, convb, recb, rescb, nrmb, zkb
  (`python3 r15/<b>.py build --s build/r15/<k>.s --out build/r15/<b>; ... run --out ...`, all compare with verbatim C++).
- design8 builds each kernel from its flag (ew_object / asm_object); plan8 P60G (recorded in build json, applied by step8 / run8).
- ew8.cc: div768 slot counts (always on), copy_run / in_copy by cpy64, norm_rows_k (NRMK: one function for both modes), ZRING.
- r8/ewprof.py: 14 phases (cp_h_pan, cp_x split out of out_copy).
## Where the 41.5 ms go (timeline r15h, before NRMK / ZRING): SSM + out 6.76, attention 7.08, norms 6.46 (now ~-0.5), GEMMs ~22.
## Next concrete actions
1. EW PROF of full12r15m (fits again) to re-rank the SSM phases (gates 652, sigk 509, ssx 509 incl. a 120-cycle prologue, sqrt 493,
   in_wait 165, copies of the x parts 289) and the norms (pass 1 + stats now dominate: asm candidates; the C++ pass 1 spills).
2. Attention S2 (running row-block max: removes the 393-cycle first-pass rerun): needs 768 B - bank 3 has only 384 B free + 384 B
   in the K object's tail (rev 10 assumed 2,688): a 3D-modifier split layout or moving sgelu_cd / serf_cd out of bank 3.
3. The output-ring store stall (~3.5 cycles per 64 B vector in situ, bank 0 with the stack): probe ewo in another bank
   (Buffer(mem_bank=)), or move the stack.
4. SSM sqrt in asm (mv-bound, ~-0.29), gates in asm, P52 (x copies by DMA), sigk + ssx fusion (one prologue less).
5. Research rev 11 (requested in RESULTS_FROM_MAIN round 15).
## Rules that bit us this round (also in the skill)
- IRON's jit cache key ignores source files: a kernel edit under the same build name silently reuses the old build.
- Extra passes over the EW output ring cost more than they save (NRMK with copies +0.6 ms; direct stores -0.40).
- A deployed C++ loop can run 3x slower in situ than its own one-core bench (the conv): price from in-situ PROF, not benches.
- CP schedules can fail msched.check only with the .p2align pad: check inside the II search.

# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-10, end of round 14)

Read `~/.claude/skills/xdna2-npu/SKILL.md` first (round 14 added: the zero-overhead-loop setup distance, mid-line FIFO restarts
need explicit pointers + position resets, scalar results read 2 cycles on, pointer-only 2D steps, the K object's junk tail as
scratch, text levers, the measured lockstep factor), then `~/git/app/.scratch/npu-research/PROPOSALS.md` (rev 9 or later) and
the round-14 part of `RESULTS_FROM_MAIN.md`. RESULTS.md 23 has every round-14 number, NOTES.md 19 the changes, facts and dead
ends. All timings: testbox, Turbo, GPU idle, pyxrt `run.wait`, 1.81 GHz. No deadline.

## 1. State: 47.31 ms at depth 12, any depth from one xclbin
| depth | dispatch ms | ms per frame | fps | DAVIS AJ (CPU bf16) |
|---|---|---|---|---|
| 12 | 47.31 | 49.5 | 20.2 | 70.84 (70.98) |
| 10 | 39.84 | 42.1 | 23.8 | 69.90 (70.07) |
| 8 | 31.87 | 34.2 | 29.3 | not run (68.22) |
| 6 | 24.43 | 27.6 | 36.2 | not run (64.77) |
Best build `full12r14f` (+ full{10,8,6}r14f):
`P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3 -DSIGK" AIECC_EXTRA=--reclaim-runtime-bds
./bb.sh full12r14f --lookahead 2 --l2-depth 8 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU,-DATTKF,-DATTKE,-DATTKB,-DKFOLD,-DZASM,-DATTV,-DVTK`.
Numerics BIT-IDENTICAL to round 9 (`step8.py regress --ref full12r9a`). Timeouts this round: 0. **GEMM text 16,320 / 16,384 B**
(64 free); EW 16,224 / 16,384. Builds in the same build dir must be sequential.

## 2. What exists (new in round 14, r14/ unless said)
- gen_attkb.py (+ attkb.json): the fused K loop over all 36 (row block, key block) iterations of a (tile, chunk); mm8.cc
  `-DATTKB` driver (chunk_k_fused + k_rb: per-row-block path for chunk 0, the last chunk and flagged row blocks; the exact
  rerun uses the same loop with clamp 71.5 = aexp_push). `-DATTKE`: chunk-0 rescale skip + last chunk with 2 trips + zero panels.
- gen_attv.py (+ attv.json; `--sums` + attvs.json): the V phase and the row sums in asm (`-DATTV`). gen_vt.py (+ vt.json): the
  qkv V^T epilogue in asm (`-DVTK`). zero2.s (`-DZASM`): acc_zero + attn_begin's zero fills.
- csize.sh (GEMM per-function text from mm8.cc; edit its flag list to the build's), deal.py / deal2.py (P60 single-TAP proof).
- r13/attkb.py one-core bench: `--defs`, `--hname`, `--extra-asm`, `--krb` (k_rb phase stamps, KRB_STAMP), `--vphase`,
  `--zero`; in-situ bank layout (K in bank 3, Q at q_buf(1), the C object at 0x78900). Twin `-DATTK_FORCE=128`: the batched
  chunks' flags ignored (timing only).
- msched check(): per-op 2D / 3D modifiers and pointer-only "pa" ops.

## 3. Where the 47.3 ms go (timeline r14f, ms per step)
SSM + out 9.78, attention 8.52, norms 6.86, up_a + up_b 6.8, mlp0 3.80, qkv 2.76, down 2.73, yx 1.97, mlp3_a 1.93, mlp3_b 1.38,
out_proj 1.04. Inside attention: the fallback of the ~2.3 % flagged row blocks + the K / V lockstep ~2.1-2.6 ms (twin r14cF128
46.15 ms).

## 4. Next concrete actions (in order)
1. **The fallback** (~2.3 ms of wall for ~0.35 ms of cycles): per flagged row block first-pass rerun 393 + exact test 164 +
   rescale 578 + exact rerun 393 cycles. Options: rescale_rb in asm (its exp32 + 9 hi/lo passes: ~578 -> ~200), the exact test
   at -O2 (164 -> ~70), a cheaper score recovery than the first-pass rerun (a QK-only variant of attkb needs ~300 B of text), and
   **P60 with multi-TAP DMA** (single-TAP is impossible: NOTES 19.2): an attention-specific memtile ld pattern for Q and per-piece
   O drains or a memtile reorder. Re-fit research's attsim lockstep on the F128 twin first.
2. **Text** (64 B free): k_rb 1,056 B (minsize), attn_chunk_k 1,184, attn_end 752, mm_kstep 320 (T3f), rescale_rb 560 + exp32 304.
3. **P29b** (SSM log_a + exp chain in asm, research -0.95): r14/ssm_exp_deployed.s is the deployed sequence (53.5 bundles per
   vector, vec bound 20); 2^n via vups.2x / vsrs.2x / vadd.16; 13 constants > 12 x registers: reload some per use.
4. Per-(tile, chunk) overheads of the batched K call (-m rows prep + flag scan ~46 cycles per row block), the V phase's sums
   (attvs 126 cycles) and calls.
5. P55, P35 + P24b, P57, P14; the product wrap of Step8 (depth 10 default, HOST_HEADS=np, OPENBLAS_NUM_THREADS=1).

## 5. Rules that bit us this round (also in the skill)
- A bare zero-overhead loop needs full-form nop bundles after `add.nc lc` (two-byte nops: the body ran once).
- Restarting a pop stream after 432 B: explicit pointer + position reset; walker copies need a maxdist after the walker's step.
- Scalar alu results are read 2 cycles later (a flag OR at +1 missed the last iteration: only the in-situ regress caught it).
- Per-row-block reruns must not write the batched flag array (separate scratch).
- A 2-loop zero kernel: build both loops identically; check `rc=` AND the elf text after every build (a stale zero2.s once).
