# ROUND 25 DONE - STATE AT PAUSE (2026-10-11) - read this block first; NPU optimisation is paused here (the maintainer), round 24's block is the base
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: best = **full12r24z** (no kept change in round 25), reproduced byte-identically from this checkout as full12r25z (all 16
runtime sequences d12 / 10 / 8 / 6 x 4 phases, the 32 elfs and the CDO equal); regress BIT-IDENTICAL vs full12r9a; asmlint 0.
| depth | sweep median (min), r25z = r24z bytes | whole step | DAVIS AJ (OA), 10 videos, this round |
|---|---|---|---|
| 12 | 28.42 (28.37) | 30.45 | 70.84 (92.57), re-run: identical to round 24 |
| 10 | 23.83 (23.78) | 25.84 | 69.90 (91.02), re-run: identical |
| 8 | 19.09 (19.05) | 20.98 | not run |
| 6 | 14.40 (14.35) | 16.86 | not run |
Box for the sweep: load 3.0-9.1 (the background job + research CP-SAT, scalar), r7/clk.py 1.808 GHz, gpu_busy 0. Timeouts: 0.
References: bf16 CPU 70.54 (10 videos); round-24 DAVIS 70.84 (92.57) d12, 69.90 (91.02) d10.
## Exact build (from a clean checkout of strix-halo-maxx; .scratch/npu-tapnext is a symlink into it; ~10 min CPU, docker only):
    cd ~/git/app/.scratch/npu-tapnext && r24/try24.sh NAME [--sweep]
  (--sweep also builds depths 10 / 8 / 6 and times them; NOREG=1 / NOTIME=1 skip regress / time). The chain is r24/try24.sh ->
  r23/try23.sh -> r22/try22.sh -> r21/try21.sh -> r20/try20.sh -> r18/try18.sh -> r17/xtry.sh -> npu-research/tools/try.sh, then
  r20/csplit20.sh (lowered-MLIR patch: CSPLIT / CSUB sub-parts 1728,3456, NRESFIX route, E7PATCH B split) rebuilds the xclbin;
  **never dispatch build/*.unpatched.xclbin**. Run: `r16/nrun.sh LOG python3 step8.py {regress|time|sweep|timeline} --build full12NAME`,
  DAVIS: `TMO=3000 ./s8.sh LOG davis --build full12NAME --depth 12 --videos 10`.
## Every switch of the best build (as the docker build sees them):
- design8 / plan8 env: P60G=1 P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 AIECC_EXTRA=--reclaim-runtime-bds XROT=4
  AL1=1 IL=1 DAPRE=1 NSEQ=1 SRES=1 AQPRE=1 AKVPRE=1 NRES=norm0,norm1,norm2,norm3 SEQPRE=1 E7=1 ROT_V=k70p A3=1
  ATTKS_JSON=r22/attks33.json ATTKB_DM=0,1,2,4,3 ATTV_JSON=r22/attv20.json MSPLIT=down_b,mlp3_b,out_proj YXSSM=1 YXR=1 ZCH=1
  OUTPRE=1 OPH1=1 OPPRE2=1 SSMPRE=1 CPPRE=1; design args --lookahead 2 --l2-depth 3.
- EW_DEFS: -DSSM_LUT -DSSM_EXP3 -DSIGK -DCPY64 -DCONVK -DRECK -DNRMK -DZRING -DSQK -DINPART_INLINE -DXONE -DGGK -DSSQ -DSIG10 -DN1K
  -DXROT -DAL1 -DNPIPE -DNPIPE_G=6 -DHRING -DPTRG -DZK -DNIOC -DCGF -DNRES -DTXT18 -DSTATS64.
- GEMM kdefs: -O2 -DMM_ASM -DMM_KSTEP -DMM_PIECES -DMM_KS -DGELU_MS -DGUP -DCPY32 -DPAN -DGELU_FLAT -DBGELU -DATTKF -DATTKE -DATTKB
  -DKFOLD -DZASM -DATTV -DVTK -DRESCK -DATTKS2MAX -DV2T -DKSC -DKS_ACC_HI=0x71A00 -DKS_C=0x78900 -DKS_A0=0x77B80 -DKS_A1=0x78000
  -DKS_A2=0x78480 -DP35 -DKS_B0=0x7D100 -DKS_B1=0x7E180 -DKS_BPROD=0 -DKS_BCONS=1 -DMM_ROT -DZHI -DCSPLIT -DCS_Z=6 -DEPLR -DCS_CONS=3
  -DKPREP -DZLOAD -DE7 -DKS_X0=0x70A00 -DKS_X1=0x70C40 -DK0B -DCSUB -DMSPLIT.
- Off by default (switches kept): round 25 P0FREE, CPPRE_OP, RRTWIN (timing twin), RRES / -DRRES (not buildable yet), -DNIO2(_NONRM)
  (stamps); round 24 MSDEFER, YXH, ZDROP; earlier ones in their blocks below. Text: GEMM 16,256 / 16,384, EW 15,728.
## Round 25 (NOTES 30, RESULTS 34): nothing kept. P0FREE + CPPRE_OP (OPPRE2's A parts configured before the waits; BD peak 16) neutral
(+0.009 median). Out's bias wait is off the critical path (the slowest SSM column waits 0.03 us). MSPLIT-C up_a priced ~0 (norm1 at the
read cap: pass 1's B re-read lands in its span). **RRTWIN**: 3-input norms are byte-bound (norm0 90 -> 67, norm2 92 -> 74 us without
the residual), 2-input norms compute-bound at ~500 (RMS) / ~560 (LN) cycles per unit; NIO2: pass 2 costs 287 / 364 per unit vs
117 / 177 static. RRES (residual rows on chip) designed and coded, blocked at aiecc. C hand-offs priced ~28 us per block.
## Known risks
- **CPU vector load drops the NPU clock**: 12 AVX-512 numpy threads -> 1.41 GHz (x1.27 slower) at the same package power that 12 spin
  threads run at 1.81 (CLKLOAD, round 23; cause unseparated, npu-research/r25/clkload2.sh not run). Keep <= 6 vector threads beside it.
- **The NPU with the GPU busy is untested**: every number here is gpu_busy 0. An LLM on the GPU shares DDR (the norms and the SSM sit at
  the 55-59 GB/s read cap) and maybe power: expect slower steps; measure before relying on 28 ms.
- Dispatch-time noise under foreign CPU load is bimodal (p90 ~32 ms with a busy box): compare A/Bs in one process (r18/ab.py).
- A changed design must be checked: SEQLOG for CP-order deadlocks, the shim BD peak <= 16 in the build log, asmlint 0, one short
  dispatch first; a kernel taking a new lock hangs on an unpatched xclbin.
## Open queue with prices (ms per step at d12; all need a build + one-process A/B)
1. **RRES** (residual rows rb < 10 of norm0 / norm2 parked in memtile half 1 during mlp3_b / down_b, fed to the EW S2MM 1): ~-0.27
   (RRTWIN bound -0.49). Code in place (RRES=norm0,norm2 + --ew-add -DRRES); blockers: memtile (0,1) runtime-BD budget ("Too many
   simultaneously active buffer descriptors") and the 2 KB R ring's placement (pin it after core_data; data_size 26,240). L.
2. **SSM glue by chunk pairs** (ssq2 / reck2 / ggk2): -0.15..-0.25 (research SSMCUT: lands to the ~2,030-cycle input floor); needs ~5 KB
   EW L1 (stack / arena: RRES takes the data tail). M-L.
3. Norm pass-2 glue (287 / 364 cycles per unit vs 117 / 177 static, every norm at that floor once bytes allow): up to ~-0.1..-0.3 if
   halved; needs research's static decomposition first. M.
4. KTL R 68-69 bank-clean (research CP-SAT): -0.2..-0.3 if a schedule exists. S once found.
5. attkb II 33 -> 32 (bound 31): -0.035..-0.065 if CP finds it (12 workers, 3,600 s, --life 1, stages 3 / 4). S.
6. C hand-offs (~28 us per block): <= -0.1..-0.15, needs LATEC-style split C locks (hung in round 21). L, risky.
7. Dead / ~0: MSPLIT-C up_a / qkv / mlp0 (read-capped norm spans), out's bias wait, CPPRE_OP, SSPM-B alone.
## Rules that bit us this round (also in the skill)
- A 2-input norm is compute-bound and a 3-input norm byte-bound: price norm levers per input count (RRTWIN).
- A pinned dst_channel on a routed Flow into a core tile is refused ("already in use"): let the compiler assign it.
- Build the DEFAULT configuration after every flag-guarded kernel edit: an #ifdef inserted inside another's #else broke the default
  build while the flagged build failed elsewhere (fixed in a57c764).
- Re-read RESULTS_FROM_MAIN right before starting a build: research may have asked for a build-free window (my slip at 15:52).

# ROUND 24 DONE (2026-10-11) - read this block first; the round-23 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r24z**, A/B in one process vs r23z **-0.48 ms** (60 x 20: -0.476 / -0.483 / -0.508), BIT-IDENTICAL vs full12r9a
| depth | r24z sweep median (min) | whole step | r23z same session | DAVIS AJ (OA) |
|---|---|---|---|---|
| 12 | 28.43 (28.38) | 30.73 | 28.95 (28.87) | 70.84 (92.57) (bit-identical: not re-run) |
| 10 | 23.85 (23.79) | 26.03 | 24.25 (24.20) | 69.90 (91.02) (bit-identical) |
| 8 | 19.10 (19.07) | 21.56 | 19.47 (19.39) | not run |
| 6 | 14.38 (14.34) | 16.62 | 14.62 (14.58) | not run |
Box for the sweeps: load 1.4-4.1 (the background job), r7/clk.py 1.808 GHz, gpu_busy 0. Timeouts this round: 0.
Build: `r24/try24.sh NAME [--sweep]` = r23/try23.sh + STATS64 + OPPRE2 + SSMPRE + CPPRE (env STATS64= / OPPRE2= / SSMPRE= / CPPRE=0 to drop).
Text: GEMM 16,256 / 16,384, EW 15,728 (656 free). Never dispatch *.unpatched.xclbin; read SEQLOG=1 before dispatching a reordered sequence.
## What changed (NOTES 29, RESULTS 33): STATS64 (norm statistics as one 64-lane chain) -0.10; OPPRE2 (out_proj's A per attention head
group, sends before the waits) -0.15; SSMPRE (SSM head-0 params before the yx wait) -0.07; CPPRE (post-wait shim fills configured
before the wait, pushed after) -0.09. step8 timeline marks for split ops fixed; TL_ONLY fast timelines (3-4 us polls).
## Where the time goes (r24e/g, us per block): attention 491 (compute), yx + SSM ~330 (the SSM compute-bound per chunk, 2,273 cycles),
up_a + up_b ~390, mlp0 ~228, qkv ~150, norms 72-77 each (DDR read cap; norm3 near it after STATS64), out ~53 after the SSM.
## Next concrete actions (round 25)
1. SSM glue by chunk pairs (ssq2 / reck2 / ggk2, ~-0.2 at the static price, SSMIN says it lands): find ~5 KB of EW L1 (stack: deepest
   C++ 1,792 of 4,864 B, asm frames unaudited; the 15 KB input ring) and re-lay the arena (convk over-read, sigk garbage stores).
2. More CP issue cuts: a post-wait shim fill costs ~0.55 us; CPPRE on OPPRE2's parts needs BD headroom (static peak 20 > 16) - free BDs
   first (finish the attention's pass-0 group earlier?) or configure only the last part; audit other post-wait runs (r24/cprest.py).
3. KTL R 68-69 bank-clean (research: not found at <= 4 workers; try more workers in a quiet window).
4. MSPLIT-C up_a / qkv / mlp0 (research -0.05..-0.10 each; main prices pass 1 as B-bound: ~1/3 tile); needs an NRES ldn split.
5. Research rev 20's SSPM-B: re-price with SSMIN (the SSM is not input-starved) before building.
## Rules that bit us this round (also in the skill)
- A shim DMA channel queues ~5 tasks like a memtile channel: 7 drain TAPs per column + the next op's drain blocked the CP (+0.44).
- A post-wait shim fill costs the CP ~0.55 us (5 instructions); per column that staggers EW starts by ~4 us and the B multicast makes
  the next GEMM wait for the last column. Configure before the wait, push after (CPPRE); keep the static BD peak <= 16.
- Measure boundaries with TL_ONLY (3-4 us polls): the 27-us-poll timeline invented a 6 us shift.
- An earlier start of an overlapping op can lengthen it (YXH: the SSM +19 us); a drop of write bytes in a read-bound span is free (ZDROP).
- Push a split op's memtile sends before its part waits (OPPRE2 v1 vs v2: -0.04 vs -0.15).

# ROUND 23 DONE (2026-10-11) - read this block first; the round-22 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r23z**, A/B in one process vs r22z **-1.89 ms** (60 x 20: -1.894 / -1.885 / -1.875), BIT-IDENTICAL vs full12r9a
| depth | r23z sweep median (min) | whole step | r22z same session | DAVIS AJ (OA) |
|---|---|---|---|---|
| 12 | 28.89 (28.85) | 30.97 | 30.81 (30.76) | 70.84 (92.57) (bit-identical: not re-run) |
| 10 | 24.22 (24.18) | 26.01 | 25.79 (25.76) | 69.90 (91.02) (bit-identical) |
| 8 | 19.40 (19.36) | 21.30 | 20.66 (20.61) | not run |
| 6 | 14.62 (14.57) | 16.67 | 15.54 (15.50) | not run |
Quiet box for the sweep (load 2.9-4.0, only the background job; r7/clk.py 1.801 GHz just before; gpu_busy 0).
Build: `r23/try23.sh NAME [--sweep]` = r22/try22.sh + KTL (r23/mm_rot_k70p, E7_V=k70p) + MSPLIT=down_b,mlp3_b,out_proj (-DMSPLIT,
acc_hi 0x71A00) + YXSSM + YXR + ZCH + OUTPRE + OPH1 + --l2-depth 3 (env switches: MSPLIT=, YXSSM=0, YXR=0, OUTPRE=0, OPH1=0, L2=).
Never dispatch *.unpatched.xclbin. Text: GEMM 16,256 / 16,384 (128 free), EW 15,824 (560 free). Timeouts this round: 2 (both CP-order
deadlocks of my own sequences, recovered; 1 left would have been the phase limit).
## What changed (NOTES 28, RESULTS 32): KTL (research r24/mm_e7k_r70b) -0.82; MSPLIT-P (last tile as row passes rb 0-5 / 6-17 for
down_b, mlp3_b, out_proj) -0.17 + -0.10; **YXSSM** (the SSM starts while yx runs: per-head waits on yx drain groups, z panels into half
1) -0.41 and **YXR** (yx steps y / x interleaved: the SSM starts after step 2 of 8) -0.38; OUTPRE small (-0.02).
## Where the time goes: attention ~430 us per block (compute), SSM ~290 (now overlapping yx, DDR-contended: yx wall 94 -> 164 us),
GEMM k-steps (KTL R 70), the remaining GEMM -> EW and EW -> GEMM boundaries (norm1 after out, the consumers' first tiles).
## Next concrete actions (round 24)
1. out -> norm1 split: dead in this form (r23o3 +0.43 / r23w +0.14: out's 4th memtile push waits behind yx's tasks and delays the
   SSM's head fills); retry with out's pass sends pushed after the SSM's fills (a second issue point), or norm1 after a half split.
2. KTL at R 68: CP-feasible but 35 modelled bank conflicts per k-step (r23/solve_k_r68.log) = no gain; research: a bank-clean R 68-69.
3. SSM glue: gate fails (tcall ~1,930 > 1,880 without FSX); FSX needs a 12-x-register ssq schedule (DEAD_ENDS E16) - research.
4. MSPLIT-C only fits yx (the other consumers' epilogues are gated / qkv / gelu_pan); norm3 LN glue; OPPRE-W.
5. step8 timeline: MSPLIT ops write their last tile's rows twice, so per-op marks after down_b / mlp3_b are misassigned: fix op_marks.
## Rules that bit us this round (also in the skill)
- A memtile MM2S channel holds 5 tasks (1 running + 4 queued): a column's 6th push blocks on that column's own tile, whose B multicast
  needs every column's A sends issued - push every column's first five before any sixth (r23d hung).
- Reordering the CP stream: a group pushed through the lookahead can be finished early, and a CP wait on work that needs later-issued
  instructions is a deadlock (r23o2 hung); design8 SEQLOG=1 prints every group wait with the op being issued - read it before dispatch.
- An EW op can overlap its producer when its per-unit inputs are finer than the op: split the producer's drain into groups in the
  consumer's fill order (YXSSM + YXR -0.79). An on-chip panel target (NRES / SRES) must be a slot half the running GEMM does not hold.
- The GEMM cores' L1 is full (data 124 B free, bank 3 full): new constant tables do not fit; patch the existing table in place and
  check the result with a static_assert against the intended table (MSPLIT on KTL).
- --l2-depth 3..8 is neutral (3 / 5 / 6 / 8 within 0.02 ms): the B forward depth is a free memtile BD budget.
- CLKLOAD (r23/clkload12.out): the package sits at ~99 W with 12 spin or 6 vector threads at 1.81 GHz; 12 vector threads at the same
  ~99 W drop the NPU to 1.408 GHz with Tctl 59-71 C: not thermal, not package power alone (a current / vector-load rule: INFERRED).

# ROUND 22 DONE (2026-10-11) - read this block first; the round-21 block below is the base it builds on
## Working rules (unchanged): all compute on testbox (nice -n 10, <= 12 CP workers), edit on testbox, laptop only ssh + round-end rsync.
## State: **full12r22z**, A/B in one process vs r21k1 **-0.41..-0.49 ms** (30 x 20: -0.408 / -0.488 / -0.422), BIT-IDENTICAL vs full12r9a
| depth | r22z sweep median (min) | whole step | r21k1 same session | DAVIS AJ (OA) |
|---|---|---|---|---|
| 12 | 30.78 (30.74) | 32.53 | 31.20 (31.13) | 70.84 (92.57) (bit-identical: not re-run) |
| 10 | 25.79 (25.75) | 27.55 | 26.14 (26.10) | 69.90 (91.02) (bit-identical) |
| 8 | 20.71 (20.61) | 23.39 | 20.95 (20.90) | not run |
| 6 | 15.58 (15.54) | 18.01 | 15.79 (15.72) | not run |
Quiet box for the sweep (load 2.5-4, no CP running; r7/clk.py 1.806 GHz). (A first r22z sweep inside its build had d12 31.26 /
whole 34.49: a slow-mode window; the re-sweep minutes later is the number.)
Build: `r22/try22.sh NAME [--sweep]` = r21/try21.sh + A3=1 + -DTXT18 + ATTKS_JSON=r22/attks33.json (ATTKB_DM=0,1,2,4,3) +
ATTV_JSON=r22/attv20.json (each with an env switch: A3=0, TXT18=0, ATTKS_JSON=r17/attks.json, ATTV_JSON=r14/attv.json). Never dispatch
*.unpatched.xclbin. Text: GEMM 15,856 / 16,384 (528 B free), EW 15,824 (560 free). Timeouts this round: 2 (KTL; both recovered).
## What changed (NOTES 27, RESULTS 31): 3A (third A slot region: down_b / mlp3_b k 12-23 load during the previous op) -0.075,
TXT18 (research's EW text patch) -0.07, attkb (attention K loop) II 36 -> 34 -> 33 -0.13 / -0.035, attv II 21 -> 20 (3 stages) -0.04.
## Where the time goes: attention IS compute-bound in its body chunks (attkb II cuts land 1:1; chunk 0 is hidden: K2 -0.02),
GEMM k-steps (E7 R 68), the GEMM -> norm boundaries (NTWIN: norm0 / norm2 not waiting = -0.92 ms, walls shift, not grow), SSM glue.
## Next concrete actions (round 23)
1. KTL: research's mm_e7k_r70 HANGS in situ, also called once per k-step (KTL1 bisect): research's KTLP one-core DMA-fed bench
   decides (in-kernel B release, movs restarts, mul masks, no tail). Main side is ready (E7_V=k70 KS_ACC_HI=0x719C0 r22/try22.sh).
2. MSPLIT-P (research rank 4): NTWIN passed its gate (-0.92 upper bound for norm0 + norm2): the producer's last tile in row halves.
3. SSM glue (ssq2 / FSX / reck2 / ggk2, -0.25..-0.45): EW text is there now (560 B).
4. Attention: attvs (II 18 vs bound 11, 2 stages), attkb II 32 (none in 1,800 s at life 2), S3 re-priced -0.05..-0.10.
5. E7'' at R 64-66: the CP stays at its hints (nx 10 and the reweighted objective did not help); research: only after KTL.
## Rules that bit us this round (also in the skill)
- A modulo schedule can come out U 2 when lifetimes may reach 2 II: kernels whose C++ asserts U 1 need max_life = II (gen_attkb --life 1).
- `vshuffle bmXXn` refuses dm4 as the target ("invalid operand", assembler): allocators must keep vshuffle bm targets in dm0-3.
- A text lever can be a time lever (TXT18 -0.07 for "0 +- 0.02"): time every refactor.

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
