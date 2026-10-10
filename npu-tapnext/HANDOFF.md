# ROUND 16 IN PROGRESS (2026-10-10/11) - read this block first; the round-15 block below is the base
## State: **full12r16e 40.25 ms** median at depth 12 (min 40.13; r15m re-timed 41.50), regress BIT-IDENTICAL vs full12r9a
Build: `r16/try16.sh NAME [--ew-add ...]` (research tools/try.sh on r16e's flags: EW_DEFS = r15m's - -DSSX + -DSQK
-DINPART_INLINE -DXONE -DGGK -DSSQ; kdefs = r15m's). Every experiment: a fresh NAME; the summary prints lint, text, regress, time.
Commits (repo holding npu-tapnext/, local only): SQK 0250b26, P-IN fcb0325, XONE 0f21eca, GGK 451f962, SSQ d1db154.
## Changes this round (r16/)
- SQK (gen_sqk.py, sqkb): exact sqrt, II 13, exponent bias folded into the table (r9/ssm_lut.h takes SQK_TAB). 493 -> 321.
- INPART_INLINE (research P-IN; the flag is not PIN: a header #undef-s PIN): part pointers inline, the straddling part by cpy64.
- XONE (research X1b): x / x-1 / x-2 in one out_copy when contiguous. cp_x 432 -> 269.
- GGK (gen_ggk.py, ggkb, ggkref): both gate GEMMs in one loop, II 24. gates 655 -> 365.
- SSQ (gen_ssq.py, ssqb): ssx + sqk fused, II 30 U 2. 832 -> 715.
- Tools: r16/nrun.sh (python in npu-iron-torch with the NPU), b16.sh (build on r15m flags + EWX), try16.sh, commit.sh,
  popprobe.s + popb.py (FIFO pop probes), ggkref.py (host bfp reference / hypothesis tests).
## EW PROF r16eP (cycles per chunk): in_wait 80, conv 440, gates 365, sigk 509, ssq 715, recur 313, z 264, rest 213,
  par 26 + 125, cp_h_pan 192, cp_x 269 = 426 us per SSM op (r15m 532).
## Rules found (to the skill at round end)
- A dense 576-bit pop stream underflows at its 9th pop without a fill after every 8 pops (probes 6 / 7).
- `#ifdef PIN` is false even with -DPIN (some header #undef-s it): check new flags change the elf.
## Next
1. N1 (norm pass 1 + stats in asm), FSX / sigk II (sigk 509 is now the largest SSM phase), X1 full rotation (cp_x 269 -> ~0),
   S2 (attention, research rev 11: stack slots), round-end sweep + DAVIS.

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
