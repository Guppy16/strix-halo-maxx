# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-10, end of round 9)

Read `~/.claude/skills/xdna2-npu/SKILL.md` first (round 9 added LUT, lock, zero-overhead-loop and Peano pipelining rules),
then `~/git/app/.scratch/npu-research/PROPOSALS.md` (revision 4) and the round-9 part of `RESULTS_FROM_MAIN.md`.
RESULTS.md §18 has every round-9 number, NOTES.md §14 the changes, facts and dead ends. Round 8's hand-off:
r8/HANDOFF_r8.md. All timings: testbox, Turbo, GPU idle, pyxrt `run.wait`, 1.81 GHz. No deadline: research project.

## 1. State: 58.1 ms at depth 12, any depth from one xclbin

| depth | dispatch ms | ms per frame | fps | DAVIS AJ (CPU bf16) |
|---|---|---|---|---|
| 12 | 58.1 | 60.3-60.6 | 16.5 | 70.84 (70.98) |
| 10 | 48.5 | 51.0 | 19.6 | 69.90 (70.07) |
| 8 | 38.9 | 41.3 | 24.2 | not run (68.22) |
| 6 | 29.2 | 31.7 | 31.6 | not run (64.77) |
Best build `full12r9s` (+ full{10,8,6}r9s): `A_BANKS=1,2,2 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3"
AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r9s --lookahead 2 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS`.
Regress reference with the round-9 numerics: `regress_full12r9a` (r9a..r9s all BIT-IDENTICAL to it). Tracks within
0.038 px of CPU bf16 over 4 frames. Timeouts this round: 2 (device recovered both times).

## 2. What exists (new in round 9)
- r9/: ssmemu.py (SSM numerics on real gates vs torch), ssmlut.py -> ssm_lut.h, ssm9.h (table sigmoid / sqrt, exp passes),
  lprobe.py/.cc (primitive probe, every bf16), cpuemu.py (CPU model with emulated RG-LRU numerics: track noise of a
  numeric change, no build), gen_ks.py (the pipelined asm k-step + its checker + `--banks` bank-conflict model),
  ksb.py/.cc (one-core k-step bench, no DMA, hash-compares ref vs mm_ks), bankeval.py (mm_core_p layouts), x9.cc / xs.sh
  (loop-count experiments).
- Flags: ew8 `SSM_LUT`, `SSM_EXP3`; mm8 `MM_KSTEP`, `MM_PIECES`, `MM_KS` (each requires the previous); design8 pins the A
  ring locks and binds mm_kstep under MM_KSTEP; run8 `MISMAP=1` prints where a CHECK's mismatches sit.
- Program memory: GEMM core 15,840 / 16,384 B, EW core ~16.2 KB (exp passes made the ring helpers minsize).

## 3. Where the 58 ms go
- GEMM k-steps are **A-port-bound** (~4 B/cycle into each core: 10.4 KB of A per k-step = 2,590 cycles; the asm k-step
  alone needs ~1,850). K = 768 ops with few N tiles are slower again (out / out_proj 3,300 cycles per k-step).
- SSM ~0.79 ms per block (t3 946 us unprofiled): conv, recurrence, output copies and the remaining sigmoid now similar.
- attention ~0.93 ms per block; norms ~0.54; op-start waits as round 8.

## 4. Next concrete actions (in order)
1. **A delivery** (now the GEMM bound): (a) split A across the core's two S2MM channels (B shares one: research P6
   revisited; needs a 2-stream A ring in mm_ks); (b) P5a (load the K = 768 slot once) and understand why K = 768 / small-nt
   ops run at ~3.1 B/cycle; (c) longer-term: more N per A (n = 128 tiles) to halve A bytes per MAC.
2. SSM: research P27 (log_a fold + `a` table indexed by log_a's bits, exact for a >= 0.95) needs ~1.7 KB per copy: free EW
   memory first (norm_rows' 2.9 KB stack frame). Then conv / recurrence / out-copy passes.
3. Op-start waits (P5 chunked slot, P7 norm fold), acc_zero into the first k-step (mm_ks can MAC from zero), P23.
4. P14 two frames in flight (the SSM / EW work hides behind GEMMs).
5. Product: wrap Step8 for app (depth 10 default).

## 5. Rules that bit us this round (also in the skill)
- acq with a positive value is acquire_EQUAL: a "no-op acquire" with 0 waits for an empty ring (hang). Use acq.cond.
- Zero-overhead loops in hand asm: 16-B-aligned start (.p2align 4), full-form first / last bundles, init every FIFO reg.
- Test a new asm kernel in a one-core bench (r9/ksb.py) before a multi-core design: it found the bugs without timeouts.
- IRON jit caches builds by CompileTime args: different compile flags need a CompileTime key.
