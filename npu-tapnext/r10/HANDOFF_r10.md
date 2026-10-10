# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-10, end of round 10)

Read `~/.claude/skills/xdna2-npu/SKILL.md` first (round 10 added runtime memtile tasks, the router multicast bug, the
median rule and the forced-II warning), then `~/git/app/.scratch/npu-research/PROPOSALS.md` (revision 5) and the round-10
part of `RESULTS_FROM_MAIN.md`. RESULTS.md §19 has every round-10 number, NOTES.md §15 the changes, facts and dead ends.
Round 9's hand-off: r9/HANDOFF_r9.md. All timings: testbox, Turbo, GPU idle, pyxrt `run.wait`, 1.81 GHz. No deadline.

## 1. State: 53.9 ms at depth 12, any depth from one xclbin

| depth | dispatch ms | ms per frame | fps | DAVIS AJ (CPU bf16) |
|---|---|---|---|---|
| 12 | 53.8 | 56.3-56.9 | 17.6 | 70.84 (70.98) |
| 10 | 45.4 | 49.2 | 20.3 | 69.90 (70.07) |
| 8 | 36.0 | 38.5 | 26.0 | not run (68.22) |
| 6 | 27.1 | 29.6 | 33.8 | not run (64.77) |
Best build `full12r10r` (+ full{10,8,6}r10r): `A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3"
AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r10r --lookahead 2 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS`.
Numerics BIT-IDENTICAL to round 9 (`regress --ref full12r9a`); DAVIS identical. Timeouts this round: 1 (recovered).

## 2. What exists (new in round 10)
- Flags (plan8 / design8 env, recorded in build/<name>.json; step8 / run8 apply them): A_HALF (single A fills, reuse_a for
  up_b; with A_RT off it also multicasts into the two slot halves), A_RT (memtile A slot by runtime tasks), P23 (round-robin
  tiles for yx / qkv). mm8 `-DGELU_II` (rejected, off).
- r10/: gdist.py (per-tile medians of PROF records), rtm.py/.cc (runtime memtile task bench, also the memtile->core rate:
  7.95 B/cycle), pf1-5.mlir (router multicast bug), gx.cc / gxb.py (GELU loop bundles / one-core cycles), mmasm.sh, bak/
  (round-9 sources before the edits).

## 3. Where the 53.9 ms go (timeline r10r, per block ~4.45 ms)
- attention 0.75 + 0.15 gap (10.9 ms per step), SSM + out 0.86 (10.3), norms 4 x 0.11-0.13 (6.6), GEMM k-steps ~1.47
  (steady k-step 2,055 cycles + ~70 between calls), GELU epilogues ~0.27 (gelu_pan 8.19 us / tile, gated 7.03, yx gelu
  ~6.1; the deployed GELU loop is 34 cycles per 32 lanes), op-start waits now small (~20-60 us per op incl. the first tile).

## 4. Next concrete actions (in order)
1. **GELU epilogue by a hand modulo schedule** (gen_ks style generator for elementwise chains, reusable for the SSM chains,
   research P29): floor ~16-20 cycles per 32 lanes vs 34-44 (n = 8 calls): -1.2 to -1.5 ms. Bench on one core (r10/gxb.py)
   before building in; gated / gelu_pan also spend ~6 k cycles per tile on products + bfp pushes (look at those loops).
2. **SSM**: P42 (EW stack 6,144 -> ~4.4 KB; deepest path 4,224 B) then P27 (a by table, one copy in bank 2 first).
   EW text 16,240 / 16,384 B: the fused pass must replace loga_p + exp_pA..D.
3. **P7a** numerics emulation (no device), then the norm fold (largest lever: -4.6 to -6.8 ms predicted).
4. Attention K phase (P12 / P19, -4 ms predicted); P35 (B locks in mm_ks), P24b (acc_zero fold), P38 counters.
5. P14 two frames in flight; product wrap of Step8 for app (depth 10 default).

## 5. Rules that bit us this round (also in the skill)
- Never read a k-step rate from means over tiles: per-tile medians (r10/gdist.py).
- IRON pathfinder: multicast + unicast to the same DMA channels drops a destination (hang). Check masterset / rules.
- A runtime-task BD needs a lock acquire AND release (or neither); push an op's tasks before any wait of that op.
- A forced pipeline_initiation_interval can be 3x slower on the core than its bundle count: bench it.
