# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-09, end of round 8)

Read `~/.claude/skills/xdna2-npu/SKILL.md` first (round 8 added the L1 bank rule, the SSM / attention stamps and
the one-loop control code). Then `~/git/app/.scratch/npu-research/PROPOSALS.md` (revision 3) and the round-8 part
of `RESULTS_FROM_MAIN.md`. RESULTS.md §17 has every round-8 number, NOTES.md §13 the changes, facts, dead ends and a
designed-not-built accumulator layout. Round 7's hand-off: r7/HANDOFF_r7.md. All timings: testbox, Turbo, GPU idle,
pyxrt `run.wait`, 1.81 GHz. No deadline: research project, measured understanding over quick wins.

## 1. State: 67.9 ms at depth 12, any depth from one xclbin

| what | number (MEASURED) |
|---|---|
| **Step at depth 12** (`build/full12r8c`) | **67.89 ms dispatch** (time), 67.9-68.4 in the sweep; 70.3-72.2 ms per frame (r7g re-timed today: 72.15) |
| DAVIS 10 videos | depth 12 **AJ 70.78** OA 92.58; depth 10 **AJ 69.83** OA 91.13 (bit-identical numerics to r7) |
| tracks vs CPU bf16 | within 0.035 px (regress BIT-IDENTICAL vs full12r7a) |
| timeouts | 0 |

| depth | ms per frame (dispatch) | fps | DAVIS AJ (CPU bf16) |
|---|---|---|---|
| 12 | 70.3-72.2 (67.9-68.4) | 14.2 | 70.78 (70.98) |
| 10 | 60.6 (57.1) | 16.5 | 69.83 (70.07) |
| 8 | 48.0 (45.4) | 20.8 | 68.25 r6 numerics (68.22) |
| 6 | 36.6 (34.1) | 27.4 | 64.77 CPU |
GPU (compiled bf16): 22 ms at depth 12. Depth 10 stays the default pick.

## 2. What exists
- **Best build**: `A_BANKS=1,2,2 EW_OPT=-O2 AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r8c --lookahead 2
  --kdefs=-O2,-DMM_ASM` and the same for full{10,8,6}r8c (identical elfs; `Step8("full12r8c", d)`).
- Sources changed this round (all in place, no r7 copies: every change is bit-identical or env-gated): design8.py
  (`A_BANKS`, `ONE_SEG=1` default, EW bus enables with EW_DEFS=-DPROF, EW data_size 27,648), ew8.cc (SSM rewrite,
  -DPROF phase stamps, -DEW_NO_NORM), nk8.h (sqrt unroll), mm8.cc (-DPROF: gaps, handoff, acc_zero, attention
  head-group records), run8.py (`PROF_OUT`), r7/gen_mm.py (`--split`, `--noconf`, `--noconf-ab`), bb.sh / r8.sh env.
- Tools r8/: mmb.py/.cc (GEMM-call bank bench), gprof.py (GEMM + attention stamps: run8 tests or `--step`),
  ewprof.py (SSM phases), cmpact.py (bit-compare two run8 ACT images; SSM reference `build/t3base.npy`).
- Profiling builds: GEMM `--kdefs=...,-DPROF` fits the full step now (14.8 KB + stamps); EW `EW_DEFS=-DPROF` fits
  the full step only by 96 B (16,288 B); SSM-only tests use `EW_DEFS="-DPROF -DEW_NO_NORM"`.
- Program memory: GEMM core 14,848 / 16,384 B (1.5 KB free); EW core 15,792 B (592 B free).

## 3. Where the 67.9 ms go (per block 5.6 ms; full-step stamps + timeline)
- GEMM ops ~3.1 ms: k-loop calls ~283 cycles of 225 static (in-place accumulators: +33 bank stalls; DMA ~+12), call
  gaps 47 per call / 66 per k-step, op-start waits (A slot 27-50 us; after norms ~180-200 us; `out` after the SSM).
- SSM ~1.12 ms (EW cores only; GEMM cores idle): transcendental chains 52% of it.
- attention ~0.93 ms: K phase 66%; fixed costs only ~70 us.
- norms ~0.54 ms.

## 4. Next concrete actions (in order)
1. **SSM transcendentals** (largest lever left, ~-5 ms if 3x): sigmoid x2, exp (a, 1 - a^2), sqrt at ~80 cycles per
   32-lane vector each. Options, emulate first (r6/emu.py style, real inputs `build/acts_clip3.npz` rglru_gx/ga,
   sp from the safetensors): (a) sigmoid by table with an exact-enough x < -8 tail (inputs reach -105; gate_a feeds
   a = exp(-8 sp sigma), so a ~3e-4 error near a = 1 is a different bf16 a); (b) a and mult as functions of the bf16
   log_a; (c) exact bf16 sqrt by mantissa / exponent-parity table. Judge by the bf16 mismatch vs torch and by DAVIS
   (h is fp32 state over frames: decay errors compound). EW memory ~6 KB free for tables.
2. **P20 + the conflict-free accumulator layout** (GEMM, ~-5..-8 ms predicted): one asm call per k-step with the
   A-ring locks inside (gen_mm.py -> a stripe hardware loop), accumulators as 1 KB pieces with consecutive groups in
   different banks (NOTES 13.4). Program memory: 1.5 KB free now; P22 (c) (mm_core group loop) frees ~0.8 KB more.
3. **Op-start waits** (P5 A-slot ping-pong / chunk streaming; P7 norms folded): first tiles wait 27-50 us even
   without a dependency.
4. acc_zero folded into the first k-step (0.49 us per tile, ~0.5 ms); P23 row balance (~0.6 ms, host only: yx's
   drain needs two tasks per column for the balanced tile order).
5. Attention K phase (P12) / P19 (attention on the EW cores too: EW program memory is the blocker).
6. Product: wrap `Step8` for app (depth 10 default; `set_depth` exists).

## 5. Rules that bit us this round (also in the skill)
- Two accesses to one 16 KB L1 bank in one bundle stall; hand-scheduled code with 3 memory ops per bundle hits it
  where Peano's code did not. Check placement (`input_with_addresses.mlir`, `llvm-nm -n`) for every asm kernel.
- Sched barriers inlined into a pipelined Peano loop crash clang: stamp via a noinline function.
- A relative object path in IRON `Kernel(...)` resolves inside build/<name>.prj: use absolute paths.
