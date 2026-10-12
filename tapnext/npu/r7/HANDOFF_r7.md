# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-09, end of round 7)

Read `.claude/skills/xdna2-npu/SKILL.md` first (round 7 added the clock, the core timer, stream re-seek and asm
GEMM rules). Then the analysis proposals `PROPOSALS.md` (revision 2) and `RESULTS_FROM_MAIN.md` (round 7
verdicts on P10, P12, P15, P16-lite, P17, P18 at the end). RESULTS.md §16 has every round-7 number, NOTES.md §12 the
changes, dead ends and tools. Round 6's hand-off: r6/HANDOFF_r6.md. All timings: test machine, Turbo, GPU idle
(gpu_busy 0), pyxrt `run.wait` kernel time. **The clock is 1.81 GHz** (measured; the old 1.43 was wrong).
No deadline: research project, measured understanding over quick wins.

## 1. State: 71.2 ms at depth 12, any depth from one xclbin

### Measured
| what | number |
|---|---|
| **Step at depth 12** (`build/full12r7g`, step8.py, one dispatch) | **71.19 ms dispatch** (time), 71.8 in the sweep; 75.0 ms per frame incl. host (round 6: 76.3 / 78.9) |
| attention, t4 directed test | 1706 -> 1506 us (O rms_rel 6.987e-3 -> 6.938e-3) |
| GEMM call (mm_acc) | 321 -> 225 bundles (asm, bit-identical) |
| DAVIS 10 videos 'first' (full12r7g) | depth 12 **AJ 70.78** OA 92.58 (r6 70.83, CPU bf16 70.98); depth 10 **AJ 69.83** OA 91.13 (r6 69.94, CPU 70.07) |
| tracks vs CPU bf16 (the original 30-frame test clip (private, not included), 4 points, 4 frames) | within 0.035 px (full12r7a; r7c..r7g are BIT-IDENTICAL to r7a) |
| timeouts this round | 0 |

### The Pareto front (latency MEASURED, sweep on one xclbin full12r7g; AJ: see the DAVIS row)
| depth | ms per frame (dispatch) | fps | DAVIS AJ |
|---|---|---|---|
| 12 | 75.0 (71.8) | 13.3 | 70.78 |
| 10 | 64.0 (60.3) | 15.6 | 69.83 |
| 8 | 52.2 (48.4) | 19.2 | 68.25 (r6 numerics; r7 changed only the attention exp: re-measure) |
| 6 | 39.7 (36.4) | 25.2 | 64.77 (CPU) |
GPU (compiled bf16): 22 ms at depth 12. Depth 10 stays the default pick.

## 2. What exists (`$SHM_SCRATCH/npu-tapnext`)
- Sources: mm8.cc (attention: table exp `aexp_*`, unscaled scores, Q popped per key block; `-DMM_ASM` calls the asm
  `mm_core`; `-DPROF` stamps), exp_lut.h (`python3 r7/explut.py exp_lut.h 0.125`), design8.py (stack 0xC00,
  data_size 28,160; `asm_object()` builds mm8.cc + r7/gen_mm.py's mm_core.s into build/mm8asm.o and links it with
  `Kernel` when the kdefs hold -DMM_ASM; bus enables for -DPROF), pack8.py (`B_PAIR4`: GEMM weights in the asm's
  pop order; step8 / run8 set it from the build's kdefs), step8.py (timeline DETAIL=bN_attn), run8.py.
  Round-6 sources: r7/r6src/.
- **Best build**: `EW_OPT=-O2 AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r7g --lookahead 2 --kdefs=-O2,-DMM_ASM`
  and the same for full{10,8,6}r7g (one xclbin serves every depth; `Step8("full12r7g", d)`).
- Tools (r7/): clk.py/.cc (clock probe, `--stamp` = core timer test), prof.py (per-tile GEMM cycle profile of a
  -DPROF build), t4var.sh (build + time a t4 variant), expemu.py (attention exp candidates on real scores),
  explut.py, gen_mm.py (the asm GEMM call: exact-cycle schedule, EDF on lda, built-in conveyor checks; run
  `../../moondream/npu/p11/asmcheck.py build/mm_core.s mm_core` after edits).
- Regression reference: `build/regress_full12r7a_d12.npz` (`./s8.sh LOG regress --build X --ref full12r7a`).
- Program memory is the binding constraint of the GEMM cores: full12r7g uses 16,176 of 16,384 B of text (208 free)
  (-DPROF together with -DMM_ASM overflows by 80 B).

## 3. Next concrete actions (in order)
1. **Why the asm call gave -1.9 ms, not the ~-7 ms its bundle count implies** (extrapolated: -96 cycles x 9 calls per
   k-step x ~16k k-steps per core). Measure in situ: a -DPROF -DMM_ASM build without the attention code (a g11
   GEMM-only test has the program memory) -> cycles per mm_core call and the k-loop wall. Candidates: memory stalls
   inside the asm (the C++ loop had none), or the k-step now meeting the data-movement time (P1: ~1.04 us per k-step
   without MACs) - then P5 / P20 / prefetch, not the loop, are next.
2. **asm GEMM, next schedule** (if 1 says the call is still the bound): skew the 4 chains so group boundaries do not
   coincide (each boundary = 16 serialised stores + loads on 5 accumulator registers: groups start every 48 cycles
   for 32 MACs; the ldb bound is 36), then P20 (one call per k-step; ~400 cycles per k-step of call + lock code).
3. **Attention** (~0.93 ms per block now): per-head-group fixed costs (~25-35 us pass 0), the exp at 49 bundles per
   64 (the 16 vldb lookups bound it at 16), P12's K/V per pass, P19 (EW cores idle during attention).
4. **SSM / EW bubbles** (P13, P7): the profile shows GEMM tiles waiting 1.37 ms for the SSM before `out` and ~0.2 ms
   after each norm. P14 (two frames in flight) for fps.

## 4. Rules that bit us this round (also in the skill file)
- read_tm needs the core's processor bus enabled (npu_maskwrite32 0x32038 bit 0 in the sequence) + sched barriers.
- Re-seeking a block-vector input stream after pops returns garbage; make a fresh stream object.
- A host-side layout switch must cover only the consumers that changed (pack8's B order hit the SSM's parameter
  tiles too: `step8.py blocks` localised it in one run).
- A blanket edit of pragmas removed two `unroll(disable)` that kept the epilogue in program memory: diff sizes.
