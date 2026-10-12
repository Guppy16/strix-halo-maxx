# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-09, end of round 5)

Read `.claude/skills/xdna2-npu/SKILL.md` first (toolchain, rules, safety; round 5 added rules to §2, §3, §8). Then
the analysis proposals (PROPOSALS.md, not in this repo; they list experiments and tools:
run the worthwhile ones and record results against them). RESULTS.md §14 has every number of round 5, NOTES.md §10
the changes and what failed, REDESIGN.md the next architecture (written, not built).
All timings: test machine, Turbo, GPU idle (gpu_busy 0), pyxrt `run.wait` kernel time.

**Priorities changed (2026-10-09): no deadline. Treat this as a research project: thorough, measured
understanding over quick wins.**

## 1. State: 89.4 ms at depth 12, any depth from one xclbin

### Measured
| what | number |
|---|---|
| **Step at depth 12** (`build/full12m`, `step8.py`; 205 ops, one dispatch) | **89.4 ms dispatch, 91.9 ms per frame incl. host** (round 4: 106.8 / 110) |
| per block | 7.4 ms (round 4: 8.9) |
| **DAVIS 10 videos 'first'** | **depth 12 AJ 71.00**, depth 10 69.87, depth 8 68.28 (CPU bf16: 70.98 / 70.07 / 68.22) |
| tracks vs CPU bf16 (clip30, 4 points, 4 frames) | depth 12 within 0.021-0.027 px, depth 10 within 0.042 px |
| depth at run time | `Step8(build, depth)` + `set_depth(d)`: one xclbin, per-depth .bin (elfs + CDO byte-identical across depths) |
| host patch embedding (corrected) | 0.39 ms per frame at 8 torch threads (conv alone 0.28; 1 thread 0.93 / 1.57) |

### The Pareto front (latency MEASURED; AJ NPU-measured except depth 6)
| depth | ms per frame (dispatch) | fps | DAVIS AJ |
|---|---|---|---|
| 12 | 91.9 (89.4) | 10.9 | 71.00 |
| 10 | 77.0 (74.6) | 13.0 | 69.87 |
| 8 | 62.0 (59.7) | 16.1 | 68.28 |
| 6 | 47.2 (44.9) | 21.2 | 64.77 (CPU) |
GPU (compiled bf16): 22 ms at depth 12. Depth 10 stays the default pick (13 fps, AJ ~69.9).

### Per block today (block 5, MEASURED)
| part | ms |
|---|---|
| GEMM compute at the loop's 63% MAC rate (ideal) | ~1.65 |
| GELU chain inside the epilogues (`-DNO_GELU` differential) | 0.77 |
| other epilogue work over the GEMM rate (mlp0 0.46 / up_a 0.43 without GELU vs ~0.25) | ~0.6 |
| attention (qkv end -> attention end) | 1.35 (PV ~0.5, exp ~0.35, QK^T + row max + stores the rest) |
| SSM | 1.38 (transcendentals 0.63, other math 0.37, ring I/O 0.38) |
| 4 norms | 0.10-0.13 each (DDR-bound, ~57 GB/s) |
| gaps before ops | ~0.05-0.12 each, ~1 ms in total |

## 2. What exists (`$SHM_SCRATCH/npu-tapnext`)
- Round-5 sources: `plan8.py` (`full_step(depth)` always lays out 12 blocks; ops only for `depth`), `design8.py`
  (EW worker = one `ew_op` call; GEMM cores = one block per worker iteration; `--kdefs` accepts commas),
  `ew8.cc` (`norm_rows<LN>`, `ew_op`, SSM with `exp_a2_n`), `mm8.cc` (attention with lazy max + `expd32x2`;
  `-DNO_GELU`, `-DATTN_NOEXP`, `-DATTN_NORESCALE` timing differentials), `nk8.h` (`expd_i`, `exp32a_i`,
  `exp_a2_n`), `pack8.py` (`EW_KIND`; norm params carry eps and the kind), `step8.py` (`Step8.set_depth`, `sweep`).
  Round-4 sources are in `r4src/`. Scratch for this round: `r5/` (probes, gelufit*.py, patch_time.py).
- **Best build**: `EW_OPT=-O2 AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12m --lookahead 2 --kdefs=-O2`, and the
  same for full10m / full8m / full6m / full1m (each build only matters for its .bin; `Step8("full12m", d)` loads
  `full12m.xclbin` + `full{d}m.bin`). full1m is the depth-1 build for `step8.py blocks`.
- `t5` (patch embedding test) no longer builds: ew_op has no patch kind (the patch runs on the host).
- Commands: `./s8.sh LOG smoke|blocks|davis|time|timeline|sweep --build full12m [--depth d]`; DAVIS 10 videos takes
  ~7 min now. `./r8.sh LOG run t1|t2|t3|t4` random-data directed tests (build them first with bb.sh).

## 3. Next concrete actions (in order; each measured with the timeline tool, accuracy by smoke + DAVIS)
1. **Epilogue structure (~0.6 ms per block, the cheapest-looking item)**: `acc_epilogue` is `minsize` (no unrolling
   or pipelining). Count its loop bundles per 64-element block (build/asm/loopcount.py), then try `-O2` for the hot
   paths within the GEMM elf's budget (15,168 of 16,384 B today).
2. **GELU chain (0.77 ms per block)**: the chain is latency-bound (~60 ops, 111 bundles per vector with 2 chains).
   Before any kernel: an emulation harness for candidate formulations against the current kernel's accuracy (15%
   bf16 flips, <= 2 ulp); r5/gelufit3.py has the method. The reciprocal-free forms failed (NOTES 10.2). Try: three
   chains in one block (only 2 and 4 were measured), a shorter reciprocal (better seed, one hi/lo Newton step).
3. **Attention (1.35 ms per block)**: PV loads and stores the fp32 O tile every key chunk (~0.5 ms); the QK^T loop
   spills the Q registers (~4 bundles per mmul); `attn_end` runs `aie::inv` on 64 lanes (scalar unit) where 8 are
   needed. A one-column trace build (skill §4) would show the stall classes.
4. **SSM sigmoids (~0.37 ms per block)**, then the gaps (split A-slot load, NOTES 10.1 item 4: ~25 us per op).
5. **REDESIGN.md**: estimated ~62 ms with today's kernels, ~44 ms with 2x faster vector kernels; 4-5 weeks. Items 1-4 are needed by either design.

## 4. Rules that bit us this round (also in the skill file)
- LLVM fully unrolls a worker's constant-trip block loop (12 blocks -> 60 calls, 2 KB of program memory). Use IRON's
  endless worker loop instead, with the op kind read from the op's params: the core code then no longer depends on
  the op list, and one xclbin serves every depth.
- Row reductions over 8 lanes: `interleave_unzip/zip` + accumulator adds (native, ~40 bundles); `reduce_add` on fp32
  vectors is emulated.
- Online softmax with fp32 O and bfp16 P: update the max lazily (TAU = 8); eager rescaling cost 0.42 ms per block.
- Timeouts this round: 0.
