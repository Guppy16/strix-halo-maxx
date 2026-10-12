# TAPNext++ on the XDNA2 NPU — hand-off (2026-10-09, end of round 6)

Read `.claude/skills/xdna2-npu/SKILL.md` first (round 6 added rules to sections 2, 3, 7, 8, 10). Then
the ranked analysis proposals (PROPOSALS.md, not in this repo) and
`tapnext/research/RESULTS_FROM_MAIN.md` (what round 6 measured against them: P1, P2, P8, P9, P11 + the LUT work).
RESULTS.md section 15 has every round-6 number, NOTES.md section 11 the changes, dead ends and tools. Round 5's
hand-off is in r5/HANDOFF_r5.md. All timings: test machine, Turbo, GPU idle (gpu_busy 0), pyxrt `run.wait` kernel time.
No deadline (2026-10-09): research project, measured understanding over quick wins.

## 1. State: 76.3 ms at depth 12, any depth from one xclbin

### Measured
| what | number |
|---|---|
| **Step at depth 12** (`build/full12r6e`, step8.py, one dispatch) | **76.34 ms dispatch, 78.9 ms per frame incl. host** (round 5: 89.4 / 91.9) |
| per block (block 5) | 6.33 ms (round 5: 7.42) |
| **DAVIS 10 videos 'first'** | depth 12 **AJ 70.83** (r5 71.00, CPU bf16 70.98), depth 10 **69.94** (r5 69.87, CPU 70.07), depth 8 **68.25** (r5 68.28, CPU 68.22) |
| tracks vs CPU bf16 (the original 30-frame test clip (private, not included), 4 points, 4 frames) | within 0.037 px (full12r6b; r6e is bit-identical to it) |
| directed tests | t1/t2 epilogues = round 5 except GELU sites (all closer to their references now); t4 attention O rms_rel 6.99e-3 |
| timeouts this round | 1 (the core-timer read in the one-core probe; recovered, not retried) |

### The Pareto front (latency MEASURED, sweep on one xclbin; AJ NPU-measured except depth 6)
| depth | ms per frame (dispatch) | fps | DAVIS AJ |
|---|---|---|---|
| 12 | 78.9 (76.3) | 12.7 | 70.83 |
| 10 | 66.4 (63.8) | 15.1 | 69.94 |
| 8 | 53.6 (51.0) | 18.7 | 68.25 |
| 6 | 40.9 (38.3) | 24.5 | 64.77 (CPU) |
GPU (compiled bf16): 22 ms at depth 12. Depth 10 stays the default pick (15 fps, AJ ~70).

### Where a block's time goes now (block 5, MEASURED shares from the host-polled timeline of full12r6e)
attention 1.19 (the largest op), out 1.44 (= the SSM, whose last write the poll cannot see, + out's GEMM ~0.1),
up_a + up_b ~1.0, mlp0 0.54, yx 0.30, qkv 0.36, down/mlp3 4 x ~0.2, norms 4 x 0.11-0.15, out_proj 0.13.
`python3 r6/tlview.py build/tl_full12m_d12.json build/tl_full12r6e_d12.json --html build/tlview_r6.html` prints the per-op roofline + diff.

## 2. What exists (`$SHM_SCRATCH/npu-tapnext`)
- Sources: plan8.py, design8.py (GEMM workers: stack 0x1000, data_size 27,136), mm8.cc (pass-structured epilogue,
  LUT GELU, attention at -O2 with `expd_n`), nk8.h (`recipa_i`, `gelu_n`), gelu_lut.h (generated:
  `cd r6 && python3 lutgen.py -3 ../gelu_lut.h data,3`), ew8.cc (unchanged), pack8.py (`gelu_erf_ref`), run8.py
  (`<test>-<variant>` names), step8.py (`regress`, timeline JSON). Round-5 kernel sources: r6/r5src/.
- **Best build**: `EW_OPT=-O2 AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r6e --lookahead 2 --kdefs=-O2` and the
  same for full{10,8,6,1}r6e (elfs byte-identical across depths; `Step8("full12r6e", d)`).
- Tools (r6/): asmstat.py, ccs.sh, tlview.py, kprobe.py/.cc, emu.py, capture.py (build/acts_davis3.npz), numh.py,
  lutgen.py. See NOTES 11.3.
- Regression reference: `build/regress_full12r6b_d12.npz` (`./s8.sh LOG regress --build X --ref full12r6b`).

## 3. Next concrete actions (in order; each measured with the timeline + regress, accuracy by smoke + DAVIS)
1. **Attention (1.19 ms per block, the largest op).** Static per-chunk-tile split: exp ~46% (65 bundles per 64 scores:
   the fp32 hi/lo argument, magic rounding and n/f extraction dominate, not the polynomial; a 2^f table would not
   be cheaper), PV ~27% (42 bundles per 12 mmuls; the fp32 O tile is loaded and stored every 48-key chunk), QK^T ~27%
   (~70 bundles per 16 mmuls, 4 accumulators, no overlap across the 3 cb-pairs of a row block). Ideas, cheapest first:
   (a) exp from a bf16 hi/lo split of d = s - m and a table of exp over d (needs ~3 KB per copy: check bank space);
   (b) longer QK^T / PV loops (all 6 key blocks per pass, fewer accumulator round trips);
   (c) analysis P12: K/V re-streamed per pass (the second pass has 1 query tile for all K/V traffic).
2. **GEMM inner loop by hand (asm route).** P1 measured the GEMMs compute-bound (no-MAC twin -16.1 ms, 2x-MAC twin
   +24.7 ms of 84.0). The loop is 28 bundles per 16 mmuls = its ldb-slot count; a schedule holding A in registers
   and reusing B pop destinations before consumption needs ~17-18. Upper bound ~-6 to -8 ms, capped by the measured
   ~1.0 us per k-step of data movement. Use p11/gen_gemm.py + asmcheck.py (stage-3 rule).
3. **analysis P5 (k-chunk-streamed A slot)**: est. -3 ms, -144 MB DDR; medium hang risk (lock arithmetic).
4. **SSM (1.3 ms per block on the EW cores)**: the LUT sigmoid is fine for gate_x only (NOTES 11.2); exp_a2_n and
   sqrt_n are the other transcendentals; numh.py has the a-near-1 yardstick.
5. **Settle the clock** (SKILL 7: >= 1.75 GHz implied, 1.43 assumed) with P10's port microbenchmarks.

## 4. Rules that bit us this round (also in the skill file)
- `read_tm(0x340F8)` hangs the core; no cycle builtin exists.
- `minsize` hot paths: 10x; a variable trip count stops pipelining; the ldb slot bounds every pop-heavy loop.
- `aie::linear_approx` reloads its object each call; write table lookups by hand.
- Table error is absolute: a sigmoid table broke the a-near-1 accuracy of the RG-LRU (measured before building).
- The ViT MLP's GELU is the erf form in torch.
