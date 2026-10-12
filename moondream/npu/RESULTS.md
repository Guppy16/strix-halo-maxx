# moondream2 vision encoder on the XDNA2 NPU -- phase 1 results (2026-10-07)

## Stack
Image `npu-iron` (docker/Dockerfile): ubuntu:25.10 + locally built XRT 2.21.0 debs (base, base-dev, npu; plugin shim extracted only)
+ mlir-aie 1.4.4 wheels + llvm-aie (Peano) 22.0.0.2026092101 + python 3.13 + torch-cpu.
Container gets only `--device /dev/accel/accel0 --ulimit memlock=-1:-1` and the workspace. No host changes.
xrt-smi: "NPU Strix Halo", NPU FW 1.1.0.60, power mode Default.
Version note: mlir-aie 1.4.4's build_drivers.sh pins xdna-driver 31c01f2 (2026-08-05); the host has 273a586 (2026-01-02, XRT 2.21).
README asks for Ubuntu 24.04/24.10 + kernel >= 6.17 + python 3.12 (wheels exist for 3.11-3.14). No incompatibility hit.

## Hello NPU
vector_vector_add PASS. whole_array matmul (npu2, 8 columns) PASS for i16, bf16 and i8 at 512^3 (~0.2 ms, overhead-bound).

## Encoder (HF config, rev 2025-06-21)
SigLIP ViT: 378x378, patch 14 -> 729 tokens, 27 blocks, dim 1152, MLP 4304 (tanh GELU), 16 heads x 72, pre-LN; projection MLP
2304 -> 8192 -> 2048 (input cat([x, x])). One crop per frame in md2fast.

| GEMM | M x K x N | calls | GFLOP/call | total GFLOP |
|---|---|---|---|---|
| patch_emb | 729x588x1152 | 1 | 0.99 | 1.0 |
| qkv | 729x1152x3456 | 27 | 5.81 | 156.7 |
| attn_out | 729x1152x1152 | 27 | 1.94 | 52.2 |
| fc1 (+GELU) | 729x1152x4304 | 27 | 7.23 | 195.2 |
| fc2 | 729x4304x1152 | 27 | 7.23 | 195.2 |
| proj_fc1 | 729x2304x8192 | 1 | 27.5 | 27.5 |
| proj_fc2 | 729x8192x2048 | 1 | 24.5 | 24.5 |
GEMMs 652 GFLOP. Attention 4*729^2*72*16 = 2.45 GFLOP/layer = 66 GFLOP. Per layer: 2 LayerNorms over 729x1152, GELU over 729x4304, 2 residual adds.

## GEMM benchmark (NPU, 8x4 = 32 cores)
Design: mlir-aie's asymmetric-tile-buffering GEMM (bf16 activations, bfp16ebs8 weights, bf16 out; the kernel converts A to bfp16 in-core,
so the math is bfp16 x bfp16 with bf16 C between k=64 steps), generalised in atb/ to m=64 and arbitrary n / K-chunk.
Padding: M 729->768, K to a multiple of 64 (588->640, 4304->4352), N to a multiple of 8n.
Data: A ~ N(0,1), W ~ N(0,0.03); error vs float64 reference on the bf16 inputs (bf16 output rounding alone gives 0.17% rms).
Kernel = submit + wait of one dispatch, includes ~95 us fixed launch cost (measured with a near-empty GEMM).

| GEMM | padded | tile m,n | kernel us | useful TOPS | sync in / out us | rms rel err | max abs / max ref |
|---|---|---|---|---|---|---|---|
| patch_emb | 768x640x1152 | 64,144 | 174 | 5.7 | 8 / 14 | 1.31% | 1.5% |
| qkv | 768x1152x3456 | 64,144 | 486 | 11.9 | 14 / 42 | 1.35% | 1.4% |
| attn_out | 768x1152x1152 | 64,144 | 212 | 9.1 | 14 / 14 | 1.35% | 1.5% |
| fc1 | 768x1152x4608 | 64,144 | 618 | 11.7 | 14 / 56 | 1.34% | 1.5% |
| fc2 | 768x4352x1152 | 64,144 | 570 | 12.7 | 52 / 14 | 1.59% | 2.3% |
| proj_fc1 | 768x2304x8192 | 64,128 | 2060 | 13.4 | 28 / 99 | 1.42% | 2.3% |
| proj_fc2 | 768x8192x2048 | 64,128 | 1862 | 13.1 | 99 / 25 | 1.83% | 3.4% |
With x20 outlier channels in A: qkv 1.49%, fc2 1.70% rms.
Per encode: 55.0 ms of NPU GEMM (111 dispatches) vs 22.2 ms hot / 20.7 ms in situ for the HIP bf16 GEMMs on the GPU.
Peak reference: the same design reaches 23.6 TFLOPS (config1) and 31.7 TFLOPS (all-bfp16 config2) at 4096-scale shapes on this box;
theoretical bfp16/int8 peak ~50 TOPS. Our M=729 shapes reach 9-13 TOPS, 20-27% of peak.
Exact-precision alternatives are far slower: generic whole_array bf16->f32 (rel err 3e-7) 1.6-2.8 TFLOPS, i8->i32 2.8-3.8 TOPS.

Other overheads:
- Switching between two xclbins (hw_contexts) costs ~2.8 ms per switch (attn_out 275 -> 3012 us when alternated with qkv).
  So the encoder must be one xclbin, ideally one runtime sequence. A single tiling (n=144, K-chunk 128) for all shapes costs <=10%.
- Buffer syncs are small (8-100 us) and would mostly disappear when activations stay in NPU-visible buffers between layers.

## Encoder estimate (NPU)
GEMMs 55 ms as measured; ~45 ms with the launches folded into one sequence. Attention (66 GFLOP, hd 72, no causal mask) at an assumed
5-8 TOPS: 8-13 ms. LN/GELU/residual: 2-3 ms fused into GEMM prologues/epilogues, ~7 ms as separate passes. Total ~55-70 ms per frame.
Same work on the GPU today: ~30 ms (vision GEMMs 20.7 + vision attention ~4.5 + elementwise ~4). The GPU "encode 79.6 ms" also
contains the text prefill of the 730 image tokens (int8 GEMMs 48.4 ms), which stays on the GPU.

Pipelining (NPU encodes frame N+1 while the GPU prefills + queries frame N): the GPU per-frame work becomes ~88.7 - 30 = ~59 ms,
the NPU ~55-70 ms -> ~15-17 frames/s vs 10.9 today (+40-55%), with per-frame latency rising to ~115-130 ms.
The ceiling is the GPU's prefill + query (~59 ms -> ~17 fps); making the NPU faster than that buys nothing unless prefill shrinks too.
(These GPU-side figures come from RESULTS.md in moondream-opt; they were not re-measured because the GPU was not available.)

## Recommendation
Worth pursuing, but only after a cheap precision check, and the gain is capped at ~+50% throughput.
1. Go/no-go first (no NPU work): emulate bfp16ebs8 (shared exponent per 8, 8-bit mantissas, both operands, bf16 C every 64 k) in the
   existing GPU fast path and run bench.py on the 493 frames. Per-token int8 vision collapsed to 89.2%; block-FP should hold up much better,
   but it is the main risk. Also re-measure the GPU vision-vs-prefill split.
2. Phase 2 build (~3-5 weeks): one xclbin for the whole encoder; dispatch-time-shaped ATB GEMM with weights pre-converted and pre-shuffled
   to bfp16 at load; fused epilogues (bias, tanh-GELU, residual) that emit bfp16 activations directly; LayerNorm kernel; non-causal
   flash attention for hd 72 (pad 80), heads spread over columns, with a more accurate exp than the AIE2P exp2 interpolant (up to 6% off);
   patch embedding as a GEMM on host-im2col'd input; host side from the application process (pyxrt is cp313, the application runs a newer Python -> small C++ shim
   or rebuilt bindings), 729x2048 hand-off to the GPU.
Risks: precision; mlir-aie API churn (we track 1.4.4, driver pinned to Jan 2026); no NPU sharing/preemption story (2.8 ms context switches);
turbo power mode needs sudo; driver hangs (none seen today).

## Precision check on the GPU (2026-10-07, 493 frames, moondream-opt/bench_bfp.py + bfpemu.py)
bfpemu.bfp_quantize is bit-exact with aie.utils.bfp.quantize(rounding="conv_even") (589,824 values, npu-iron).
Vision encoder only (all vision linears incl. patch_emb and projection); text model stays HI8Lin. --nograph.

| vision linears | acc | tv fp (fn) | agreement w/ stock |
|---|---|---|---|
| HLin (today's GPU path) | 98.0% | 10 (0) | 99.4% |
| AccLin: bf16 accumulator every 64 K only | 97.8% | 11 (0) | 99.6% |
| BfpWAcc32Lin: bfp weights, fp32 acc | 97.4% | 13 (0) | 98.4% |
| BfpWLin: bfp weights, bf16 acc/64 | 96.6% | 17 (0) | 98.0% |
| BfpLin: mlir-aie ATB as benchmarked (bfp both, bf16 acc/64) | 96.6% | 17 (0) | 98.4% |
| **BfpAcc32Lin: bfp both, fp32 acc over all K** | **97.6%** | 12 (0) | 99.0% |
| BfpRotLin: + 64-wide Hadamard along K | 96.1% | 19 (0) | 98.0% |

=> The kernel must keep the accumulator in fp32 across the whole K (no bf16 C between K chunks); then bfp16 costs ~2 frames.
Hadamard rotation hurts; dropped.
GPU today: bench.py 96.3 ms (encode 87.3, query 9.0), 98.0%; --pool 16: 64.3 ms, 98.2% but fn=3 tv;
test_throughput.py 10.97 frames/s (91.1 ms/frame). Runner with GTT guard: moondream-opt/gpu_safe.sh, logs in moondream-opt/runs-1007/.


## Phase 2a: fp32-accumulating bfp16 GEMM (2026-10-07; stopped early, see below)
Kernel p2/mm_f32acc.cc + design p2/design.py: bfp16ebs8 A and B (both host-packed with round-to-nearest-even, B = weights
packed once), fp32 accumulator over the whole K (never rounded between K chunks), bf16 out. Core tile 96x96, k-step 64,
6 A sub-tiles of 16 rows per k-step; core column c = M-tile c (8 x 96 = 768 >= 729), core row = run of N-tiles.

| GEMM | best config | dispatch us (incl ~95 us launch) | useful TOPS | % of 50 | GPU us | error vs reference |
|---|---|---|---|---|---|---|
| qkv 729x1152x3456 | 96x96x64, rho 6 | 463 | 12.5 | 25% | 195 | bit-exact up to final bf16 rounding (0.0000% differ from bf16(exact); max 1 ulp; rms 1.66e-3 = bf16 floor) |
| others | not measured (stopped) | | | | | |
Phase 1 (bf16 accumulator, 1-2 ppt worse frame accuracy) did qkv in 486 us. So: precision fixed at no speed cost, but no speedup.

Why it is not faster, measured:
1. DDR bandwidth. NPU DMA probes (p2/bw.py): ~6.4 GB/s per shim stream, ~55-60 GB/s aggregate read, ~63 GB/s read+write.
   With no compute at all the qkv data movement alone takes 515-549 us (18.8 MB: A re-read once per N-pass, B once, C once).
2. Kernel: 68 bundles per 32 MACs = 47% of core peak. The fp32 accumulator costs 16 x 512-bit loads + 16 stores per 16x16 block
   per k-step on the same load unit as the A stream (floor ~34); Peano does not software-pipeline the block loop.
3. Stream ports: an fp32 96x96 tile is the largest that fits L1 next to double-buffered A/B; at 4 B/cycle per core input
   channel that caps any fp32-accumulating output-stationary design at ~67% of peak (~33 TOPS), vs ~89-100% for bf16 C.

Per-frame floors (vision GEMMs, 652 GFLOP; GPU HIP bf16 = 20.7 ms in situ):
- DDR, every operand moved exactly once, bf16 outputs: ~40 MB per block (qkv 10.8, attn_out 4.3, fc1 14.1, fc2 11.1) x 27
  + projection ~ 1.1 GB -> ~19 ms at 58 GB/s. With bfp16 outputs (epilogue packs C) ~15.5 ms; with fc1->GELU->fc2 fused on
  chip ~12 ms. Weights alone 0.23 GB -> 4 ms.
- Compute at the 67% stream cap: ~20 ms; at today's kernel (~47%): ~28 ms.
=> Standalone NPU GEMMs cannot beat the GPU's 20.7 ms; at best they match it while running in parallel (offload). Beating it
needs activations kept on chip (fusion) AND a better kernel, and DDR contention with the GPU is a further unknown.

One xclbin / few dispatches: design.py already runs any schedule of GEMMs from one runtime sequence (compile-time list of
(K, N-tiles) per op, `reps` for 27 blocks, offsets into 3 flat buffers). Not measured for the full block before the stop.
design2.py (A kept resident in memtiles, re-sent per N-tile with per-op repeat counts from the runtime sequence: cuts qkv
traffic 18.8 -> 10.8 MB) builds but deadlocked on first run -> NPU work stopped (see NOTES.md).
Power mode: still Default; `xrt-smi configure --pmode turbo` not tried.


## Phase 2b: A resident on chip, all shapes, one sequence per block (2026-10-07/08; Default power mode throughout)
Code: p2b/ (design3.py, mm_f32acc.cc with -DBLK12, pack.py, run.py, host.cpp). Details and every experiment: NOTES.md.

Power mode: still Default (turbo was not switched on during the session). Known-good phase-2a qkv (p2/q_r6p1)
re-run in Default mode: 547 us (phase 2a: 463). Run-to-run noise is per process and bimodal (same build: 423 vs 946
us in consecutive processes, every iteration inside a process agrees); numbers below are the fast mode (min over
processes), medians of 10-15 iterations, pyxrt with no wait timeout, incl. ~80 us launch.

design3 = design.py's B/C ObjectFifos + a static memtile DMA chain that keeps each column's A M-tile in two 124 KB
memtile slots and re-sends it per N-tile (per-op repeat = lock values on the load BDs; nothing can wait for data
that never comes; verified in the lowered MLIR before the first run). Replaces design2 (runtime memtile tasks),
which deadlocked; its lowered MLIR showed no channel/lock/chain fault, so the runtime-task path itself is the suspect.
Every build's first dispatch was a single run under a timeout; no timeouts or driver messages this session.

| GEMM (real K x N) | config | kernel us | total us | DDR MB (A/B/C) | TOPS (real) | % of 50 | GPU us | error vs float64 ref |
|---|---|---|---|---|---|---|---|---|
| qkv 1152x3456 | d3 (c_streams 2, BLK12) | ~340 | 423 | 10.8 (1.0/4.5/5.3) | 13.7 | 27% | 195 | bf16 floor, 0 mismatches, max 1 ulp |
| qkv, same, old kernel | d3 | ~365 | 446 | 10.8 | 13.0 | 26% | 195 | same |
| attn_out 1152x1152 | d3 | ~140 | 223 | 4.3 (1.0/1.5/1.8) | 8.7 | 17% | 80 | same |
| fc1 1152x4304 (N 4608) | d3 | ~475 | 558 | 14.0 (1.0/6.0/7.1) | 13.0 | 26% | 251 | same |
| fc2 4304x1152 (K 4352, A streamed) | d3 | ~420 | 504 | 18.7 (11.3/5.6/1.8) | 14.3 | 29% | 223 | same |
| proj_fc1 2304x8192 (N 8448) | d3 | ~1480 | 1564 | 36.9 (2.0/21.9/13.0) | 17.6 | 35% | 878 | same |
| proj_fc2 8192x2048 (N 2304, A streamed) | d3 | ~1380 | 1467 | 67.2 (42.5/21.2/3.5) | 16.7 | 33% | 1051 | same |
"kernel us" = total minus the ~80 us dispatch floor. All seven are bit-exact up to the final bf16 rounding
(rms_rel 1.65-1.66e-3 = the rounding floor, 0.0000% of outputs differ from bf16(exact), max 1 ulp).

One runtime sequence per block (c_linear: C leaves blocked, un-blocked on the host; lookahead 2):
| schedule | us | per block |
|---|---|---|
| qkv + attn_out + fc1 (1 dispatch) | 951 | vs 1227 as three dispatches |
| the same x27 blocks in ONE dispatch | 22636 | 838 |
| fc2 x27 in one dispatch | 10605 | 393 |
fc2 cannot join the block chain: memtile BDs come in two halves of 24 per channel group, and fc2's streamed A needs
24 load BDs per block pass (4608/1152 chunks x 3 N-tiles x 2); a whole-block single sequence needs runtime memtile
tasks or the fused MLP.

Per-frame estimate (NPU, encoder GEMMs only): 22.6 (3 ops x27) + 10.6 (fc2 x27) + 3.0 (projection, 2 dispatches)
= **36.2 ms in 4 dispatches**, vs 20.7 ms on the GPU (49.8 ms as 111 dispatches). Phase 1 was 55 ms.
Where the time is (qkv): NOCOMPUTE twin of the same design 369-377 us; compute adds ~50-80 us on top. The floor is
the per-core DMA input port (~4 B/cycle): 1.1 MB of B and 1.1 MB of A per core per qkv = ~176 us each, in parallel,
plus ~80 us launch plus ramp/drain bubbles. Neither aggregate DDR (19-29 GB/s used of ~58) nor shim stream count
(C over 16 streams: no change) is the limit any more.

Launch: ~76-82 us per dispatch for a 1-column, 75-word probe from pyxrt; C++ saves 15-25 us; independent of
instruction count (62200 words: same). Amortised to 2-3 us per block by the x27 sequences.

Kernel: -DBLK12 (1x2 blocks, prefetched accumulators, A panels hoisted into registers) 27 bundles per 16 MACs =
59% of core peak (was 47%); unit B (16 B pops per 16 MACs) is the saturated resource now.

Not done: bfp16-emitting epilogue and the fused fc1-GELU-fc2 (design notes in NOTES.md: the 4304-wide intermediate
fits on chip only with two passes over M, two memtiles per M-tile).

What it would take to beat 20.7 ms (phase 2c): neighbour sharing of A and B through shared L1 (halves the port floor
to ~90 us per qkv), A prefetch across ops (four half-slots or runtime tasks), bfp16 epilogue (C -44%), then the kernel
is the limit (~59-67%): qkv ~160-180 us, block ~600 us, frame ~18 ms. Marginal vs the GPU; the real win stays the
offload/pipelining (GPU free for prefill while the NPU encodes), as in phase 1's recommendation.


## Phase 2c (2026-10-08; Default power mode for every number; NPU core clock measured ~1.4 GHz)
Code: p2c/ (design3.py + mm_f32acc.cc -DBLK12, mode.py, spin.py, host.cpp). Experiments: NOTES.md "Phase 2c".
*Note (2026-10-11):* the "% of peak" figures in phases 2c and 3 assumed a ~1.4 GHz clock; the NPU runs at ~1.8 GHz (common/MACHINE.md section 1). Bundles per MAC are unaffected.

**Timing mode.** The 2x-slow processes of phase 2b could not be reproduced in ~45 processes on the idle box
(zero/random/real data, 3 BO sets per process, alternating xclbins, CPU burn before timing): BO placement, flags,
IOMMU (identity-mapped), xclbin and hw-context are all innocent. Two real effects were found instead: a DPM ramp
(590 -> 410 us over the first ~25 dispatches of a process; run.py now warms up 20) and DDR contention (six CPU
memcpy hogs: +21%; a GPU job moves 10x more). The slow processes all fell in one window of GPU activity from another workload
on the shared box -> cross-tenant contention. run.py prints gpu_busy% with every number; report min over
processes.

**Clock and launch** (spin.py: one core, kernel on resident data): 0.261 us per 364-bundle call -> ~1.4 GHz;
intercept 37 us = dispatch latency without input DMA, so ~40 of the usual ~80 us is firmware DMA-task handling.

| GEMM (real K x N) | config d6 | total us (min) | DDR MB | TOPS | % of 45.9 peak @1.4 GHz | GPU us | error |
|---|---|---|---|---|---|---|---|
| qkv 1152x3456 | BLK12, A ring 6, C 2 streams | 416 (405) | 10.8 | 14.0 | 30% | 195 | bf16 floor, 0 mismatch |
| attn_out, fc1, qkv as one sequence x27 | c_linear, lookahead 2 | 20764 (20316) per frame = 769 per block | 785 | 19.5 | 42% | 526 per block | same |
| fc2 4304x1152 x27 | A streamed | 9629 (9152) = 357 per fc2 | 505 | 20.3 | 44% | 223 | same |
| proj_fc1 2304x8192 | | 1484 (1454) | 36.9 | 18.5 | 40% | 878 | same |
| proj_fc2 8192x2048 | A streamed | 1351 (1338) | 67.2 | 18.1 | 39% | 1051 | same |
**Per frame: 20.76 + 9.63 + 1.48 + 1.35 = 33.2 ms (min 32.3) in 4 dispatches** vs 36.2 ms (phase 2b) and 20.7 ms GPU.

**Latency breakdown, image-in to encoder-GEMMs-out (today's 4-dispatch structure):** host syncs in/out 0.7 ms
(A 50 us, C 0.66 ms; in a pipeline where activations stay in NPU-visible buffers only the 0.4 MB image in and the
3 MB encoder output remain: ~60 us), 4 launches ~0.3 ms, NPU execution 33.2 ms -> ~34 ms end to end; the frame
waits ~1-3% on launches and syncs. Inside the NPU time (qkv, per op): kernel 253 us (60%), launch 80, acc_zero/store
13, C-tile stall (L1 C depth 1) ~26, A-load bubble ~20, misc ~25. The kernel at 59% of core peak is the limit:
the frame's 652 GFLOP is 14.2 ms at 100% of 45.9 TOPS, 24 ms at 59%, so in Default mode the GEMMs cannot beat
the GPU without a >= 75% kernel and small overheads. Kernel attempts this phase (K2 two-k-step 2x2, K3 half-hoisted
2x2) all spill or tie the old 68/32; the saturated resource in BLK12 is load unit B (one B pop per MAC with 8-row
blocks), and 16-row blocks need 16 loop-invariant A panels but AIE2P has 12 ex registers.

**Shared operands.** AIE2 cores read only their own, north, south and west neighbour's L1, so symmetric
horizontal sharing is impossible and the viable design is a K-split pair (same tile, two K-halves, partial summed
through shared L1; ports 1728 -> 864 cycles per k-step). Parked: ports are not the limit until the kernel is ~2x
faster. A ring depth 6, B L1 depth 3 and B L2 depth 4 all changed qkv by <= 2%.

**Phase 3 sketch (per frame, NPU, Default mode):** LayerNorm 54 x (729x1152): ~40 us each as its own pass
(DMA-bound, 1.7 MB in+out) -> 2 ms, ~1 ms if the LN cores write the bfp16 A slot of the following GEMM directly
(the LN kernel is the packer). Attention 16 heads x hd 72 x 729 tokens: QK^T with K=72 is accumulator-traffic bound
(9 k-subs per acc load/store) unless S stays in registers per 96-column chunk with online softmax; exp as an fp32
polynomial (~10 ops per 32 values, the native exp2 interpolant is 6% off) ~60 us per block; PV (K=729) is
efficient -> ~150-250 us per block, 4-7 ms per frame (GPU ~4.5). GELU fused into fc1's acc_store ~0.9 ms (fp32
tanh polynomial), residual adds and bias free in the epilogues. Patch embedding: im2col as a 3-dim shim DMA gather
of the 378x378x3 image plus a uint8->bf16 normalise kernel, then a 729x588x1152 GEMM ~100 us. Full encoder on the
NPU today: ~40-42 ms per frame vs ~30 ms on the GPU; with turbo and a 75% kernel it crosses under the GPU.

Timeouts: one (a nosync=2 probe that by construction never releases B hung its second dispatch); driver recovered,
known-good bit-exact. No other NPU incidents. Containers removed, no temp files.


## Phase 3 (2026-10-08; NPU in **Turbo**, verified per container; uncontended = gpu_busy 0% and no SLOW-MODE flag)
Code: p3/ (design3.py, mm_f32acc.cc -DBLK12 -DBLKU=2, run.py with slow-mode detection, mode2.py, spin.py).
*Note (2026-10-11):* see the clock note under phase 2c: "% of peak" here assumed ~1.4 GHz, the NPU runs at ~1.8 GHz; bundles per MAC are unaffected.

**Turbo vs Default.** Core clock 1.43 vs 1.39 GHz (+3%, spin probe); validate's 51 TOPS in both modes agrees.
Turbo mostly trims DMA/latency: the d6 set went 33.2 -> 31.4 ms per frame; with the unroll-2 kernel (d7) 29.7 ms.

**Kernel.** Peano's BLK12 loop (27 bundles / 16 MACs) is bound by its two-stage modulo schedule exposing the
pop->MAC lead-in once per block, not by resources (unit B needs 18): tried split accumulators (28), hoisting 0/2/4/6
of the 8 A panels (45/43/33/29), two-block manual unroll (74, spills), unroll 3 (92/48). Unroll 2 of the block
loop gives **51 bundles / 32 MACs = 1.59 bundles per MAC = 63% of core peak** (was 59%, 47% in phase 2a); bit-exact.
A hand-written VLIW modulo schedule (II ~18 = 89%) is the remaining step; not attempted (assembly-level, high risk).

| GEMM (real K x N) | d7, turbo | us (min) | TOPS | GPU us |
|---|---|---|---|---|
| qkv 1152x3456 | alone | 375 (361) | 15.5 | 195 |
| qkv + attn_out + fc1, x27 in one dispatch | c_linear, lookahead 2 | 18783 (18735) = 696 per block | 21.5 | 526 per block |
| fc2 4304x1152 x27 in one dispatch | A streamed | 8328 (8291) = 308 each | 23.4 | 223 |
| proj_fc1 2304x8192 | | 1326 (1324) | 20.7 | 878 |
| proj_fc2 8192x2048 | A streamed | 1247 (1242) | 19.6 | 1051 |
All bit-exact to the bf16 floor (0 mismatches vs bf16(exact), max 1 ulp).
**Per frame (encoder GEMMs, 4 dispatches): 18.78 + 8.33 + 1.33 + 1.25 = 29.7 ms** vs 36.2 (phase 2b), 33.2 (2c
Default) and **20.7 ms on the GPU**. 652 GFLOP / 29.7 ms = 22 TOPS = 48% of the 45.9 TOPS peak at 1.43 GHz.

**The slow mode, finally characterised (not root-caused).** Sub-millisecond dispatches sometimes complete at ~1 ms
multiples (946 / 1944 us for a 390 us kernel) for whole minutes, per process, GPU idle, in Default and (more often)
Turbo; 20 ms dispatches are unaffected. Eliminated: BO placement/flags, IOMMU, xclbin/context, D3hot resume, the
core clock (spin rate unchanged inside a slow process), wait(0) vs wait(), the per-iteration sync, BLAS threads,
DDR contention (separately measurable: +21% with six CPU memcpy hogs). Signature = completion notification falling
back to a ~1 ms timer path in driver/firmware. run.py flags it (SLOW-MODE) and it is amortised by one encoder-long
dispatch in a full pipeline.

**Not done this phase:** bfp16 epilogue / C depth 2 (needs n=64 tiles: the fp32 accumulator and a second C buffer
do not fit L1 at 96x96; worth ~26 us per qkv-sized op, ~2 ms per frame) and the K-split pair (ports are not the
limit while the kernel is at 63%).

**Pipelined frame-rate estimates (NPU: full vision encoder = GEMMs 29.7 + attention ~6 + LN/GELU/patch ~2 = ~38 ms;
GPU: everything after the encoder):**
- exact path (730 image tokens): GPU side 96.3 - ~30 = ~66 ms -> GPU-bound, ~15 fps (today 10.4), latency ~104 ms.
- pooled path (27x27 -> 16x16 after the encoder, the most accurate config): GPU side 64.3 - ~30 = ~34 ms -> NPU-bound,
  ~26 fps (today 15.6), latency ~72 ms.
Caveat: the two run concurrently and share DDR; the NPU's DMA-bound phases slowed 21% under CPU memcpy hogs and a
GPU prefill moves far more, so the pooled estimate is more like 20-24 fps until measured together (not measured:
the GPU was not available).

**What limits further gains:** the kernel schedule (63% of core peak; the frame's MACs alone are 22 ms at that
efficiency, 14 ms at 100%), then the per-op bubbles (C-tile stall, A load between ops, ~85 us per op), then launch.
Next step: a hand-scheduled inner loop (II 18, 89%) -> ~21 ms per frame; then the bfp16 epilogue (-2 ms) and the
K-split pair, which together would put the GEMMs at ~18 ms, under the GPU's 20.7.


## Phase 4 (2026-10-08; NPU in Turbo; uncontended, gpu_busy 0%): int8 W8A8 text-model GEMMs
Code: p4/ (mm_i8.cc, design_i8.py (design3 slots), design_i8s.py (design.py streaming + lookahead + run-length
schedule), pack_i8.py, run_i8.py). Details and every build in NOTES.md "Phase 4".

**Numerics (goal 1): done, matched.** int8 x int8 -> int32 exact; epilogue ((float)acc * sa[token]) * sw[row] +
bias[row] in fp32 in the HIP kernel's order; tanh-GELU with an fp32 exp polynomial; bf16 RNE; per-channel weight
scale max|w|/127 (>= 1e-8, RNE, clamp) and per-token max(amax, 1e-5)/127 with round-half-away as in quant_rows.
vs the numpy reference mirroring hgemm.py HI8Lin / hip/i8gemm.hip: plain ops 0.0000-0.0005% of outputs differ
from bf16(reference) (max 1 ulp = the bf16 floor); GELU ops 0.009-0.010% differ, max 2-3.5 ulp (exp polynomial vs
numpy's exp; the GPU's __expf differs from both at this level).

**Hardware fact that decides the outcome.** The AIE2P dense int8 mmul is 8x8x8 = 512 MACs per instruction, the
same rate as bfp16 (the 1024-MAC 8x16x8 form needs a sparse B); int8 panels are 64 B vs bfp16's 72 B. So int8 buys
nothing in MACs and 11% in bytes relative to the vision path. The int8 kernel (same 1x2 structure, unroll 2) is
48 and 70 bundles per 32 MACs in its two row halves (~54% of core peak).

**Shapes.** Text: dim 2048, 24 layers, fused qkv N=6144, proj 2048, fc1 8192 (GELU), fc2 K=8192; weights 1.21 GB
per frame. Orientation: weights are the per-column operand (96-row blocks, sw/bias appended), token tiles (96 tokens,
sa appended) are broadcast per row; pooled 278 tokens -> 384 slots (nt 1), exact 750 -> 768 (nt 2); 80-token tiles
were slower (703 vs 639 us: the port-bound weight stream serves fewer MACs per byte).

| text GEMM, Turbo | design | us (min) | weights GB/s | real TOPS | note |
|---|---|---|---|---|---|
| qkv pooled 6144x2048, 278 tok | i8_qkv (slots) | 660 (660) | 19 | 10.6 | slot reload per op cannot overlap |
| qkv pooled | i8s_qkv (streaming, lookahead 2) | 639 (637) | 19.7 | 11.0 | old 64-lane epilogue |
| qkv pooled | i8s_qkv_new (16-lane epilogue + GELU flag) | 971 (958) | 13 | 7.2 | epilogue ~40 us per op |
| qkv pooled, NOCOMPUTE | | 769 (767) | 16.4 | - | the data path alone |
| qkv exact 750 tok | i8_qkvx (slots, nt 2) | 1247 (1240) | 10 | 15.1 | |
| qkv exact | i8s_qkvx (streaming, nt 2) | 1508 (1278) | 8.3 | 12.5 | weights read twice |
| whole layer pooled (25 ops, 53.5 MB) | i8s_layer, one dispatch | 7630 (7617) | 7.0 | 3.9 | NOCOMPUTE twin 2750 |
| whole layer exact (750 tok) | i8s_layerx | 15330 (15320) | 3.5 | 5.2 | |
| 24 layers pooled in ONE dispatch (1.28 GB) | i8s_frame | 179728 (179712) | 7.1 | 4.0 | instruction buffer 1.85 MB |
Per frame: **pooled 180 ms measured in one dispatch (183 by extrapolation), exact 368 ms (x24)** of text GEMMs on the NPU vs the GPU's 48.4 ms (exact, in situ) and
~25 ms pooled (estimated from pool-16 encode 57.9 ms minus ~30 ms vision; not broken out in moondream-opt).
Even the DMA-only floor of the layer (2.75 ms x 24 = 66 ms) is above the GPU.

**DDR / weights finding (goal 2).** The weight stream never exceeds ~20 GB/s in these designs although the shims
can move ~51 GB/s and DDR ~58: with one token tile per core row (nt 1) every core must receive every weight byte of
its column through its 4 B/cycle input port (ceiling 8 columns x 5.7 GB/s = 46 GB/s at 1.43 GHz), and the ObjectFifo
hops (1152-byte sub-tiles, depth 2 in L1) reach only ~45% of that; the 16-lane dequant epilogue adds ~40 us per op
and the GELU ~25 us more; DMA and compute overlap poorly across op boundaries. Text prefill is bandwidth-shaped
work (0.5 MAC per weight byte pooled) that this NPU's per-core port topology does not suit: the weight bytes must
be replicated to 4 core rows per column instead of once per chip.

**Rest of the text model on the NPU (goal 4, costed, not built):** LayerNorm 48 x 278x2048: negligible (~0.5 ms);
rope on 32 of 64 dims and the KV writes: ~0.3 ms; per-token activation quant needs a full-row amax over 2048 / 8192
columns from all 8 columns' tiles -> a small reduction kernel per op (~4 us each, ~2.5 ms per frame) or host-side;
attention pooled (278 tokens, 32 heads x 64, bidirectional prefix then causal): 0.6 GFLOP per layer, ~0.2 ms per layer
at 10 TOPS -> 5 ms; exact 4.6 GFLOP per layer -> ~1 ms per layer, 24 ms; LM head for yes/no: two rows of the vocab
matrix, ~0. Full-NPU pooled frame: vision ~38 + text GEMMs 183 + rest ~8 = ~230 ms vs the GPU-only pooled 64.3 ms.
Conclusion: the text model stays on the GPU; the NPU's role is the vision encoder in a pipeline (phase 3 estimate:
~26 fps pooled, NPU-bound, before DDR-contention effects).

Timeouts: none this phase (1 of 3 used in 2c). Device healthy; containers and temp files removed; GPU untouched.


## Phase 5 (2026-10-08; NPU Turbo; GPU through gpu_safe.sh, venv-rocm714, one process at a time)
Code: p5/nputloop.py (NPU frame loop, per-frame timestamps), moondream-opt/textloop.py (pooled text-only GPU loop
from cached projected image tokens, CUDA-graphed). Logs: p5/build/{npu_joint.log, gpu_alone.log(.gpu), gpu_joint.log(.gpu)}.

### (1) Joint NPU + GPU measurement, pooled pipeline
| side | alone | during the other side | slowdown |
|---|---|---|---|
| NPU: vision-encoder GEMMs, d7 (b3x27 + fc2x27 + pfc1 + pfc2, four xclbins) | 40.4 / 40.7 ms (2 runs) | 44.8 ms (p90 46.2, 1273 frames) | +10% |
| GPU: pooled text prefill + query (text-only; prefill 24.9 + query 8.3) | 33.2 ms | 34.4 ms (prefill 25.1 + query 9.2) | +3.6% |
GPU busy 100% during its loops; GTT 11%; no other GPU process; answers unchanged. The NPU frame carries ~10.7 ms of
hw-context switches (b3x27 looped alone 18.8 ms; GEMM work 29.7 ms); attention / LN / patch embedding (~8 ms est.)
are not on the NPU yet.
**Pipelined frame rate today: 1 / max(44.8, 34.4) = 22 fps, NPU-bound** (GPU-only pooled: 15.6 fps). With the
encoder in one xclbin (no switches) plus the missing pieces, ~41 ms -> ~24 fps, still NPU-bound; the GPU side
alone would allow ~29 fps. Latency per frame in the pipeline ~45 + 34 = ~79 ms.

### (2) Hand-scheduled inner loop: not available at that point
With Peano (llvm-aie 22.0, the only compiler here) inline assembly was rejected ("unable to translate instruction: call"
for any `asm volatile`), no AIE2P ISA document (register classes, slot/latency tables, stream fill/pop
state) was available, and no assembler was shipped (no `llvm-mc`; only `llc`/`llvm-objdump`), so a hand-written .s kernel could not
be assembled at that time. The route was not attempted in this phase (later phases built it). Phase 3's six structural variants stand: BLK12 unroll 2 = 51 bundles / 32 MACs =
1.59 per MAC = 63% of core peak; the per-frame GEMM number stays 29.7 ms (GPU 20.7). The bfp16 epilogue / C depth
2 (needs n=64 tiles) was not built either.
What would have enabled the kernel: an AIE-ML v2 (XDNA2) ISA reference or an AIE2P scheduling model for Peano
(instruction latencies + slot assignment), or an `aie-opt`/Vitis `chess` toolchain that honours
`chess_prepare_for_pipelining` / `chess_loop_range` with an initiation-interval hint.

Timeouts this phase: none. Device healthy; containers removed; GPU used only through gpu_safe.sh (text loops).


## Phase 6 (2026-10-08; NPU Turbo; GPU only for the final joint re-measure via gpu_safe.sh)
Code: p6/ (design6.py, mm_f32acc.cc -DGELU, ln.py/ln.cc/runln.py, run.py --gelu-ops, nputloop.py). NOTES.md "Phase 6".

### (1) One NPU program for the 27 encoder blocks: done
design6 = design3 with one 31104-word A buffer per memtile and one lock pair, so fc2's streamed A fits the 24-BD
channel half (block chain 15 loads + 2 sends; verified in the lowered MLIR before the first run).
| program | per frame | note |
|---|---|---|
| four xclbins (phase 5 loop) | 40.4 ms | three ~2.7 ms context switches |
| pure GEMM work (phase 3) | 29.7 ms | |
| **d8_blkx27: 27 blocks, one dispatch, one xclbin** | **33.8 ms** (run.py 33.8/33.78; loop 33.8 median, p90 33.9) | all four ops bit-exact; fc2 ~550 us instead of 308 (single buffer cannot double-buffer its chunks, +6.7 ms) |
The projection (K 2304 = 2 buffers x 22 N-tiles = 44 loads) cannot join a static chain and stays on the GPU
(2 ms there; hand-off 1.7 MB instead of 3 MB). Fixing fc2's serialisation needs the runtime memtile-task path.

### (2) Encoder pieces (verified against float64 / fp32 references; moondream numerics: F.layer_norm eps 1e-5, tanh-GELU, bf16 RNE)
| piece | status | result |
|---|---|---|
| tanh-GELU fused in fc1's epilogue | built, correct, slow | 0.0226% one-ulp flips (exp polynomial), max abs 3.1e-3 of max, rms at the bf16 floor; fc1 5.1 ms vs 0.56 without |
| LayerNorm 768x1152 (ln.py, 32 cores, memtile split/join, gamma/beta fifo) | built, correct | 0.0018% one-ulp flips, rms at the floor; **246 us** per LN (55 per frame -> ~13.5 ms as separate passes) |
| patch embedding (im2col on the host, GEMM K 588->640; pos_emb add not fused) | built on design3 (d8_pe) | bit-exact (0 mismatches, max 1 ulp); **147 us** |
| residual adds, attention (16 x 72, accurate exp) | not built | costed below |
**Finding that changes the plan:** AIE2P has no native fp32 vector arithmetic. The AIE API emulates fp32 mul/add/
convert with multi-instruction sequences: ~21 bundles per 16-lane op (acc_store with two inlined GELU chains is
1806 bundles for 32 elements), i.e. ~1.3 cycles per element per op. The 40-op GELU costs 55 cycles per element
(120 ms per frame over 27 x 729 x 4304), LayerNorm 5 ops per element (13.5 ms per frame), and a softmax with an fp32
exp per score would cost ~3 ms per block (80 ms per frame). Elementwise math on this NPU must be bf16 (native
32-lane) with fp32 only for the reductions, or stay on the GPU. bf16 tanh/exp lose bf16-level agreement (output
flips of up to one ulp on a large share of values); a Pade 7/6 tanh halves the fp32 op count.
End-to-end encoder on the NPU for a real frame: not reached (attention and residual adds not built; the GELU and
LN costs above make the full tower ~34 + 13.5 + 120 + 80 ms with fp32 elementwise math, so the elementwise design
has to change first).

### (3) Joint NPU + GPU re-measure with the single-program encoder (pooled pipeline)
| side | alone | during the other | slowdown |
|---|---|---|---|
| NPU d8_blkx27 frame | 33.8 ms | 36.8 ms (p90 37.8, 1545 frames) | +9% |
| GPU pooled text-only (prefill + query) | 33.2 ms | 34.6 ms (25.1 + 9.4) | +4% |
**Pipelined frame rate: 1 / max(36.8, 34.6) = 27 fps** (phase 5 with four xclbins: 22 fps; GPU-only pooled: 15.6 fps),
latency ~71 ms per frame; the two sides are now balanced within 6%. Still missing on the NPU side: attention, LN,
GELU, residuals, patch embedding (the GPU currently does them with the vision tower).
*Caveat (2026-10-11):* the two loops were measured separately; attention, LN, GELU and the projection (~10.5 ms) were in neither, so the realistic rate is ~22 fps (EXTRAPOLATED); see moondream/book/overview.md.

Timeouts this phase: none (1 of 3 used in 2c). Device healthy; containers removed; GPU used only through gpu_safe.sh.


## Phase 7 (2026-10-08; NPU Turbo; a bf16-emulation job kept the GPU at 96-99% busy throughout, so every
## timing here is DDR-contended (labelled); correctness results are unaffected. BF16_ELEMENTWISE.md had not landed.)
Code: p7/ (mm_f32acc.cc -DGELU_BF16, ln.cc/ln.py (bf16 LayerNorm), ln_res.py (LayerNorm + residual add),
run.py --gelu-mode bf16, runln.py with references in the BF16_ELEMENTWISE numerics). NOTES.md "Phase 7".

### (1) bf16 elementwise kernels: done
| kernel | numerics | accuracy vs the bf16-chain reference | time (contended) |
|---|---|---|---|
| tanh-GELU in fc1's epilogue (-DGELU_BF16) | bf16 op chain; exp = 2^z with z from a bf16 x bf16 -> fp32 MAC, int/frac split on the conversion unit, 2^f by bf16 MACs, bf16 Newton reciprocal | 29.96% one-ulp flips, rms 3.55e-3 (floor 1.67e-3), max abs 4.2e-3 of max; vs fp32 GELU 61% one-ulp flips | 6.6 bundles per element (fp32 version 56); fc1 1168 us min vs 560 without -> ~0.6 ms per fc1, ~16 ms per frame |
| LayerNorm (bf16, fp32 mean/var via the bf16 MAC unit, bf16 rstd) | as specified: (x-mean), *rstd, *gamma, +beta each rounded | **bit-exact (0 mismatches)** | **142 us median / 108 min** (fp32 version 246) -> 55 per frame ~7 ms |
| LayerNorm + residual add (ln_res: y = LN(x + delta), x' = x + delta) | residual add in bf16 | **both outputs bit-exact** | 168 us median / 153 min -> 54 per frame ~8.5 ms, residual adds free |
Engineering notes: the shim's 10-bit wrap means a [objects][cores][x/delta][row] gather needs one-row objects
(1152-word runs are too long; two-row objects decomposed into >16 BDs); the memtile L2 object interleaves the four
cores' objects, so the gather is one 4-D pattern per column. Residual adds fused into attn_out's / fc2's GEMM
epilogues were not built: the core has only two input channels (A, B) and the residual tile would need a third
stream or 166 KB per column in the memtile; the LN pass is where the residual is already streaming.
GELU accuracy option not built: a fp32 2^f polynomial (6 emulated ops) would halve the extra noise at ~2x the cost.

### (2)-(4) Attention, the whole tower, the joint measurement: not reached
Attention needs a fused kernel (S kept in registers per 96-key chunk, online softmax, P -> bfp16 -> PV), not the
GEMM structure: with the measured costs (bfp16 GEMMs at K=72 ~45% efficient, bf16 exp ~3-4 bundles per score),
a straightforward QK^T / softmax / PV would cost ~1 ms per block (softmax 0.74 ms: 8.5 M scores x ~4 bundles / 32
cores), ~27 ms per frame. Tower estimate with everything measured or costed: GEMMs 34 (one program) + GELU 16 +
LN/residual 8.5 + attention ~27 + patch 0.15 = ~86 ms per frame on the NPU vs ~30 ms on the GPU for the same work.
The pipelined pooled design (phase 6: 27 fps with the GPU still doing the tower's elementwise/attention) remains
the practical shape until a fused attention kernel and a cheaper GELU exist.

Timeouts this phase: none (1 of 3 used in 2c). Device healthy; containers removed; GPU not used by me.


## Phase 8 (2026-10-08; NPU Turbo; GPU idle: uncontended; BF16_ELEMENTWISE verdict applied)
Code: p8/ (mm_f32acc.cc -DGELU_BF16 sigmoid form; expsplit.h; attn.cc/attn.py/runattn.py; ln*.py from phase 7). NOTES.md "Phase 8".

### 1. GELU (sigmoid form, bf16emu.py gelu_sig numerics), LN re-timed
| kernel | vs reference | time |
|---|---|---|
| fc1 + GELU, bf16 ops with a correctly-rounded exp and reciprocal (split hi/lo bf16 operands in the fp32 accumulator, MAC-unit only) | **0.998% one-ulp flips, rms 1.5e-4, max 3.9e-4 of max** vs gelu_sig | **1117 us** (560 without GELU; 492 bundles per 64 elements) -> ~15 ms per frame |
| the same with a plain bf16 Horner exp / bf16 reciprocal | 38.8% one-ulp flips, rms 3.8e-3 | 948 us |
| LayerNorm (bf16, fp32 reductions) | bit-exact | **116 us** |
| LayerNorm + residual | both outputs bit-exact | **146 us** |
The split-operand exp/reciprocal (expsplit.h) match the emulation's "correctly rounded exp from a bf16 input" to ~1%
of outputs and cost ~1.3 bundles per element; they are shared with the softmax.

### 2. Fused attention (16 heads x 729 x 72, non-causal): correct, slow
Per core 6 items of (head, 64-query tile), 32-key chunks, two passes (row max; exp / fp32 row sum / fp32 PV),
bf16 8x8x8 mmul straight from the qkv GEMM's blocked bf16 tiles, K transposed in registers, bf16 1/sum applied once
after PV. Accuracy vs the bf16emu attn reference: **rms 6.2e-3, max abs 2.0e-3 (outputs up to 0.18)**, of which
2.5e-3 is the normalise-after-PV ordering (the emulation's own softmax deviation from fp32 is 4-8e-3). Timing per
block (16 heads): scalar version 18 ms -> vectorised 4.4 -> static masks and vector max/sum state **3.78 ms**
(min 3.74) -> **~102 ms per frame** (plain-structure estimate was 27; target "far below"). The kernel runs ~14k
cycles per 32-key chunk for ~6k cycles of counted vector work (9.9 KB of spills, 288 register transposes + 576
block loads per chunk); the exp is not the bottleneck (a 3x cheaper exp changed 4.36 -> 4.18 ms), PV is 0.8 ms.
One timeout this phase (IRON `acquire(n)` = "hold n in total": `acquire(1)` twice vs `release(2)` drifted the
locks); the device recovered (known-good bit-exact). Toolchain/API pitfalls met and worked around: Peano aborts on
vectors assigned under runtime conditions (if-conversion "Bad PHI"); no fp32 64-lane transpose; sub-native 8-lane
bf16 max returns wrong values; scalar fp32 divide/compare are software calls.

### 3-4. Whole tower, joint measurement: not reached
With the measured pieces the tower would be GEMMs 34 + GELU 15 + LN/residual 8 + attention 102 + patch 0.15 =
~160 ms per frame on the NPU (GPU: ~30 ms for the same work). The phase-6 split (NPU GEMMs, GPU elementwise +
attention + projection, 27 fps pooled) stays the practical shape until the attention kernel is ~10x faster:
K^T emitted by the qkv epilogue (no per-use transposes), 64-key chunks (needs ~9 KB more L1: 48-query tiles or a
bf16 hi/lo O accumulator), spill-free inner loops (event-counter profiling), and ideally the GEMM-style fp32-accumulate
bfp16 mmul for QK^T/PV once P can be packed on chip.

Timeouts this phase: 1 of 3. Device healthy; containers removed; GPU not used.


## Phase 9 (2026-10-08; NPU Turbo; GPU idle; NPU-only work in npu-iron)
Code: p9/ (deliverable: attn15.cc + attn13.py-style design attn15.py + runattn8.py; trace tools tracesum.py/tracekv.py). NOTES.md "Phase 9".

### 1. Attention: 3.78 -> 0.675 ms per block (5.6x), 102 -> 18 ms per frame
Fused one-pass attention, 16 heads x 729 x 72, bfp16ebs8 operands with fp32 accumulation (host-packed K natural /
V^T panels, P converted on chip), online softmax with the running max in bf16, row sums from a ones panel in the PV
mmul, exact exp (magic-add integer split + Horner cubic), native bf16 reciprocal for 1/sum, K/V broadcast to the 4
cores of a column (4 query tiles of 48 rows in lockstep), one fifo object held at a time.
| version | ms/block (median / min) | rms vs bf16emu attn | note |
|---|---|---|---|
| phase 8 v2 (bf16 API mmul, two passes) | 3.82 / 3.74 | 6.2e-3 (floor rounding mode!) | 102 ms/frame |
| v3 one pass, 48x48 | 3.18 | 1.0e-2 | |
| v4 bfp operands | 2.04 / 1.97 | 1.49e-2 | P truncation bias |
| v7r + set_rounding(conv_even) | 0.887 / 0.885 | 1.03e-2 | all earlier kernels ran in floor mode |
| v9 (4 chains, bf16 max) | 0.906 / 0.901 | 9.6e-3 | |
| v13 (K / V^T objects separately: DMA overlapped) | 0.846 / 0.843 | 9.6e-3 | |
| v14 (native exp2) | 0.510 / 0.507 | 2.7e-2 | exp2 is a 4-bit approximation: rejected |
| **v15 (v14 infrastructure, exact exp)** | **0.675 / 0.672** | **9.8e-3** (6.5e-3 vs bfp-operand ref; max abs 2.0e-3 of 0.18) | **18.2 ms/frame** |
Accuracy floor with bfp16 Q/K/V operands is 8.1e-3 vs the bf16emu reference on this random test (host model), so
v15 is at the operand-rounding floor; the comparison against the phase-8 6.2e-3 is not like for like (v2 ran with
floor rounding and bf16 P normalised before PV).

Cycles per 48-key x 48-query chunk (hardware trace of core (0,2), one-column build; Turbo 1.43 GHz):
| | v9 (paired acquire) | v13 | v14 | v15 (derived) |
|---|---|---|---|---|
| K phase: scores (36 bfp mmul x 9 + conversions), max, rescale, exp, P panels | 8.4k incl. ~1.0k lock stall | 6.7k | 4.4k (native exp2) | ~5.7k |
| V phase: P [V | 1] (60 mmul x 6), O load/store | (in the above) | 1.5k | 1.5k | 1.5k |
| gap between chunks | 1002 (DMA not overlapped) | 36 | 36 | 36 |
| item boundary (begin / end / Q copy / Q objects) | 21.6k (x8 per block) | 21.6k | 5.5k | 5.5k |
| INSTR_VECTOR share of chunk cycles | 25% | 26% | 23% | ~25% |
Differential variants (full 8-column design, v15 family): exp 0.24 ms/block, PV ~0.13, scores ~0.2, DMA floor 0.29
(empty kernel) now hidden under compute; the remaining ~0.1 is max/rescale/P conversion/stream setup. The trace shows
memory/stream/lock stalls under 5% inside chunks: the gap to the ~0.3 ms the instruction count would allow is the
compiler's schedule (dependent chains, ~280 spilled words per call), i.e. the queued hand-scheduled route.

### 2. GELU: not reached (15 ms/frame unchanged). The elementary unit's exp2/tanh are too coarse to replace the
split-operand exp; the cheaper path is scheduling the existing epilogue (492 bundles / 64 elements).

### 3. Tower budget (ms per frame, 27 blocks, Turbo, NPU alone; target: whole tower <= ~35 ms to beat the pooled GPU)
| piece | phase 8 | now | target | how |
|---|---|---|---|---|
| GEMMs (qkv, proj, fc1, fc2; 63% of bfp peak) | 34 | 34 | ~20 | hand-scheduled GEMM kernel (queued) |
| GELU (fc1 epilogue, sigmoid form, split exp/recip) | 15 | 15 | ~4 | reschedule the epilogue; 492 bundles/64 elem today |
| LN + residual (x2) | 8 | 8 | 4 | fuse into GEMM prologues / fewer passes |
| attention (16 heads, 729 x 72) | 102 | **18.2** | 10 | schedule (INSTR_VECTOR 25%), 64-key chunks |
| patch embed | 0.15 | 0.15 | 0.15 | |
| **sum** | **159** | **75** | **~38** | |
The NPU tower is now ~2.2x the pooled GPU side (34 ms) instead of ~5x; attention is no longer the largest piece.

Timeouts this phase: 0 of 3. Device healthy; no containers left; GPU not used.


## Phase 10 (2026-10-08; NPU Turbo; GPU idle; NPU-only)
Code: p10/. NOTES.md "Phase 10".

### 1. Toolchain proof: Peano assembly round trip is byte-identical and links
attn15.cc -> `clang++ -S` -> asmfix.py (`##sym` -> `#sym`, `.word` -> `.4byte`; events compiled out) -> `clang -c`
-> object identical to the direct compile (cmp, sections, disassembly). Linked with IRON `Kernel(... "x.o" ...)`:
NPU output identical, 0.673 ms/block (compiled: 0.676). Hand-written .s merged with a normally compiled C++ object
by `ld.lld -r` keeps the trace markers.

### 2. Machine model: aie2p_lat.json, now common/tools/aie2p_isa_timing.json (879 instructions) + asmcheck.py (0 violations on Peano's output)
mmul/bf16 MAC: result 6, accumulator read 4 (II 2 per chain); pop 8; load 7; conversions 2 (fp32->bf16 in the st
slot); one instruction per slot (lda, ldb, st, mv, vec, alu, lng).

### 3. Hand-scheduled score block
| | Peano (attn15, ONE_CHAIN) | hand-scheduled (scores.s) |
|---|---|---|
| score block per row block | ~160 cycles (at the mmul chain bound) | 162 cycles, checker-clean |
| K-phase per chunk (trace, 1 column) | 6730 median | 6730 median |
| output | identical | identical (bit-exact) |
| ms/block (8 columns) | 0.671 | 0.672 |
The score mmuls were not the K-phase's cost; the exp and softmax bookkeeping are (see NOTES). Attention stays at
0.675 ms/block = 18.2 ms/frame.

### 4. Tower budget (ms/frame, 27 blocks, Turbo; unchanged this phase)
| piece | now | target | next |
|---|---|---|---|
| GEMMs (63% of bfp peak) | 34 | ~20 | hand-scheduled inner loop (II ~18) - not started |
| GELU epilogue | 15 | ~4 | reschedule (492 bundles / 64 elements) - not started |
| LN + residual | 8 | 4 | fuse |
| attention | 18.2 | 10 | exp block in asm (~13 vec ops per 32 lanes vs ~37 cycles now), then the max/rescale path |
| patch | 0.15 | | |
| **sum** | **75** | **~38** | |

Timeouts this phase: 2 of 3 (raw `.byte` event bundles hung the core; known-good test recovered the device).
No containers left; GPU not used.


## Phase 11 (2026-10-08; NPU Turbo; GPU idle; NPU-only)
Code: p11/. NOTES.md "Phase 11". Timeouts: 0 of 3.

### 1. GEMM inner loop (d8_blkx27 kernel, BLK12 unroll 2)
| | Peano | bound (slot classes) | hand schedule |
|---|---|---|---|
| bundles per 32 MACs | 51 (1.59/MAC) | 36: ldb slot = 32 B pops + 4 fills (89% of the MAC slot) | II 18 -> 1.32/MAC per call: WRONG (odd k-subs only) |
| | | | II 26 (3-cycle chains, stage-3 accumulator reads) -> 1.83/MAC: **bit-exact, all 4 ops** (d11_asm6) |
| frame (27 blocks, run.py) | 34.1 ms (d11_ref; d8: 33.8) | | 31.5 ms with the wrong 1.32/MAC loop; 33.6 ms with the correct 1.83/MAC loop |
Finding 1: the mmul reads its accumulator operand at stage 3, not the tables' 4: a dependent `vmac.f` 2 cycles after
its predecessor (with another mmul in between) loses the predecessor's product, and a loaded quarter landing in the
read cycle is stale. Back-to-back dependent macs at distance 2 work (phase-10 score block): the bypass is one deep. With 5 accumulator registers and 2 outputs per block the realistic hand-scheduled II is ~20-22 (1.35/MAC).
Finding 2: the GEMM frame is DDR-bound: 1317 MB per frame (A 403 re-read per N tile, B 484, C 430) at 37-39 GB/s
= 34 ms; a 17% faster loop moved the frame 7.6%. The 20 ms target needs the A traffic halved, not a faster loop.

### 2. GELU epilogue: bound 114 vec-slot ops / 64 elements vs 492 bundles today (4.3x; ~11 of 15 ms/frame)
Not scheduled this phase (see NOTES).

### 3. Attention exp block: not reached.

### 4. Tower budget (ms/frame, 27 blocks, Turbo)
| piece | now | target | what it takes |
|---|---|---|---|
| GEMMs | 34 (31.5 shown possible by a faster loop) | ~20 | halve the A traffic (DDR-bound), loop to ~1.35/MAC |
| GELU epilogue | 15 | ~4 | hand schedule: 4.3x gap measured |
| LN + residual | 8 | 4 | fuse |
| attention | 18.2 | 10 | exp block (0.13 ms/block), max/rescale path |
| patch | 0.15 | | |
| **sum** | **75** | **~38** | |


## Phase 12 (2026-10-08; NPU Turbo; GPU idle; NPU-only). NOTES.md "Phase 12". Timeouts: 0 of 3.

### 1. GELU epilogue (fc1 + GELU, 729x1152x4608, one dispatch)
| variant | bundles / 64 elements | spills | fc1 + GELU us (min) | vs reference build |
|---|---|---|---|---|
| gelu_b2 (phase 8, 2 chains) | 492 | 74 | 1108 | - |
| gelu_staged, rolled stage loops (GV 8) | (dynamic ~600) | 89 | 1177 | identical |
| gelu_staged unrolled, GV 2 / 4 / 8 | 471 / 469 / 481 | 70 / 122 / 302 | (not run: no gain) | identical (GV 8 run) |
Peano issues ~1 op per bundle regardless of the exposed ILP (allocation-bound); the slot bound is ~114 per 64. A
real hand schedule (list scheduler + register allocation over ~80-op chains) is required: estimated 150-200 per 64
-> GELU 15 -> ~5-6 ms/frame. Not done.

### 2. GEMM DDR traffic: A is read ~2x (403 vs ~200 MB inherent); half-K memtile residency (-100 MB) and a bfp16
fc1 -> fc2 hand-off (-190 MB) are the reachable cuts (~300 of 1317 MB, ~7 ms/frame). Design changes, not built.

### 3. Attention exp: not reached.

### 4. Tower budget (ms/frame, 27 blocks, Turbo; unchanged this phase)
| piece | now | target | next |
|---|---|---|---|
| GEMMs | 34 | ~20 | A residency / bfp16 fc1->fc2 hand-off (DDR-bound) |
| GELU epilogue | 15 | ~4-6 | list-scheduled asm (allocation-bound today) |
| LN + residual | 8 | 4 | fuse |
| attention | 18.2 | 10 | exp block in asm |
| patch | 0.15 | | |
| **sum** | **75** | **~40** | |
