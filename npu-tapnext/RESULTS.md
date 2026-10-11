# TAPNext++ step on the XDNA2 NPU — results (2026-10-08/09 overnight)

MEASURED unless marked EXTRAPOLATED. NPU turbo (1.43 GHz). The GPU was at 96-100% from another agent's job all
night, so every NPU timing below is DDR-contended (the moondream phases saw +10% or worse under GPU load) and every
GPU ms/frame in the DAVIS logs is meaningless. Raw: build/*.log, logs_davis_*.txt, davis_npu.jsonl.

## 1. What runs on the NPU tonight (random data, bit-level checks against fp32 references)
| piece | program | shape | ms per step (median / min) | check |
|---|---|---|---|---|
| all 96 trunk GEMMs of a step (the 8 GEMMs x 12 blocks), image rows only | tn_gemm_A (design6, m 128 x 8 columns = 1024 rows, n 64, k 64, RA_K 768, 14 ops x 12 reps, one dispatch) | 1024 x {768 -> 1536, 768, 6144(=4x768 K-chunks for the two K=3072 GEMMs, see below), 2304, 3072 ...} | **36.1 / 35.2** (contended; 873 MB/step at 24 GB/s: A 149, B weights 215, C 510) | all 14 ops bit-exact vs the fp64 reference rounded to bf16 (0.0000% mismatch) |
| all 1072 rows (1024 image + 48 query, padded to 1280 = 8 x 160) | tn_gemm_D (m 160, n 48, rho 10, -Os) | - | NOT BUILT: core program memory over by 3.6 KB (16 ops x 10 calls per k-step of control code) and L1 over by 3.6 KB | the open item for the query tokens; EXTRAPOLATED: 1280/1024 x the 1024-row time = ~44 ms contended |
| attention, 12 heads x 1072 tokens x 64 dims (all 12 blocks of a step) | attn_tn (attention v15 re-shaped: 8-panel groups, 23 query tiles -> 6 passes, 72 (head, pass) items dealt to 8 columns, K/V broadcast) | 12 x 1072 x 64 | **1.15 / 1.14** per step (12 attentions) | rms 1.7e-2 vs the bf16emu reference on random data (1.5e-2 vs the bfp-operand reference; moondream's 729-key case was 1.0e-2 - more keys = more diffuse = larger relative error on random data); DAVIS says it does not matter (section 3) |
Notes: (a) the two K=3072 GEMMs (ffw_down, mlp down) run as four K=768 chunks each because design6 caches one
768-deep A tile in the memtile and its static BD chain cannot hold the 12-BD streamed variant (the 22-BD half);
the four bf16 partial outputs must be summed outside (fp32 on the host or in the NPU elementwise kernel): a numerics
deviation from "fp32 over the whole K" that the DAVIS run in section 3 does NOT yet include. (b) The 48 query
rows: the CPU cannot take them (32 ms/step for 48 rows through the 96 GEMMs, bf16, 8 threads), so the NPU program
must cover 1072 rows -> m 160 (1280 rows, 84% useful).

## 2. The rest of the step (not run on the NPU tonight; EXTRAPOLATED from the moondream kernels)
| piece | per step | basis |
|---|---|---|
| GELU (88.9 M elements) | ~15 ms (~5 after the list-scheduled asm) | phase-8 fc1 epilogue: 492 bundles / 64 elements |
| 48 norms (RMSNorm x2, LayerNorm x2 per block) | ~10 ms | phase-7 bf16 LN kernel rate (116 us per 729 x 1152) |
| RG-LRU + conv1d + gates (12 x [1072, 768], fp32 state) | ~5-6 ms | DDR: ~200 MB of state read + written per step; the arithmetic is ~1 ms of bf16 vector work |
| residual adds, K-chunk partial sums, bfp packing of activations between ops | ~3-5 ms | elementwise passes over 1072 x 768 (fused into the norm kernels ideally) |
| heads (48 rows) | < 0.1 ms | CPU |
| dispatch / context switches | 0.1 ms per program; 2.7 ms per xclbin switch | one program per step is the target; tonight's pieces are 2 xclbins |

## 3. Accuracy (TAP-Vid DAVIS 'first', 10 videos, vendored model, GPU emulation of the NPU numerics)
| numerics | AJ | OA | <thr | note |
|---|---|---|---|---|
| vendored model, bf16 trunk (what runs live) | 70.54 | 92.49 | 83.29 | per-video AJ within 0.3 of the repo model's 30-video bf16 run (66.35 on all 30) |
| + bfp16ebs8 operands on all 96 trunk GEMMs, fp32 accumulate, bf16 out (= tn_gemm) | 70.47 | 92.39 | 83.29 | -0.07 AJ |
| + bfp16 Q/K/V/P in the attention (= attn_tn, v15 numerics) | 70.37 | 92.29 | 83.30 | -0.17 AJ vs bf16 (per video within 0.2-0.4) |
| **the real NPU path (npu_step.py): trunk GEMMs + attention executed on the NPU, rest CPU bf16** | **70.68** | 92.50 | 83.32 | +0.14 AJ vs bf16 (per video within 0.3; camel +1.0, car-roundabout +2.2, car-shadow -1.2); 4.4 s/frame (prototype, section 5) |
Not yet emulated: the four-chunk K=3072 partial sums (bf16 partials), the NPU GELU/LN/exp numerics (all bit-exact or
within 1 ulp of the bf16 emulation in the moondream phases).

## 4. Step budget on the NPU vs GPU
| | ms per step | basis |
|---|---|---|
| GPU today (compiled max-autotune, bf16, uncontended) | 22.6 (eager 31.2) | tapnet/RESULTS.md |
| NPU, pieces measured tonight (GEMMs 1024 rows + attention) | 37.3 contended | section 1 |
| NPU, whole step, EXTRAPOLATED | ~75-80 ms now (GEMMs ~40 for 1280 rows + attention 1.2 + GELU 15 + norms 10 + SSM 6 + glue 4), ~55-60 after the GELU/LN work queued in npu-moondream | sections 1-2 |

## 5. End-to-end step with the NPU doing the trunk GEMMs and the attention (scratch prototype, MEASURED)
`npu_step.py` (runs in the `npu-iron-torch` image = npu-iron + torch CPU + safetensors): the vendored model runs on the
CPU in bf16; every trunk `F.linear` goes to the NPU for the 1024 image rows (five single-op xclbins, K 768, N tiles 3/6/9/
12/24; a K=3072 Linear = four K=768 runs summed in fp32 on the host) and to the CPU (bf16) for the 48 query rows;
`F.scaled_dot_product_attention` goes to the NPU attention program for all 1072 tokens. Everything else (patch
embedding, norms, GELU, RG-LRU, conv1d, residuals, heads) stays on the CPU.
| | per step | note |
|---|---|---|
| NPU kernel + dispatch time | 350 ms | 168 GEMM + 12 attention dispatches; each op is its own dispatch on one of six xclbins, so ~2 ms per dispatch of hardware-context switch and DMA setup dominate - the kernels alone are ~37 ms (section 1) |
| host bfp16 packing of the activations | 3.5 s | numpy codec (aie.utils.bfp), 168 x [1024, 768] + the attention panels; a torch version was not bit-equal and not faster; the real fix is NPU kernels that emit bfp panels |
| the 48 query rows on the CPU (bf16, 12 threads) | 50 ms | |
| CPU elementwise / SSM / heads (bf16) | ~0.5 s | |
| **total** | **4.4 s** | a correctness vehicle, not a latency one |
Correctness on 3 real frames (test clip, 4 points): tracks within **0.03 px** of the all-CPU bf16 step, visibility
logits within 0.2. DAVIS with this path: section 3 (row "npu_real").

# Session 2 (2026-10-09, daytime) — stage 0 baseline, transfer costs, depth frontier

## 6. Stage 0: trunk GEMMs with the GPU idle (MEASURED, Turbo)
| build | gpu_busy | kernel ms median / min | DDR MB | GB/s | check |
|---|---|---|---|---|---|
| tn_gemm_A (m 128 x 8 cols, c_linear 1, 14 ops x 12 blocks, one dispatch) | **0%** | **28.3 / 26.8** | 873 | 30.8 | all 14 ops bit-exact (0.0000% mismatch vs fp64->bf16) |
| same, overnight | 96-100% | 36.1 / 35.2 | 873 | 24 | same |
So the GPU costs the GEMM phase +28%. Neither DDR-bound (NPU reads 55-60 GB/s idle) nor MAC-bound (13.5 ms at the
measured 63% MAC-slot rate): the A re-reads, the C writes (510 MB, the largest stream) and per-op fill/drain bubbles
share the blame; the one-column trace (stage 0's second half) was not taken - design7 changes the op set anyway.

## 7. Host <-> NPU transfer costs (MEASURED, pyxrt, 300 runs each; GPU at 93-94% during this run, so DDR-contended)
`xfer_run.py` with the trivial `xfer_<n>.xclbin` programs (8 cores xor-fold their column's input, 256 B out per column).
| what | median | p99 | notes |
|---|---|---|---|
| launch floor: run() -> wait(), 4 KB in / 2 KB out | **97 us** | 116 | the per-dispatch cost of any program; the 1-column probe saw 58 us min |
| launch incl. DMA of a 256x256x3 uint8 frame (192 KB) | 101 | 115 | the frame DMA is ~free next to the floor |
| launch incl. 256 px bf16 frame (384 KB) | 100 | 117 | |
| launch incl. 720p uint8 (2.7 MB) | 128 | 200 | |
| launch incl. 1080p uint8 (6.2 MB) | 186 | 1999 | p99 hit a slow-mode tick |
| upload 192 KB uint8: bo.write + sync (host_only) | 14 + 13 = 27 | 33 + 19 | mapped np.copyto instead of write: 11 + 14 |
| upload 384 KB bf16 | 37 + 27 = 64 | 67 + 38 | mapped: 20 + 28 |
| upload 2.7 MB (720p) | 298 + 96 (host_only); 119 + 50 (cacheable) | ~400 / ~240 | cacheable bos are faster to write at MB sizes |
| upload 6.2 MB (1080p) | 579 + 169 (host_only); 297 + 122 (cacheable) | ~800 / ~660 | |
| read back 576 B (48 points x (y, x, vis) fp32): sync + read | 0.2 + 0.5 us | 1.6 | effectively free |
| read back 1.5 MB (a [1024, 768] bf16 activation) | 26 + 59 us | 44 + 200 | only relevant to the prototype |
| end to end: frame (192 KB) write+sync -> run -> result sync+read | **94 us** (run 77) | 137 | host overhead on top of the kernel ~20 us + the launch floor |
Flags: host_only vs cacheable make no difference below 1 MB; sync is never zero (12-27 us per sync of a frame), so a
step's fixed host cost is ~110-150 us (frame sync + launch + result). Preallocate and map the frame bo once; copy
into it with numpy; one sync; one dispatch; the result sync is free.
Resize + normalise on the CPU (`cpu_resize.py`, torch, bilinear, no antialias): 256 px source 0.4-0.6 ms, 720p
0.5 ms, 1080p 1.0 ms single-threaded (antialias: 1.8 / 5 ms). Uploading the full frame instead costs 0.25-0.75 ms
more than the 256 px upload, and an NPU resize kernel would still have to run: **resize on the CPU** (bilinear, as
`frame_to_model` does today), upload 192 KB uint8 (or 384 KB bf16) - the normalisation can move into the patch
embedding GEMM later (fold /127.5 - 1 into the conv weights + bias).

## 8. Depth / support-point frontier (MEASURED accuracy, CPU bf16 vendored model, 10 DAVIS videos; latency see 9)
`davis_cpu.py`, same loader / queries / metrics as the NPU prototype; `davis_cpu.jsonl`.
| config | AJ | OA | <thr | vs depth 12 | 30-video GPU run (tapnet, repo model) |
|---|---|---|---|---|---|
| depth 12 | 70.98 | 92.76 | 83.42 | - | 66.28 |
| depth 10 | 70.07 | 91.21 | 83.59 | -0.9 | 65.48 (-0.8) |
| **depth 10 + 8 support points / query (radius 8 px)** | **71.33** | 91.66 | 84.48 | **+0.35** | 66.54 (+0.3) |
| depth 8 | 68.22 | 90.90 | 82.14 | -2.8 | 63.47 (-2.8) |
| depth 6 | 64.77 | 90.25 | 78.12 | -6.2 | 60.69 (-5.6) |
(The 10-video CPU bf16 depth-12 number, 70.98, is 0.4 above the GPU-emulated bf16 70.54 and 0.3 above the real-NPU
prototype 70.68: the three differ only in reduction order; treat +-0.4 as noise on 10 videos.) Layer dropping =
run the first `depth` blocks, skip the rest (the upstream notebook's trick; `model.blocks[:depth]`).
Query slots: on the NPU the row count is padded to 8 x 144 = 1152 (design7), so **up to 128 query slots are free**
(1024 image + 128); the support-point trick (8 extra co-tracked points per real query, discarded on output) costs
nothing for up to 14 real queries. Input size: the 256 px checkpoint accepts nothing smaller (pos-emb), 512 px is a
different checkpoint at 4x tokens - not a knob.

## 9. Step latency per depth (NPU path; MEASURED pieces, EXTRAPOLATED step)
Per block the NPU pieces measured so far: GEMMs 28.3/12 = 2.36 ms (idle) and attention 1.15/12 = 0.1 ms. The fused
design7 step (NOTES.md) is estimated at **~40 ms for depth 12 idle** (GEMM ~29 with 24 cores, elementwise ~7 on the
8 EW cores, attention ~2, ~2 of op-boundary bubbles) and scales linearly with depth: ~34 (10), ~27 (8), ~21 (6).
Sustainable fps = 1 / step while the camera and WebRTC deliver (the phone is capped at 30 fps today), so the
**frontier points are depth 10 + 8 support points (best accuracy, ~34 ms / ~29 fps extrapolated) and depth 8
(~27 ms / ~37 fps, -2.8 AJ)**; depth 12 is dominated by depth 10 + support; depth 6 loses 6 AJ for 6 ms.
All step numbers are extrapolated until design7 runs; the GPU path's measured 22 ms (depth 12) / 18.8 (depth 10)
stays the reference until then.

## 10. design7 — the fused-step frame (MEASURED, Turbo, GPU idle unless noted)
`design7.py` + `mm7.cc` + `ew7.cc` + `pack7.py` + `run7.py`; builds `build/s7_*`.
| build | what | kernel ms median / min | check |
|---|---|---|---|
| s7_t1 | [768x768 (nt 4)], [1536x768 (nt 4)], 1 rep: 3 GEMM rows, m 144 (1152 rows), bias epilogue, packet-switched A loads, K=768 double-fill, K=1536 chunk | 3.16 (cold) | both ops bit-exact incl. bias |
| **s7_block** | one block's 11 GEMM ops x 12 blocks (yx 1536, out 768, up 2 x 3072, down 2 x K1536, qkv 2304, out_proj 768, mlp0 3072, mlp3 2 x K1536), 24 cores | **35.1 / 35.0** | all 11 bit-exact; DDR 981 MB (A 263 = 2x for the 7 K=768 ops, B 230, C 488) at 27.9 GB/s, 11.7 real TOPS |
| s7_g12 | [768x768 nt 4] x 12 | 1.99 / 1.59 | bit-exact; = 166 us per small op (0.68 GMAC: the per-op fixed cost - slot load latency + drain tail - dominates small ops) |
| s7_ew1 | [rmsnorm -> E], [GEMM 768x768 from E], 1 rep: the first dependent chain through the elementwise core | 3.4 (cold) | panels bit-exact vs numpy (torch bf16 numerics + device bfp semantics), GEMM bit-exact on them |
| **s7_ew12** | the same x 12 | **2.83 / 2.81** | bit-exact; **(2.83 - 1.99) / 12 = 70 us per RMSNorm op incl. the dependency bubble** (port-bound: 221 KB in + 124 KB out per core at 4 B/cycle = 61 us if not overlapped) |
Compared with tn_gemm_A (32 cores, 1024 rows, no bias): 35.1 ms for 1152 rows on 24 cores = 30.5 us/row vs 27.6
us/row - the query rows and the 3-row split cost 10% per row. The doubled A reads (+130 MB) can go later with the
memtile self-copy (NOTES 7).
Extrapolated step from these: GEMM 35 + 4 norm-type EW ops x 12 x 70 us = 3.4 + SSM (~4, port-bound) + attention
(~2) + qkv/O relayout (in epilogues, ~0) = **~45 ms depth 12 idle**, before the per-op bubble work (NOTES 7).

## 11. Elementwise numerics facts (MEASURED)
- The AIE2P fp32 -> bfp16ebs8 conversion (`to_vector<bfp16ebs8>()` / `vst.push.conv`) rounds to nearest even at the
  block's max exponent and, when a mantissa would round to +128, bumps the shared exponent and re-rounds every
  mantissa from the exact values; a mantissa of -128 is kept (two's complement). `aie.utils.bfp.encode` instead
  saturates at +127 (and bumps nothing), so device-produced panels differ from host-encoded ones in ~2.5% of panels
  (one row by one mantissa step). `pack7.encode_hw` reproduces the device bit-exactly (0 of 13,824 panels differ);
  use it for any reference that models a device-side conversion.
- RMSNorm with torch's bf16 numerics (bf16(x^2) summed in fp32, mean -> bf16, + eps -> bf16, rsqrt -> bf16 (16-lane
  fp32 Newton), x * rstd -> bf16, * (scale + 1) -> bf16) was bit-exact against numpy on 1152 x 768 random rows:
  no rsqrt rounding flips observed in this sample.

# Round 3 (2026-10-09) — design8: the fused step

## 12. design8 (MEASURED, Turbo, GPU idle unless noted; random data; run8.py tests)
### 12.1 Correctness of the pieces (one dispatch each, 1152 rows, checks on the 1072 real rows)
| test | ops | kernel ms | result |
|---|---|---|---|
| t1 | GEMM -> RMS(2 in) -> GEMM -> LN(2 in) -> GEMM(GELU) | 1.4 (min, 2 iters) | GEMMs and X' sums bit-exact; RMS panels bit-exact; LN panels 0.09% differ (<= 2.6 ulp of row max); GELU 15% flips, <= 2 ulp |
| t2 / t2a-f | gated pairs x2 -> K=1536 x2 -> LN(3 in) -> GELU->panels -> K=1536 x2 | 5.0 (min) | gated/gelu_pan panels 1.2-1.8% of mantissas differ (<= 6.6 ulp of row max); all K=1536 GEMMs from device panels bit-exact; LN3 panels 0.1% |
| t3 | SSM (first=0) -> GEMM from z | 2.6-4.7 (5 iters; run-to-run variance) | C0/C1/C2 shift register bit-exact; h rms_rel 4e-3; z panels: 10% of mantissas, 55% of panels differ by >= 1 step |
| t5 | patch embedding -> RMS(pe + pos) -> GEMM | 6.8 (cold) | patch bit-exact vs bfp-emulated conv; norm bit-exact |
| t4q | qkv GEMM with the object epilogue (12 drains/column) -> GEMM | 4.5 (cold) | Q, K, V^T objects bit-exact |
| t4 | qkv -> attention (2 passes) -> out_proj | 5.8 (cold) | runs; out_proj bit-exact on the device O; O wrong (12.2) |
### 12.2 Attention output (t4, `attn_dbg*.py` on `t4_dbg.npz`)
1072 keys: corr(got, ref) 0.69, rms_rel 1.0, row norms 1.40x. 48 keys (`--attn-keys 48`): corr 0.91, rms_rel 0.41,
row norms 1.001x; every got dim best-matches its own ref dim (corr 0.7-0.86), uniform over dim blocks, heads and
query tiles; not the scale (peak at 1/8), not key permutations, not bfp-P quantisation, not block transposes.
### 12.3 Timing (5 timed iterations after 3 warm-ups unless noted)
| build | content | ms median / min | notes |
|---|---|---|---|
| g11 | the block's 11 GEMM ops x 12 (all epilogues, deps, lookahead 2), `-Os` | 64.8 / 64.7 | s7_block (design7, no GELU, no deps): 35.1 |
| g11o2 | same, mm8 at `-O2` | 61.7 / 61.5 | `-Os` costs ~5% |
| **full12r** | **patch + 12 blocks + encoder norm, 206 ops, one dispatch** (`--reclaim-runtime-bds --lookahead 1`) | **269.1 / 269.1** | .bin 7.5 MB; ACT 324 MB, W 232 MB |
| full12na | same without the 12 attention ops | 252.5 / 252.4 | attention = 1.4 ms/block in situ |
| full12ns | same without the 12 SSM ops | 188.5 / 188.5 | SSM = 6.7 ms/block in situ |
| t3 / t3ng / t3o2 | SSM op: baseline / gates stubbed / EW kernels `-O2` | 2.59 / 4.71 / 3.07 | run-to-run variance exceeds the differences: not interpretable |
Host side unchanged from round 2 (launch floor 97 us). Timeouts this round: 10, device recovered every time.

# Round 4 (2026-10-09) — a correct step with the real weights, then speed

## 13. design8 with real weights (MEASURED, Turbo, GPU idle (gpu_busy 0); step8.py, one dispatch per frame)
### 13.1 Correctness
| test | result |
|---|---|
| attention `t4` (the round-3 late fix in mm8.cc: g_srb 384 -> 768 B; the maxima aliased score blocks 3-5) | O rms_rel 7.2e-3, corr 1.000, every head/tile/column 0.01; out_proj bit-exact |
| `step8.py blocks --build full1a` (depth 1, real weights, clip30 frames 0-1, every region vs the torch block) | after the GELU fix every region corr >= 0.9995, rms_rel 0.6-3% (bfp operand noise); before it y corr 0.89 |
| **GELU bug**: `nk8::gelu32` clamped x to [-8, 8] and multiplied the CLAMPED x (gelu(39) = 8): random-data tests never hit it | fixed (the clamp only bounds the exp argument) |
| `step8.py smoke --build full12e` (depth 12, 4 frames, 4 tracked points vs the vendored model on the CPU in bf16) | **tracks within 0.030-0.055 px**, query-token corr 0.9995-0.9999 |
| **DAVIS 10 videos, first** (`step8.py davis --build full12a`) | **AJ 70.72** (OA 92.48, <delta> 83.38) vs CPU bf16 70.98 (davis_cpu.py), GPU bf16 70.54, prototype 70.68; 282 ms/frame incl. host |
### 13.2 Where the time went (the host-polled timeline, `step8.py timeline`)
The host polls the first and last word each op writes (column 0, 64 B cache-line invalidate per probe, ~17 us per
poll round) while the dispatch runs: a per-op wall-clock timeline of the real step without trace hardware.
| build | change | dispatch ms (median) | per block |
|---|---|---|---|
| full12a | round 3 + attention fix + GELU fix | 279.5 | SSM 7.2, mlp0 6.5 (5.3 of it waiting), attn 2.2, norms 0.55-0.85 each |
| full12b | R4 placed after R3 (norm3s 2-input fill had a 10.5 MB stride shim BDs 2^20-word step: it ran column after column, 8 x 0.75 ms) | 217.2 | mlp0 1.38 |
| full12c | EW kernels -O2 (round 3 used -Oz; -O2 is SMALLER for ew8 and 2.4x faster) | 146.3 | SSM 2.85, norms 0.31-0.35 |
| full12d | + mm8 -O2 | 143.1 | |
| **full12e** | new SSM kernel (per-head broadcast tables, transcendentals as array loops, slot-wise output), vector reciprocal (bit-trick seed + 2 Newton; `aie::inv` is SCALAR: 32 extracts + inverses per vector), EW -Os (program memory) | **125.1** | SSM 1.88, attn 2.2, up_a+up_b 1.6, mlp0 1.1, norms 0.30-0.36, yx 0.55, qkv 0.43 |
| **full12f** | + host patch embedding (X0 written per frame; the device patch op and its 1.2 ms gone), 2-chain exp / sigmoid / sqrt / gelu helpers (nk8.h), EW -O2 again (fits without ew_patch) | **106.8** (min 106.8, 20 iters) | attn 1.75, SSM 1.45, norms 0.31-0.36, up_a + up_b 1.33, mlp0 0.85, yx 0.43, qkv 0.43 |
SSM differentials (t3 = SSM + one GEMM, 20 iters): -Oz 7.45 ms, -O2 3.06, new kernel 2.11 (no transcendentals
0.89, ring traffic only 0.51). Host side of a step (frame -> patches, query tokens, writes, read-back, heads): ~2.5 ms.
### 13.3 The current best step and the depth front (MEASURED; `step8.py time`, 20 iters after 5 warm-ups, clip30)
| build | depth | dispatch ms median / min | whole step incl. host |
|---|---|---|---|
| full12f | 12 | 106.83 / 106.77 | 109.93 |
| full10f | 10 | 89.56 / 89.23 | 92.38 |
| full8f | 8 | 71.56 / 71.51 | 74.03 |
| full6f | 6 | 53.83 / 53.79 | 56.62 |
| full10s (`--attn-keys 1152`: all 128 query slots are tokens) | 10 | 89.79 / 89.77 | 92.99 |
8.9 ms per block. full12f smoke: tracks within 0.023-0.034 px of CPU bf16 (4 points, 4 frames).
### 13.4 DAVIS 10 videos, 'first' (NPU = `step8.py davis`; CPU = davis_cpu.py, same loader/queries/metrics)
| run | AJ | OA | <delta> | ms/frame |
|---|---|---|---|---|
| NPU full12a (279 ms build) | 70.72 | 92.48 | 83.38 | 282 |
| **NPU full12f** | **70.81** | 92.65 | 83.35 | 110 |
| CPU bf16 depth 12 (round 2) | 70.98 | 92.76 | 83.42 | - |
| NPU full10s, support 4, 128 slots | 69.97 | 91.37 | 83.77 | 98.6 (dispatch 93.1: a CPU DAVIS job ran alongside) |
| CPU bf16 depth 10, support 4, 128 slots | 70.14 | 91.52 | 83.77 | 221 |
Support points need slots: 8 per query = up to 225 slots on DAVIS (CPU 71.33, round 2); this program has 128.

# Round 5 (2026-10-09)

## 14.0 Correction: the host patch embedding is not "0.2 ms" on one core (MEASURED, r5/patch_time.py)
Real-weights `lin_proj` (Conv2d 3->768, 8x8 stride 8, bf16 weights [768, 3, 8, 8]) on a 256x256x3 bf16 frame ->
[1, 768, 32, 32] bf16 = 1024 x 768 x 192 MAC = 302 MFLOP. 200 runs after 10 warm-ups, npu-iron-torch (torch 2.14 CPU),
testbox host load average ~6.6 from other tenants during the run.
| torch threads | conv median (min) | effective | as put_x0 runs it (+ permute, bf16 convert, + pos-emb) |
|---|---|---|---|
| 1 | 0.93 (0.92) ms | 325 GFLOP/s | 1.57 ms |
| 2 | 0.77 (0.49) ms | 394 GFLOP/s | 0.81 ms |
| 4 | 0.28 (0.27) ms | 1.1 TFLOP/s | 0.44 ms |
| 8 (step8.py default, THREADS env) | 0.28 (0.25) ms | 1.1 TFLOP/s | 0.39 ms |
| 16 (the container's torch default) | 0.22 (0.22) ms | 1.4 TFLOP/s | 0.29 ms |
| 32 | 24 (16) ms | SMT oversubscription | 36 ms |
Round 4's "0.2 ms" was the multi-threaded conv alone; the per-frame cost in step8 (8 threads) is ~0.39 ms. One Zen 5
core with AVX-512 BF16 peaks near 640 GFLOP/s, so 325 GFLOP/s single-threaded is plausible (oneDNN).

## 14. Round 5 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), step8.py, one dispatch per frame)
### 14.1 Speed, change by change (depth 12; block 5 of the host-polled timeline)
| build | change | dispatch ms | per block |
|---|---|---|---|
| full12f (round 4) | | 106.8 | attn 1.77 wall, SSM 1.45, norms 0.31-0.36 each |
| full12h | new norm kernel, ew_op dispatch, cores one block per worker iteration | 96.4 (time, 20 iters) | norms 0.10-0.13 each |
| full12i | attention: lazy max (TAU 8) + expd_i | 91.6 | attention 1.77 -> 1.35 (qkv end -> attn end) |
| full12j | `--lookahead 2` | 90.4 | gaps before up_b / down_a / down_b / mlp3_b 0.08-0.12 -> 0.05-0.08 |
| full12k | SSM: e2 = a^2 from the fp32 a | 89.5 | SSM 1.45 -> 1.38 |
| **full12m** | the same sources, built for depths 12/10/8/6/1 (one xclbin serves all) | **89.40** | 7.4 |
### 14.2 Differentials (timing only, wrong outputs; block 5 wall time)
| build | removed | result |
|---|---|---|
| full12h{ne,nr,nv,nk} | attention exp / rescale / PV (chunk_v) / whole K phase | attention 1.77 -> 1.31 / 1.35 / 1.24 / 0.47 ms |
| full12jng / full12jcp | SSM transcendentals / SSM math (ring copy only) | SSM 1.45 -> 0.75 / 0.38 ms |
| full12kng | the GELU chain in every epilogue (`-DNO_GELU`) | step 89.5 -> 80.3 ms; up_a 0.54 -> 0.43, mlp0 0.73 -> 0.46 |
### 14.3 Correctness
| test | result |
|---|---|
| t1 (random): RMS(2 in) panels / LN(2 in) panels | bit-exact / 0.0148% of mantissas, <= 2.56 ulp (round 3: 0.09% of panels) |
| t2 (random): LN(3 in) panels | 0.0125%, <= 2.37 ulp |
| t4 (random): attention O | rms_rel 7.16e-3 (round 4 7.2e-3) |
| t3 (random): SSM h / z panels | h rms_rel 4.05e-3, z 10.2% of mantissas (round 3/4: 4e-3, 10%) |
| `blocks --build full1h` (depth 1, real weights) | every region corr >= 0.9997, rms_rel 0.3-2.2% (round 4: 0.6-3%) |
| smoke full12k (4 frames vs CPU bf16) | tracks within 0.021-0.027 px |
| smoke depth 10 on the depth-12 xclbin (full12m + full10m.bin) | tracks within 0.025-0.042 px |
| **DAVIS 10 videos, full12m depth 12** | **AJ 71.00** (OA 92.70) vs CPU bf16 70.98, round-4 NPU 70.81; 92.6 ms/frame |
| **DAVIS 10 videos, full12m depth 10** (shared xclbin) | **AJ 69.87** (OA 91.11) vs CPU bf16 70.07; 77.1 ms/frame |
| **DAVIS 10 videos, full12m depth 8** (shared xclbin) | **AJ 68.28** (OA 91.05) vs CPU bf16 68.22; 62.5 ms/frame |
Timeouts this round: 0.
### 14.4 Depth at run time (`step8.py sweep --build full12m --depths 12,10,8,6,12`, one process, one xclbin)
| depth | dispatch median / min | whole step incl. host | fps |
|---|---|---|---|
| 12 | 89.40 / 89.38 | 91.86 | 10.9 |
| 10 | 74.64 / 74.62 | 76.95 | 13.0 |
| 8 | 59.73 / 59.72 | 62.04 | 16.1 |
| 6 | 44.87 / 44.86 | 47.23 | 21.2 |
| 12 (again) | 89.40 / 89.38 | 91.96 | |
The elfs and CDO of full{12,10,8,6,1}m are byte-identical (cmp); the xclbins differ only in their headers.
### 14.5 The Pareto front after round 5 (latency MEASURED; AJ = NPU DAVIS 10 videos where marked, else CPU bf16)
| depth | NPU ms per frame (dispatch) | fps | DAVIS AJ |
|---|---|---|---|
| 12 | 91.9 (89.4) | 10.9 | **71.00 NPU** (CPU 70.98) |
| 10 | 77.0 (74.6) | 13.0 | **69.87 NPU** (CPU 70.07) |
| 8 | 62.0 (59.7) | 16.1 | **68.28 NPU** (CPU 68.22) |
| 6 | 47.2 (44.9) | 21.2 | 64.77 (CPU) |
All from one xclbin (full12m) with Step8.set_depth. GPU compiled bf16: 22 ms at depth 12.

## 15. Round 6 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), step8.py, one dispatch per frame)
### 15.1 Speed, change by change (depth 12, `step8.py time`, median of 20)
| build | change | dispatch ms |
|---|---|---|
| full12m (round 5) | | 89.40 |
| full12r6a | pass-structured -O2 epilogue (bias pass 5.5 bundles per 64 elements, from ~77) + vector 1/sum in attn_end | 84.14 |
| full12r6b | GELU by lookup table (tanh table for yx / gated, erf table for mlp0) | 78.16 |
| full12r6c | attention chunk_k / chunk_v without minsize | 77.46 |
| full12r6d / **full12r6e** | attention exp as one array pass per row block (expd_n) | **76.34** |
Block 5 (timeline, share = last write to last write): r5 7.42 -> r6a 6.95 -> r6b ~6.46 ms; up_a 0.61 -> 0.51, up_b
0.58 -> 0.49, mlp0 0.73 -> 0.54, yx 0.38 -> 0.30 for the GELU step.
### 15.2 Depth sweep (one xclbin, `step8.py sweep --build full12r6e --depths 12,10,8,6,12`)
| depth | dispatch median / min | whole step | fps |
|---|---|---|---|
| 12 | 76.34 / 76.30 | 78.91 | 12.7 |
| 10 | 63.75 / 63.71 | 66.44 | 15.1 |
| 8 | 51.02 / 51.01 | 53.61 | 18.7 |
| 6 | 38.33 / 38.32 | 40.85 | 24.5 |
| 12 again | 76.32 / 76.29 | 78.88 | |
### 15.3 npu-research P1 twins (full step timelines, round-6 sources at r6a)
| build | dispatch |
|---|---|
| full12r6a | 84.0 |
| full12p1n (-DMM_NOCOMPUTE) | 67.9 |
| full12p1t (-DMM_TWICE) | 108.7 |
down_a (96 k-steps per core): 0.20 / 0.10 / 0.35 ms -> data movement ~1.0 us, compute ~1.56 us per k-step: compute-bound.
### 15.4 One-core primitive probe (`r6/kprobe.py`, every bf16 input; time = repetition-timed, copy subtracted)
| primitive | != RNE(exact) | ulp p99 (all finite) | ns per 32 lanes |
|---|---|---|---|
| nk8 gelu (round 5) | 1.07% | 0.51 | 63 |
| LUT gelu, aie::linear_approx | 0.61% | 0.47 | 43 |
| LUT gelu, hand lookup (deployed) | 0.61% (bit-equal to the above) | 0.47 | 16 |
| aie::tanh (vtanh) | 1.73% | 1.21 (19 at 0.5) | - |
| aie::exp2 | 53.3% | 10.3 | - |
| aie::inv | 0.00% | 0.49 | (scalar) |
| nk8 exp32 / sigmoid | 0.99% / 0.97% | 0.50 | - |
Per range (gelu): LUT is closer to RNE than nk8 on [-1, 4) (e.g. [0.01, 1): 1.7% vs 10.2% mismatches) and worse on
[-4, -1) (up to 7 ulp) and below -4 (s clamped: |gelu| ~2e-4 instead of ~0).
### 15.5 Real activations (clip30 3 frames, 12 blocks; `r6/numh.py`)
GELU candidates, rms rel / bfp ratio (ssm_gelu, ffw_gate x up, vit_gelu vs tanh): nk8 4.41e-4 / 0.059, 2.05e-3 /
0.264, 4.42e-4 / 0.093; LUT 64 x 1/8 4.40e-4 / 0.059, 2.05e-3 / 0.264, 4.77e-4 / 0.100; one Newton step 0.40-0.60;
quadratic 2^f = nk8. vit_gelu vs the erf reference: nk8 (tanh) p99 39 ulp, erf LUT p99 3.3 ulp, same rms (8.7e-4).
RG-LRU sigmoid: nk8 18.6% != RNE, |da|/(1-a) 1.4e-2; symmetric LUT [0, 8) 5.1% but |da|/(1-a) 1.6e3 (rejected).
### 15.6 Correctness
| test | result |
|---|---|
| t1 (r6b): g1/g2 plain, norms | bit-exact as round 5 (n2 LN 0.0146% = round 5's post-t1h norm) |
| t1 (r6b): g3 gelu vs tanh ref | 9.30% mismatches (round 5 15.38%), rms_rel 1.04e-3 (1.03e-3) |
| t2 (r6b): gated panels | 0.84 / 0.86% (round 5 1.16 / 1.19%) |
| t2 (r6b): mlp0 gelu_pan vs the ERF reference | 0.81%, rms_rel 1.61e-3 (round 5 vs tanh: 1.76%, 2.48e-3) |
| t4 (r6a..r6d): attention O | rms_rel 6.99e-3 (round 5 7.16e-3; the fp32 1/sum) |
| regress full12r6d / full12r6e vs full12r6b | BIT-IDENTICAL |
| smoke full12r6b (4 frames vs CPU bf16) | tracks within 0.015-0.037 px, tokens rms_rel 1.6-3.0e-2 (round 5 1.6-2.9e-2) |
| **DAVIS 10 videos depth 12** (full12r6b = r6e) | **AJ 70.83**, OA 92.6, 80.8 ms/frame (r5 71.00, CPU 70.98) |
| **DAVIS 10 videos depth 10** (full12r6e) | **AJ 69.94**, OA 91.12, 67.0 ms/frame (r5 69.87, CPU 70.07) |
| **DAVIS 10 videos depth 8** (full12r6e) | **AJ 68.25**, OA 90.98, 54.6 ms/frame (r5 68.28, CPU 68.22) |
Timeouts this round: 1 (core timer read in the probe; recovered; known-good full12r6a 83.97 ms right after).

## 16. Round 7 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0))
### 16.1 Clock (r7/clk.py + clk.cc: pure-register hardware loops, runtime trip count, host-timed slope)
| loop | bundles/trip | fit (median / min) |
|---|---|---|
| scalar LCG chain | 4 | 1.8086 / 1.8088 GHz |
| dependent bf16 vmac chain | 3 | 1.8076 / 1.8083 GHz |
Time ratio of the two loops at 32M trips 1.3329 (4/3 = 1.3333): no stalls. **The core clock is 1.81 GHz in Turbo**
(the driver table's top DPM level, 1.8 GHz), not 1.43. Dispatch intercept 50-77 us. Every cycle/us conversion uses 1.81.
### 16.2 Attention, t4 directed test (qkv + attention + out_proj; t4q = qkv + a plain GEMM = 582 us)
| build | change | t4 us | attn O rms_rel |
|---|---|---|---|
| t4-r7a (= r6e kernels) | | 1706 | 6.987e-3 |
| t4-r7b | P = exp by table (h = 1/4 on [-10, 9), zero below), fused with the bfp push from fp32 | 1625 | 6.938e-3 |
| t4-r7c | attn_begin / attn_end at -O2, ones panels as 6 bfp pushes, O zeroed per used tile | 1570 | 6.938e-3 |
| t4-r7e | QK^T: Q popped per key block on the other load slot (no 8-register hold), constant trip count | 1526 | 6.938e-3 |
| t4-r7f | scores unscaled (1/8 folded into table / TAU / rescale; bit-identical) | 1506 | 6.938e-3 |

Splits (t4, r7b sources, twin method = the phase run twice per call): K phase +599 us (exp + push +298, so QK^T +
max ~300), V phase +192 us; all stubbed 759 us. Round-6 differentials: the exp alone 406 us of ~1.12 ms.
Per head group (b5, host-polled O drains of full12r7a, +-21 us): pass 0 ~155-170 us (compute 132), pass 1 ~80 (66).
Not kept: carrying the looked-up lines across trips (65 bundles, spills); chunk_v with P popped (67 vs 42 per pair).
### 16.3 Step (depth 12, step8.py time, median of 20)
| build | dispatch ms | regress |
|---|---|---|
| full12r6e (today) | 76.74 | |
| full12r7a (LUT exp) | 74.47 | differs at noise level; smoke tracks within 0.035 px of CPU bf16 |
| full12r7c | 74.19 | BIT-IDENTICAL vs r7a |
| full12r7e | 73.35 | BIT-IDENTICAL vs r7a |
| full12r7f | 73.10 | BIT-IDENTICAL vs r7a |
| full12r7g (asm GEMM call, -DMM_ASM; B pair4 order) | 71.19 | BIT-IDENTICAL vs r7a |

### 16.4 The asm GEMM call (r7/gen_mm.py -> mm_core.s)
225 bundles per call (128 MACs) vs 321 for the C++ loop; asmcheck 0 violations; t1 / t2 / t4 / g11 directed tests
give round 6's values exactly; full12r7g regress BIT-IDENTICAL vs full12r7a. Step 73.10 -> 71.19 ms (-1.9 ms), far
less than the bundle count implies (EXTRAPOLATED ~-7 ms): not yet measured in situ (-DPROF with -DMM_ASM overflows
program memory by 80 B). First full-step run was wrong (pack8's B order also permuted the SSM parameter tiles).
### 16.5 Depth sweep (one xclbin, `step8.py sweep --build full12r7g --depths 12,10,8,6,12`)
| depth | dispatch median / min | whole step | fps |
|---|---|---|---|
| 12 | 71.76 / 71.72 | 75.00 | 13.3 |
| 10 | 60.34 / 60.22 | 64.04 | 15.6 |
| 8 | 48.38 / 48.09 | 52.18 | 19.2 |
| 6 | 36.38 / 36.08 | 39.65 | 25.2 |
| 12 again | 72.07 / 71.87 | 75.02 | |
### 16.6 Accuracy
| test | result |
|---|---|
| smoke full12r7a (= r7g bit-level) vs CPU bf16, 4 frames | tracks within 0.024-0.035 px, tokens rms_rel 1.6-2.7e-2 |
| **DAVIS 10 videos depth 12** (full12r7g) | **AJ 70.78**, OA 92.58, 75.4 ms/frame (r6 70.83, CPU 70.98) |
| **DAVIS 10 videos depth 10** (full12r7g) | **AJ 69.83**, OA 91.13, 63.5 ms/frame (r6 69.94, CPU 70.07) |
Timeouts this round: 0.

## 17. Round 8 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz)
### 17.1 Why the asm GEMM call gave -1.9 ms, not ~-7 (g11 -DPROF, r8/gprof.py; r8/mmb.py)
In-situ stamps, g11 (11 GEMM ops x 12, no EW / attention: program memory for -DPROF + -DMM_ASM). The PROF window
adds ~40 cycles per call (stamps + the round-8 gap bookkeeping).
| build | cycles per call in situ | static | g11 ms (PROF) |
|---|---|---|---|
| g11-r8pc (C++ mm_acc) | 361 | 321 | 42.70 |
| g11-r8pa (asm mm_core, r7 placement) | 337 | 225 | 41.23 |
| g11-r8pb (asm, A ring pinned to banks 1,2,2) | 323.5 | 225 | 40.12 |
Gaps between calls: 43 (C++) / 47 (asm) cycles inside a k-step (call + A-ring lock code), 62-66 at k-step starts
(B lock); yx 84-88, qkv 57-61 (P23's row imbalance).
**Explanation: L1 bank conflicts inside the asm call** (the C++ loop has none). mmb (one core, no DMA, operands at
chosen offsets, mm_core called n times between timer reads; banks = 16 KB, 0x70000 + 16 KB i):
| placement | cycles per call |
|---|---|
| A, B, C-load, C-store in 4 different banks (split variant mm_core_split) | 238 (= 225 + ~13 loop/call) |
| A with C-load (same slot lda, never co-issued) | 238 |
| A and B in one bank | 252 (+14) |
| C loads and stores in one bank (in place, as deployed) | 271 (+33) |
| A + B one bank and C in place | 285 |
| B with C | 280-322 (+42..) |
In situ (r7 placement: A slots 1, 2 in B's bank 3) the call runs ~283-297 instead of 225. Remaining after the A pin:
~283 = 271 (C in place) + ~12 (DMA writes). P21's R1 (no acc vlda + acc vst in one bundle) in gen_mm.py
(`--noconf`): 273 static bundles (a 4-chain group boundary needs 32 same-bank accesses in ~16 cycles); R2-like A/B
exclusion (`--noconf-ab`): 255. Both worse than the conflicts they remove: needs a layout fix (C pieces of
consecutive groups in different banks) or a restructured loop (P20).
### 17.2 Tile boundary (P24 stamps, g11-r8pd)
Per tile: epilogue 0.64 us (plain) / 2.4 (qkv) / 3.3 (yx gelu) / 7.0-7.8 (gated, gelu_pan); C handoff (epilogue end
-> next acc_zero) **1.31 us** median (0.03 within a gated pair); acc_zero **0.49 us**; last call -> epilogue 69
cycles. The op's first tile waits 27-50 us for the A slot load in g11 (no EW dependency): P5's case.
### 17.3 SSM phase profile (EW_DEFS=-DPROF, r8/ewprof.py; cycles per chunk = 8 rows x 64 channels; 216 chunks per op)
| phase | full12r8ep (r7 kernel) | t3-r8bp (round 8 kernel) |
|---|---|---|
| input wait | 114 | 115 |
| conv | 792 | 1058 (incl. 7 in_part views) |
| gate GEMMs (2 x 64 mmuls + bfp panels) | 2383 | 652 |
| sigmoid (2 x 16 vectors) | 2542 | 2541 |
| log_a + exp (a, 1 - a^2) | 1507 | 1506 |
| sqrt | 1208 | 1093 |
| recurrence | 1032 | 889 |
| z panels | 233 | 244 |
| output wait | 31 | 31 |
| output copies + release | 1085 | 915 |
| params (per head, /18) | 301 | 303 |
| **op total** | **1340 us** | **1115 us** (PROF; t3 1554 -> 1263 us unprofiled) |
Compute-bound: waits are 1.3% of the op. The transcendental chains (sigmoid, exp, sqrt) are now 52%.
Real pre-activation ranges (build/acts_clip3.npz, block 0): gate_x input [-87, 79] (1-99%: -13.6..4), gate_a
[-105, 93] (1-99%: -10.6..13.8): a sigmoid table must keep the x < -8 tail (gate_a feeds a = exp(-8 sp sigma)).
### 17.4 Attention per head group (full12r8cp -DPROF, 192 records, median us)
| pass | Q copies + waits | begin | K phase | V phase | waits between calls | end | total |
|---|---|---|---|---|---|---|---|
| 0 (2 tiles) | 5.5 | 0.96 | 90.6 | 33.8 | 1.8 | 2.1 | 135 |
| 1 (1 tile) | 3.8 | 0.71 | 45.8 | 17.4 | 1.8 | 1.1 | 71 |
Fixed costs ~17 us per head-group pair = ~70 us per block (8%): attention is its K phase (QK^T + max + exp/push, 66%).
### 17.5 Step (depth 12, step8.py time, median of 20)
| build | change | dispatch ms | regress |
|---|---|---|---|
| full12r7g (re-timed today) | | 72.15 | |
| full12r8a | A ring pinned to L1 banks 1, 2, 2 (A_BANKS) | 70.62 | BIT-IDENTICAL vs full12r7a |
| full12r8b | + SSM: gate GEMM specialised, contiguous part views, fast ring copies, sqrt unrolled | 68.26 | BIT-IDENTICAL |
| full12r8c | + one table-walked op loop (attention = a table entry), attention trip counts from L1 | 67.89 | BIT-IDENTICAL |
GEMM core text 16,176 -> 14,848 B (full12r8c); EW core 15,264 -> 15,792 B.
### 17.6 Depth sweep (one xclbin, `step8.py sweep --build full12r8c --depths 12,10,8,6,12`)
| depth | dispatch median / min | whole step | fps |
|---|---|---|---|
| 12 | 68.40 / 68.00 (67.90 at the end) | 72.23 (70.26) | 14.2 |
| 10 | 57.08 / 56.67 | 60.56 | 16.5 |
| 8 | 45.40 / 45.32 | 48.02 | 20.8 |
| 6 | 34.11 / 34.09 | 36.55 | 27.4 |
### 17.7 Accuracy
full12r8c is BIT-IDENTICAL to full12r7a (regress, 4 frames: tracks within 0.035 px of CPU bf16 as r7a).
DAVIS 10 videos (full12r8c): depth 12 **AJ 70.78**, OA 92.58, 71.1 ms/frame; depth 10 **AJ 69.83**, OA 91.13, 60.0 ms/frame (= r7g, bit-identical numerics). Timeouts this round: 0.

## 18. Round 9 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz)
### 18.1 Step (depth 12, step8.py time, median of 20) — every build bit-identical to the one before unless marked
| build | change | dispatch ms | regress |
|---|---|---|---|
| full12r8c (re-timed) | round 8 best | 67.89 | |
| full12r9a | SSM sigmoid + sqrt by table (EW_DEFS=-DSSM_LUT) | 65.07 | NEW NUMERICS (DAVIS below) |
| full12r9e | + sigmoid table copies in two banks (SSM arena compacted) | 64.73 | BIT-IDENTICAL vs r9a |
| full12r9f | + exp_a2 as 5 pipelined passes (-DSSM_EXP3) | 64.51 | BIT-IDENTICAL |
| full12r9k | + one call per k-step, A-ring locks in the kernel (kdefs -DMM_KSTEP) | 62.49 | BIT-IDENTICAL |
| full12r9q | + accumulator pieces, greedy bank map (-DMM_PIECES, round-9 intermediate) | 61.61 | BIT-IDENTICAL |
| full12r9r | pieces with the affine layout (strided epilogues; kept) | 62.17 | BIT-IDENTICAL |
| **full12r9s** | + the k-step in asm, 9 stripes software-pipelined (-DMM_KS, r9/gen_ks.py) | **58.14** | BIT-IDENTICAL |
Build: `A_BANKS=1,2,2 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3" AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r9s --lookahead 2
--kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS` (and full{10,8,6}r9s). GEMM core text 15,840 / 16,384 B.

### 18.2 Depth sweep (one xclbin, `step8.py sweep --build full12r9s --depths 12,10,8,6,12`)
| depth | dispatch median / min | whole step | fps | DAVIS AJ (OA) | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 58.10 / 57.98 (58.08 at the end) | 60.3-60.6 | 16.5 | 70.84 (92.57) | 70.98 |
| 10 | 48.53 / 48.49 | 50.98 | 19.6 | 69.90 (91.02) | 70.07 |
| 8 | 38.85 / 38.80 | 41.27 | 24.2 | (not run; r6 numerics 68.25) | 68.22 |
| 6 | 29.21 / 29.18 | 31.66 | 31.6 | (not run) | 64.77 |

### 18.3 SSM transcendentals
Primitive probe (r9/lprobe.py: every bf16 input, one core, cycles per 32 lanes from the core timer, device vs bit-level
emulation):
| primitive | cycles / 32 lanes | != emulation | != RNE(exact) |
|---|---|---|---|
| sigmoid nk8 (deployed) | 75.0 | 0.08% | 21.4% (all bf16) |
| sigmoid table [-12, 8) / 8 (r9/ssm9.h) | 27.4 (35.4 with both copies in one bank) | 0 | 21.3% (all bf16; tails clamped) |
| a = exp(la) nk8 / table 2^f (64 seg) | 60.7 / 61.5 | 0.19% / 0.19% (1-ulp, both) | 0.53% / 0.23% |
| sqrt nk8 / exact table (256 mantissas) | 65.7 / 28.3 | 0 / 0.09% (-0 only) | 33.7% / 0.09% |
Real gates (r9/ssmemu.py, 5 blocks x 288 rows x 768, vs torch bf16): deployed nk8: gate_x 18.7% / gate_a 16.9% / a 13.1% /
mult 40.3% != torch, fixed-point drift p99 3.3%; tables (sigmoid /8 + nk8 exp + exact sqrt): 5.3% / 4.9% / 8.5% / 6.6%,
drift p99 2.9%. Direct exp tables on log_a: 29% / 17% a mismatches (rejected); range-reduced 2^f table: as accurate,
not faster (rejected for speed).
CPU model with emulated RG-LRU numerics (r9/cpuemu.py, 48 grid points x 30 frames of clip30, |track - torch bf16|):
| variant | mean px | p95 | max | vis flips |
|---|---|---|---|---|
| exact (fp64, RNE) | 0 | 0 | 0 | 0 |
| deployed nk8 | 0.0148 | 0.0385 | 0.101 | 0 |
| tables (r9a) | 0.0146 | 0.0392 | 0.126 | 0.07% |
| random 1-ulp flips of 5% / 18% of the gates | 0.0141 / 0.0146 | 0.038 / 0.037 | 0.195 / 0.150 | 0.07% / 0 |
Any ulp-level change of the SSM numerics is the same chaotic noise (~0.1-0.2 px max over 30 frames).
NPU smoke vs CPU (clip30, 4 demo points): 4 frames max 0.038 px (r8c 0.035); 30 frames max 0.131 (r8c 0.070; inside
the noise envelope above).
SSM op (t3, us, 40 iters): r8b 1260 -> r9a 1022 -> r9e 987 -> r9f 946. In-situ phases (t3-r9ap, PROF, cycles per chunk):
sigmoid 2541 -> 1168, sqrt 1093 -> 493, log_a + exp 1507 (then -> passes), conv 1058, recurrence 888, out copies 798,
gate GEMMs 650. Step effect of r9f's exp passes: -0.22 ms (t3: -40 us per op).

### 18.4 GEMM k-step (P20)
- mm_kstep (C++ loop, locks inside): g11 35.72 -> 33.93 ms.
- Pieces: in-situ cycles per stripe (g11 PROF) down_a 346 -> 323 (greedy map) / 333 (affine), qkv 349 -> 317 / 342;
  plain epilogue 0.64 -> 1.45 us (table-indirect pass) -> 0.72 us (affine strided pass).
- mm_ks (pipelined, r9/gen_ks.py): period 192 bundles per stripe (225 per call before), prologue 58, epilogue 15; one-core
  bench (r9/ksb.py, no DMA): 2655 -> 2200 cycles per k-step, 36/36 pieces bit-identical. g11 32.86 -> 29.47 ms.
- In situ (g11-r9sp PROF): cycles per k-step down_a 2588, qkv 2663, out / out_proj 3297-3320, yx 3033. Static 1,830
  + modelled bank conflicts 24 (gen_ks.py --banks). **The k-step is now A-port-bound: 10,368 B of A per k-step at
  ~4 B/cycle = 2,592 cycles** (down_a measured 2,588). K = 768 ops with nt 4-8 (out, out_proj, yx) are slower still.
- Timeouts this round: 2 (both mm_ks in g11 before the fixes below; the known-good g11 ran right after each).
  1: loop start not 16 B aligned, r25 uninitialised (no `mov r25` before the first B fill); 2: `acq` with value 0 =
  acquire_EQUAL 0 (waits for an empty ring). The bench then ran it bit-exact; g11 and the step after.

## 19. Round 10 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz)
### 19.1 Step (depth 12, step8.py time, median of 20) — every build BIT-IDENTICAL to full12r9a (regress, 4 frames)
| build | change | dispatch ms |
|---|---|---|
| full12r9s (re-timed) | round 9 best | 58.11 |
| full12r10j | A_HALF=1: K=768 / attention A loaded once (multicast into both slot halves), up_b reuses up_a's slot (P5a + P5r) | 56.76 |
| full12r10k | + P23=1: yx / qkv N tiles round-robin over the 3 rows | 55.91 |
| **full12r10r** | + A_RT=1: memtile A slot driven by runtime tasks (chunked loads, chunk-gated first tile, next op's load under the current op) | **53.89** (sweep 53.78-53.83) |
Build: `A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3" AIECC_EXTRA=--reclaim-runtime-bds ./bb.sh full12r10r
--lookahead 2 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS` (and full{10,8,6}r10r). GEMM text 15,840 B, EW 16,240 B (unchanged).
g11 (GEMM ops only, 132 ops): r10b (= r9s flags) 29.72 -> r10j 27.85 -> r10k 27.58 -> r10r 25.21 ms. t2: 2.09 -> 1.95 -> 1.90; t4: 1.34 -> 1.30 -> 1.25.

### 19.2 Depth sweep and DAVIS (one xclbin full12r10r)
| depth | dispatch median | whole step | fps | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 53.83 | 56.3-56.9 | 17.6 | 70.84 (92.57), 57.6 ms/frame in the DAVIS loop | 70.98 |
| 10 | 45.35 | 49.18 | 20.3 | 69.90 (91.02), 47.5 ms/frame | 70.07 |
| 8 | 35.98 | 38.49 | 26.0 | not run (bit-identical numerics to r9a) | 68.22 |
| 6 | 27.07 | 29.57 | 33.8 | not run | 64.77 |
Host part of the step (whole - dispatch) was 2.5-3.8 ms this round vs 2.2-2.5 in round 9 (CPU load from other work; not studied).

### 19.3 Findings
- Round 9's "A-port-bound k-step" was a mean: per-tile medians (r10/gdist.py on g11sp.npy) 2,050-2,065 cycles for every op;
  the first tile of each op held the op-start A-slot wait. With A_RT the g11 PROF means are ~2,058 per k-step (cyc/call 229)
  except the first ops after the dispatch start.
- Memtile -> core stream 7.95 B/cycle (r10/rtm.py; 64-bit). Runtime memtile tasks: hashes correct first time; op-start
  waits 0.04 us when the next op's A has no dependency on the current one.
- IRON router bug: multicast + unicast packet flows to the same two memtile DMA channels drop the multicast's second
  destination (r10/pf1.mlir vs pf2.mlir). It cost the round's only timeout (t2-r10h); known-good t2-r9k ran right after.
- GELU: deployed loop 34.2 cycles / 32 lanes (n = 144), 44.1 at n = 8 (one-core bench r10/gxb.py); forced-II pipelined
  loop 17 bundles static but 54.7 cycles measured (in situ yx epilogue 3.39 -> 5.02 us): rejected.
- Epilogue cost on the GEMM critical path (g11 PROF, us per tile): gelu_pan 8.19, gated 7.03, yx gelu ~6.1, qkv 2.70,
  plain 0.71. Timeline r10r per block (ms, last to last): yx 0.15, ssm + out 0.86, up_a 0.31, up_b 0.31, down_a / b 0.10,
  qkv 0.21, attn 0.75 (+0.15 gap), out_proj 0.05, mlp0 0.36, mlp3_a / b 0.10, norms 0.11-0.13 each. Wall share: attention
  10.9 ms, SSM + out 10.3, norms 6.6, mlp0 4.9, up_a + up_b 8.6.
- Timeouts this round: 1.

## 20. Round 11 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz)
### 20.1 Step (depth 12, step8.py time, median of 30) — every build BIT-IDENTICAL to full12r9a (regress, 4 frames)
| build | change | dispatch ms | whole step ms |
|---|---|---|---|
| full12r10r (re-timed) | round 10 best | 53.84 | 56.18 |
| full12r11a | + GELU_II really pipelined (separate restrict parameters) | 52.96 | 55.06 |
| full12r11b | GELU_MS instead (asm modulo schedule, II 13, one call per run) | 52.43 | 54.90 |
| full12r11c | + P43 (B fills before the dependency wait) | 52.08 | 54.58 |
| **full12r11d** | = r11c + -fstack-size-section (text identical) | **52.07** (sweep) | 54.06 |
Build: `P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3" AIECC_EXTRA=--reclaim-runtime-bds
./bb.sh full12r11d --lookahead 2 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS` (and full{10,8,6}r11d). GEMM text
15,968 B, EW 16,240 B. Timeline r10r -> r11b (ms per step): mlp0 4.93 -> 4.12, up_a 4.39 -> 4.05, up_b 4.22 -> 3.84, yx 2.20
-> 2.04, qkv 3.35 -> 3.00, norms 6.43 -> 6.62 (r11/tlcmp.py).

### 20.2 Depth sweep and DAVIS (one xclbin full12r11d, HOST_FAST=2)
| depth | dispatch median | whole step | fps | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 52.07 | 54.06 | 18.5 | 70.84 (92.57), 55.5 ms/frame in the DAVIS loop | 70.98 |
| 10 | 43.52 | 45.72 | 21.9 | 69.90 (91.02), 46.0 ms/frame | 70.07 |
| 8 | 34.84 | 37.17 | 26.9 | not run (bit-identical numerics to r9a: 68.28 on r5's numerics) | 68.22 |
| 6 | 26.23 | 28.45 | 35.1 | not run | 64.77 |

### 20.3 One-core benches (r11/apb.py, cycles per 32 lanes incl. ~0.8 call overhead, n = 144 unless said)
| loop | in place | out of place | n = 8 calls |
|---|---|---|---|
| deployed gelu_lut_n (2 vectors per trip) | 34.2 (round 10) | 32.5 | 37.4-44.1 |
| Peano forced II 16 (r10/gx.cc gelu_ii16) | 17.8 | 17.8 | 27.0 |
| gelu_ms (II 13, U 1, 5 stages) | 14.6 | 15.6 | 30.4 (pieces, one call: 15.3) |
Hazard probes (r11/mk_haz.py; rd / rdl / waw / mid modes): see NOTES 16.2. In situ (g11-r11gp PROF): yx epilogue 3.39 -> 1.81 us
per tile.

## 21. Round 12 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz)
### 21.1 Step (depth 12, step8.py time, median of 60 after 20 warm-up, alternated A/B) — every build BIT-IDENTICAL to full12r9a
| build | change | dispatch ms | whole step ms |
|---|---|---|---|
| full12r11d (re-timed) | round 11 best | 52.10 | 54.0 |
| full12r12a | + GUP (gated up + product + push, asm II 6 / 64 lanes) | 51.56 | 53.6 |
| full12r12b | + CPY32 (gated_b stash copy, asm II 1 / 32 B) | 51.21 | 53.1 |
| full12r12c | + PAN (panel pushes of mlp0 and the Q / K objects, asm II 16 / row block) | 50.93 | 52.9 |
| full12r12d | + GELU_FLAT + BGELU (bias + GELU fused, II 12, flat-edge tables; gelu_ms gone) | 50.61 | 52.6 |
| **full12r12e** | + EW SIGK (SSM sigmoid, kernel-only asm, II 12) | **50.04** (sweep 50.05) | 52.0-52.3 |
Build: `P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3 -DSIGK" AIECC_EXTRA=--reclaim-runtime-bds
./bb.sh full12r12e --lookahead 2 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU`
(and full{10,8,6}r12e). Text: GEMM 16,288 / 16,384 B, EW 16,224 / 16,384 B. Stack audit: GEMM 1,536 of 2,560 B (+ bgelu's 512 B ring).
Timelines (ms per step): r11d -> r12c: up_a 4.02 -> 3.63, up_b 3.83 -> 3.35, mlp0 4.19 -> 3.99, qkv 3.00 -> 2.84; r12d -> r12e:
SSM ("out") 10.37 -> 9.85, norms 6.57 -> 6.39. P43 check: mlp3_a 1.99 (r11b) -> 1.91 ms per step (166 -> 159 us per block).

### 21.2 Depth sweep and DAVIS (one xclbin full12r12e, HOST_FAST=2, median of 60; load average ~2-3 from other tenants)
| depth | dispatch median | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 50.05 | 52.14 | 19.2 | 70.84 (92.57), 52.5 ms/frame in the DAVIS loop | 70.98 |
| 10 | 42.02 | 45.24 | 22.1 | 69.90 (91.02), 44.0 ms/frame in the DAVIS loop | 70.07 |
| 8 | 33.51 | 35.94 | 27.8 | not run (bit-identical numerics: 68.28 on r5's) | 68.22 |
| 6 | 25.20 | 27.59 | 36.2 | not run | 64.77 |

### 21.3 One-core benches (r12/*b.py, core timer, bit-exact against the deployed C++ unless said)
| kernel | asm | C++ it replaces (same bench) | notes |
|---|---|---|---|
| gup, 24 blocks of 64 | 239 cycles (10.0 per block incl. call) | 1,223 | strides 320 / 576, 3 seeds |
| pan, 6 row blocks | 161 | 346 | contiguous and grouped (1,152 B jumps) |
| cpy32, 162 x 32 B | 210 (U 3) | (in situ ~6,100 per gated_b tile) | U 2 (2-bundle body) ran once |
| bgelu, 32 vectors | 591 (II 12; 613 at II 13) | 1,644 (bias_pcs + gelu_lut_n) | 5 input ranges incl. out-of-table, tanh and erf |
| sigk, 32 vectors | 499 (kernel-only, II 12) | 898 (ssm9::sigmoid_lut_n) | guards: nothing before touched, 4 garbage vectors after |
| gelu_ms flat (II 12, IMS) | 14.46 per 32 lanes in place | 14.55 (II 13, deployed r11) | every finite bf16 (--sweep), both tables |
In situ (g11 PROF, gated_b tiles): epilogue per tile 5.11 (r11gp) -> 3.68 (GUP) -> 2.10 us (+CPY32). Differentials (r12a):
stash copy 1.70, gate GELU 1.20, gate bias 0.36, gup 0.36 us per tile; PAN_NOPUSH: mlp0 4.30 -> 3.07, qkv 2.70 -> 1.88 us per tile.
Output-stream probes (r12/gen_pushprobe.py): all eight push / flush patterns and four 2D-flush patterns byte-exact vs encode_hw.

## 22. Round 13 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz, host load ~1-3)
### 22.1 Step (depth 12, step8.py time, median of 60 after 20 warm-up) - every build BIT-IDENTICAL to full12r9a
| build | change | dispatch ms | whole step ms |
|---|---|---|---|
| full12r12e (re-timed) | round 12 best | 50.05 | 52.1 |
| full12r13a | + ASM_FIT=1 (asmfit on gup / pan / bgelu / cpy32: GEMM text -208 B) | 50.06 / 50.05 | 52.0-52.2 |
| full12r13b | + ATTKF (fused K loop, exact test on every unit) + T1 (constexpr k-step records: -0.7 KB) | 49.91 | 51.9 |
| full12r13c | + the C object re-laid (hot K-loop buffers in bank 2) | 49.84-49.89 | 51.7-52.0 |
| full12r13e | + --l2-depth 8 (memtile K / V forward 4 -> 8 objects) | 49.54 | 51.9 |
| full12r13e12 | --l2-depth 12 instead | 49.58 | 51.4 |
| **full12r13f** | + the in-loop flag (today's test only on flagged row blocks) | **49.21** (min 49.11) | 51.0 |
Build: `P43=1 A_RT=1 P23=1 A_HALF=1 A_BANKS=1,2,2 ASM_FIT=1 EW_OPT=-O2 EW_DEFS="-DSSM_LUT -DSSM_EXP3 -DSIGK" AIECC_EXTRA=--reclaim-runtime-bds
./bb.sh full12r13f --lookahead 2 --l2-depth 8 --kdefs=-O2,-DMM_ASM,-DMM_KSTEP,-DMM_PIECES,-DMM_KS,-DGELU_MS,-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU,-DATTKF`
(and full{10,8,6}r13f). Text: GEMM 16,384 / 16,384 B (0 free), EW 16,224 / 16,384. Timeouts this round: 2 (a trip count read
from the wrong register: 0 -> endless loop; both recovered, the known-good bench passed after each).
Timelines (ms per step): attention 10.95 (r12e) -> 10.48 (r13b) -> 10.13 (r13f); norms 6.39 -> 6.76 (+0.37, not understood:
other tenants, or the K / V multicast's deeper buffer shifting DDR traffic); everything else within +-0.04.

### 22.2 The K phase in situ: timing twins of full12r13c (`-DATTK_FORCE=bits`, outputs wrong, median of 40)
| variant | dispatch ms | reads as |
|---|---|---|
| ATTK_FORCE=3: no K-phase compute at all | 43.39 | the attention is compute-bound (not K / V DDR-bound) |
| ATTK_FORCE=2: the fast loop only, no test, no fallback | 47.14 | the fused loop's whole-step value: **-2.9 ms** vs r13a |
| ATTK_FORCE=16: the loop + today's test on every unit, no slow exp | 48.49 | the exact test + rows8 prep + rescales: +1.35 |
| production r13c (test on every unit, slow exp when it moves / invalid blocks) | 49.84 | the slow exp on ~10 % of units: +1.35 |
| ATTK_FORCE=8: the slow path on every unit | 52.87 | aexp_push on every unit: +4.4 ms (382 cycles per unit) |
In-situ slow-unit count (full12r13cc, `-DATTK_COUNT`, 3 frames, 192 records over the attention output): 8.6-10.9 % of the
units per core take the slow path (chunk 0 + chunk 22 are 8.7 %; the moved maxima 0-2 %). The CPU model on the NPU's own 1152
rows (r13/lazycount3.py) says 8.8-13.6 % per core, the captured-score replay (lazycount.py) 8.4 %.
So a 10 % slow fraction costs ~1.35 ms where its own cycles are ~0.45 ms: the K / V multicast makes the 8 columns advance in
lockstep per chunk (the slowest column's units set the wall). --l2-depth 8 recovered 0.36 ms of it.

### 22.3 One-core bench (r13/attkb.py, core timer, bit-exact vs mm8.cc chunk_k over 144 cases: 4 seeds x 6 scales x nkb 6 / 4 x
### the stored max at NEG / 64 / 128)
| path (cycles per row block = 6 key blocks) | C++ chunk_k | round-13 |
|---|---|---|
| warm, the max does not move (90 % of units in situ) | 636 | **496** (fused loop ~280 static + preamble / drain + rows8(-m) prep) |
| warm, the max moves (flagged: fast pass + today's test + rescale + exp) | 928 | 1,002 |
| cold (chunk 0: m = NEG; rescale of zeros + exp) | 1,220 | 1,508 |
| invalid key blocks (nkb 4: the slow exp always) | 662 | 940 |
| the split loops (superseded): asm QK^T (II 11) + C++ glue + C++ exp | 636 | 609 (QK 168 -> 141; glue 80; exp 347) |
Timing twins in the bench (r13c driver): the fused loop 390 cycles in situ-like layout (280 static: II 35 x (6 + 2 stages));
rows8 prep + call ~68; the exact test ~66; aexp_push 358-382.

### 22.4 Host (H3, step8 `HOST_HEADS=np`, 40 steps, 8 torch threads, OPENBLAS_NUM_THREADS=1)
heads: torch median 0.943 ms (p90 1.163) -> numpy 0.718 (p90 0.753); max |d| tracks 7.6e-5 px, logits 7.3e-5, vis 1.9e-6.
With OpenBLAS's default threads beside torch's pool: 11.7 ms (thread thrash). Not in the step's default yet.

### 22.5 Depth sweep and DAVIS (one xclbin full12r13f, step8 sweep, median of 60; DAVIS 10 videos, vendored model, 48 slots)
| depth | dispatch median | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 49.22 | 51.05 | 19.6 | 70.84 (92.57), 52.2 ms/frame in the DAVIS loop | 70.98 |
| 10 | 41.25 | 43.27 | 23.1 | 69.90 (91.02), 43.8 ms/frame in the DAVIS loop | 70.07 |
| 8 | 33.44 | 36.66 | 27.3 | not run (bit-identical numerics: 68.28 on r5's) | 68.22 |
| 6 | 25.16 | 27.21 | 36.8 | not run | 64.77 |
Numerics are bit-identical to rounds 9-12 (regress BIT-IDENTICAL at every step; the DAVIS numbers reproduce exactly).

## 23. Round 14 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz, step8 time / sweep median of 60)
### 23.1 Step at depth 12 (every build BIT-IDENTICAL to full12r9a)
| build | change | dispatch ms | predicted |
|---|---|---|---|
| full12r13f (re-timed this round) | round 13 best | 49.23-49.33 | |
| full12r14a | + ATTKE (chunk 0 no rescale, last chunk 2 trips + zero panels) | 48.70-48.71 | -0.58 +- 0.2 (hit: -0.63) |
| full12r14c | + ATTKB (batched K loop) + KFOLD + text work | 48.24-48.73 (bimodal) | -1.5 +- 0.4 (MISS: -0.04 .. -0.47) |
| full12r14cz | + ZASM | 48.10-48.18 (min 48.01) | -0.12 (hit) |
| full12r14e / **r14f** | + ATTV (V phase + row sums in asm) + VTK (V^T epilogue in asm) | **47.25-47.34** | -0.92 (hit: -0.9) |
Timing twin full12r14cF128 (batched chunks' flags ignored, outputs wrong): 46.15 ms. full12r14cF2: 50.33 (that twin also skipped
chunk 0's exact path: every later row block flagged; not usable).
Timeline r13f -> r14f (ms per step): attention 10.13 -> 8.52, qkv 2.87 -> 2.76, the rest within +-0.1 (norms +0.1).
### 23.2 Depth sweep (one xclbin full12r14f; full{10,8,6}r14f sequences)
| depth | dispatch median | whole step | fps (whole step) |
|---|---|---|---|
| 12 | 47.31 | 49.53 | 20.2 |
| 10 | 39.84 | 42.07 | 23.8 |
| 8 | 31.87 | 34.15 | 29.3 |
| 6 | 24.43 | 27.64 | 36.2 |
### 23.3 One-core benches (r13/attkb.py: in-situ banks since this round: K in .aie.bank3, Q at q_buf(1), the C object at 0x78900)
| K phase per row block (cycles) | r13 fused (r14a driver) | r14 batched |
|---|---|---|
| warm, unflagged | 444 | **261** |
| chunk 0 (cold) | 933 | 971-1,038 |
| last chunk (nkb 2), unflagged | 321 | 358 |
| warm, flagged (max moves) | 870 | 1,129 |
k_rb phases on one flagged row block (--krb, KRB_STAMP): first pass 393, exact test 164, rescale_rb + -m rows 637 (chunk 0: 59),
exact rerun 393. V phase per (tile, chunk) (--vphase): C++ 1,316 -> attv + C++ sums 899 -> attv + attvs 813 (O and sums hashes
equal on 3 seeds). acc_zero (--zero): zero2 627 cycles. Static: attkb 36 x 34 + 68 = 1,292; attv 24 x 21 + 21 = 525; attvs 126;
vtk 24 x 13 + 13 = 325 per call (6 calls per V^T tile).
### 23.4 DAVIS (10 videos, vendored model, 48 slots) on full12r14f
| depth | AJ (OA) | ms per frame in the DAVIS loop (dispatch) | CPU bf16 AJ |
|---|---|---|---|
| 12 | 70.84 (92.57) | 51.1 (48.05) | 70.98 |
| 10 | 69.90 (91.02) | 42.9 (40.51) | 70.07 |
Identical to rounds 9-13 (bit-identical numerics).

## 24. Round 15 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz, step8 time median of 60; a foreign
CPU job (another project's worker, 100 % of one core) ran throughout; baselines re-timed in the same sessions)
### 24.1 Step at depth 12 (every build BIT-IDENTICAL to full12r9a)
| build | change | dispatch ms | predicted | verdict |
|---|---|---|---|---|
| full12r14f (re-timed) | round 14 best | 47.26 | | |
| full12r15a | + SSX | 46.47 | -0.75 +- 0.15 (rev 10: -1.0 +- 0.07) | hit (-0.79) |
| full12r15b | + CPY64 (SSM out copies) | 46.27-46.30 | (ANALYSIS rev 8: -0.78 from a stale 798) | -0.18 |
| full12r15c | + CONVK | 45.54-45.61 | -0.9 +- 0.3 | hit (-0.72) |
| full12r15d | + P60G | 44.49 | rev 10 -1.08 +- 0.22 | hit (-1.06) |
| full12r15e2 | + RESCK | 44.20 | (rev 10 S1+S2+S3 -0.74) | -0.29 (S1 alone) |
| full12r15f | + RECK | 43.33-43.42 | -0.80 +- 0.15 | hit (-0.84) |
| full12r15g | in_copy by cpy64 | 43.19 | -0.30 +- 0.1 | miss (-0.15..-0.2) |
| full12r15h | in_need / in_done by div768 | 42.62 | ~-0.17 (SSM only) | -0.55 (the norms too) |
| full12r15i | out_begin / out_end by div768 | 42.09 | | -0.5 |
| full12r15j | NORM_SPLIT | 42.13 | | 0 (dead end) |
| full12r15iN1 (twin) | NORM_TWIN=1: norm pass-2 math skipped (outputs wrong) | 40.62 | | the pass-2 math = 1.5 ms |
| full12r15k2 | NRMK with X' / panel copies | 42.67 | | +0.6 (copies into the output ring) |
| full12r15k3 | NRMK, direct ring stores / pushes | 41.67-41.72 | | -0.40 vs r15i |
| **full12r15m** | + ZRING (z panels straight into the ring) | **41.52** (41.91 noisy; min 41.39-41.41) | | ~-0.12 (mins) |
P60G per-block attention wall (gap + span, us, blocks 1-11; r14f / predicted / r15d): 710 660 850 750 660 650 800 730 840 540 510 /
572 589 727 637 599 574 676 629 712 525 514 / 620 570 720 630 590 590 650 650 680 510 500 (mean 701 / 614 +- 18 / 610).
Timeline r14f -> r15h (ms per step): SSM + out 9.78 -> 6.76, attention 8.52 -> 7.08, norms 7.03 -> 6.46, GEMMs unchanged.
### 24.2 EW PROF (full12r15bQ / bR, cycles per chunk, mean of 8 columns, 216 chunks per SSM op)
in_wait 165, conv 1,125, gates 652, sigmoid 509, log_a + exp (ssx) 509, sqrt 493, recurrence 902, z panels 237, out_wait 31,
copies 477 (input-ring x parts 289, h + panels 217... ~3.5 cycles per vector into the output ring), rest 300, par 26 + 282.
### 24.3 One-core benches (cycles per call incl. ~20 of stamps; all bit-exact vs verbatim C++ on several seeds, guard zones clean)
| kernel | asm | C++ (same bench) | notes |
|---|---|---|---|
| ssx (16 vectors) | 497 (543 with the table on the stack) | 1,057 | II 20, 7 stages |
| cpy64 (16 / 32 vectors) | 59 / 75 | | 1 vector per cycle across banks |
| convk (16 vectors) | 251 | 377 | in situ the C++ phase is ~1,000 |
| resc (9 x 64 fp32) | 158 | 353 (-O2; minsize ~500 in situ) | |
| reck (16 vectors) | 292 | 558 | U 4, II 11 |
| nrm_rms / nrm_ln (one k-step, with X' stores) | 156 / 230 | 279 / 405 | II 9 (U 2) / 13 (U 2) |
zk (z panels, r15/gen_zk.py): 100 cycles against the C++ 97 on one core (not used; ZRING moved the push target instead).
### 24.4 Depth sweep and DAVIS (full12r15m, the round's final build; one xclbin, full{10,8,6}r15m sequences)
| depth | dispatch median (min) | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 41.52 (41.40) | 43.94 | 22.8 | 70.84 (92.57) | 70.98 |
| 10 | 35.03 (34.71) | 38.13 (r15k3 36.97) | 26.2 | 69.90 (91.02) | 70.07 |
| 8 | 27.86 (27.78) | 30.12 | 33.2 | not run | 68.22 |
| 6 | 20.96 (20.90) | 23.27 | 43.0 | not run | 64.77 |
DAVIS loop on full12r15m: 44.3 ms per frame (dispatch 41.96) at depth 12, 37.0 (35.07) at depth 10.
full12r15i: 42.06 / 35.26 / 28.24 / 21.24; full12r15k3: 41.83 / 34.97 / 27.93 / 21.01 (AJ the same: bit-identical numerics).

## 25. Round 16 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), cycles at 1.81 GHz, step8 time median of 60 after 20
warm-up; a foreign CPU job (another project's python worker, 100 % of one core) ran throughout; baselines re-timed in-session)
### 25.1 Step at depth 12 (every build BIT-IDENTICAL to full12r9a; lint = research tools/asmlint.py, 0 errors each)
| build | change | dispatch ms (min) | predicted | verdict |
|---|---|---|---|---|
| full12r15m (re-timed x3) | round 15 best | 41.52 / 41.50 (41.38-41.40) | | |
| full12r16a | + SQK | 41.32 (41.13) | -0.33 +- 0.05 (research -0.31..-0.35) | miss: -0.18..-0.25 (priced the loop, not the ~30-bundle preamble + call) |
| full12r16b | + PIN (flag #undef-ed by a header: no change) | 41.38 (41.18) | | dead build (same elf) |
| full12r16b2 | + INPART_INLINE, straddle by in_part | 41.23 (40.97) | research P-IN -0.25 +- 0.06 | miss alone (-0.09..-0.16) |
| full12r16b3 | + straddle by cpy64 | 41.03 (40.94) | | P-IN hit (-0.29) |
| full12r16c | + XONE | 40.75 (40.64); re-timed 40.85 / 41.18 | research X1b -0.18 +- 0.03 | miss high (-0.28: the phase was 432, not 289) |
| full12r16d | + GGK | 40.84 noisy; re-timed 40.61 / 40.37 (40.28 / 40.27) | main -0.32 +- 0.05; research G1 -0.50 +- 0.07 | ~-0.41 by cycles: main miss (better), research slight miss |
| full12r16e | + SSQ (-SSX) | 40.25 (40.13); re-timed 40.23 | -0.17 +- 0.04 (phase ~715 +- 20) | phase hit (715), step slight miss (-0.125 by cycles) |
| full12r16f | + SIG10 | 40.18 (40.08); re-timed 40.20 (40.10) | -0.10 +- 0.03 | miss (-0.05..-0.07) |
| full12r16g | + N1K, one call per unit | 40.13 (39.96) | research N1 -0.6 +- 0.25 | miss (-0.05..-0.12) |
| full12r16h | + N1K _begin / _step / _end | 39.96 (39.85); re-timed 40.04 (39.82) | | -0.16..-0.28 vs r16f: N1 miss |
| **full12r16z** | r16h's configuration, fresh build (the round's final) | **39.94 (39.83)** | | |
Twins (outputs wrong): r16tw2 NORM_TWIN=2 (pass 1 + statistics skipped) 38.20; r16tw4 NORM_TWIN=4 (the n1k steps skipped) 38.43.
### 25.2 EW PROF (cycles per SSM chunk, mean of 8 columns, 216 chunks per op)
| phase | r15m (r16p0) | r16a | r16b | r16d | r16e |
|---|---|---|---|---|---|
| in_wait | 80 | 79 | 79 | 79 | 80 |
| conv (+ part pointers) | 638 | 638 | 497 | 440 | 440 |
| gates | 651 | 651 | 655 | 365 | 365 |
| sigmoid | 509 | 509 | 509 | 509 | 509 |
| log_a + exp (ssx / ssq) | 508 | 508 | 508 | 511 | 715 |
| sqrt | 493 | 321 | 321 | 321 | 30 (empty, stamps) |
| recurrence | 304 | 304 | 309 | 313 | 313 |
| z panels | 259 | 259 | 258 | 264 | 264 |
| rest / out_wait | 212 / 30 | 212 / 30 | 213 / 30 | 213 / 30 | 213 / 30 |
| par wait / copy | 26 / 125 | 26 / 125 | 26 / 125 | 26 / 125 | 26 / 125 |
| cp_h_pan / cp_x | 192 / 432 | 192 / 432 | 192 / 432 | 192 / 269 | 192 / 269 |
| SSM op us | 532.1 | 511.5 | 495.6 | 436.5 | 426.1 |
Norms (NPROF, full12r16hN): n1k step 161.5 cycles per unit call (incl. ~20 of stamps), in_need 104.8 cycles per unit.
### 25.3 One-core benches (cycles per call incl. ~20 of stamps; bit-exact vs verbatim C++, guards clean)
| kernel | asm | C++ (same bench) | notes |
|---|---|---|---|
| sqk (16 vectors) | 311 | 609 (tables in one bank; in situ 493) | all 32,768 non-negative bf16 |
| ggk (both gates) | 310 | 533 (2 x gate_gemm) | 4 seeds + per-panel isolation |
| ssq (16 vectors) | 701 | 1,648 (loga_exp_a2 + sqrt_lut_n) | ssx 497 + sqk 311 = 808 separately |
| sigk II 10 / U 4 (32 vectors) | 431 | 898 | II 12 / U 2 (deployed before): 499 |
| n1p (pass 1, one unit) | 133 (call alone) | 211-217 | superseded by n1k |
| n1k (12 units, LN / RMS) | 2,449 / 2,427 | 2,545 / 2,451 | n_in 1-3, wrap offsets; stage + fp32 accumulators |
FIFO probes (r16/popprobe.s, popb.py): fill + 8 pops, restarted (probe 3), interleaved with a second FIFO (2): correct; ggk's B
timing (4): pops 8-12 wrong; dense 24 pops without (6) / with (7) a fill every 8: pops 8-12 wrong / all correct.
### 25.4 Depth sweep and DAVIS (full12r16z, one xclbin, full{10,8,6}r16z sequences; `step8.py sweep`)
| depth | dispatch median (min) | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 40.30 (39.86); time run 39.94 (39.83) | 43.55 (time run 41.98) | 23.0 | 70.84 (92.57); loop 42.3 ms per frame (dispatch 40.25) | 70.98 |
| 10 | 33.84 (33.43) | 37.06 | 27.0 | 69.90 (91.02); loop 36.3 ms per frame (dispatch 33.96) | 70.07 |
| 8 | 26.86 (26.74) | 29.12 | 34.3 | not run | 68.22 |
| 6 | 20.18 (20.11) | 22.53 | 44.4 | not run | 64.77 |

## 26. Round 17 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), 1.81 GHz; a foreign CPU job (another project's python
worker, 100 % of one core) ran throughout, and for part of the session my own CP-SAT run on testbox (load 7-14): dispatch times are
bimodal (~0.17 ms apart), so the table gives SPREAD lines (200 iterations after 20 warm-up, interleaved with r16z) where taken)
### 26.1 Step at depth 12 (every build BIT-IDENTICAL to full12r9a; asmlint 0 errors each)
| build | change | measured (ms) | predicted | verdict |
|---|---|---|---|---|
| full12r16z (re-timed x6) | round 16 best | median 39.94-40.42, p10 39.87-39.91, mean 40.00-40.29 | | |
| full12r17s2 | + S2 (attkb II 36, running max on the stack) | p10 39.56, median 39.71, mean 39.69 (-0.31 at every percentile) | research -0.38 (-0.29..-0.45); main -0.35 +- 0.08 | hit |
| full12r17al2 | + AL1 (params padded to 3 KB multiples) | p10 39.82, mean 39.87 (-0.05..-0.15) | research -0.21 +- 0.05 | miss |
| full12r17al (1 KB padding) | | TIMEOUT (1): a fill group must stay a 768 B slot multiple | | dead |
| full12r17np (G 2) / np3 / np4 / np6 | + NPIPE, NPIPE_G units per group | p10 39.96-39.98 / 39.89 / 39.82 / 39.80 (G 4-6: -0.05..-0.15; G 2: +0.07) | research -0.55 (-0.54..-0.56) | miss |
| full12r17x3 (sliding X1, N = 3) | rotating frames vs wrap | rotating 39.67-39.74, wrap 40.00-40.08 (240 iters x2) | main -0.36 +- 0.05 (rotating frame) | hit (-0.33) |
| full12r17x8 (sliding, N = 8) | sweep | d12 39.71 (cycle mean; rotating 39.61-39.74, wrap 39.92) | | superseded |
| full12r17c (sliding N 8 + S2 + AL1 + NPIPE 6) | | rotating 39.17-39.21, wrap 39.47, cycle mean 39.23 | | superseded by mod 4 |
| **full12r17d** | S2 + AL1 + NPIPE_G=6 + X1 mod 4 (XROT=4) | **sweep 39.16 (39.13); time run 39.19**; in a noisier slot p10 39.19, median 39.62, mean 39.55 | X1 research -0.34 alone / -0.26 after AL1 | hit (timeline ssm -0.47 incl. AL1) |
Timeline (one dispatch, r11/tlcmp): r16z 40.07 -> r17d 39.28 ms: attn 7.08 -> 6.78, ssm 4.45 -> 3.99, norms 4.93 -> 4.87, GEMMs +-0.06.
Text r17d: GEMM 16,336 / 16,384 (48 free), EW 15,808.
### 26.2 Probes
- NPROF (norm unit waits, cycles per unit, all norm ops): r16z in_need 101.5, out_begin 45.0 (new stamp), n1k step 161.5; NPIPE_G=2
  in_need 61.0, out_begin 46.6, step 160.6. Research rev 12's ewio predicted out_begin 147 (RMS) / 90 (LN) at r16z: the norms are
  not output-bound at the measured waits; the input side waits ~2x as long.
- CPROF (r17d config without N1K / NPIPE for text): convk alone in situ 280 cycles per chunk incl. ~25 of stamps (one-core bench 251),
  part pointers + mapping 101, the rest of the chunk 2,562; SSM op 351 us (r16e PROF 426, with ~14 stamps). The round-16 "conv 440"
  was convk + pointers + the straddle copy (AL1 removed it) + stamps: no in-situ penalty on convk itself.
- attkb one-core bench (r13/attkb.py --hashes, S2 vs deployed, 36 cases): P / maxima hashes identical; cycles per call warm
  unflagged 1,634 vs 1,558, flagged 4,482 / 3,751 vs 5,730 / 4,528, chunk 0 5,772 vs 6,225, last chunk 4,140 vs 4,953 (warm 2,202
  vs 2,145). The bench's compare against its C++ chunk_k reports BAD P for the deployed attkb as well (pre-existing).
### 26.3 Depth sweep and DAVIS (full12r17d, one xclbin, 4 phase sequences per depth; `step8.py sweep`, phases p0-p3 within 0.04)
| depth | dispatch median (min) | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | CPU bf16 AJ |
|---|---|---|---|---|---|
| 12 | 39.16 (39.13) | 41.01 | 24.4 | 70.84 (92.57); loop 42.2 ms per frame (dispatch 39.55) | 70.98 |
| 10 | 32.78 (32.74) | 34.86 | 28.7 | 69.90 (91.02); loop 35.0 ms per frame (dispatch 32.95) | 70.07 |
| 8 | 26.29 (26.25) | 28.37 | 35.2 | not run | 68.22 |
| 6 | 19.79 (19.72) | 21.87 | 45.7 | not run | 64.77 |

## 27. Round 18 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), 1.81 GHz; the foreign CPU worker ran throughout: A/B
timing by r18/ab.py = both builds in ONE process, 20-22-dispatch blocks interleaved, 2 discarded per block)
### 27.1 Step at depth 12 (every kept build BIT-IDENTICAL to full12r9a; asmlint 0 errors each)
| build | change (cumulative) | A/B delta vs previous (p10 / median / mean ms) | predicted | verdict |
|---|---|---|---|---|
| full12r18a | r17d + HRING (reck stores h into the output ring, 2D wrap) + PTRG (part pointers) | -0.25 / -0.35 / -0.34 (vs r17d) | tcall -0.20..-0.30 | hit |
| full12r18b | + ZK (z panels by r15/gen_zk.py, re-solved II 5 U 2) | -0.19 / -0.14 / -0.17 | tcall -0.18 | hit |
| full12r18d | + KSC (k-step records build-time data) | -0.14 / -0.14 / -0.16 | ~0 (text lever, -288 B) | hit (bonus: per-call check ~16 cycles x 15,552) |
| full12r18e | + NIOC (norm ring glue inline, ring state in locals) | -0.23 / -0.20 / -0.17 | tcall static -0.55 | miss (pass 1 waits on input) |
| full12r18g | + V2T (attv / attvs once per chunk for both tiles) | -0.06 / -0.08 / -0.08 | -0.06..-0.10 | hit |
| full12r18h | + CGF (convk pushes xc's bfp panels) | -0.05 / -0.06 / -0.06 | research -0.10..-0.13, tcall ~-0.06 | miss vs research, hit vs tcall |
| full12r18j | + P35 (mm_ktile: a tile's k-steps per call, B locks by id) | 30 rounds: -0.11 / -0.10 / -0.08 | research -0.44 +- 0.1 | miss |
| full12r18r6 = r18m | + ROT g2a (rotating-accumulator k-step, research pass 14 + acq gap 6) | -3.34 / -3.27 / -3.35 (vs r18j) | -3.84 +- 0.4 (calibrated), -3.46 (uncalibrated) | slight miss |
| r18r4 / r18r5 | ROT g2f as delivered (acq.cond 1 bundle before the A fill) / + plain rel | regress DIFFERENT, run-to-run varying (rms_rel 0.15-0.25) | | race: fixed by g2a |
| r18hG4 / r18hG12 | NPIPE_G 4 / 12 with NIOC | +0.06..+0.13 | | dead |
| K1C (not kept) | attention -m prep loop two row blocks per trip | tcall -28 per (tile, chunk), +96 B | | dropped (text) |
Text at r18m: GEMM 15,536 / 16,384, EW 15,808.
### 27.2 Probes
- DMAB (r18/dmab.py): memtile -> core DMA 63 KB at 8.07 B/cycle into bank 3. Core one 64 B access per cycle, 256-access segments:
  same bank loads 306 -> 364, stores 314 -> 373 cycles (DMA slows to 7.45 / 7.31 B/cycle); bank 1 and nops unchanged. ~1 core
  cycle per 32 B DMA write into the bank in use.
- KS-P38 (mm8 -DKSP38, r18/ksp38.py, per k-step, median over 6,240 tiles): mm_ks MEMORY_STALL 173.7, LOCK_STALL 13.3, TRUE 2,046.7,
  timer 2,056.7. mm_rot (r18r6K): TRUE 1,603.2, MEMORY 167.5, LOCK 13.4, timer 1,613.2.
- NIO2 (ew8 -DNIO2, cycles incl. stamps): r18d RMS nrm 144.8 / pass-2 unit 341.8 / pass-1 unit 302.0, LN 229.5 / 428.4 / 287.5;
  r18h (NIOC, ZK off for text) RMS 147.2 / 302.0 / 293.2, LN 231.9 / 390.7 / 248.6.
- ROT one-core bench (r18/rotbench.py, research's rotb): g2f and g2a 36/36 pieces identical after 1 and 4 k-steps, warm 1,431.
### 27.3 Depth sweep and DAVIS (full12r18m, one xclbin, 4 phase sequences per depth)
| depth | dispatch median (min) | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | r17d dispatch |
|---|---|---|---|---|---|
| 12 | 35.20 (34.80) | 38.44 | 26.0 | 70.84 (92.57); loop 37.0 ms per frame (dispatch 34.98) | 39.16 |
| 10 | 29.11 (29.07) | 30.99 | 32.3 | 69.90 (91.02); loop 31.7 ms per frame (dispatch 29.32) | 32.78 |
| 8 | 23.30 (23.29) | 25.31 | 39.5 | not run | 26.29 |
| 6 | 17.53 (17.50) | 19.61 | 51.0 | not run | 19.79 |
(r18k = r18h + P35 before ROT: 38.11 / 31.91 / 25.57 / 19.23; DAVIS 70.84 / 69.90.)

## 28. Round 19 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), 1.81 GHz; the foreign CPU worker ran throughout; A/B by
r18/ab.py = both builds in ONE process, 22-dispatch blocks interleaved, 2 discarded per block)
### 28.1 Step at depth 12 (every kept build BIT-IDENTICAL to full12r9a; asmlint 0 errors each; timeouts this round: 0)
| build | change (cumulative) | A/B vs previous (p10 / median / mean ms) | predicted | verdict |
|---|---|---|---|---|
| full12r19a | r18m + ROT g1a (padda restart, acq gap 6 / rel gap 8) | -0.104 / -0.111 / -0.079 (vs r18m) | -0.13 (15 cycles x 8.59 us) | hit |
| full12r19b | r18m + ROT g0a (B wraps inside the last pop) | -0.174 / -0.183 / -0.152 (vs r18m) | -0.23 (26 cycles) | slight miss (~0.75 lands) |
| full12r19c | + ZHI (acc_hi zeroed after the C release) | 40 rounds: -0.156 / -0.149 / -0.160 | -0.15..-0.20 | hit |
| full12r19d | + CSPLIT (C drains in two parts; MLIR patch) | 40 rounds: -0.250 / -0.476 / -0.261 | -0.33 | hit (p10 / mean slightly low) |
| full12r19e2 | + zero tail calls (text -208 B, acc_hi 0x719C0) | +0.001 / -0.009 / +0.004 (vs r19d) | 0 | hit |
| full12r19f | + EPLR (plain epilogues release part 1 after rb 11) | -0.100 / -0.119 / -0.096 (vs r19d) | -0.06 | better |
| full12r19g | + KPREP (attention K prep straight-line) | -0.130 / -0.093 / -0.147 | -0.12..-0.18 | hit |
| full12r19h | SEQPRE v1 (loads + sends + drains before the wait) | +1.098 / +1.015 / +1.091 (vs r19g) | -1.4 +- 0.5 (research) | dead (queue block) |
| full12r19h2 | SEQPRE v2 (sends after the fills) | +0.109 / +0.131 / +0.145 | | dead |
| r19h3m2 / r19h3m1 | SEQPRE only drains / only slot loads before the wait | +0.150 / +0.158 / +0.151; -0.039 / -0.042 / -0.032 (40) | | m1 kept |
| **full12r19z** | r19g + SEQPRE=1 (the round's best) | **-0.880 / -0.888 / -0.874 vs r18m** (30 rounds: 34.88 -> 33.99 median) | | |
Text at r19z: GEMM 15,984 / 16,384, EW 15,808.
### 28.2 Probes
- Wrap probes (r19/wrapgen.py + wrapb.py, one core, 128 pops over two passes of 4,608 B): control 128/128; `vldb.pop.576.2d` with
  m2 = -4544 64/128 (pass 2 wrong), m2 = -4608 128/128 (the 2D step is ADDED to the 64 B line step); padda [p1], m2 one bundle before
  the fill 128/128, in the fill's bundle 64/128.
- TPROF (mm8 -DTPROF, r19/tprof.py, r19b-based, 6,240 tile records, cycles median): first k-step 1,487, k-step 1,576, epilogue plain
  1,310 / gelu_pan 4,918 / gated 3,263 / qkv 1,941, hand (epilogue end -> next mm_ktile) 3,010 (gated second tile 682 = zero + loop).
- Timelines (step8 timeline, 3 runs each; r19/tlsum.py): r18m attn 6.66, norms 5.01, ssm 3.50, GEMMs 19.9 (k-step model 13.4);
  r19d attn 6.69, norms 4.88, ssm 3.49, GEMMs ~19.2. Norm walls per column (r19d): norm0 95.6 / norm1 67.7 / norm2 98.0 / norm3 82.7
  us, each gap ~16.5 us.
- CPRATE (design8 env CPRATE=400 [CPRATE_POLL=1] on mlp3_a's path, vs r19g): write32 +0.268 / +0.296 / +0.274 ms (0.06 us each);
  maskpoll (always satisfied) +0.520 / +0.553 / +0.522 (0.11 us each).
- DDRCAP (r19/ddrcap.py, shim -> core streams, 4 MB each, awaited; ~0.08 ms dispatch overhead inside): 4 x 1 and 2 x 2 streams
  16 MiB in 0.413 ms; 8 x 1 and 4 x 2 32 MiB in 0.650; 8 x 2 64 MiB in 1.314 -> one stream ~7 B/cycle, aggregate read cap ~55-59 GB/s.
### 28.3 Depth sweep and DAVIS (full12r19z, one xclbin, 4 phase sequences per depth; r18m re-swept in the same session)
| depth | dispatch median (min) | whole step | fps (whole step) | DAVIS AJ (OA), 10 videos | r18m same session |
|---|---|---|---|---|---|
| 12 | 33.88 (33.86) [first sweep 34.22, slow mode] | 35.95 | 27.8 | 70.84 (92.57); loop 36.8 ms per frame (dispatch 34.15) | 34.79 / 37.03 |
| 10 | 28.37 (28.34) | 30.25 | 33.1 | 69.90 (91.02); loop 30.7 ms per frame (dispatch 28.55) | 29.20 / 32.09 |
| 8 | 22.74 (22.70) | 24.96 | 40.1 | not run | 23.33 / 25.55 |
| 6 | 17.09 (17.07) | 18.99 | 52.7 | not run | 17.56 / 20.02 |
