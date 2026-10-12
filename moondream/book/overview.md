# moondream: the workload, the split and the budget

moondream is low-latency VLM inference for local use; the hardware can be the NPU or a strong GPU. Here moondream2 is used as an object-presence (yes/no) query ("Is there a mug in this image? Answer yes or no.") on camera frames. The GPU alone runs it in
**88.7 ms per frame** (MEASURED, 10.9 fps, 98.0 % on 493 frames; [GPU path](gpu.md)). The NPU project tried to take the vision tower off
the GPU so that the two sides work on different frames at once. It built a 27-block GEMM program at **33.8 ms per frame**, a fused
attention at **0.675 ms per block**, and bf16 GELU and LayerNorm kernels. Summed, the tower on the NPU costs **~75 ms** against
~30 ms for the same work on the GPU (MEASURED pieces, summed in MD RESULTS phase 9-12). So the pipelined design is real but half-built:
the NPU does the GEMMs, the GPU keeps the text model, and the tower's other pieces are not placed yet. The work stopped after phase 12
(the project is paused, not finished). This chapter is the map; the other chapters have the mechanisms.

## The workload

- **Yes/no presence gate.** One 720x1280 camera frame is squashed to 378x378 (one crop, 27x27 = 729 patches of 14 px), the question is
  a fixed sentence, and the answer is the yes/no logit margin of the first decoded token. Test set: 493 frames, tv 402 (46 found) and
  mug 91 (7 found); label yes means a VLM judge said "found" (`moondream/gpu/RESULTS.md`, head).
- **The model.** A SigLIP vision tower (27 blocks, dim 1152, MLP 4304, 16 heads x 72, pre-LN), a projection MLP (2304 -> 8192 -> 2048
  from `cat([x, x])`), and a 24-layer text model (dim 2048, int8) that prefills ~730 tokens and answers from one logit pair
  (MD RESULTS phase 1, phase 4).
- **Grounding** (boxes and points for a named object) uses the same model family but a longer decode. It was only
  evaluated, never optimised; [moondream3](moondream3.md) compares the newer model on it.
- **Where the 79.6 ms of encode goes** (MEASURED in situ on the GPU, MD NOTES phase 1): text int8 GEMMs 48.4 ms, vision bf16 GEMMs
  20.7 ms, flash attention ~10 ms (vision ~4.5), LN/GELU/adds/quant ~6 ms. "Encode" includes the text prefill. The part an NPU can take over is
  the vision tower plus projection, **~30 ms**, not the whole 79.6.

## Which side runs what

| stage | GPU (fast path) | NPU (built) | status |
|---|---|---|---|
| patch embedding | in the tower | GEMM 147 us (K 588 -> 640) | MEASURED, bit-exact ([npu-gemm](npu-gemm.md)) |
| 27 x (qkv, attn_out, fc1, fc2) | bf16 HIP WMMA, 20.7 ms | one program, **33.8 ms** (design6) | MEASURED, bit-exact |
| attention, 16 heads x 729 x 72 | flash attention, ~4.5 ms | 0.675 ms/block = 18.2 ms | MEASURED per block; not in the program ([npu-attention](npu-attention.md)) |
| GELU (fc1 epilogue) | fused in the GEMM | ~15 ms/frame | MEASURED per op; not in the program ([npu-elementwise](npu-elementwise.md)) |
| LayerNorm + residual | fused kernels | 54 x 146 us = ~8 ms | MEASURED per op; not in the program |
| projection MLP | ~2 ms | does not fit a static chain (44 loads) | stays on the GPU |
| text prefill + query | int8 WMMA, 24.9 + 8.3 ms pooled | 180 ms pooled | stays on the GPU (phase 4) |

The reason the text model stays: the NPU's int8 mmul is the same 512 MACs per instruction as bfp16, and weight streaming through
the per-core input port peaks at ~20 GB/s (MD RESULTS phase 4; [numerics](numerics.md#int8)).

## The pipelined 27 fps design

Frame N is encoded by the NPU while the GPU prefills and queries frame N-1, so the rate is 1 / max(NPU, GPU) instead of their sum.
Phase 5 measured the two halves running together on the shared DDR; phase 6 re-measured with the one-program encoder.

| side | alone | during the other | slowdown | evidence |
|---|---|---|---|---|
| NPU, 27 blocks of GEMMs in one dispatch | 33.8 ms | 36.8 ms (p90 37.8, 1,545 frames) | +9 % | MEASURED, [p6](../npu/p6/), MD RESULTS phase 6 |
| GPU, pooled text prefill + query (24.9 + 8.3 ms) | 33.2 ms | 34.6 ms | +4 % | MEASURED, same |

1 / max(36.8, 34.6) = **27 fps**, latency ~71 ms per frame, the sides balanced within 6 % (MEASURED as two concurrent loops; GPU-only
pooled is 15.6 fps; phase 5 with four xclbins gave 22 fps). Two cautions on the 27:
- The GPU loop starts from cached projected tokens, and the NPU loop has GEMMs only. Attention, LN, GELU and the projection (~8.5 ms of
  tower work plus ~2 ms, MD RESULTS phase 1 and 6) belong to neither loop. If the GPU kept them, its side would be ~45 ms and the rate
  ~22 fps (EXTRAPOLATED; arithmetic from the table, not a run).
- The DDR is shared: CPU memcpy hogs slowed a qkv GEMM by 21 % and a GPU job moves ~10x more DDR traffic
  (MEASURED, MD NOTES phase 2c; [measurement](../../common/book/measurement.md#check-the-box-before-timing)). The +9 % above is that effect.

## The tower budget

The NPU alone, 27 blocks, Turbo (MEASURED pieces; the sum is arithmetic, MD RESULTS phase 9-12):

| piece | phase 8 | phase 12 | target | what limits it |
|---|---|---|---|---|
| GEMMs (qkv, attn_out, fc1, fc2) | 34 | **34** | ~20 | DDR-bound: 1,317 MB per frame at 37-39 GB/s |
| GELU epilogue | 15 | **15** | ~4-6 | allocation-bound schedule, 492 bundles per 64 elements vs a bound of ~114 |
| LN + residual | 8 | **8** | 4 | separate passes, not fused |
| attention | 102 | **18.2** | 10 | exp block and softmax bookkeeping (~25 % vector issue) |
| patch embedding | 0.15 | 0.15 | | |
| **sum** | **159** | **75** | **~40** | GPU does the same work in ~30 ms |

Attention was the largest piece until phase 9; after it, the GEMMs are. Beating the GPU needs the whole tower at about 35 ms, a 2x cut
that no single item delivers (phase 12 prices the reachable GEMM traffic cut at ~7 ms and the GELU schedule at ~10 ms).

## How the phases built on each other

| phase | question | result |
|---|---|---|
| p1 | Is a bfp16 GEMM on the NPU worth it? | generic GEMM design: 55 ms per frame in 111 dispatches vs 20.7 ms on the GPU; a GPU emulation says keep an fp32 accumulator ([numerics](numerics.md)) |
| p2a-p2c | Why not faster? | DDR-bound -> A resident in the memtile (design3, 36.2 ms) -> kernel and timing study (33.2 ms) ([npu-gemm](npu-gemm.md)) |
| p3 | Turbo, a better inner loop | 1.59 bundles per MAC, 29.7 ms (4 dispatches) |
| p4 | Can the text model move too? | no: 180 ms pooled |
| p5 | Do NPU and GPU overlap? | 22 fps NPU-bound; the asm route was declared closed (wrongly, see p10) |
| p6 | One program for the encoder | 33.8 ms, 27 fps; fp32 GELU and LN built but emulated and slow |
| p7-p8 | bf16 elementwise | GELU, LN, LN + residual correct; first fused attention, slow (3.78 ms/block) |
| p9 | Attention | 0.675 ms/block, tower sum 159 -> 75 ms |
| p10-p12 | Hand schedules and traffic | asm route proven; hand-scheduled score block and GEMM loop gave no frame gain; GELU and GEMM traffic priced, not built |

## What worked

- **A resident in the memtile with lock-counted repeats** removed the first DDR bound (qkv 18.8 -> 10.8 MB, 547 -> 446 us, MEASURED,
  [p2b](../npu/p2b/)) and made one dispatch possible ([npu-gemm](npu-gemm.md)).
- **One program for 27 blocks** saved three context switches of ~2.7 ms each (40.4 -> 33.8 ms, MEASURED, [p6](../npu/p6/);
  [fusion](../../common/book/fusion.md#one-dispatch)).
- **Accuracy gates before kernels.** The GPU emulations (`moondream/gpu/bfpemu.py`, `bf16emu.py`) said which numerics to build, and the
  kernels came out bit-exact to the emulations ([numerics](numerics.md)).
- **Attention 5.6x faster in one phase** (3.78 -> 0.675 ms/block, MEASURED, [p9](../npu/p9/)), by fixing the rounding mode, then the
  exp, then the stream structure ([npu-attention](npu-attention.md)).

## What didn't work

- **Standalone NPU GEMMs never beat the GPU's 20.7 ms** (best 29.7 ms with four dispatches, 33.8 ms as one program). Only the pipelining
  pays. The 1.4 GHz clock behind the "% of peak" figures was a lower bound, not the clock ([H1](../../common/book/negative-results.md#h1) <!-- nr: H1 -->;
  [machine](../../common/book/machine.md#array-and-clock)). Bundles per MAC are clock-free and stay valid.
- **The text model on the NPU**: 180 ms pooled (MEASURED, one dispatch of 24 layers) against ~25-48 ms on the GPU. A port-topology
  limit, not a tuning gap ([N5](../../common/book/negative-results.md#n5) <!-- nr: N5 -->).
- **Pricing from structure**: attention was priced at ~27 ms per frame (phase 7) and measured 102 ms (phase 8). The fp32 elementwise
  estimate (120 ms GELU) was the first to turn the plan around ([N11](../../common/book/negative-results.md#n11) <!-- nr: N11 -->, [N17](../../common/book/negative-results.md#n17) <!-- nr: N17 -->). See [method](../../common/book/method.md) for the general pattern.
- **Calling the asm route closed** after one failed inline-asm attempt (phase 5); phase 10 built it from `clang -S` output
  ([N16](../../common/book/negative-results.md#n16) <!-- nr: N16 -->, [A23](../../common/book/negative-results.md#a23) <!-- nr: A23 -->, [npu-attention](npu-attention.md#what-didnt-work)).
- Per-chapter failures are in [npu-gemm](npu-gemm.md#what-didnt-work), [npu-attention](npu-attention.md#what-didnt-work),
  [npu-elementwise](npu-elementwise.md#what-didnt-work), [numerics](numerics.md#what-didnt-work), [gpu](gpu.md#what-didnt-work) and
  [moondream3](moondream3.md#what-didnt-work).

## Lessons

- Offloading wins by overlap, not by speed: the NPU never matched the GPU on the same work, but a balanced pair is faster than either.
  Measure both halves running together on the shared DDR.
- Check what the sum of the pieces is before polishing one. The tower budget table was the most useful artefact of the project: it showed
  attention (phase 8), then the GEMMs, as the item to attack.
- A pipeline rate is only as honest as its inventory. List what is on neither side.
- Gate numerics with a host emulation first, then demand bit-exactness from the kernel.

## Going deeper

- Raw record: `moondream/npu/NOTES.md`, `moondream/npu/RESULTS.md` (phases 1-12; code in `moondream/npu/pN/`), `moondream/gpu/RESULTS.md`;
  import commits `258ba90` (NPU, phases 1-12) and `cddae92` (GPU, moondream3).
- Machine level: [gemm](../../common/book/gemm.md), [data-movement](../../common/book/data-movement.md), [fusion](../../common/book/fusion.md),
  [elementwise](../../common/book/elementwise.md), [attention](../../common/book/attention.md).
- `moondream/npu/p6/nputloop.py` (the NPU frame loop), `moondream/gpu/textloop.py` (the GPU text-only loop).
