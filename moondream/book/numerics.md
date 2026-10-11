# moondream numerics: bfp16 vs bf16, Hadamard, int8

Every numeric decision for the NPU was made on the GPU first, by emulating the NPU arithmetic in the existing fast path and scoring the labelled
frames, before any kernel was written. The answers: **bfp16 operands are fine if the accumulator stays fp32 across the whole K** (97.6 % against
98.0 % on 493 frames, two more false positives; MEASURED, MD RESULTS phase 1); a **bf16 C between K chunks costs 1.4 points**; **a Hadamard rotation hurts**;
**int8 breaks the vision tower** (89.2 %); and **bf16 elementwise ops are fine with the sigmoid form of GELU** (99.6 % with the full NPU numerics at 16x16 pooling against 99.4 % for the GPU path;
MEASURED, `moondream/gpu/BF16_ELEMENTWISE.md`). The kernels were then built to match the emulations bit for bit. General rules are in
[gemm](../../common/book/gemm.md#is-it-compute--or-data-bound) and [elementwise](../../common/book/elementwise.md#judging-accuracy).

## The gate and the emulators

The gate is the 493-frame set of the [GPU chapter](gpu.md) (tv 402 / 46 found, mug 91 / 7 found); a change is acceptable if it costs about two frames.
The emulators are `moondream/gpu/bfpemu.py` (bfp16 with a shared exponent per block of 8 values and 8-bit mantissas; verified bit-exact with
`aie.utils.bfp.quantize(rounding="conv_even")` on 589,824 values) and `bf16emu.py` (every op computes in fp32 and rounds to bf16 immediately; constants bf16; exp correctly rounded from its bf16 input,
the best case for a bf16 exp unit). They replace the GPU path's linears or `_vision` (`bench_bfp.py`, `frame_preds.py --bf16`).

## bfp16 operands and the accumulator

Vision encoder only (patch embedding, all blocks, projection), text model unchanged, 493 frames (MD RESULTS phase 1; tv false positives, no false negatives in any row):

| vision linears | accuracy | tv fp | agreement with stock |
|---|---|---|---|
| HLin, today's GPU path | 98.0 % | 10 | 99.4 % |
| bf16 accumulator every 64 K only | 97.8 % | 11 | 99.6 % |
| bfp weights, fp32 accumulator | 97.4 % | 13 | 98.4 % |
| bfp both, bf16 accumulator every 64 K (the stock mlir-aie design) | 96.6 % | 17 | 98.4 % |
| **bfp both, fp32 accumulator over all K** | **97.6 %** | 12 | 99.0 % |
| bfp both + 64-wide Hadamard along K | 96.1 % | 19 | 98.0 % |

Reading: operand rounding alone is cheap; what the stock design lost was the bf16 C between 64-K chunks ([F4](../../common/book/negative-results.md#f4) <!-- nr: F4 -->, 1.4 points
against the GPU path, 1.0 against the fp32-accumulating bfp variant). The kernel (`p2/mm_f32acc.cc`) therefore keeps an fp32 accumulator for the whole K, split between the output buffer and
a static buffer because an fp32 96x96 tile does not fit L1 beside a bf16 tile. Its output matches `bf16(float64 product of the decoded bfp operands)`: 0.0000 % of outputs differ, max 1 ulp, rms 1.66e-3 = the bf16 rounding
floor (MEASURED on all seven encoder shapes, [p2](../npu/p2/), [p2b](../npu/p2b/)). Fixing the precision cost no speed (qkv 463 us against 486 us with bf16 accumulation);
it did cap the output-stationary stream at ~67 % of peak, since an fp32 tile needs more input-port cycles ([npu-gemm](npu-gemm.md)).

## Hadamard rotation

A 64-wide Hadamard transform along K before the bfp quantisation is the standard trick to flatten outliers within a shared-exponent block. Here (BfpRotLin) it made things worse:
96.1 % against 96.6 % for the same design without it, 19 false positives against 17 (MEASURED emulation, [F5](../../common/book/negative-results.md#f5) <!-- nr: F5 -->). The cause was not investigated; it was dropped without a kernel.

## Int8

- **On the GPU**: per-token int8 for the vision tower gave 113 ms but 89.2 % (53 tv false positives), against 97.8 % for int8 on the text model only. SigLIP activations do not survive per-token int8
  (MEASURED, `moondream/gpu/RESULTS.md`, `fast-all-int8-graph`). GPU int8 WMMA also peaks at 57 TOPS against bf16's 55 TFLOPS, so the assumption "int8 is 2x" was wrong there as well ([gpu](gpu.md)).
- **On the NPU**: the dense int8 `mmul` is 8x8x8 = 512 MACs per instruction, the same rate as bfp16 (the 1,024-MAC 8x16x8 form needs a sparse B operand); int8 panels are 64 B against 72 B
  for bfp16, 11 % fewer bytes. Int8 buys nothing in MACs and little in bytes ([F6](../../common/book/negative-results.md#f6) <!-- nr: F6 -->; MACHINE: MMUL rate).
- The text-model W8A8 kernels (`p4/mm_i8.cc`) did match the GPU's numerics: int32 exact accumulation, the dequant epilogue in fp32 in the HIP kernel's order, 0.0000-0.0005 % mismatch
  (0.009-0.010 % with GELU, max 2-3.5 ulp, exp polynomial against numpy's exp). The text model still stays on the GPU for speed reasons ([npu-gemm](npu-gemm.md)).

## bf16 elementwise

LayerNorm, GELU, softmax and residual adds rounded to bf16 at every step, fp32 only for reductions (the NPU has no fp32 vector arithmetic; [npu-elementwise](npu-elementwise.md)). 462 frames, 24 positives
(`BF16_ELEMENTWISE.md`; flips are against the exact path, or against pool16 for pooled rows):

| config | accuracy | fp | decision flips | max margin shift |
|---|---|---|---|---|
| exact (GPU path) | 98.3 % | 8 | - | - |
| bfp16 GEMMs only | 97.8 % | 10 | 6 | 0.93 |
| bf16 LayerNorm | 97.6 % | 11 | 5 | 0.75 |
| bf16 GELU, tanh form | 97.0 % | 14 | 10 | 2.22 |
| bf16 softmax | 97.2 % | 13 | 7 | 0.71 |
| bf16 GELU, sigmoid form | 97.2 % | 13 | 5 | 0.97 |
| all bf16 + sigmoid GELU + bfp16 GEMMs | 97.8 % | 10 | 6 | 1.42 |
| pool16 (GPU path) | 99.4 % | 3 | - | - |
| **all bf16 + sigmoid GELU + bfp16 GEMMs, pool16** | **99.6 %** | 2 | 1 | 0.75 |

Per-op relative rms error against fp32: LayerNorm 3.7-4.4e-3 (GPU path 1.6e-3), softmax 4.2-7.6e-3, tanh-form GELU **1.4-2.1e-2**, sigmoid-form GELU 0.5-4.5e-3. The tanh form
`1 - 2/(exp(2u)+1)` loses precision to cancellation; the sigmoid form `x / (1 + exp(-2u))` is the same function. The verdict: bf16 elementwise is acceptable with the sigmoid form; at 729 tokens the full NPU numerics
match the bfp16-GEMM-only result (+2 false positives against exact), at pool 16 they are one frame better than the GPU path.

## Attention operands

The kernel (`p9/attn15.cc`) uses bfp16 Q, K, V and P with an fp32 accumulator, a bf16 running max, row sums from a ones panel and a bf16 reciprocal. Emulated as `attn_bfp`, it has relative rms 1.0-1.6e-2
against fp32 (bf16 attention 0.4-0.8e-2). Accuracy: 97.6 % against 97.8 % with bf16 attention at 729 tokens (one more false positive), 99.6 % at pool 16, no missed positive (smallest true-positive margin +4.8, same as exact).
On random data the kernel is at the operand-rounding floor: rms 9.8e-3 against `bf16emu`, a host model of the bfp-rounded inputs already 8.1e-3 away (MEASURED, [npu-attention](npu-attention.md)).
The conversions also depend on the core's rounding mode, which defaults to floor, not round-to-even.

## Reading the accuracy numbers

- The deciding frames are few. Almost every flip is a tv frame of session `sessB` that the GPU path already scores within +-0.8 of zero, a band of ~15 monitor-like negatives; false-positive counts move up and down by noise
  (bf16_all has fewer than bf16_gelu alone). No configuration misses a positive.
- Cross-process noise on the GPU is of the same size: Triton autotune choices changed the accumulation order and flipped frames (97.0 to 98.0 % for one config). Compare emulations within one run.
- Two frame sets appear: 493 frames (the phase-1 table, 53 positives) and 462 frames with 24 positives (the bf16 study, "user-labelled"). Their absolute accuracies differ (98.0 vs 98.3 % for the exact path), so the tables above must not be mixed.

## What worked

- Emulate first, score on the labelled frames, then build the kernel and demand bit-exactness to the emulation.
- fp32 accumulation across the whole K with bfp16 operands: ~2 frames of accuracy, with 72 B per 64-value panel instead of 128 B.
- The sigmoid GELU form, found from per-op error measurements before any NPU time was spent.

## What didn't work

- [F4](../../common/book/negative-results.md#f4) <!-- nr: F4 --> bf16 C between 64-K chunks (the stock mlir-aie design): -1.4 points. Lesson: fp32 accumulator over the whole K.
- [F5](../../common/book/negative-results.md#f5) <!-- nr: F5 --> Hadamard rotation along K: 96.1 % against 96.6 %.
- [F6](../../common/book/negative-results.md#f6) <!-- nr: F6 --> int8: 89.2 % in the vision tower and no MAC-rate gain on the NPU.
- Not yet in the index (see the hand-back):
  - **GELU in the tanh form** in bf16 (up to 2.2 margin shift, 10 flips).
  - **A plain bf16 exp and reciprocal in the GELU**: 38.8 % one-ulp flips against 0.998 % for the split-operand form.
  - **The folded projection (its two K halves summed) and the fused residual epilogue on the GPU** (98.4 % and 98.8 % agreement): bf16 rounding of a changed order moves decisions.
  - **fp16 on the GPU**: 96.1 %.

## Lessons

- Spend accuracy on the accumulator, not on the operands: bfp16 operands with fp32 accumulation are within two frames; bf16 C is not.
- Rank elementwise ops by per-op error before choosing which to keep in fp32; one of five (GELU in its tanh form) carried the damage.
- The test set is small and its decisions concentrate on ~15 borderline frames. Look at margins and flips, not only at accuracy, and keep one frame set per comparison.
- A rounding mode or reference that differs between two kernels makes their errors incomparable.

## Going deeper

- `moondream/npu/RESULTS.md` "Precision check on the GPU" (phase 1), `moondream/gpu/BF16_ELEMENTWISE.md` (method, tables, margins, attention follow-up), `moondream/gpu/bfpemu.py`, `bf16emu.py`, `bench_bfp.py`, `rescore.py`.
- Kernel references: `p2/mm_f32acc.cc`, `p4/mm_i8.cc`, `p8/expsplit.h`, `p9/attn15.cc`.
- General: [gemm](../../common/book/gemm.md), [elementwise](../../common/book/elementwise.md), [machine](../../common/book/machine.md).
