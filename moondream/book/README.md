# Book: moondream on Strix Halo

The model-specific chapters for moondream2 on the gfx1151 GPU and the XDNA2 NPU, plus the moondream3-preview evaluation.
The work predates the TAPNext port and is not active. Read [the machine-level book](../../common/book/README.md) first.

Each chapter uses the same sections, so links stay stable: a lead, the topic sections, `## What worked`,
`## What didn't work` (anchor `#what-didnt-work`), `## Lessons`, `## Going deeper`.

| # | chapter | covers | general part |
|---|---|---|---|
| 1 | [overview.md](overview.md) | the yes/no and grounding workload, which side runs what, the pipelined 27 fps design, the NPU tower budget | [fusion](../../common/book/fusion.md) |
| 2 | [gpu.md](gpu.md) | the GPU fast path: static shapes, Triton and HIP WMMA GEMMs, int8 text model, image-token pooling, 88.7 ms per frame | — |
| 3 | [npu-gemm.md](npu-gemm.md) | design2 / 3 / 6: A resident in the memtile, BD halves, one program for 27 blocks, the DDR-bound frame, the k-loop | [gemm](../../common/book/gemm.md), [data movement](../../common/book/data-movement.md) |
| 4 | [npu-attention.md](npu-attention.md) | attention v2 → v15: bfp operands, one pass, rounding mode, K/V broadcast, exp, 3.78 → 0.675 ms per block | [attention](../../common/book/attention.md) |
| 5 | [npu-elementwise.md](npu-elementwise.md) | GELU, LayerNorm, LN + residual, the fp32-emulation finding, bf16 numerics | [elementwise](../../common/book/elementwise.md) |
| 6 | [numerics.md](numerics.md) | bfp16 vs bf16 accuracy on the 493-frame set, bf16 C chunks, Hadamard, int8 | [gemm](../../common/book/gemm.md#is-it-compute--or-data-bound) |
| 7 | [moondream3.md](moondream3.md) | moondream3-preview vs moondream2 on grounding | — |

The raw record: `../npu/{NOTES,RESULTS}.md`, `../gpu/RESULTS.md`, `../moondream3-eval/NOTES.md`.
