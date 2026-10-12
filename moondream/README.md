# moondream on Strix Halo

moondream2 for object-presence (yes/no) and grounding queries on camera frames: low-latency VLM inference for local use. The hardware can be a strong GPU (gfx1151 here) or the XDNA2 NPU, and this folder covers both. The work predates the
TAPNext port and is not active now.

| folder | what | headline |
|---|---|---|
| [gpu/](gpu/README.md) | moondream2 latency on the GPU: fast path, Triton and hand-written HIP WMMA GEMMs, int8 text model | **88.7 ms per frame** (encode 79.6 + query 9.0) vs ~340 ms stock HF bf16 (3.8x), 98.0 % accuracy on the 493-frame set; ~65 ms with image tokens pooled to 16 x 16 |
| [npu/](npu/RESULTS.md) | the moondream2 vision tower on the NPU, phases 1-12 (GEMMs, bfp16, LN, GELU, attention, the asm route) | the 27-block GEMMs in one NPU program: 33.8 ms per frame; tower budget ~75 ms per frame on the NPU alone (phase 12); pipelined with the GPU text side: **27 fps** |
| [moondream3-eval/](moondream3-eval/NOTES.md) | moondream3-preview vs moondream2 on the grounding set | not better (44 correct by eye for both), ~2.6x slower (1.2-1.3 s vs 0.46 s per detect), ~4.5x the GPU memory |

- `npu/NOTES.md` is the lab log per phase and `npu/RESULTS.md` the measured tables. Phase 10 started the asm route and
  the instruction timing table that now lives in `../common/tools/`.
- `gpu/RESULTS.md` holds the full ladder and the discarded avenues. `gpu/gpu_safe.sh` is the one-GPU-job wrapper.
- `gpu/` and `moondream3-eval/` are snapshots; see their SOURCES.md for where the live copies are.
- `book/`: the moondream chapters ([book/README.md](book/README.md)); the machine-level book is `../common/book/`.
