# strix-halo-maxx

Hand-optimised model inference on AMD Strix Halo (Ryzen AI Max+ 395): the XDNA2 NPU and the gfx1151 GPU.

## The hardware

The Ryzen AI Max+ 395 puts Zen 5 CPU cores, a gfx1151 iGPU and an XDNA2 NPU on one die. All three share one pool of
unified LPDDR5X memory. The NPU is an AIE2P array: 8 columns x 4 rows of compute tiles (32 VLIW vector cores, each with
64 KB of data memory and 16 KB of program memory), one memtile per column, and a shim row that talks to DDR. Its clock
was measured at 1.81 GHz. The measured performance model is [common/MACHINE.md](common/MACHINE.md).

## Projects

| project | headline | start here |
|---|---|---|
| TAPNext++ point tracker on the NPU | **28.43 ms per step at depth 12**, one fused dispatch per frame; DAVIS AJ 70.84; bit-identical to the round-9 reference | [tapnext/README.md](tapnext/README.md) |
| moondream2 on the GPU | 88.7 ms per frame (stock HF bf16: ~340 ms), 98.0 % accuracy on the 493-frame yes/no set | [moondream/README.md](moondream/README.md) |
| moondream2 vision tower on the NPU | 27 blocks in one NPU program; tower ~75 ms per frame on the NPU alone (phase 12); 27 fps with the GPU pipelined | [moondream/README.md](moondream/README.md) |
| moondream3-preview evaluation | not better at grounding than moondream2, ~2.6x slower | [moondream/README.md](moondream/README.md) |

## Where things live

- `CLAUDE.md`: the short entry point for an agent (safety rules, how to verify, commit conventions).
- `common/book/`: **the book**, the machine-level manual (hardware, toolchain, measurement, GEMM, data movement, fusion,
  elementwise, attention, method) and `negative-results.md`, the classified index of refuted ideas.
  `tapnext/book/` and `moondream/book/` hold the model chapters.
- `.claude/skills/`: `book-read` (find things in the book), `book-write` (add to it), `xdna2-npu` (NPU hardware and
  toolchain rules, safety; this repo copy is canonical).
- `common/`: `MACHINE.md` (the hardware model with every fact tagged), `tools/` (asmlint, tcall, the simulators, try.sh, the
  timing table `aie2p_isa_timing.json`).
- `tapnext/`, `moondream/`: one folder per model; the lab notebook (NOTES, RESULTS, HANDOFF, rNN/) lives there.

Measured symptoms to keep in mind when reading a timing (observations under our load, not rules): heavy floating-point CPU
load lowered the NPU clock (1.81 -> 1.41 GHz), and the NPU's DDR reads and writes each capped at ~55-59 GB/s aggregate with the
GPU idle. See [common/book/measurement.md](common/book/measurement.md).

Only hand-written code, generators, docs and small result tables are tracked: builds, weights, venvs and logs stay out.
On the test machine, `$SHM_SCRATCH/npu-{tapnext,research,moondream}` are symlinks to `tapnext/npu`, `tapnext/research` and
`moondream/npu`. Containers that mount .scratch at /work also mount this repo at its own path, so the links resolve.

## Reading the book

The Markdown here is also built as a MkDocs Material site (`mkdocs.yml`, links to code handled by `tools/mkdocs_hooks.py`).
Preview locally with `uvx --with mkdocs-material --with mkdocs-same-dir mkdocs serve`. Once the repo is pushed to GitHub
and Pages is set to "GitHub Actions", `.github/workflows/pages.yml` publishes it on every push to `main`.
