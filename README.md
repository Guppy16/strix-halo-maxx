# strix-halo-maxx

Optimising models on AMD Strix Halo (Ryzen AI Max): the XDNA2 NPU (AIE2P) and the gfx1151 GPU.

- tapnext/npu/: TAPNext++ point tracking on the NPU, one fused dispatch per frame, bit-exact against the round-9
  reference: design8.py, plan8.py, mm8.cc, step8.py, per-round experiments (rN/), HANDOFF.md / NOTES.md / RESULTS.md.
- tapnext/research/: offline analysis of the NPU (MACHINE.md, ANALYSIS.md, PROPOSALS.md, RESULTS_FROM_MAIN.md), simulators,
  asmlint and try.sh in tools/.
- moondream/npu/: moondream on the NPU (phases 1-12), NOTES.md / RESULTS.md.
- moondream-gpu/, moondream3-eval/: snapshots of the GPU moondream work and the moondream3 evaluation (see SOURCES.md).
- .claude/skills/xdna2-npu/SKILL.md: snapshot of the hardware/toolchain rules (live copy: ~/.claude/skills/xdna2-npu on the laptop).

Only hand-written code, generators, docs and small result tables are tracked: builds, weights, venvs and logs stay out.
One commit per experiment or round: the message carries the measured result and predicted-vs-measured verdicts.
On testbox, ~/git/app/.scratch/npu-{tapnext,research,moondream} are symlinks into this repo, and containers that mount
.scratch at /work also mount this repo at its own path so the symlinks resolve.
