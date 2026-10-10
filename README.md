# tapnext-xdna2

TAPNext++ point tracking on the AMD XDNA2 NPU (Strix Halo, AIE2P): one fused dispatch per frame, bit-exact against the
round-9 reference, plus the research model of the hardware behind it.

- npu-tapnext/: the implementation (design8.py, plan8.py, mm8.cc, step8.py), per-round experiments (rN/), and
  HANDOFF.md / NOTES.md / RESULTS.md.
- npu-research/: offline analysis (MACHINE.md, ANALYSIS.md, PROPOSALS.md, RESULTS_FROM_MAIN.md), simulators and tools/.
- skill/SKILL.md: snapshot of the hardware/toolchain rules (live copy: ~/.claude/skills/xdna2-npu on the laptop).

One commit per round: the message carries the measured result and predicted-vs-measured verdicts.
Working copy, not yet restructured for publication.
