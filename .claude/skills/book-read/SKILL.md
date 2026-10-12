---
name: book-read
description: Find project knowledge fast in the strix-halo-maxx repo (Strix Halo NPU / GPU optimisation, TAPNext and moondream). Load before proposing, pricing or debugging any NPU or GPU optimisation, before answering "was X tried?" or "why is X slow?", and before reading NOTES / RESULTS / HANDOFF / ANALYSIS / MACHINE. Covers the reading order, checking the negative-results index before trying an idea, when to drop to the raw rNN/ evidence, and how to cite.
---

# Reading the book

The book is the manual; the rNN/ folders, NOTES, RESULTS, HANDOFF and the commit log are the lab notebook. Start in the
book and drop to the notebook only for raw evidence.

## Reading order
1. `README.md` (root): the projects, headline numbers, where things live.
2. `common/book/README.md`: the machine-level table of contents. Read the chapter for your topic (machine, toolchain,
   measurement, gemm, data-movement, fusion, elementwise, attention, method). Each is a 5-10 minute read.
3. The model's book, `tapnext/book/` or `moondream/book/`, only if the shapes, layouts or past attempts of that model
   matter. Common chapters link down; model chapters link back up.
4. For hands-on NPU work also load the `xdna2-npu` skill (hardware and toolchain rules, safety).

## Before trying an idea
- Search `common/book/negative-results.md` (by id, name, or keyword) and the chapter's `## What didn't work`.
  Also check `tapnext/research/PROPOSALS.md` (the queue at the pause).
- Read the class before you decide:
  - **fundamental**: a named hardware or arithmetic limit. Don't retry unless you can show that limit does not apply.
  - **measured-here**: dead in that design or layout by measurement. Retry only if what made it lose has changed (another model,
    layout or bottleneck), and say what changed.
  - **unfinished**: abandoned for time, an unchased bug or an untested hypothesis. It's a candidate: read why it stopped.
  - **revived**: once called dead, later made to work. Cite the working form, not the old verdict.
  - **method**: a measurement or pricing mistake. Apply its lesson to your own plan.
- If an entry has no class or its reason no longer holds, say so in your report rather than assuming.

## When to drop to raw evidence
- A number you will build on or quote: open the source the chapter cites (MACHINE M-id, NOTES / RESULTS section, commit).
- What exactly was built: `git show <hash>`, the round's `tapnext/npu/rNN/` scripts, and the `try2N.sh` switch lists.
- Why a proposal was ranked or priced: `tapnext/research/ANALYSIS.md`, `PROPOSALS.md`, and their history
  (`git log --follow -p <file>`; old revisions `git show eed3870:npu-research/revN/<file>`).
- Measured verdicts per proposal id: `tapnext/research/RESULTS_FROM_MAIN.md`.
- The state at the pause and the best build's switches: `tapnext/npu/HANDOFF.md`, top block.
- These files are big (NOTES, RESULTS, MACHINE run to thousands of lines). Grep for the id or the M-id; don't load them whole.
- Where sources disagree, the newer measurement wins. Check the negative-results "sources disagree" notes and the MACHINE tags
  ([DOC] / [MEAS] / [INF]; superseded values appear only as Pitfall lines).

## How to cite
- Every claim gets a number and its evidence: `common/MACHINE.md` M1.4, `tapnext/npu/NOTES.md` 25.2, commit `516d3a9`,
  `tapnext/npu/r19/ddrcap.py`, negative-results `C13`.
- Keep the status the source gives: MEASURED, MODEL or EXTRAPOLATED. Quote A/B deltas as p10 / median / mean.
- Give the conditions with absolute timings: build, depth, NPU clock (1.81 GHz), gpu_busy, power mode.
- The DDR caps and the CPU-load clock drop are observations under our conditions, not rules.
