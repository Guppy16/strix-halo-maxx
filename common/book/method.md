# Research method

The TAPNext port ran as two alternating workstreams. The **optimisation rounds** ("main") built, verified and timed one change at a
time. The **analysis passes** ("research") read the measurements, kept the hardware model and the time ledger, and proposed a ranked,
priced queue of experiments with pre-registered predictions. It worked: rounds 3-25 took the step from 269 ms to 28.42 ms,
bit-identical to the round-9 reference from round 9 on. The analysis side's mispredictions were the expensive part. Many of
the 187 negative results are refuted prices, not refuted mechanisms, and the method entries ([H1-H29](negative-results.md#h1), M1, L3)
record how each misprediction happened. This chapter turns them into rules.

## The two-workstream loop

| | optimisation rounds (`tapnext/npu/`) | analysis passes (`tapnext/research/`) |
|---|---|---|
| owns | the builds, the device, NOTES / RESULTS / HANDOFF | ANALYSIS (time ledger), PROPOSALS (ranked queue), MACHINE (hardware model) |
| per round | builds the queue's top items, one commit per experiment, kept or failed, with predicted vs measured | re-reads the measurements, scores the last predictions, re-prices the queue, proposes probes |
| hands back | RESULTS_FROM_MAIN.md: a verdict per proposal id | a new revision of ANALYSIS / PROPOSALS / MACHINE (git history keeps every revision) |

What made it work:
- **Stable ids.** Every proposal and probe has an id (P60g, ROT, SSMCUT). The commit, the verdict and the negative-results entry all use it.
- **Separate probes from levers.** A probe answers one question and is designed to separate two hypotheses (CPST, DDRCAP, SSMCUT, RRTWIN). A
  lever is priced only from probe-backed facts. Probes were cheap, and a few of them moved the queue more than any lever did. DDRCAP
  killed four levers and re-ranked the norms in one run.
- **Scheduling.** One GPU or NPU user at a time. The analysis passes' CP-SAT runs pause during the optimisation rounds' timing windows
  (`tapnext/research/r26/cpgate.sh`), because CPU load drops the NPU clock ([measurement.md](measurement.md#clock-and-cpu-load)).
- **The analysis passes build kernels too.** ROT, E7'' and KTL were generated and replay-verified in the analysis passes, then integrated in the optimisation rounds. The hand-off was
  a kernel `.s` + its header + a recipe, with a one-core bench (`r20/rotdb.py`, `r24/ktlpb.py`).

## Pricing

- **Price from the deployed build**, like for like, not from an older round's measurement ([H6](negative-results.md#h6) <!-- nr: H6 -->).
- **Price the whole call**: body + call/return delay slots + loop setup + prologue + drain + glue. Use `tcall`
  ([H8](negative-results.md#h8) <!-- nr: H8 -->, [H29](negative-results.md#h29)). Loop-only prices missed by 1.5x (P50: 87 → 136 cycles).
- **Every component measured, or flagged with a wide bar.** A guessed "half a stash copy" was 7x off ([H7](negative-results.md#h7) <!-- nr: H7 -->).
- **Known discounts**: k-step static cuts land at ~0.75 ([A18](negative-results.md#a18)). A compute cut lands only down to the op's input floor
  ([M2](negative-results.md#m2)). A rare per-unit cost under a multicast costs the max over destinations, not the mean
  ([H19](negative-results.md#h19)).
- **State which code path each part of a saving depends on**, and score a partial build part by part ([H27](negative-results.md#h27) <!-- nr: H27 -->).
- Give a range, the hypothesis it tests, and the in-between outcome. Score each prediction as hit or miss in the commit message.

## How mispredictions happened

Grouped by mechanism. Each one recurred until a rule or a tool stopped it.

| mechanism | examples | what fixed it |
|---|---|---|
| **wrong constants** compounding | the 1.43 GHz clock and the 4 B/cycle port made the A floor 2.5x too high ([H1](negative-results.md#h1) <!-- nr: H1 -->) | every constant carries a source tag; [DOC] beats an old [MEAS] of something else; `r7/clk.py` |
| **one number matching a bound** | t_ks 1.857 vs floor 1.813 "decided" the A port ([H2](negative-results.md#h2) <!-- nr: H2 -->) | report "consistent with"; run the perturbation (twin) first |
| **fitted free constants** absorbing unseparated time | sim9's t_ks, sim10's t_tile 7.6 µs ([H3](negative-results.md#h3) <!-- nr: H3 -->); EW fits with a nuisance parameter ([H13](negative-results.md#h13) <!-- nr: H13 -->); the CP boundary fit ([H14](negative-results.md#h14)) | compose from the binary; fix every term that has a stamp, fit at most one; stamp the hypothesised wait itself; name the separating probe before ranking |
| **means instead of distributions** | k-step means with the op-start wait ([H4](negative-results.md#h4)); PROF walls for coupling ([H5](negative-results.md#h5) <!-- nr: H5 -->); a CPU lazy-max map too hot by 1/0.6 ([H18](negative-results.md#h18)) | per-tile medians; full-step timelines; one fitted constant (κ) with per-unit one-core costs |
| **near a cap read as on the cap** | the SSM at 95 % of the read cap priced as capped ([M1](negative-results.md#m1) <!-- nr: M1 -->); a window average ([L3](negative-results.md#l3)) | removal twin + input-wait stamps; the binding op's own span |
| **benches that are not the deployed code** | the forced-II GELU, C++ serial in situ, pre-released locks ([H9-H12](negative-results.md#h9)) | bench the deployed binary, feed rings by DMA, guard zones |
| **twins priced from the step** | the norm twin −1.98 vs −0.2 real ([H17](negative-results.md#h17)) | per-op walls; twins only where downstream work is data-independent |
| **post-hoc hits** | ranges written after the measurement ([H21](negative-results.md#h21) <!-- nr: H21 -->) | not counted |
| **false nulls** | a stale jit cache, an `#undef`-ed flag: same elf ([H22](negative-results.md#h22)) | new build name; check the elf changed |

The pattern behind most rows: a model that reproduced the past was used to price the future before a probe had separated its
terms. The cure was to name the probe that would distinguish the hypotheses, run it, and only then rank levers. CPST did this for
the CP-bound boundary model, DDRCAP for "84 GB/s", and SSMCUT for "the SSM is read-capped".

## Keeping the record

- One commit per experiment, kept or failed, with the measured result and predicted vs measured in the message
  (`feat(npu-tapnext): ROT ... -3.3 ms (round 18)`). Commit dead experiments too, so their order can be read later
  (OPPRE's patch was not committed, and its cause could not be re-examined; [C11](negative-results.md#c11)).
- Every kept change sits behind an env or `-D` switch. The best build is a list of switches (HANDOFF round 25), and any change can be
  turned off for an A/B.
- Write every failure into the negative-results index with its class and reason, and into the chapter it belongs to
  ([book-write skill](../../.claude/skills/book-write/SKILL.md)). Several "dead" ideas were revived because the entry said *why*
  they died, and the reason later changed (C8, B2, E15, D13).

## What didn't work

The method entries: [H1](negative-results.md#h1)-[H29](negative-results.md#h29), [L3](negative-results.md#l3), [M1](negative-results.md#m1).
The measurement-specific ones (H4, H9-H12, H16, H17, H20, H22, H26, H28) are explained in [measurement.md](measurement.md). Also:
- [H15](negative-results.md#h15) <!-- nr: H15 --> "5 ms of boundaries" counted first-tile compute as boundary.
- [H18](negative-results.md#h18) <!-- nr: H18 --> the CPU lazy-max map overstated in-situ flags; "lockstep × 6" came from compensating fits.

## Lessons

- Twin first, then fit at most one constant, then price from the deployed binary's whole call, with known discounts applied.
- Write the prediction before the build, including what result would mean "neither hypothesis".
- Separate measured from extrapolated in every number you write ([MEASURED], [MODEL], [EXTRAPOLATED]).

## Going deeper

- `tapnext/research/RESULTS_FROM_MAIN.md` (verdicts per proposal, rounds 6-25), ANALYSIS / PROPOSALS history
  (`git log --follow -p tapnext/research/PROPOSALS.md`; older revisions `git show eed3870:npu-research/revN/<file>`).
- [measurement.md](measurement.md) for the timing rules, [negative-results.md](negative-results.md) for the full list.
