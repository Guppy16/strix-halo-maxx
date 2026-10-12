# TAPNext++ on the XDNA2 NPU

TAPNext++ point tracking with the whole step (12 blocks of SSM + ViT + attention, 48 query slots) in one fused NPU
dispatch per frame. The output is bit-identical to the round-9 reference build `full12r9a`. Work is paused after round 25.

## Status (best build full12r24z; full12r25z rebuilt it byte-identically)

| depth | step ms (sweep median) | DAVIS AJ, 10 videos |
|---|---|---|
| 12 | 28.42 | 70.84 |
| 10 | 23.83 | 69.90 |
| 8 | 19.09 | not run |
| 6 | 14.40 | not run |

Reference: CPU bf16 AJ 70.98 on the same 10 videos (70.54 is the GPU emulation of a bf16 trunk, RESULTS 3). Conditions: Turbo, NPU clock 1.808 GHz, gpu_busy 0.

## Rebuild and verify
From `tapnext/npu` (the script `cd`s into the working copy of this folder under `$SHM_SCRATCH`, a symlink to it).
It takes ~10 min of CPU and builds in Docker only.

    r24/try24.sh NAME [--sweep]

- `--sweep` also builds depths 10 / 8 / 6 and times them. `NOREG=1` / `NOTIME=1` skip the regress / timing.
- Every switch of the best build is listed in HANDOFF.md (round-25 block). Each round's kept change can be turned off by
  an env switch (e.g. `STATS64=0`, `OPPRE2=0`), see the try2N.sh headers.
- The chain is r24/try24.sh -> r23 -> r22 -> r21 -> r20 -> r18 -> r17/xtry.sh -> common/tools/try.sh, then
  r20/csplit20.sh patches the lowered MLIR and rebuilds the xclbin. Never dispatch `build/*.unpatched.xclbin`.

Verify:
- `r16/nrun.sh NAME.reg python3 step8.py regress --build full12NAME --depth 12 --ref full12r9a` must print BIT-IDENTICAL.
- `r24/lint.sh full12NAME` must report 0 asmlint errors.
- `r16/nrun.sh NAME.ab python3 r18/ab.py --a full12r24z --b full12NAME` gives the A/B in one process (p10 / median / mean).
- DAVIS: `TMO=3000 ./s8.sh LOG davis --build full12NAME --depth 12 --videos 10`.

## Docs
- `npu/HANDOFF.md`: the state at the pause, newest round block first (switches, risks, next steps).
- `npu/NOTES.md`: the lab log per round. `npu/RESULTS.md`: the measured tables per round.
- `research/ANALYSIS.md`: where the step's time goes (the time ledger). `research/PROPOSALS.md`: the ranked queue of
  next experiments. `research/RESULTS_FROM_MAIN.md`: measured verdicts on the research proposals.
- `../common/MACHINE.md`: the hardware model. `../common/book/negative-results.md`: refuted ideas.
- `book/`: the TAPNext chapters ([book/README.md](book/README.md)); the machine-level book is `../common/book/`.

## Progress (depth-12 step, best build at the end of each round)

| round | build | ms |
|---|---|---|
| 12 | full12r12e | 50.05 |
| 13 | full12r13f | 49.21 |
| 14 | full12r14f | 47.31 |
| 15 | full12r15m | 41.52 |
| 16 | full12r16z | 39.94 |
| 17 | full12r17d | 39.16 |
| 18 | full12r18m | 35.20 |
| 19 | full12r19z | 33.88 |
| 20 | full12r20z | ~31.8 (quiet-box estimate; -2.07 vs r19z in one process) |
| 21 | full12r21k1 | 31.30 |
| 22 | full12r22z | 30.78 |
| 23 | full12r23z | 28.89 |
| 24 | full12r24z | 28.43 |
| 25 | full12r25z (= r24z bytes) | 28.42 |

## Crop study
Tracking a point group from a moving crop, with the recurrent state shifted along with the crop, is effectively lossless
at W = 192 / 160 and costs ~1 AJ at W = 128 (CPU, 10 videos). Without the state shift AJ collapses. The NPU speed-up is
estimated, not built. See [crop-study/README.md](crop-study/README.md).
