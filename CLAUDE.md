# strix-halo-maxx — agent entry point

This repo holds the NPU and GPU optimisation work for AMD Strix Halo. Paths below are relative to the repo root;
`$SHM_SCRATCH` is the scratch/working directory (default: `<repo root>/../.scratch`). Start with `README.md` for the map. For a project, read its README, then:
- TAPNext: `tapnext/npu/HANDOFF.md`, top block first.
- moondream: `moondream/README.md`.

Project knowledge is in the book: `common/book/README.md`, then `tapnext/book/` or `moondream/book/`. Skills (loaded from
this repo): `book-read` to find things and check past attempts, `book-write` to record results without letting the book rot,
`xdna2-npu` for NPU hardware / toolchain rules (read it before any NPU work). The hardware model is `common/MACHINE.md`;
timing advice is `common/book/measurement.md`.

## Safety (the rules that matter)
- NPU work runs in Docker only (`npu-iron`, `npu-iron-torch`). Mount `-v $SHM_SCRATCH:/work` together with
  `-v <repo root>:<repo root>` (the same path inside, so symlinks resolve). Check `docker ps` first. Leave no container running.
- New or changed kernels: one dispatch with a short timeout first. After a timeout, check `journalctl -k | grep -i xdna`
  and run a known-good design. At most 3 timeouts per phase; stop if the device does not recover.
- GPU: one job at a time, through `moondream/gpu/gpu_safe.sh` (it checks GPU memory before and during the run).
  Never `kill -9` a GPU process.
- The test machine may be shared with other users and workloads. Ask before installing anything, starting services,
  binding ports or using the GPU. No sudo, no systemd.
- Never touch other users' processes or containers. Match processes with the bracket trick (`[p]attern`) so a
  command never matches itself.
- Nothing in `/tmp`: temporary files go in `$SHM_SCRATCH` or a gitignored folder here. Never read `.env` files.
- Compute (generators, CP-SAT, sims) runs niced (`nice -n 10`) on the test machine.

## Verify a TAPNext build (from `tapnext/npu`, via its working copy `$SHM_SCRATCH/npu-tapnext`)
- Anything that runs the model needs `TAPNEXT_SRC` (the TAPNext++ torch `model.py`) and `TAPNEXT_WEIGHTS`
  (`tapnextpp_256.safetensors`); the scripts mount them read-only into the container.
- Numerics: `r16/nrun.sh NAME.reg python3 step8.py regress --build full12NAME --depth 12 --ref full12r9a` must print
  BIT-IDENTICAL.
- Asm hazards: `r24/lint.sh full12NAME` must report 0 errors.
- Speed: `r16/nrun.sh NAME.ab python3 r18/ab.py --a full12r24z --b full12NAME` (A/B in one process; see
  `common/book/measurement.md`).

## Before trying an idea
Load `book-read`. Search `common/book/negative-results.md` (classified index) and `tapnext/research/PROPOSALS.md`. Many ideas
were already refuted; the class (fundamental / measured-here / unfinished / revived / method) says how sure that is.
After a round, record what changed with `book-write`.

## Commits
Commit locally and never push. Make one commit per experiment, kept or failed. The message carries the measured result
and the predicted vs measured verdict, e.g. `feat(tapnext): X - -0.12 ms (round 26)`. Use `type: summary` style.
