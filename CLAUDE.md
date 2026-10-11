# strix-halo-maxx — agent entry point

This repo holds the NPU and GPU optimisation work for AMD Strix Halo on `testbox` (`ssh testbox`, repo at
`~/git/app/strix-halo-maxx`). Start with `README.md` for the map. For a project, read its README, then:
- TAPNext: `tapnext/npu/HANDOFF.md`, top block first.
- moondream: `moondream/README.md`.

Project knowledge is in the book: `common/book/README.md`, then `tapnext/book/` or `moondream/book/`. Skills (loaded from
this repo): `book-read` to find things and check past attempts, `book-write` to record results without letting the book rot,
`xdna2-npu` for NPU hardware / toolchain rules (read it before any NPU work). The hardware model is `common/MACHINE.md`;
timing advice is `common/book/measurement.md`.

## Safety (the rules that matter)
- NPU work runs in Docker only (`npu-iron`, `npu-iron-torch`). Mount `-v $HOME/git/app/.scratch:/work` together with
  `-v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx`. Check `docker ps` first. Leave no container running.
- New or changed kernels: one dispatch with a short timeout first. After a timeout, check `journalctl -k | grep -i xdna`
  and run a known-good design. At most 3 timeouts per phase; stop if the device does not recover.
- GPU: only when allowed, one job at a time, through `~/git/app/.scratch/moondream-opt/gpu_safe.sh`
  (snapshot in `moondream/gpu/gpu_safe.sh`).
- testbox is shared (another service, LLM server, llama servers, the live app server). Ask before installing anything,
  starting services or binding ports. No sudo, no systemd.
- Never `kill -9`. Never touch other tenants' processes or containers. Use the bracket trick with `pgrep -f`.
- Nothing in `/tmp`: temporary files go in `~/git/app/.scratch/` or a gitignored folder here. Never read `.env`.
- Compute (generators, CP-SAT, sims) runs on testbox, niced (`nice -n 10`).

## Verify a TAPNext build (from `tapnext/npu`, i.e. `~/git/app/.scratch/npu-tapnext`)
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
