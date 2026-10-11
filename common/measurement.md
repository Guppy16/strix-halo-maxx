# Measuring on testbox

These are observations from the TAPNext and moondream rounds, with advice for reading timings on this shared box. They
describe our measurement conditions; a production load can differ. Section numbers such as R21.2 point into
[MACHINE.md](MACHINE.md).

## Compare builds in one process
- Use `tapnext/npu/r18/ab.py --a BUILD_A --b BUILD_B [--rounds 10] [--block 22] [--skip 2]`, run through
  `r16/nrun.sh`. It alternates blocks of dispatches of the two builds in one process, so foreign CPU load and drift hit
  both alike. The first dispatches of each block are skipped, because a hardware-context switch precedes them.
- It reports **p10 / median / mean** per build and the deltas b - a. Quote all three: a change is real when they agree
  in sign and size. Single medians of 60 dispatches under foreign load were bimodal (negative-results H20).
- Absolute numbers come from `step8.py time` / `sweep` (median and min of many dispatches after 20 warm-up). The
  sweep times every depth from one xclbin.
- Host-polled timelines quantise at the poll interval (~16.5 us), so per-op medians snap to multiples of it (H16).

## Check the box before timing
- Load: `uptime`. Note the foreign workers (the background job, CP-SAT jobs) in the result.
- NPU clock: `python3 r7/clk.py run --out build/clk` from tapnext/npu in `npu-iron` with the device (build it once with
  `clk.py build`). It fits host time against trip counts of a pure-register loop. Quiet box: 1.81 GHz (MACHINE §1).
- GPU: `cat /sys/class/drm/card1/device/gpu_busy_percent`. Record it; GPU load shares DDR with the NPU.
- Power mode: Turbo for every TAPNext number. moondream phase 2 ran in Default and saw ~1.4 GHz.

## CPU load and the NPU clock
Observed (R20.1, R21.2; probes `tapnext/research/r23/clkload.sh`, `r25/clkload2.sh`, 12 threads per kind, NPU clock
by r7/clk.py):

| CPU load | package power | NPU clock (GHz) |
|---|---|---|
| idle | 27 W | 1.809 |
| 12 scalar spin threads | ~99 W | 1.810 |
| 6 numpy matmul threads | 96-100 W | 1.809 |
| 12 numpy 512 x 512 fp32 matmul threads (AVX-512) | ~99 W | 1.408 |
| the same with AVX2 (OPENBLAS_CORETYPE=Haswell) | 99 W | 1.408 |
| 12 threads of 64 x 64 matmul in L1 | 99 W | 1.584 |
| 12 threads of np.copyto on 64 MB arrays | 42 W | 1.808 |

- The drop appears with many threads of floating-point compute while the package sits at its ~100 W limit. DRAM copies
  and scalar spin at the same power do not cause it, and vector width does not matter.
- It was not thermal (Tctl 59-71 C). The mechanism (SMU power or current arbitration) is unverified.
- With BLAS threads at 6 or fewer, the clock stayed at 1.81 GHz. While timing, set `OPENBLAS_NUM_THREADS` (and torch
  threads) to 6 or fewer, or check the clock in the same session.
- GPU load as a trigger is untested.

## DDR bandwidth seen from the NPU
- Aggregate read: ~55-59 GB/s with 8 or more shim -> core streams. One stream alone does ~7 B/cycle (~12.6 GB/s)
  (DDRCAP, `tapnext/research/RESULTS_FROM_MAIN.md`; MACHINE R20.2, R21.1).
- Aggregate write: ~50-60 GB/s with 8 columns draining, ~12.1 GB/s for one column (DDRW). Reads plus writes together:
  63-74 GB/s.
- An earlier "at least 84 GB/s" reading was not reproduced (negative-results C12). These were measured with the GPU
  idle. A span reading at ~95 % of the cap is not necessarily capped (M1): measure before you price it.

## Timeouts and recovery
- Dispatch every new or changed design once, with a short timeout (`run.wait(3_000_000)` us), before timing it.
- After a timeout: `journalctl -k | grep -i xdna`, then run a known-good design (bit-exact) to confirm the device
  recovered. All timeouts so far recovered this way.
- At most 3 timeouts per phase. Stop if the device does not recover. Never loop retries on a failing design. Report the
  timeout count with the round.
- Common hang causes: lock values and conditions (keep conditions 0 / 1), immediates that wrap silently, stale record
  headers, and dispatching an `*.unpatched.xclbin`. SKILL §3 and negative-results G12, G16 list them.

## Pre-register predictions
- Before a build or probe, write the predicted effect as a range, with the hypothesis it tests, in the notes or
  PROPOSALS. Then score predicted vs measured in the commit message.
- Hits found after the measurement do not count as hits (H21). A single number that matches a bound cannot separate two
  hypotheses (H2). Price the whole call, not the loop (H8), and against the current build, not an older round (H6).
