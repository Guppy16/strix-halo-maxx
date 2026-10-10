# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 2, 2026-10-09, after round 6 (`RESULTS_FROM_MAIN.md`). Revision 1 is in `rev1/PROPOSALS.md`, which keeps the
full text of P1-P14. Savings marked [sim9] come from `tools/sim9.py` (ANALYSIS §4): depth 12, each change alone,
against the r6e fit (76.3 ms). Every entry gives the hypothesis, the procedure, the expected outcome, the cost
(S / M / L), the device risk, and a **Result** field for the main agent.

Common rules (SKILL §3, §9):
- one dispatch with a short timeout first for every new design, and the known-good test after any timeout;
- record Turbo / Default and gpu_busy with every number;
- the clock is 1.8 GHz (MACHINE §1): convert cycles with it, and say so.

## Ranking (revision 2)

| rank | id | what | cost | risk | expected [sim9 unless noted] |
|---|---|---|---|---|---|
| 1 | **P15** | core timer that does not hang: processor-bus enable + scheduling barriers | S | low-medium | unblocks P16 and P2's stamps; measures the clock |
| 2 | **P18** | hand-asm GEMM loop: 2x2 register blocking, A popped on lda and B on ldb | L | medium | t_ks 1.854 -> 0.95 us: **-14 ms** (pre-registered prediction) |
| 3 | **P17** | L1 bank placement of the GEMM core (B fifo, A ring, accumulators apart) | S | low | part of the 12.6% k-step stalls: up to -3.6 ms |
| 4 | **P16** | performance-counter profiler: busy / lock / stream / memory-stall cycles per core per op | M | low (after P15) | attribution for P13, P12, P17, P18 |
| 5 | **P13** | SSM (1.30 ms per block on the EW cores) | M | low | -> 0.5 ms: **-9.6 ms** |
| 6 | P12 | attention (1.20 ms per block; main agent's current item) | M | low | -> 0.6 ms: -7.2 ms |
| 7 | P7 | fold the norms into the GEMMs | L | low-medium | -4.5 ms, removes 48 `finish_all` brackets |
| 8 | P19 | attention on all 32 cores (the EW cores idle during it) | M | medium | up to -3.6 ms |
| 9 | P11 | cheaper tile boundaries (options 1-2 left) | M | low | t_tile halved: -3.2 ms |
| 10 | P5 | k-chunk-streamed A slot | M | medium | -2.4 ms, -144 MB DDR |
| 11 | P14 | two frames in flight | L | medium | fps x1.3 at ~2x latency (today 13 -> 17 fps at depth 12) |
| 12 | P20 | one call per k-step, then k = 128 per call | M | medium | ~-2 ms, then the vec slot binds (after P18) |
| - | P3, P4, P9, P10 | tooling (txnlint, event simulator, objdump auditor, port microbenchmarks) | S-M | none-low | see below |

All of P7, P11-P13 and P5 plus P18 together: **35.2 ms serial at depth 12** (29.3 at depth 10). With two frames in
flight, 27.8 ms per frame = 36 fps at 67 ms latency (ANALYSIS §4).

## Status of P1-P14 (from RESULTS_FROM_MAIN.md)

| id | status | verdict / what is left |
|---|---|---|
| P1 twin builds | **DONE** | Compute-bound, refuting revision 1's A-port theory. sim9 refit: no-MAC t_ks 0.753 us (data movement, >= 7.6 B/cycle per core), 2x-MAC +1.544 us against 2,718 static bundles, so the clock is >= 1.76 GHz. Follow-ups: P18, P17 |
| P2 in-kernel cycle stamps | **BLOCKED -> cause found** | The hang is the disabled processor bus (Core_Processor_Bus 0x32038, Enable resets to 0), plus a bundle-packing hazard on back-to-back tm accesses. Retry as P15 |
| P3 txnlint | open | Unchanged. Still the cheapest race insurance; feeds P4 and the two-frame sequence (P14) |
| P4 event-driven simulator | open, partly superseded | sim9 adds a segment fit (handles invisible ops) and a two-resource list scheduler for overlap. The CP / queue / DMA-level simulator is still unbuilt; build it before P14 |
| P5 k-chunk A slot | open, value lowered | L_dep fitted at 34 us (r6e) vs 40: -2.4 ms (was -3.0) |
| P6 two A streams | **REFUTED** | The memtile -> core port is not binding (>= 7.6 B/cycle available, 3.1 used). Revisit only if P18 brings t_ks to ~0.75 us |
| P7 fold the norms | open | Unchanged: -4.5 ms. Norm cost is the shim channel + ring protocol, not a core port |
| P8 numerics harness | **DONE** (parts 1-3) | Found the erf-vs-tanh GELU and the input ranges. Parts 4-5 (bfp damage, depth error growth) open |
| P9 object auditor | **PARTLY** (r6/asmstat.py) | Left: objdump of the linked elfs (control code, program-memory headroom for P19 / P13 option 2), baseline diffs. Note: asmstat's loop table counts slot ops but does not say which slot or chain binds (see P18) |
| P10 port microbenchmarks | open, re-scoped | Now: confirm 1.8 GHz (with the main agent's r7/clk.cc) + `xrt-smi examine -r platform` while a context is open; the exact memtile -> core rate (8 B/cycle?) and broadcast to 1 vs 3 cores |
| P11 tile boundaries | **PARTLY** | The epilogue pass gave -5.3 ms. Options 1-2 (drain as two sub-objects, acc_zero folded into the first k-step) open: t_tile 5.77 us |
| P12 attention | in progress (main agent) | 1.20 ms per block. P16's counters would split exp / PV / QK^T / waiting on the device instead of statically |
| P13 SSM | open | 1.30 ms per block, the longest EW stretch, and the main bubble in every overlap schedule (ANALYSIS §4) |
| P14 two frames in flight | open, priced | sim9: today 58.2 ms per frame (17.2 fps) at 140 ms latency; after the fixes 27.8 ms (36 fps) at 67 ms. Build only after P4 |

---

## P15. A core timer that does not hang (processor-bus enable) — **S, risk: low-medium**

**Hypothesis.**
- `read_tm` reaches tile registers over the core's processor bus.
- The bus's Enable bit (Core_Processor_Bus, 0x32038 bit 0) resets to 0, and a core that touches the bus while it is
  disabled hangs.
- mlir-aie's `test/npu-xrt/tile_mapped_read/aie.mlir` says so verbatim, and its npu2 example
  `programming_examples/basic/dma_compression` sets the bit from the IRON runtime sequence before using
  `read_tm` / `write_tm`.
- A second, documented hazard: tm accesses packed back to back into one bundle stall the core (mlir-aie issue #2346).

**Procedure.**
1. kprobe with stamps, plus in the runtime sequence, first op:

   ```python
   npu_maskwrite32(address=0x32038, value=1, mask=1, column=0, row=2)   # aie.dialects.aiex, IRON 1.4.4
   ```

2. In the kernel, wrap each read in `__builtin_aie2p_sched_barrier()`. Read `0x340F8` (core Timer_Low); `0x140F8`
   (memory-module timer) is the fallback.
   - The first read must come after the core's first acquire of sequence-fed data, so the maskwrite has landed.
   - `llvm-objdump -d` the object: each `lda.tm` alone in its bundle.
3. Dispatch once with a 3 s timeout; known-good full12r6e after any timeout.
4. Stamp a pure-register loop of known bundle count (r7/clk.cc's chains): cycles per trip = bundles proves the
   timer runs at the core clock, and timer delta / host time gives the clock.

**Expected.** Stamps that read sane values, and 1.80 GHz.

**What it changes.**
- P2's stamp plan (RESULTS-free op timing inside the real step) becomes buildable.
- P16 needs it to read the counters.

**Risk.** If the bus enable is not the (only) cause, one more timeout. The device recovered from the last one. If it
hangs again, stop and use P16 option (b) (control packets) or the trace unit.

**Result:**

---

## P16. Performance-counter profiler: where the stalls are, per core and op — **M, risk: low (after P15)**

**Hypothesis.** Each core module has 4 hardware counters with event-controlled start / stop. They can count, per op
and core, at zero cost inside the kernels:
- cycles inside `mm_acc` (start INSTR_EVENT_0 = 33, stop INSTR_EVENT_1 = 34; the markers already exist);
- LOCK_STALL (26), STREAM_STALL (24) and MEMORY_STALL (23) cycles (start = stop = the event: count while asserted
  [INF, verify]).

That splits the 12.6% of the GEMM k-step beyond its static schedule, and the attention / SSM time into compute vs
waiting.

**Procedure.**
1. Configure from the runtime sequence (`npu_write32`):
   - `Performance_Control0` 0x31500 = (cnt1 stop << 24) | (cnt1 start << 16) | (cnt0 stop << 8) | cnt0 start;
   - `Control1` 0x31504 the same for cnt2 / cnt3;
   - optionally `Control2` 0x31508 reset events.
2. (a) Core-side readout: at the end of each op, the kernel reads `0x31520-0x3152C` with `read_tm` (P15's bus enable +
   barriers) into the P2 stamp slots (ANALYSIS rev 1 / PROPOSALS rev 1 P2 list the padding that can hold them).
   (b) Host-side readout without kernel changes: control-packet reads (IRON 1.4.4
   `aie.utils.trace.utils.create_ctrl_pkt`, `setup.configure_packet_ctrl_flow`) at the end of a one-column run.
3. Validate on one core first: cnt0 against timer stamps around the same call; the stall counters against a twin
   that adds a known stall (e.g. a lock wait).
4. `step8.py prof`: per op and core, busy / lock / stream / memory cycles; across columns for the spread.

**Expected.** GEMM: ~88% busy, the rest mostly MEMORY_STALL if P17's bank theory holds (STREAM_STALL if the A ring
runs dry at tile boundaries). Attention and SSM: the waiting share that P12 / P13 must design around.

**Result:**

---

## P17. L1 bank placement of the GEMM core — **S, risk: low**

**Hypothesis.**
- The deployed k-step runs 3,337 cycles against 2,963 static (12.6% stalls, sim9).
- In `ldScripts_main_core_0_2.ld.script` bank 3 (0x7C000-0x7FFFF) holds both B fifo buffers, two of the three A ring
  slots and the C object's tail.
- In the loop the core pops B from bank 3 every bundle while the DMA writes A and B into bank 3.
- Bank conflicts stall the core (MEMORY_STALL).

**Procedure.**
1. Read the allocation of every GEMM-core buffer from `input_with_addresses.mlir`.
2. Re-place:
   - B fifo in one bank;
   - the A ring in another;
   - the C object and `acc_hi` in the remaining two.
   - Use aiecc's bank-aware allocation, explicit `Buffer` placement or the linker's `.aie.bankN` sections. The data
     area holds the LUT tables and `acc_hi`; check the 64 KB budget.
3. Differential timeline vs full12r6e. `regress` must be BIT-IDENTICAL (placement only).
4. If P16 exists, compare MEMORY_STALL before and after.

**Expected.** t_ks 1.854 -> ~1.70 us if all stalls are bank conflicts: up to -3.6 ms [sim9: 12.6% of 28.8 ms]. No
change means the stalls are elsewhere (stream / lock): P16 decides.

**Result:**

---

## P18. Hand-asm GEMM loop: 2x2 register blocking, A on lda, B on ldb — **L, risk: medium**

**Hypothesis** (ANALYSIS §3, §5).
- The deployed `mm_acc` spends 321 bundles per 128 mmuls. Its 28-bundle loop is not ldb-bound (18 ldb ops); it is
  bound by two accumulator chains (2 macs per 3 bundles).
- 134 of the 321 bundles are per-row-block prologue and tail.
- AIE2P can pop block vectors on both load slots (`vlda.pop.576`, `vldb.pop.576`; FIFOs lf0 / lf1) and has 5
  accumulators.
- With 2x2 blocking, per pass of 32 mmuls:
  - lda carries 16 A pops + 2 fills + 16 accumulator quarter loads = 34;
  - ldb carries 18;
  - st 16, vec 32.

  Four chains at distance 4, and accumulator traffic overlapped across passes: ~160 bundles per call.

**Procedure.**
1. Extend `npu-moondream/p11/gen_gemm.py` to emit the 2x2 pass:
   - A on (p1, lf1) via `vlda.pop.576`, B on (p0, lf0) via `vldb.pop.576`;
   - `vmac.f dmD, dmS` with the next accumulator staged in dm4;
   - fills every 8 pops per FIFO;
   - one prologue and drain per call.
2. Check it before assembling:
   - `asmcheck.py` with the stage-3 rule;
   - assert which pop every mac reads;
   - count slots per pass.
3. Merge with `ld.lld -r` and link via `Kernel(...)`. Keep the `mm_acc` interface and IRON's per-call locks.
4. Correctness: t1 / g11 BIT-EXACT; then `regress` vs full12r6e (bit-identical expected: same MAC order per
   accumulator over k).
5. Then the timeline, and sim9's refit of t_ks.

**Pre-registered prediction** (sim9, before the build):
- t_ks = 0.95 +- 0.10 us;
- step 62.2 +- 2 ms at depth 12.

Outside that band, the model is wrong somewhere: then P16 before anything else.

**Cheaper first step.** Same data flow as today (A held in registers), but 3-4 accumulator chains via the staging
trick: the loop at the ldb bound (18 per 16 mmuls). Predicted t_ks 1.23 us, -9.6 ms.

**Risk.** Hand asm hung the core before only through raw `.byte` bundles (never do that). Lock / FIFO misuse shows as
wrong results first; one dispatch with a 3 s timeout.

**Result:**

---

## P19. Attention on all 32 cores — **M, risk: medium**

**Hypothesis.**
- Attention (1.20 ms per block, 14.4 ms per step) runs on the 24 GEMM cores while the 8 EW cores idle.
- A fourth core per column takes 25% of the (head, query-tile) units.

**Procedure.**
1. P9's objdump: the EW elf's free program memory. The attention code is ~3 KB (chunk_k 1.7 KB, chunk_v, rescale,
   begin/end).
2. If it fits, extend plan8's attention units to row 5. Its K/V inputs arrive like the GEMM cores' (broadcast from the
   memtile), and its O goes out on its own S2MM.
3. t4 directed test, then the step.

**Expected.** Attention x0.75: -3.6 ms today; -1.8 ms after P12.

**Result:**

---

## P20. One call per k-step, then k = 128 per call — **M, risk: medium (after P18)**

**Hypothesis.**
- Per k-step, 9 calls cost ~225 cycles of call and lock overhead (~7% of today's k-step, ~15% after P18).
- With k = 64 per call, every accumulator is loaded and stored every 8 macs. That makes accumulator traffic, not
  the MACs, the lda bound in P18.

**Procedure.**
1. One `mm_kstep` entry that loops over the 9 stripes and takes the A ring locks inside the kernel
   (`acquire_greater_equal` on id + 48, as ew8 does).
2. Then k = 128:
   - acquire both B objects of a k-step pair (the B fifo is 2 deep);
   - change the memtile A send pattern to [k-pair][stripe][2 k-steps] (a TAP change on the same slot).

**Expected.** ~-2 ms for step 1. After step 2, lda drops to ~26 per 32 mmuls and the vec slot binds: t_ks toward the
A-port floor (~0.75 us).

**Result:**

---

## Unchanged tooling proposals (full text in rev1/PROPOSALS.md)

- **P3 txnlint** (S, no device): races, BD reuse, queue head-of-line, bytes per op from the `.bin`. Run it in `bb.sh`.
  Needed before P14, whose sequence interleaves two frames.
- **P4 event-driven simulator** (M, no device): sim9 covers segment fits and two-resource overlap. The command
  processor / queue / DMA model is still the gate for P5 and P14.
- **P9 auditor, remainder** (S): objdump of the linked elfs (control code per op, program-memory headroom, which slot
  and which chain bounds each loop).
- **P10 port microbenchmarks, re-scoped** (S): the clock and the exact memtile -> core rate (ANALYSIS §0, MACHINE §9).
