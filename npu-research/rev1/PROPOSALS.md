# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Ranked by expected value per unit of cost. Each entry gives: the hypothesis, the procedure, the expected outcome and
what it changes, the cost (S / M / L), the device risk, and a **Result** field for the main agent to fill in.

Common rules (SKILL §9):
- one dispatch with a short timeout first for every new design;
- run the known-good test after any timeout;
- record Turbo / Default and gpu_busy with every number.

Savings marked [sim8] come from `tools/sim8.py` what-ifs (ANALYSIS §6): serial schedule, each change alone.

---

## P1. Is the GEMM k-step bound by the A port or by compute? (twin builds) — **S, risk: none**

**Hypothesis.**
- One k-step per GEMM core costs 1.857 us (fitted, ANALYSIS §2). The A-port floor is 1.813 us (10,368 B at 4 B/cycle).
- Compute plus per-call overhead could give the same number (9 calls x (204 + ~115) cycles).
- The fix differs:
  - port-bound: two A streams (P6) or a different tile shape;
  - compute-bound: one call per k-step or the asm loop.

**Procedure.**
1. Make two `-D` variants of mm8.cc's `mm_acc`:
   - `MM_NOCOMPUTE`: return right after `event0()`. Locks and DMA are untouched: IRON's control code still acquires
     and releases the A ring and the B fifo.
   - `MM_TWICE`: run the `rb` loop twice on the same A and B, i.e. 2x the MACs and the same bytes.
2. Build the g11-style GEMM-only test (11 ops x 12, real deps) with `-DNO_GELU`:
   - baseline;
   - NOCOMPUTE;
   - TWICE.
3. Time 20 iterations each with `r8.sh`. Also run `step8.py timeline` on the full step for baseline vs NOCOMPUTE
   (outputs are garbage in NOCOMPUTE; only the timings count).

**Expected and what changes.**

| outcome | conclusion | next |
|---|---|---|
| NOCOMPUTE within ~5% of baseline, and TWICE well below 2x | A-port bound | P6 (two A streams) becomes the top GEMM item; kernel tuning of the loop is worth ~0 |
| NOCOMPUTE far faster, and TWICE ~ 2x of the compute part | compute / overhead bound | build a "k-step call" (9 stripes per call, ring locks taken inside the kernel as ew8 does). [sim8] -0.7 ms if overhead-bound to the port floor, more if the port is not the limit |

Feed the measured t_ks back into sim8.

**Result:**

---

## P2. In-kernel cycle stamps ("probe words") for every core kind — **S-M, risk: low**

**Hypothesis.** The timeline sees only op boundaries in column 0. Per-core busy vs wait cycles inside
GEMM / attention / SSM / norm would decide P1, P12 and P13 directly. They can be collected at zero DMA cost, in bytes
that are already drained into padding.

**Procedure (spec).**
1. A compile flag `PROBE` in mm8.cc and ew8.cc. Each kernel keeps static counters from
   `aie::tile::current().cycles()`:
   - GEMM: cycles inside `mm_acc` (sum), calls, the wall from `acc_zero` to the end of `acc_epilogue`, cycles in the
     epilogue.
   - Attention: cycles in `chunk_k` / `chunk_v` / `qcopy` / `end` per pass, and the wall per (pass, head).
   - SSM: per chunk, cycles in conv / gate GEMMs / transcendentals / output.
   - Norm: pass 1 / stats / pass 2.
2. Where the stamps go:
   - **GEMM:** a 64 B record in the junk tail of the C object for epilogues that leave one: gated_b and gelu_pan
     write 10,368 B of an 18,432 B window, and the APAN `kstride >= WIN` puts the tail in padding. Plain objects
     have no tail: probe a gelu_pan op (mlp0) or a gated op (up_a).
   - **Attention:** the last 64 B of pass 2's window, which lands at +25,280 < O_KSTRIDE 25,344. Write it after
     `attn_end`.
   - **SSM:** the 448 B pad of each chunk's z slot.
   - **Norm:** it has no pad. Append one 64 B stamp slot per op: the drain TAP grows by 64 B into a new small ACT
     region.
   - **Check:** have the txnlint (P3) prove each stamp lands in padding before dispatching.
3. A host collector in step8 (`probe` command) reads the stamps after one dispatch and prints, per core and op:

   ```
   busy%, wait cycles (wall - busy), cycles per mm_acc call
   ```

   Repeat on 2-3 columns to see the spread across columns.

**Expected and what changes.**
- GEMM busy ~70% (if A-port bound) vs ~100% (compute) settles P1 without twins.
- Attention: the split between B-port waits (K/V re-streamed per pass) and exp/convert work picks P12's variant.
- SSM: which of conv / gates / exp / sqrt / sigmoid dominates picks P13's variant.
- Unambiguous op stamps also fix the timeline's mis-assignment when a write leaves a word unchanged (r5_tl12jcp
  shows negative spans).

**Result:**

---

## P3. `txnlint`: a static linter and DMA-accounting pass over the instruction stream — **S, risk: none (no device)**

**Hypothesis.** Most DMA bugs and many bubbles are visible in the `.bin` itself. mlir-aie 1.4.4 ships a decoder
(`aie.utils.txn_trace.trace`) that turns a stream into push / wait events with BD dims, lengths, `(arg, offset)`
addresses, repeat and iteration (verified on full12k.bin: `tools/txnstats.py`, `tools/txnq.py`).

**Procedure (spec)** — `tools/txnlint.py BUILD` runs in `npu-iron` with `--network none`, no device:
1. Decode `BUILD.bin`. Load `BUILD.sched` and the plan8 regions so every `(arg1, offset)` maps to a region name
   and an op.
2. Expand every push to its byte ranges, per column (dims + repeat + iteration).
3. Checks:
   - **(a) step limit:** any step >= 2^20 words.
   - **(b) RAW race:** a push reading ACT bytes that an earlier drain writes, without a `tct` wait on that drain's
     token in between.
   - **(c) WAR:** a drain overwriting bytes an earlier, un-awaited fill still reads.
   - **(d) BD reuse:** a `blockwrite` to a BD id whose previous push has not been awaited (the `finish()` hazard).
   - **(e) queue / head-of-line report:**
     - per channel, outstanding pushes > 4 between waits;
     - for each `maskpoll`, which pushes of other channels or columns wait behind it.
   - **(f) bytes per op per channel**, against plan expectations: flag the K=768 double reads, junk windows, and
     fills larger than the consumer needs.
   - **(g) burst length** and axcache uniformity.
   - **(h) stamp placement** for P2.
4. Output: a violations list, then a per-op table:

   ```
   op | DDR read MB | DDR write MB | pushes | waits | maskpolls
   ```

   This table feeds P4 and the roofline.

**Expected and what changes.**
- Either a proof that full12k has no races (today correctness relies partly on timing; round 3 found such a race),
  or the races themselves.
- The HOL report says whether reordering pushes (columns interleaved, B pushed before EW ops) is worth doing.
- Run it in `bb.sh` after every build; cheap insurance.

**Result:**

---

## P4. Event-driven schedule simulator, validated before use — **M, risk: none**

**Hypothesis.** A resource-level simulator predicts step latency from a schedule well enough to rank the
data-movement redesigns (P5, P6, P7, micro-batching, frame pipelining) before building them.

**Procedure.**
- Extend `tools/sim8.py` per ANALYSIS §6:
  - CP in-order with queue depth 4 and `maskpoll` blocking;
  - shim channels at 6.4 GB/s (DDR cap 55);
  - memtile A slot locks;
  - per-core A / B / C rings;
  - EW rings.
- Inputs:
  - tasks from txnlint (P3);
  - compute costs from P2 stamps (until then, sim8's fitted constants).
- Validation gates, all required before trusting it:
  - (a) r5 per-op within 10% and the step within 3%;
  - (b) no-GELU and SSM_COPY_ONLY builds predicted without refitting;
  - (c) one hold-out change predicted before it is measured, e.g. lookahead 1 vs 2, or EPI_OVERRIDE.

**Expected.** sim8 (first cut, fitted) is already within 1% on totals. The event-driven version adds overlap and
queue effects, which the redesigns need.

**Result:**

---

## P5. k-chunk-streamed A slot (removes the dependent-load latency and the K=768 double read) — **M, risk: medium (lock hangs)**

**Hypothesis.**
- Every dependent GEMM waits ~40 us for the whole 248 KB A slot. One BD releases `full` only at its end.
- The slot is k-step-major, so releasing per 10,368 B chunk lets the cores start after ~2 us.
- K=768 ops then load their A once.

**Procedure.**
1. Replace the static load chain with one S2MM BD using `BdIteration` over the chunks: acquire `empty` by nt, release
   `full` by nt per chunk.
2. Replace the sends with one MM2S BD iterating over the chunks (acquire `full` 1 / release `empty` 1). It is
   re-walked nt times: repeat count or chained.
3. Simulate the lock trace in P4 first. Dispatch one GEMM-only test (`t1`-style) with a 3 s timeout, then g11, then
   the full step.

**Expected and what changes.**
- [sim8] -3.0 ms per step (L_dep 40 -> ~5 us).
- -144 MB per step of DDR.
- Frees ~11 memtile BDs. That headroom later allows deeper prefetch: the next op's first chunks during this op's
  last tile.

**Result:**

---

## P6. Two A streams per column through neighbour memory (if P1/P2 say A-port bound) — **L, risk: medium**

**Hypothesis.** The GEMM loop runs at the A-port floor: 2,592 cycles of A per k-step vs 1,832 of compute.

**Design.**
- Split each A sub-tile into X and Y.
- X lands in row 3's memory, which rows 2 and 4 read as their N/S neighbour.
- Y is broadcast to rows 2 and 4; row 3 reads Y from a neighbour.
- Each core's A port then carries half: 1,296 cycles.

**Procedure.**
1. Prototype on one column with one op. Use hand-built rings and cross-tile lock handshakes (row 3 coordinates), and
   a second memtile MM2S for Y.
2. Verify bit-exactness against `run8` g11.
3. Measure t_ks.

**Expected and what changes.**
- [sim8] t_ks 1.857 -> 1.28-1.35 us: **-8 to -9 ms per step**.
- After it, the GEMM loop is compute-bound, so the asm loop work (1.59 -> ~1.35 bundles/MAC) and cheaper epilogues
  finally pay.
- The B port (1,152 cycles) is then the next bound, so n grows only with the B traffic.

**Result:**

---

## P7. Fold the norms into the GEMMs (row-scale epilogues, LN as a rank-1 correction) — **L, risk: low-medium**

**Hypothesis.** The 48 EW norm ops per step cost ~6.5 ms plus `finish_all` serialisation, and the GEMM cores idle
during them.
- `rms(x) @ W = rstd_row * (x @ diag(s+1) W)`.
- `LN(x) @ W = rstd_row * (x @ W' - mu_row * colsum(W')) + beta W + b`.

**Procedure.**
1. Numerics first (P8 harness): emulate the folded order on recorded real activations, then DAVIS 10 videos on the
   CPU emulation.
2. Prototype the epilogue row-scale with stats delivered as a per-stripe header in the A slot.
3. Add the residual in the producer epilogue: the residual tile comes as an extra C-sized object on the B port,
   which has 44% slack.

**Expected.** Norms 6.5 -> ~0-1.5 ms per step; with the removed brackets, [sim8] ~-5 ms. Also enables EW/GEMM overlap
later.

**Result:**

---

## P8. Numerics harness with real activations (value ranges, per-op error, bfp damage) — **M, risk: none (CPU only)**

**Hypothesis.** Random-weight tests miss range-dependent failures; the GELU clamp passed every one.
- Each nonlinearity and approximation has a valid input range.
- The real activations of TAPNext on real video can be checked against those ranges systematically.

**Procedure (spec)** — `tools/numharness.py`, CPU torch in `npu-iron-torch`:
1. **Capture.**
   - Run the vendored model (bf16) on clip30 + 3 DAVIS videos.
   - Record for every op input: per-channel min / max / absmax, the percentiles, and 1% random row samples.
   - Record the arguments of every nonlinearity: GELU x, softmax `s - m`, sigmoid args, the SSM's `log_a`, `a`,
     `1 - a^2`, the norm variance + eps, and the LN `|mean| / std`.
2. **Range check.** Compare against a table of each kernel's limits in nk8.h / ew8.cc / mm8.cc:
   - exp32 clamp +-60;
   - expd n >= -120;
   - lazy-max TAU = 8;
   - gelu exp-argument clamp;
   - recip seed domain;
   - rsqrt `max(vh, 1e-30)`;
   - the sqrt seed near 0.

   Report how close real data gets: margin to each limit, and the count beyond it.
3. **Per-op numerics.** Feed the recorded inputs to each kernel's numpy bit-level emulation (pack8 references,
   `encode_hw`) and to fp32 torch. Output the per-op error distribution. Save the worst 64 rows of each op as
   directed test vectors for run8 / t-tests, replacing random data.
4. **bfp damage report.** For every bfp16 operand (activation panels and weights), compute the per-8-block ratio
   `max / |x|`:
   - the fraction of values flushed to 0 or keeping fewer than 3 mantissa bits;
   - which channels are outliers.
5. **Depth error growth.** All depths share one xclbin, so build `.bin`s for depths 1..12. Read the residual stream
   after each and compare with torch block outputs: the error growth curve (extends `step8.py blocks`).

**Expected and what changes.**
- A standing regression suite that catches the next clamp-style bug.
- It decides whether cheaper approximations are safe: GELU (r5 gelufit), SSM transcendentals, norm folding P7.
- A per-block error budget.

**Result:**

---

## P9. Disassembly / object auditor — **S, risk: none**

**Hypothesis.** The big kernel regressions so far all leave fingerprints in the object code:
- -Oz 2.4x slower;
- `aie::inv` scalar;
- the non-pipelined chains;
- calls with a live output stream.

**Procedure (spec)** — `tools/objaudit.py OBJ [--baseline OBJ2]`:
- Run `llvm-objdump -d` (Peano), and find hardware-loop bodies (loop-start/end registers) and their bundles.
- Report per loop:
  - bundles, nop-slot fraction;
  - `jl` calls inside;
  - stack spills (`st` / `ld` against `sp`);
  - scalar elementary or transcendental ops;
  - vector extract/insert counts (scalarisation);
  - calls to `__divsi3`, `__udivsi3` and soft-float `__*sf*`.
- Report per symbol: text size, with the total checked against the 16 KB program memory and the stack estimate
  against the Worker stack.
- Diff against a baseline. Warn when a hot loop grows more than 10% in bundles or gains a call or spill.

It extends the existing `build/asm/loopcount.py`; run it in `bb.sh`.

**Expected.** Regressions caught before dispatch, and a program-memory headroom number that P13 and §5.4 of ANALYSIS
need.

**Result:**

---

## P10. Port-rate microbenchmarks (is aie2p's stream 32- or 64-bit, and what does broadcast cost?) — **S, risk: low**

**Hypothesis.**
- Every path we measured runs at ~4-4.5 B/cycle.
- AM027 documents 64-bit streams for aie2ps.
- If some path (memtile -> core, packet vs circuit, broadcast to 1 vs 3 cores) can do 8 B/cycle, the GEMM bound in
  ANALYSIS §2 moves.

**Procedure.**
1. Build one-column designs that move 4 MB over each path:
   - shim -> memtile;
   - memtile -> 1 core;
   - memtile -> 3 cores (broadcast);
   - shim -> core packet;
   - core -> memtile -> shim.
2. The cores only acquire and release, and stamp cycles (P2 technique). Report B/cycle per path; the
   `xfer*.py` / `pkt_test.py` designs are the starting point.

**Expected.** ~4 B/cycle everywhere (then MACHINE §3 is confirmed), or a faster path that changes P6's design.

**Result:**

---

## P11. Cheaper tile boundaries (the C drain and epilogue stall the A port) — **M, risk: low**

**Hypothesis.** Each tile costs 8.9 us of fixed time while the A stream idles (3-deep ring): the C object is
single-buffered and must drain (3.2 us) before acc_zero.

**Options.**
1. Drain the C object as two sub-objects, so rows 0-71 release while rows 72-143's epilogue runs.
2. Fold acc_zero into the first k-step: `mul` instead of `mac` for q == 0, which removes 288 stores per tile.
3. Deepen the A ring by 1-2 slots, if P9 shows the L1 headroom.

**Expected.** [sim8] t_tile 8.9 -> 3 us: -6.5 ms; realistically -3 to -4 ms.

**Result:**

---

## P12. Attention: let P2 decide, then restructure — **M, risk: low**

**Hypothesis.** Attention costs 1.37 ms per block against a ~0.6-0.75 ms estimate:
- QK^T + PV ~0.17 ms;
- expd ~0.2 ms;
- conversions ~0.15 ms;
- K/V re-streaming 885 KB per core per pass at 4 B/cycle = 0.155 ms x 2 passes.

The second pass (1 query tile) re-streams all K/V for half the work.

**Procedure.** After P2's stamps, choose between:
- all 3 query tiles in one pass, by freeing L1 for the online-softmax state;
- K/V kept for 2 heads per pass;
- Q objects pre-placed so the 0.04 ms Q slot load overlaps qkv's drain.

**Expected.** [sim8] attention 1.37 -> 0.7 ms: -8 ms.

**Result:**

---

## P13. SSM: 15.7 ms per step on 8 of 32 cores — **M, risk: low**

**Hypothesis.**
- The SSM is compute-bound on the row-5 core: the ring traffic alone takes 0.28 of 1.30 ms per block (measured,
  full12jcp).
- It is elementwise over tokens.

**Options, in order (after P2's breakdown and P8's accuracy checks).**
1. Cheaper transcendentals where the bf16 reference allows. Be careful with `a` near 1: the recurrence amplifies
   errors over frames, so test over 30+ frames.
2. Split the heads across 2 EW-capable cores per column, if P9 shows program-memory room on one GEMM row.
3. Overlap via frame pipelining (P14).

**Expected.** [sim8] SSM -> 0.5 ms per block: -9.7 ms.

**Result:**

---

## P14. Frame pipelining for fps (two frames in flight) — **L, risk: medium**

**Hypothesis.**
- Frame t+1's block b depends only on frame t's block b (the SSM state) and on its own block b-1.
- Interleaving two frames' op lists lets the GEMM cores and the EW cores work at the same time.

**Expected.** fps toward 1/64 ms (from 1/89) at ~1.3x per-frame latency. This is a Pareto point for throughput
rather than latency; only after P5-P7, when EW and GEMM times are closer.

**Procedure.** Price it in P4 first.

**Result:**

---

### Not proposed, and why

| idea | why not |
|---|---|
| activation residency in the memtiles as the next step | DDR is at 1/3 of peak; the binding traffic is memtile -> core (ANALYSIS §4) |
| a faster mm_acc inner loop before P1 | if the loop is A-port bound, it gains nothing |
| more GEMM epilogue fusion before P11 | the epilogue's cost is mostly the A port idling at tile boundaries |
