# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 5, 2026-10-10, after round 9 (`full12r9s`, 58.14 ms at depth 12; `RESULTS_FROM_MAIN.md` round 9). Revision 4 is in
`rev4/PROPOSALS.md`, older ones in `rev3/`, `rev2/`, `rev1/`. Analysis behind every number: ANALYSIS rev 5.

**Headline correction.** The round-9 statement "the GEMM k-step is A-port-bound at ~4 B/cycle" does not hold.
- The steady in-situ k-step is **2,055 cycles in mm_ks + 67 between calls** (median of 13,440 g11-r9sp records,
  `tools/profdec.py`).
- 2,588 was the mean including each op's first tile. That tile now holds the op-start A-slot wait (28-34 us), because the
  A acquire moved into the kernel.
- The K = 768 / nt 4 ops' "3.1 B/cycle" is the same wait over half the k-steps.
- The thing that runs at ~4 B/cycle per column is the **DDR -> memtile slot load** (all 8 columns at once: 55-60 GB/s, the
  DDR peak).
- See ANALYSIS rev 5 §1. Please correct the skill's "per-core input port ~4 B/cycle binds a fast GEMM" line and NOTES 14.2
  accordingly.

Savings marked [sim12] come from `tools/sim12.py`:
- depth 12, each change alone, against r9s;
- calibrated on the measured g11 op walls plus one fitted constant;
- output in `r10/sim12_r9s.out`.

Every entry gives the hypothesis, the procedure, the expected outcome (written before the build), the cost (S / M / L) and
the device risk. Results go into RESULTS_FROM_MAIN as before.

Common rules (SKILL §3, §9):
- one dispatch with a short timeout first for every new design, and the known-good test after any timeout;
- record Turbo / Default and gpu_busy with every number;
- **report stamp medians and percentiles, not only means** (`profdec.py` does).

## Ranking (revision 5)

| rank | id | what | cost | risk | expected [sim12] |
|---|---|---|---|---|---|
| 1 | **P38** | stall counters in the PROF record (core LOCK_STALL / MEMORY_STALL, memory-module S2MM lock / backpressure) | S | low | decides whether A binds in situ and what the 201 extra cycles per k-step are; 0 ms itself |
| 2 | **P5r + P5a** | up_b reuses up_a's slot (same A); K = 768 slots loaded once (packet multicast into both halves) | S (P5r) / M (P5a) | low / medium | **-0.4 (P5r) and -1.2 (P5a) = -1.6 +- 0.3 ms** |
| 3 | **P23** | balance epilogue kinds across a column's 3 rows (qkv, yx) | S | low | **-1.2 +- 0.3 ms** |
| 4 | **P7** | P7a numerics emulation, then the norm fold | L | medium | **-6.8 ms full / -4.6 conservative** |
| 5 | **P5** | k-chunk streaming of the slot (on top of P5r / P5a) | M | medium (lock values, BD iteration) | **-1.7 more (-3.3 total) +- 0.7 ms** |
| 6 | P27 | SSM a by table: one A1 copy in the free bank-2 tail first, the second copy after the stack shrink (P42) | S-M | low | -0.5 +- 0.2 (one copy), -0.7 (two) |
| 7 | P12 / P19 | attention 930 -> 600 us | M | medium | -4.0 ms |
| 8 | P35 | B locks inside mm_ks, one call per tile | S-M | medium | -0.47 +- 0.15 ms |
| 9 | P24b | acc_zero folded into the first k-step | S | low | -0.54 ms |
| 10 | P29 | SSM chains hand-scheduled | L | low | -3.5 ms |
| 11 | P34 | A-rate twin of mm_ks (timing build) | S | low | measures R_A in situ: 1,364 +- 150 cycles per k-step predicted |
| 12 | P37 | DMA-collision placement (after P38) | M | medium | -0.4 to -0.9 ms |
| 13 | P14 | two frames in flight, after P7 | L | medium | today 1.33x fps (43.7 ms per frame), after P5 + P7 + P23 1.22x |
| parked | P36, P39, P40, P41 | deep-K accumulator visits, neighbour-memory A, A split over 2 S2MM, a 4th GEMM row | L | high | only after P34 / P38 and only with L1 freed (ANALYSIS §2) |
| tools | P42, P30, P33, P3 | EW stack / frame audit in bb.sh, bank linter, r9s timeline, txnlint | S | none | |

Combined (sim12, depth 12):
- P5 -> P7 -> P23 -> P35 + P24b -> P27 -> attention = **40.9 ms** (today's k-step);
- + SSM hand-scheduled + P37 + a 1,600-cycle k-step = **34.7 ms** (28.8 fps), 36.9 if R_A is only 5.5 B/cycle;
- depth 10: 29.0 ms.

## Status of the earlier proposals

| id | status | verdict / what is left |
|---|---|---|
| P1 twin builds | DONE | compute-bound (rev 5: the no-MAC twin also gives the A path >= 5.5 B/cycle, hard; ~7.6 fit) |
| P2 / P15 stamps | DONE | P38 extends the record with counters |
| P3 txnlint | open | before P5 (chunked) and P14 |
| P4 command-processor simulator | open | |
| P5 A slot | **rewritten (rev 5)**: P5r, P5a (packet multicast), P5 (chunks) | the op-start wait is a DDR-bound load (55-60 GB/s for 8 columns) |
| P6 two A streams | superseded by P40 (parked) | |
| P7 norm fold | open | the largest single lever now; P7a first |
| P8 numerics harness | DONE | |
| P9 object auditor | PARTLY | P42 adds the stack audit |
| P10 port microbenchmarks | clock DONE; port: P34 | |
| P11 tile boundaries | PARTLY | P24b, P35 |
| P12 / P19 attention | open | |
| P13 SSM | PARTLY | P26, P28 done; P27, P29 left |
| P14 | open | after P7 |
| P16 counters | **revived as P38** | |
| P17 | REFUTED (rev 3) | |
| P18 2x2 blocking | open, **blocked by L1** (GEMM core ~0.9 KB free) and gated by R_A | see P36 |
| **P20** one asm call per k-step | **DONE** (round 9, -6.4 ms) | in-situ k-step 2,055 + 67 = 2,122 cycles vs 1,940 (1,850-2,100) predicted: static +34, IRON / B gap +42, in-situ extra +~80. Not A-bound |
| P21 bank placement | DONE / rules REFUTED | |
| P22 program memory | (a) + (b) DONE | GEMM text 15,840 B, EW 16,240 B (144 B free: P27 must shrink the exp passes) |
| P23 row balance | open, **re-priced -1.2** (was -0.7) | the r9s walls: qkv's slowest row costs 80 us per block |
| P24 / P24b | P24 DONE; P24b open | |
| P25 bundles.py | DONE | |
| **P26** exact sqrt table | **DONE** (round 9) | with P28: -2.8 ms vs -2.9 predicted |
| P27 a by table | open, **unblocked** (one copy fits the free bank-2 tail today) | |
| **P28** sigmoid table | **DONE** (round 9) | DAVIS 70.84 / 69.90 |
| P29 SSM hand-scheduled | open | |
| P30 bank linter | open | |
| P31 rotating accumulator | parked | program memory |
| P32 DMA bank bench | **replaced by P38** (in situ, no new bench) | |
| P33 timeline | open (r9s) | would split sim12's fitted 873 us (SSM + syncs) |

---

## P38. Stall counters in the PROF record — **S, risk: low** (rank 1)

**Hypothesis.** mm_ks runs 2,055 cycles per k-step in situ against 1,830 static + 24 modelled bank cycles. The other ~201
cycles are DMA collisions, lock waits or record loads, and nothing measured so far separates them. Whether A binds in situ
(it needs 5.0 B/cycle; idle, the path gave >= 5.5) is the same question. The hardware counts both:
- core module: 4 counters, `Performance_Control0/1` at 0x31500 / 0x31504, `Counter0-3` at 0x31520-0x3152C;
  events LOCK_STALL 26, MEMORY_STALL 23, TRUE 1;
- memory module: 2 counters, `Performance_Control0` at 0x11000, `Counter0/1` at 0x11020 / 0x11024; events
  DMA_S2MM_{0,1}_STALLED_LOCK 31 / 32, DMA_S2MM_{0,1}_MEMORY_BACKPRESSURE 39 / 40, DMA_S2MM_{0,1}_STREAM_STARVATION 35 / 36
  [DOC aie-rt `xaie2pgbl_params.h`, `xaie_events_aie2p.h` in the npu-iron image].

**Procedure.**
1. Runtime sequence (as for the timer's bus enable), per GEMM core, `npu_write32`:
   - core counters 0-3 = LOCK_STALL, MEMORY_STALL, TRUE (self-check), ACTIVE 28, each with start event = stop event =
     the event;
   - memory-module counters 0-1 = S2MM_A STALLED_LOCK, S2MM_A MEMORY_BACKPRESSURE. Take A's S2MM channel number from
     `input_with_addresses.mlir` (`aie.dma_start(S2MM, n)` of the A ring BDs).
2. mm8.cc PROF: `read_tm(0x31520..0x3152C)` and `read_tm(0x11020 / 0x11024)` at acc_zero and at the epilogue (fenced like
   the timer reads). Write the 6 deltas into record words 12-17.
3. g11 PROF build, one dispatch (3 s timeout), then `profdec.py` with the extra columns.

**Self-check.** The TRUE counter must equal the timer delta of the same window to +-1%. If not, "start = stop" does not
count occurrences on aie2p; then try start = event, stop = never (0) with a reset per window.

**Expected (pre-registered), per steady k-step:**
- core LOCK_STALL **<= 40 cycles** (A is not binding);
- core MEMORY_STALL **120-200** (24 bank + DMA collisions);
- S2MM_A STALLED_LOCK **400-800** (the DMA waits for a free slot, because A arrives faster than mm_ks consumes it);
- S2MM_A MEMORY_BACKPRESSURE > 0.
- First tile: core LOCK_STALL ~ the first-tile excess (51-62 K cycles).

**If instead** steady LOCK_STALL >= 150 and S2MM_A STALLED_LOCK ~0: A binds in situ at ~5.0 B/cycle. Then P37 (fewer DMA
collisions) and P34 / P39 / P40 move up, and every compute lever on the k-step is capped at 2,055.

---

## P5r + P5a. The A slot: reuse up_a's load for up_b; load K = 768 slots once — **S / M** (rank 2)

**Hypothesis.**
- Measured first-tile waits (g11-r9sp, from mm_ks entry): K = 768 ops 32.8-34.3 us, K = 1536 28.2-34.2 us. The load is
  248,832 B per column x 8 columns in ~33-36 us: DDR-bound at 55-60 GB/s.
- **P5r:** up_a and up_b have the same A TAP (11,064,320 in full12r9s.sched), yet up_b reloads it.
- **P5a:** K = 768 ops load their 124 KB twice (TAP dim 0 = 2 with stride 0) because the static send chain alternates the
  slot's halves per tile.

**Procedure.**
- **P5r (plan-level, design8 chain).**
  - Drop up_b's load from `load_plan`: up_a's load releases `full` with nsend(up_a) + nsend(up_b) = 32, and the next load
    (down_a) acquires `empty` with 32 (<= 63).
  - Drop up_b's A fill from the runtime sequence.
  - Assert in plan8 that the two TAPs are identical.
- **P5a (packet multicast).**
  - The shim's A fill for a K = 768 op carries packet id 3. The memtile's packet rules route it to **both** S2MM channels
    0 and 1; each writes one half, from one DDR read. The memtile has 6 S2MM; A uses 1 today.
  - K = 1536 ops send id 1 (-> half 0) and id 2 (-> half 1). That is 3 packet rules on the memtile's slave port from the
    shim; 4 are allowed [DOC TM].
  - IRON 1.4.4's PacketFlow has one destination, so this needs raw `aie.packet_flow` with two `aie.packet_dest`, or a
    hand-written switchbox. Check the lowered packet rules (`input_with_addresses.mlir`) before dispatching.
  - Alternative without multicast: rewrite the memtile send BDs per op from the runtime sequence. That is more
    command-processor traffic; not preferred.
- Directed test: g11 and the full-step regress bit-identical (the data are unchanged).

**Expected:**
- P5r: up_b's first wait 33 -> ~2 us, **-0.4 ms**;
- P5a: K = 768 first waits 33 -> ~17 us (DDR bytes halved), **-1.2 ms**;
- together **-1.6 +- 0.3 ms** and -107 MB of DDR reads per step.

**Risk.** P5r low (lock arithmetic; assert <= 63). P5a medium: packet multicast is untested here, and a wrong rule
deadlocks the memtile S2MM (a timeout).

---

## P23. Balance epilogue kinds across a column's rows — **S, risk: low** (rank 3)

**Hypothesis.** The A broadcast makes a column's 3 cores advance together, so an op's wall is its slowest row's.
- qkv: rows run epilogues of 3,700 / 3,700 / 7,260 cycles per tile; the fast rows' k-steps stretch to 2,350.
- yx: one row runs all-GELU epilogues (10,980 cycles per tile), one all-plain; the plain tiles' k-steps stretch to
  2,860.
- Walls (g11wall): qkv 337 us vs ~257 balanced; yx 210 vs ~188.

**Procedure.** In plan8, deal N tiles to rows round-robin (row r takes tiles r, r + 3, ...) instead of in contiguous
ranges, so each row gets the same mix of q / k / v and GELU / plain tiles. The C drain TAPs and the qkv packing offsets
follow the mapping. Regress bit-identical.

**Expected.**
- qkv wall 337 -> 255-265 us, yx 210 -> 185-192 us (g11wall);
- step **-1.2 +- 0.3 ms**;
- mlp3_a's column spread (16.7 us per block) is not addressed.

---

## P7 (rev 4, re-priced). Fold the norms — **L, risk: medium** (rank 4)

Unchanged procedure (P7a numerics emulation on acts_clip3 first: the bfp ratio of each consumer's output <= 1.1, RMS sites
first).

**Expected [sim12]:** full fold **-6.8 ms** (4 norm ops and 4 EW boundaries gone, +0.3 us of epilogue per tile);
conservative (40 us stats passes) -4.6 ms. It also shrinks the EW side for P14 and frees the EW core (§ P42) of the norm
code: 4.2 KB of stack path and ~2.9 KB of text.

---

## P5 (rev 5). Stream the slot in k-chunks — **M, risk: medium** (rank 5, after P5r + P5a)

**Hypothesis.** With P5a the first tile still waits for the whole slot: 17 us for K = 768, 33 for K = 1536. Streaming the
slot in chunks lets the first k-steps start after the first chunk. The first tile then runs at max(compute, DDR):
8 columns x 10,368 B per 1.17 us = 71 GB/s of demand, above DDR's ~55. So it is DDR-paced, not free.

**Procedure.**
- One load BD with `Bd(iteration=...)` per op (chunks of 4 k-steps = 41 KB), releasing `full` per chunk.
- The send BDs acquire per chunk.
- **Lock budget:** up_a's 16 sends x 6 chunks = 96 > 63. Use chunks only for the first tile (a 2-BD pattern per op:
  chunked first half, whole remainder) or acquire per (tile, half). Run P3 txnlint first.

**Expected [sim12]:** first waits ~3 us (K = 768), ~6 us (K = 1536); **-1.7 ms beyond P5r + P5a (-3.3 total) +- 0.7**.

---

## P27 (rev 5). a by table, unblocked — **S-M, risk: low** (rank 6)

Unchanged numerics (rev 4: exact for a >= 0.95; ANALYSIS rev 4 §4). Memory and program memory, from the r9s EW elf
(ANALYSIS rev 5 §3):
- **Step 1:** ONE A1 copy (1,696 B) in `.aie.bank2`. The bank-2 region is 5,120 B and holds only ssig_cd (2,560 B), so
  2,560 B are free. Both LUT pointers point at it; the round-9 probe measured +8 cycles per 32 lanes for that.
- **Program memory:** the fused pass replaces loga_p + exp_pA..D (1,728 B of text). The EW core has 144 B free, so the new
  pass must stay under ~1.8 KB, which it will (~600-800 B).

**Expected:** -0.5 +- 0.2 ms with one copy; -0.7 with the second copy after P42 step 1.

---

## P42. EW memory: stack audit and shrink — **S, risk: low** (tool + memory)

**Hypothesis.** The EW stack is 6,144 B; the deepest call path in the linked r9s elf (`llvm-readelf --stack-sizes`) is
4,224 B:
- core 64 + ew_op 64 + ew_norm 128 + norm_rows<LN> 2,944 + ln_stats 1,024; the SSM path is 768 B.
- norm_rows<LN>'s 2,944 B and ln_stats' 1,024 B frames are spills: A32 / B32 passed by value and by reference into
  noinline minsize helpers, and pass 2's ~10 live vectors.

**Procedure.**
1. bb.sh: after linking, compute the max call path from `--stack-sizes` plus the call graph (`llvm-objdump -d` `jl`
   targets). Assert path + 128 <= the Worker's stack_size, for every core.
2. EW stack 6,144 -> 4,416. IRON places ewo0 after the stack, so the bank-0 free tail grows to ~2,752 B; put A1's second
   copy there beside ssqrt_ab. Check ewo0's address and the ldscript bank0 LENGTH in the build.
3. Shrink the frames:
   - ln_stats / rms_rstd take pointers into a static 512 B scratch (g_ptmp is idle during norms);
   - pass 2's per-cb LN math becomes a noinline function per k-step.
   - Target: norm path <= 1.8 KB, stack 2 KB: 4.1 KB of bank 0 freed in all.
   - Time the 4 norms before / after (124-148 us each).

**Expected.** Step 2 frees 1,728 B with no speed change. Step 3 frees +2.4 KB, with norms within +-3% (if slower by more,
keep step 2 only).

---

## P35. B locks inside mm_ks, one call per tile — **S-M, risk: medium**

**Hypothesis.** Between mm_ks calls, IRON's B acquire / release and loop code cost 67 cycles per k-step (g11-r9sp gap
median).

**Procedure.**
- One asm call per tile, `mm_tile(rec, nks)`: the k-step loop goes inside, and B's two objects alternate.
- Use B's lock ids from `input_with_addresses.mlir` (rev 4 read B 0/1). Pin them, or assert them in design8 as for A's
  4 / 5.
- `acq` B with -1 (acquire >= 1), and `rel` after the last B pop of the k-step plus 6 cycles.

**Expected:** gap 67 -> ~12 cycles; step **-0.47 +- 0.15 ms**; GEMM text about -60 B (one call site).

**Risk.** B is an IRON ObjectFifo forward, so its lock ids are allocated, not chosen: assert them in the design.

---

## P24b. acc_zero folded into the first k-step — **S, risk: low**

**Mechanism.** In a tile's first k-step, mm_ks's piece loads read a 1 KB block of zeros instead of the pieces. The record
stream points every load cursor of that k-step at the zero block; the stores still go to the real pieces. No change to
the loop body.
- acc_zero (0.49 us x 92 tiles per block) disappears.
- Needs 1 KB of L1. The GEMM core has ~0.9 KB free, split over 3 banks. Apply P42's stack audit to the GEMM cores first:
  they reserve 0xB00; round 6 measured ~2.5 KB of need, before mm_ks replaced the C++ loop.

**Expected:** **-0.54 ms**, bit-identical (0 + x = x exactly in fp32).

---

## P34. A-rate twin of mm_ks — **S, risk: low** (timing only)

**Hypothesis.** P38 tells whether A binds today; P34 measures how fast A can come in situ with today's lock protocol and
ring (the idle-core twin says >= 5.5, ~7.6 B/cycle). It gates P36 / P40: a k-step below ~1,600 is worth -3.9 ms at R_A 7.6
but only -1.5 at 5.5.

**Procedure.** gen_ks.py `--aonly`: the same zero-overhead stripe loop, per stripe `acq`, the 16 + 16 A pops (vlda.pop,
results discarded), `rel`; no MACs, no B pops, no accumulators. IRON keeps B's acquire / release. g11 timing build, PROF
on; outputs are garbage.

**Expected:** steady **1,364 +- 150 cycles per k-step** (R_A 7.6). >= 1,700 means R_A < 6.1: the A path is the next wall
after any compute work.

---

## P37. DMA-collision placement — **M, risk: medium** (after P38)

If P38 shows MEMORY_STALL >> 24 per k-step: the B DMA writes the next B object into bank 3 while mm_ks pops the current
one from bank 3 at ~63% of cycles. Put B's two objects in different banks, so the DMA writes a bank the stripe touches
least.
- Needs 4.6 KB free in another bank of the GEMM core, which has ~0.9 KB free. Only after P39-style freeing or a
  smaller C object.
- **Expected:** MEMORY_STALL -60 to -140 per k-step, **-0.4 to -0.9 ms**.

---

## P14 (re-priced). Two frames in flight — **L, risk: medium**

sim12's list bound: today 43.7 ms per frame (1.33x fps, latency ~87 ms); after P5 + P7 + P23 38.2 ms (1.22x); at the
ceiling 29.4 vs 34.7 ms (1.18x, latency ~59 ms). Its value falls as P7 and the SSM work land, and it doubles latency.
Last fps lever.

---

## Parked layer-ii (memtile -> core) options — revisit only after P34 / P38 (ANALYSIS rev 5 §2)

| id | option | mechanism | rate | blocker |
|---|---|---|---|---|
| P36 | deep-K accumulator visits (k = 128 per piece visit; the "2x2" successor) | 16 MACs per accumulator visit, group boundary amortised over 64 MACs: static ~1,430 per 64-K | compute ~1,600 in situ | +8 KB L1 (B fifo + A ring) in a full GEMM core |
| P39 | A from the middle core's memory | rows 2 / 4 pop A from row 3 (N / S neighbour); row 3 takes A on both S2MM; B for row 3 via row 2 | A per column up to 15 B/cycle; frees 3.4 KB on rows 2 / 4 | neighbour locks in asm; 3-core bank behaviour unmeasured; row 3 needs a deeper ring. Bench one column first |
| P40 | A split over the core's 2 S2MM | two A half-rings; B shares one channel in S2MM out-of-order mode [DOC] | port floor 1,364 -> 985 | out-of-order BDs not in IRON 1.4.4; B forward must become a PacketFlow |
| P41 | 4th GEMM row (EW elsewhere) | m 72 x 2 groups x n 128 per column | A 5,184 / B 9,216 per k-step | needs the EW work gone (P7 + SSM elsewhere) |

Per-core A streams instead of the broadcast (ANALYSIS §2.2) are not a speed lever on their own: the op still ends with its
slowest row, so P23 is the fix.

## Tools

- **P38** above.
- P30 bank linter in the build: unchanged; add `--stack` from P42.
- **P33 r9s timeline**: unchanged, and more useful now. sim12's one fitted constant (SSM + syncs, 873 us per block) is
  the only unmeasured part of the block.
- `profdec.py` / `g11wall.py`: run them on every g11 PROF image. They need only the ACT `.npy` and the `.sched`.
