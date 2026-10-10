# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 3, 2026-10-09 evening, after round 7 and the first round-8 stamps (`RESULTS_FROM_MAIN.md`, round 7 part;
`npu-tapnext/build/g11p{a,c}.out`, `mmb_run.out`, `tl_full12r7g_d12.json`). Revision 2 is in `rev2/PROPOSALS.md`
(full text of P15-P20), revision 1 in `rev1/` (P1-P14). Savings marked [sim10] come from `tools/sim10.py` (ANALYSIS
rev 3 §2-3): depth 12, each change alone, against the r7g fit (71.7 ms). Every entry gives the hypothesis, the
procedure, the expected outcome (written before the build), the cost (S / M / L), the device risk and a **Result**
field for the main agent.

Common rules (SKILL §3, §9):
- one dispatch with a short timeout first for every new design, and the known-good test after any timeout;
- record Turbo / Default and gpu_busy with every number; cycles at 1.81 GHz;
- every prediction below is costed from the deployed binary (`tools/bundles.py` on an objdump) plus sim10 terms.

## Ranking (revision 3)

| rank | id | what | cost | risk | expected |
|---|---|---|---|---|---|
| 1 | **P21** | bank-aware asm GEMM call: no same-bank accumulator load + store in one bundle; A / B / C apart | S | low | in-call stall 44 -> <= 10 cycles per call: **-3.0 ms** [sim10, mmb] |
| 2 | **P20** (rev 3) | one asm call per k-step: A-ring locks inside, the 9 stripes pipelined | M | medium | **-5.0 ms** [sim10] (call + IRON code 395 -> ~40 cycles per k-step, 33 cycles of ramp/drain per stripe overlapped) |
| 3 | **P13** | SSM (1.32 ms per block, on the critical path, GEMM cores idle) | M | low | -> 0.5 ms: **-9.8 ms** [sim10]; stamp it first (P13a) |
| 4 | **P22** | program-memory diet: one GEMM segment, attention loops from a table, mm_core group loop, exp32 -> table | S-M | low | frees ~3.5 KB of the GEMM elf; bit-identical; time +-0.3 ms |
| 5 | **P24** | stamp the tile boundary (C handoff, acc_zero), then halve it (P11) | S, then M | low | t_tile 7.6 us fitted, ~1 us explained; halving it: **-4.2 ms** [sim10] |
| 6 | P7 | fold the norms into the GEMMs | L | low-medium | -4.6 ms [sim10] |
| 7 | P12 / P19 | attention 0.94 -> 0.6 ms per block (exp at the vldb floor; EW cores take a quarter) | M | medium | -4.1 ms [sim10] |
| 8 | P18 (full) | 2x2 register blocking inside P20's k-step | L | medium | mm_core 225 -> ~150: -5.8 ms alone [sim10]; then the A port binds (~0.77 us per k-step) |
| 9 | P5 (rev 3) | A slot: ping-pong halves for K = 768 ops, chunk-streamed loads for K = 1536 | M | medium | op-start wait (27-50 us measured, 21 fitted) on all 11 ops: -2.4 to -4 ms; -107 MB DDR per step |
| 10 | **P23** | balance epilogue kinds across the 3 rows of a column (yx, qkv) | S | low | -0.7 ms [sim10] |
| 11 | P14 | two frames in flight | L | medium | today 13.9 -> 18.1-18.6 fps at depth 12, latency 72 -> 124-130 ms (ANALYSIS §6) |
| - | P3, P4, P9, P10, P25 | tooling (txnlint, CP simulator, objdump auditor, port microbenchmark, `tools/bundles.py`) | S-M | none-low | see the end |

Combined: P21 + P20 -> 63.7 ms; every lever + the 2x2 GEMM -> **34.1 ms serial at depth 12** (28.5 at depth 10);
two frames in flight then 26.9 ms per frame (37 fps) at 65 ms latency (ANALYSIS §3, §6).

## Status of P1-P20 (from RESULTS_FROM_MAIN.md and the round-8 stamps)

| id | status | verdict / what is left |
|---|---|---|
| P1 twin builds | DONE | compute-bound. Note: round 6's "1.04 us per k-step" for down_a divided a 3-tile span by 4 tiles' k-steps; the whole-step -16.1 ms still implies an A port >= ~7 B/cycle |
| P2 cycle stamps | DONE via P15 | the stamps exist (mm8.cc -DPROF, r7/prof.py, r8/gprof.py) |
| P3 txnlint | open | needed before P14 |
| P4 event simulator | open, partly superseded | sim10 composes the k-step from the binary; the command-processor / DMA queue model is still unbuilt |
| P5 k-chunk A slot | open, **revalued up** | every GEMM op waits for its slot load at its first tile (27-50 us in g11), not just the dependent ones; rewritten below |
| P6 two A streams | REFUTED for now | revisit after P18 full: the 2x2 k-step needs ~7.5 B/cycle of A |
| P7 fold the norms | open | -4.6 ms [sim10] |
| P8 numerics harness | DONE (parts 1-3) | parts 4-5 open |
| P9 object auditor | PARTLY | rev 3 adds the linked-elf objdumps (`r8/*.dis`) and `tools/bundles.py`; baseline diffs open |
| P10 port microbenchmarks | clock DONE (1.81 GHz) | the memtile -> core rate is still open (matters after the 2x2 GEMM) |
| P11 tile boundaries | PARTLY | epilogue pass done (-5.3 ms); the rest waits for P24's split |
| P12 attention | PARTLY (round 7: exp by table, Q popped, unscaled scores) | 0.94 ms per block left |
| P13 SSM | open | the largest lever; P13a below |
| P14 two frames in flight | open, **re-priced** | 1.30-1.33x fps at ~1.8x latency today (ANALYSIS §6) |
| P15 core timer | DONE | bus enable + sched barriers |
| P16 counters | stamps DONE (P16-lite); counters not needed now | stamps + the bank microbenchmark answered the stall question |
| P17 bank placement (C++ loop) | **REFUTED** for the C++ loop | ... but the asm call DOES stall on banks: P21 |
| P18 asm GEMM | first step DONE: 225 bundles, bit-identical, -1.9 ms | the gap to -7.4 is explained (ANALYSIS §1): +2.6 wrapper, +3.4 bank stalls. Full 2x2 after P21 / P20 |
| P19 attention on 32 cores | open | needs P22's program memory on the EW elf (15,264 B) |
| P20 one call per k-step | open, **rewritten** below | now -5.0 ms |

---

## P21. Bank-aware asm GEMM call — **S, risk: low**

**Hypothesis** (ANALYSIS §1; MACHINE §2.2).
- The asm call stalls ~44 cycles per call (20% of its 225 bundles) on L1 bank conflicts. The evidence:
  - The mmb sweep: in-place accumulators (C load and store in one bank) cost +33. A and B in one bank cost +14. A or B
    with C cost +36-42.
  - The deployed placement predicts +44.6.
  - sim10 fits +44.0, and the g11 stamps show +53 over a static count.
- Two rules in `gen_mm.py` remove most of it:
  - **(R1)** never put an accumulator `vlda` and an accumulator `vst` in the same bundle;
  - **(R2)** never put an A `vlda.pop` / `fill` in the same bundle as an accumulator `vst`.
- A placement change removes the rest: A ring and B fifo in different banks, both away from the C object's bank.
- Why no placement can fix it alone: the 36 KB of fp32 accumulators span 3 banks.
- Why "same bundle" is the right rule: llvm-aie's model gives `vlda`, `vst` and the FIFO pops / fills the same
  memory cycle (`MemoryCycles<[5]>`, AIE2PGenSchedule.td), so only co-issued accesses can collide.

**Procedure.**
1. `gen_mm.py`: mark every lda / st op as acc / A / B. Add R1 and R2 to `put()` as slot-pair exclusions, so a
   double-booking moves the later op to the next free cycle. Keep the EDF order and the existing checks. Report the
   new bundle count; expect 225-232.
2. mmb with the new schedule over the same sweep. Pass: every placement <= 245 cycles (today 238-322); in
   particular (1, 3, 2, 2), which is today's 271.
3. Optional placement: the A ring's three buffers in one bank that holds neither B nor the accumulators. IRON
   `Buffer(..., address=)` if it exists in 1.4.4, otherwise the `.aie.bankN` sections or an explicit `data_size`
   layout. Check the ld script.
4. Full step: `regress` BIT-IDENTICAL (same MAC order), then the time, then g11 -DPROF.

**Expected** (pre-registered):
- g11 PROF cyc/call 337 -> **295 +- 8**;
- full step 71.2 -> **68.2 +- 0.7 ms** (s 44 -> <= 10).

If cyc/call stays above 315, the conflicts are with DMA writes (MACHINE §9 Q6). Then try placement (step 3) and
measure with the A / B DMA active in mmb.

**Risk.** Schedule-only change, verified by asmcheck and the conveyor asserts; bit-identical expected.

**Result:**

---

## P20 (rev 3). One asm call per k-step, A-ring locks inside, the stripes pipelined — **M, risk: medium**

**Hypothesis** (ANALYSIS §4).
- Per k-step, 395 cycles of non-MAC code sit between the MACs: IRON's 89 + 9 x 34 of the C++ wrapper.
  That is 14% of the in-situ k-step.
- Each call also pays a ~13-cycle ramp (A pops land at +8) and a ~20-cycle store drain.
- The lock operations themselves are nearly free:
  - one LCKREQ slot per 4 cycles;
  - ~1 cycle each when the lock is free (r7p stamps).
- So one entry per k-step, with the A-ring locks taken inside, costs ~40 cycles per k-step. With stripe j+1's A
  pops issued during stripe j's last group, the steady period is 4 x 48 = 192 cycles per stripe.

**Procedure.**
1. design8: give the A-ring locks fixed ids (`Lock(tile, init=a_ring, lock_id=n)`, as ew8 does), and stop the worker
   acquiring / releasing them. The worker becomes `b = in_b.acquire(1); mm_kstep(a0, a1, a2, b, cbuf); in_b.release(1)`.
   Pass the three A buffers as arguments: their addresses are fixed but not contiguous.
2. `gen_mm.py` -> `mm_kstep`:
   - a zero-overhead hardware loop over the 9 stripes;
   - per stripe: `acq #(48+full), -1` before the A fill, `rel #(48+free), 1` after the last A pop has landed
     (pop +8);
   - A slot pointer = j mod 3, kept in a pointer register rotated with `sel`, or 3 unrolled stripes per loop trip;
   - accumulator base += 4 KB per stripe, switching to `acc_hi` at ACC_LO. Pass both bases, or place `acc_hi` right
     after the C object's fp32 rows;
   - the stripe boundary modulo-scheduled: A_q of stripe j+1 re-popped after the last reader of ex_q in stripe j,
     as `gen_mm.py` already does between row blocks;
   - keep P21's R1 / R2.
3. Directed tests (t1 / t2 / g11) bit-exact, `regress` bit-identical, then the step and g11 -DPROF (busy region now
   per k-step).

**Expected** (pre-registered, from the P21 state):
- k-step static ~1,770 cycles (today 2,420 + 44 x 9 stalls);
- full step **-5.0 +- 1.5 ms**;
- GEMM elf text -400 B per GEMM loop copy.

**Risk.**
- A lock misuse hangs the core: one dispatch with a 3 s timeout.
- `acq` in a hardware-loop body: check that llvm-aie's assembler accepts it in the loop's last bundles; otherwise put
  it at the top.
- Deadlock if the kernel releases A "free" before the last pop of that slot has landed: assert it in the generator.

**Result:**

---

## P13a. Stamp the SSM in place before redesigning it — **S, risk: low**

**Hypothesis.**
- The SSM's 1.32 ms per block is the largest single item, and the only split so far is round 5's differential:
  transcendentals vs ring I/O.
- P15's processor-bus enable works on any core. Stamps in ew8's SSM give the phase split in situ:
  - ring input wait;
  - gate math (sigmoid / softplus / exp);
  - the scan over tokens;
  - output pushes.

**Procedure.**
- `npu_maskwrite32(0x32038, 1, 1, column=c, row=EW_ROW)` for the 8 EW cores.
- Stamps around the SSM's phases (sched barriers).
- A record written into the SSM's own output tail (timing build only).
- Program memory: the EW elf is 15,264 B, ~1.1 KB free.

**Expected.** Compute-bound phases summing to >= 1.0 ms of the 1.32. If the input-ring wait is > 0.2 ms, the fix is
I/O (a second input ring or the packet channel), not math.

**Result:**

---

## P22. Program-memory diet for the GEMM cores — **S-M, risk: low**

**Hypothesis** (ANALYSIS §5). 16,176 of 16,384 B are used.

| change | frees |
|---|---|
| (a) one GEMM segment walked by a table in which attention is an entry (today two copies of the loop nest, 992 + 1,104 B) | ~1.0 KB |
| (b) the attention control's 3 x 3 Q-copy loop and k / v loop with trip counts from an L1 table (today unrolled: 9 `jl attn_qcopy`, 928 B) | ~0.8 KB |
| (c) `mm_core`'s 4 groups as a modulo-scheduled hardware loop (today 1,524 B straight-line) | ~0.8 KB |
| (d) `rescale_rb`'s factor from the attention exp table, dropping `nk8::exp32` | 304 B |

**Procedure.** One change at a time. `tools/bundles.py X.dis --sizes` and `llvm-size` after each. `regress`
BIT-IDENTICAL for (a)-(c). For (d), the expemu check on real scores, then DAVIS.

**Expected.**
- text <= 12.8 KB (-3.4 +- 0.5 KB);
- step time +-0.3 ms (the loop trips add < 10 us per block);
- -DPROF fits in the full step again, and `attn_begin` can go back to -O2 (+304 B, ~-0.1 ms).

**Result:**

---

## P23. Balance epilogue kinds across the rows of a column (yx, qkv) — **S, risk: low**

**Hypothesis.**
- The A slot is broadcast to the column's 3 GEMM rows through a 3-object ring. A row that is more than ~3 calls
  behind holds the stream, and the other two rows wait at their A acquires.
- plan8 gives yx's flags as `["gelu"] * 12 + ["plain"] * 12` over the 3 x 8 tiles. With row r taking tiles
  r*8..r*8+7 (the B TAP's `r * nt * lw`):
  - row 0 runs 8 GELU epilogues;
  - row 1 runs 4;
  - row 2 runs none.
  The measured extra waits are yx +5.04 us per tile in the full step (r7p) and +40 cycles per call in g11, close to
  t_gelu (4.7 us).
- qkv: rows 0 / 1 / 2 pack Q / K / V objects. That costs +1.23 us per tile, and +14 cycles per call in g11.

**Procedure.**
1. Assign N tiles so that at every tile index the 3 rows run the same epilogue kind. yx: row r takes y tiles
   4r..4r+3, then x tiles 4r..4r+3. qkv: each row gets 4 Q, 4 K and 4 V objects in the same order.
2. The flags list, the weight packing order and the per-tile drain TAPs follow (all host side, plan8 / pack8).
3. `regress` must be BIT-IDENTICAL (only the tile-to-core mapping changes).

**Expected.**
- yx median k-wait per tile 7.7 -> <= 3.0 us; qkv 3.9 -> <= 3.0 (r7/prof.py);
- step **-0.6 +- 0.2 ms**.

**Result:**

---

## P24. Stamp the tile boundary, then shrink it — **S (stamps), then M; risk: low**

**Hypothesis.**
- sim10 fits t_tile = 7.3-7.6 us per GEMM tile: 92 tiles per block, 8.4 ms per step.
- The stamps account for ~1 us of it: plain epilogue 0.63 us, the tail 69 cycles, acc_zero ~0.33 [INF].
- The rest lies between the epilogue's end and the next `acc_zero`: the C object handoff.
  - The C L1 object is single-buffered (`depths=[1]`).
  - Its 18 KB move to the memtile join (>= 1.3 us at <= 8 B/cycle) only when a 55 KB L2 slot (depth 2) is free.
  - The shim drains a slot in ~8.6 us at ~6.4 GB/s.

**Procedure.**
1. Two more fields in the -DPROF record:
   - cycles from the end of `acc_epilogue` to the next `acc_zero` entry (the C acquire wait plus loop code);
   - `acc_zero`'s own cycles.
   g11 has the program memory.
2. Per op: handoff mean / max.

**Expected.** Handoff 3-6 us per tile, rising on ops whose tiles are short relative to the 8.6 us column drain.
- If it is the L1 -> L2 move: fold `acc_zero` into the first k-step (the asm's first k-step MACs from zero with no
  accumulator loads), and split C into two 9 KB objects so the first half drains during the second half's epilogue.
- If it is the L2 slot: c_depth 3, which costs memtile space; check the 512 KB budget.
- Then P11's gain [sim10: -4.2 ms for a halved t_tile].

**Result:**

---

## P5 (rev 3). A slot: ping-pong halves for K = 768 ops, chunk-streamed loads for K = 1536 — **M, risk: medium**

**Hypothesis.**
- The memtile A slot (248,832 B) is single-buffered. Op i+1's load waits for op i's last send, so every GEMM op's
  first tile waits for a whole-slot load. g11 stamps: 27-34 us for K = 768 ops, 27-50 us for K = 1536.
- K = 768 ops read their 124 KB source twice to fill both halves, 107 MB of DDR per step.

**Procedure** (two parts):
- (a) K = 768 ops:
  - load only the half the op will send from, alternating halves between consecutive K = 768 ops, so op i+1 loads
    into the half op i does not use. This needs the send side to follow the op's half, not a fixed [half 0, half 1]
    cycle;
  - two memtile MM2S channels as packet sources into the column's A stream, one per half, ordered by the slot locks;
  - check the memtile channel and BD budget (6 MM2S, 48 BDs).
- (b) K = 1536 ops:
  - `Bd(iteration=...)` loads in k-chunks with a lock release per chunk;
  - sends acquire chunk by chunk;
  - lock values <= 63.

**Expected.** sim10: L_dep 21 -> 5 us on 11 ops, -2.4 ms; up to -4 ms if the measured 27-50 us first-tile waits go
entirely. DDR -107 MB per step.

**Risk.** Memtile BD / lock accounting (the static chain is near 22 BDs). P3 (txnlint) should check the new
sequence.

**Result:**

---

## P14 (re-priced). Two frames in flight

See ANALYSIS §6.
- Today: 53.8-55.3 ms per frame (18.1-18.6 fps) at depth 12, latency 124-130 ms. Serial is 71.7 ms (13.9 fps).
- After P21 + P20: 47.6 ms per frame (21 fps).
- After every lever: 26.9 ms per frame (37 fps) at 65 ms latency.
- The gain is the SSM hidden behind the other frame's GEMM work, so P13 takes most of it first.
- Build only after P3 (txnlint) and with the norms kept off the A-load channel's critical time.

**Result:**

---

## Tooling

- **P25 `tools/bundles.py`** (done, rev 3): static bundle counts per function / address range / call target from a
  linked elf's objdump. Use it for every prediction. It found the 34-bundle wrapper and the 89-bundle IRON k-step.
- **P3 txnlint** (S, no device): races, BD reuse, queue head-of-line, bytes per op from the `.bin`. Before P5 and P14.
- **P4 command-processor simulator** (M): sim10 covers ops; the sequence / queue / DMA level is still missing.
- **P9 remainder** (S): diff `--sizes` and the k-step bundle count between builds in `bb.sh`, so a program-memory or
  wrapper regression shows at build time.
- **P10 port microbenchmark** (S): the memtile -> core A rate, before P18 full.
