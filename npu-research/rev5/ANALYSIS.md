# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 5, 2026-10-10. State analysed: round 9's `full12r9s` (58.14 ms per depth-12 dispatch, Turbo, gpu_busy 0) with the
round-9 GEMM stamps (`npu-tapnext/build/g11sp.npy`, the ACT image of the `g11-r9sp` PROF run) and the r9s depth sweep.
Revision 4 is in `rev4/`. Tags as in MACHINE.md: [DOC] vendor source or the deployed binary, [MEAS] measured by us,
[INF] model or arithmetic. Cycles at **1.81 GHz** [MEAS].

New tools this revision (`tools/`, none needs the device; outputs in `r10/`):
- `profdec.py`: decodes the GEMM PROF records straight from a saved ACT image + the build's `.sched` (no pyxrt).
  It prints medians, percentiles and per-column spreads, not only means.
- `profdist.py`: per-tile records of one op instance in drain order (which tile is slow).
- `g11comp.py`, `g11wall.py`: the GEMM-only step composed from the records. Per op: steady k-steps, B gaps, first-tile
  wait, row coupling, epilogue, handoff, acc_zero. The wall time is the slowest of a column's 3 cores. The composition
  reproduces the measured g11 time to 1-2%.
- `twinfit.py`, `twinspans.py`: refit of the round-6 no-MAC twin timeline with t_ks held fixed, plus the raw spans.
- `sim12.py`: per-block model of r9s with an A-port term, built on the measured g11 op walls. It fits one constant
  (SSM + EW-boundary syncs) and prices every lever.

## 0. Scorecard: revision 4 against round 9

| rev-4 prediction | outcome | verdict |
|---|---|---|
| P20 (c): in-situ k-step 1,940 (1,850-2,100) cycles | **steady state 2,055 in mm_ks + 67 between calls = 2,122** (median of 13,440 records). The main agent's 2,588 is the MEAN, which includes each op's first-tile wait (section 1) | just above the range. Not A-port-bound |
| P20: step -8.2 +- 1.5 ms | -6.4 ms. From the measured k-step, sim11 gives -6.7: the k-step miss is +182 cycles = static +34, gap +42 (67 vs 25), in-situ extra +~80 (201 vs ~145) | the model was right; three small terms were each optimistic |
| P26 + P28 (SSM tables): -2.9 ms | -2.8 ms | confirmed |
| bank rule reproduces the one-core bench | (a): 2,127 modelled vs 2,200 measured | confirmed (a); the mm_ks bench (2,200) runs SLOWER than in situ (2,055): its buffers share banks |
| sim11 ceiling 35.7 ms | sim12: 34.7 ms (R_A 7.6) to 36.9 ms (R_A 5.5), section 4 | about the same, and now on measured GEMM walls |

Method rule from this round: **report the median and the distribution of a stamp, never only its mean.** One slow tile
per op (the first) moved the mean by 25-60%. A mean that happens to equal a plausible hardware constant (10,368 B / 4 =
2,592) is a coincidence until the distribution says otherwise.

## 1. The "A-port bound": post-mortem and reconciliation

### 1.1 What the records say (`r10/profdec_g11sp.out`, `profdist_g11sp.out`)
Every GEMM op runs its k-steps at **2,051-2,065 cycles** inside mm_ks (p10 to median), whatever K or nt. One tile per core
per op is slow: **the op's first tile** (drain-order records 0, nt, 2nt: tile 0 of each of the 3 rows).

| op (K, nt) | median cyc/ks | first tile cyc/ks | first-tile excess | mean cyc/ks (main's number) |
|---|---|---|---|---|
| down_a (1536, 4) | 2,055 | 4,164 | 28.2 us | 2,591 |
| down_b, mlp3_a/b (1536, 4) | 2,055-2,062 | ~4,600 | 33.6-34.2 us | 2,695-2,705 |
| out, out_proj (768, 4) | 2,052 | 6,567-7,401 | 32.8-33.5 us | 3,291-3,319 |
| yx (768, 8) | 2,053 / 2,860 coupled | 7,225 | 34.3 us | 3,039 |
| qkv (768, 12) | 2,054 / 2,350 coupled | 7,291 | 34.3 us | 2,667 |
| up_a, up_b, mlp0 (768, 16) | 2,051-2,058 | (records incomplete) | | 2,057-2,376 |

- The first-tile excess is the op-start A-slot wait that P24 measured (27-34 us). Round 9 moved the A-ring acquire
  into the kernel (P20 (b)), so that wait now sits inside the stamped k-step time instead of before it.
- 2,588 = 2,055 + 28.2 us x 1,810 cycles/us / 96 k-steps: the mean of 95 fast k-steps and one long wait. It matched
  10,368 / 4 by coincidence.
- **The K = 768, nt = 4 ops' "3.1 B/cycle" is the same ~33 us wait spread over 48 k-steps instead of 96**: 2,052 + 33.5 x
  1,810 / 48 = 3,315 (measured mean 3,319). No rate is involved.
- **Independent agreement:** the main agent's round-10 early note (RESULTS_FROM_MAIN, `npu-tapnext/r10/gdist.py` on the
  same records) reached the same verdict: medians 2,050-2,065, wall 2,130 with the B acquire, first-tile waits 30-40 us,
  DDR-bound slot load ~58 GB/s. It also has a full-step r9s timeline (block 5). sim12's one fitted constant (SSM + EW
  syncs, 873 us per block) and its norm figures (r8 stamps, 541 us) should be refit from that timeline (P33).
- The same artefact runs through the round: r9k's steady state is 2,594 and r9r's 2,484. Each compute change moved the
  steady state, which a port-bound loop would not do.

### 1.2 What does deliver A at ~4 B/cycle: the DDR -> memtile slot load
The A slot (248,832 B per column; K = 768 ops load their 124 KB twice, P5a) is loaded by one shim channel per column
while all 8 columns load at once. That is 1.99 MB in ~33 us from mm_ks entry (~36 us from the previous op's last send):
**55-60 GB/s, the DDR read peak**. Per column it is 7 GB/s = **4.1 B/cycle**, the second "4 B/cycle" of this round. The
op-start wait is DDR-bound, not port-bound. That changes how P5 is priced (§2.7).

### 1.3 Reconciling with the no-MAC twin (rev 2's >= 7.6 B/cycle)
There is no contradiction left: the steady k-step needs 10,368 / 2,055 = 5.0 B/cycle, and the twin showed more than that.
- `r10/twinspans.out`: in `full12p1n` the span up_b -> down_a is 99.5 us. In that span every core received 96 k-steps of
  A (995 KB) **plus** the op-start slot load (28-34 us today). So the A path delivered >= 995 KB / 99.5 us = **5.5
  B/cycle as a hard bound**, and ~7.7 if the wait was the same as today. This bound does not depend on any model.
- `r10/twinfit.out`: holding t_ks at the 4 B/cycle value (1.43 us) forces t_tile to -2.7 us (unphysical) and doubles the
  fit's rms. The twin really did move A at ~7.6 B/cycle with the core idle.
- Rev 2's caveat about the round-6 down_a span (3 of 4 tiles) was about a different number. The twin itself stands.

### 1.4 The candidate causes, one by one

| candidate | verdict | evidence |
|---|---|---|
| MAC kernel's pop pattern (A popped in bursts at the stripe start and mid-stripe) | not binding | the steady state is the same for every op; A needs 5.0 B/cycle and the path gives >= 5.5 |
| stream switch width | >= 5.5 B/cycle per port (hard), ~7.6 (fit): more than 32-bit at 1 word/cycle. aie2p's width is not on disk | [MEAS twin]; [DOC] data memory and core load/store bus 256 bit |
| memtile MM2S rate per channel (one channel broadcasts A to 3 cores) | same bound as above | the twin used the same broadcast |
| lock round-trips per 1,152 B buffer | not binding | 9 per k-step; the stamps show 51 cycles from acc_zero to the first k-step |
| DMA-core bank collisions | **real, and part of the 201 cycles** mm_ks runs above its static 1,854 (1,830 + 24 modelled bank). Not yet split | P38 measures it |
| 2-slot ring (the kernel holds <= 2 of 3 slots) | lead >= 1 slot = 1,152 B = 152 cycles at 7.6 B/cycle against a 228-cycle stripe. It would bind only if R_A < ~5 | [INF] |
| **first-tile A-slot wait inside the stamped region** | **the cause of the 2,588 / 3,300 means** | §1.1 |

**Open:** the steady state is 5.0 B/cycle. The idle-core twin proves >= 5.5, but nothing yet proves the DMA keeps 5.5 while
mm_ks hammers the banks. So 2,055 could still be marginally A-bound in situ. P38 (stall counters in the PROF record:
core LOCK_STALL, MEMORY_STALL; memory-module S2MM STREAM_STARVATION, MEMORY_BACKPRESSURE) settles it in one PROF run.
P34 (an A-only twin of mm_ks) measures R_A with today's lock protocol. Both are pre-registered in PROPOSALS.

### 1.5 The A-port term (sim12)
    t_ks = max(C_ks, A_ks / R_A, B_ks / R_B) + gap,   C_ks = 2,055, gap = 67, A_ks = 10,368 B, B_ks = 4,608 B
    R_A = 7.6 B/cycle (twin fit; hard bound 5.5; in situ >= 5.03), R_B ~7.6 (B broadcast across the 8 columns of a row)
- A floor = 1,364 cycles at 7.6 B/cycle, or **1,885 at the hard bound**.
- Today: t_ks = 2,122 cycles = 1.172 us. The A term binds only once compute drops below ~1,364-1,885.
- sim12 reproduces 58.10 ms (measured 58.14) with one fitted constant: SSM + EW-boundary syncs = 873 us per block, so ~14
  us per boundary if the SSM is ~790 in situ.

## 2. Ways to deliver A faster

Two layers are separate, and the more important one today is the first:
- **(i) DDR -> memtile.** The op-start slot load, ~33 us x 11 ops = 365 us per block = **4.4 ms per step**, DDR-bound.
- **(ii) memtile -> core.** Not binding at 2,055 cycles; binds below ~1,364-1,885.

Every option below states which layer it acts on. Predictions are sim12 at depth 12, each option alone.

### 2.1 A split across the core's two S2MM channels (layer ii)
- **Mechanism.** Both core S2MM channels carry A, e.g. stripes' row block 0 on channel 0 and row block 1 on channel 1:
  two memtile MM2S broadcasts per column. B shares channel 1 in **S2MM out-of-order mode**.
  - [DOC] `MEMORY_MODULE_DMA_S2MM_0_CTRL.Enable_Out_of_Order`, `XAie_DmaSetOutofOrderBdId`: the packet header selects the BD,
    so one channel can land two packet streams in two rings.
- **Rate.** The port floor goes from A_ks / R_A = 1,364 to (A_ks + B_ks) / (2 R) = 985 cycles at 7.6 B/cycle.
- **Budget.**
  - Memtile: +1 MM2S per column (A uses 1 today and B's forward 1 on memtiles 0-2: 3 of 6), +2 BDs.
  - Core: +3 BDs (16 available, ~7 used), +2 locks (6 used of 16).
  - Packet routes: B's forward is an IRON ObjectFifo, so it must become a PacketFlow with hand-built rings (SKILL §2:
    ObjectFifo(packet) on a PacketFlow channel is refused), and out-of-order BDs are not exposed by IRON 1.4.4 (raw
    `aie.dma_bd` needed).
- **Program memory.** mm_ks: two A cursors and two acq / rel pairs per stripe, ~+40 B; records +2 words per stripe.
- **Risks.** High. It is new IRON territory (out-of-order S2MM), and a mis-set header lands B in A's ring.
- **Prediction.** **0 ms today.** At compute 1,200 it gives -1.4 ms beyond the single-channel A-bound case (50.8 vs 52.2
  ms alone). Build only after P38 / P34 show R_A < ~6.5 and a k-step below ~1,600 exists.

### 2.2 Broadcast from the memtile vs per-core streams (layer ii)
- **Mechanism.** 3 MM2S per column, one per core, each sending the same slot. The `full` lock is released with 3 and
  acquired once per channel; `empty` is acquired with 3.
- **Rate.** The same per-core port rate. The memtile reads 3x the bytes (3 x 5 B/cycle: no limit known to bind).
- **Budget.** Memtile MM2S 4 of 6 on memtiles 0-2, +4 BDs per column (in the even/odd channel halves), lock values <= 63.
- **What it changes.** It removes the A broadcast's coupling of the 3 rows *within* a tile run, but the op still ends when
  the slowest row ends, and the next op's slot load still waits for every channel's last send. **Row imbalance remains
  the cost; balance (P23) is the fix, not decoupling.**
- **Prediction.** **~0 ms alone, and -1.4 ms only together with P23** (which gives the same -1.4 ms alone). Not worth
  building for speed. It is worth it later as an enabler: per-core A lets cores take different row ranges (§2.5).

### 2.3 A read from a neighbour core's memory (layer ii)
- **Geometry.** Rows 2, 3 and 4 of a column all reach row 3's memory: row 2 via N (0x60000), row 4 via S (0x40000) [DOC TM].
- **Mechanism.** A lands once in the middle core's L1, which can use **both** of its S2MM channels for A. Rows 2 and 4 pop A
  through their own load units, at 64 B per load: no stream limit. Row 3's B arrives via row 2's freed S2MM and row 3
  pops it from row 2's memory.
- **Rate.** A per column up to 2 x 7.6 = 15 B/cycle: an A floor of ~680 cycles per k-step.
- **Budget.**
  - Rows 2 and 4 drop their A rings (3.4 KB of L1 each, **freed**).
  - Row 3 needs a deeper ring (3 consumers' jitter): 6 slots = 6.9 KB (+3.5 KB, but row 3's L1 is full: needs P37's
    placement work or the C object shrunk).
  - Locks: the A ring's locks live in row 3. The other cores acquire neighbour locks: the lock-id offset differs per
    direction, so gen_ks takes it as a parameter.
- **Risks.**
  - The bank rule with 3 cores' accesses in one memory module is unmeasured; the A pops of 3 cores co-issue with row 3's
    piece accesses.
  - A 3-consumer ring is a release-count protocol: the slot is free after 3 releases, which a semaphore lock does
    naturally (acquire >= 1 per consumer, the DMA acquires the "free" count 3).
  - The neighbour lock ids are unverified in asm.
- **Program memory.** ~+100 B.
- **Prediction.** 0 ms today. It is the most powerful layer-ii option (2x the A supply plus 3.4 KB freed on 2 of 3 rows).
  Pre-register only after a one-column bench (P39).

### 2.4 Larger N tiles: more MACs per A byte (layer ii)
- A per MAC is proportional to 1/n and B per MAC to 1/m. With m = 144 fixed by the 8 x 144-row column layout, n = 128 needs
  144 x 128 x 4 = 72 KB of accumulators: more than L1.
- The GEMM core has **~0.9 KB of L1 free** today (data region 100 B, bank 1 192 B, bank 3 640 B; `ldScripts_main_core_0_2`,
  llvm-size).
- n = 80 needs +9 KB of accumulators. **Not feasible without freeing L1 (§2.3 frees 3.4 KB on two rows only).**
- Prediction: none. Parked.

### 2.5 Different 3-row / column work splits (layer ii)
Per k-step bytes per core for accumulator budgets of 36 KB (m x n = 9,216):

| split | A per core | B per core | port floor at 7.6 B/cycle | fits the 1152-row x 8-column layout? |
|---|---|---|---|---|
| today: m 144 (A broadcast per column), n 64 | 10,368 | 4,608 | A 1,364 | yes |
| m 96, n 96 | 6,912 | 6,912 | 909 | no: 1152 / 96 = 12 row groups across 8 columns |
| m 72, n 128 | 5,184 | 9,216 | B 1,213 | 16 row groups = 2 per column with per-core A (§2.2), but then 3 cores do not divide 2 groups |
| m 48 per core (A per core), n 192 | 3,456 | 13,824 | B 1,819 | B-bound; worse |

- Every split that lowers A's floor either breaks the 8 x 144 layout or moves the bound to B. Today's split is the right
  one while 3 GEMM rows share a column.
- A 4th GEMM row (EW moved off row 5) would allow m 72 x 2 groups x n 128: **parked** until the EW work shrinks (P7).

### 2.6 Keeping A resident across N tiles
- Per core, A is 144 x K: 124 KB (K 768) or 248 KB (K 1536). It cannot stay in a 64 KB L1. Residency across N tiles *is*
  the memtile slot (A is re-streamed from it per N tile). Nothing to gain at layer ii.
- Residency across **ops** is the real lever, and it is layer i:
  - **up_a and up_b read the same A** (identical TAP at 11,064,320 in `full12r9s.sched`). Today up_b reloads the whole 248
    KB slot (~33 us). Keeping the slot gives that back: **P5r**, part of P5a below.
  - There is no room for a second 248 KB slot: the memtile holds 376 KB of 512 (slot 248, B forward 18, C join 110).
    Ping-pong only works at k-chunk granularity (P5).

### 2.7 Layer i: the op-start slot load (the lever that pays today)

| option | mechanism | prediction (sim12) |
|---|---|---|
| **P5a** | K = 768 ops load their 124 KB once (today the TAP's first dim has stride 0 and loads it twice): the wait halves (DDR-bound) | **-1.6 ms with P5r**; g11 first waits K = 768: 33 -> ~17 us |
| **P5r** | up_b skips its load: same TAP as up_a (the memtile load BD chain gets no load entry for up_b; its sends re-run) | in P5a's -1.6 (-0.4 of it) |
| **P5** | slot streamed in k-chunks (Bd iteration + a lock release per chunk): the first tile runs at max(compute, DDR) instead of load + compute | **-3.3 ms** with P5a / P5r (K = 768 first wait ~3 us, K = 1536 ~6 us) |

The first tile's demand (8 columns x 10,368 B per 1.17 us = 71 GB/s) exceeds DDR. So a chunked first tile runs at DDR
speed, not compute speed: this is why P5's gain is 3.3 ms, not the 4.4 ms of waits.

## 3. EW-core memory: a plan that makes room for P27

### 3.1 The EW core's 64 KB today (full12r9s, tile 0,5; ldScripts + llvm-size + llvm-nm + `llvm-readelf --stack-sizes`)

| region | bytes | bank(s) | note |
|---|---|---|---|
| stack | 6,144 | 0 | **max call path 4,224 B**: core 64 + ew_op 64 + ew_norm 128 + norm_rows<LN> 2,944 + ln_stats 1,024. The SSM path is 768 |
| ewo0 (out ring, 12 x 768) | 9,216 | 0 | |
| ssqrt_ab | 1,024 | 0 (IRON free tail) | |
| .data: ssig_ab | 2,560 | 1 | |
| .bss: g_arena 22,016 (SSM needs 21,568), g_panels 1,024, g_ptmp 1,024, globals | 24,128 | 1-2 | 448 B spare inside the arena |
| **data region tail** | **960 free** | 2 | 0x7A840-0x7ABFF |
| .aie.bank2: ssig_cd | 2,560 | 2 | |
| **bank-2 region tail** | **2,560 free** | 2 | 0x7B600-0x7BFFF: the region is 5,120 B, half used |
| ewi0 (in ring, 20 x 768) | 15,360 | 3 | |
| ssqrt_cd | 1,024 | 3 | |
| program memory | 16,240 of 16,384 | | **144 B free** |

The round-9 note "no 2 x 1.7 KB left" holds for two copies *in two banks*. One copy fits today, and the stack holds 1.9 KB
it never uses.

### 3.2 The plan, in order (each step separately measurable)
1. **P27 with ONE copy of A1 (1,696 B) in the bank-2 tail** (`.aie.bank2`; 2,560 B free). Both LUT pointers point at it. The
   round-9 probe measured one-bank tables at +8 cycles per 32 lanes, about +15 us per block (216 chunks x 16 vectors x 8 /
   1.81 GHz).
   - No other memory change.
   - **Program memory:** the fused a-pass replaces `loga_p` (272 B) + `exp_pA..D` (1,456 B) with one table pass (~600-800 B).
     That frees ~1 KB of text, which the core needs (144 B free).
   - Prediction: **-0.5 +- 0.2 ms**.
2. **Stack 6,144 -> 4,416 B** (the measured max path 4,224 + 192 margin). That frees 1,728 B at the end of bank 0 (IRON
   places ewo0 right after the stack, so the bank-0 free tail grows from 1,024 to ~2,752 B). A1's second copy goes in
   `.aie.bank0` beside ssqrt_ab (1,024 + 1,696 = 2,720).
   - Check `input_with_addresses.mlir` for ewo0's new address and the ldscript's bank0 LENGTH before dispatching.
   - Prediction: a further **-0.2 ms** (two banks).
   - Risk: low. The stack sizes come from the linked elf. Re-run `llvm-readelf --stack-sizes` after every EW change and
     assert max path + 128 <= stack in bb.sh.
3. **Shrink norm_rows<LN>'s 2,944 B frame and ln_stats' 1,024 B** (for the next tables: N2 / P29, and the rings):
   - ln_stats takes A32 by value and returns B32 / A32 through references: noinline + minsize puts every one on the
     stack. Pass pointers into one static 512 B scratch (g_ptmp is free during norms).
   - Move pass 2's per-cb LN math into a noinline function per k-step (its live set: nb0/nb1, r0/rl0/r1/rl1, ga, be: ~10
     vectors) so norm_rows keeps only s0/s1/q0/q1 live.
   - Target: norm path <= 1.8 KB -> stack 2 KB, **+2.4 KB more in bank 0** (4.1 KB in all vs today).
   - Measure norm time per op before / after: 124-148 us each, so +5% would cost +0.3 ms per step.
4. Later, only if needed: out ring 12 -> 8 slots (-3 KB) or in ring 20 -> 16 (-3 KB). The norms are input-port-bound,
   so measure their op time against ring depth first.

## 4. The ceiling (sim12, with the A-port term)

Levers alone (depth 12, from 58.1 ms; `r10/sim12_r9s.out`):

| lever | ms |
|---|---|
| P7 full (norms folded) | **-6.8** (conservative -4.6) |
| attention 930 -> 600 us | -4.0 |
| SSM chains hand-scheduled (790 -> ~500) | -3.5 |
| P5 (P5a + P5r + k-chunks) | **-3.3** (P5a + P5r alone -1.6) |
| P23 row / column balance (yx, qkv, mlp3_a) | -1.4 (rev 4 said -0.7: the r9s walls show qkv's slow row costs 80 us per block) |
| P37 DMA-collision placement (201 -> ~100) | -0.9 (needs P38 first) |
| P27 a by table | -0.7 (-0.5 with one copy) |
| P24b acc_zero folded | -0.5 |
| P35 B locks inside mm_ks, one call per tile | -0.5 |
| k-step compute 1,600 (deep-K / 2x2; no L1 for it today) | -3.9 at R_A 7.6, **-1.5 at R_A 5.5** |

Combined: P5 54.8 -> + P7 48.0 -> + P23 46.5 -> + P35, P24b 45.5 -> + P27 44.8 -> + attention 40.9 (**ceiling with today's
k-step**) -> + SSM hand-scheduled 38.1 -> + P37 37.2 -> + k-step 1,600 **34.7 ms (28.8 fps)**, or 36.9 if R_A is only 5.5.
Depth 10: 29.0 ms, depth 8: 23.3.

**P14 (two frames in flight)**, re-priced as a list bound (each frame >= the busier side + 15% of the other):
- Today: GEMM side 3,386 us, EW side 1,414 us per block -> **~43.7 ms per frame (22.9 fps, 1.33x), latency ~87 ms**.
- After P5 + P7 + P23 the EW side shrinks to 817 us: 38.2 ms per frame (26.2 fps), against 46.5 serial: 1.22x.
- At the ceiling: 29.4 vs 34.7 ms (1.18x), latency ~59 ms.
- P14's value falls as P7 and the SSM work land. It doubles latency (bad for a tracker's feel). Keep it as the last fps
  lever, after P7.

**P7 re-priced:** -6.8 ms full (no norm ops; 4 EW boundaries gone; +0.3 us per tile of epilogue on 92 tiles), -4.6
conservative (40 us stats passes). This is unchanged from rev 4 in size; it is now the largest single lever. P7a
(numerics emulation on acts_clip3) gates it.

Critical path at the ceiling, per block: k-steps 1,154 us, attention 600, SSM + syncs 527, tile boundaries + epilogues 436,
op starts 136. Still serial; the GEMM side is 82% of it.

## 5. What still stands from revision 4
- The bank rule (exact), the conflict-free piece layout (24 modelled bank cycles in situ: confirmed small), the SSM table
  numerics.
- Compute, not DDR, bounds the step overall (~22 GB/s average), **but the op-start slot loads are DDR-bound bursts** (55-60
  GB/s for ~33 us, 11 times per block).
- Compose from the binary, fit at most one constant, pre-register. Added: read distributions, not means.
