# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 15, 2026-10-11, research pass 15, written during main's round 19.

- **Base:** `full12r19d` (main, round 19) = full12r18m + ROT g0a + ZHI + CSPLIT.
  - full12r18m measured 35.20 ms (sweep median) at depth 12.
  - The round-19 A/B deltas are −0.18, −0.15 and −0.25, so r19d ≈ 34.6.
- **Data:** main's r18m and r19d timelines and TPROF (RESULTS_FROM_MAIN round 19), KS-P38 and NIO2 (round 18), and the deployed
  `full12r19b.bin` / elfs (the CP analysis; r19d's sequence differs only by CSPLIT's core-side BDs).
- **Archive:** revision 14 is in `rev14/`; older ones are in `rev13/` .. `rev1/`.
- **Prices:** deltas at depth 12, Turbo, 1.81 GHz. 1 cycle per GEMM k-step = 8.59 µs per step.
  - Round 19 measured that k-step static cuts land at **~0.75** of their static price (g0a −0.18 for 26 cycles, g1a −0.11 for 15).
    Rev 15 prices k-step changes with that factor.

## Headlines (rev 15)

1. **Where the 35 ms go now** (r18m timeline + `r20/tilefit.py`): GEMM 19.9 ms, attention 6.7, norms 5.0, SSM 3.5.
   - GEMM k-step compute is only 13.4–13.7 ms. The **~6.3 ms of per-op overhead** splits into:

     | part | ms per step | what |
     |---|---|---|
     | dependent-op start-up | ~3.3 | ~33–38 µs per op whose A is the previous op's output (up_a ~50) |
     | per-tile overhead | ~2.2 | epilogue + C-object hand-over (main's TPROF: hand 3,010 = C-drain wait 2,330 + zero / loop; now CSPLIT + ZHI) |
     | residual | ~0.8 | |

   - The ops whose A was ready early (down_b, mlp3_b, and up_b by reuse) show **no** start-up.
   - Op boundaries cost **~5 ms per step** overall: GEMM 3.3, attention 1.2, norms 0.8.
2. **The op-boundary start-up looks command-processor-bound** (strong circumstantial case, NOT measured).
   - **The pattern:**
     - After every token-wait run, the runtime sequence executes 92–144 CP instructions before the dependent op's A path is armed:
       the shim A fills, then the memtile slot loads, then the memtile → core sends.
     - The drains, the memtile tasks and (after norms) the B fills all sit behind the sync.
   - **The fit:** with **t_op ≈ 0.17 µs per CP instruction** (0.13–0.22), plus the A-bound first tile and ~3 µs of DMA latency, the
     model reproduces the measured boundary of yx, out, qkv, out_proj, mlp0, mlp3_a and the norms within ~5 µs. up_a is +20 µs.
   - **The lever, SEQPRE:** issue everything that does not read the producer's output BEFORE the wait. That leaves 36–40
     instructions after it (`r20/cpseq.py`).
   - **Price:** −1.37 / −1.79 / −2.31 ms at t_op 0.13 / 0.17 / 0.22, so **−1.4 ± 0.5 ms** after realisation.
   - **Gate:** the CPRATE probe (a few lines in design8, one A/B run) measures t_op first.
3. **GEMM L1 layout vs DMA collisions** (`r20/collide.py`, a time-resolved model): **dropped.**
   - The model fits KS-P38's 167.5 with α = 0.72, the same per-write core loss DMAB measured directly (0.68–0.75). So the model
     holds (split: A 75, B 92).
   - **B objects in different banks is infeasible.** B is popped in 72 of 74 cycles, so a bank holding a popped B object can hold
     no hot data. Two B banks would need 23 KB of cold data; the GEMM core has 14.6 KB.
   - **A slots:** the best feasible assignment (slots in banks 1 / 2 / 0) saves 15 ± 10 cycles (≈ 0.1 ms after realisation), and
     it needs bank-0 room that the 28 KB data region does not leave.
   - The bound, A slots in a bank no piece uses, is −48 cycles (−0.31 ms). Capacity forbids it: 36 KB of accumulator pieces
     cannot fit in two banks.
   - **E7' as posed is void.** ROT's 72 + g bound is the single B stream's ldb slot and its bank together. Splitting B over banks
     needs B on TWO FIFOs (E7'', research item R2).
4. **ROT g0f / g1f:**
   - **Done by main** (g0a deployed, −0.18; g1a −0.11) with the probes PWRAP / PADDA. The 2D post-modify ADDS to the line step;
     padda must sit one bundle before the fill.
   - The pass-15 generator (`r20/gen_rot.py`) has the lock gaps as hard floors: ACQ_GAP 6, REL_GAP 8. Its replay rejects
     round 18's g2f ("A fill 1 cycle(s) after its acquire").
   - Its independent re-solves reproduce main's statics (g0a 1,357, g1a 1,368, OPTIMAL). They are different schedules with the
     same R; main's are deployed.
5. **`r20/rotdb.py` is the DMA-fed A ring bench main asked for.**
   - The three A slots are IRON Buffers fed by a shim → core flow. The S2MM BD chain acquires aF (lock 4) and releases aU (lock 5)
     per 1,152 B object, so every acquire waits on real data.
   - Built (CPU) for g0a and for the g2f negative control. Not run.
6. **asmlint** has three new rules:
   - **LOCK-ACQ-GAP** (error, measured r18);
   - **LOCK-REL-GAP** (warning, unverified threshold);
   - **IMM-SRC** (error, measured r18): immediates checked on the SOURCE, because `add #64` → `#-64` is invisible in a disassembly.

   The historic suite passes, with g2f, g2a and an add64 snippet added. full12r19b is clean.

7. **Norms (main's round-19 walls):** 3-input norms run at 3.8 B/cycle. That is ONE shim MM2S channel, which feeds the EW core
   (shared with the A fills), so the rev-13 fit's R_in 5.75 is refuted.
   - The lever is a second input channel (NIN2: shim MM2S 1 is free on columns 3–7 and B's on 0–2; the EW core's S2MM 1 is free)
     together with N-IO-ASM: **−0.7..−1.5 ms**, depending on the unmeasured DDR ceiling.
   - MACHINE's "DDR 55–63 GB/s peak" was 8 × the single-channel rate; today's norms already move ≥ 84 GB/s.
8. **The C object:** moving the accumulators out is infeasible (+12,288 B needed, 576 B free). After main's CSPLIT, a progressive
   4-part release with rb-major epilogues (CSPLIT4) hides most of the remaining 1,555-cycle wait: −0.4..−0.5.

## Ranking (rev 15; base full12r19d ≈ 34.6 ms = r19b + ZHI −0.15 + CSPLIT −0.25, main round 19)

| rank | id | what | cost | risk | expected (depth 12) | pre-registered check |
|---|---|---|---|---|---|---|
| 0 | **CPRATE** | probe: N dummy CP writes after one dependent op's sync (recipe below) | S (1 build, 1 A/B) | none | measures t_op | Δstep = 12 · N · t_op; N = 400 → +0.6..+1.1 ms if t_op is 0.13–0.22 µs |
| 1 | **SEQPRE** | design8 sequence: pre-issue a dependent op's memtile slot loads + sends, drains, B fills, EW params fills before its producer wait; norms pushed to `pending` instead of `finish()`ed at once | M (design8 sequence code only) | medium (queue / BD budgets, ordering) | **−1.4 ± 0.5** (cpseq: −1.37..−2.31 before realisation) | regress BIT-IDENTICAL; timeline gaps: yx / out / qkv / out_proj / mlp0 / mlp3_a −10..−18 µs each, norms −6..−9 µs each |
| 0' | **DDRCAP** | probe: the NPU's aggregate DDR ceiling with 2 shim MM2S channels per column streaming (+ the 8 drains) | S | none | gates rank 2's price | aggregate read GB/s; today's 3-input norms already move ≥ 84 GB/s (in + out) |
| 2 | **NIN2 + N-IO-ASM** | the norms' inputs over TWO shim channels (a second packet flow into the EW core's free S2MM 1; B on shims 0–2 becomes a hand-built packet ring) + the norm unit loops in asm | L | medium | **−0.7..−1.5** (DDR-ceiling dependent; each alone −0.26..−0.52) | per-unit cycles below; norms' timeline share −0.7..−1.5 |
| 3 | **CSPLIT4** | after main's CSPLIT + EPLR (plain epilogues release part 1 early, −0.12, main r19): the gated / gelu_pan / qkv epilogues rb-major + part 1 released in 4 sub-parts as written | M | medium | −0.3..−0.35 (the plain share went to EPLR) | TPROF hand after the epilogue: plain ≤ 900, gated / gelu_pan / qkv ≤ 750 (today ~2,240 = 1,555 + zero / loop) |
| 4 | attention S3 / K1 / K2 | rev 10 / 11 prices; GEMM text has ~0.3–0.8 KB free | M | low | ~−0.5 (±30 %) | rev 11 checks |
| 5 | **DAPRE** | down_a's A load during up_b (the only non-dependent op still paying; overhead 0.52 ms vs down_b 0.26) | S–M | medium | −0.2..−0.3 | down_a gap → ~0, per-tile period ≈ down_b's |
| 6 | SSM ssq2 / FSX | rev 13 | M | low | −0.11 / −0.07 | rev 13 |
| 7 | LATE-C (R3) | C's 12 pieces = the rows processed LAST (pair 0 of rb 6–17); the tile's first k-step loads zeros (no acc_zero); conditional C acquire inside mm_rot at rb 6 | L (gen_rot op + record flag) | medium | −0.3 on top of CSPLIT4 | |
| — | GEMM A-slot layout | best feasible slot banks (1, 2, 0) | M (IRON data region) | medium | ≤ −0.1 | dropped |
| — | B objects in two banks | | | | infeasible (capacity proof above) | dropped |
| — | accumulators out of the C object | main's request | | | infeasible: +12,288 B needed, ~576 B free (below) | dropped; CSPLIT4 / LATE-C instead |

Ceiling from ≈ 34.6, stacked:

| step | ms |
|---|---|
| SEQPRE −1.4 | 33.2 |
| NIN2 + N-IO-ASM −1.0 | 32.2 |
| CSPLIT4 −0.3 (after EPLR −0.12) | 31.8 |
| attention −0.5 | 31.25 |
| DAPRE + SSM −0.4 | 30.85 |
| LATE-C −0.3 | **~30.5** |

- **Beyond that, structural levers:**
  - An activation-stationary DDR bypass for dependent A: the remaining ~7–10 µs per boundary plus the A-bound first tile,
    ~−1.0 ms.
  - E7'' (B on two FIFOs): −0.9 ms realised, if feasible.
  - The B-collision floor (~92 cycles per k-step ≈ 0.8 ms) has no lever.
- **Floor ≈ 28.5–29.5 ms.** Stacked prices below 33 carry ±1 ms.

## CPRATE — measure the command processor's cost per instruction (rank 0)

design8 `sequence()`, GEMM branch, right after `finish_deps(op)`, behind an env flag:
```python
if os.environ.get("CPRATE") and op["name"].endswith("_mlp3_a"):
    from aie.dialects.aiex import npu_write32
    for i in range(int(os.environ["CPRATE"])):  # harmless: Performance_Control0 of GEMM core (i % 8, 2) = 0 (no events)
        npu_write32(address=0x31500, value=0, column=i % 8, row=2)
```
- **Build and time:** build with CPRATE=400, then A/B vs r19b (r18/ab.py, 20 × 20).
- **Pre-registered:**
  - Δstep = 12 × 400 × t_op = **+0.62 / +0.82 / +1.06 ms** for t_op 0.13 / 0.17 / 0.22 µs.
  - mlp3_a's gap in the timeline grows by 400 · t_op (52–88 µs).
  - Regress BIT-IDENTICAL.
- **If Δ < 0.15 ms** (t_op < 0.03 µs), the CP hypothesis is wrong and SEQPRE is worth ≤ 0.3 ms. The boundary would then be a DMA /
  DDR latency, and the next check is a TPROF `first` stamp on the dependent ops' first tiles.
- **Optional:** the same with `npu_maskpoll32` on that register (value 0, mask 0: always satisfied) prices a poll. maskpoll32 is
  42 of every ~158 CP instructions.

## SEQPRE — the dependent op's independent work before the wait (rank 1)

**Today** (`design8.py sequence()`, 2026-10-11):
- **GEMM:** P43's B fill (pre-wait), then `finish_deps(op)` → `fill_a` × 8 → `a_loads` (memtile S2MM tasks) → `a_sends` (memtile
  MM2S tasks t0 / mid / last) → drains → `push`.
- **Norm:** `finish_all()` → per column: params fill, input fills, drain → `tg.finish()` at once. This is a full sync, so the
  NEXT op's whole issue (B fills included) sits behind the norm's completion.
- **Attention:** `finish_deps` → `fill_a` → `a_loads` → per pass: B (K / V) fills, `a_sends` (pass 0), drains.

**After SEQPRE:**
1. **GEMM:** move `a_loads(op)`, `a_sends(op)` and the drain loop BEFORE `finish_deps(op)`; keep `fill_a` after it. Order the
   post-wait shim fills column-major, the critical path's order. The memtile tasks only arm lock-governed chunk transfers:
   - op k + 1's slot load waits on `E[h]`, which op k's `last` send releases after the core has taken chunk h for its last tile;
   - its data arrives only after `fill_a`.

   So moving them is data-safe by the locks (check: the lowered `aiex.dma_start_task` order, then one dispatch with a 3 s
   timeout). The memtile MM2S queue can then hold op k's t0 / mid / last plus op k + 1's three (6 > 4). A full queue makes the CP
   poll until op k's sends finish. They finish before op k's drains complete, so there is no deadlock, only the wait it already
   had.
2. **Norm:** emit its params fill and its drains, then `finish_deps` on its input producers (not `finish_all`), then the input
   fills. Push the group to `pending` like a GEMM's, instead of `tg.finish()`. The following GEMM's B / memtile / drain work then
   issues while the norm runs.

   Mind the EW ring order: params must stay the first bytes of the op in the ring. The packet BD fills slots in push order, so
   push params first.
3. **Attention:** `a_loads`, `a_sends` and the drains before `finish_deps`. Post-wait order: `fill_a` × 8, pass 0's K / V fills,
   then later passes' fills.
4. **BD accounting:** run design8's `bd_peak` print. More tasks are live across a wait, and the limit is 16 per shim. If it
   overflows, keep the drains behind the wait; they are not on the start path.

- **Price** (`r20/cpseq.py` on full12r19b.bin): post-sync CP instructions on the A path, per block:

  | op | instructions saved |
  |---|---|
  | yx | 72 |
  | ssm0 | 72 |
  | out | 72 |
  | norm1, norm2, norm3, norm0 | 52 each |
  | up_a | 56 |
  | qkv | 83 |
  | attn | 80 |
  | out_proj | 60 |
  | mlp0 | 72 |
  | mlp3_a | 108 |

  That is 10,515 instructions per step: −1.37 / −1.79 / −2.31 ms at t_op 0.13 / 0.17 / 0.22 µs; **−1.4 ± 0.5** after a ~0.75–1.0
  realisation.
- **Pre-registered (timeline, median of 3):**
  - gaps of yx / out / qkv / out_proj / mlp0 −10..−18 µs each;
  - mlp3_a −15..−24;
  - norms −6..−9;
  - down_a / down_b / mlp3_b / up_b unchanged (±3);
  - BIT-IDENTICAL.

## The C object after main's CSPLIT (rank 3 CSPLIT4, rank 7 LATE-C; main's round-19 request)

**Measured (main, TPROF):** `hand` was 3,010 cycles per tile: ~2,330 C-drain wait plus ~680 zero / loop.
- **Note:** this section's first draft (written before those numbers) pre-registered `hand` 1,900–2,700 and `epi` plain
  1,350–1,600. Measured were 3,010 (incl. zero + loop, so the drain part 2,330 lies inside the range) and 1,310 (below). They
  are not claimed as hits.
- ZHI (−0.15) and CSPLIT (−0.25) are deployed. **Left:** part 1 (12,288 B at 7.9 B/cycle = 1,555 cycles) drains after each
  epilogue. That is 912 C objects per core per step (76 per block), **~0.78 ms**.

**Can the 12,288 B of accumulators leave the C object? No (capacity).** The core 0_2 L1 total:

| item | bytes |
|---|---|
| stack | 2,560 |
| data region | 28,672 (.data 3,964 + .bss 24,616) |
| tbk / tbt / tbc | 192 |
| A slots | 3,456 |
| C object | 18,432 |
| B | 9,216 |
| tbn / tba | 128 |
| .aie.bank3 | 2,304 |
| **total** | **64,960 of 65,536** (576 B free) |

- Moving the accumulators out needs +12,288 B while the C object keeps its 18,432 B of output.
- **The reducible items cover ~3.4 KB:** the stack (~1.5 KB above the 832 B deepest path; stackaudit), CSPLIT's free bytes, and
  `rot_rec` (as .rodata it is already counted).
- **The GELU / erf / exp tables (5.6 KB) are needed** by the epilogues and the attention.
- So nothing short of shrinking the tile hides the drain.

**CSPLIT4 — progressive part-1 release (rank 3):**
- **Today:** the epilogue writes the output piece-major: bias_to_c is three bias_pcs calls (pair 0 rb < 12 in place, pair 0
  rb ≥ 12 → the tail, pair 1 → C + 256 h). C bytes [1 KB rb] are complete only after the third call, so part 1 can only be
  released at the end.
- **Change:**
  1. Make the epilogues rb-major for rb < 12. Per rb: read piece(rb, 0) (in C) and piece(rb, 1) (acc_hi), write the 1 KB of bf16
     at C + 1 KB rb. This is safe in place, because the reads of the rb come before its writes.
  2. Release part 1 in 4 sub-parts of 3 KB (rb 0–2, 3–5, 6–8, 9–11) as a 4-BD chain with 4 lock pairs. This is main's CSPLIT
     patch with more BDs; the memtile join still receives one 18,432 B stream in order.
- **The drain then overlaps the epilogue:**

  | epilogue | today's exposed wait | after CSPLIT4 |
  |---|---|---|
  | plain (1,310) | 1,555 | ~580 |
  | gated (3,263), gelu_pan (4,918) | 1,555 | ~0 |
  | qkv (1,941) | 1,555 | ~65 |

- **Price:** per block per core 32 · 975 + 32 · 1,555 + 12 · 1,490 ≈ 99k cycles, × 12 = 0.66 ms; realised ~0.75 → **−0.4..−0.5**.
  Main's EPLR (round 19, −0.12) has since taken the plain share (32 · 975), which leaves **−0.3..−0.35** for the gated /
  gelu_pan / qkv epilogues.
- **Cost:** every epilogue variant (bias_pcs, bgelu3 / bgelu, gup, pan, vtk) must become rb-major for rb < 12. pan / gup write bfp
  panels: check that their drained layout is rb-ordered. Text: one loop per variant, a few hundred bytes. **Bit-identical by
  construction** (the same values, written in another order).

**LATE-C (rank 7, research R3):**
- **Map:** C holds the accumulators of the rows ROT processes last: pair 0 of rb 6–17. acc_hi holds pair 0 rb 0–5 and pair 1
  of all rbs.
- **First k-step:** a tile's first k-step loads its accumulator quarters from a 256 B zero block instead of the pieces (records
  only: J0 / J1 point p3 at the zero block, no stride). It is bit-identical (0 + Σ) and makes acc_zero unnecessary.
- **C acquire:** mm_rot takes the C lock itself with a second conditional acquire, flagged only at rb 6 of a tile's first k-step.
  That hides ~min(1,555, 6/18 · 1,576 + zeroing) ≈ 775 cycles per object: **−0.3** after CSPLIT4.
- **Cost:** gen_rot gets a CACQ op (alu slot, ACQ_GAP ≥ 6 before the first C store, not load: the first k-step only stores into
  C). The records get a flag, and the epilogue's in-place order moves (the rb ≥ 6 outputs land in C [6 KB, 18 KB)).

## The norm plan after the r19d walls (rank 2; main's round-19 request: re-check the fit)

**Measured (main, r19d timeline, per column):**

| op | kind | inputs | µs | cycles per unit (216 units) |
|---|---|---|---|---|
| norm0 | RMS | 3 | 95.6 | 801 |
| norm1 | RMS | 2 | 67.7 | 567 |
| norm2 | LN | 3 | 98.0 | 821 |
| norm3 | LN | 2 | 82.7 | 693 |

- Every norm's gap is 16.5 µs.
- n1k reads three streams in both kinds, so n2 and n3 compute is the same. **The 3-input ops are input-bound:** 3,072 B / 801 =
  3.84 B/cycle and / 821 = 3.74.

**The rev-13 fit (R_in 5.75) is refuted.**
- **The input is ONE shim MM2S channel per column** (lowered MLIR: `Ein{c}_src` shares shim MM2S 0 with `Ain{c}`). That
  channel's measured rate is ~6.4–7 GB/s = 3.5–3.9 B/cycle (MACHINE §7, memtile ← DDR through one shim channel).
- **What hid it:**
  - The fit had a free per-op constant F and the fitted pass-2 excess s2, and both absorbed the n3 input time.
  - Its one miss was this very signal: n3 − n2 walls 10.7 µs modelled vs 15–19 measured. At NIOC's faster compute the
    difference grew to +28 µs (RMS) / +15 µs (LN).
- **Also refuted, MACHINE §5's "DDR ~55–60 GB/s read, ~63 read + write":** that was 8 × the single-channel rate.
  - Today's 3-input norms move (3,072 + 1,600) B × 8 per 801 cycles ≈ 47 B/cycle ≈ 84 GB/s of reads + writes.
  - The real ceiling is unmeasured (DDRCAP).

**Per-unit model** (cycles, 216 units per op; compute from the n2 walls; N-IO-ASM = pass-1 glue −70, pass-2 glue −115 in situ;
input at 3.8 B/cycle per channel):

| case | RMS n2 (12) | RMS n3 (12) | LN n2 (12) | LN n3 (13) | norms ms per step | Δ |
|---|---|---|---|---|---|---|
| today | 567 | 808 | 693 | 808 | 4.21 (+ 0.8 gaps: the 4.88 measured) | — |
| N-IO-ASM only | 539 (input) | 808 | 539 (input) | 808 | 3.95 | −0.26 |
| NIN2 only | 567 | 567 | 693 | 693 | 3.69 | −0.52 |
| both | 382 | 404 | 508 | 508 | 2.64 | −1.57 |
| both, DDR capped at 100 GB/s | 529 | 677 | 529 | 677 | 3.53 | −0.68 |

So the price is **−0.7..−1.5**, set by DDRCAP. SEQPRE takes ~−0.3 of the 0.8 ms of norm gaps on top (in rank 1's price).

**NIN2 design:**
- **Stream:** `Ein2[c] = PacketFlow(4, shims[c], ew_tiles[c], src_channel=1, dst_channel=1)`, a second input ring (e.g. 8 slots;
  `ewi0` 20 → 12 slots to pay for it) with locks 4 / 5, and a TileDma S2MM 1 chain like `ewi`'s.
- **Split:** the plan puts stream 0 (and the params) on ring 1 and streams 1–2 on ring 2. The units then need 1–2 KB per ring.
- **Shims 0–2:** MM2S 1 carries B (`B{r}_shim_alloc`), and IRON refuses an ObjectFifo on a channel a PacketFlow uses. Make B a
  hand-built ring as A is (`PacketFlow(3, shims[r], mems[r], src_channel=1, ...)` into the memtile's existing B S2MM channel + a
  memtile TileDma forward to the row's cores). B (GEMM phases) and Ein2 (norm phases) never overlap in time.
- **Kernel:** ew8.cc's NPIPE loop gets per-ring in_need / in_done (the ring state is already in locals since NIOC). The n1k_step
  pointers come from both rings.
- **Pre-registered (NIN2 alone, timeline):** norm0 / norm2 walls → norm1 / norm3 (67.7 / 82.7 µs ± 4); norm1 / norm3 unchanged;
  BIT-IDENTICAL.

**DDRCAP probe (rank 0'):** one op that streams 2 × 512 KB per column from DDR through both shim MM2S channels into two sinks
(memtile or the EW core's two rings, cores discarding), with the 8 shim S2MM draining 256 KB each.
- Report aggregate GB/s for 8 / 16 channels.
- Pre-registered: ≥ 84 GB/s (today's norms prove it); the open question is whether 16 channels reach ~110–120.

## GEMM L1 layout (rev 14 ranks 4–5, main's round-18 request): negative, with the evidence

**Model** (`r20/collide.py`). Per static cycle, the banks mm_rot touches come from gen_rot's replay. The A ring is modelled as
36 writes of 32 B per object at 7.95 B/cycle, starting at the release of the slot three objects back. B goes into the idle
buffer from the end of the k-step that last used it.

| layout (core 0_2, full12r18m) | MEM A | MEM B | total | note |
|---|---|---|---|---|
| deployed (slots b1 b2 b2) | 75.3 | 91.6 | **166.9** | measured 167.5 (calibration point; α = 0.72, DMAB 0.68–0.75) |
| slots b1 b2 b0 | 63.2 | 89.2 | 152.3 | −15 ± 10; bank 0 has no room (data region) |
| slots b0 b1 b2 | 75.8 | 85.8 | 161.6 | |
| all slots b2 | 93.8 | 85.1 | 178.9 | worse |
| bound: A slots in a piece-free bank | 28.9 | 90.3 | 119.2 | −48: capacity forbids it |

- Mean over 12 DMA phases; single-phase spread 11–22 cycles. `r20/collide_scan.out`: all 27 slot-bank assignments × 4 affine
  piece placements (single phase).
- **Why the A part stays ~60–75:** every row block's two pieces lie in two of banks 0–2. So whichever bank a slot sits in sees
  piece traffic ~45 % of the time it is written.
- **The piece map cannot be time-aware.** The epilogues (bias_pcs, bgelu, gup, pan, GELU_RUN) and the attention use acc_hi as one
  contiguous 24 KB array, so the map can only shift.
- **B objects in two banks:**
  - The bank of the popped B object is busy 0.97 (64 pops + 8 fills per 74 cycles), so it can hold nothing the k-step touches.
    Two such banks need 2 × (16 − 4.5) = 23 KB of k-step-cold data.
  - The GEMM core has 14.6 KB: stack 2.5, tables 5.6, the C object's tail 6, misc 0.5.
  - A partial version (B1 in bank 0 with 5 pieces) costs ~125 cycles per k-step of internal conflicts for a saving of ~70.
- **E7' (B over two banks with one stream) is void.** The stream's 72 line reads per row block all issue from ONE FIFO (lf1); a
  FIFO cannot pop twice in a bundle. The ldb slot and bank 3 both bind at 72 + g. R2 below is the only way past it.

## Research items (offline, next passes)

- **R1 (SEQPRE support):** a txn-level checker that SEQPRE did not hoist a fill that reads a region written by a pending drain.
  `cpseq.py` + plan8 deps.
- **R2 (E7''):** B on BOTH FIFOs with A merged into lf0's restarts, B lines woven between two banks by a 2D post-modify
  (dn = 1: dj, m, dj, m ...).
  - Needs: CP-SAT feasibility at R = 64; register (ex) and bank budgets; a host interleave.
  - **UNVERIFIED** that `vlda.pop.576` accepts lf1 (MACHINE says both FIFOs pop on both slots).
  - Prize: −144 static cycles per k-step (−0.9 ms realised).
- **R3 (LATE-C):** the gen_rot CACQ op (a conditional C acquire at rb 6 of a tile's first k-step), zero-load records for the
  first k-step, and the new piece map (pair 0 of rb 6–17 in C). The CP feasibility at R = 72 comes first.
- **R4 (DAPRE):** whether a K = 768 op's A really occupies both halves of the memtile A slot (A_HALF multicasts it to both). If
  one half suffices, down_a's first 12 chunks can load during up_b.

## Tools (rev 15, `r20/`)

- **`gen_rot.py`:** main's r18 generator + pass-15 changes:
  - ACQ_GAP 6 / REL_GAP 8 hard floors (env may only widen);
  - the replay asserts the acquire → A-fill distance;
  - `PLACE_R18M`;
  - per-cycle bank trace (`st["trace"]`).
- **`collide.py`:** the time-resolved DMA-collision model: `--scan`, `--alpha` (calibrated 0.72), `LAYOUTS`, phase-averaged.
- **`tilefit.py`:** per-tile periods and overheads per GEMM kind from timelines.
- **`cpseq.py`:** CP instructions on each boundary's critical path, now vs after SEQPRE (npu-iron).
- **`rotdb.py` / `rotdb.cc`:** the one-core k-step bench with a DMA-fed A ring. Builds in `r20/build/` (g0a, g2f).
- **`build_rot.sh`, `solve_g01.sh`:** re-solve, fit, lint, tcall and replay of ROT variants on the r18m placement.
- `mm_rot_g{0,1}a.*`: pass-15's independent schedules; main's deployed ones are in npu-tapnext/r19.
- `dis/full12r19b_0_{2,5}.dis`: the deployed disassemblies.
- **`tools/asmlint.py`:** LOCK-ACQ-GAP, LOCK-REL-GAP, IMM-SRC. `asmlint_hist.py` has 23 cases (+3), all pass; full12r19b is
  clean.

## Status of the rev-14 items

| rev 14 | status (rev 15) |
|---|---|
| ROT g2f (rank 1) | done by main as g2a (−3.27): the acquire gap |
| PWRAP / PADDA → g0f / g1f (rank 3) | done by main: g0a deployed −0.18, g1a −0.11 (0.75 of static) |
| E7' (rank 4) | void (single-FIFO bound); E7'' is R2 |
| A slots into the least-busy bank (rank 5) | ≤ −0.1 feasible: dropped |
