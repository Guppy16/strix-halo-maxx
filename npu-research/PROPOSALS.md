# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 15, 2026-10-11, research pass 15, written during main's round 19.

- **Base:** `full12r19b` = full12r18m + ROT g0a. full12r18m measured 35.20 ms (sweep median) at depth 12; g0a's A/B is −0.18.
- **Data:** main's r18m timeline (RESULTS_FROM_MAIN round 19, `npu-tapnext/build/tl_r18m_run{1,2,3}.json`), KS-P38 and NIO2
  (round 18), and the deployed `full12r19b.bin` / elfs.
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
     | per-tile overhead | ~2.2 | epilogue + C-object hand-over |
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

## Ranking (rev 15; base full12r19b ≈ 35.0 ms)

| rank | id | what | cost | risk | expected (depth 12) | pre-registered check |
|---|---|---|---|---|---|---|
| 0 | **CPRATE** | probe: N dummy CP writes after one dependent op's sync (recipe below) | S (1 build, 1 A/B) | none | measures t_op | Δstep = 12 · N · t_op; N = 400 → +0.6..+1.1 ms if t_op is 0.13–0.22 µs |
| 1 | **SEQPRE** | design8 sequence: pre-issue a dependent op's memtile slot loads + sends, drains, B fills, EW params fills before its producer wait; norms pushed to `pending` instead of `finish()`ed at once | M (design8 sequence code only) | medium (queue / BD budgets, ordering) | **−1.4 ± 0.5** (cpseq: −1.37..−2.31 before realisation) | regress BIT-IDENTICAL; timeline gaps: yx / out / qkv / out_proj / mlp0 / mlp3_a −10..−18 µs each, norms −6..−9 µs each |
| 2 | **N-IO-ASM** (+ NS) | the norm unit loops in asm (pass-1 glue ~90 → ~20, pass-2 glue ~155 → ~40 in situ), stats in asm | M–L | medium | **−0.5..−0.8**; NS −0.1 more | NIO2: pass-1 unit RMS ≤ 230, pass-2 unit RMS ≤ 200 / LN ≤ 290 |
| 3 | **CDRAIN** (TPROF first; main already has it) | per-tile C-object hand-over: the next tile's C acquire waits for the 18 KB drain of a depth-1 C object | M | medium | if TPROF `hand` ≈ 2,000–2,600 cycles: −0.4..−1.0 via a split drain (below) | TPROF pre-registration below |
| 4 | attention S3 / K1 / K2 | rev 10 / 11 prices; GEMM text now has ~800 B free | M | low | ~−0.5 (±30 %) | rev 11 checks |
| 5 | **DAPRE** | down_a's A load during up_b (the only non-dependent op still paying; overhead 0.52 ms vs down_b 0.26) | S–M | medium | −0.2..−0.3 | down_a gap → ~0, per-tile period ≈ down_b's |
| 6 | SSM ssq2 / FSX | rev 13 | M | low | −0.11 / −0.07 | rev 13 |
| — | NIN2 (dual EW input stream) | a second shim → EW packet flow (the EW core's 2nd S2MM is taken by ew_op's header: fold it into the ring) | M | medium | after N-IO-ASM: −0.1..−0.3 (n3 ops become input-bound at ~534 cycles per unit) | NIO2 pass-1 waits → ~0 |
| — | GEMM A-slot layout | best feasible slot banks (1, 2, 0) | M (IRON data region) | medium | ≤ −0.1 | dropped |
| — | B objects in two banks | | | | infeasible (capacity proof above) | dropped |

Ceiling from ≈ 35.0, stacked:

| step | ms |
|---|---|
| SEQPRE −1.4 | 33.6 |
| N-IO-ASM + NS −0.7 | 32.9 |
| CDRAIN −0.6 (if TPROF confirms) | 32.3 |
| attention −0.5 | 31.8 |
| DAPRE −0.25 + SSM −0.18 | **~31.4** |

- **Beyond that, structural levers:**
  - An activation-stationary DDR bypass for dependent A: the remaining ~7–10 µs per boundary plus the A-bound first tile,
    ~−1.0 ms.
  - E7'' (B on two FIFOs): a static R of 72 → 64 per row block would be −0.9 ms realised, if feasible.
  - The B-collision floor (~92 cycles per k-step ≈ 0.8 ms) has no lever.
- **Floor ≈ 29–30 ms.** Stacked prices below 33 carry ±1 ms.

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

## CDRAIN — the per-tile C-object hand-over (rank 3; main's TPROF measures it)

- **Why it matters:** every GEMM core has ONE C object (`C{c}_{i}` depth 1, 18,432 B). It holds 12 KB of the fp32 accumulator
  pieces (pair 0 of rb < 12) and receives the tile's bf16 output, so the next tile's `acc_zero` and first k-step must wait until
  the core MM2S has drained it.
- **Model:** 18,432 B / ~8 B/cycle ≈ 2,300 cycles per tile, × 1,104 tiles per core per step ≈ **1.4 ms** if nothing overlaps.
  - The timeline fit allows 1.3–2.2 µs of plain per-tile overhead (2,400–4,000 cycles).
  - The static plain epilogue (tcall: acc_epilogue glue ~221 + 3 bias_pcs = 374 + 218 + 530) is ~1,340.
- **Pre-registered for TPROF** (cycles per tile, median):
  - `epi` plain 1,350–1,600;
  - `hand` (previous epilogue end → this tile's start) **1,900–2,700** if the drain wait is real, < 400 if it is not;
  - `ks` g0a 1,560–1,620;
  - `first` of dependent ops' first tiles: + (A_load − K / 64 k-steps), ≈ +11k cycles at K 768, +22k at K 1536.
- **The lever, if `hand` ≈ 2,300:** a split drain. The epilogue writes rows 0–8 of the output first and releases a first C
  half. That half drains while it writes rows 9–17. The C object becomes two 9 KB objects (two join inputs per core in the
  memtile).
  - Saves ~the second half's epilogue time per tile (~600–700 cycles): **−0.4..−0.5**.
  - A bigger cut needs the first k-step not to touch C: process rb 12–17 (acc_hi only) first. That needs a different A order in
    the slot (A_RT chunk order), **−0.8..−1.0**. Research item R3: CP-solve ROT's row order.

## N-IO-ASM and the norm plan (rank 2)

**After NIOC** (NIO2 r18h, in situ incl. stamps):

| | pass-1 unit | pass-2 unit | nrm call |
|---|---|---|---|
| RMS | 293 | 302 | 147 |
| LN | 249 | 391 | 232 |

- tcall on full12r19b: pass-1 iteration 215–233 static (n1k_step 134 + ~90 glue); pass 2 ~196 (RMS) / ~256 (LN) static
  (nrm 117 / 177 + glue).
- **Input floor** per unit at R_in 5.75: n2 356, n3 534 cycles.
- Per unit today, RMS 595 and LN 640 are above both floors, so ops are compute-bound on average. Pass 1 alone runs below its
  input rate, which is why NIOC's pass-1 cut half-missed.
- **Model** (whole-call per unit, 216 units per op):

  | | n2 RMS | n2 LN | n3 RMS | n3 LN |
  |---|---|---|---|---|
  | glue cuts (−70 pass 1, −115 pass 2) | ~410 | ~455 | — | — |
  | n3 ops pinned at the floor | — | — | 534 | 534 |

  Δ per step:

  | ops | ms |
  |---|---|
  | RMS n2 (12) | −0.27 |
  | LN n2 (12) | −0.27 |
  | RMS n3 (12) | −0.09 |
  | LN n3 (13) | −0.17 |
  | **total** | **−0.80** |

  Allowing for the unlocated in-situ pass-2 excess (s2): **−0.5..−0.8.**
- **NS** (stats in asm, −250 per row block of 300–420): −0.1 (n3 ops input-bound).
- **NIN2:** raise R_in by a second shim → EW stream. Matters only after N-IO-ASM: the n3 floor drops 534 → ~310, worth
  **−0.1..−0.3**. It needs the EW core's second S2MM, which ew_op's per-op header uses (fold the header into the ring's params
  fill).
- **Order:** SEQPRE (the norm boundary, −0.3..−0.45 of its price is the norms') → N-IO-ASM → NIN2 / NS.
- **Pre-registered NIO2 after N-IO-ASM:** pass-1 unit RMS ≤ 230 (n2 ops), pass-2 unit RMS ≤ 200 / LN ≤ 290; norms' timeline span
  −0.4..−0.7 ms.

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
- **R3 (CDRAIN-B):** ROT with a row-block order that touches acc_hi pieces first (rb 12–17), so the next tile's first k-steps
  overlap the C drain. Needs an A slot chunk order change and gen_rot with a free row order.
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
