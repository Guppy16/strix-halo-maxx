# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 17, 2026-10-11, research pass 17, written during main's round 21 (which consumed r22/ as it appeared).

- **Base:** `full12r21e3` = r20z + CSUB + CSUB-QK + ZLOAD + E7'' R 68 (main, BIT-IDENTICAL). r20z was ~31.8 ms quiet at d12
  (EXTRAPOLATED); round 21's A/B gains (−0.13, −0.13, E7'' −0.22..−0.30 at 1.81 GHz) put r21e3 at **~31.3 EXTRAPOLATED**.
- **Data:**
  - The deployed builds' GEMM-core disassembly (`r22/full12r20z_0_2.dis`, tcall).
  - RESULTS_FROM_MAIN rounds 20–21; npu-tapnext NOTES §25, RESULTS §29, HANDOFF (round 21 block); SKILL additions of round 21.
  - r21 `ddrbudget`, main's `r21/mtmap.py`.
- **Archive:** revision 16 is in `rev16/`.
- **Prices:** depth 12, 1.81 GHz.
  - One cycle per GEMM k-step = **8.25 µs per step** (14,930 k-steps on the GEMM critical path: 13.0 ms / 1,576 cycles).
  - k-step body cuts land at ~0.75 (round 19). E7'' R 68 landed at ~1.4–1.9× its static cut (−26..−36 in situ for −19 static):
    its B writes moved off the busiest bank [INFERRED].
- **CPU load:**
  - The NPU clock falls to 1.405 GHz under ~20 busy CPU threads (main, round 21). This pass ran 12 CP-SAT workers until 05:56,
    then ≤ 4.
  - Every CP run below used 4 workers.

## Headlines (rev 17)

1. **E7'' was built from `r22/gen_e7.py` in round 21 and kept: R 68, −0.28 ms at 1.405 GHz (~−0.22..−0.30 at 1.81), bit-identical.**
   - The generator: a CP-SAT model, emission, prologue fit, whole-call replay; asmlint is clean (0 / 0 after the LOCK-REL-GAP fix
     for register-copy restarts).
   - R 64 is neutral (main: +0.04 / −0.04 / −0.08). The replay had said so: 1,244 static + 129 bank stalls vs 1,338 + 0.
   - Bank 3 must take 63 line reads per row block with a 576 B X segment. R 66 (hard banks) found nothing in 15 min.
   - The A-port floor (1,304 cycles per k-step) caps what R < 68 buys anyway.
2. **KTL is built: `r22/gen_e7k.py`, `r22/mm_e7k_r70.s / .h` (rank 1).**
   - It is one modulo loop over all of a tile's k-steps. The deployed call costs 1,338 + 13 of mm_ktile glue per k-step, of which
     114 + 13 are setup / prologue / ret / loop.
   - mm_e7k: 15,196 (nks 12) and 30,316 (nks 24) cycles per tile call (= 76 + 1,260 × nks), replayed bit-correct over 1–25 k-step tiles at three acc_hi
     placements, 0 bank stalls. asmlint 0 / 0, text 1,534 B against the deployed E7'' kernel's 1,648.
   - Per k-step 1,264–1,268 static vs 1,351 today: **−83..−87 static → −0.50..−0.70 ms** pre-registered.
   - The k-step-0 ZLOAD config is switched off in-loop: `and r6, r6, ~flag` at rb 17.
3. **KTL needs E7'''s two FIFOs (this corrects rev 17's first draft).**
   - Consecutive k-steps read different B buffers, so no stream instance may serve two row blocks.
   - One row block's 64 pops span ~85 cycles. ROT's single FIFO therefore cannot do it (DEAD_ENDS A28).
   - E7'' can, with lf1 split L | N and every segment instance serving one row block (`--aligned`). CP-feasible at R 70, not found
     at R 68 in 5 min.
4. **OPPRE's +0.18 is not aggregate DDR.** Attention pass 1 runs at ~17 + ~6 = ~23 GB/s against ~55. The lead is H_CP (per-group
   drain waits ahead of the K / V pushes in the in-order CP); OPPRE-W separates it from H_SHIM.
5. **C waits: done.** CSUB + ZLOAD (round 21) took the hands to 345 / 668. LATE-C hangs; RTC is not worth it.
6. **3A (third A region): −0.07..−0.10 for L.**
   - First-tile paces now: out_proj 6.7, down_b 4.15, mlp3_b 3.92 µs per block. 3A takes down_b + mlp3_b.
   - It needs 124,416 B contiguous on memtiles 0–2 (repack B + the C joins, or l2 depth 7).
   - GEMM → norm tails: no lever above ~0.1 ms.

## Status at hand-off (main's round 22, read 08:15)

- Round 21 ended at **full12r21k1: 31.30 ms at d12** (quiet sweep, MEASURED), −0.70 vs r20z in one process.
- Round 22 so far:
  - 3A built and kept: −0.08 (this revision's re-price −0.07..−0.10: hit).
  - K2 −0.02 (pre-registered −0.05..−0.10: miss), off by default for its 608 B of text.
  - E7'' nx = 10: no better than nx = 8.
- So S3 / K2 drop in rank: chunk 0's compute is mostly hidden behind the K / V stream start.

## Ranking (rev 17, final; base full12r21k1 31.30 MEASURED, + 3A ≈ 31.22)

| rank | id | what | cost | whole-call price | expected (d12) | separating probe / pre-registered check |
|---|---|---|---|---|---|---|
| 1 | **KTL** | mm_e7k: one call per tile (r22/mm_e7k_r70.s, fitted) | M (C++ + records + one aiecc) | per tile 76 + 1,260 × nks vs 1,351 × nks | **−0.50..−0.70** | replay bit-correct for 1–25 k-steps (done); in situ: BIT-IDENTICAL, TPROF steady k-step −60..−85 cycles; a hang = B lock order (one-core KTLP first if in doubt) |
| 2 | E7'' + KTL at R 66–68 | aligned + KTL schedules at lower R (`gen_e7.py --ktl --R 68 --hint r22/e7k_r70_n16.json`, longer runs) | S (CP time) | −36..−72 static per k-step vs R 70 | −0.15..−0.3 on top of KTL | replay 0 stalls first; the A-port floor (1,304) bounds it |
| 3 | SSM ssq2 + FSX | rev 13 | M | FSX −50 per unit | −0.18 | EW text 112 B free: text first (norm_rows_k 3,504 B of C++) |
| 4 | S3 | the attention's exact test + rescale + −m store in asm (T + R −60 per flagged unit) | M | rev 10 | −0.05..−0.10 | K2's miss says chunk 0 is hidden; price S3 on flagged non-chunk-0 units only |
| 5 | OPPRE-W → OPPRE-lag | probe, then out_proj part g after head group g + 2's K / V pushes | S | out_proj pace 6.7 µs per block | −0.04..−0.10 | OPPRE-W: H_CP +0.12..+0.20, H_SHIM ≤ +0.03 |
| — | LATE-C | — | — | — | dead (hang) | round 21 LATEC |
| — | RTC, NTAIL-PF, NTAIL-M | — | — | — | not proposed | DEAD_ENDS B14, C25 |

**Ceiling, stacked:** r21k1 31.30 (MEASURED) → 3A −0.08 → 31.22 → KTL −0.6 → **30.6** → R 66–68 −0.2 → 30.4 → SSM −0.18 → 30.2 →
S3 −0.07, OPPRE-lag −0.07 → **~30.1** (EXTRAPOLATED).

- **Floors:**
  - GEMM k-steps: ~1,460 in situ after KTL against the A-port floor of 1,304.
  - Epilogues ~1.6 ms; attention ~6.3; the EW DDR read floor 6.6; boundaries ~0.75 (tails 0.58 + paces ~0.17).
- **Floor ≈ 27.5 ± 1 ms.** Stacked prices below 31 carry ±1 ms.

## KTL — what is built and how to drop it in (rank 1)

**Kernel** (`r22/gen_e7k.py` on a `gen_e7.py solve --R 70 --nx 8 --n1b 16 --ktl` schedule, `r22/e7k_r70_n16.json`):
- **Segments:** lf0 = A (fill + 8) → X (8 panels, bank 0) → Y (24, bank 3); lf1 = L (16) → N (16, serving the NEXT row block).
  Every instance serves one row block.
- **Restarts:** `movs p0 / p1, rBase` from Y r16, L r17, N r18, X r19.
- **Buffer toggle** at the row block whose record carries the B flag (rb 17):
  - `mul rT, r29, rΔ; xor rBase, rBase, rT`, Δ = B0 ^ B1 per segment (r20–r23), temps r0 / r1 / r2 / r7.
  - Each toggle sits between the restart of the instance serving rb 17 and the one serving the next rb 0 (L's restart BP restarts
    the next period's L: offset −1).
- **Locks:** acq.cond / rel.cond take r26 only (assembler), so the four r26 groups are disjoint in the schedule:
  - A acquire: `mul r26, r26, r27` (flag × remaining-rb counter: 0 on the trip after the last).
  - A release: RQ1 → r26.
  - B acquire: `mul r26, r29, r30`, r30 = r27 − 1 (none beyond the tile); ≥ 6 bundles before the first fill of every instance
    serving the next rb 0.
  - B release: `mov r26, r29`; ≥ 8 after the last op of every instance serving rb 17.
- **ZLOAD across the tile:** the kp-0 MACs read r6; r6 = 781 on entry, and `xor r31, r29, r4; and r6, r6, r31` at rb 17 (after
  k-step 0's last kp-0 MAC, before k-step 1's first) leaves 780.
- **Records:** six fields per row block (A pointer, J0, J1, A acquire, A release, B flag).
  - Table = head (the prologue's plain loads of rb 0's stage −1 fields) + one k-step (108 words); the body's `lda.2d .., [p2], d2`
    wraps it (dn2 = 107, dj2 = 4, m2 = −428).
  - rb 0's J0 is the "from rb 17" value; the caller passes pc' = piece(0, 0) − J0.
- **Signature:** `mm_e7k(const int32_t *rec, float *pc', int32_t cfg = 781, int32_t trips = 18 nks, int32_t y0, int32_t x0)`.
  The caller acquires B(0) and nothing else; the kernel acquires B(1..nks − 1) and releases all of them.

**Main's side:**
1. `r21/mke7.py`-style import of mm_e7k (it is self-contained, no p6 / p7 frame).
2. mm_ktile: acquire B(0); `mm_e7k(rec, pc', 781, 18 * nks, KS_B0 (Y0), KS_X0)`; no per-k-step loop. Check which buffer IRON hands
   out first (P35's parity) and pass that one's Y / X; Δ is symmetric.
3. The record table from `mm_e7k_order` (head + period) as constexpr records (KSC).
4. B order from `mm_e7k_border` in pack8 (Y region: lf0-Y, L, N panels; X region: lf0-X).
5. The E7PATCH layout is unchanged: same X / Y buffers, the hand-off lock e7h.

**Pre-registered:**
- Static per tile 76 + 1,260 nks (replay); in situ −83..−87 static per k-step → **−0.50..−0.70 ms** (0.75–1.0 landing).
- TPROF steady k-step −60..−85 cycles. BIT-IDENTICAL.
- Watch: B lock order inside the kernel (the replay models the DMA refilling a released buffer at once).
- Not modelled: the in-situ cost of the in-kernel B acquire stall (today C++ waits there).

## E7'' — status and recipe (rank 2: lower R with KTL)

- **Built and deployed (main, round 21):** R 68 from `gen_e7.py` (residue matching), via E7PATCH (X buffers at 0x70A00 / 0x70C40,
  Y 4,224 B, aexp_ab in bank 3, tbn / tba in bank 1, the B S2MM split with a hand-off lock).
- **Always deliver the FITTED .s.** An unfitted prologue lost rb 0's first jobs in situ twice (SKILL, round 21).
- **Lower R:**
  - R 64 neutral (bank stalls). R 65 soft (2 excess + 43 conflicts). R 66 hard: nothing in 15 min at 4 workers.
  - For KTL the target is aligned + KTL at R 68 / 66. Give CP-SAT more time with `--hint` from the R 70 KTL schedule.
  - Price R 70 → 68: −36 static per k-step → −0.2 ms; → 66: −0.35. In-situ k-step ~1,390–1,420 then, near the A-port floor.

## OPPRE — what +0.18 is (main's request)

- **Not aggregate DDR:** the attention reads 5.31 MB of K / V per block over ~554 µs; pass 1 (~160 µs) at ~17 GB/s, + 1 MB A
  (~6 GB/s) = ~23 against ~55.
- **H_CP (lead):**
  - The attention pushes all K / V fills with `wait=False` into 4-deep shim queues, so the CP spends the attention inside that push
    loop.
  - OPPRE's per-group drain waits and extra task groups sit in that in-order stream: 3 waits × ~5 µs of drain tail ≈ 15 µs per
    block = +0.18 ms.
  - Also possible: `--reclaim-runtime-bds` waiting on BDs freed by the extra groups.
- **H_SHIM:** shims 0–2 carry K / V (MM2S 1) and an A part (MM2S 0) at once.
- **OPPRE-W** (the same CP order, 64 B A parts): H_CP +0.12..+0.20 and ATPROF pass-1 groups' first K +3..6 µs; H_SHIM ≤ +0.03.
  OPPRE-37 (A parts on columns 3–7 only) is optional. The patch was not saved: rebuild from NOTES 25.4.
- **If H_CP:** OPPRE-lag (rank 6).

## C waits (main's request)

- Done: CSUB −0.13, ZLOAD −0.13. LATE-C hung. RTC is not worth it.
- The residual hand is ~0.2 ms per step. The drain collisions now sit in the epilogues (+219 / +561), the bank the epilogue writes:
  no lever priced above 0.05.

## 3A and the GEMM → norm tails (main's request)

- **3A:**
  - Paces now 4.15 / 3.92 µs per block (down_b / mlp3_b): −0.07..−0.10 ms.
  - Needs 124,416 B contiguous: memtiles 3–7 have 141,312; memtiles 0–2 have 128,000 in five holes (B objects scattered). Repack B +
    C joins (C loses its 64 KB alignment, memtile bank behaviour UNMEASURED), or l2 depth 7 (127,488).
  - Plus own E / F locks and 3 runtime tasks per memtile (count BDs).
- **Tails:** every row's last 64 columns arrive in the producer's last tile at the write cap. NTAIL-PF ≤ −0.07; NTAIL-M −0.1..−0.25
  for L with B re-read ×2. Not proposed.

## Probes

| id | what | separates / gates | pre-registered |
|---|---|---|---|
| **OPPRE-W** | OPPRE's CP order, 64 B A parts | H_CP vs H_SHIM | H_CP +0.12..+0.20; H_SHIM ≤ +0.03 |
| KTLP (optional) | one-core mm_e7k bench, DMA-fed A and B rings (r20/rotdb pattern), 3 k-steps | in-kernel B locks, lda.2d wrap on hardware | bit-exact; 76 + 3 × 1,260 + B waits |
| ACQ0 | done by main | — | acquire by 0 REFUSED at CDO generation; hand-off lock used |
| **CLKLOAD** (main's MACHINE question) | r7/clk.py (NPU clock) at 0 / 4 / 8 / 12 / 16 / 20 busy CPU threads, two kinds of load: vector-heavy (numpy matmul threads) vs scalar spin (`yes > /dev/null`), with hwmon package power where readable (no installs, no sudo) | SoC power budget (STAPM / PPT shared by CPU and NPU) vs a DPM policy | power budget: the NPU clock tracks package power, so the vector load drops it at fewer threads than the spin; DPM: the clock depends on thread count (or not at all), not on the load's power |

## Tools (rev 17, `r22/`)

| tool | what |
|---|---|
| `gen_e7.py` | E7'' CP-SAT model + emit / fit / whole-call replay. Options: `--R --nx --n1 --n1b` (lf1 split), `--aligned` (segment instances serve one row block), `--ktl` (KTL ops + constraints), `--hint`, env `E7_DROP` (`soft,nb,bank3,bank0,qey,bank3soft`) |
| `gen_e7k.py` | KTL emission (records head + 2D-wrapped period, toggles, counter masks, in-kernel B locks, ZLOAD switch), fit, multi-k-step whole-tile replay (B buffer ownership, DMA refill on release, A stripes across k-steps) |
| `build_e7.sh`, `build_e7k.sh` | fit + assemble + link + asmlint + tcall + replay |
| `mm_e7_r68_x8.*` (deployed by main as e7r68b), `mm_e7_r64_x8.*` (neutral), `mm_e7k_r70.*` (KTL) | kernels + .h + .dis |
| `diag64*.out`, `solve_*.log` | feasibility diagnostics (R 64: bank 3 binds; absolute matching infeasible) |
| `tools/asmlint.py` | register-copy restarts open unguarded streams (E7'' lf0 B segments) |
| `DEAD_ENDS.md` (repo root) | 167 entries + the revived list |
