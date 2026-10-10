# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 9, 2026-10-10 (evening), after round 13 (`full12r13f`, 49.21 ms at depth 12; `RESULTS_FROM_MAIN.md` round 13).
Revision 8 is in `rev8/PROPOSALS.md`, older ones in `rev7/` .. `rev1/`. The analysis behind every number is ANALYSIS rev 9.
Attention prices come from **`tools/attsim.py`** (the K / V multicast lockstep, calibrated on the five round-13 twins to 0.02
ms rms, per-block check corr 0.97) inside **`tools/sim16.py`** (`r14/sim16_r13f.out`); "thin / raw" = the two calibrations of
the flag map (ANALYSIS 2.2). Every price is a delta against the deployed r13f at depth 12; error bars give every unmeasured
component +-100 %.

**Headlines.**
- **The fallback's lockstep is a layout problem.** 99 % of flagged row blocks really move the max (flag-not-move 0.055 %), so
  the flag cannot get rarer and the exact test is nearly free now; what costs is that the moves cluster in the lower image
  (columns 5-6 hold 48 % of the "slowest column" events) and each K / V object waits for the slowest column. Dealing the
  query row blocks to the columns round robin (**P60**, DMA only, 0 B of core text, bit-identical) removes the lockstep
  (0.80 -> 0.06 ms) and the per-core imbalance at the head-group barriers: **-0.8 +- 0.3 ms**.
- The V phase (222 cycles per row block, uniform) is now the largest bit-identical attention lever: **P50 -1.2 (+0.3 / -0.8)**.
- Bit-identical attention stack P60 + P61 + P62 + P50 + P63: **-4.0 +- 1.2 ms** (attention 10.1 -> ~6 ms per step).
- GEMM text is the gate (0 B free): **T3** frees ~1.6 KB with no timing risk; the first 464 B come from deleting aexp_push
  (the slow path re-runs attk with the new max: bit-identical).
- Ceiling 31.1 ms at depth 12 (35.5 without the k-step row); rev 8 31.7 / 36.0.

Common rules (SKILL 3, 9): one dispatch with a short timeout first for every new design; the known-good test after any timeout;
Turbo / gpu_busy recorded; one-core bench (apb.py, guard zones) before building in; read the emitted loop of the binary you
bench; name the build each baseline comes from; regress BIT-IDENTICAL for every lever marked so.

## Ranking (revision 9)

| rank | id | what | cost | risk | expected (sim16, depth 12) |
|---|---|---|---|---|---|
| 1 | **P60** (new) | query row blocks dealt round robin to the 8 columns in half-tile pieces (attention A fill, memtile send pattern, O drain) | S-M | low | **-0.8 +- 0.3 ms**; tile pieces only: -0.5 +- 0.2 |
| 2 | **T3** (new) | GEMM text: aexp_push out (slow exp = attk rerun), chunk wrappers folded, acc_zero / attn_qcopy / attn_begin loops, mm_kstep tail call | S | low | 0 ms; ~1.6 KB free |
| 3 | **P50** | V phase in asm (chunk_v: 4 x 42 bundles per row block -> ~12-bundle body per dim pair) | M | low | **-1.2 (+0.3 / -0.8) ms**; text ~-150 B |
| 4 | **P29b** | SSM log_a + exp chain as 2 generated kernel-only passes | M | low | **-0.95 +- 0.2 ms**; EW text -1.4 KB (unchanged, not attempted) |
| 5 | **P61 + P62** (new) | slow path: chunk 22 through the fused loop (2 trips + zero panels); rescale_rb's O update in asm | S-M | low | after P60: **-0.7 +- 0.4 ms** (P61 -0.33 +- 0.1, P62 -0.38 +- 0.38) |
| 6 | P63 | the 36-iteration K loop (one call per (tile, chunk); probe the 3D Q pop first: Q2) | M-L | medium | -1.25 +- 0.6 ms |
| 7 | **Q1** (new) | diagnostic twins: F2 (fast loop only) at l2-depth 4 vs 8 | S | none | 0 ms; explains the unmodelled -0.30 of l2-depth |
| 8 | P55 | tile-granular A dependency (mlp3_a / out_proj) | S-M | low | -0.36 +- 0.12 |
| 9 | P29c + P52 | sqrt + recurrence generated; SSM outputs in place | M | medium | -1.15 +- 0.3 / -0.78 |
| 10 | P44 -> P7-E | EW norm stamps, then port-bound norm passes | M-L | medium | -1.5 to -3.1 ms; EW text -3 KB |
| 11 | P35 + P24b | B locks inside mm_ks; acc_zero folded | S-M | medium / low | -0.98 together |
| 12 | P57 | C object released in two halves | M | medium | ~-0.5 |
| 13 | P29d | conv + z panels generated | M | low | -0.84 (x 0.85) |
| 14 | H1 + H2 | host overlap / warm pool (H3 done) | S | none | <= -0.8 (throughput mode only) |
| 15 | P51 | norm writes the next GEMM's A slot | L | high | -0.96 |
| parked | P66 (new) | m seeded with q_i . k_i (numerics: DAVIS gate) | M | numerics | alone -1.4 (cost ~0.4 +- 0.4 not incl.); after P60 + P61 ~-0.3 +- 0.4 |
| parked | P67 (new) | TAU 16 nats (exp table to +17, +1 KB tables) (numerics) | S | numerics | alone -1.1; after P60 + P61 ~-0.4 |
| parked | P64, P65 | 3,456 B K / V objects (-0.04: producer not binding); chunk 0 QK-only loop (-0.13, needs a second asm loop) | | | |
| parked | P59, P14, H4 | as rev 8 | | | |
| retired | see the status table | | | | |

Combined (sim16, model base 50.28 = the polled timeline; shift by -1.07 to the timed 49.21): P60 + P61 + P62 48.72 -> P29b 47.78 ->
P50 46.51 -> P63 45.23 -> P29c + d + P52 42.44 -> P55 42.08 -> P7-E high 38.96 -> P35 + P24b 37.98 -> P51 37.02 -> P57 36.54 ->
k-step 1,600 **32.20** (= 31.1 from 49.21; depth 10 ~26).

## Status of the revision-8 proposals

| id | status | verdict / what is left |
|---|---|---|
| T1 | DONE (-0.7 KB + -208 B) | hit |
| P54 (1) | **refuted** (no base-register form of `vldb.4x64`) | retired; P54 (2) (vfloor from bm) stays a probe, low value |
| P56 | DONE per row block: -0.84 ms (predicted -3.2 +- 0.5) | **miss explained** (ANALYSIS 0): the loop's -2.9 was right; the exact test everywhere, chunk 22 and the multicast lockstep were unpriced. Follow-ups: P60, P61, P62, P63 |
| H3 | DONE (-0.22) | hit |
| P29b, P55, P57, P35, P24b, P29c-d, P52, P7-E, P51 | not attempted | re-based on r13f, prices unchanged |
| "cheaper exact test" (HANDOFF 2) | **retired** | the test runs only on flagged units now (10.8 %): < 0.1 ms left |
| "make the flag rarer" | **retired** without a numerics change | flag-not-move is 0.055 %; with numerics: P66 / P67 |
| "defer slow rows outside the lockstep" | **retired** | -0.80 alone, but needs a V copy per deferred chunk (no L1) and P60 leaves 0.06 ms of lockstep |
| more L1 K / V slots | **retired** | -0.66 alone, 4.6 KB per slot, moot after P60 |
| asm aexp (gen_aexp, HANDOFF 2b) | **superseded** by P62b | attk with the new max computes the same P at the same cost (L = E = 400) and deletes 464 B |
| P48 / P49 | superseded (rev 8) | |

---

## P60. Query row blocks dealt round robin to the columns — **S-M, risk: low** (rank 1, new)

**Mechanism.** Each core row's K / V objects are multicast to 8 columns with 2 L1 slots each, so per object the row waits
for the slowest column (ANALYSIS 2.1). The slow row blocks (2.26 % after chunk 0) cluster in rows 720-1007 = columns 5-6
(4.2-4.9 % vs 0.25 % in column 0). Rows are independent in attention, so any assignment of query row blocks to columns gives
bit-identical outputs.

**Design.** Piece n (3 consecutive row blocks = half a tile, 24 rows, 1,728 B of Q per head) of the head's 48 pieces goes to
column n mod 8, position n div 8 (positions 0-3 = pass 0's tiles 0-1, 4-5 = pass 1's tile 2). Data movement only:
- **A fill** (plan8 `attn`, `a` per column): 6 shim TAPs per column instead of 1, each [2 (both slot halves), 12 heads stride
  QKV_PER_H, 1,728 B] from the source (column n div 6, tile (n mod 6) div 2, half n mod 2) of the QKV region; the memtile slot
  then holds [piece][head][1,728], so the memtile -> core sends (A_RT tasks) read it with a 2-dim pattern back into [head]
  [tile][3,456] order (the ring and the cores are unchanged).
- **O drain**: per column and pass, one drain TAP per piece (4 + 2 instead of 1 + 1) to the natural rows of the out_proj A
  region (APAN [c][head][rb 18][576 B]: a piece is 1,728 contiguous bytes per head).
- Simplest first step (P60b, tile pieces): retarget the qkv epilogue's Q-object drains to slot (n mod 8, n div 8) of tile n and
  the attention O drain per tile; no A-fill change. -0.5 +- 0.2.
- Check: shim BD / task budget for the extra TAPs (the A fill is once per op), regress BIT-IDENTICAL.

**Price** (sim16 thin / raw): half tiles -0.84 / -1.00, row blocks -0.96 / -1.14, tiles -0.54 / -0.62. Per-block slope
(measured / model 0.80-0.88) and the unmeasured extra DMA task cost (+0.05 +- 0.05) give **-0.8 +- 0.3 ms**.

**Pre-registered** (timeline of the P60 build vs r13f, 11 blocks): attention mean 839 -> 765 +- 25 us per block; the per-block
spread (max - min, 176 us today) falls below 110 us, and blocks 10 / 11 move least (< 25 us: they have almost no flags).

## T3. GEMM text recovery — **S, risk: low** (rank 2, new)
Sizes from `r14/textmap_r13f_0_2.out`:
- T3a: delete aexp_push (464 B): the slow path re-runs `attk` with rows8(-m_new) over the same K object and Q (both still in
  L1); attk's P is op-for-op aexp_push's (round-13 bench). Chunk 22 then needs P61 first (invalid blocks: P must be 0).
- T3b: fold attn_chunk_k (304 B) / attn_chunk_v (256 B) into chunk_k_fused / chunk_v as the bound entry points (~-250 B).
- T3c: acc_zero as a 3-bundle zero-overhead loop (336 B, 15 unrolled full-form bundles; ~-230 B; time it).
- T3d: attn_qcopy as one pointer-bumped loop (400 B minsize; ~-200 B). T3e: attn_begin's zero loops merged (~-250 B).
- T3f: mm_kstep (320 B): one C object per core, so the k-step records can be complete constants (tail call; ~-200 B).
**Expected:** ~1.6 KB free, timing +-0.05 ms, BIT-IDENTICAL.

## P50. V phase in asm — **M, risk: low** (rank 3)
**Baseline:** chunk_v = a 42-bundle loop per dim-block pair (12 mmuls) x 4 + the ones panels + P pops: 222 cycles per row
block in situ (r8 stamps; static count agrees). **Target:** per pair 12 mmuls (vec slot) with 4 accumulator chains in today's
order (bit-identical), V pops on both load slots, O loads / stores 8 + 8 per pair: ResMII ~12 per pair -> ~80-100 per row block.
**Price:** sim16 -1.19 ms (the pass-1 send / shim limits take ~0.25 of the static 1.45). The new cost is unmeasured: 60-180
cycles -> **-1.2 (+0.3 / -0.8) ms**. Text: chunk_v 608 B -> ~450 B. Pre-registered one-core bench: 80-110 cycles per row block.

## P29b. As rev 8 (rank 4)
Pre-registered: the deployed `loga_exp_a2` benches 1,080-1,200 cycles per 16 vectors, the generated passes 420-520; step
-0.95 +- 0.2 ms; EW text -1.4 KB.

## P61 + P62. The slow path — **S-M, risk: low** (rank 5, new)
- **P61 chunk 22**: call attk with 2 trips (the valid key blocks; the loop has 3 stages: check the trip-count floor) and write 4
  zero P panels per row block (4 x 72 B); the flag covers the valid lanes, flagged blocks take the slow path as today. Invalid
  P is exactly zero today too: bit-identical. Uniform (all columns at once): **-0.33 +- 0.1 ms**. ~60 B of C++.
- **P62 rescale_rb**: today a minsize C++ loop over 9 x 64 fp32 lanes (hi / lo bf16 products) after exp32; an asm O-update loop
  (exp32 stays) is ~3x faster. R is degenerate in the fit (0-450 cycles), so **-0.38 +- 0.38 ms after P60** (-0.70 alone).
- **P62b** = T3a (aexp_push -> attk rerun): 0 ms, -464 B.
Pre-registered: with P60 in, timeline attention -30 +- 15 us per block for P61 + P62.

## P63. The 36-iteration K loop — **M-L, risk: medium** (rank 6)
One call per (tile, chunk): the drain (2 stages x 35) and preamble (~40) per row block go, -m loaded per iteration (round 13's
design). Static saving ~110 cycles per row block [DOC attk.json II 35, 3 stages]; HANDOFF's 115; the in-situ ratio (400 in situ
vs 280 static) is unmeasured: **-1.25 +- 0.6 ms**. **Q2 first**: does `VLDA_POP_576_3D` give the Q stream's per-row-block return
+ advance (one-core probe)? Text ~neutral after T3.

## Q1. Why did l2-depth 8 help? — **S, risk: none** (rank 7, new)
attsim reproduces the twins at l2-depth 4 to 0.02 ms but has no depth effect (the multicast is gated by the 2 L1 slots; the
shim fill binds only without K compute), so it misses r13c -> r13e (-0.30). Two timing twins: ATTK_FORCE=2 (fast loop only,
uniform units) at l2-depth 4 and 8. If 8 still gains ~0.3 ms with uniform units, the gain is the producer at head-group
boundaries (then P64's bigger objects / fewer bytes matter again); if it gains ~0, the multicast path has slack the model lacks
(then attsim needs a slack term and the P60 price drops by up to 25 %).

## P66 / P67 (parked, numerics)
- P66: seed the stored max with s_ii = q_i . k_i: flagged units after chunk 0 2.26 -> 0.27 %, chunk 0 100 -> 9.7 %. Data path: the
  column's own K objects (QKV slot 3i + 1) ride the A slot beside Q (A load and ring traffic 2x: ~0.2 + 0.2 ms, unmeasured), an
  8x8 diagonal mmul per row block at qcopy. Not bit-identical (P scales differ): DAVIS gate. Worth ~-0.3 +- 0.4 ms after P60 + P61.
- P67: TAU 8 -> 16 nats: flags after chunk 0 0.25 %; the exp table to +17 (112 segments, +1 KB of L1 that is not there).

## Unchanged from rev 8
P55, P57, P29c, P29d, P52, P44 / P7-E, P35, P24b, P51, H1 / H2, H4, P59, P14 (rev8/PROPOSALS.md).
