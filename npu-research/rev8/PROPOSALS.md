# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 8, 2026-10-10, after round 12 (`full12r12e`, 50.05 ms at depth 12; `RESULTS_FROM_MAIN.md` round 12). Revision 7 is in
`rev7/PROPOSALS.md`, older ones in `rev6/` .. `rev1/`. The analysis behind every number is ANALYSIS rev 8. Savings marked
[sim15] come from `tools/sim15.py` on the r12e timeline (base 50.14 ms vs 50.05 measured; `r13/sim15_r12e.out`). Loop IIs
marked [GEN] are msched's (`tools/attk.py`, `r13/attk.out`), [CP] cpsched's. **Every baseline is from the deployed r12e
build** (timeline, static loop counts `r13/loops_r12e_*.out`, or a bench of the deployed kernel): the method change of ANALYSIS
0.2. All figures depth 12, each change alone against r12e unless stated.

**Headlines.**
- Scorecard: G1 hit in total (-1.49 vs -1.76 +- 0.35); G1a missed on an unmeasured component (the stash copy), P29a on a stale
  baseline, the P43 check on a wrong mechanism: **dependent GEMM boundaries are the A slot load** (out_proj 17.6 us = 124 KB,
  mlp3_a 37 us = 248 KB at the slot rate). P43b is retired; P55 replaces it.
- Text: GEMM 37 % attention, 31 % epilogues; EW 47 % SSM, 42 % norms. No dead entry points. ~1.8 KB (GEMM) and ~3.4 KB (EW)
  are recoverable without numerics changes; the cheapest 1 KB is mm_kstep's once-per-run record builder (0.7 KB) + asmfit.
- **Attention K phase = P56**: the lazy max makes P independent of the current block's max, so one asm loop over all 36
  (row block, key block) iterations of a (tile, chunk) computes QK^T and exp + push with no score buffer; the clamp's own
  vmin_ge mask flags the rare row blocks that go to today's C++. [GEN] II 37 with the biased LUT base: **-3.2 +- 0.5 ms**;
  -1.7 without the biased base. **P54 (the probe) decides which.**
- SSM re-priced from the r12e elf: the r9 phases still hold (code unchanged); P29b -0.95, P29c -1.15.
- Ceiling unchanged: 31.7 ms (36.0 without the k-step row); P14 today 37.1 ms per frame.

Common rules (SKILL 3, 9): one dispatch with a short timeout first for every new design and the known-good test after any
timeout; Turbo / gpu_busy recorded; one-core bench (apb.py, guard zones both sides) before building in; read the emitted loop of
the binary you bench; **name the build each baseline comes from**.

## Ranking (revision 8)

| rank | id | what | cost | risk | expected [sim15] |
|---|---|---|---|---|---|
| 1 | **P29b** | SSM log_a + exp chain as 2 generated kernel-only passes (replaces loga_p + exp_pA-D) | M | low | **-0.95 +- 0.2 ms**; EW text -1.4 KB |
| 2 | **T1** | GEMM text: mm_kstep's records at design time (-0.7 KB) + asmfit gup / pan / bgelu / cpy32 (-0.3 KB) | S | low | 0 ms; GEMM free 96 B -> ~1.1 KB (enables P56) |
| 3 | **P54** | probes: biased LUT base (`vldb.4x64` base operand), vfloor from a bm source | S | low | 0 ms itself; decides P56 -3.2 vs -1.7 and -1 cycle per 32 on every LUT loop |
| 4 | **P56** | attention K phase: one fused asm loop per (tile, chunk) over 36 iterations, lazy-fast with a C++ fallback | M-L | medium | **-3.2 +- 0.5 ms** (with P54), -1.7 (without) |
| 5 | **H1 + H3 (+ H2)** | host: lean numpy heads, heads(t) overlapped with dispatch(t+1), warm pinned pool | S | none | whole step -1.0 +- 0.3 ms (rev 7, unchanged) |
| 6 | P29c + **P52** | sqrt + recurrence generated; SSM outputs in place | M | medium | -1.15 +- 0.3 / -0.78 |
| 7 | P44 -> P7-E | EW norm stamps, then port-bound norm passes (one body for LN / RMS) | M-L | medium | -1.5 to -3.1 ms; EW text -3 KB |
| 8 | **P55** (new) | tile-granular A dependency: mlp3_a / out_proj A chunks wait per producer drain group | S-M | low | **-0.36 +- 0.12 ms** |
| 9 | P35 + P24b | B locks inside mm_ks; acc_zero folded | S-M | medium / low | -0.98 together |
| 10 | P50 | attention V phase in asm | M | medium | -1.4 ms |
| 11 | P29d | conv + z panels generated | M | low | -0.84 (x 0.85) |
| 12 | **P57** (new) | C object released in two halves: drain overlaps the second half's epilogue | M | medium | ~-0.5 ms |
| 13 | **T2** (new) | GEMM: V^T (flag 7) epilogue in asm (-0.5 KB); one exit in acc_epilogue (-0.15); EW: no `__udivsi3` (-0.15), SSM tables from the host (-0.55) | S-M | low | text only (+ a few us) |
| 14 | P51 | norm writes the next GEMM's A slot (mechanism confirmed by 0.1) | L | high | -0.96 ms |
| 15 | H4 | patch embedding on the NPU | M | numerics | host -0.4 to -0.9 |
| probe | **P59** (new) | program-memory overlay by `npu_blockwrite` from the runtime sequence (one core, a parked core, rate + safety) | S | medium (hang) | 0 ms; only if text blocks a > 1 ms lever |
| last | P14 | two frames in flight | L | medium | 37.1 ms per frame today (x1.35) |
| retired | P43b | B was never the bound (ANALYSIS 0.1) | | | |
| superseded | P48, P49a, P49b | by P56 | | | |
| parked | P7 full fold, P7c, P24c, P36-P41, k-step 1,600 | see rev 6 | | | |

Combined (sim15, each on the previous): P29b 49.20 -> P56 46.09 -> P29c + d + P52 43.30 -> P55 42.94 -> P7-E high 39.84 ->
P35 + P24b 38.86 -> P50 37.45 -> P51 36.49 -> P57 36.01 -> k-step 1,600 **31.7 ms** (depth 10: 26.4).

## Status of the revision-7 proposals

| id | status | verdict / what is left |
|---|---|---|
| G1a | DONE (-0.55; predicted -1.02 +- 0.2) | miss explained: unmeasured stash copy (then CPY32 -0.30) |
| G1b | DONE (PAN -0.30, BGELU -0.32; predicted -0.74) | hit; G1 total -1.49 vs -1.76 +- 0.35 |
| P29a | DONE (-0.61; predicted -1.0) | miss explained: stale baseline. P29b / c re-priced from the deployed elf |
| P43 check | MISS (mlp3_a 158-161 vs 137 +- 8) | wrong mechanism; P43b retired, P55 new |
| P53 | DONE differently (cpsched) | II 6 / 12 / 16 where msched failed. On the 80-op attention lists CP-SAT finds nothing in 150-400 s where msched's IMS finds II 30 / 37: give cpsched a warm start (msched's schedule as a hint, `AddHint`) and a smaller horizon |
| P54 | open, **rank 3** | now gates P56 |
| H1-H3 | open | rank 5 |
| P48 / P49 | superseded by P56 | |
| P29b-d, P52 | open, re-priced | ranks 1, 6, 11 |
| P44 / P7-E, P35, P24b, P50, P51, H4 | open | |
| P42 | partly (GEMM stack 0xA00) | EW stack 6,144 vs 4,288 used: 1.8 KB of EW L1 free for P29 tables |

---

## P29b. SSM log_a + exp chain, generated — **M, risk: low** (rank 1)

**Baseline (r12e elf, static):** loga_p 23 bundles x 8 trips, exp_pA 20 x 16, exp_pB 12 x 16, exp_pC 6 x 16, exp_pD 4 x 16, plus
~275 straight-line bundles (5 calls, Peano prologues / epilogues) = **~1,130 cycles per chunk**; round 9's PROF 1,170.

**Procedure.**
1. Bench the deployed `ssm9::loga_exp_a2` on one core (apb `--cc`, 16 vectors): this is the baseline the saving is scored on.
2. One op list per vector (32 lanes): the loga products, the MAC-unit round (magic add), f = z - n, the cubic, 2^n by int16
   `(n << 7) + 0x3F80`, a and 1 - a^2 (the deployed passes' exact sequence). msched needs int16 vector ops and a pass cut at
   5 dm: two kernel-only passes.
3. CP-schedule both passes; bench bit-exact against step 1 with guard zones; then t3 and regress BIT-IDENTICAL.

**Expected (pre-registered):** step 1 reads 1,080-1,200 cycles; the two passes 420-520 together; step **-0.95 +- 0.2 ms** (SSM op
-80 us per block on the timeline); EW text 2,080 B -> ~700 B.

## T1. GEMM text: mm_kstep's record table + asmfit — **S, risk: low** (rank 2)

- mm_kstep (832 B) builds mm_ks's 58 records once per C object; each GEMM core has one C object (C0_0_buff_0), so it runs once.
  Emit the records at design time: design8 reads the placed addresses of a00_0..2, C0_x_buff_0 and acc_hi (input_with_addresses
  .mlir after a first aiecc pass, or pin them with `Buffer(address=)`), computes the table as pack8 / gen_ks do, and passes it
  as an L1 `Buffer(initial_value=np.int32[58])`; mm_kstep becomes the call (~100 B). Bit-identical (mm_ks unchanged).
- asmfit the four GEMM-core asm kernels (round 13's `r13/textmap.py` lists the full-form bundles).

**Expected:** GEMM text 16,288 -> ~15,300 B (-0.7 KB, -0.3 KB); timing unchanged (+-0.05 ms).

## P54. The LUT probes — **S, risk: low** (rank 3)
As in rev 7: (1) does `vldb.4x64` take a base register so the index adds can go (the biased base); (2) vfloor from a bm source.
Now worth more: the fused attention loop reaches II 37 [GEN] with (1) and none up to II 48 without it, so (1) decides P56
between -3.2 and -1.7 ms. Run it before writing P56.

## P56. The attention K phase as one lazy-fast loop — **M-L, risk: medium** (rank 4)

**Baseline:** 571 cycles per (row block, key chunk) unit in situ (r8 stamps; the code is unchanged since r8c; static check in
ANALYSIS 3.1). 1,728 units per core per block.

**Design** (ANALYSIS 3.3): one asm call per (tile, chunk), 36 iterations (6 row blocks x 6 key blocks), kernel-only:
- Q pops on lda (2D post-modify returning to the row block's start; FIFO refilled per key block), K pops on ldb;
- QK^T in today's two-chain order, `vadd.f`, 2 `vconv` -> s;
- exp with the *stored* max m_old (the lazy max makes it independent of this block's max): -m by `vlda.conv` into an fp32
  half (or `vbcstshfl` of g_maxb: no new L1), `vadd.f`, today's LUT line, one push per 64 in the last stage;
- the clamp's `vmin_ge` bound lowered to TAU - 0.25 SU; its r16 mask OR-ed into a per-row-block flag;
- after the loop, flagged row blocks rerun today's C++ (QK^T, max, exact test, rescale, aexp_push) over their P slot; chunk 0
  calls today's chunk_k. Bit-identical by construction.

**Procedure.**
0. CPU: count flagged row blocks per chunk on real scores (r7/expemu captures, TAU - 0.25 bound). Expected <= 3 % after chunk 0.
1. T1 and P54 first. Then attk.py's fused list -> a generator (gen_attk.py on gen_ks's pop handling + gen_gup's push);
   schedule with msched IMS (II 37 with the biased base) and try cpsched with msched's schedule as a hint for II 30-34.
2. One-core bench: one (tile, chunk) against today's chunk_k on recorded Q / K / max state, bit-exact P panels and maxima,
   guard zones on both sides; with a forced flag on one row block to test the fallback.
3. t4 attention bit-identical; regress BIT-IDENTICAL; timeline.

**Expected (pre-registered):** step 0 <= 3 % flagged; bench ~1,480 cycles per (tile, chunk) call at II 37 (6 units);
attention 911 -> ~650 us per block; step **-3.2 +- 0.5 ms** (with P54). Without the biased base and with no fused schedule: the
two-loop-per-row-block fallback, -1.7 ms. GEMM text net ~+0.3 KB.

## P55. Tile-granular A dependency — **S-M, risk: low** (rank 8, new)

**Hypothesis.** mlp3_a's boundary (+15 us) and first-tile wait (22.5 us) are its 248 KB-per-column A load after mlp0's whole drain
group; out_proj's 17.6 us is its 124 KB A load after attention (ANALYSIS 0.1).

**Procedure.** Split mlp0's C drains into 4 task groups by N range (its N tiles are mlp3_a's K chunks; order mlp0's tiles by N
within each column) and make each mlp3_a A-chunk fill (A_RT's chunked runtime tasks) depend only on its group. For out_proj, a
group per head (the attention's per-head O drains). Check the BD / task budget (txnlint if it exists).

**Expected (pre-registered):** timeline mlp3_a 161 -> 135 +- 8 us, out_proj 84 -> 75 +- 5 us per block; step **-0.36 +- 0.12 ms**.

## P57. The C object in two halves — **M, risk: medium** (rank 12, new)
The handoff is 1.3 us per tile (80 us per block): the single 18 KB C object drains at ~8 B/cycle before acc_zero may start the
next tile. Two 9 KB sub-objects with their own locks: the epilogue finishes row blocks 0-8, releases that half (its drain runs)
and does 9-17. The next tile's first k-steps need both halves, so at most half the drain hides. ~-0.5 ms; check that the
pieces map (MM_PIECES) and the stash offsets respect the split.

## T2. Smaller text recoveries — **S-M, risk: low** (rank 13)
V^T (flag 7) epilogue in asm (~860 B of C++ -> ~350 B), a single exit in acc_epilogue (-150 B), `__udivsi3` out of out_begin /
out_end (-144 B), SSM tables (ssm_tables + bias_tab, 640 B) precomputed on the host into the parameter stream.

## P59. Program-memory overlay probe — **S, risk: medium** (probe, new)
Only if a > 1 ms lever is blocked by text after T1 / T2 / P29. One core: the kernel parks on a lock in resident code; the runtime
sequence `npu_blockwrite`s a small function into program memory (tile 0x20000 + offset) and releases the lock; the core calls it.
Measure the write rate (us per KB) and check that the core runs it correctly; a hang ends the probe. A per-block swap of
~3 KB on 24 cores would add 1.7 MB of sequence per step: price it from the measured rate before any design.

## Unchanged from rev 7 / rev 6
H1-H3, P29c (re-priced -1.15), P29d, P52, P44 / P7-E, P35, P24b, P50, P51, H4, P14 (rev7/PROPOSALS.md, rev6/PROPOSALS.md).
