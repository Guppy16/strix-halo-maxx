# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 9, 2026-10-10 (evening). State analysed: round 13's deployed `full12r13f` (49.21 ms per depth-12 dispatch, Turbo,
gpu_busy 0, BIT-IDENTICAL to full12r9a), its timeline `npu-tapnext/build/tl_full12r13f_d12.json`, its elfs
(`build/full12r13f.prj/elfs_main_core_0_{2,5}`), its lowered MLIR, and the round-13 timing twins (RESULTS 22.2). Revision 8 is
in `rev8/`. Tags: [DOC] vendor source or the deployed binary, [MEAS] measured by the main agent, [INF] model or arithmetic,
[MODEL] sim16 / attsim output. Cycles at 1.81 GHz. All prices are deltas against the deployed r13f, depth 12.

New this revision (`tools/`, outputs in `r14/`; CPU only: one container per CPU replay, no device, no builds):
- `r14/lazymap.py` (container npu-iron-torch, 4 CPU threads, ~1 min): the lazy-max flag / move map per (block, head, column,
  tile, row block, chunk) from the CPU bf16 model on the NPU's 1152 rows, for today's kernel and four numerics policies
  (TAU 96 / 128, key-block interleave, self-score seed). `r14/lazymap.npz`.
- `r14/lazystat.py`: where the slow units fall and what the multicast makes of them (`r14/lazystat.out`).
- **`tools/attsim.py`**: the attention op as an event simulation of the K / V multicast lockstep (per core row: shim fill ->
  memtile depth D -> one multicast send -> 8 columns with 2 L1 slots each; head-group barriers; per-row-block costs per
  variant from the map). `r14/attcal.py` calibrates it on the twins, `attscan.py` (parameter valleys), `attblk.py` (per-block
  check against the timelines), `attrobust.py` (map imbalance), `attlev.py` / `attlayout.py` (levers).
- **`tools/sim16.py`**: sim15's block model on r13f with attsim for the attention (`r14/sim16_r13f.out`).
- r13f disassembly, text and loop maps: `r14/r13f_0_{2,5}.dis`, `textmap_r13f_*.out`, `loops_r13f_0_2.out`; per-op walls
  r12e / r13b / r13f: `r14/tlops_r12e_r13f.out`.

## 0. Scorecard: revision 8 against round 13

| rev-8 prediction | measured | verdict | why |
|---|---|---|---|
| T1: GEMM text -0.7 KB (records) + -0.3 KB (asmfit) | -0.7 KB + -208 B | **hit** | |
| P54(1): `vldb.4x64` has a base-register form (the "biased base") | no such form | **refuted** | rev 8 read a slot name as an operand; the index is an absolute address. The fused loop reached II 35 anyway |
| P56: -3.2 +- 0.5 ms (fallback "updates <= 3 %, +17 cycles per unit") | fast loop alone -2.9 (twin F2); production -0.84 (with l2-depth -0.36) | **miss** (loop right, fallback ~3x off) | three unpriced components: (1) the exact test on every unit (+1.35 ms; rev 8 assumed the clamp mask made it free, the first build ran the C++ test everywhere), (2) chunk 22 (invalid key blocks) forced slow: 4.35 % of units, uniform, (3) **the K / V multicast lockstep**: the slowest column pays per object, and the slow units cluster in two columns (section 1) |
| H3: heads -0.15 to -0.25 ms | -0.22 | hit | |

### 0.1 Method changes (rev 9)
1. **Rare events go through the lockstep.** Any per-unit cost that is not uniform across the 8 columns of a core row is priced
   by the max over columns per K / V object (attsim), never by its mean. "Where" comes from the NPU-layout map (`lazymap.npz`),
   not from a global rate.
2. **Per-block validation.** The attention wall varies 744-920 us per block in r13f; a model that prices rare events must
   reproduce that variation (attblk: corr 0.93-0.97, slope 0.70-0.88). The slope is the correction factor for flag-driven prices.
3. The rev-8 rules stand (price from the deployed build, like-for-like deltas, every component measured or flagged +-100 %).

## 1. The fallback, statistically (`r14/lazystat.out`; CPU map, 3 frames; caches not replayed: every frame is a first frame)

| quantity | value | note |
|---|---|---|
| slow units (flag or chunk 22), per core | **8.8-13.6 %** (mean 10.8) | in situ 8.6-10.9 % (`-DATTK_COUNT`): the map's spread is ~2x the measured one (2.2) |
| flagged | 6.41 %: chunk 0 100 %, chunks 1-21 **2.26 %**, chunk 22 0 (forced anyway) | |
| flagged but the max does not move | **0.055 %** | the 63.75 bound is essentially exact: the flag cannot get rarer without a numerics change |
| by chunk (1-21) | 0.7 % (chunk 1) rising to 4.4 % (chunks 12-13), 0.8 % (21) | keys in the middle of the image |
| by column (1-21) | c0 0.25, c1 0.90, c2 1.36, c3 2.79, c4 2.87, **c5 4.19, c6 4.92**, c7 0.79 % | rows 720-1007 (lower image): spatial clustering |
| by block | b3 4.6 % .. **b10 0.22, b11 0.00 %** | the late blocks almost never move: matches their measured walls (763 / 744 us) |
| slow rb per K object, pass 0 (12 rb per core) | mean over columns 0.265, **max 1.463: 5.5x** | iid at the per-chunk rate would give 4.0x; the rest is the column clustering |
| pass 1 (6 rb) | mean 0.141, max 0.838: 5.9x (iid 4.9x) | |
| slowest column per object (pass 0) | c6 28 %, c5 20 %, c3 15 %, c4 14 %, c2 10 %, c7 8 %, c1 5 %, c0 1 % | |

So the round-13 "3x" is (a) the max over 8 columns of a rare event (~4x even for independent flags), (b) the clustering of the
moves in columns 5-6 (to 5.5x), diluted by the uniform part (chunks 0 and 22: 8.7 % of units at 1:1).

## 2. sim16: the attention as a lockstep simulation

### 2.1 Mechanism (lowered MLIR of r13f) [DOC]
- Per core row r: `B<r>` memtile fifo on memtile r with **8 buffers of 4,608 B** (the GEMM's B object type; the K / V payload
  is 3,456 B), filled by shim r; **one MM2S channel** multicasts each object to the 8 columns' `B<r>L1_<c>` (2 buffers, lock init
  2). A multicast send starts only when all 8 destinations have a free slot: a column runs at most one object ahead of the
  slowest.
- Per core: pass 0 (tiles 0-1, 12 row blocks) and pass 1 (tile 2, 6 rb), each 4 head groups (head r + 3g) x 24 chunks x (K, V).
  Chunk 23 has no keys (calls only); chunk 22 has 2 of 6 key blocks.
- Head-group boundary: the column's A ring (Q objects broadcast to its 3 cores, 3 slots) and the single C object make the 24
  cores start each head group together (a barrier) after pre + begin + end (r8 stamps: 8.56 / 5.61 us).

### 2.2 Calibration (`r14/attcal.out`, `attscan.out`, `attblk*.out`)
Per-row-block costs PREP 68 (bench), V 222 (r8 stamps), send 580 cycles per object (4,608 B at 7.95 B/cycle) fixed; fitted on
the five l2-depth-4 twins (F3 no K compute, F2 fast loop only, F16 + test everywhere, r13c production, F8 slow path everywhere)
with one base:

| twin | measured | model | |
|---|---|---|---|
| F3 | 43.39 | 43.41 | fixes the shim fill: **P_SHIM 1,600 cycles per 4,608 B object (2.9 B/cycle)**; binding only here |
| F2 | 47.14 | 47.11 | **L 400** cycles per row block in situ (bench 390) |
| F16 | 48.49 | 48.49 | |
| r13c | 49.84 | 49.84 | |
| F8 | 52.87 | 52.88 | **E 400** (aexp_push) |
| r13e (holdout, l2 8) | 49.54 | 49.84 | **+0.30: the l2-depth gain is not in the model** (2.3) |
| r13f (holdout, deployed) | 49.21 | 49.36 | +0.15 (49.19 with the thinned map) |

- **T and R are degenerate** (only ~T + 0.1 R is determined): T 66 (bench) -> R ~300; T 130 -> R ~0. Every R-priced lever
  carries +-100 %.
- **Per-block check** (model vs measured attention walls, blocks 1-11): r13f corr 0.93 (slope measured / model 0.70), r13b 0.95
  (0.78). With the map's heavy cores thinned to the in-situ 10.9 % (`Att.thinned(0.109)`, R 450): corr **0.97 / 0.98, slope
  0.80 / 0.88**. This is the primary calibration ("thin"); the raw map is the alternative.
- Lockstep cost in r13f (columns independent vs lockstep, same work): **0.80 ms** (thin) / 0.82 (raw).

### 2.3 What the model does not explain
- **l2 depth 4 -> 8 (-0.30 to -0.36 ms measured)**: in the lowered design the memtile depth only decouples the shim from the
  multicast, and the fitted shim rate binds only in F3, so depth has no effect in the model. Candidates: a producer slowed at
  head-group boundaries (C drains and A sends share the shims and DDR) that a deeper memtile hides, or slack in the multicast
  path the model lacks (the per-block slope < 1 points the same way). PROPOSALS Q1 settles it with two twins.
- The model's absolute attention is ~5 % above the timeline (879 vs 839 us per block); prices are deltas.

## 3. Attention levers (sim16 / attlev; ms per step, thin / raw; each alone against r13f unless stated)

| lever | thin | raw | kind |
|---|---|---|---|
| **P60 query row blocks dealt to the columns round robin** (pieces of 3 rb = half tiles; 6 pieces per column per head) | **-0.84** | -1.00 | bit-identical; DMA only (A fill, memtile send pattern, O drain); 0 B of core text |
| P60 with pieces of 1 / 2 / 6 rb (6 = whole tiles) | -0.96 / -0.91 / -0.54 | -1.14 / -1.08 / -0.62 | the tile form only retargets the qkv Q drain + the O drain |
| P61 chunk 22 fast (fused loop over its 2 valid key blocks + zero P panels for the 4 invalid ones) | -0.33 | -0.33 | bit-identical (invalid P is exactly 0 today too); uniform: 1:1 |
| P62 rescale_rb as an asm O-update loop (R / 3) | -0.70 | -0.59 | bit-identical; R +-100 % |
| P62b slow exp by re-running attk with the new max (aexp_push removed) | 0.00 | 0.00 | bit-identical (attk's P is op-for-op aexp_push's); **-464 B text** |
| P50 V phase in asm (222 -> ~90 per row block; chunk_v is a 42-bundle loop per dim pair today) | -1.19 | -1.19 | bit-identical; uniform; pass 1's send / shim limits take ~0.25 of the static 1.45 |
| P63 36-iteration K loop (L 400 -> 285) | -1.26 | -1.26 | bit-identical; uniform |
| P65 chunk 0 QK-only loop | -0.13 | -0.13 | needs a second asm loop (text) |
| asm aexp (E 400 -> 250) | -0.43 | -0.52 | superseded by P62b (text) and P60 (lockstep) |
| 3 / 4 L1 K / V slots | -0.66 / -0.76 | -0.67 / -0.77 | no L1 (4.6 KB per slot); moot after P60 |
| defer flagged units past the lockstep (a V copy per deferred chunk) | -0.80 | -0.82 | no L1; moot after P60 |
| P64 3,456 B K / V objects (shim 1,600 -> 1,200) | -0.04 | -0.04 | the producer does not bind even after P50 + P63 (-0.05) |

**After P60 the lockstep is gone**: columns-independent minus lockstep = 0.06 ms (from 0.80). Every "cut the lockstep" idea
(more slots, deferral, cheaper exact test, rarer flag) is then worth < 0.1 ms; the slow path's own cost is what is left.

Stack (each adds to the previous; thin / raw): P60 -0.84 / -1.00; + P61 -1.17 / -1.33; + P62 -1.55 / -1.62; + P50 -2.82 / -2.89;
+ P63 **-4.05 / -4.12**; + P64 -4.10. Attention 10.07 -> ~6.0 ms per step.

Numerics-changing options (not bit-identical; DAVIS gate), from the map's policies (`lazymap.out`, `attlev_thin.out`):

| policy | flagged after chunk 0 | chunk 0 | alone | after P60 + P61 |
|---|---|---|---|---|
| today | 2.26 % | 100 % | | |
| TAU 12 nats (table to +13) | 0.84 % | 100 % | -0.48 | ~-0.2 |
| TAU 16 nats (table to +17, +1 KB of tables) | 0.25 % | 100 % | -1.12 | ~-0.4 |
| keys dealt to chunks by 8-key block, stride 23 | 0.93 % | 100 % | -0.92 | ~-0.2 |
| **m seeded with q_i . k_i** (the row's own key; rows >= 1072 need none) | 0.27 % | **9.7 %** | -1.40 | ~-0.7, minus its data path (~0.4 +- 0.4) |

The self-score is nearly every row's max. Its key is in this column's own K objects (QKV slot 3i + 1 of (h, c)), so the seed
can come through the A slot beside Q and an 8x8 diagonal mmul per row block at qcopy (exact: same operands and MAC order as
the loop). Parked: after P60 + P61 it nets ~-0.3 +- 0.4 ms for a numerics change.

## 4. Program memory (r13f; `r14/textmap_r13f_*.out`)

GEMM 16,384 / 16,384 B (867 all-nop bundles = 1,874 B, mostly delay slots); EW 16,224 (160 free, unchanged). GEMM attention
code: chunk_k_fused 1,360, attk 896, attn_begin 816, attn_end 784, chunk_v 608, rescale_rb 560, aexp_push 464, attn_qcopy 400,
attn_chunk_k 304, attn_chunk_v 256, exp32 304, rows8 112 = 6,864 B.

| candidate (GEMM) | frees | cost / risk | note |
|---|---|---|---|
| T3a aexp_push removed: the slow path re-runs attk with the new max (P62b) | **464** | S / none | bit-identical; the K object and the scores are still in L1; time 0 (L = E) |
| T3b attn_chunk_k / attn_chunk_v folded into their kernels (tile loop + nkb inside chunk_k_fused / chunk_v as the entry points) | ~250 | S / low | the wrappers are 560 B, mostly delay-slot nops and callee saves |
| T3c acc_zero as a 3-bundle zero-overhead loop (today 15 unrolled full-form bundles) | ~230 | S / low | the ATTK_COUNT build already ran it un-unrolled; time it (0.49 us per tile) |
| T3d attn_qcopy as one pointer-bumped loop over 54 vectors | ~200 | S / none | minsize today, 400 B |
| T3e attn_begin: one zero loop over the contiguous O / osum span | ~250 | S / low | |
| T3f mm_kstep: one C object per core, so the records can be fully constant (mm_kstep -> tail call of mm_ks) | ~200 | S / low | |
| P62 rescale_rb as asm O-update (exp32 stays C++) | ~150-250 | M / low | and -0.38 to -0.70 ms |
| P50 chunk_v in asm | ~150 | M / low | and -1.19 ms |
| exp32 -> the exp table in rescale_rb | 300 | S / numerics | DAVIS-gated, last resort |

T3a-f: ~1.6 KB with no timing risk; enough for P61 (~60 B), P63 (+-200 B) and the asm preambles.

## 5. Other parts (mechanisms unchanged; re-based on r13f, sim16)
- GEMM k-steps 18.2 ms (compute-bound, 2,051-2,066 cycles), SSM 9.0, norms 6.35, boundaries 1.8, handoff 1.8, epilogues 1.7.
- r12e -> r13f per op: attention 911 -> 839 us per block; every other op within +-5 us (`tlops_r12e_r13f.out`). The "norms
  +0.37 ms" of RESULTS 22.1 is not in these timelines (norm0 113 -> 117 us, the others +-3).
- SSM (P29b -0.95 +- 0.2, P29c -1.15, P29d -0.84 x 0.85, P52 -0.78), P55 -0.36, P57 ~-0.5, P35 + P24b -0.98, P7-E, P51: rev 8
  prices stand (their code is unchanged).

## 6. Ceiling (sim16, depth 12)

| added lever | model ms | from 49.21 |
|---|---|---|
| P60 + P61 + P62 | 48.72 | 47.7 |
| + P29b | 47.78 | 46.7 |
| + P50 | 46.51 | 45.4 |
| + P63 (+ P64) | 45.23 | 44.2 |
| + P29c + P29d + P52 | 42.44 | 41.4 |
| + P55 | 42.08 | 41.0 |
| + P7-E high | 38.96 | 37.9 |
| + P35 + P24b | 37.98 | 36.9 |
| + P51 | 37.02 | 36.0 |
| + P57 | 36.54 | **35.5** |
| + k-step 2,055 -> 1,600 | 32.20 | **31.1** |

The model base is the polled timeline (50.28 ms; the timed median is 49.21), so the right column shifts by the measured base.
Depth 10 at the ceiling ~26 ms. Rev 8: 31.7 / 36.0; the attention now reaches ~6.0 ms instead of ~7.8.
P14 (two frames in flight): today ~37 ms per frame (x1.35), after the attention stack ~33, at the ceiling ~25 (latency 2x).
