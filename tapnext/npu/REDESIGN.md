# TAPNext++ on the XDNA2 NPU: the activation-stationary redesign (written round 5, 2026-10-09; NOT built)

Status of the current design (design8, round 5): 89.4 ms per step at depth 12, 74.6 ms at depth 10, 7.4 ms per
block, all measured (Turbo, GPU idle). This document proposes the next architecture, estimates what it would buy
and lists the risks, so that the estimate can be weighed against the expected gain. Numbers marked MEASURED come from RESULTS.md 13-14;
everything else is an estimate with its reasoning.

## 1. Where a block's 7.4 ms goes today (MEASURED, block 5 of full12k/full12m, host-polled timeline + differentials)

| part | ms / block | what it is |
|---|---|---|
| GEMM compute at the loop's 63% MAC rate | ~1.65 | 18.3 GMAC (1152 rows x 768 x 20,736) on 24 cores; plain K=1536 ops run at this rate already (down_a 0.16 vs 0.12 ideal) |
| GELU chain in the epilogues | 0.77 | `-DNO_GELU` differential: 89.5 -> 80.3 ms per step |
| other epilogue work (bias, gated products, bfp conversion) | ~0.6 | up_a without GELU 0.43 vs 0.25 ideal; mlp0 0.46 vs 0.25 |
| attention (12 heads x 1072 keys, on the GEMM cores) | 1.37 | by differential (round-5 builds): PV ~0.5, exp ~0.35, the rest QK^T + row max + stores (round 5 already removed 0.42 of rescale and 0.12 of exp) |
| SSM (RG-LRU + conv, on the EW cores) | 1.38 | transcendentals 0.63, other math 0.37, ring I/O 0.38 |
| four norms (EW cores) | 0.5 | ~57 GB/s of DDR traffic: DDR-bound now |
| gaps before ops | ~1.0 | A-slot load (248 KB per column) + first tile + drain tail per op; dependent ops wait for DDR |

Two facts shape the redesign:
1. **DDR round trips cost ~1.5 ms per block, not most of it.** The plain GEMMs already run at the loop rate with
   their A and C in DDR. What DDR costs is the gaps (~1.0) and the norm passes (0.5).
2. **Vector work is the bulk: ~4.1 ms per block** (GELU 0.77 + epilogues 0.6 + attention 1.37 + SSM 1.38) against
   1.65 of GEMM compute. On AIE2P the vector unit and the mmul are the same unit, so a core does one or the other.
   The round-4 hand-off's 20-25 ms assumed this work at its bound; it is not, and no data placement changes it.

## 2. The design

### 2.1 Placement (per column c of 8; a column owns rows 144c .. 144c + 143, as today)
| where | what | bytes |
|---|---|---|
| memtile (512 KB) | residual X, bf16 [rb 18][k-step 12][8 x 64] | 221,184 |
| memtile | A panels of the next GEMM (normalised X or the gated hidden chunk), bfp16 | 124,416 (K = 768) |
| memtile | B forward ring (columns 0-2 forward the weights to all columns, as today) | 18,432 |
| memtile | C staging / hidden-chunk ring (2 x 144 x 64 bf16 objects + a 48-row chunk of hidden panels) | ~100,000 |
| L1 of the 3 GEMM cores (64 KB) | fp32 accumulators of one 144 x 64 tile, A ring (3 x 1152 B), B (2 x 4608 B) | as today |
| L1 of the EW core | norm stats, SSM working set | as today |
| DDR | weights (215 MB, streamed once per step), SSM state (1.1 MB per column per block: conv cache 3 x 144 x 768 bf16 + h 144 x 768 fp32), QKV for the attention (K/V must reach every column) | |
Total memtile use ~465 KB of 512 KB per column: it fits, with little slack (risk 4.1).

### 2.2 Weight streaming order
Weights stay where they are today: every N-tile's 64 x 64 bfp16 tiles stream once from DDR through the B fifos of
memtiles 0-2 and broadcast across the 8 columns (one stream per GEMM core row). Order per block: yx, out, up (gate
and up columns interleaved per tile pair, as pack8 does), down, qkv, out_proj, mlp0, mlp3. 215 MB per step at the
measured ~50 GB/s is ~4.3 ms of DDR time per step, overlapped with compute: weight traffic is not the limit.

### 2.3 Norm handling
Two options, in order of risk:
- **(a) on-chip norm passes (recommended first).** The EW core of a column reads X from its memtile and writes the A
  panels back to the memtile. 221 KB in at the core's ~4 B/cycle port = ~40 us, versus 100-150 us today through DDR,
  and it can run while the GEMM cores work on the previous row chunk (2.4). Numerics stay identical to today.
- **(b) folded norms.** RMSNorm: fold (1 + scale) into the next weight's rows on the host, apply rstd per row in the
  GEMM epilogue (C x rstd). LayerNorm: fold gamma into W and beta into the bias, C = rstd (x W' - mean colsum(W')).
  The row statistics come from the producer's epilogue as partial sums per core, reduced through a small memtile
  buffer. Risk: with A = x quantised to bfp16 before the mean is removed, rows whose mean is large against their
  spread lose precision in the subtraction. This needs a numerics study on the real activations before it is built.

### 2.4 Row-chunk pipelining (the part that actually saves time)
Every op except the attention is row-independent. Split a column's 144 rows into 3 chunks of 48 and pipeline across
chunks: while the GEMM cores run op k on chunk i, the EW core runs the norm or the SSM of chunk i - 1. This overlaps
the EW cores' work (SSM 1.38 + norms 0.5 ms per block today) with GEMM-core work instead of serialising it. The
attention is the barrier: all rows' K and V must exist before any query proceeds, so each block has one full
synchronisation point (qkv -> attention -> out_proj).

### 2.5 FFN chunk fusion
The gated hidden layer is 144 x 3072 per column (497 KB as bfp): it does not fit next to X. Per 48-row chunk it is
166 KB and does: up (768 -> 6144 interleaved, gated GELU epilogue -> hidden panels in the memtile), then down
(3072 -> 768, K = 3072 in two 1536 halves as today, fp32 accumulators in L1: 48 rows x 256 columns per core = 48 KB),
then + residual in the epilogue straight into X. The FFN weights then stream three times per block (once per row
chunk). mlp0/mlp3 work the same way. FFN + MLP weights are 13.3 MB per block, so two extra passes add ~27 MB per
block (~0.5 ms of DDR at ~50 GB/s), overlapped with compute.

## 3. Estimated latency and why
| component per block | today (MEASURED) | redesign, today's kernels | redesign + 2x faster vector kernels |
|---|---|---|---|
| GEMM compute (GEMM cores) | 1.65 | 1.65 | 1.65 |
| GELU + epilogues (GEMM cores) | 1.37 | 1.37 | 0.7 |
| attention (GEMM cores) | 1.37 | 1.37 | 0.7 |
| SSM + norms (EW cores) | 1.88 | overlapped (2.4); ~0.3 exposed | ~0.2 exposed |
| gaps and DDR waits | ~1.0 | ~0.3 (pipeline fill, attention barrier) | ~0.3 |
| **per block** | **7.4** | **~5.0** | **~3.5** |
| **depth 12 step (+1.6 host/final)** | **89.4 (MEASURED)** | **~62** | **~44** |
Reasoning: the GEMM cores' sum is a hard serial sum (vector and mmul share the unit); the EW cores' work overlaps
only if the chunk pipeline keeps both busy, and the attention barrier exposes one chunk's worth per block.
**GPU parity (~22 ms) needs more than this design**: the GEMM loop at ~85% of the MAC slot (1.2 ms/block) and the
vector work moved off the GEMM cores' critical path (e.g. the attention's exp onto the EW cores) on top. The round-4
estimate of 20-25 ms for the redesign was optimistic because it took the vector work at its bound.

What the same effort buys inside the current design (for comparison): 2x faster vector kernels apply to both
designs, so the current design would go from 7.4 to ~5.8 ms per block (~71 ms). The redesign's own gain is
~1.5-2.3 ms per block (gaps, norms, EW overlap), i.e. ~18-28 ms per step at depth 12.

## 4. Risks
1. **Memtile capacity**: ~465 of 512 KB per column, with the B forward ring only in columns 0-2. Any extra buffer
   (a second A slot to hide loads, a 128-slot query layout at 1152 + 80 rows) breaks it. Mitigation: 48-row chunks
   everywhere, X kept as the only full-height tensor.
2. **BD and lock budgets**: static memtile BD chains (48 per memtile, two halves of 24) must express X reads by
   chunk, panel writes, the hidden ring and the C ring. design6 hit 22 BDs per chain; this needs several chains per
   memtile. A plan-level BD allocator with asserts is required before the first build.
3. **IRON expressiveness**: memtile-to-core flows with many hand-built buffers and locks (design6/7 style) work, but
   every new pattern so far cost 1-3 timeouts to get right (round 3: 10). Expect hangs during bring-up.
4. **Attention still needs DDR**: K/V of all 1072 keys must reach every column. Either keep the QKV round trip
   through DDR (as now, ~0.1 ms) or broadcast through the memtiles (memtile-to-memtile is only to neighbours).
5. **Folded norms (2.3 b)** may cost accuracy; option (a) does not.
6. **The SSM state** (1.1 MB per column per block) stays in DDR and must stream in and out around the SSM op of each
   chunk: ~17.7 MB per block (~0.35 ms of DDR), which must overlap with GEMM work.

## 5. Effort
| step | estimate |
|---|---|
| plan + BD/lock allocator for memtile-resident X, A, hidden ring; directed tests per transfer pattern | 1 week |
| GEMM core program reading A from the memtile ring, writing C into X (residual add in the epilogue) | 1 week |
| EW core programs: on-chip norms, chunked SSM with DDR state streaming | 1 week |
| row-chunk pipeline + attention barrier + FFN fusion, whole-step bring-up, real-weights checks, DAVIS | 1-2 weeks |
| **total** | **4-5 weeks** of focused work, with hang risk concentrated in weeks 1 and 4 |

## 6. Recommendation
Do the vector-kernel work first (GELU formulation, attention QK/PV loops and register pressure, SSM sigmoids, the
epilogue loops' `minsize`): it is needed by either design, is measurable op by op with the timeline tool, and takes
the current design from 89 to roughly 70 ms. Then build the redesign for its ~20-28 ms on top, taking the step
towards ~45 ms at depth 12. Parity with the GPU additionally needs the GEMM loop near the MAC-slot bound.
