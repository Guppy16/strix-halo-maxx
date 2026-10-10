# ANALYSIS.md — data movement, bubbles and a scheduling model for the TAPNext++ step

State analysed: round 5, `full12k` / `full12m` (89.4 ms per depth-12 step, Turbo, gpu_busy 0).

Inputs:
- the host-polled timelines `build/r5_tl12k.out` (GELU on) and `r5_tl12kng.out` (`-DNO_GELU`);
- the depth sweep `r5_sweep_m.out`;
- the decoded instruction stream `full12k.bin` (`tools/txnstats.py`, `tools/txnq.py`);
- the plan and the kernels (plan8 / design8 / mm8 / ew8).

The tags [MEAS], [INF] and [DOC] mean what they mean in MACHINE.md.

## 1. Where the 89.4 ms go

Each op's cost is its last write minus the previous op's last write. The model in §2 reproduces these costs
(per op rms 7.9 us; step 88.6 ms predicted vs 89.5 measured). The SSM's last write is invisible to the timeline, so
its cost is out's measured cost minus out's modelled cost.

| part | per block (ms) | per step (ms) | share | binding resource [INF] |
|---|---|---|---|---|
| GEMM k-step loops (1,296 tile-k-steps per core per block) | 2.41 | 28.9 | 32% | **A input port of each GEMM core** (§2) |
| SSM (row-5 cores only) | 1.30 | 15.7 | 18% | compute on the EW core. Ring traffic alone is 0.28 ms (SSM_COPY_ONLY build `full12jcp`) |
| attention | 1.37 | 16.4 | 18% | unknown: no cycle counts inside it (P2) |
| per-tile fixed cost (92 C objects per core per block x 8.9 us) | 0.81 | 9.8 | 11% | C drain (18 KB at 4 B/cycle = 3.2 us, C single-buffered) + bias epilogue + acc_zero |
| GELU epilogues (36 tiles per core per block x 18.2 us) | 0.65 | 7.9 | 9% | core compute (2-chain gelu) |
| norms (4 x 0.12-0.15) | 0.54 | 6.5 | 7% | input port of the EW core (~4 B/cycle) |
| latency before a dependent GEMM (7 per block x 40 us) | 0.28 | 3.4 | 4% | the A-slot load: one 248 KB BD, `full` released only at its end |
| final norm, host side | | 0.15 + ~2.4 | | |

How the parts are used:

| resource | busy | idle |
|---|---|---|
| GEMM cores (24) | ~64 ms | ~25 ms: during the norms, the SSM and the dependent loads |
| EW cores (8) | ~22 ms | ~67 ms |
| DDR | 1.67 GB per step = 18.7 GB/s average (~1/3 of the read peak) | |

The DDR total comes from decoding the instruction stream [MEAS]:
- A loads + EW inputs (shim MM2S ch0): 663 MB;
- weights + bias objects + attention K/V (MM2S ch1, shims 0-2): 294 MB;
- C drains (S2MM ch0): 446 MB;
- EW outputs (S2MM ch1): 263 MB.

The engine is also not compute-bound. The step's 220 GMAC would take 19.9 ms on 24 cores at the measured 1.59
bundles/MAC, or 15.0 ms on 32 cores. Today the step is bound by per-core ports and by serialisation.

## 2. The GEMM cost law (fitted and validated)

For one GEMM op (cores split N three ways, nt = N-tiles per core, g = GELU tiles per core):

```
T_op = L_dep [if its producer is the op right before it, or an EW op]
     + nt * (K/64 * t_ks + t_tile) + g * t_gelu
```

Least squares over blocks 1-10 of the no-GELU build (`tools/sim8.py`) gives:

| constant | fitted value |
|---|---|
| **t_ks** (one k-step of one tile) | **1.857 us** |
| L_dep | 40.4 us |
| t_tile | 8.86 us |
| t_gelu | 18.2 us per tile (from the GELU build minus the no-GELU build) |

The per-op residual is 7.9 us rms. Predictions:

| case | predicted | measured |
|---|---|---|
| step, GELU build | 88.6 ms | 89.5 ms |
| step, no-GELU build | 80.8 ms | 80.3 ms |
| depth 10 | 73.9 ms | 74.64 ms |
| depth 8 | 59.1 ms | 59.73 ms |
| depth 6 | 44.4 ms | 44.87 ms |

The depth rows are not independent tests (the model is linear in depth). The no-GELU step total is the independent
check.

**t_ks sits on the A-port floor.** One k-step moves 144 rows x 64 k of bfp16 (10,368 B) into every GEMM core
through one stream, and at 4 B/cycle and 1.43 GHz that takes 1.813 us. The fit is 2.4% above it. The arithmetic
compute floor is 9 calls x 128 MAC instr x 1.59 = 1,832 cycles = 1.28 us. Matching 1.857 by compute would need
~115 cycles of overhead per mm_acc call. That is possible, and only the twin experiment P1 can separate the two
hypotheses.

If the port bounds it, the tile shape is the problem. Per core per k-step:

| tile m x n (fp32 acc bytes) | A bytes -> cycles @ 4 B | B bytes -> cycles | compute cycles @ 1.59 | bound |
|---|---|---|---|---|
| **144 x 64 (36 KB), today** | 10,368 -> **2,592** | 4,608 -> 1,152 | 1,832 | A port, 41% over compute |
| 96 x 96 (36 KB) | 6,912 -> 1,728 | 6,912 -> 1,728 | 1,832 | compute (balanced) |
| 128 x 72 (36 KB) | 9,216 -> 2,304 | 5,184 -> 1,296 | 1,832 | A port |
| 144 x 64 with A on two streams (§5.3) | 5,184 -> 1,296 per stream | 4,608 -> 1,152 | 1,832 | compute |

The per-tile constant (8.9 us) is also inflated by the port bound. While the core runs the epilogue and the C drain,
its 3-deep A ring (3.4 KB) fills in ~0.6 us and then the A stream stops. Port time lost at tile boundaries is never
recovered.

## 3. Bubbles: what the timeline's "gap" column means, and the real idle time

- **"gap" (first write minus the previous op's last write) is not idle time.** It contains the op's own latency to
  its first output.
  - GEMM: the A-slot load (~40 us when dependent) plus the whole first tile (22-45 us).
  - Attention: the 0.25 ms "start gap" is mostly the first (pass, head) unit of work (1.37 ms / 8 units = 0.17 ms),
    not a stall.
  Use per-op costs (last to last) and a model, or in-kernel stamps (P2), not gaps.
- **Real idle periods [INF from §1-2]:**
  1. **GEMM cores during EW ops:** SSM 15.7 + norms 6.5 = 22 ms per step. Every EW op is bracketed by
     `finish_all()`: the sequence drains everything before the op and waits for its drain after it. Nothing of the
     next GEMM can start, not even its weights (B) or a dependency-free A load.
  2. **Dependent A-slot load:** 7 x 40 us per block (3.4 ms per step). The slot is one BD that releases `full`
     only when all 248 KB have landed, although the cores need the first 10 KB (k-step 0, all stripes) to start.
     The slot layout is k-step-major (`[K/k][rho][...]`), so releasing per k-step chunk would let the cores start
     after ~2 us.
  3. **Tile boundaries:** ~6-9 us per tile in which the A port idles (§2), 9.8 ms per step.
  4. **EW cores during GEMM and attention:** ~67 ms per step.
  5. **K=768 double read:** every K=768 op (7 of 11) loads its 124 KB of A twice into the 248 KB slot, so each half
     holds the whole A. This doubles that op's A DDR traffic and its load latency.
  6. **Head-of-line blocking in the command processor:** up to 16-19 pushes on one channel between waits against a
     queue of 4. The 6,960 `maskpoll`s show the sequence stalls on full queues and delays the other channels'
     pushes. Its cost is not measured; the simulator in §6 bounds it and P3 lints it.
- **Overlaps that are legal today without changing kernels:**
  1. The B (weight) stream of the GEMM after an EW op could be pushed before the EW op, filling the B fifos (depth
     4 in L2, 2 in L1) ahead. Small (~10-20 us per EW op).
  2. Independent GEMM pairs (up_a/up_b, down_a/down_b, mlp3_a/mlp3_b) already prefetch: down_a/down_b/mlp3_b have
     no L_dep in the fit.
  3. Norm statistics in the consumer (folding, §5.2).

## 4. Critique of the round-4 "activation-stationary" sketch (HANDOFF §4)

The sketch says DDR round trips (~1.5 GB per step) put a ~40 ms floor under design8; keep the residual stream and the
per-block intermediates in the memtiles and stream only the weights; the step then becomes compute-bound at
~15-20 ms.

1. **DDR is not the binding constraint now [MEAS].** The decoded stream moves 1.67 GB per step: 18.7 GB/s over
   89 ms, a third of the peak. The 40 ms "floor" assumed 37-39 GB/s, which design8 never reaches. DDR becomes
   binding only below ~30 ms per step (1.67 GB / 55 GB/s = 30 ms), so the traffic must fall eventually, but it is
   not the next wall.
2. **Residency does not remove the dominant traffic.** The binding transfer is memtile -> core: every N-tile
   re-streams A from the memtile into each core at 4 B/cycle (§2). An activation-stationary design still feeds the
   cores from the memtiles, so the GEMM loop (29 ms per step) does not shrink at all unless the tile shape or the
   number of A streams changes. The "15-20 ms compute-bound" figure assumes a compute-bound loop the 144 x 64 tile
   cannot reach.
3. **Capacity [DOC + arithmetic], per column (144 rows), against 512 KB of memtile:**

   | tensor | size per column |
   |---|---|
   | residual X (bf16) | 221 KB |
   | A slot of a K=1536 op (bfp16) | 248 KB |
   | the FFN / MLP hidden (144 x 3072 bfp16) | 498 KB |
   | fp32 partial sums for a fused up -> down (144 x 768) | 442 KB |

   So residual + slot already uses 469 of 512 KB. The hidden activation and the fused-FFN partial sums do not fit.
   Neighbour memtile access (channels 0-3) does not help, because every column is equally full. The FFN hidden must
   go through DDR, or the FFN must be chunked over rows. Row chunking re-streams the weights once per chunk
   (215 MB x chunks; 3 chunks ~13 ms of DDR, which can overlap with compute). Folding the K-chunks with bf16
   partials costs accuracy (moondream: bf16 C every 64 K cost ~1.4 points).
4. **What the sketch gets right:**
   - Folding norms into GEMM prologues and epilogues removes the EW ops from the critical path; that is the
     biggest idle period on the GEMM cores.
   - Fusing more work per op removes per-op fixed costs.
   - Both can be done inside design8's DDR-based structure (§5.2), without residency.
5. **Recommended order:**
   1. pipeline the A slot per k-chunk (§5.1);
   2. balance the GEMM ports (§5.3, after P1);
   3. take EW work off the GEMM cores' critical path (§5.2 / §5.4);
   4. cut DDR traffic when the step approaches ~35 ms: no K=768 double read (-144 MB), no junk C windows, norms
      folded (-~300 MB).

## 5. Data-movement designs, ordered by expected value

### 5.1 k-chunk-streamed A slot (removes L_dep and the K=768 double read)

- **Load.** One memtile S2MM BD with `BdIteration` over the K/64 chunks of 10,368 B. Per chunk it acquires
  `empty` by nt and releases `full` by nt. Lock values must be <= 63, and nt <= 16 is fine.
- **Send.** One MM2S BD iterating over the same chunks, acquiring `full` by 1 and releasing `empty` by 1 per chunk.
  It re-walks the chunks once per N-tile: the BD is re-armed nt times by a repeat count, or the send is chained.
- **Result.** Cores start ~2 us after the load starts instead of ~40 us. K=768 ops load 124 KB once. The 22-BD
  memtile budget gets easier (2 BDs instead of 13).
- **Gain [INF, sim8].** -3.0 ms per step, plus -144 MB of DDR. With lookahead, an independent op's first chunks
  can also arrive while the previous op finishes.
- **Risk.** Lock arithmetic. Prove the lock trace in the simulator first (P4); one dispatch with a short timeout.

### 5.2 Fold the norms into the GEMMs (removes 48 EW ops per step)

- **RMSNorm before a GEMM.** `rms(x) @ W = rstd_row * (x @ (diag(s+1) W))`.
  - Fold `diag(s+1)` into the packed weights.
  - Apply `rstd_row` to the fp32 accumulator rows in the epilogue (a row scale: 144 values per tile).
- **LayerNorm.** `LN(x) @ W = rstd_row * (x @ W' - mu_row * colsum(W')) + (beta @ W + bias)`, with
  `W' = diag(gamma) W`.
  - This is one extra rank-1 term in the epilogue.
  - `colsum(W')` goes with the bias object (B stream, shared by all columns).
- **The row statistics** (`sum x`, `sum x^2` per row) are needed before the consumer's first epilogue.
  - They are per row, so per column: they cannot ride the B stream, which is broadcast across columns.
  - Two options:
    - The producer epilogue (the GEMM writing the residual sum) emits partial row sums per N-tile, and the
      consumer's A slot carries them as a small per-stripe header.
    - A tiny EW pass computes them. Statistics are reads only: 221 KB per core at 4 B/cycle = 39 us, against
      120-150 us for the full norm today, and the EW core then writes nothing.
- **The residual sums** (`R1 + R0`, `D1 + D2 + R2`, ...) must still be formed. The simplest is to add the residual
  in the producer's epilogue. The residual tile (18 KB) arrives through a C-sized object on the B port, which has
  44% slack.
- **Numerics.**
  - Different rounding order from torch's bf16 path: `x` is quantised to bfp16 before the row scale instead of
    after it.
  - The per-8-value exponent makes the two near-equivalent, but this must be measured: the numerics harness (P8)
    first, then DAVIS.
- **Gain [INF].** Norms 6.5 ms -> ~1.5 ms (stats-only EW passes) or ~0 (producer stats). The `finish_all` brackets
  go with them.

### 5.3 Port-balanced GEMM: A on two streams through neighbour memory (if P1 confirms the A-port bound)

Each column has GEMM cores in rows 2, 3 and 4.
- **Split each A sub-tile into halves X and Y** (e.g. stripes or k-sub-blocks).
- **Placement:**
  - X goes into row 3's memory, which rows 2 (its north neighbour) and 4 (its south neighbour) can both read.
  - Y is broadcast into rows 2 and 4; row 3 reads Y from either.
- **Result.** Each core's A S2MM carries half the bytes: 1,296 cycles per k-step < 1,832 compute, so the loop
  becomes compute-bound.
- **Cost.** A cross-tile lock handshake per ring slot:
  - row 3 acquires `full` from its DMA and releases per-reader locks;
  - the readers release `done`;
  - row 3 releases `free` after both.

  The memtile needs a second A MM2S channel (it has 6).
- **Gain [INF, sim8 what-if].** t_ks 1.857 -> ~1.28-1.35 us: **-8 to -9 ms per step**. The deeper win: the port is
  no longer binding, so kernel work (fewer bundles per MAC, cheaper epilogues) starts paying.
- **Alternative.** Re-map to 96 x 96 tiles (12 row groups x 2 N-halves). This is awkward with 144 rows per column
  and would break the plan's layouts.

### 5.4 Overlap EW and GEMM work (only after 5.1-5.3)

The block is a strict residual chain. Only attention mixes rows; everything else is per row, and the SSM is
recurrent over frames, not tokens. Overlap therefore needs two independent pieces of work.
- **Row micro-batches (two halves per column).**
  - The EW core works on half 1 while the GEMM cores work on half 0. This could hide most of the 22 ms of EW time.
  - m = 72 breaks the 16-row stripe (rho 4.5), and smaller m makes the B port bind (m=48: 1,152 B-cycles vs 864
    A-cycles per k-step). Feasible as 80 + 64 rows.
  - The simulator must price it before anyone builds it.
- **Frame pipelining (throughput, not latency).**
  - Block b of frame t+1 needs only block b of frame t (the SSM state) and block b-1 of frame t+1.
  - Two frames in flight, phase-shifted by half a block, give the GEMM cores one frame's GEMMs while the EW cores run
    the other frame's SSM and norms.
  - fps could rise toward 1 / max(GEMM-core time, EW time) ~ 1/64 ms vs 1/89 ms, at a per-frame latency of ~1.3x
    the step. This suits the fps side of the Pareto front, not the latency side.
- **Put the EW work on more cores.**
  - Norms are port-bound per core, so 4 cores per column would make them 4x faster.
  - The obstacle is the 16 KB program memory of the GEMM cores. Measure the headroom first: the disassembly auditor
    (P9) reports text size per entry point.

## 6. A scheduling model / simulator

`tools/sim8.py` (pure python, runs on the host) is the validated first cut:
- it holds the cost law of §2, the measured EW and attention constants, and what-if overrides;
- per-op rms is 7.9 us; the step total is within 1% on the GELU and no-GELU builds;
- each what-if (depth 12, alone, ms) moves the step from 88.6 to:

| change | step | delta |
|---|---|---|
| port-balanced GEMM (t_ks 1.28 us) | 79.7 | -8.9 |
| SSM -> 0.5 ms per block | 79.0 | -9.7 |
| attention -> 0.7 ms per block | 80.6 | -8.0 |
| C drain overlapped (t_tile 3 us) | 82.2 | -6.5 |
| norms 4x faster or folded | 83.8 | -4.9 |
| GELU halved | 84.7 | -3.9 |
| k-chunk A slot | 85.7 | -3.0 |
| one call per k-step (only if call-overhead bound) | 87.9 | -0.7 |

All of them together give **43.8 ms**, still serial. Getting below ~40 ms needs the overlap of §5.4 or fewer EW ops.

**Next version (P4): an event-driven simulator, specified so the main agent can build it.**

Resources:
- the command processor: in order; push, poll and wait semantics with queue depth 4;
- 16 shim channels at 6.4 GB/s each with a DDR cap of 55 GB/s;
- per column: the memtile A slot with its locks;
- per GEMM core: an A ring of 3, a B fifo of 2 and a C object of 1;
- 8 EW cores with in/out rings of 20/12 slots of 768 B.

Each op is a list of tasks taken from the decoded `.bin` (TXN events: push sizes per channel, waits). Per-object
compute costs come from P2's stamps (or the fitted constants above).

Outputs:
- per-op start and end;
- per-resource utilisation;
- critical path attribution.

Validation targets:
- (a) the r5_tl12k per-op costs within 10%;
- (b) the step within 3%;
- (c) the no-GELU and SSM_COPY_ONLY builds without refitting;
- (d) one hold-out schedule change (e.g. lookahead 1 vs 2, or EPI_OVERRIDE) predicted before it is measured.

Only after (d) should it be trusted to choose between §5.1-5.4.
