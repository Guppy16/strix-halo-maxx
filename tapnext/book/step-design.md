# TAPNext: step design

The whole TAPNext++ step (patch-embedded frame in, 12 blocks, encoder norm, tracks out) is **one NPU dispatch of 206 ops**
over one fixed layout: 8 columns × 144 rows, three GEMM cores and one elementwise (EW) core per column, every activation
in one DDR buffer under a memory plan, the weights streamed once per step. The block count is chosen at run time by swapping
the instruction stream, not the program. At the pause the step takes **28.42 ms at depth 12** (MEASURED, quiet sweep, 1.808 GHz,
gpu_busy 0; HANDOFF round 25), 23.83 / 19.09 / 14.40 ms at depth 10 / 8 / 6, bit-identical to the round-9 reference
`full12r9a` (DAVIS AJ 70.84, [numerics.md](numerics.md)). Only about 2 ms per frame (conv, heads) runs on the host. The general
mechanism for one dispatch is in [fusion](../../common/book/fusion.md#one-dispatch). This chapter is the TAPNext instance: what is
in the dispatch, how rows and cores are laid out, and how the design got here.

## The step: 17 ops per block

Per block, in plan order (`tapnext/npu/plan8.py` `full_step`). Norm kinds: norm0 and norm1 are RMS, norm2 and norm3 are LayerNorm.

| op | runs on | shape (K, N tiles per core of 64) | notes |
|---|---|---|---|
| norm0 | EW | 3 inputs: residual + the two K-half partials of the previous block's mlp3 | block 0: 1 input (X0) |
| yx | GEMM | K 768, N 1536 (8 tiles) | y half with GELU, x half plain |
| ssm | EW | 12 heads, conv + RG-LRU | [ssm.md](ssm.md) |
| out | GEMM | K 768, N 768 (4) | A = the SSM's z panels |
| norm1 | EW | 2 inputs | |
| up_a, up_b | GEMM | K 768, N 3072 each (16) | gated pair epilogue |
| down_a, down_b | GEMM | K 1536 each, N 768 (4) | the K = 3072 GEMM as two K halves; two partial outputs |
| norm2 | EW | 3 inputs (D1 + D2 + residual) | LN |
| qkv | GEMM | K 768, N 2304 (12) | Q, K and V^T objects |
| attn | GEMM cores | 12 heads × 1,072 keys | [attention.md](attention.md) |
| out_proj | GEMM | K 768, N 768 (4) | A = the attention output |
| norm3 | EW | 2 inputs | LN |
| mlp0 | GEMM | K 768, N 3072 (16) | GELU to panels |
| mlp3_a, mlp3_b | GEMM | K 1536 each, N 768 (4) | K halves again |

That is 11 GEMM ops, 4 norms, the SSM and the attention: 17 per block, 12 × 17 + the patch embedding and the final norm = 206.
Per core and block the 11 GEMMs run 1,296 k-steps (MODEL, [ledger.md](ledger.md)). The plan has 11 GEMMs per block (RESULTS 10, `s7_block`; `plan8.py`); an earlier draft of the table of contents said 12.

## Rows, columns and core roles (design8)

- **Rows.** The step has 1,024 image tokens and 48 query slots = 1,072 real rows, padded to **1,152 = 8 columns × 144 rows** (18 row
  blocks of 8). Column *c* owns rows 144c … 144c + 143, for every op. 80 rows are padding (RESULTS 8; REDESIGN).
- **Cores.** Per column: three GEMM cores (24 in all; each owns a third of an op's N tiles, so a tile is 144 rows × 64 columns, a k-step is
  64 K) and one EW core (8 in all; norms and SSM). Attention also runs on the GEMM cores, 12 heads dealt as head r + 3g to core row r.
  The memtile holds the A slot and the B/C fifos; the shim moves DDR. Why the vector-heavy work sits on the GEMM cores too: the mmul and
  the vector unit are one unit, so a core does one or the other (REDESIGN §1).
- **Tile deal.** yx and qkv deal their N tiles round-robin over the three core rows (P23, −0.85 ms, round 10); mlp0 is dealt the same
  way for IL ([boundaries.md](boundaries.md)). Weights are packed in that order on the host (`tapnext/npu/pack8.py`).
- **GEMM core program.** One table-driven loop (`[k-steps, C objects, tiles per object]` per op in L1) because 16 KB of program memory cannot
  hold 12 unrolled ops. The epilogue kind is a flag word in each N tile's bias object ([gemm.md](gemm.md)). Attention is one more entry of
  the same table (ONE_SEG, round 8: text 16,176 → 14,848 B).
- **EW core program.** `ew_op` is called once per iteration of IRON's endless worker loop and reads its op kind from header word 3 of the
  op's first parameter slot. Round 5 (NOTES 10.1): LLVM had unrolled a 12-block loop into 60 calls (2 KB of the 16 KB).

## Layouts

All in `plan8.py` / `pack8.py`; the reason for each is that **one op's C drain writes exactly the next op's A layout**, so no repack pass exists.

- **Q16** (residual stream and norm outputs): per column `[k-step][row block 18][1,600 B = 1,024 B bf16 (8 rows × 64) | 576 B bfp panels]`. A GEMM
  drains its 1,024 B parts linearly. A norm emits `X' | panel` per k-step with one 3-dim drain. A GEMM's A gathers the panels with one
  4-dim TAP. A residual norm gathers 1-3 streams with one 4-dim TAP (inputs at equal distances; sorted by address, since the sum is commutative).
- **APAN**: regions of epilogue-made k-step chunks (gated hidden layer, mlp0 output), 18,432 B stride equal to the C object, so the junk tail of
  every drained object lands in padding.
- **QKV**: `[head][column][…]` Q, K and V^T objects of 4,608 B (K and V use 3,456 B of it; the tail is scratch while the core holds the object,
  NOTES 19.3). V^T panels share exponents over keys, so the host reference must encode the transposed matrix (NOTES 8.3).
- **SSM area** per block: `[YX (2 S)][conv history R0..R3][h lo, h hi][z]`. The conv history rotates mod 4 with the frame (X1, round 17, `d3c671d`), so
  the state never moves. Four runtime sequences per depth select the phase ([ssm.md](ssm.md)).
- **A slot** in the memtile: 24 chunks × 10,368 B = 248,832 B (K = 1536 ops hold all 24, K = 768 ops half). Driven by runtime tasks (A_RT,
  −2.0 ms, round 10). 3A (round 22, `d14c8cf`) grows it to 36 chunks = 373,248 B for down_b / mlp3_b.
- **W** (weights and parameter objects, 232 MB, RESULTS 12.3) and **ACT** (324 MB at round 3). Weight bytes per step ~215-230 MB, streamed once through the
  B fifos of memtiles 0-2 and multicast across the row (REDESIGN 2.2).

## One dispatch

- Round 2 (design7) proved the pieces: PacketFlows sharing a shim channel, an EW core on each column, GEMM epilogues that emit the next op's
  operands (RESULTS 10: 35.1 ms for one block's 11 GEMMs × 12 blocks). Round 3 (design8) put everything in one runtime sequence:
  **269.1 ms** for 206 ops, `.bin` 7.5 MB, with ~200 serialising sync points at 0.1-0.3 ms each (NOTES 8.4, RESULTS 12.3). The scratch prototype
  before it ran 168 GEMM + 12 attention dispatches on six xclbins at ~2 ms of context switch each (4.4 s per frame, RESULTS 5).
- **Dependencies are explicit.** `plan8` records the last writer of every region. The sequence finishes exactly the producer's group
  before a consumer (a lookahead pipeline alone passed a test "only by timing", NOTES 8.2). Two ops stay in flight (`--lookahead 2`).
- **Budgets the sequence must respect**: ≤ 16 live shim BDs, ≤ 5 tasks per memtile or shim channel before the CP blocks, a fill group a
  multiple of the 768 B EW ring slot ([data-movement](../../common/book/data-movement.md#dma-bookkeeping)). `SEQLOG=1` prints every group wait
  with the op being issued, and is read before any reordered sequence is dispatched (NOTES 28.3).
- 59,720 CP instructions and 421 sync runs per step at r24 (RESULTS 33.2).

## Depth switching and phases

- `plan8.full_step` lays ACT and W out for 12 blocks whatever the depth; ops of blocks ≥ depth are dropped. Every depth shares one layout, one packed
  W and one xclbin: the elfs and CDO of the depth builds are byte-identical (NOTES 10.1). `Step8.set_depth(d)` swaps the instruction buffer: no rebuild,
  no new hardware context. After raising the depth, `reset()` clears the SSM state (stale blocks hold old recurrent state).
- With X1 each depth has 4 phase sequences (frame mod 4), so 16 `.bin` files; `step8` rotates the instruction bo per dispatch (`select(phase)`).
  Phases agree within 0.04 ms (RESULTS 26.3).
- Cost is linear: (28.42 − 14.40) / 6 = **2.34 ms per block plus ~0.4 ms** fixed (MEASURED depth sweep, RESULTS 34.3). DAVIS AJ falls with depth:
  70.84 / 69.90 at depth 12 / 10, and 68.2 / 64.8 for 8 / 6 on the CPU model (RESULTS 8; [numerics.md](numerics.md)).

## The host side

Per frame the host runs the patch embedding, packs it into the mapped ACT buffer, dispatches, and runs the tracking heads. Measured pieces
(MACHINE R7.5, `r11/hostprof.py`, round 11): conv 0.48-1.1 ms, heads 0.6-1.0 ms, query tokens 0.12, X0 packing 0.14, result read 0.08; the
launch floor is 97 µs. Total **2.0-2.2 ms**, and whole step 30.45 ms vs dispatch 28.42 (RESULTS 34.3).

- **Patch embedding on the host** (round 4): torch's bf16 conv (Conv2d 3 → 768, 8 × 8) is 0.28 ms at 8 threads and 0.39 with permute, pos-emb
  and bf16 conversion (RESULTS 14.0; an earlier "0.2 ms" was the 16-thread conv alone). It gives torch's exact numerics, frees 0.7 KB of EW text and
  1.2 ms of device time. X0 = `[conv + pos-emb; query tokens; 0]` (2.8 MB) is written every frame.
- **X0 packing**: torch's strided copy into a persistent buffer whose panel parts stay zero, 0.75 → 0.14 ms (HOST_FAST=2, byte-identical).
- **Heads**: tiny (48 rows), on the CPU. numpy fp32 + `torch.erf` with `OPENBLAS_NUM_THREADS=1` takes 0.72 ms vs torch 0.94 (HOST_HEADS=np, round 13;
  not the default).
- **Variance** comes from the OpenMP pool waking after the 28 ms wait, and from other tenants. A busy box makes dispatch times bimodal (~0.17 ms
  apart), so A/Bs run in one process ([measurement](../../common/book/measurement.md#ab-in-one-process)).
- **CPU load slows the NPU**: 12 threads of FP compute at ~100 W cut the clock from 1.81 to 1.41 GHz (MACHINE R21.2). Cap BLAS/OpenMP threads at 6
  beside the NPU ([M3](../../common/book/negative-results.md#m3)).

## Progress

The step time at the end of each round (depth 12; the table for rounds 12-25 is also in [tapnext/README.md](../README.md#progress-depth-12-step-best-build-at-the-end-of-each-round)).
All kept changes from round 9 on are bit-identical. Timings from rounds 3-11 predate the final clock fix and the A/B-in-one-process rule.

| round | ms | what moved it most | evidence |
|---|---|---|---|
| 3 | 269.1 | design8: the whole step in one program | NOTES 8, RESULTS 12 |
| 4 | 106.8 | correct with real weights; SSM 7.2 → 1.45 ms/block, norm3's 20-bit BD step (6 ms) | NOTES 9 |
| 5 | 89.4 | norms rewritten (0.31 → 0.10-0.13 ms each), depth at run time, lazy attention max | NOTES 10 |
| 6 | 76.3 | epilogues as array passes (−5.3), LUT GELU (−6.0) | NOTES 11 |
| 7 | 71.2 | table exp, first asm GEMM call (−1.9), the 1.81 GHz clock and the core timer | NOTES 12 |
| 8 | 67.9 | A ring pinned off the B bank (−1.5), SSM kernel (−2.4) | NOTES 13 |
| 9 | 58.1 | SSM tables (−2.8), pipelined k-step `mm_ks` (−3.5, `mm_kstep` −2.0) | NOTES 14 |
| 10 | 53.9 | A slot loaded once and runtime-driven (A_HALF −1.35, A_RT −2.0) | NOTES 15 |
| 11-14 | 52.1 → 47.3 | modulo-scheduled asm: GELU, gated epilogues, the SSM sigmoid, the attention K and V loops | NOTES 16-19 |
| 15 | 41.52 | SSM and norm passes in asm, query deal P60g (−1.06), slot-count divides (−1.05) | NOTES 20 |
| 16-17 | 39.94 → 39.16 | SQK, GGK, SSQ, N1K; S2, X1 | NOTES 21-22 |
| 18 | 35.20 | ROT rotating-accumulator k-step (−3.3, `1ffff40`) | NOTES 23 |
| 19 | 33.88 | C drain (ZHI, CSPLIT, EPLR), ROT g0a | NOTES 24 |
| 20 | ~31.8 | op boundaries taken apart: IL, DAPRE, NSEQ, NRES, SRES, AQPRE, AKVPRE (−2.07 A/B) | NOTES 25 |
| 21-22 | 31.30 → 30.78 | CSUB, ZLOAD, E7''; 3A, TXT18, attention IIs | NOTES 26-27 |
| 23 | 28.89 | KTL (−0.82), MSPLIT-P, YXSSM + YXR (−0.79) | NOTES 28 |
| 24 | 28.43 | STATS64, OPPRE2, SSMPRE, CPPRE (−0.48) | NOTES 29 |
| 25 | 28.42 | nothing kept; work paused | NOTES 30 |

## What worked

- **One program, fixed layout, table-driven cores.** Every later lever (asm kernels, on-chip residency, op overlap) changed the sequence or a kernel,
  never the layout of the step. 269 → 28.4 ms (−89 %) with the same 206 ops. (NOTES 8-30)
- **Epilogue-emits-next-A.** No repack pass anywhere between GEMM, norm, SSM and attention: the drain TAPs and the A gathers do the permutation.
- **Run-time depth.** Depth 10 or 8 costs one instruction buffer swap, which is how the accuracy/latency frontier was measured on the device
  (RESULTS 14.5, 34.3).
- **Explicit dependencies plus SEQLOG** found every ordering hazard before it hung the box; timeouts per round fell from 10 (round 3) to 0 (rounds 24-25).
- **Host kept small** (2 ms): the conv is torch's exact bf16 numerics, and the heads are 48 rows.

## What didn't work

- The activation-stationary redesign (REDESIGN.md: X, A and a hidden ring resident in the memtile, 48-row chunk pipelining) was priced at ~62 ms with
  round-5 kernels and ~44 ms with 2x faster vector kernels (EXTRAPOLATED, 4-5 weeks of effort), and **never built**. The incremental route reached 28.4 ms
  inside design8 by speeding the kernels and moving only the boundaries that mattered on chip. Row-chunk pipelining was priced again with the sim9
  list-schedule and came out slower: [C1](../../common/book/negative-results.md#c1) <!-- nr: C1 --> 76.3 → 76.9-77.2 ms (MODEL, never built): every chunk repeats the
  tile and dependent-load costs. The chunked FFN needs the hidden ring in the memtile: see [B11](../../common/book/negative-results.md#b11)
  ([boundaries.md](boundaries.md#what-didnt-work)).
- [E17](../../common/book/negative-results.md#e17) <!-- nr: E17 --> a sliding SSM area with a wrap frame: −0.33 ms per rotating frame, but +232 MB of ACT at N 8
  phases. The mod-4 rotation gave the same gain with no wrap frame. Do not rebuild the sliding form.
- [F1](../../common/book/negative-results.md#f1) <!-- nr: F1 --> the heads on one thread: the next frame's conv slows (OpenMP pool re-spin). Number not recorded.
- [F2](../../common/book/negative-results.md#f2) <!-- nr: F2 --> numpy heads with OpenBLAS's default threads beside torch's pool: 11.7 ms vs 0.72 ms with one thread. Fixed.
- [F3](../../common/book/negative-results.md#f3) <!-- nr: F3 --> host/device overlap levers (H1, H2): not attempted. The product runs in latency mode, where H1 cannot help.
- Not a failure but a trap: a 1 KB-padded EW fill hung the step ([E12](../../common/book/negative-results.md#e12), [data-movement](../../common/book/data-movement.md#dma-bookkeeping)).

## Lessons

- Fix the layout once and make every op's drain write its consumer's A. Then the dispatch is the only place left to optimise.
- Keep cores and sequence independent of the op list (table-driven loops, header-driven EW) so depth is a runtime knob.
- Price whole-step ideas (row chunks, a resident-activation redesign) with measured per-op fixed costs before building them; the incremental route won.
- Account the host: 2 ms of 30.45 is conv and heads, and CPU load beside the NPU changes the clock.

## Going deeper

- [`tapnext/npu/NOTES.md`](../npu/NOTES.md) 8-10 (design8, correctness, depth at run time), [`REDESIGN.md`](../npu/REDESIGN.md), [`HANDOFF.md`](../npu/HANDOFF.md) top block (state and every switch).
- Code: `tapnext/npu/plan8.py` (memory plan, `full_step`), `design8.py` (sequence, `SEQLOG`), `pack8.py` (weight order), `step8.py` (host, `set_depth`, `select`, `reset`).
- Evidence: [RESULTS](../npu/RESULTS.md) 12-14, 33-34 (sweeps); MACHINE R7.5 (host part), R9.1 (the A and K/V flows), R10.2 (memtile map).
- Next: [gemm.md](gemm.md), [attention.md](attention.md), [ssm.md](ssm.md), [norms.md](norms.md), [boundaries.md](boundaries.md), [ledger.md](ledger.md).
