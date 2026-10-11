# moondream NPU GEMMs: design2, design3 and design6

The encoder's four GEMMs (qkv, attn_out, fc1, fc2; 652 GFLOP per frame with the projection) run on the NPU as **one program for all
27 blocks in 33.8 ms** (MEASURED, [`p6`](../npu/p6/), Turbo). That is still slower than the GPU's 20.7 ms, and the reason is not
the kernel: the frame is **DDR-bound**, 1,317 MB per frame at 37-39 GB/s (MEASURED, MD RESULTS phase 11). The path there had three
designs. design.py streamed A and B from DDR and hit the DDR cap. design2 tried to keep A in the memtile with runtime tasks and
deadlocked. design3 did it with a static chain, and design6 squeezed the whole block into the memtile's buffer-descriptor (BD) budget.
The numerics are bit-exact to bf16 throughout (rms 1.65e-3, the bf16 rounding floor). General mechanics are in
[gemm](../../common/book/gemm.md) and [data-movement](../../common/book/data-movement.md); the clock-free unit here is bundles per MAC.

## Setup: shapes and tiles

M = 729 pads to 768 = 8 columns x 96 rows. Each core owns a 96x96 output tile, bfp16 A and B (host-packed, round to nearest), an fp32
accumulator over the whole K, and a bf16 C. Column c holds M-tile c; the four rows split the N tiles. Per frame: patch embedding 1.0 GFLOP,
27 x (qkv 5.81 + attn_out 1.94 + fc1 7.23 + fc2 7.23) and the projection MLP 52 GFLOP (MD RESULTS phase 1).

## Phase 1: the generic design is 20-27 % of peak

mlir-aie's asymmetric-tile-buffering GEMM (bf16 A converted in the core, bfp16 B) at the encoder shapes: 9-13 TOPS (20-27 % of the ~50 TOPS
peak), **55.0 ms per frame in 111 dispatches** (MEASURED, `moondream/npu/atb/`, MD RESULTS phase 1) against 20.7 ms on the GPU. Two lessons
shaped everything after: switching between two xclbins costs ~2.8 ms (MEASURED, `atb/ctx_switch.py`), so the encoder must be one
xclbin; and the design keeps bf16 C between K chunks, which the GPU emulation showed costs accuracy ([numerics](numerics.md)).

## Phase 2a: fp32 accumulator, and the DDR bound (design.py)

The new kernel (`p2/mm_f32acc.cc`) keeps the accumulator fp32 for the whole K. The L1 cannot hold an fp32 96x96 tile next to a bf16 output
tile, so the accumulator is split: 48 rows inside the output buffer, 48 in a static buffer. Result: 0.0000 % of outputs differ from
bf16(exact), but qkv takes 463 us, no faster than phase 1's 486 (MEASURED, [p2](../npu/p2/)). Why: DMA probes (`p2/bw.py`) found ~6.4 GB/s
per shim stream and ~55-60 GB/s aggregate, and a build with no compute at all still took 515-549 us. qkv moved 18.8 MB (A re-read once per
N pass) at ~50 GB/s. The inner loop was 68 bundles per 32 MACs (47 % of core peak). The early DDR numbers were read as a rule; the later
caps are in [data-movement](../../common/book/data-movement.md#the-ddr-caps) ([C12](../../common/book/negative-results.md#c12) <!-- nr: C12 -->).

## design2 and design3: A stays on chip

Hypothesis: keep each column's A tile in the memtile and re-send it per N tile; qkv traffic drops 18.8 -> 10.8 MB.
- **design2** drove the memtile with per-op runtime tasks. First run: ERT timeout and an amdxdna TDR. The lowered MLIR looked right
  (no channel or lock clash); the suspect is the runtime memtile-task path itself, which nothing in the mlir-aie examples uses. It was never
  debugged (MD NOTES phase 2a-2b).
- **design3** replaced it with a static chain: two 124 KB slots per memtile, the memtile MM2S chain is always [slot0, slot1] and loops,
  and "repeat for this op" is a lock value on the load BD. Nothing can wait for data that never comes. First dispatch was correct and bit-exact.
  The verification step that mattered: reading the lowered MLIR for lock values before the first run.

| op (real K x N) | design3 us (median) | DDR MB | GB/s | GPU us |
|---|---|---|---|---|
| qkv 1152x3456 | 446 | 10.8 | 24 | 195 |
| attn_out 1152x1152 | 223 | 4.3 | 19 | 80 |
| fc1 1152x4304 | 558 | 14.0 | 25 | 251 |
| fc2 4304x1152 (A streamed) | 504 | 18.7 | 37 | 223 |
| proj_fc1 2304x8192 | 1,564 | 36.9 | 24 | 878 |
| proj_fc2 8192x2048 (A streamed) | 1,467 | 67.2 | 46 | 1,051 |

MEASURED, Default power mode, ~80 us launch included, [p2b](../npu/p2b/). Frame: 22.6 + 10.6 + 3.0 = **36.2 ms in 4 dispatches**
(27 blocks per dispatch; 49.8 ms as 111). The K = 1152 ops are no longer DDR-bound (19-25 GB/s). Only fc2 and the projection, whose A
is too big for the slots and streams per N tile, are near the cap.

## Where an op's time goes

A twin of qkv with the kernel compiled out (NOCOMPUTE) takes 369 us of 446: the floor is data movement plus launch, not MACs. The
per-core input port (~4 B/cycle) must carry 1.1 MB of B and 1.1 MB of A per qkv, ~176 us each in parallel. Later breakdown (417 us, 1.4 GHz
estimate): launch 80, kernel 253, accumulator zero/store 13, C-tile stall ~26, A-load bubble ~20, other ~25 (MD NOTES phase 2c).
Each of these changed qkv by 2 % or less: A ring depth 6, B depth 3 in L1, B depth 4 in the memtile, C over 16 shim streams (a second B
stream per row is impossible: a core has two input channels, A and B). Removing the A handshakes saved ~12 us.

## The kernel

| version | bundles per MAC | share of core peak | evidence |
|---|---|---|---|
| phase 2a 2x2 blocks | 2.13 | 47 % | [p2](../npu/p2/) |
| BLK12 (1x2 blocks, prefetched accumulators, A hoisted into registers) | 1.69 | 59 % | [p2b](../npu/p2b/) |
| BLK12 unrolled by 2 | **1.59** | **63 %** | [p3](../npu/p3/) |
| hand schedule, II 26 | 1.83 | 55 % | [p11](../npu/p11/) |

Per frame this was 36.2 -> 33.2 (Default) -> 29.7 ms (Turbo, four dispatches looped; MEASURED). Turbo moved the core clock by ~3 %; it mostly cut dispatch latency
(33.2 -> 31.4 ms for the same build). The "slow mode" (sub-millisecond dispatches finishing at ~1 ms multiples for minutes) was found and
characterised, not root-caused; a 30 ms encoder dispatch amortises it ([measurement](../../common/book/measurement.md#clock-and-cpu-load)).

## design6: one program

Four dispatches meant three hardware-context switches (~2.7 ms each, 10.7 ms per frame: 40.4 ms looped against 29.7 ms of GEMM work,
MEASURED, MD NOTES phase 5). Putting qkv, attn_out, fc1 and fc2 in one block needed more memtile BDs than exist: BDs come in two halves of
24, and fc2's streamed A alone needs 24 load BDs per block pass ("Allocator exhausted available BD IDs"). design6 merges the two half slots
into **one 31,104-word A buffer with one lock pair**, so a cached op is one load BD and an fc2 K chunk is one. The block chain is 15 loads
+ 2 sends = 17 BDs (checked in the lowered MLIR before the first run). Result: **33.8 ms per frame**, 27 blocks in one dispatch, all ops
bit-exact (587 frames in the loop, p90 33.9). The price: one buffer cannot double-buffer fc2's chunks, so fc2 runs ~550 us instead of 308
(+6.7 ms per frame). The projection (44 loads) cannot join the static chain and stays on the GPU.

## Phase 11-12: a faster loop does not help a DDR-bound frame

A hand-scheduled inner loop (II 18, 1.32 bundles per MAC) ran the frame in 31.5 ms against 34.1 for the Peano build, but computed the sum over
only the odd k sub-steps: with two accumulator chains interleaved one MAC per cycle, a dependent MAC two cycles after its predecessor
read the old accumulator (MEASURED; the hardware reads it earlier than the tables say; [toolchain](../../common/book/toolchain.md#the-asm-route)).
The correct schedule (chains 3 apart) is II 26, 1.83 bundles per MAC, bit-exact, **33.6 ms**: a correct hand loop lands where Peano's 1.59 does.
Even the wrong, 17 % faster loop moved the frame only 7.6 %, which is the point: the frame moves A 403 MB (re-read ~2x; ~200 inherent),
B 484 MB and C 430 MB. Phase 12 priced the cuts: half-K residency of A in the memtile (-100 MB), and a bfp16 hand-off from fc1 to fc2 (-84 MB written,
-107 MB read), about 300 of 1,317 MB or ~7 ms, none built. A fused fc1-GELU-fc2 does not fit: the 4,304-wide intermediate is 465 KB per M tile.

## Text-model GEMMs: tried, stay on the GPU

Phase 4 mirrored the GPU's W8A8 int8 text GEMMs (`p4/`): 0.0000-0.0005 % mismatch to the reference. But a pooled qkv takes 639 us (~19 GB/s of
weights), a whole layer 7.6 ms, and 24 layers in one dispatch **180 ms** pooled (MEASURED; 368 ms exact) against ~25 ms (pooled) to 48.4 ms (exact) on the
GPU. Every weight byte reaches four core rows through four per-core ports, so prefill (0.5 MAC per weight byte) is bandwidth-shaped work
the topology does not suit ([numerics](numerics.md#int8) for the arithmetic).

## What worked

- A static memtile chain with lock-counted repeats (design3): qkv 547 -> 446 us, DDR 18.8 -> 10.8 MB.
- The 17-BD single-buffer block (design6): 40.4 -> 33.8 ms, one xclbin.
- fp32 accumulation with bit-exact output, and the twin experiments (NOCOMPUTE) that located the floor each time.
- Reading the lowered MLIR for lock values before the first run: after the design2 hang, design3 and design6 both ran correctly on their first dispatch.

## What didn't work

- [A22](../../common/book/negative-results.md#a22) <!-- nr: A22 --> Peano k-loop variants (pipeline off, II hint, unroll, full unroll): slower or spilled, because AIE2P has five `dm` registers.
  Phase 2c's K2 (154 bundles per 64 MACs, 89 spills) and K3 (80 per 32, 36 spills) and phase 3's split accumulators, partial A hoists and unroll 3 (28-92 bundles per block against 27) belong here too. Lesson: the asm route, not C++ tweaks.
- [A23](../../common/book/negative-results.md#a23) <!-- nr: A23 --> The first hand-scheduled loop (1.32 bundles per MAC) was silently wrong; the correct one (1.83) was not faster than Peano's. Lesson: a checker
  needs conveyor-intent assertions, and any hand loop must be bit-compared. TAPNext's later generators made hand scheduling pay.
- [C12](../../common/book/negative-results.md#c12) <!-- nr: C12 --> The "~55-60 GB/s" DDR ceiling from the early probes; superseded by the aggregate caps.
- [H1](../../common/book/negative-results.md#h1) <!-- nr: H1 --> The 1.4 GHz clock from the spin probe and the "% of peak" figures built on it.
- Not yet in the index (see the hand-back):
  - **design2, runtime memtile tasks**: ERT timeout, TDR, cause never found; replaced, not fixed.
  - **A whole-block static chain with fc2's streamed A**: 24 BDs per channel half; solved only by design6's merged buffer, at +6.7 ms for fc2.
  - **Port and depth tuning**: A ring 6, B depths 3 and 4, C over 16 streams, no handshakes: all <= 2 % on qkv.
  - **Sharing operands between neighbours**: an AIE2 core reads only its own, north, south and west L1, so a 2x2 quad cannot halve both ports. The K-split pair that can is parked (the kernel,
    not the ports, was the limit).
  - **Text model on the NPU** (180 ms pooled), and a residual add in the GEMM epilogue (a third input stream is not available).
  - **Turbo as a free multiplier**: the plan said "scales every compute number"; it moved the clock ~3 %.

## Lessons

- Find the bound with a twin before changing the kernel. Here it was DDR traffic and the input port, not MACs, three times in a row.
- A faster loop is worth its percentage of the *bound*, not of the loop: 17 % of the loop gave 7.6 % of the frame.
- Make the repeat structure static (lock values, merged buffers) and verify it in the lowered MLIR; a runtime task path that nobody has used is a risk.
- Dispatch granularity decides the frame: 111 dispatches 49.8 ms, 4 dispatches 36.2, one program 33.8.
- Treat timing noise (DPM ramp over the first ~25 dispatches, +21 % under CPU memcpy hogs, the slow mode) as part of the measurement: warm up, print `gpu_busy`, take the minimum over processes.

## Going deeper

- `moondream/npu/NOTES.md` and `RESULTS.md`, phases 1-6 and 11-12; code in `p2/` to `p12/` (`design3.py`, `design6.py`, `mm_f32acc.cc`, `run.py`); commit `258ba90`.
- The tables of all seven ops per phase are in MD RESULTS phases 2b, 2c and 3. The hand-schedule generator is `p11/gen_gemm.py`, its checker `p11/asmcheck.py`.
- General: [gemm](../../common/book/gemm.md#is-it-compute--or-data-bound), [data-movement](../../common/book/data-movement.md#keeping-data-on-chip),
  [fusion](../../common/book/fusion.md#one-dispatch), [toolchain](../../common/book/toolchain.md#program-memory).
