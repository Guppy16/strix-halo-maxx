# NPU moondream2 vision encoder -- phase 1 notes (2026-10-07)

## Environment
- Host: Ubuntu 26.04, kernel 7.0.0-34, amdxdna DKMS 2.21 from ~/git/xdna-driver @273a586 (2026-01-02). Host shell is zsh.
- User's XRT debs: xrt_202610.2.21.0_25.10-amd64-{base,base-dev,npu}.deb (built on Ubuntu 25.10: boost 1.88, pyxrt cp313),
  plugin xrt_plugin.2.21.0_25.10-amd64-amdxdna.deb (postinst = dkms -> only extract libxrt_driver_xdna.so into the image).
- mlir-aie: cloned to ./mlir-aie, checked out v1.4.4 (2026-10-06). README: Ubuntu 24.04/24.10, kernel >= 6.17, XRT from
  ppa:amd-team/xrt or build_drivers.sh; env_install.sh hard-requires python3.12 but wheels exist cp311-cp314.
  build_drivers.sh pins xdna-driver 31c01f2 (2026-08-05, "update xrt submodule") -> MISMATCH: host driver/XRT is 7 months older
  (273a586, 2026-01-02, XRT 2.21). Try v1.4.4 against 2.21 first; fall back to an older mlir-aie release (v1.2.x, Jan 2026) if formats differ.
- Peano pin: llvm-aie==22.0.0.2026100501+94b320c0 (utils/peano-requirements.txt).

## Image npu-iron (docker/Dockerfile)
ubuntu:25.10 + user's 2.21 debs + python3.13 venv /opt/ironenv + mlir_aie==1.4.4 + llvm-aie pin + torch-cpu.
`nohup docker build -t npu-iron docker > build.log 2>&1 &`

## Encoder config (HF snapshot 9a7d402..., config.py VisionConfig, moondream2 rev 2025-06-21)
enc_dim 1152, patch 14, 27 layers, ff 4304, 16 heads (hd 72), crop 378 -> 27x27 = 729 tokens, proj_inner 8192, proj_out 2048.
md2fast: single crop (global == local for 378x378), projection input cat([x,x]) -> fc1 K=2304 (not folded: folding costs agreement).

## Important: what "encode 79.6 ms" means on the GPU
md2fast encode = vision tower + projection + TEXT PREFILL of 730 tokens. RESULTS.md in-situ: text int8 GEMMs 48.4 ms,
vision bf16 GEMMs 20.7 ms, flash attention ~10 ms (vision ~4.5 ms of it), LN/GELU/adds/quant ~6 ms.
So the part the NPU could take over is the vision tower + projection, ~30 ms, not 79.6 ms.

## GPU baseline re-measurement
Attempted (gpu_run.sh: GTT pre-check 44 MB/66 GB, busy 0%) -> blocked by the Claude Code permission classifier; not retried.
Using RESULTS.md numbers instead (88.7 ms, encode 79.6, 10.9 fps; vision bf16 GEMMs 20.7 ms in situ).

## Stack bring-up (works)
- `docker build -t npu-iron docker` (~10 min, slow apt mirror). Inside: XRT 2.21.0 (hash 92d731b, amdxdna 2.21.0_20260103 273a586),
  `xrt-smi examine` -> "NPU Strix Halo" [0000:c3:00.1], NPU FW 1.1.0.60, Power Mode: Default. mlir-aie 1.4.4, llvm-aie 22.0.0.2026092101 (v1.4.4 pin),
  python 3.13.7, torch 2.14.1+cpu. The 7-month driver gap vs mlir-aie's pin caused no problem for anything below.
- Run pattern: `docker run -d --name npu-X --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v $PWD:/work npu-iron <cmd>; timeout N docker wait npu-X`
- dmesg is not readable as testbox; `journalctl -k` is. No amdxdna messages during any run.

## Hello NPU
- basic/vector_vector_add (-n 1024): PASS, 0.7 s total.
- basic/matrix_multiplication/whole_array, npu2, 8 cols, 512^3: i16->i16 PASS 210 us (1.1 TOPS); bf16->bf16 (32x32x32) PASS 264 us (1.0 T);
  i8->i8 (64x128x64) PASS 177 us (1.5 T). All overhead-dominated at this size.

## GEMM at encoder shapes, generic whole_array design (bench_gemm.py, runs/sweep1.sh)
bf16->f32 64x64x32: qkv 2.8 ms (2.1 T), attn_out 1.2 ms (1.6 T), 2048^3 2.6 T; error ~3e-7 rel (exact fp32 accumulation).
bf16->bf16 64x64x64: same speed, 3% rms error (bf16 accumulation across k-tiles). bfp16 emulation: 2.8-3.0 T, 0.9% rms.
i8->i32 64x128x32: 2.8-3.8 T (exact). i8->i16: 5-7.5 T but overflows (useless). => generic design is 10-20x below peak.

## ATB design (programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering)
Stock: config1 (bf16 A, bfp16ebs8 B, bf16 C) 4096x4096x2048: 2909 us = 23.6 TFLOPS; config2 (all bfp16) 3072x4096x1536: 1221 us = 31.7 TFLOPS.
(Default power mode; README quotes the same numbers with `xrt-smi configure --pmode turbo`, which needs sudo -- not tried.)
NOTE: config1's kernel converts A to bfp16ebs8 in-core, so the math is bfp16 x bfp16 with bf16 C accumulation between k=64 steps.
atb/: generalised copy -- enc_gemm.py (m, n, mtk parameters; any number of task groups), mm_bfp_mixed.cc (-DDIM_M/-DDIM_N),
test.cpp (real vs padded dims, A~N(0,1), W~N(0,0.03), optional outlier channels, full double reference, sync timings), run_shapes.sh.
m=64 so M pads 729->768 (m=128 would need 1024).

## Dispatch + context costs
- 256x128x1152 (6 us of math): 99 us per dispatch -> ~95 us fixed submit+wait per launch.
- atb/ctx_switch.py: alternating two xclbins (separate hw_contexts, both 32-core): attn_out 275 -> 3012 us, qkv 483 -> 3255 us
  => ~2.8 ms per context switch. A per-layer pipeline must live in ONE xclbin (ideally one runtime sequence).
- Uniform tiling n=144, mtk=128 for every shape (shapes_uniform.txt) costs <=10% on the projection GEMMs only, so one
  dispatch-time-shaped xclbin for all encoder GEMMs is realistic.

## GPU baseline numbers used (moondream-opt/RESULTS.md, not re-measured)
HIP bf16 hot microbench: 729x1152x3456 195 us, x1152 80, x4304 251, 729x4304x1152 223, 729x2304x8192 878, 729x8192x2048 1051 -> 22.2 ms/encode;
in situ 20.7 ms. Per-token int8 vision dropped accuracy to 89.2% (RESULTS.md l.80).

# Phase 2a -- fp32-accumulating bfp16 GEMM (2026-10-07, evening)
Code: p2/ (design.py, design2.py, mm_f32acc.cc, pack.py, run.py, bw.py/runbw.py, b.sh build, r.sh run, pb.sh probes,
bdis.sh/loopii.sh = compile one kernel + count inner-loop bundles). Builds/logs/outputs in p2/build/.

## Kernel (mm_f32acc.cc)
bfp16 A (host-packed, conv_even) x bfp16 B (host-packed, conv_even = round to nearest), 2x2 register block of 8x8x8 bfp mmuls
(ATB microkernel structure), fp32 accumulator for the whole K. L1 cannot hold an fp32 96x96 tile plus a bf16 output tile, so
the accumulator is split: rows [0,48) live inside the bf16 output ObjectFifo buffer (exactly 96*96*2 B = 48 rows fp32), rows
[48,96) in a static buffer; acc_store() compacts in place and rounds to bf16 (RNE) once at the end. Hooks for fused epilogues
= acc_store().
- Correctness (qkv 729x1152x3456): 0.0000% of outputs differ from bf16(float64 product of the decoded bfp operands);
  rms rel 1.66e-3 = the bf16 rounding floor; max err 1.00 ulp. I.e. bit-exact up to final bf16 rounding.
- Compiler pitfalls (Peano 22.0.0.2026092101):
  * Because every 16x16 block re-reads the same A band, the compiler hoisted the 16 A pops out of the block loop and spilled
    them to the stack, reloading via 3 scalar/vector loads each (~100 bundles/block). Fix: seek the A stream by `g_zero*blk`
    with g_zero a volatile 0 (-DOPAQUE_A). KEEP.
  * fp32 C prefetch of the next block (ATB style) needs 6 dm registers (4 acc + 2 prefetch) but AIE2P has dm0-dm4 -> spills.
    -DNO_PREFETCH is better. KEEP.
  * Inner block loop: 68-70 bundles per 32 MACs (47% of the core's 512 MAC/cycle). Not software-pipelined: loads at top,
    MACs, stores at bottom. Load unit A carries 16 fp32-C loads (4x512b per 8x8 acc) + 16 A pops + 2 fills = 34/block, so
    II ~34-36 is the floor with perfect pipelining. Tried: pipeline(disable)+min count (68), II hint 40 (ignored, 70),
    unroll 2 (140/2 blocks), full unroll (spills, worse). DISCARD all; next idea is hand-rotated accumulators.
  * rho (A sub-tiles per k-step) 3 vs 6: same speed. rho=2: 2x slower (unexplained; bank conflicts?). Use rho=6.

## Data movement probes (bw.py; launch overhead ~95 us included in "us")
| streams | data | us | aggregate |
|---|---|---|---|
| 8 (1/col), fan 4 | 9 MB | 270 | 33 GB/s incl launch; 51 GB/s excl |
| 16 (2/col), fan 4 | 18 MB | 398 | 45 / 59 GB/s |
| 8, fan 1 | 9 MB | 244 | 37 / 60 GB/s |
| 16, fan 1 | 18 MB | 399 | 45 / 59 GB/s |
| 16, 2x data | 36 MB | 719 | 50 / 58 GB/s |
| 8 read + 8 write | 9+9 MB | 381 | 63 GB/s combined excl launch |
=> One shim stream does ~6.4 GB/s (4 B/cycle at ~1.6 GHz); the NPU's DDR ceiling is ~55-60 GB/s read, ~63 GB/s read+write
combined. Broadcast fan-out costs nothing. This is THE constraint for the encoder (see RESULTS Phase 2a).

## design.py (all through ObjectFifos; core column c = M-tile c of 96 rows, core row r = run of N-tiles; B read from DDR once,
A re-read once per N-tile)
| variant | qkv us (median) | note |
|---|---|---|
| first cut, prefetch, rho 3 | 616 | A pops spilled to stack |
| OPAQUE_A + prefetch | 946 (min 493) | noisy, spills |
| OPAQUE_A + NO_PREFETCH | 459 | |
| + PIPE=1, rho 6 / rho 3 | 464 / 463 | KEEP rho 6 |
| rho 2 | 945 | DISCARD |
| no compute at all (DMA only) | 549, 515 with L2 depth 4 | => DMA/DDR-bound, not kernel-bound |
| no compute, C linear (no un-block) | 577 | C path is not the limit |
qkv traffic in this design: A 1 MB x 9 N-passes + B 4.5 + C 5.3 = 18.8 MB -> ~50 GB/s => DDR-bound.

## design2.py (A resident in memtiles, explicit Flow/Buffer/Lock/TileDma + runtime-sequence memtile tasks with per-op repeat)
Builds. First run: ERT timeout (design deadlock) -> amdxdna TDR "Device isn't making progress... Count 11" at 22:43:15.
Earlier, 22:33:04-22:34:00 (Count 1-10): my bw5 probe (--l1 1728 --l2 6912 --total 279936: total not a multiple of the
L2 object, so the last memtile object never completed -> hang until the 60 s timeout). The driver recovered after the first
episode (later runs fine). Per the hard rule, NPU work STOPPED here; nothing run since. Suspects for the design2 deadlock:
memtile task lock credits (Acquire(value=nt) on the slot-empty locks), the core S2MM ping-pong vs core acquire order, or the
memtile BD/channel assignment for runtime tasks next to ObjectFifo channels. Next step: run it with a 1-op, nt=1 schedule
under a short XRT timeout once the maintainer OKs more NPU runs.

## Power mode
Not tried (would have been after the stop). Command to try (in container, as root with /dev/accel/accel0, or on the host):
`xrt-smi configure --pmode turbo` (phase 1: host needs sudo). README numbers for ATB were measured with turbo.

# Phase 2b -- A resident on chip, all shapes, fusion, kernel, launch (2026-10-07 late / 10-08)
Code: p2b/ (design3.py = the fix for design2; design.py/pack.py/run.py copied from p2 and extended;
b.sh/r.sh/bdis.sh/loopii.sh as before but under /work/p2b). Builds in p2b/build/.

## Power mode
`xrt-smi examine -r platform` in the container at 23:11: **Power Mode: Default** (turbo was not switched on).
Known-good qkv (p2/build/q_r6p1, same build and command as phase 2a): 547 us median / 531 min at 23:12,
vs 463/452 in phase 2a (same mode). Box idle (GPU 0%, load 0.2). Re-measured later in the session (see table);
treat +-20% run-to-run drift of DDR-bound numbers as real.
Driver recovered from the 22:43 TDR: the run was bit-exact, no new xdna kernel messages.

## pyxrt run.wait(timeout)
`run.wait(N)` in this XRT 2.21 pyxrt returns no earlier than ~N *microseconds* (N=5000 -> 5007 us, 1000 -> 1016 us,
20 -> real time ~450 us; 0 = wait forever). So it is a usable short timeout for first runs (use a few million) but it
inflates timings: benchmark with --timeout-ms 0 after a design is proven. run.py takes --timeout-ms.

## design2 deadlock: analysis (no NPU run)
Lowered MLIR of p2/build/d2_qkv (input_with_addresses.mlir) looks right: memtile (0,1) got S2MM 5 / MM2S 1 for A next
to C's S2MM 0-3 + MM2S 0, no channel or shim collision (IRON 1.4.4 places ObjectFifos on *logical* tiles, so this had
to be checked); lock values 9 are in range; the core S2MM chain loops (DmaChannel.loop defaults True); the memtile tasks
lower to `aiex.dma_configure_task(%mem_tile, S2MM, 5)` with the use_locks intact. The remaining suspects are the
runtime-task path itself (memtile BD register encoding from the sequence; nothing in programming_examples uses
`.task()` on a memtile) and the task-queue depth (shim queue is 4 deep and does not backpressure). Not debugged
further: replaced by a static design that needs none of it.

## design3.py (KEEP): static memtile A chain
Two 124 KB slots per memtile; every op's A M-tile is split by K-halves over the two slots; the memtile MM2S chain is
always [slot0, slot1] (one N-tile's A per pass) and loops forever; a load BD releases "full" by nt and the next load into
that slot acquires "empty" >= nt. So per-op repeat = lock values on the load BDs (compile-time schedule), no runtime
tasks, nothing can wait for data that never comes. Ops with K > 2304 (fc2, proj_fc2) stream A through the slots as a
ring, re-read per N-tile (same traffic as design.py for them). One slot "unit" per schedule: cached ops fix it at half
an M-tile, streaming ops need K padded to a multiple of 2*unit (fc2 alone: unit 29376 words; in a block schedule with
qkv/attn/fc1 the unit is 15552 and fc2 must pad K to 4608).
Cost: no prefetch of the next op's A while the current op runs (~10-20 us bubble per op).
Verified in the lowered MLIR before running: chain [As0_0: acq Ae>=9, rel Af+9] -> [As0_1 ...] -> loop; MM2S
[acq Af>=1, rel Ae+1]; lengths 15552+15552 = the 31104-word shim fill; core chain 2 x 288 words.
First dispatch (1 run, timeout): correct, bit-exact. Driver quiet.
qkv: **446 us median / 442 min** (timeout 0), 13.0 real TOPS, 10.8 MB moved (A 1.0, B 4.5, C 5.3) = 24 GB/s.
=> Not DDR-bound any more (limit ~55-60 GB/s); design.py today 547 us for 18.8 MB. The remaining time is kernel
(47% of core peak), per-N-tile epilogues (acc_zero/acc_store), the A-load bubble and the ~95 us launch.

## design3, all encoder shapes (Default power mode, pyxrt, --timeout-ms 0, median of 10; first dispatch of each
## build was a single run with a timeout: all correct, bit-exact, driver quiet)
| build | sched (K, nt) | real KxN | us median / min | real TOPS | DDR MB (A/B/C) | GB/s | GPU us |
|---|---|---|---|---|---|---|---|
| d3_qkv | 1152, 9 | 1152x3456 | 446 / 442 | 13.0 | 10.8 (1.0/4.5/5.3) | 24 | 195 |
| d3_attn | 1152, 3 | 1152x1152 | 223 / 222 | 8.7 | 4.3 (1.0/1.5/1.8) | 19 | 80 |
| d3_fc1 | 1152, 12 (N 4608) | 1152x4304 | 558 / 548 | 13.0 | 14.0 (1.0/6.0/7.1) | 25 | 251 |
| d3_fc2 | 4352, 3 (A streamed) | 4304x1152 | 504 / 497 | 14.3 | 18.7 (11.3/5.6/1.8) | 37 | 223 |
| d3_pfc1 | 2304, 22 (N 8448) | 2304x8192 | 1564 / 1563 | 17.6 | 36.9 (2.0/21.9/13.0) | 24 | 878 |
| d3_pfc2 | 8192, 6 (N 2304, A streamed) | 8192x2048 | 1467 / 1456 | 16.7 | 67.2 (42.5/21.2/3.5) | 46 | 1051 |
All: rms_rel 1.65-1.66e-3 = bf16 rounding floor, 0.0000% mismatch vs bf16(exact), max 1 ulp.
Per frame (27 blocks + projection): 27 x 1731 + 3031 = 49.8 ms vs GPU 20.7 ms (still 2.4x slower);
~10 ms of that is 4 x 27 launches at ~95 us.
Observations: fc2/proj_fc2 stream A per N-tile (K too large for the slots) and are the only ones near the DDR
ceiling (37-46 GB/s). The K=1152 shapes move 19-25 GB/s, so they are NOT DDR-bound any more; their time is
kernel (47% of core) + the A input stream per core (at 96x96 tiles the A stream per core needs 1728 cycles per
k-step vs 1152 cycles of MACs at peak -> 67% cap, see Phase 2a) + per-N-tile epilogues + launch.

## Where qkv's 446 us go (design3)
d3_qkv_nc (same design, kernel compiled with -DNOCOMPUTE: cores only acquire/release): **369 us** median.
So data movement + launch is 369 us and compute adds only 77 us on top. The per-shim-stream limit (~6.4 GB/s) is
the reason: B (4.5 MB) enters through only 4 shim MM2S channels (one per core row) = 1.1 MB per stream = ~176 us;
C (5.3 MB) leaves through 8 S2MM streams = ~104 us; A 1 MB / 8 = 20 us; plus ~95 us launch. Not aggregate DDR (29 GB/s).
Fix (next): B over 8 streams (2 per row, second fifo of each row on the 4 memtiles without one) and C over 16
(2 S2MM shim channels per column, join of 2 rows each) -> qkv data time ~100 us; then the kernel (~220 us of MACs
at 47%) is the limit, then launch.

## Memtile BD budget: 48 BDs are two halves of 24 (channels 0-2 get BDs 0-23, channels 3-5 get 24-47)
Whole-block build (qkv, attn, fc1, fc2 K->4608, c_linear) failed: "Allocator exhausted available BD IDs (maximum 24
available for channel 5)". The static A chain for fc2 alone needs 4608/1152 x 3 N-tiles x 2 = 24 load BDs per
block pass (each streamed chunk is its own BD), plus 6 for the cached ops; the half holding the A channel also holds
one C channel (2 BDs) -> at most ~22 A loads. Consequences: fc2 (K=4304) cannot share a static chain with the other
ops; fc2 alone fits (unit 29376, 4 chunks x 3 = 12). A whole-block single sequence needs either runtime memtile
tasks (the design2 path, unverified) or the fused fc1->GELU->fc2 design that never streams fc2's A from DDR.
c_linear=1 (C leaves the memtile in the core's blocked layout, 2 BDs instead of 8 for the join's MM2S) is kept for
multi-op schedules; run.py un-blocks on the host.

## One sequence per block (design3, c_linear, lookahead 2; pyxrt timeout 0)
| build | schedule | us | per block / per op |
|---|---|---|---|
| d3_b3 | qkv, attn_out, fc1 | 951 | vs 446+223+558 = 1227 as 3 dispatches |
| d3_b3x27 | the same x27 in ONE dispatch | 22636 | 838 per block-of-3 |
| d3_fc2x27 | fc2 x27 in one dispatch | 10605 | 393 per fc2 (504 alone) |
All ops bit-exact (c_linear un-blocked on the host). => encoder GEMMs today: 22.6 + 10.6 + 3.0 (projection) = 36.2 ms
per frame in 4 dispatches (GPU 20.7 ms). The launch cost is now negligible; what remains is the per-stream DMA floor
(B over 4 shims, C over 8) and the kernel.
Launch probes (p2/bw.py, 576 words per stream, pyxrt): 1 column / 75 instruction words: 82 us median, 76 min;
8 columns / 572 words: 90 / 87. The ~80-90 us is per-dispatch latency, not instruction or DMA count.

## C over both shim S2MM channels per column (design3 c_streams=2, now the default; b_streams stays 1)
d4_qkv: 442-447 us (two processes), NOCOMPUTE twin 377 us (d3: 369) -> the memtile->shim C stream was not the floor.
A second B stream per row is impossible without a memtile join (a core has only 2 S2MM channels: A and B), and
would not help anyway: the per-core input port (~4 B/cycle) carries the same 1.1 MB of B per qkv op (~176 us) and
the same 1.1 MB of A (9 re-sends of the 124 KB M-tile) in parallel. That port floor (176 us + launch + ramp) is
what the NOCOMPUTE 369-377 us measures. Only sharing A/B between neighbouring cores through their shared L1
(each core DMA-receives half the tiles, reads the other half from its neighbour) halves it. Not built this session.

## Run-to-run noise is per process and bimodal
Consecutive processes, same build, same box (idle): d3_qkv 522 then 946 us median (min 489 / 934); d4_qkv 442, 447.
Earlier d4_qkv 701 (min 451). Phase 2a saw 946 vs 493 too. Within a process all iterations agree, so it is not the
NPU clock but something per process (host BO placement?). Report min-of-processes; worth checking pyxrt bo flags.

## Launch overhead from C++ (p2b/host.cpp, zero data, same xclbin/instr)
qkv 432 us (pyxrt 446), attn_out 197 (223), b3 966 (951), b3x27 22581 (22636). Python costs ~15-25 us per dispatch;
the rest (~75-80 us floor, see the 1-column probe: 76-82 us) is XRT/firmware per-dispatch latency, independent of
instruction count (75 vs 62200 words) and DMA count. validates 50 us is not reachable from the XRT run API here
## C over both shim S2MM channels per column (design3 c_streams=2, now the default; b_streams stays 1)
d4_qkv: 442-447 us (two processes), NOCOMPUTE twin 377 us (d3: 369) -> the memtile->shim C stream was not the floor.
A second B stream per row is impossible without a memtile join (a core has only 2 S2MM channels: A and B), and
would not help anyway: the per-core input port (~4 B/cycle) carries the same 1.1 MB of B per qkv op (~176 us) and
the same 1.1 MB of A (9 re-sends of the 124 KB M-tile) in parallel. That port floor (176 us + launch + ramp) is
what the NOCOMPUTE 369-377 us measures. Only sharing A/B between neighbouring cores through their shared L1
(each core DMA-receives half the tiles, reads the other half from its neighbour; explicit Buffers/Locks like the A
path, lock values 2 per tile) halves it. Not built this session.

## Run-to-run noise is per process and bimodal
Consecutive processes, same build, idle box: d3_qkv 522 then 946 us median (min 489 / 934); d4_qkv 442, 447;
d5_qkv 423 then 946. Earlier d4_qkv 701 (min 451). Phase 2a saw 946 vs 493 too. Within a process all iterations
agree, so it is not the NPU clock but something per process (host BO placement / IOMMU mapping?). Report
min-of-processes; worth checking pyxrt bo flags (host_only vs device) before the next round.

## Launch overhead from C++ (p2b/host.cpp, zero data, same xclbin/instr)
qkv 432 us (pyxrt 446), attn_out 197 (223), b3 966 (951), b3x27 22581 (22636). Python costs ~15-25 us per dispatch;
the rest (~75-80 us floor, see the 1-column probe: 76-82 us) is XRT/firmware per-dispatch latency, independent of
instruction count (75 vs 62200 words) and DMA count. validate's 50 us is not reachable from the XRT run API here;
amortising (27 blocks per dispatch = 2-3 us per block) makes it irrelevant.

## Kernel v2: -DBLK12 (KEEP, without OPAQUE_A / PIPE)
1x2 register blocks (8 rows x 16 cols), next block's two fp32 accumulators prefetched (4 of 5 dm registers), A
packed row-half-major (pack.py half=True, run.py picks it from the kdefs). Peano now hoists the row-half's 8 A panels
into ex registers (12 exist) for the whole block loop, no spills: inner loop 27 bundles for 16 MACs (16 vldb pops,
8 acc loads, 8 stores) = 1.69 bundles/MAC = 59% of core peak, vs 68 per 32 (2.13, 47%) before. With OPAQUE_A it is
57 bundles (the volatile defeats the hoist) -> do not combine. d5_qkv (design3 + BLK12): bit-exact, 423 us / 418 min
(vs 442-446), i.e. ~5% at the op level because the MACs are mostly hidden behind the DMA-port floor.
Unit B (16 pops per 16 MACs) is now the saturated resource; B-sharing between neighbour cores or 2x2 blocks with
16 A panels in registers (does not fit: 12 ex) would be the next step.

## bfp16 epilogue / fused MLP (not built; design notes)
Epilogue: acc_store would push each 8x8 fp32 block as accum<accfloat,64>::to_vector<bfp16ebs8>() through an
OutBufStream in A order ([2 row halves][k/8] per 16-row group, i.e. exactly the half=True pack order), so a 96x96
tile leaves the core as 10368 B instead of 18432 (C bytes -44%) and the next op's A needs no host repack; the memtile
C object shrinks accordingly and the drain tap becomes [nt][rows][tile bytes] linear. Requires the next op's k to
divide 96 (k=32 or 96; k=96 fits L1: B 2x10368 + A 2x1728 + C 18432 + acc_hi 18432 + stack 2048 = 63104 B).
Fused fc1->GELU->fc2 with the 4304-wide intermediate on chip does not fit as a whole: per 96-row M-tile it is
465 KB of bfp16 next to the memtile's C/B buffers (512 KB per memtile), and fc2's fp32 accumulators per core (3 N-tiles
x 36 KB) do not fit L1. It fits if the array works on 4 M-tiles at a time (2 memtiles = 2 columns per M-tile, 232 KB
each), i.e. two passes over M with N split over 8 cores; that is the shape of the phase-2c design, together with
neighbour sharing of A/B.

# Phase 2c (2026-10-08; Default power mode throughout, labelled)
Code: p2c/ (design3.py evolves; mode.py = timing-mode reproducer; spin.py/runspin.py = clock probe; host.cpp).

## 1. Bimodal per-process timing: what it was
Reproducer p2c/mode.py (zero or random operands, N BO sets per process, per-iteration trace, optional CPU burn).
~45 processes tonight on the idle box (zero data, random data, run.py real data, alternating xclbins, CPU burn
before timing, 3 BO sets per process): NO slow (2x) process at all; medians 405-483 us for d5_qkv.
Found instead: (a) a DPM ramp: per-iteration time falls 590 -> 410 us over the first ~25 dispatches of a process,
so short runs and "set 0" look 10-20% slow -> run.py now warms up 20 dispatches by default. (b) DDR contention is
measurable: six CPU memcpy hogs make qkv 413 -> 501 us (+21%); a GPU job moves 10x more DDR traffic. The slow 2x
processes last night (23:44-23:55) fell in a window when the GPU-side App server may have been encoding frames;
no slow process has occurred since. Conclusion: cross-tenant DDR contention on the shared box, not BO placement
(addresses/flags make no difference), not IOMMU (identity-mapped), not the xclbin or hw-context. Fix: run.py prints
gpu_busy% around each measurement; take min over processes; warm up >= 20 dispatches.

## Clock and launch (p2c/spin.py: one core loops the BLK12 kernel on resident L1 data, no input DMA)
2000 calls 559 us min, 4000 calls 1081 us -> 0.261 us per call; a call is ~364 bundles (2 row halves x 6 blocks x
27 + prologue) -> **NPU core clock ~1.4 GHz in Default mode**. Intercept 37 us = dispatch latency of a design with
one 64-byte drain; the ~80 us of real designs is therefore ~40 us firmware DMA-task handling (BD writes, patches,
waits) + ~37 us fixed. At 1.4 GHz the qkv kernel (162 k-steps x 1944 bundles) is 225 us of the 423: launch 80,
stalls ~118. Per frame the MACs alone at the current 59% kernel are ~24 ms -> the kernel is the limit, not DMA.

## Where the remaining qkv time goes (d6 = design3 + BLK12 + c_streams 2; 1.4 GHz)
| variant | us | note |
|---|---|---|
| d5_qkv (A ring 2) | 423 | |
| d6_qkv (A ring 6) | 416 | +4.6 KB L1, -2% only: A-ring latency was not the stall |
| + B L2 depth 4 | 417 | no change |
| A ring 2 + B L1 depth 3 | 416 | no change |
| nosync 1 (no A handshakes) | 404 | handshakes cost ~12 us |
| nosync 2 (no B handshakes) | first dispatch ok; second hung because the probe never releases B (timeout 1/3; driver recovered, known-good bit-exact) |
| NOCOMPUTE (ring 6, L2 4) | 350 | DMA + launch floor (was 369) |
| spin, compute only, 1 core | 0.261 us per call = 253 us of pure kernel for qkv on 32 cores |
Breakdown: launch 80 + kernel 253 + acc_zero/store 13 + C-tile stall (L1 depth 1, 18 KB drain per N-tile) ~26
+ A-load bubble ~20 + misc ~25 = 417. The kernel is 60% of the op and the only big lever left; the ~85 us of
bubbles need the bfp16 epilogue (C depth 2 fits with n=64 tiles) and A prefetch.
Kernel variants this phase (bundles per block): K2 (2x2 over two k-steps, two B streams) 154 per 64 MACs with 89
spills, also as two loops: DISCARD. K3 (2x2, one row-half of A hoisted, the other popped) 80/32 with 36 spills, 68/32
with PIPE=1 (= the old 2x2), 96/32 without OPAQUE_A (132 spills): DISCARD. BLK12 at 27/16 stays.
Peak arithmetic at 1.4 GHz: 32 x 512 MAC = 45.9 TOPS; the frame's 652 GFLOP is 14.2 ms at 100%, 24 ms at the
kernel's 59%. In Default mode the GEMMs cannot beat the GPU's 20.7 ms without a >= 75% kernel AND small overheads;
turbo (the user's call) scales every compute number by its clock ratio.

## Shared-operand design: why it is a K-split, and why it is parked
AIE2 affinity (AIETargetModel.cpp isLegalMemAffinity): a core reads its own, north, south and WEST neighbour's L1
only. Vertical pairs in the mapping share A (same M-tile) but need different B; symmetric horizontal sharing is
impossible, so a 2x2 quad cannot halve both ports. The design that halves both without horizontal access is a
K-split pair: cores (c,2)/(c,3) compute the same 96x96 tile over the two K-halves (each DMA-receives half of A and
half of B: slots = K-halves, B interleaved per K-half on the host, two A and two B MM2S flows per memtile); the
north core's fp32 partial is summed by the south core through shared L1 (~2000 cycles per N-tile, ~10%). Ports go
1728 -> 864 cycles per k-step. The kernel already needs 1944 cycles per k-step, so ports are not the limit until
the kernel is ~2x faster: parked behind the kernel work.

# Phase 3 (2026-10-08, from ~02:20; NPU in **Turbo** (`xrt-smi examine -r platform`), checked at every container start)
Code: p3/ (copies of p2c + mode2.py; builds d7_* = BLK12 unrolled 2).

## Turbo vs Default (uncontended, gpu_busy 0)
spin probe: 0.254 us per 364-bundle kernel call (Default 0.261) -> core clock **1.43 GHz vs 1.39 GHz, +3%**.
Correction to the phase-2c note: the spin Worker free-runs (while_true), so back-to-back dispatches just collect
finished objects; the slope is the core rate but the "37 us intercept" was an artefact, not dispatch latency.
d6 builds in turbo: b3x27 19.93 ms (Default 20.76), fc2x27 8.80 (9.63), pfc1 1399 (1484), pfc2 1308 (1351)
-> per frame 31.4 ms vs 33.2: turbo mainly trims DMA/latency, not the MAC clock (validate: 51 TOPS in both modes).

## The slow mode, continued (turbo made it frequent)
Observed in turbo: d6_qkv 945 us in 5 of 6 processes in one container (GPU 0%), then 1466/1944/1954 in the next
container, then 6 of 6 fast (382-414). Values are quantised at ~1 ms multiples (946 ~ 1 unit, 1944 ~ 2 units) around a
390 us kernel; 20 ms dispatches (b3x27) are unaffected (19.9 ms in the same slow process). mode2.py (two contexts in
one process: qkv, then the spin core-rate probe, then qkv) was fast 6/6 with the spin rate unchanged -> not the core
clock, not D3hot/D0 at start (autosuspend_ms 5000), not BO placement, not run.wait(0) vs wait(), not the per-iteration
A sync, not BLAS threads (OPENBLAS_NUM_THREADS=1: 382-387 vs 388-414, all fast). It clusters in time (whole minutes
slow, then fast) and only shows on sub-millisecond dispatches: the signature of completion notification falling
back to a ~1 ms timer path in driver/firmware (mailbox_polling=0 is the interrupt mode; HZ-granular wait). Not
root-caused from user space. run.py now prints xdna interrupts per dispatch and flags SLOW-MODE (median >= 1.6 x min
or ~950 us with a sub-700 min); report numbers from processes without the flag. For App it is amortised by
one ~30 ms encoder dispatch.

## Phase 3 kernel + d7 results (turbo, uncontended): see RESULTS.md "Phase 3". BLK12 unroll 2 (-DBLKU=2) KEEP:
51 bundles / 32 MACs (63% of core peak). Per frame 29.7 ms. One SLOW-MODE process caught by the new flag (pfc2 1910
vs min 1271; repeat 1247). No timeouts this phase; three-timeout budget: 1 used in 2c, none since.

# Phase 4 (2026-10-08, Turbo): int8 W8A8 text-model GEMMs
Code: p4/ (mm_i8.cc kernel + epilogue; design_i8.py = design3 structure with int8 sizes; design_i8s.py = design.py
streaming structure; pack_i8.py, run_i8.py with the numpy reference mirroring moondream-opt HI8Lin / i8gemm.hip).
Numerics mirrored: sw = max|w_n|/127 (>= 1e-8), RNE, clamp; sa = max(amax,1e-5)/127, round half away from zero;
int32 exact accumulate; epilogue ((float)acc * sa[token]) * sw[row] + bias[row] in fp32, tanh-GELU
(1 - 2/(1+exp(2u)), exp = 2^x with a degree-6 polynomial, reciprocal by Newton), bf16 RNE.
Hardware fact: AIE2P dense int8 mmul is 8x8x8 = 512 MACs per instruction, the SAME rate as bfp16 (the 1024-MAC
8x16x8 form needs a sparse B); int8 panels are 64 B vs bfp16s 72 B, so int8 is only 11% fewer bytes than what the
# Phase 4 (2026-10-08, Turbo): int8 W8A8 text-model GEMMs
Code: p4/ (mm_i8.cc kernel + epilogue; design_i8.py = design3 structure with int8 sizes; design_i8s.py = design.py
streaming structure + lookahead; pack_i8.py, run_i8.py with the numpy reference mirroring moondream-opt HI8Lin /
hip/i8gemm.hip).
Numerics mirrored: sw = max|w_n|/127 (>= 1e-8), RNE, clamp; sa = max(amax, 1e-5)/127, round half away from zero;
int32 exact accumulate; epilogue ((float)acc * sa[token]) * sw[row] + bias[row] in fp32; tanh-GELU as
1 - 2/(1 + exp(2u)) with exp = 2^x (degree-6 polynomial on the fraction) and a Newton reciprocal; bf16 RNE.
Hardware fact: the AIE2P dense int8 mmul is 8x8x8 = 512 MACs per instruction, the SAME rate as bfp16 (the 1024-MAC
8x16x8 form needs a sparse B operand); int8 panels are 64 B vs 72 B for bfp16, so int8 moves only 11% fewer bytes
than the vision path already does. The text model therefore has no int8 speed advantage on this NPU beyond those 11%.
Text model: dim 2048, 24 layers, fused qkv N=6144, proj 2048, fc1 8192 (GELU), fc2 K=8192, LayerNorm, partial
rotary 32 of 64 dims, mask = bidirectional over the 730-token prefix then causal, post_ln + lm_head 51200.
Weights 50.3 M int8 per layer = 1.21 GB per frame; MACs 0.62 TOP pooled (278 tokens), 1.76 TOP exact (750).
Orientation: "A" (per column) = 96-row blocks of W with sw/bias appended per 16-row sub-tile (1152 B = the same
words as a bfp16 sub-tile); "B" (per row) = X^T token tiles 64 x 96 with sa appended (6528 B); C^T tile 96 rows x
96 tokens, un-blocked + transposed on the host. One op = 768 weight rows; N/768 ops stream W once; tokens padded
to 384 (4 rows x 96, nt = 1). Kernel (BLKU=2): 48 bundles / 32 MACs in one row half, 70 in the other.
Results (qkv pooled: W 6144 x 2048 = 12.6 MB, 278 tokens):
| design | us | weight GB/s | note |
|---|---|---|---|
| i8_qkv (design3 slots) | 660 | 19 | nt = 1: each weight block serves one tile; the slot reload (31 us) cannot overlap the 45 us op |
| i8s_qkv (design.py streaming, finish per op) | 797-862 | 15-16 | pipeline drains at every op boundary |
Both: mismatch 0.0005% vs the reference, max 1 ulp (bf16 floor).

## Phase 4, continued: streaming design, whole layer, lessons
- design_i8s.py (design.py structure) + lookahead 2: qkv pooled 639 us (lookahead 4: same). n=80 token tiles 703
  (slower: port-bound weight stream, fewer MACs per byte). Exact (750 tok, nt 2): slots 1247, streaming 1508.
- Whole text layer in one xclbin (25 ops: qkv 8, proj 3, fc1 11 with a runtime GELU flag carried in the token
  tile, fc2 3 with K 8192). Three things had to change to make it link and run:
  (1) core program memory is 16 KB: the schedule must be run-length compressed into runtime loops in core_fn (the
      first attempt compared a list slice with a tuple and silently did nothing);
  (2) the 64-lane fp32 GELU epilogue spilled 5.5 KB of stack: rewritten on native 16-lane vectors (two accumulator
      rows per vector, sw/bias as two 8-lane broadcasts, sa as an 8-float block concatenated), no tables, stack 0x1200;
  (3) run_i8.py packs one token set per distinct K.
  Results: pooled layer 7630 us (NOCOMPUTE 2750), exact 15330; GELU ops 0.009-0.010% mismatch (max 2-3.5 ulp),
  others <= 0.0005%. The 16-lane epilogue costs ~40 us per op (qkv 8 ops: 639 -> 971 us) and GELU ~25 us more.
- Weight stream rate stays at 7-20 GB/s in every variant: the per-core input port replicates every weight byte to
  the 4 cores of a column (46 GB/s chip ceiling at nt 1) and the ObjectFifo hops reach ~45% of it; compute and
  DMA overlap poorly across op boundaries. Conclusion in RESULTS.md "Phase 4": text prefill stays on the GPU.

# Phase 5 (2026-10-08, NPU Turbo, GPU via gpu_safe.sh only)
Code: p5/ (nputloop.py = NPU frame loop over the d7 xclbins with per-frame timestamps;
moondream-opt/textloop.py = GPU pooled text-only loop: one real vision pass caches the pooled projected tokens,
then _encode is replaced by the text prefill from the cache, CUDA-graphed like bench.py, query per frame).

## (1) Joint NPU + GPU measurement (pooled pipeline)
| side | alone | during the other | slowdown |
|---|---|---|---|
| NPU frame (d7: b3x27 + fc2x27 + pfc1 + pfc2, 4 xclbins) | 40.4-40.7 ms (two runs) | 44.8 ms median, p90 46.2 (1273 frames inside the GPU window) | +10% |
| GPU pooled text-only frame (prefill 24.9 + query 8.3) | 33.2 ms (1196 frames) | 34.4 ms (prefill 25.1, query 9.2; 1738 frames) | +3.6% |
GPU busy 100% during its loops, 0% otherwise; GTT 11%; no other tenant in the .gpu logs; answers unchanged (yes).
NPU-side note: b3x27 looped alone is 18.8 ms, the four-xclbin frame 40.4: the three hw-context switches cost
~10.7 ms per frame (~2.7 ms each, as phase 1 found); the GEMM work itself is 29.7 ms. Attention / LN / patch
embedding are not built (phase-3 estimate ~8 ms more on the NPU).
Pipelined frame rate = 1 / max(NPU, GPU) under contention: 1 / 44.8 ms = **22 fps NPU-bound today** (vs GPU-only
pooled 15.6 fps = 64.3 ms); with one xclbin for the encoder (no switches, ~33 ms + ~8 ms for the missing pieces)
it would be ~41 ms -> ~24 fps, still NPU-bound; the GPU side alone would allow 29 fps.

## (2) kernel: Peano rejects inline asm (any asm volatile -> "unable to translate instruction: call"), ships no ISA docs and no llvm-mc assembler (bin: llc, llvm-objdump, clang); the hand-scheduled loop cannot be written with this toolchain. See RESULTS.md Phase 5.

# Phase 6 (2026-10-08, Turbo): one NPU program for the encoder; encoder pieces
Code: p6/ (design6.py, mm_f32acc.cc with -DGELU, run.py with --gelu-ops, nputloop.py).

## (1) One xclbin for all 27 blocks: design6 = design3 with ONE 31104-word A buffer per memtile
The phase-2b blocker was fc2's streamed A needing 24 load BDs (4 chunks x 3 N-tiles x 2 half-slots) next to a
24-BD channel half. Merging the two half-slots into one buffer with one lock pair makes a cached op ONE load BD
(release "full" by 2 x nt: the send chain is still two half-buffer BDs, so one N-tile = two sends) and an fc2
K-chunk of 1152 one load BD (release 2, sent once). Block chain: qkv 1 + attn_out 1 + fc1 1 + fc2 4 x 3 = 15 loads
+ 2 sends = 17 BDs. Verified in the lowered MLIR before the first run: empty init 2; loads acq 2 / rel 18 (qkv),
acq 18 / rel 6 (attn_out), acq 6 / rel 24 (fc1), fc2 chunks acq 24 then 2 / rel 2; sends acq 1 / rel 1; lengths
31104 (loads) and 15552 (sends). fc2's K is padded 4352 -> 4608 (+6% on fc2).
d8_blk (one block, 4 ops): first dispatch bit-exact on all four ops. d8_blkx27 (27 blocks, one dispatch,
1.32 GB moved): **33.8 ms per frame** (run.py 33.8 median / 33.78 min; nputloop 33.8 median, p90 33.9, 587 frames),
all ops bit-exact. vs 40.4 ms for the 4-xclbin frame loop (three ~2.7 ms context switches) and 29.7 ms of pure GEMM
work. The price of the merged buffer is fc2: one buffer cannot double-buffer its chunks, so each chunk loads and
then sends serially (~550 us per fc2 vs 308 in its own xclbin, ~6.7 ms per frame). Two buffers would need a
per-op send pattern a static chain cannot express; the fix is the runtime memtile-task path (hardware repeat
counts) or a BD-aware fc2 restructure. The projection (K 2304 = 2 buffers, 22 N-tiles -> 44 loads) cannot join a
static chain: it stays on the GPU (2 ms there; the NPU->GPU hand-off becomes the 729x1152 bf16 tower output, 1.7 MB).

## (2) Encoder pieces
### tanh-GELU fused into fc1's epilogue (-DGELU in mm_f32acc.cc): correct, but fp32 vector math is emulated
fc1 alone (d8_fc1g, 729x1152x4304): vs the float64 reference with GELU: 0.0226% of outputs differ by one bf16 ulp
(the exp polynomial vs numpy's exp), max abs error 3.14e-3 of the max, rms at the bf16 floor. Time 5.1 ms vs 0.56 ms
without GELU. Getting it to build at all took: no inlining into the 12x12 block loop (program memory 16 KB), the
GELU on 16-lane vectors (64-lane temporaries spill 5.5 KB of stack), two chains per out-of-line call (four = 3.4k
bundles, 560 B over), worker stack 0x1400. The remaining cost is intrinsic: acc_store with the two inlined chains is
1806 bundles for 32 elements = ~21 bundles per 16-lane fp32 op. AIE2P has no native fp32 vector arithmetic; the
AIE API emulates mul/add/convert with multi-instruction sequences, so every fp32 elementwise op costs ~1.3 cycles per
element and the 40-op GELU ~55 cycles per element -> 120 ms per frame for 27 x 729 x 4304 values. (The int8
dequant epilogue of phase 4, 3 ops, cost 6 cycles per element for the same reason.)
Options, not built: tanh in bf16 (native 32-lane ops, ~1 ms per frame, output flips up to one bf16 ulp for a large
share of values, rms slightly above the floor); a Pade 7/6 tanh in fp32 (~24 ops, 1e-7 accurate, ~2.5 ms per block);
or GELU applied by the consumer (same cost anywhere on the NPU; on the GPU it is free but costs 14 MB per frame of
hand-off). LayerNorm (5 ops per element over 729x1152) and softmax (exp per score) pay the same emulation tax.

### LayerNorm (p6/ln.py + ln.cc): 768x1152 bf16, 32 cores (column shim -> memtile split over 4 cores, 6 objects of 4 rows each; join back; gamma/beta as a one-shot fifo), fp32 math with four partial sums, scalar rsqrt (no libm on the core: bit-trick + 3 Newton steps; bf16->fp32 via an fp32 accumulator since aie::to_float has no bf16 overload). 246 us median (316 first process); 0.0018% one-ulp flips, rms at the bf16 floor.
### Patch embedding GEMM (d8_pe on design3, K 588->640, N 1152, nt 3): bit-exact, 147 us.
### Joint re-measure with d8_blkx27: NPU 36.8 ms during the GPU text loop (33.8 alone), GPU 34.6 (33.2 alone) -> 27 fps pipelined. Logs p6/build/{npu_joint1x.log, gpu_joint1x.log(.gpu)}.

# Phase 7 (2026-10-08, Turbo): bf16 elementwise kernels, towards the whole tower on the NPU
Code: p7/ (mm_f32acc.cc -DGELU_BF16, ln.cc/ln.py (bf16 LN), ln_res.py (LN + residual), run.py --gelu-mode bf16,
runln.py with the BF16_ELEMENTWISE-style references). The Opus emulation job kept the GPU at 96-99% busy
throughout this phase: every timing below is DDR-contended (labelled); correctness is unaffected.

## bf16 tanh-GELU in fc1's epilogue (-DGELU_BF16)
All native bf16/int ops except the exponent split: u in a bf16 op chain; z = 2u log2e exactly via a bf16 x bf16 ->
fp32 MAC; n = floor(z) and f = z - n on the conversion unit (one fp32 sub); 2^f by a degree-3 Horner with bf16 MACs
(bf16 rounding per step); e = 2^n 2^f by an int add into the exponent bits; d = e + 1; 1/d by a bf16 bit-trick seed
+ two Newton steps; t = 1 - 2/d; y = 0.5 x (1 + t). Cost: gelu_b2 = 420 bundles per 64 elements = 6.6 per element
(the fp32 version: 56); fc1 1168 us min (contended; 560 without GELU) -> ~0.6 ms per fc1, ~16 ms per frame.
Accuracy (d9_fc1b, 729x1152x4304): vs the BF16_ELEMENTWISE-style reference (every op rounded to bf16, exp exact on
its bf16 input) 29.96% of outputs differ by one bf16 ulp, rms_rel 3.55e-3 (floor 1.67e-3), max abs 4.15e-3 of max;
vs the fp32 reference 61% one-ulp flips, max 4.0e-3. The extra ~1 ulp of noise over the emulation's own chain comes
from the bf16-rounded 2^f Horner and the bf16 reciprocal; an fp32 2^f polynomial (6 emulated ops) would roughly
double the cost and get within ~0.5 ulp. Decision deferred to the emulation verdict.

## bf16 LayerNorm (ln.cc, 768x1152, 32 cores)
fp32 mean and variance through the bf16 MAC unit (x * 1 and d * d with d = bf16(x - bf16(mean))), rstd scalar
rsqrt (bit-trick + 3 Newton, no libm on the core) rounded to bf16, then bf16(bf16(bf16(d * rstd) * gamma) + beta).
**Bit-exact vs the BF16_ELEMENTWISE-style reference (0 mismatches), 142 us median / 108 min (contended)** vs 246 us for
the fp32 version. 55 LNs per frame -> ~7 ms as separate passes.

## LayerNorm + residual (p7/ln_res.py): one-row objects [x row | delta row] gathered per column by one 4-D shim pattern (sizes [24, 4, 2, 1152], strides [1152, 27648, 884736, 1]; two-row objects exceed the 10-bit wrap and decomposed into >16 BDs); outputs [y | x+delta] the same way. Both outputs bit-exact; 168 us median / 153 min (contended, GPU 99%).
## Not reached: attention, the whole tower, the joint re-measure (see RESULTS.md Phase 7 for the cost model).

# Phase 8 (2026-10-08, Turbo, GPU idle: all timings uncontended unless flagged)
Code: p8/ (mm_f32acc.cc -DGELU_BF16 = sigmoid form; expsplit.h = shared exp/reciprocal; attn.cc/attn.py/runattn.py;
ln.py/ln_res.py from phase 7; run.py --gelu-mode bf16 mirrors bf16emu.py gelu_sig).

## 1. GELU in the sigmoid form, LN re-timed
BF16_ELEMENTWISE.md verdict: bf16 elementwise acceptable (npu_full_sig_pool16 99.6% vs pool16 99.4%); GELU must be
x / (1 + exp(-2u)). bf16emu.py gelu_sig: x3 = bf(bf(x*x)*x); u = bf(C0*bf(x + bf(C1*x3))); e = bf(exp(-2u)) with exp
correctly rounded from its bf16 input; y = bf(x * bf(1/bf(1+e))).
| GELU variant | vs gelu_sig reference | gelu_b2 bundles / 64 elem | fc1 729x1152x4304 |
|---|---|---|---|
| bf16 ops, 2^f by bf16 Horner, bf16 Newton reciprocal | 38.8% one-ulp flips, rms 3.8e-3 | 392 | 948 us |
| **same with split-operand (hi+lo bf16) exp and reciprocal in the fp32 accumulator** | **0.998% one-ulp flips, rms 1.5e-4, max 3.9e-4 of max** | 492 | **1117 us** (560 without GELU: +0.56 ms per fc1, ~15 ms per frame) |
The split trick: a = hi + lo (two bf16 vectors from one fp32 accumulator via one MAC), then every product a*b is
three MACs (hi*hi, hi*lo, lo*hi) accumulated in fp32 -> ~2^-16 precision through a degree-4 Horner and two Newton
steps using only native MAC-unit ops. The same exp_acc / recip_b (expsplit.h) serve the softmax.
LayerNorm uncontended: 116 us (bit-exact); LN + residual 146 us (both outputs bit-exact).

## 2. Fused attention (attn.cc, attn.py)
Per core 6 items of (head, 64-query tile); 32-key chunks; Q arrives as the first two objects of the K fifo and
stays resident; pass 0 = row max over bf(bf(QK^T)*scale) (bf16 mmul 8x8x8, fp32 accumulate, Q/K/V blocks straight
from the qkv GEMM's blocked bf16 tiles, K blocks transposed in registers); pass 1 = e = bf(exp(bf(s - max))) with
exp_acc, fp32 row sum, fp32 PV accumulation; the bf16 1/sum is applied once at the end (the reference normalises
each p before PV: a <= 1-ulp-of-p deviation, measured). L1: Q 9.2 KB, S chunk 4 KB, O_acc 18.4 KB, K/V fifos 2 x 2 x
4.6 KB, O out 2 x 4.6 KB. Streaming per core per block: 6 x (2 x 105 + 105) KB = 1.9 MB.

## Attention: build trail (attn.cc / attn.py, 16 heads x 729 x 72, bf16emu numerics)
| step | result |
|---|---|
| first design: separate K and V fifos per core | memtile needs 9 MM2S (two 4-way splits + O), has 6 -> K and V share one stream (pass 0: K chunks; pass 1: K, V pairs) |
| first dispatch | **TIMEOUT (1 of 3 this phase)**: IRON `acquire(n)` means "hold n objects in total", so `acquire(1)` twice held one object against a `release(2)`; the lock counts drifted until the cores waited on objects that never came. Known-good run after it: bit-exact, device fine. Fixed with `acquire(2)` + element indexing. |
| scalar kernel | correct, rms 8e-3, **18 ms per block** (scalar row max/sum loops, per-element masks, 8x redundant K transposes) |
| vectorised: shuffle_down + filter_even row reductions | 4.1 ms but rms 2.9e-2: the debug build (output tile = per-row sum / max / 1/sum) showed the row maxima wrong (0.39 vs 0.82) and the sums exactly e^(dmax) off -> the shuffle lane mapping was wrong |
| transpose + extract reductions | fp32 64-lane `aie::transpose` does not exist (incomplete type); bf16 8x8 does |
| bf16 transpose, 8-lane bf16 `aie::max` | sums right, maxima still wrong: sub-native (8-lane, 128-bit) bf16 max is broken on the NPU; 8-lane fp32 max works |
| final reductions: three native 16-lane bf16 maxes / four MAC-unit sums over the transposed block + one 8-lane fp32 op | maxima within 1 bf16 ulp (fp32 accumulation order of QK^T), sums within 0.5%, **output rms 6.2e-3 vs the emulation reference** (2.5e-3 of it = normalising once after PV instead of per p), max abs 2e-3 on 0.18 |
| other Peano pitfalls | a vector assigned under a runtime condition (`if (padded) m = ...`, `aie::select`) crashes Peano's EarlyIfConverter ("Bad PHI"): masks are applied unconditionally; two inlined 32-lane exp chains spilled 13.5 KB of stack -> out-of-line exp, 9.9 KB stack; L1 forced O depth 1 (two half-tile writes), per-use K transposes (no 4.6 KB scratch); scalar fp32 divide/compare are software calls (`__divsf3`, `__gesf2`) |
| cost (4.4 ms per block) | PV 0.8 ms (variant without PV: 3.6); exp is not it (EXP_FAST: 4.18, rms 6.7e-3); static masks + vector max/sum state: 3.78 ms (min 3.74) |
Per chunk the kernel still takes ~14k cycles against ~6k of counted vector work: register spills (9.9 KB stack) and the
per-row-block structure (32 score blocks of 8x8 per 32-key chunk, 288 transposes + 576 loads per chunk) dominate.
Next: K stored transposed by the producer (the qkv GEMM epilogue can emit K^T blocks), 64-key chunks (halves the
per-chunk overhead; needs ~9 KB more L1: drop O_acc to bf16 hi/lo pairs or 48-query tiles), event-counter profiling.


# Phase 9 (2026-10-08; NPU Turbo; GPU idle; all work in npu-iron on the NPU only)
Code: p9/ (attn4..attn15 .cc/.py, runattn4/8/10.py, tracesum.py / tracekv.py / tracetl.py, model4.py). The deliverable
is **attn15.cc + attn15.py (attn13.py's design) + runattn8.py**. Builds live in p9/build/NAME.{xclbin,bin,json,log,prj}.

## Attention: build trail v4 -> v15 (per block = 16 heads x 729 x 72; Turbo; rms vs the bf16emu `attn` reference)
| step | result / what it taught |
|---|---|
| v4: bfp16ebs8 operands (host-packed panels, 8x8x8 bfp mmul, P converted on chip with `accum.to_vector<bfp16ebs8>()`) | ran (2.0 ms) but NaN: the block-vector streams load 512-bit words and reset their position on `seek`, so a seek target must be 64 B-aligned; 9 panels = 648 B is not. Panels are now grouped and each group padded to a 64 B multiple (Q/K: 9 panels -> 704 B, V^T: 6 panels -> 448 B; objects 4224 B) |
| v4 fixed: 2.04 ms, rms 1.49e-2 | the host error model (model4.py) showed the kernel 9e-3 away from every model: the on-chip fp32->bfp conversion of P truncates and the row sum came from the un-truncated bf16 e |
| v4b: row sums as a 10th O block = P x a ones V^T panel (sum of exactly the P the mmul multiplies), symmetric exp fraction | 1.82 ms, rms 1.13e-2 |
| v5: exp as 3 stage loops with L1 staging | 1.43 ms; disassembly: ~98 bundles per 32-lane exp, mostly 32-lane int32 (two-register) ops, srs/ups moves and mode-register writes |
| v6: magic-constant exp (z + 1.5*2^23 in the MAC unit), 3 K streams / 2 V streams | slower (1.61) and less accurate: the core has two FIFO load units; 3-4 live streams spill their state. The exp looked like a MAC precision loss - it was not (see v7r) |
| v7: Q row block (9 panels) and P row block (6 panels) in block-vector registers, two K / two V streams, bf16-arithmetic integer split | 0.88 ms, rms 1.40e-2; the exp ramp dump (DEBUG_EXP) showed `n = floor(z)` and truncated conversions everywhere |
| **v7r: `aie::set_rounding(conv_even)`** | the core's conversions default to **floor**; ln.cc and mm_f32acc.cc set RNE, no attention kernel (v2..v7) ever did. With RNE: rms **1.03e-2**, 0.885 ms; the v6 magic exp is exact (94% of a ramp within 0.5 ulp) |
| v8: K/V broadcast through the memtile (`f.cons().forward(tile=mem)`, 4 consumers), 4 Q objects per item on the same fifo, 9.7 MB per block instead of 33.8 | 0.955 ms: not DMA-bound after all (nokernel variant: 0.29) |
| v9: magic exp + Horner on f, f^2, f^3; bf16-only running max (16-lane max / `aie::lt` mask; 16-lane loads need 32 B alignment -> `g_maxb[RB*16]`); 4 accumulator chains (even/odd panels, `aie::add(acc, acc)`) | rms **9.6e-3** (best), 0.90 ms: still no faster |
| v10/v11: K and V on separate fifos | 0.82 ms |
| v12: one stream per row block with `seek_byte(0)` between groups | NaN: the stream's fill pointer is not where I assumed after a partial 64 B word; dropped |
| **hardware trace** (`Program.enable_trace`, one-column build - the trace unit cannot be routed beside 8 broadcast columns; parse with `aie.utils.trace.parse.parse_trace(buf, input_with_addresses.mlir)`) | per chunk 8.4k cycles with ~1000 cycles of lock stall between chunks: DMA-in ran only *between* chunks. IRON gives the forwarded L1 fifo at most 3 free slots (lock init 3 for depth 4 or 6, 2 for depth 3), and the paired `acquire(2)` left one slot: no prefetch. Inside chunks INSTR_VECTOR is only 25% of the cycles, MEMORY/STREAM/LOCK stalls 5%: the rest is the dependent-chain schedule |
| v13: chunk split into attn_chunk_k (K object) / attn_chunk_v (V^T object), one object held at a time; P panels per row block padded to 448 B | 0.845 ms; gaps 36 cycles, DMA overlapped; K-phase 6.7k, V-phase 1.5k cycles; **item boundary 21.6k cycles (25% of the item)**: scalar begin/end/copyq |
| v14: elementary-function unit (`aie::exp2<bfloat16>(vector<float>)`, `aie::inv(bf16)`), once-per-core init, vector Q copy and zeroing | 0.51 ms but rms 2.7e-2: the native exp2 is a ~4-bit approximation (ramp: 8% rms, 23% max error). Boundary 5.5k cycles |
| **v15: v14 with v9's exact exp** | **0.675 ms/block (min 0.672), rms 9.8e-3 vs emu, 6.5e-3 vs the bfp-operand reference** (the native bf16 reciprocal for 1/sum costs 2e-4). v15s (stage-major exp) was wrong and no faster; dropped |

Error floor: with exact exp and bf16 P, bfp-rounded Q/K/V alone put the host model 8.1e-3 from the bf16emu reference
(diffuse random-data attention: o is a mean of ~300 v's, so operand rounding noise does not average out). v15 is at
that floor; v2's 6.2e-3 came from the API's bf16 mmul (which also converts to bfp, but P was normalised first).

## Learned (keep)
- **Set `aie::set_rounding(aie::rounding_mode::conv_even)` in every kernel.** Default is floor; it silently applies to
  all bf16/bfp16 conversions. Phase-8 attention numbers were taken in floor mode.
- Block-vector streams: start/seek only at 64 B-aligned addresses; pad panel groups; one stream = one FIFO load unit,
  two units per core; never try to continue a stream across a group pad.
- `acquire(n)` holds n objects: with n > 1 and IRON's forwarded-fifo lock cap (3 free slots) there is no prefetch;
  split the kernel per object instead of deepening the fifo (depth 6 still lowers to 3).
- The elementary unit's exp2/inv are coarse (exp2 ~4 bits; inv ~1 ulp): inv is fine for 1/sum, exp2 is not for P.
- A vector of fp32 (8 or 32 lanes) ops is emulated; keep the max/compare bookkeeping in bf16 (exact for bf16 scores)
  and integer work in int16 (one register); `aie::add(acc, acc)` is native, `aie::filter_even` extracts low halves.
- Hardware trace works from IRON (`Program.enable_trace(trace_size, workers=[w], coretile_events=[...])`, trace buffer
  = last kernel arg, `--trace` in runattn8.py); the default events give INSTR_EVENT_0/1 windows, INSTR_VECTOR,
  MEMORY/STREAM/LOCK_STALL and the DMA port activity. ACTIVE/GROUP_STALL produced no intervals.
- Peano packs only ~1.5 ops per bundle in these kernels and spills ~280 words per call; INSTR_VECTOR 25% of cycles
  means the next 2x is scheduling (the queued hand-scheduled route), not instruction count.

## Not done
GELU (9.2): untouched; the native exp2/tanh are too coarse to replace the split-operand exp, so the 15 ms stays until
the GEMM epilogue is rescheduled. Also not done: 64-key chunks (L1: two fifos x depth 3 already ~58 KB), K^T from the
qkv epilogue (K is consumed in its natural layout now - no transposes anywhere, so this item is void), 2 timeouts spare.
Timeouts this phase: 0 of 3.


# Phase 10 (2026-10-08; NPU Turbo; GPU idle for all timings below; NPU-only work in npu-iron)
Code: p10/ (asmproof.sh, asmfix.py, objcmp.sh, tdparse.py, asmcheck.py, learnexact.py, gen_scores.py -> scores.s,
buildasm.sh / buildmerge.sh, attn15.cc with -DASM_SCORES / -DONE_CHAIN / -DNO_EVENTS, attn16.py = attn13's design
with ATTN_OBJ=<prebuilt .o>, td/ = llvm-aie aie-public AIE2P TableGen (read-only copies)). Builds in p10/build.

## 1. Toolchain proof (done, bit-exact)
`clang++ -S` of attn15.cc -> `clang --target=aie2p-none-unknown-elf -c k.s` fails on three dialect gaps of Peano's
own output: `movxm p, ##sym` (needs `#sym`), `.word` (needs `.4byte`) and `event #n` (the integrated assembler has
no parse for it at all). asmfix.py rewrites the first two; with the kernel compiled -DNO_EVENTS the reassembled
object is **byte-identical** to the direct `-c` object (cmp + objdump -s + objdump -d: zero differences, objcmp.sh).
Linked into the design with IRON's `Kernel("attn_chunk_k", "<path>.o", [...], stack_size_override=0x2800)`
(`.object_file.bind(...)` for the other entry points; the Worker needs 0x2900 because aiecc adds the libcalls on
top of the override), the NPU output is identical to the compiled kernel's (np.array_equal) at the same speed
(0.673 vs 0.676 ms/block).
Keeping event markers in a hand-built object: compile the C++ part normally (events intact) and assemble only the
hand-written .s, then `ld.lld -r cpp.o asm.o -o merged.o` (buildmerge.sh). Injecting the standalone `event`
encoding as `.byte` into a bundle stream hung the core twice (the two timeouts of this phase; known-good test OK).

## 2. Latency table and checker (done, calibrated)
tdparse.py (now common/tools/gen_isa_timing.py) turns AIE2PGenInstrInfo.td + AIE2PGenSchedule.td into
aie2p_lat.json (now common/tools/aie2p_isa_timing.json): 879 instruction defs, slot class
(lda, ldb, st, mv, vec, alu, lng - one instruction per slot per bundle) and per-operand stage latencies. The ones
that matter here: bfp/bf16 `vmul.f`/`vmac.f` result 6, accumulator operand read at stage 4 (a dependent chain can
issue every 2 cycles); `vldb/vlda.pop.576` result 8, `vlda/vldb` 7, conversions `vconv.*` 2 (`vconv.bf16.fp32` is
in the st slot, `vconv.fp32.bf16` in mv), `vsrs` 4, `vups` 3, `vshuffle`/`vbcst`/`vmax_lt` 2. Calling convention:
pointers p0-p5, ints r0-r7, callee-saved r8-r15/p6/p7, no vector register is preserved across calls. The fifo
pops' pointer field is 1 bit (p0/p1 only), `vmax_lt.bf16` names r16, `vst`/`padda` immediates are 10-bit.
asmcheck.py walks a function's bundles (one per cycle) with the register aliasing x/ex/y, dm/cml/cmh/bm* and
reports every read whose program-order producer has not landed; reads of an older value while a newer write is in
flight are listed as info (Peano itself pops into the same register every cycle and reads at the landing cycle).
Calibration: 0 violations on all of Peano's own functions in attn15.s.

## 3. Hand-scheduled score block (done, bit-exact, no speed-up)
gen_scores.py emits attn_scores_asm (162 bundles per row block): Q panels through the lda-slot fifo (`vlda.fill/
pop.576 [p0, lf0, r24]`), K through the ldb-slot one, one panel pair every 2 cycles into 2-register conveyors read
exactly at landing, a single 9-deep mmul chain per key block (mac every 2 cycles, the chain's own bound),
conversions / scale multiply / stores / running max overlapped with the next block's pops. Checker: 0 violations.
Output identical to the C++ (a -DONE_CHAIN build with the same accumulation order - which also equals the 4-chain
build's output). But the K-phase trace is unchanged (6730 cycles median per chunk, both builds): Peano's own score
section was already at the chain bound; the phase-9 "scores 0.2 ms" differential had also removed the P stream and
the pops. The K-phase cycles are the exp (~440 per row block = 12 vectors x ~37; bound ~13/vector) and the softmax
bookkeeping, not the mmuls.

## Not done
The exp block in asm (the real target: ~2.6k of the 6.7k K-phase cycles; ~13 vec-slot ops per 32 lanes, so ~1.1k
is possible), the GEMM inner loop, the GELU epilogue. Timeouts: 2 of 3 (both the `.byte` event experiment).


# Phase 11 (2026-10-08; NPU Turbo; GPU idle for all timings; NPU-only work in npu-iron)
Code: p11/ (gen_gemm.py -> gemm_blk.s, mm_f32acc.cc with -DASM_BLK, design6.py with GEMM_OBJ=<prebuilt .o>,
buildgemm.sh (C++ wrapper compiled normally + assembled .s merged by ld.lld -r), run.py with DUMP=, gemmhyp.py /
gemmdiag.py (output diagnosis), mm_gelu.cc = the phase-8 GELU kernel, asmcheck.py with a default latency for the
table-less moves). Builds in p11/build: d11_ref = d8_blkx27 rebuilt, d11_asm* = the hand-scheduled GEMM loop.

## 1. GEMM inner loop: bound, schedule, what the hardware did
Peano's deployed loop (d8_blkx27, BLK12 unroll 2): 51 bundles / 32 MACs. Its resource bound from the slot classes:
the ldb slot carries all 32 B pops + 4 fills = 36 per 32 MACs (A is hoisted in ex registers; 16 fp32 quarter loads
in lda, 16 stores in st, 32 mmuls in vec) -> 1.125 bundles/MAC = 89% of the MAC slot; the compiler loses 15
bundles to its two-stage modulo schedule (bursts of pops, 8 accumulator shuffles through vmov).
The hand schedule (gen_gemm.py, II 18 per 8x16 block, 253 bundles per 192-MAC call = 1.32 bundles/MAC, 0 checker
violations) ran at 31.5 ms per frame (reference 34.1) but computed exactly the sum over the odd k-subs (gemmhyp.py
fits the output to A[:, mask] @ W[:, mask]^T: odd k-subs rms 1.65e-3 = the bf16 floor, everything else ~1):
with two accumulator chains interleaved one mac per cycle, a chain's dependent mac 2 cycles after its predecessor
(acc read stage 4, result 6 in the tables) reads the accumulator from before that predecessor, so every other mac is
lost. The phase-10 score block used distance 2 on ONE chain with nothing in between and was bit-exact: the
accumulator bypass only covers the immediately preceding vector op. Peano's loop keeps >= 3 cycles between dependent
macs when other macs intervene (and renames: `vmac.f dm4, dm3, ...`). Register choice is not the cause (A/B
registers swapped to Peano's own sets: same odd-k-sub result).
Second schedule: dependent macs 3 apart (pairs of macs, one idle vec cycle), pops following the same pattern so the
4-register B conveyor is still read exactly at landing (the first attempt at this moved the macs but not the pops
and read newer pops: different garbage, caught afterwards by the generator's new intent check): II 26, 352 bundles
per call = 1.83 bundles/MAC, plus two more fixes found from the output map (gemmmap.py: which 8x8 blocks are
right): a loaded accumulator quarter landing in the same cycle a mac reads it is stale (so the accumulator operand
is read at stage 3, not 4 - the same off-by-one as the chain rule), and the B-stream re-init (r25, p1) must come
after the row block's last pop. d11_asm6: **bit-exact on all four ops (0.0000% mismatch)**, 33.6 ms per frame
(reference 34.1) - a correct hand-scheduled loop at 1.83 bundles/MAC lands where Peano's 1.59 does.
What this means for the GEMM budget: with 5 accumulator registers, two outputs per block and a 3-cycle chain
distance, one mac per cycle needs four interleaved chains (even/odd k-subs per output) and there is no room to
double-buffer the next block's accumulators; the realistic hand-scheduled II is ~20-22 per block (~1.35/MAC with
the per-call prologue and drain), i.e. ~15% over Peano's 1.59 - and the frame is mostly DDR-bound anyway: run.py
moves 1317 MB per frame (A 403, B 484, C 430) at 37-39 GB/s = the whole 34 ms; the (wrong but 17% faster) kernel
only took the frame from 34.1 to 31.5 ms. The 20 ms target needs less traffic (A is re-read per N tile: 403 MB
for ~190 MB of real A), not a faster loop.

## 2. GELU epilogue: bound measured, not scheduled
gelu_b2 (64 elements per call, phase-8 sigmoid form) compiles to 492 bundles with 94 spills and 154 nops. Per
call it issues 114 vec-slot ops (52 vmac.f, 16 vmul.f, 20 vsub.f, 10 vadd.f, 8 vadd, 6 vsub, 2 vneg), 107 st-slot
(55 vconv.bf16.fp32, 36 vst of which 34 are spills, 14 vsrs), 79 mv-slot and 70 vmov (any slot): the resource
bound is ~114 bundles per 64 elements, 4.3x below the current 492 -> ~11 of the 15 ms/frame are at stake. It is
the largest remaining win and a pure schedule problem (no fifo streams, two independent 32-lane chains), but the
chain is long (40+ dependent steps) and the int conversions (to_fixed/to_float emulations, vups/vsrs with mode
registers) need their exact forms copied from the compiler's output; not started.

## 3. Attention exp block: not reached (bound 13 vec ops per 32 lanes vs ~37 cycles; ~0.13 ms/block).

## Learned (keep)
- A dependent `vmac.f` on the same accumulator needs 3 cycles when any other vector op is issued in between; 2 is
  enough only back-to-back (the accumulator bypass is one deep). The tables (acc1 stage 4, dst 6) say 2; the checker
  now cannot know this - keep chains >= 3 apart in hand schedules.
- The generator must assert conveyor intent (which pop a read sees); the checker's "latest landed write" semantics
  accept any landed value, so a read of the wrong (newer) pop is not a violation.
- `movs`/`mov` pointer writes have no itinerary in the tables: give them 2-3 cycles before a load uses the pointer.
- GEMM frames are DDR-bound at 37-39 GB/s; kernel speed-ups show only partially.
Timeouts this phase: 0 of 3.


# Phase 12 (2026-10-08; NPU Turbo; GPU idle; NPU-only work in npu-iron)
Code: p12/ (mm_f32acc.cc = the phase-8 GELU kernel + -DGELU_STAGED, design6.py with GEMM_OBJ, run.py (p8's bf16
GELU reference) with DUMP=/DUMP_OP=, fnsize.py). Builds p12/build/d12_gelu_ref (fc1 + GELU, = phase 8's d10_fc1s),
d12_gelu_st. Timeouts: 0 of 3.

## 1. GELU epilogue: measured what a hand schedule is up against; not hand-scheduled
Reference (gelu_b2, two 32-lane chains per call): 492 bundles / 64 elements, 74 spills; fc1 + GELU 1108 us (min),
0.9986% one-ulp flips vs the bf16 sigmoid reference (phase 8's number, reproduced).
Experiment: the same arithmetic as three stage loops over GV independent vectors with L1 staging (gelu_staged, bit-
identical output - np.array_equal against the reference build). As rolled loops (GV 8): 1177 us, slower - a rolled
stage iteration is still one dependent chain. Fully unrolled: GV 2 / 4 / 8 -> 471 / 469 / 481 bundles per 64
elements (70 / 122 / 302 spills): exposing 2, 4 or 8 independent chains per stage changes nothing. Peano issues
about one operation per bundle here (467 non-nop ops per 64 elements in the reference = 470 bundles): the schedule
is neither latency- nor slot-bound but allocation-bound - every extra chain turns into spills. The slot bound stays
~114 (vec) / ~110 (st: the 55 fp32->bf16 conversions + 14 vsrs + 2 stores) per 64 elements, 4x away.
What a hand schedule needs: a list scheduler with explicit register allocation over ~80 ops per 32-lane chain
(12 x + 5 dm registers allow ~3 chains in flight), the exact Peano forms for the fp32->int and int->fp32 emulations
(magic-constant vadd.f/vsub.f, vups/vsrs with the s0/s1 shift registers and crupsmode/crsrsmode), vsel.32 for the
clamp, and the stage-3 accumulator rule. Realistic result ~150-200 bundles / 64 elements -> GELU 15 -> ~5-6 ms per
frame. Not started: it is the single largest remaining win and the largest remaining effort.

## 2. GEMM DDR traffic: analysed, not changed
run.py's accounting per 27-block frame: A 403 MB, B 484 MB, C 430 MB = 1317 MB at 37-39 GB/s (~34 ms). B (the
bfp16 weights, 17.9 MB per block) and C (bf16 outputs, 16 MB per block) are read / written once: inherent. A is
read ~2x (7 MB per block of real bfp16 input vs 14.9 streamed): the memtile A slots (a_ring 6 x one 16-row sub-tile
per k-step) re-stream A for every N tile. Keeping an op's A resident in the memtile needs 4 cores x 96 rows x K x
1.125 B = 498 KB for K = 1152 (the memtile has 512 KB, of which the B/C paths use ~150 KB): only a half-K
residency fits (halves the A re-reads, -100 MB), fc2 (K 4608) not at all. bfp16 C would cut C by 44% (430 ->
242 MB) but every op except fc1 -> fc2 has a LayerNorm / residual in bf16 between it and the next op; fc1 -> fc2
(191 MB of C, 215 MB of fc2's A re-reads) is the one pair where a bfp16 output could feed the next A directly
(-84 MB written, -107 MB read). Total reachable ~300 MB of 1317 (23%) -> ~7 ms per frame if the frame stays
DDR-bound; each piece is a design6 change (memtile BD chains, A ring, host packing). Not built this phase.

## 3. Attention exp block: not reached.
