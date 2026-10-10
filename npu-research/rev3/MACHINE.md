# MACHINE.md — the XDNA2 / AIE2P NPU in testbox, as a performance model

Revision 3, 2026-10-09 evening: the clock is MEASURED at 1.81 GHz (r7/clk.py), the core timer works with the processor
bus enabled, and section 2.2 (L1 banks and locks, from the round-8 microbenchmark and the TableGen model) is new.
Revision 2 is in `rev2/`. Revision 2, 2026-10-09, written against round 6 (`full12r6e`, 76.3 ms per depth-12 step) and the main agent's
results in `RESULTS_FROM_MAIN.md`. Revision 1 (round 5) had two wrong constants, the clock (1.43 GHz) and the
memtile -> core stream rate (4 B/cycle). Both are corrected below; ANALYSIS §0 explains how they misled the analysis.

Every fact carries a tag:
- **[DOC]**: from a vendor source. Each one cites its file or page.
- **[MEAS]**: measured in our own work. Each one cites the result or file.
- **[INF]**: inferred: a model fitted to measurements, or arithmetic on documented facts. Treat it as a hypothesis
  until a proposal in PROPOSALS.md confirms it.

Sources on disk (read without a device):
- In the `npu-iron` image: mlir-aie 1.4.4's `AIETargetModel.h` (TM), aie-rt's `xaie2pgbl_params.h` (PARAMS) and
  `xaie_events_aie2p.h` (EVENTS), Peano's `aie2pintrin.h`, the AIE API headers.
- On testbox's host: the amdxdna 2.21 driver source in `/usr/src/xrt-amdxdna-2.21.0/driver/amdxdna/` (DRV).
- llvm-aie's AIE2P TableGen files, `npu-moondream/p10/td/` (TD).
- The mlir-aie main checkout of 2026-10-06, `npu-moondream/mlir-aie/` (UPSTREAM).

Web sources are AMD AM020 (AIE-ML) and AM027 (AIE-ML v2).

**AIE2P is not AIE-ML v2.** In mlir-aie, `aie2ps` is the AIE-ML v2 part (VE3858, Versal AI Edge Gen 2), with its own
target model. `aie2p` is Strix XDNA2. testbox's device is PCI 1022:17f0 revision 0x11, "NPU Strix Halo"
(`/sys/class/accel/accel0/device/{device,revision,vbnv}`), which the driver runs as `npu5` with the `npu4` family's
tables (DRV `amdxdna_pci_drv.c`, `npu5_regs.c`).

## 1. Array and clock

| item | value | tag |
|---|---|---|
| geometry | 8 columns. Row 0 = shim, row 1 = memtile, rows 2-5 = cores: 32 cores, 8 memtiles | [DOC TM `NPU2`] [MEAS design8: rows 2-4 GEMM, row 5 EW] |
| **AIE clock (H-clock)** | **1.81 GHz measured in Turbo** (r7/clk.py: two pure-register loops, host slope and core timer agree); 1.8 GHz is the driver's top DPM level. Default mode not measured | [MEAS round 7] [DOC DRV] |
| peak MAC rate | 32 cores x 512 MAC x 1.8 GHz = 29.5 TMAC/s = **59 TOPS** (bfp16 / int8). The driver's own formula (`NPU4_DPM_TOPS`: 4096 ops x columns x H-clock) gives the same | [DOC DRV `aie2_smu.c`] |
| one program at a time | one hw context per xclbin; switching costs ~2.7 ms; launch floor 97 us median from pyxrt | [MEAS RESULTS 7] |
| the "slow mode" | sub-ms dispatches sometimes finish on a ~1 ms timer for minutes (driver/firmware) | [MEAS moondream 2b] |

**The clock, from the driver source** (DRV, amdxdna 2.21):
- `npu4_regs.c` holds the DPM table, `{npuclk, hclk}` in MHz:

  | level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | MP-NPU clock | 396 | 600 | 792 | 975 | 975 | 1056 | 1152 | 1267 |
  | H-clock (AIE array) | 792 | 1056 | 1152 | 1267 | 1267 | 1408 | 1584 | **1800** |

  `npu5_regs.c` (Strix Halo) uses `NPU4_COMMON_DEV_PRIV`, so the same table. The H-clock is the array clock: the
  driver computes TOPS from it.
- `aie2_pm.c`, the power modes:

  | mode | DPM level | clock gating |
  |---|---|---|
  | TURBO | max (7) | off (only settable with no context open) |
  | HIGH | max (7) | on |
  | DEFAULT | the highest level any open context requested (`dft_dpm_level`); 0 with no context | on |
  | MEDIUM | min(max, 3) | on |
  | LOW | 0 | on |

- `aie2_ctx.c` `aie2_calc_ctx_dpm`: a context whose QoS is incomplete (no gops, or no fps and no latency) and which
  is not low priority requests the max level. pyxrt's `hw_context(dev, uuid)` passes no QoS, so it should get
  level 7 [INF: we have not seen the QoS XRT sends; one `xrt-smi examine -r platform` while a context is open settles
  it, P10].
- **So Default and Turbo both run the array at 1.8 GHz while our context is open.** Turbo only disables clock gating,
  which matches the old observation that Turbo "mostly cuts dispatch latency" (validate 50 -> 29 us).
  - The old "~1.4 GHz Default / 1.43 Turbo" came from the moondream phase-2c spin loop: a 364-bundle call in 0.261 us
    (1.39 GHz if every bundle took one cycle). The call includes stalls and control code. At 1.8 GHz it took ~470
    cycles. It was a lower bound, read as the clock.
  - 1.408 GHz is DPM level 5's H-clock. A level-5 reading somewhere may also have fed the number; we cannot tell.

**Measured lower bounds, consistent with 1.8 GHz** [MEAS]:
- The 2x-MAC twin (P1) adds 1.544 us per tile-k-step. The added code is 2,718 bundles (static schedule of the
  deployed `mm_acc`, r6/mm8o2.s), so f >= 1.76 GHz if the added loop never stalls (`tools/sim9.py`).
- Over the whole step, 15,552 k-steps x 2,718 bundles = 23.5 ms at 1.8 GHz and 29.6 ms at 1.43 GHz; measured
  +24.7 ms. The 1.43 GHz clock cannot produce that number.
- The GELU kprobe: 110 bundles per vector in 63 ns, so f >= 1.75 GHz.
- The main agent's round-7 clock probe (`npu-tapnext/r7/clk.cc`: pure-register loops of known bundle count, several
  trip counts) will measure it directly. Expect 1.80 GHz.

## 2. Compute tile (core + 64 KB memory module)

| item | value | tag |
|---|---|---|
| VLIW slots | one instruction per slot per bundle: lda, ldb, st, mv, vec, alu, lng | [DOC TD, p10/aie2p_lat.json] |
| registers that matter for GEMM | **5 accumulators** dm0-dm4 (2048 bit = one 8x8 fp32 block each, quarters bmll/bmlh/bmhl/bmhh); **12 ex registers** ex0-ex11 (576 bit = one bfp16ebs8 block of 64 values); 6 ey pairs | [DOC TD `AIE2PRegisterInfo.td`: eDM, mEXa, eEY] |
| MMUL rate | 512 MAC per `vmac.f` (bfp16 8x8x8 or 4x8x16; dense int8 the same) | [MEAS moondream] [DOC TD] |
| bfp16 mmul forms | `vmac.f dmD, dmS, exA, exB, r(780)`: 8x8 x 8x8^T, accumulator in dmS, result in dmD (D != S allowed). EX x EY = 4x8x16 (the AIE API's 8x8x16 is two of these plus an A shuffle). EX x QEY / EY x QEX = structured-sparse B | [DOC TD lines 8872-8908; AIE API `aie2p/mmul_bfp16_bfp16.hpp`] |
| **FIFO pops** | Block-vector pops exist on **both** load slots: `vlda.pop.576` and `vldb.pop.576` (also .544/.640/.704, 2d/3d forms). Two FIFOs: (p0, lf0, r24) and (p1, lf1, r25), each usable from either slot. A pop delivers 576 bits and refills 512; one `fill` per 8 pops makes up the 72/64 B difference. Peano puts every pop of the deployed GEMM on ldb | [DOC TD `VLDA_POP_576*`, `VLDB_POP_576*`, `plfr0/plfr1`] [MEAS r6/mm8o2.s] |
| accumulator timing | a `vmac` reads its accumulator at stage 3 (the .td says 4): dependent macs at distance 2 or >= 3 | [MEAS p11] |
| other latencies | pop +8, load +7, conversion +2, vsrs +4, vups +3; pointer moves 2-3 cycles before use | [MEAS/DOC p10] |
| no fp32 vectors | fp32 vector add/mul/convert are emulated (~21 bundles per 16 lanes). bf16 is native at 32 lanes | [MEAS] |
| rounding | defaults to FLOOR; `conv_even` must be set per core | [MEAS] |
| elementary unit | `aie::inv` scalar; `aie::exp2` ~10 ulp p99; `aie::tanh` crude; 4-way LUT load (`load_lut_2x_float`) is fast | [MEAS round 6] |
| program memory | 16 KB per core | [DOC TM] |
| data memory | 64 KB, 4 banks for allocation | [DOC TM] |
| neighbour memory | own + S, W, N neighbours (TM bases: S 0x40000, W 0x50000, N 0x60000, own 0x70000) | [DOC TM] |
| locks | 16 per core tile, 64 per memtile; values 6-bit; a kernel reaches its own locks at id + 48 | [DOC TM] [MEAS] |
| core DMA | 2 S2MM + 2 MM2S, 16 BDs (fields in PARAMS `MEMORY_MODULE_DMA_BD*`) | [DOC PARAMS] |

### 2.1 Timers, performance counters and the processor bus (for a safe profiler)

| item | value | tag |
|---|---|---|
| timers | 64-bit, one per module: core module `Timer_Low` 0x340F8 / `High` 0x340FC; memory module 0x140F8 / 0x140FC; memtile 0x940F8 | [DOC PARAMS] |
| core access to tile registers | `read_tm(addr)` / `write_tm` (Peano `aie2pintrin.h`) compile to `lda.tm` / `st.tm` at 0x80000 + addr: the core's **processor bus** | [DOC Peano headers, TD `LDA_TM*`] |
| **why `read_tm(0x340F8)` hung in round 6** | **Core_Processor_Bus (0x32038) bit 0 `Enable` resets to 0, and "without this the core will hang on access to the processor bus"**. Bit 2 is `SLVERR_ON_ACCESS` | [DOC UPSTREAM `test/npu-xrt/tile_mapped_read/aie.mlir`, `programming_examples/basic/dma_compression/` (runs on npu2); PARAMS `CORE_MODULE_CORE_PROCESSOR_BUS*`] |
| second hazard | back-to-back `st.tm` / `lda.tm` packed into one bundle (or into adjacent bundles) can stall the core: separate them with `__builtin_aie2p_sched_barrier()` and check with `llvm-objdump` | [DOC UPSTREAM dma_compression README, mlir-aie issue #2346] |
| `aie::tile::cycles()` | calls `get_cycles()`, which aie2p's compat header only declares (link error); no timer register exists in the core register file (TD control/status registers list none) | [DOC] |
| perf counters | 4 per core module: `Performance_Control0/1` 0x31500/0x31504 (start/stop event per counter, 8 bits each), `Control2` 0x31508 (reset events), `Counter0-3` 0x31520-0x3152C, `Counter*_Event_Value` 0x31580-0x3158C | [DOC PARAMS] |
| core events (ids) | TRUE 1, PERF_CNT_0 5, GROUP_STALL 22, MEMORY_STALL 23, STREAM_STALL 24, CASCADE_STALL 25, LOCK_STALL 26, ACTIVE 28, DISABLED 29, INSTR_EVENT_0 33, INSTR_EVENT_1 34, INSTR_CALL 35, INSTR_VECTOR 37, INSTR_LOAD 38, INSTR_STORE 39, INSTR_LOCK_ACQUIRE_REQ 44, PROCESSOR_BUS_ERROR 72, PORT_RUNNING_0 75, PORT_STALLED_0 76 | [DOC EVENTS `XAIE2P_EVENTS_CORE_*`] |
| host register writes | the runtime sequence can write any tile register: `npu_write32` / `npu_maskwrite32(address, value, mask, column=, row=)` (IRON 1.4.4 `aie.dialects.aiex`) | [DOC IRON] |
| host register reads | control packets: a read packet to a tile's TileControl port returns the value as a packet to a shim S2MM (`aie.utils.trace.utils.create_ctrl_pkt`, `setup.configure_packet_ctrl_flow`, `config_ctrl_pkts_aie`, one target, one word per transfer) | [DOC IRON 1.4.4] |
| trace unit | event packets with timestamps; IRON `enable_trace`. Works on one-column builds. UPSTREAM main adds lateral routing to a spare column and `kernel_design.cycles_per_call` (whole-call intervals from event0/event1 markers) | [MEAS round 4] [DOC UPSTREAM] |

### 2.2 L1 banks, locks and the call ABI (revision 3)

| item | value | tag |
|---|---|---|
| L1 banks | 4 x 16 KB: 0x70000, 0x74000, 0x78000, 0x7C000 (the linker script's bank0-3) | [DOC ld script] |
| bank conflicts, measured | the asm GEMM call (225 bundles) alone, A / B / C-load / C-store placed per bank: all four apart 238 cycles; C loads and stores in one bank (in-place accumulators) 271; A and B in one bank 252; both 285; A or B with C 274-280; all in one ~322. So a same-bank pair costs ~14-36 cycles per 225-bundle call, depending on how often the schedule co-issues them | [MEAS round 8 `build/mmb_run.out`] |
| the C++ GEMM loop | runs at its static schedule in place (333 = 321 + 12 of stamps): its lower access density (28 bundles per 16 MACs) avoids the conflicts | [MEAS r7 prof] |
| deployed GEMM-core placement (core 0,2, full12r7g) | B fifo 0x7C800 / 0x7DA00 (bank 3); A ring 0x77A00 (bank 1), 0x7EC00 and 0x7F080 (bank 3); C object 0x78000-0x7C7FF (bank 2, its last 2 KB in bank 3); `acc_hi` 0x71900 (banks 0-1); IRON tables 0x77E80 / 0x7F500. The mmb table predicts +44.6 cycles per asm call for this placement; sim10 fits +44.0 | [DOC] [INF] |
| when an access touches memory | `vlda` (accumulator quarters), `vst`, `vlda/vldb.pop.576` and `.fill.512` all access data memory at cycle 5 of their itinerary (`MemoryCycles<[5]>` in AIE2PGenSchedule.td), so ops in the SAME bundle hit memory in the same cycle: a same-bank pair in one bundle is the conflict; one bundle apart is not | [DOC TD] |
| lock instructions | `acq #id, rN` / `rel #id, rN` (id = 48 + n for the tile's own lock n; rN = -1 acquires >= 1 and decrements); conditional and pointer forms exist (`aie2p_locks.h`: acquire_cond / release_cond) | [DOC Peano] |
| lock unit | ACQ / REL hold the LCKREQ unit 4 cycles (one lock request per 4 cycles per core) | [DOC AIE2PGenSchedule.td] |
| lock cost when free | ~1 cycle beyond issue: 15 dynamic cycles per k-step over 20 lock operations | [MEAS r7p stamps] |
| call ABI | `jl` + 5 delay slots, `ret lr` + 5 delay slots, lr spilled when the callee calls; a load result is usable 7 cycles later (the wrapper pads with nops). A C++ wrapper around an asm kernel costs 34 bundles per call (mm_acc in full12r7g) | [DOC objdump] |
| IRON control code per GEMM k-step | 89 bundles for 9 calls + the B acquire / release + counters (identical in r7a/f/g/p) | [DOC objdump] |

## 3. Streams and ports

| item | value | tag |
|---|---|---|
| stream switch width | AM020 (AIE-ML): 32-bit. AM027 (aie2ps): 64-bit. aie2p: not documented on disk | [DOC] |
| **memtile -> core (A broadcast to 3 cores)** | **>= 13.8 GB/s = 7.6 B/cycle per core.** The no-MAC twin moves one tile-k-step (10,368 B of A per core) in 0.753 us (sim9 segment fit of `tl_full12p1n`). That is 96% of a 64-bit stream at 1.8 GHz, and above any 32-bit stream (7.2 GB/s). So aie2p's core-side streams are 64-bit, or the A path is not a single 32-bit stream | [MEAS P1 + INF] |
| per shim stream (DDR -> array) | ~6.4 GB/s = **3.6 B/cycle** at 1.8 GHz | [MEAS moondream p2/bw.py] |
| EW core input (packet ring from the shim) | 2-3 x 221 KB per column in 0.12-0.15 ms = 3.7-5.6 GB/s: bound by the **shim channel** (6.4 GB/s) and the ring protocol, not by a core port | [MEAS round-6 fit] [INF] |
| per-core input ports | 2 (A and B in the GEMM) | [DOC TM] [MEAS] |
| packet switching | packet id 5 bit, 4 packet-rule slots per slave port; PacketFlows can share one shim MM2S channel | [DOC TM] [MEAS pkt_test] |

Take-away: inside the array, data movement is fast (>= 7.6 B/cycle per core port). The slow links are the shim
channels (~6.4 GB/s each) and, in total, DDR.

## 4. Memtile (row 1)

| item | value | tag |
|---|---|---|
| size | 512 KB; TM uses 8 banks for allocation | [DOC] |
| channels | 6 MM2S + 6 S2MM; channels 0-3 reach the east/west neighbour memtile | [DOC TM] |
| BDs | 48: even channels BDs 0-23, odd 24-47; a static chain gets ~22 | [DOC TM] [MEAS] |
| BD fields | length 17 bit (words), 4D addressing, zero padding on MM2S, iteration wrap 6 bit, next-BD 6 bit, lock ids 8 bit | [DOC PARAMS] |
| design8 use per column | A slot 248,832 B (144 rows x K 1536 bfp16) + B forward + C join | [MEAS design8] |

## 5. Shim (row 0) and DDR

| item | value | tag |
|---|---|---|
| channels | 2 MM2S + 2 S2MM per shim, 16 BDs | [DOC TM] |
| BD fields | length 32 bit; D0/D1 wrap 10 bit; steps 20 bit (words); iteration wrap 6 bit; packet id 5 bit | [DOC PARAMS] [MEAS] |
| step > 2^20 words | moves the right bytes but serialises the columns (8x slower) | [MEAS round 4] |
| task queue | 4 tasks per channel; repeat <= 255; the sequence `maskpoll`s before pushing | [DOC TM] [MEAS full12k.bin] |
| DDR | ~55-60 GB/s read, ~63 GB/s read+write. The step moves 1.67 GB (round 5): ~22 GB/s at 76 ms | [MEAS] |

## 6. The command processor (the runtime sequence)

Unchanged from revision 1 (decoded `full12k.bin`: 7,716 BD writes + pushes, 6,960 maskpolls, 5,228 token waits in 289
sync points per step; `tools/txnstats.py`, `tools/txnq.py`). The fitted dependent-load latency is now 34 us per
dependent GEMM (round 6), still ~ one 248 KB A-slot load through one shim channel.

## 7. Bandwidth and latency per level (summary)

| path | rate | latency / fixed cost | tag |
|---|---|---|---|
| core register <- L1 | 2 load slots x 64 B (+ pops) per cycle | load +7, pop +8 cycles | [DOC TD] [MEAS] |
| L1 <- memtile (stream) | >= 7.6 B/cycle = 13.8 GB/s per core | | [MEAS P1] |
| memtile <- DDR (one shim channel) | ~6.4 GB/s | ~34 us for the 248 KB A slot | [MEAS] |
| DDR aggregate | ~55-60 GB/s read | | [MEAS] |
| host -> NPU dispatch | | 97 us launch, 12-27 us per bo sync | [MEAS] |

## 8. Ops in flight

| resource | limit | tag |
|---|---|---|
| DMA tasks queued per channel | 4 (+1 running); repeat <= 255 | [DOC TM] |
| BDs | 16 per shim / core tile; 48 per memtile | [DOC TM] |
| shim runtime BDs live at once | 16 per shim | [MEAS round 2] |
| lock value | 0-63 | [DOC TM] |
| block-vector streams per core | 2 FIFOs (lf0, lf1), each poppable from lda or ldb | [DOC TD] [MEAS] |
| forwarded (broadcast) L1 fifo depth | at most 3 free slots | [MEAS] |

## 9. Open questions

1. ~~A port or compute?~~ **Settled by P1: compute (and per-call overhead).** See ANALYSIS §0.
2. ~~The clock?~~ **1.81 GHz measured in Turbo** (round 7). Default mode is still unmeasured (needs sudo to switch).
3. Is aie2p's stream 64-bit? The no-MAC twin's whole-step delta (-16.1 ms over 15,552 k-steps) still implies >= ~7 B/cycle
   on the A path. Caveat: round 6's "1.04 us per k-step" for down_a divided a span that covers 3 of its 4 tiles by all
   96 k-steps. A one-column port microbenchmark (P10) gives the exact number; it matters again once the 2x2 GEMM
   brings the k-step to ~0.77 us (7.5 B/cycle needed).
4. ~~Where do the 12.6% of GEMM k-step cycles go?~~ **Settled by the round-7/8 stamps: nowhere.** The C++ k-step runs
   at its static schedule; the 12.6% was op / tile time the fit spread over k-steps (ANALYSIS rev 3 section 0). The
   asm call does stall in place: 44 cycles per call of L1 bank conflicts (section 2.2).
5. What does a `maskpoll` stall cost the schedule? -> P3, P4.
6. Do the core's L1 accesses or the DMA writes win a bank conflict? The mmb sweep ran without DMA traffic and the
   in-situ 44 matches it, so DMA writes add little at today's rates -> P21's in-situ check.
7. How long is a GEMM tile boundary's C handoff (release C -> the next acquire)? sim10 fits 7.6 us per tile, of which
   ~1 us is accounted for -> P24.

Sources (web): [AM020 memory tile](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Tile-Memory),
[AM020 stream interconnect](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AXI4-Stream-Interconnect),
[AM027 comparison of generations](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/Comparison-of-AIE-Generations),
[AM027 tile architecture](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Tile-Architecture).
