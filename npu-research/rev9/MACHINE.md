# MACHINE.md — the XDNA2 / AIE2P NPU in testbox, as a performance model

Revision 9 additions at the end (2026-10-10 evening, after round 13; rev 8 in rev8/). Revision 8 additions before them (after round 12; rev 7 in rev7/). Revision 7 additions before them (after round 11): register timing, itinerary ports, the slot model reconciled. Revision 6 additions before them; rev 6 is in `rev6/`. Revision 5, 2026-10-10: section 3 corrects the round-9 reading of the A path. The k-step is not A-port-bound: its
steady state is 2,055 + 67 cycles, and the ~4 B/cycle figures were a mean that includes the op-start wait, plus the
DDR-bound slot load. Section 2.4 (perf counters and DMA events for P38) and section 2.5 (the L1 maps of the GEMM and
EW cores) are new. Revision 4 is in `rev4/`.

Revision 4, 2026-10-09 night: section 2.2 now states the L1 bank rule exactly (fitted to every round-8 mmb placement
with zero error) and what it implies for scheduling; section 2.3 (vector slot classes) is new; open question 6 is
partly answered. Revision 3 is in `rev3/`, revision 2 in `rev2/`. Revision 3 measured the clock at 1.81 GHz and added
the first bank table. Revision 1 (round 5) had two wrong constants, the clock (1.43 GHz) and the memtile -> core stream
rate (4 B/cycle); ANALYSIS rev 2 section 0 explains how they misled the analysis.

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
| accumulator timing | a `vmac` reads its accumulator at stage 3 (the .td says 4): dependent macs at distance 2 or >= 3. **Rev 7: same fact, stated as the .td stage 4 under the LLVM visibility rule (Revision 7 section)** | [MEAS p11, r11] |
| other latencies | pop +8, load +7, conversion +2, vsrs +4, vups +3; pointer moves 2-3 cycles before use. **Rev 7: these are .td latencies w; a reader at stage r sees the value w - r + 1 cycles after issue** | [MEAS/DOC p10, r11] |
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

### 2.2 L1 banks, locks and the call ABI (revision 4)

| item | value | tag |
|---|---|---|
| L1 banks | 4 x 16 KB: 0x70000, 0x74000, 0x78000, 0x7C000 (the linker script's bank0-3) | [DOC ld script] |
| **the bank rule (exact)** | **A bundle costs max(1, max over banks of the number of its memory accesses in that bank) cycles.** n accesses to one bank issued in one bundle serialise into n cycles (n - 1 stall cycles). Accesses in different bundles never conflict: there is no adjacent-bundle penalty. There is no read / write asymmetry: load + load, load + store and pop + store each cost exactly 1. Ops on the same slot never co-issue, so A pops and accumulator loads (both lda) are free together. The offset inside a bank does not matter (16 KB granularity; offsets 0-10 KB apart tested) | [MEAS: `tools/bankmodel.py` reproduces all 207 placements of the round-8 mmb sweep (`npu-tapnext/build/mmb_run.out`) to 0.0 cycles with one constant, 13 = loop + call] |
| why each mmb pair cost what it did | co-issued bundles in `mm_core_split` (225 bundles): B-Cs 42, B-Cl 36, Cl-Cs 33, A-B 14, A-Cs 3, A-Cl 0; triples B+Cl+Cs 27, A+B+Cs 3. Each pair costs exactly its co-issue count, and a triple costs 2 (B+Cl+Cs in one bank: 33 + 36 + 42 - 27 = 84, measured 84) | [DOC count (`bankmodel.py --pairs`) = MEAS] |
| deployed placement (full12r8c core 0,2) | B fifo 0xD100 / 0xE300 (bank 3); A ring 0x7B80 / 0x8000 / 0x8480 (banks 1, 2, 2; contiguous, stride 1152); C object 0x8900 (its 12 KB of fp32 rows = stripes 0-2 in bank 2); `acc_hi` 0x1900-0x78FF (stripes 3-5 bank 0, stripe 5 straddles, 6-8 bank 1); offsets from 0x70000. Rule: 33 cycles per call (C loaded and stored in place), +3 on stripes whose A slot shares C's bank: **34 per call** | [DOC input_with_addresses.mlir, llvm-nm] [INF rule] |
| DMA writes | in situ the call runs ~12 cycles above the rule (r8c: ~283 vs 271 + 0), ~17 with the round-7 placement (A slots in B's bank 3). DMA writes into a bank the core uses in that cycle cost the core cycles; whether the DMA or the core yields is not measured | [MEAS g11 stamps - mmb] [INF] |
| memory cycle | `vlda`, `vst`, `vlda/vldb.pop.576`, `.fill.512` access data memory at cycle 5 of their itinerary (`MemoryCycles<[5]>`): consistent with the rule (same bundle = same cycle) | [DOC TD] [MEAS] |
| **what the rule means for scheduling** | A conflict costs exactly the serialisation a scheduler would add by moving one access to another bundle. Rescheduling wins only where the moved access finds a free slot of its unit inside its dependency window. At a 4-chain group boundary (16 accumulator loads + 16 stores + ~13 B ops pinned into ~16 cycles, lda and st already full) it cannot: P21's rule R1 gave 273 bundles where the stalls cost 271. **Fix conflicts by layout (different banks), not by scheduling rules** | [MEAS P21] |
| LUT loads | `vldb.4x16/4x32/4x64` (the 4-way table loads) issue on ldb only, latency 7. Each table copy should sit in its own bank, away from the banks the same bundles load or store | [DOC TD] [INF rule] |
| lock instructions | `acq #id, rN` / `rel #id, rN` (id = 48 + n for the tile's own lock n; rN = -1 acquires >= 1 and decrements); conditional forms `acq.cond` / `rel.cond` (condition in r26) and pointer forms exist (`aie2p_locks.h`) | [DOC Peano, TD] |
| lock unit | ACQ / REL hold the LCKREQ unit 4 cycles (one lock request per 4 cycles per core) | [DOC AIE2PGenSchedule.td] |
| lock cost when free | ~1 cycle beyond issue: 15 dynamic cycles per k-step over 20 lock operations | [MEAS r7p stamps] |
| call ABI | `jl` + 5 delay slots, `ret lr` + 5 delay slots, lr spilled when the callee calls; a load result is usable 7 cycles later. A C++ wrapper around an asm kernel costs 34 bundles per call (mm_acc) | [DOC objdump] |
| IRON control code per GEMM k-step | 89 bundles: 9 x 8 per call (A acquire in a delay slot, slot pointer, release, `jl`) + 17 (B acquire / release, counters) | [DOC objdump] |
| in-situ GEMM k-step (r8c, non-PROF) | **~2,900 cycles = 1.60 us**: static 2,420 (9 x 259 + 89) + bank 306 + DMA ~108 + ~65 fitted (sim11). The g11 -DPROF stamps read 3,352 (stamps add ~16-40 cycles per call) | [INF sim11 from MEAS stamps + step time] |

### 2.3 Vector slots (revision 4): where an elementwise chain's ops go

| slot | ops (AIE2P TableGen classes) | result latency |
|---|---|---|
| vec | every fp32-accumulator op: `vmac.f`, `vmul.f`, `vmsc.f`, `vnegmul.f`, `vadd.f` / `vsub.f` (acc + acc), `vneg` | 6 |
| st | **every conversion** (`vconv.bf16.fp32` 2, `vsrs` 4, `vfloor.s32.bf16` 2, `vconv.bfp16ebs8.fp32` 4) **and every store** (`vst`, `vst.conv.bf16.fp32`, `vst.srs`) | |
| mv | integer / bit / shuffle vector ops: `vadd.16/32`, `vsub.*`, `vband`, `vbor`, `vmax_lt.*` / `vmin_ge.*` (incl. bf16), `vsel`, `vshuffle`, `vups` (3), `vconv.fp32.bf16` (bf16 -> fp32, 2), `vbcst`, `vextract` | 2 |
| lda / ldb | loads; table (LUT) loads on ldb only | 7 |

So an elementwise kernel's floor per 32 lanes is max(MACs, conversions + stores, int / shuffle ops, loads / 2), not
its op count. Every `NK_BF(...)` is an st-slot op: a bf16 Horner step costs one vec and one st. Peano's 2-chain loops
run at 3-7x this floor (sigmoid 77 bundles per vector for ~22 MACs). [DOC TD, `aie2p_lat.json`] [MEAS loop counts]

### 2.4 Counters and DMA events for in-situ stall accounting (rev 5, for P38)

| item | value | tag |
|---|---|---|
| core-module counters | 4: `Performance_Control0/1` 0x31500 / 0x31504 (start / stop event, 7 bits each per counter), `Control2` 0x31508 (reset), `Counter0-3` 0x31520-0x3152C | [DOC PARAMS] |
| memory-module counters | 2: `Performance_Control0` 0x11000, `Control1` 0x11008, `Counter0/1` 0x11020 / 0x11024 | [DOC PARAMS] |
| core stall events | MEMORY_STALL 23, STREAM_STALL 24, LOCK_STALL 26, ACTIVE 28 | [DOC EVENTS] |
| memory-module DMA events | S2MM_0/1 STALLED_LOCK 31 / 32, STREAM_STARVATION 35 / 36, MEMORY_BACKPRESSURE 39 / 40; MM2S_0/1 STALLED_LOCK 33 / 34, STREAM_BACKPRESSURE 37 / 38, MEMORY_STARVATION 41 / 42; S2MM / MM2S FINISHED_BD 23-26 | [DOC EVENTS `XAIE2P_EVENTS_MEM_*`] |
| counting an event's cycles | start event = stop event = the event (the AIE convention) [INF]: P38 self-checks it with a TRUE counter against the timer | |
| data memory width | 256 bit (`MEMORY_MODULE_DATAMEMORY_DATA_WIDTH`); core load/store bus 256 bit, 512-bit vectors need 64 B alignment | [DOC PARAMS, TM] |

### 2.5 L1 maps of the deployed cores (full12r9s, rev 5)

| core | layout | free |
|---|---|---|
| GEMM (rows 2-4) | stack 0xB00; data 0x70B00 + 28,416 B (acc_hi 24 KB, tables, records); A ring 3 x 1,152 (banks 1, 2, 2); C object 18 KB (bank 2); B fifo 2 x 4,608 (bank 3); .aie.bank3 2 KB (GELU cd tables) | **~0.9 KB** (data 100 B, bank 1 192 B, bank 3 640 B). Text 15,840 / 16,384 |
| EW (row 5) | stack 6,144 (max call path 4,224: norm_rows<LN> 2,944 + ln_stats 1,024 + 256); ewo0 9 KB (bank 0); data: ssig_ab 2.5 KB, g_arena 22,016, g_panels 1 KB, g_ptmp 1 KB; .aie.bank0 ssqrt_ab 1 KB, .aie.bank2 ssig_cd 2.5 KB, .aie.bank3 ssqrt_cd 1 KB; ewi0 15 KB (bank 3) | **bank-2 region tail 2,560 B + data tail 960 B; 1.9 KB of stack never used**. Text 16,240 / 16,384 (144 B) |

## 3. Streams and ports

| item | value | tag |
|---|---|---|
| stream switch width | AM020 (AIE-ML): 32-bit. AM027 (aie2ps): 64-bit. aie2p: not documented on disk | [DOC] |
| **memtile -> core (A broadcast to 3 cores)** | **rev 5: >= 5.5 B/cycle hard, ~7.6 fitted (idle core); >= 5.03 in situ under mm_ks (its steady 2,055-cycle k-step needs it and gets it).** The round-9 "~4 B/cycle" was the mean k-step including the op-start wait (ANALYSIS rev 5 §1). Rev 2's reading follows: **>= 13.8 GB/s = 7.6 B/cycle per core.** The no-MAC twin moves one tile-k-step (10,368 B of A per core) in 0.753 us (sim9 segment fit of `tl_full12p1n`). That is 96% of a 64-bit stream at 1.8 GHz, and above any 32-bit stream (7.2 GB/s). So aie2p's core-side streams are 64-bit, or the A path is not a single 32-bit stream | [MEAS P1 + INF] |
| per shim stream (DDR -> array) | ~6.4 GB/s = **3.6 B/cycle** at 1.8 GHz | [MEAS moondream p2/bw.py] |
| A-slot load, 8 columns at once (rev 5) | 8 x 248,832 B in ~33 us from mm_ks entry (~36 us from the previous op's last send): **55-60 GB/s in total = the DDR read peak, ~7 GB/s = 4.1 B/cycle per column**. The op-start wait is DDR-bound | [MEAS g11-r9sp first-tile records, tools/profdec.py] |
| S2MM out-of-order mode | `DMA_S2MM_n_CTRL.Enable_Out_of_Order` (bit 3), `XAie_DmaSetOutofOrderBdId`: the packet header picks the BD, so one S2MM channel can land several packet streams in separate rings. FoT (finish-on-TLAST) mode and decompression also exist | [DOC aie-rt params / xaie_dma.h] |
| EW core input (packet ring from the shim) | 2-3 x 221 KB per column in 0.12-0.15 ms = 3.7-5.6 GB/s: bound by the **shim channel** (6.4 GB/s) and the ring protocol, not by a core port | [MEAS round-6 fit] [INF] |
| per-core input ports | 2 (A and B in the GEMM) | [DOC TM] [MEAS] |
| packet switching | packet id 5 bit, 4 packet-rule slots per slave port; PacketFlows can share one shim MM2S channel | [DOC TM] [MEAS pkt_test] |

Take-away: inside the array, data movement is fast (>= 5.5 B/cycle hard, ~7.6 fitted, per core port). The slow links are
the shim channels (~6.4 GB/s each) and, in total, DDR. Rev 5: the op-start A-slot loads hit the DDR peak (8 columns at
once).

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
3. (rev 5) Still not on disk. In situ the A path keeps >= 5.03 B/cycle while mm_ks runs; idle it gave >= 5.5 (hard). P34
   measures it with today's ring, P38 shows whether the core ever waits for it. Rev 4 text: Is aie2p's stream 64-bit? The no-MAC twin's whole-step delta (-16.1 ms over 15,552 k-steps) still implies >= ~7 B/cycle
   on the A path. Caveat: round 6's "1.04 us per k-step" for down_a divided a span that covers 3 of its 4 tiles by all
   96 k-steps. A one-column port microbenchmark (P10) gives the exact number; it matters again once the 2x2 GEMM
   brings the k-step to ~0.77 us (7.5 B/cycle needed).
4. ~~Where do the 12.6% of GEMM k-step cycles go?~~ **Settled by the round-7/8 stamps: nowhere.** The C++ k-step runs
   at its static schedule; the 12.6% was op / tile time the fit spread over k-steps (ANALYSIS rev 3 section 0). The
   asm call does stall in place: 44 cycles per call of L1 bank conflicts (section 2.2).
5. What does a `maskpoll` stall cost the schedule? -> P3, P4.
6. Do the core's L1 accesses or the DMA writes win a bank conflict? **Partly answered (rev 4):** in situ the call runs ~12 cycles
   above the bank rule (r8c), ~17 with A in B's bank, so DMA writes do cost the core; who yields is unmeasured (P32:
   mmb with the A / B DMA running). Rev 3 note: The mmb sweep ran without DMA traffic and the
   in-situ 44 matches it, so DMA writes add little at today's rates -> P21's in-situ check.
7. ~~How long is a GEMM tile boundary's C handoff~~ **Settled by P24: 1.31 us handoff + 0.49 acc_zero.** Rev 3 note: (release C -> the next acquire)? sim10 fits 7.6 us per tile, of which
   ~1 us is accounted for -> P24.

Sources (web): [AM020 memory tile](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Tile-Memory),
[AM020 stream interconnect](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AXI4-Stream-Interconnect),
[AM027 comparison of generations](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/Comparison-of-AIE-Generations),
[AM027 tile architecture](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Tile-Architecture).

## Revision 6 additions (2026-10-10, after round 10)

| fact | value | source |
|---|---|---|
| slot classes of the LUT / epilogue ops | ldb: vldb.4x64 / 4x32 / 4x16 lo / hi (latency 7); lda: vlda, vlda.ups; st: vfloor.s32.bf16, vconv.bf16.fp32, vst.conv, vst.push.544.conv (bfp, 64 lanes); mv: vmin_ge / vmax_lt / vadd.32 / vshuffle / vbcst / vbcstshfl / vsel / vconv.fp32.bf16 / vmov x (latency 2); vec: vmac.f / vmsc.f / vmul.f / vnegmul.f (acc read stage 3 [MEAS], result 6) | [DOC aie2p_lat.json] |
| GELU-by-table floor | 8 ldb per 32 lanes (4 vldb.4x64 per 16 lanes); mv 8 per 32 lanes with the index adds (6 with a biased table pointer) | [INF from the deployed asm build/r10/gx.s] |
| register files seen in Peano output | x0-x11 (512-bit, w halves), dm0-dm4 (1,024-bit; bm quarters bmll / bmlh / bmhl / bmhh) | [DOC asm] |
| GEMM core L1 during epilogues (full12r10r core 0,2) | sgelu_ab 0x70B00 / serf_ab 0x70F00 (bank 0), sgelu_cd 0x7F580 / serf_cd 0x7F980 (bank 3), acc_hi 0x71880-0x7787F (banks 0-1: pieces 0-9 in bank 0), C object 0x78900-0x7D0FF (bank 2, last 4,352 B in bank 3), aexp_ab 0x71300 | [DOC readelf] |
| GEMM text by function (r10r) | acc_epilogue 3,168, core loop 2,080, mm_ks 1,818, chunk_k 1,456, mm_kstep 880, attn_begin 800, attn_end 784, bias_pcs 752, chunk_v 608, rescale_rb 560, aexp_push 464, gelu_lut_n 448 B | [DOC readelf] |
| attention L1 | C object 18,368 of 18,432 B used (O tile 0, osum, srb, maxb, mbuf, ones, the exp table's second copy); acc_hi 24,576 of 24,576 | [DOC mm8.cc q_map / attn_begin] |
| runtime-sequence order (design8) | per GEMM op: finish_deps -> A fills -> memtile A loads -> B fills -> A sends -> drains; EW ops finish_all first. So B of a dependent op leaves DDR only after its producer finished | [DOC design8.py] |
| first-tile B starvation | g11-r10rp mlp3_a: 2,324 cycles per k-step outside mm_ks + 33 K cycles before the first k-step on every core's first tile (~49 us); A on time | [MEAS records, tools/profdist.py] |
| boundaries in the full step (r10r) | after an EW op +21 to +38 us, after a dependent GEMM / attention +20 / +34, independent next op -6 to -11 (it starts under the previous op's last drain), up_b (reuse) +6 | [MEAS timeline - g11 composition, tools/sim13.py] |
| EW norm input rates | norm1 (2 inputs) ~2.2 B/cycle, norm0 (3 inputs) ~3.4 B/cycle over their streaming spans | [INF timeline spans / bytes] |

## Revision 7 additions (2026-10-10, after round 11)

### R7.1 Register timing (measured by the main agent, `npu-tapnext/r11/mk_haz.py`; SKILL section 3)
| rule | value | tag |
|---|---|---|
| visibility | a register written by an op with .td latency w is seen by an operand read at .td stage r of an op issued **>= w - r + 1** cycles later (LLVM's latency). Before that, readers see the OLD value (no interlocks). Measured for vlda, vldb.4x64 (w 7), vadd.32, vconv, vfloor, vshuffle, vmin_ge (w 2) with a vst reader | [MEAS r11] |
| the old model | asmcheck's c + r >= c' + w is one cycle optimistic. Its "vmac reads its accumulator at stage 3" is the .td's stage 4 under the rule above: dependent macs >= 3 apart, or back-to-back at distance 2 through the one-deep accumulator bypass (round 7) | [MEAS] |
| mv -> mv bypass | a value from an mv-slot op (vmov, vshuffle, vadd.32, vmin / vmax) reaches mv-slot readers one cycle EARLY. So a read of the old value in the cycle before an mv write lands is unsafe for mv readers. Loads as writers: d = 7 for every reader | [MEAS] |
| WAW | an older load and a younger short write to one register land in landing order; nothing is dropped | [MEAS] |
| zero-overhead loop | the LAST bundle must be 16-B aligned (otherwise the body runs about once); writing every body bundle in the 16-byte full form makes this automatic | [MEAS] |

### R7.2 Itinerary ports: the structural hazards beside the slots
`AIE2PGenSchedule.td` itineraries (aie2p_lat.json `itin`; msched.ports) name shared ports. Each holds one op per cycle at the
listed stage, reserved modulo II exactly like a slot [DOC; the DM_ADA / QEY_RS sharing is MEAS r11]:

| port | held by (stage) |
|---|---|
| **DM_ADA** (load address A) | every `vlda` incl. `vlda.ups`, `vlda.pop.*` (1); every LUT load `vldb.4x16 / 4x32 / 4x64` (1). **Not** a plain `vldb` / `vldb.pop` |
| **QEY_RS** (x / y file read) | LUT loads (0: they read the index register), `vfloor.s32.bf16` from a w register (0), `vst` of an x register (0), `vst.push.576` (0), `vconv.bfp16ebs16.ebs8` (0) |
| QEY_WA | vlda x results (6), vconv bf16 (1), vfloor (1), vconv bfp (3), vlda.pop (7) |
| QEY_WB | vldb results incl. LUT loads (6) and pops (7) |
| CRRND_RS | vconv.bf16.fp32 (0), vst.conv (0), vconv.bfp (1), vst.push.*.conv (1) |
| L_RM / L_WM | vshuffle, vbcst, vsel, vbcstshfl (L_RM 0); vmin / vmax (L_WM 1); **vst.push.576(.conv) holds both at 0** |
| DM_WM | vlda into bm (6), vshuffle into bm (1), vconv.fp32.bf16 / ups (1), vmov cm (1) |
| DM_RM | vadd.f acc + acc (3), vmov cm (0) |

Useful escape hatches found in the json (not probed):
- `VFLOOR_s32_bf16_mv_float_to_int_bm` (source = a bm quarter, `$src[$imm]`) holds only QEY_WA. It takes the floor's operand
  read off QEY_RS.
- A plain `vldb` holds only QEY_WB, so in LUT loops data loads go on ldb.

### R7.3 The slot model reconciled (rev 4 section 2.3 + rev 6)
The rev-4 slot classes stand: one op per slot per bundle, conversions and stores on st, int / shuffle on mv, fp32 MACs on vec.
What was missing:
1. **ResMII = max over slots AND ports.** For every table-lookup line (GELU, the SSM sigmoid, the attention exp), QEY_RS
   (8 LUT index reads + 2 vfloor per 32 lanes) and DM_ADA (8 LUT + the vlda loads) bind at 10 per 32 lanes. Rev 6's "ldb 8,
   mv 8" floor was a slot count.
2. **RegMII binds next.** The x file is 12 registers (512 bit) minus the pinned constants. msched reaches II 11-13 on LUT
   loops whose ResMII is 10 [GEN `r12/g1sched.out`]. The dm file is 5 accumulators: the SSM recurrence (ResMII 10, vec) stops
   at 13 on dm MaxLive 5.
3. **Latencies are .td latencies + the visibility rule**, plus the mv bypass: a modulo schedule's dependence distance is
   w - r + 1, and an mv-written register stays reserved from c + w.
4. The bank rule (rev 4) still holds on top: it is a per-bundle memory cost, independent of the ports.

Floors per 32 lanes at these rules (ResMII / generator II):

| loop | ResMII | II |
|---|---|---|
| LUT line + product (GELU) | 10 | 11-13 |
| LUT line only (sigmoid) | 10 | 11-12 |
| attention exp per 64 | 20 | 23-31 |
| fused gated per 64 | 24 (DM_ADA) | no allocation: needs 2 passes |
| gated product + push per 64 | 6 | 6 |
| bf16 -> bfp push per 64 | 2 | 2 |

### R7.4 Text and stack (full12r11d, llvm-nm / llvm-size) [DOC]

| core | text used | free | largest functions |
|---|---|---|---|
| GEMM | 15,968 / 16,384 B | 416 B | acc_epilogue 3,024, core loop 2,080, mm_ks 1,818, chunk_k 1,456, mm_kstep 880, attn_begin 800, attn_end 784, bias_pcs 752, gelu_ms 718, chunk_v 608, rescale_rb 560, aexp_push 464, acc_zero 336, exp32 304 |
| EW | 16,240 / 16,384 B | 144 B | ew_ssm 2,752, norm_rows<LN> 2,624, norm_rows<RMS> 1,968, sqrt_lut_n 784, ew_patch 704 (unused by the deployed schedule), gemm8 672, ln_stats 672, ew_norm 528, exp_pD 512, exp_pC 496, gate_gemm 480, ssm_tables 480, exp_pB 464, rsqrt_i 432, sigmoid_lut_n 384, exp_pA 336 |

Stacks [MEAS stackaudit]: GEMM 1,664 of 2,816 B used, EW 4,288 of 6,144 B.

### R7.5 Host part (per frame, measured r11/hostprof.py)

| piece | ms |
|---|---|
| conv | 0.48-1.1 |
| heads | 0.6-1.0 (one thread 0.83) |
| queries | 0.12 |
| X0 packing | 0.14 (HOST_FAST=2) |
| read | 0.08 |
| total | **2.0-2.2** |

The variance comes from the OpenMP pool waking after the 52 ms NPU wait, and from tenants. Launch floor 97 us; frame upload 27 us.

## Revision 8 additions (2026-10-10, after round 12)

### R8.1 Dependent GEMM boundaries are the A slot load [MEAS timelines + INF]
out_proj (after attention) waits 17.6 us = 124 KB per column; mlp3_a (after mlp0) 37.5 us (boundary + g11 first-tile wait) =
248 KB per column, both at the measured slot rate (~7.3 GB/s per column with 8 loading). Same-shape ops that do not depend on
their predecessor (down_a, down_b, mlp3_b) show none. B fills are not on this path (ANALYSIS rev 8, 0.1).

### R8.2 Text (full12r12e, `tools/textmap.py`) [DOC]
GEMM 16,288 B: attention 6,048, epilogues 5,024, k-step 2,992 (mm_kstep 832 is a once-per-run record builder: one C object per
core), IRON control 2,238; 849 all-nop bundles (1,842 B). EW 16,224 B: SSM 7,600, norms 6,752, ring I/O 1,296, control 590.
No dead entry points. Each GEMM core holds a single C object in L1 (C0_x_buff_0), the cause of the 1.3 us tile handoff.

### R8.3 Instruction facts used by the rev-8 designs [DOC aie2p_lat.json]
- `vlt.bf16 rC, x, x` / `vge.bf16` write a 32-lane mask to a scalar register (mv slot, latency 2); `vmin_ge` / `vmax_lt` write
  theirs to r16. A lane test therefore costs no x register.
- bfp mmul: `VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX` (acc read at stage 4, result 6); pops `VLDA/VLDB_POP_576_normal_pop` (dst 8,
  FIFO state 7), and a `_2D` pop form with a post-modify exists.
- `VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf`: bf16 straight into an fp32 accumulator half (lda only).
- `vadd.f` (`VADD_f_vmac_cm2_add_reg`) reads both accumulators at stage 4; its config register selects the mode.

### R8.4 Overlays [DOC mlir-aie 1.4.4]
`aiex.npu.load_pdi` exists for the full-ELF flow only; the pyxrt dispatch bridge rejects it. `npu_blockwrite` can address any
tile register / memory in principle; writing core program memory at run time is untested (P59).

## Revision 9 additions (2026-10-10 evening, after round 13)

### R9.1 The K / V multicast (design8 attention, lowered MLIR of full12r13f) [DOC]
| item | value |
|---|---|
| producer | per core row r: shim r -> memtile r fifo `B<r>`: 8 buffers x 4,608 B (`--l2-depth 8`; the GEMM's B object type, the K / V payload is 3,456 B), S2MM BD ring with prod / cons locks |
| multicast | **one memtile MM2S channel**, one BD per buffer, routed to the 8 columns' `B<r>L1_<c>` (2 buffers each, prod lock init 2) |
| consequence | a send starts only when all 8 destinations have a free L1 slot: a column runs at most one object ahead of the slowest. The memtile depth decouples only the shim side [INF] |
| head groups | the A ring (Q objects to the column's 3 cores, 3 slots) and the single C object per core make the 24 cores start every head group together [INF from design8 `one_attn`] |
| work per core | pass 0: tiles 0-1 (12 row blocks), pass 1: tile 2 (6); 4 head groups (head r + 3g) x 24 chunks x (K, V); chunk 23 empty, chunk 22 2 of 6 key blocks |

### R9.2 Fitted in-situ costs (tools/attsim.py on the round-13 twins, r14/attcal.out) [INF, fit rms 0.02 ms]
| item | value | note |
|---|---|---|
| fused K loop per row block | **400 cycles** (bench 390; static II 35 x 8 = 280) | |
| aexp_push per row block | **400** (twin F8: 399) | |
| exact test + rescale | T + ~0.1 R determined only: T 66 (bench) -> R 300-450 | R carries +-100 % |
| rows8(-m) prep + call | 68 (bench) | |
| V phase per row block | 222 (r8 stamps; chunk_v 42 bundles per dim pair x 4) | |
| shim -> memtile fill | **~1,600 cycles per 4,608 B object = 2.9 B/cycle** | fitted from the no-K-compute twin only; not binding in production |
| memtile -> 8 L1 send | 580 cycles per object (7.95 B/cycle) | |
| lockstep cost, r13f | 0.80 ms per step | the model's columns-independent counterfactual |
| per-block check | model vs timeline attention walls: corr 0.97, slope 0.80 (thinned map) | r14/attblk_thin.out |

Not explained: l2-depth 4 -> 8 gained 0.30-0.36 ms (r13c -> r13e); the model has no depth effect (PROPOSALS Q1).

### R9.3 The lazy max on real data (r14/lazystat.out, CPU map) [INF]
Flagged = moved to within 0.055 %. After chunk 0: 2.26 % of row blocks, peaking at chunks 12-13 (4.4 %), clustered in columns
5-6 (4.2-4.9 % vs 0.25 % in column 0); blocks 10-11 ~0. Slow row blocks per K object: max over columns 5.5x the mean (iid 4.0x).
The row's self-score q_i . k_i is nearly always its max: seeded with it, 0.27 % flag after chunk 0 and 9.7 % in chunk 0.

### R9.4 Text (full12r13f, r14/textmap_r13f_*.out) [DOC]
GEMM 16,384 / 16,384 B: attention 6,864 (chunk_k_fused 1,360, attk 896, attn_begin 816, attn_end 784, chunk_v 608, rescale_rb 560,
aexp_push 464, attn_qcopy 400, attn_chunk_k 304, exp32 304, attn_chunk_v 256, rows8 112), main 2,080, acc_epilogue 2,000, mm_ks
1,824, bgelu 848 + bgelu3 304, bias_pcs 752, gup 432, acc_zero 336, mm_kstep 320, pan 304, cpy32 176; 867 all-nop bundles
(1,874 B). EW 16,224 B (unchanged since r12e).
