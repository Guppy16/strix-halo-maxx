# MACHINE.md — the XDNA2 / AIE2P NPU in testbox, as a performance model

Research notes, written 2026-10-09 against the round-5 state of `npu-tapnext` (best build: full12k / full12m,
89.4 ms per depth-12 step, Turbo, GPU idle).

Every fact carries a tag:
- **[DOC]**: from a vendor source. Each one cites its file or page.
- **[MEAS]**: measured in our own work. Each one cites the result or file.
- **[INF]**: inferred: a model fitted to measurements, or arithmetic on documented facts. Treat it as a hypothesis
  until a proposal in PROPOSALS.md confirms it.

The sources on disk are the vendor headers inside the `npu-iron` image, read without a device. They are mlir-aie
1.4.4's `include/aie/Dialect/AIE/IR/AIETargetModel.h` (TM) and aie-rt's `xaie2pgbl_params.h` (PARAMS),
`xaie_events_aie2p.h` (EVENTS) and `xaie_lite_regdef_aie2p.h`. The tools are mlir-aie's `aie.utils.txn_trace`
(TXN, which decodes `.bin` runtime sequences) and llvm-aie's AIE2P schedule tables, parsed into
`npu-moondream/p10/aie2p_lat.json`. Web sources are AMD AM020 (AIE-ML) and AM027 (AIE-ML v2).

**AIE2P is not AIE-ML v2.** In mlir-aie, `aie2ps` is the AIE-ML v2 part (VE3858, Versal AI Edge Gen 2), with its own
target model. `aie2p` is Strix XDNA2. AM027 describes aie2ps. It matches our core on the MAC rate but not on the
stream width (see §3), so AM027 facts are marked [DOC, AM027 = aie2ps] and are not assumed to hold here.

## 1. Array

| item | value | tag |
|---|---|---|
| geometry | 8 columns. Row 0 = shim (NoC/DDR interface), row 1 = memtile, rows 2-5 = cores: 32 cores, 8 memtiles | [DOC TM `NPU2`] [MEAS design8 uses rows 2-4 for GEMM, row 5 for EW] |
| clock | ~1.4 GHz Default, 1.43 GHz Turbo (spin loop: 0.261 us per 364-bundle call) | [MEAS moondream phase 2c] |
| one program at a time | one hw context per xclbin; switching costs ~2.7 ms; launch floor 97 us median from pyxrt (58 us minimum on a 1-column probe; ~37 us of it is firmware without DMA) | [MEAS RESULTS 7, moondream 2c] |
| the "slow mode" | sub-ms dispatches sometimes finish on a ~1 ms timer for minutes (driver/firmware) | [MEAS moondream 2b] |

## 2. Compute tile (core + 64 KB memory module)

| item | value | tag |
|---|---|---|
| VLIW slots | one instruction per slot per bundle: lda, ldb, st, mv, vec, alu, lng (llvm-aie AIE2P `Slots.td`, parsed) | [DOC llvm-aie, p10/aie2p_lat.json] |
| MMUL rate | `mmul<8,8,8,bfp16ebs8>` = 512 MAC/instr, dense int8 the same. AM027 gives aie2ps 512 int8 / 256 bf16 / 512 fp8 MAC per cycle | [MEAS moondream] [DOC AM027 "Comparison of AIE Generations"] |
| achieved GEMM loop | 1.59 bundles per 512-MAC instr (63%) from Peano. Hand-scheduled 1.83 correct. The ldb-slot bound is 1.125 (B pops) | [MEAS moondream p11] |
| accumulator timing | a `vmac` reads its accumulator at stage 3 (the .td says 4): a dependent mac must be at distance 2 or >= 3 | [MEAS p11] |
| other latencies | pop +8, load +7, conversion +2, vsrs +4, vups +3; pointer moves 2-3 cycles before use | [MEAS/DOC p10] |
| no fp32 vectors | fp32 vector add/mul/convert are emulated (~21 bundles per 16 lanes). bf16 is native at 32 lanes; fp32 exists only in the accumulator | [MEAS] |
| rounding | defaults to FLOOR; `conv_even` must be set per core | [MEAS] |
| elementary unit | `aie::inv` is scalar (32 extracts per vector); `aie::exp2` is ~4-bit | [MEAS round 4] |
| program memory | 16 KB per core (`getProgramMemorySize` 0x4000) | [DOC TM] |
| data memory | 64 KB (`getLocalMemorySize` 0x10000). TM gives 4 banks for allocation; AM027 gives aie2ps 8 physical 256-bit banks paired into 4 x 512-bit | [DOC TM, AM027] |
| load / store bandwidth | AM027: aie2ps loads 128 B/cycle (2 load units) and stores 64 B/cycle. The 64 B alignment rule matches (`vector_ldst_align` 64 B for AIE2P in TM) | [DOC AM027 = aie2ps; TM] [INF for aie2p] |
| neighbour memory | a core addresses its own memory plus its S, W and N neighbours' (TM bases: S 0x40000, W 0x50000, N 0x60000, own = "E" 0x70000) | [DOC TM `getMem*BaseAddress`] |
| locks | 16 per core tile, 64 per memtile; values 6-bit (max 63); a kernel reaches its own locks at id + 48 | [DOC TM `getNumLocks`, `getMaxLockValue`] [MEAS] |
| core DMA | 2 S2MM + 2 MM2S channels, 16 BDs. BD fields: length 14 bit (words), D0/D1 wrap 8 bit, D0-D2 step 13 bit, iteration wrap 6 bit, next-BD 4 bit, lock ids 4 bit, lock values 7 bit (signed delta), zero compression | [DOC PARAMS `MEMORY_MODULE_DMA_BD*`] |
| timer | 64-bit tile timer, readable in a kernel via `aie::tile::current().cycles()` / `get_cycles()` | [DOC aie_api tile.hpp, AM027] |
| perf counters | 4 per core module, 4 per memtile; events include LOCK_STALL, STREAM_STALL, MEMORY_STALL, PORT_STALLED_0..7, INSTR_VECTOR, ACTIVE | [DOC PARAMS, EVENTS] |

## 3. Streams and ports: the limit that matters most today

| item | value | tag |
|---|---|---|
| stream switch width | AM020 (AIE-ML): 32-bit crossbar. AM027 (aie2ps): 64-bit, "two 32-bit words in parallel" | [DOC AM020, AM027] |
| **effective rate into one core (memtile -> core broadcast)** | **~3.9 B/cycle.** Fitted from 10 blocks of the round-5 timelines: one GEMM k-step (10,368 B of A per core) takes 1.857 us vs 1.813 us at exactly 4 B/cycle. See ANALYSIS §2 | [INF from MEAS, rms 7.9 us per op] |
| per shim stream (DDR -> array) | ~6.4 GB/s (~4.5 B/cycle at 1.43 GHz) | [MEAS moondream p2/bw.py] |
| EW core input port | port-bound at ~4 B/cycle (norm ops: 1-3 x 221 KB in per core in 0.12-0.15 ms) | [MEAS RESULTS 10, round-5 timeline] |
| per-core input ports | 2 (A and B in the GEMM). Each core reads only its own and N/S/W neighbours' memory | [DOC TM] [MEAS] |
| packet switching | packet id 5 bit (max 31), 4 packet-rule slots per slave port. Several PacketFlows may share one shim MM2S channel; an ObjectFifo pinned to that channel is refused | [DOC TM `getMaxPacketId`, `getNumSlaveSlots`] [MEAS pkt_test] |

Take-away: on this part, treat every DMA channel and stream endpoint as **4 B/cycle (5.7 GB/s)** until proposal
P10 shows otherwise. AM027's 64-bit figure describes aie2ps, and nothing we measured runs faster than ~4.5 B/cycle.

## 4. Memtile (row 1)

| item | value | tag |
|---|---|---|
| size | 512 KB (TM `getMemTileSize` 0x80000). AM020: 16 x 32 KB banks of 128 bit; TM uses 8 banks for allocation | [DOC] |
| channels | 6 MM2S + 6 S2MM. Channels 0-3 can reach the east/west neighbour memtile's memory (TM `getMaxChannelNumForAdjacentMemTile` 4) | [DOC TM, AM020/AM027] |
| BDs | 48: even channels use BDs 0-23, odd channels 24-47 (TM `isBdChannelAccessible`). A static chain in practice gets ~22 | [DOC TM] [MEAS] |
| BD fields | length 17 bit (words), 4D addressing (D0-D2 wrap 10 bit, D0-D3 step 17 bit), zero padding before/after per dim (6/5/4 bit) on MM2S, iteration wrap 6 bit, next-BD 6 bit, lock ids 8 bit (neighbours reachable) | [DOC PARAMS `MEM_TILE_MODULE_DMA_BD*`] |
| design8 use per column | A slot 248,832 B (144 rows x K 1536 bfp16) + B forward + C join | [MEAS design8] |

## 5. Shim (row 0) and DDR

| item | value | tag |
|---|---|---|
| channels | 2 MM2S + 2 S2MM per shim, 16 BDs | [DOC TM] |
| BD fields | length 32 bit (words); D0/D1 wrap 10 bit (so every TAP size <= 1023); **D0/D1/D2/iteration step 20 bit (words)**; iteration wrap 6 bit; burst-length 2 bit (all our BDs use encoding 3, the largest); packet id 5 bit | [DOC PARAMS `NOC_MODULE_DMA_BD*`] [MEAS TXN decode] |
| step > 2^20 words | still moves the right bytes, but the columns run one after another (8x slower) | [MEAS round 4, norm3] |
| task queue | **4 tasks per channel; repeat count <= 255.** A push to a full queue is dropped, not stalled, so the runtime sequence polls the status register (`maskpoll` on Task_Queue_Size bit 22) before pushing | [DOC TM `getDmaTaskQueueDepth`, comments on `getDmaStatusAddress`] [MEAS full12k.bin: 6,960 maskpolls] |
| DDR | ~55-60 GB/s aggregate read, ~63 GB/s read+write. Large GEMM frames ran at 37-39 GB/s. The GPU's DDR traffic adds +10-28% to DDR-heavy NPU phases; CPU memcpy hogs add +21% | [MEAS moondream, RESULTS 6] |

## 6. The command processor (the runtime sequence)

The `.bin` is a TXN stream that the NPU's firmware executes in order. Decoding `full12k.bin` with
`python -m aie.utils.txn_trace` (read-only, no device) gives these opcodes [MEAS]:

| opcode | count | meaning |
|---|---|---|
| blockwrite + patch | 7,716 each | write one BD (8 words) and patch in the host buffer address |
| write32 | 7,716 | push the BD onto a channel's task queue |
| maskpoll | 6,960 | wait until that channel's queue has room. **This blocks the whole sequence** |
| maskwrite + tct | 5,228 each | arm a task-completion token, then wait for it |

The stream is 1.35 MB and 337,712 words. The 5,228 token waits fall into 289 sync points (runs of waits) per step,
with 16-107 pushes between sync points.

A single in-order agent issues everything. A `maskpoll` on one channel stalls the pushes for every other channel
and column behind it (head-of-line blocking).

The per-sync cost of the firmware is not measured directly. The model in ANALYSIS §2 explains the latency before a
dependent op (~40 us) entirely by the A-slot load (248 KB at 6.4 GB/s = 39 us), so the firmware's own response is
at most a few us [INF].

Other facts:
- **Runtime BD semantics:** IRON's `TaskGroup.finish()` frees every BD of the group, including fills that were
  not awaited [MEAS round 3].
- **Shim BD budget:** at most 16 active runtime BDs per shim; design8 peaks at 14 [MEAS build log].

## 7. Bandwidth and latency per level (summary)

| path | rate | latency / fixed cost | tag |
|---|---|---|---|
| core register <- L1 | 2 x 64 B/cycle (aie2ps 128 B/cycle) | load +7 cycles | [DOC AM027] [MEAS td] |
| L1 <- memtile or shim (one channel) | ~4 B/cycle = 5.6-5.7 GB/s | a BD runs whole before its lock release: a 248 KB slot load = ~40 us before the first send | [INF fit] [MEAS] |
| memtile <- DDR (one shim channel) | ~6.4 GB/s | ~40 us for the 248 KB A slot | [MEAS] |
| DDR aggregate | ~55-60 GB/s read | | [MEAS] |
| host -> NPU dispatch | | 97 us launch, 12-27 us per bo sync; reading back a few hundred bytes takes < 1 us | [MEAS RESULTS 7] |

## 8. Ops in flight

| resource | limit | tag |
|---|---|---|
| DMA tasks queued per channel | 4 (+1 running); repeat <= 255 | [DOC TM] |
| BDs | 16 per shim / core tile; 48 per memtile (24 per channel parity) | [DOC TM] |
| shim runtime BDs live at once | 16 per shim | [MEAS round 2] |
| lock value | 0-63 | [DOC TM] |
| block-vector streams per core | 2 (two FIFO load units; a third spills) | [MEAS] |
| fifo depth on a forwarded (broadcast) L1 fifo | at most 3 free slots, whatever depth is asked | [MEAS] |

## 9. Open questions (each has a proposal in PROPOSALS.md)

1. Is the GEMM k-step bound by the A port (fit: 1.857 vs 1.813 us) or by compute plus per-call overhead
   (9 x (204 + ~115) cycles comes out the same)? -> P1.
2. Is the stream really 32-bit on aie2p, or is it 64-bit with a DMA-side limit? -> P10.
3. What does a `maskpoll` stall cost the schedule? -> P3 (static) and P4 (simulated).
4. How much of the attention (1.37 ms per block) and the SSM (1.30 ms per block) is compute, and how much is
   waiting? -> P2 (in-kernel cycle stamps).

Sources (web): [AM020 AIE-ML memory tile](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Tile-Memory),
[AM020 stream interconnect](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AXI4-Stream-Interconnect),
[AM027 memory tile](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Memory-Tile-Overview-and-Features),
[AM027 comparison of generations](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/Comparison-of-AIE-Generations),
[AM027 tile architecture](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Tile-Architecture).
