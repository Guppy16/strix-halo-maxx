# The machine: XDNA2 / AIE2P in testbox

The NPU in testbox's Ryzen AI Max+ 395 is an AIE2P array: 8 columns, each with a shim tile (DDR access), a memtile
(512 KB) and 4 compute tiles. A compute tile is a VLIW core with 64 KB of data memory in 4 banks, 16 KB of program
memory and 512 bfp16 MACs per instruction. The array runs at **1.81 GHz** (MEASURED). Most of what made kernels
fast or slow on this chip is in this chapter. Each fact carries its MACHINE R-id or a source path. The detailed
tables are in [MACHINE.md](../MACHINE.md), and the toolchain rules are in the
[xdna2-npu skill](../../.claude/skills/xdna2-npu/SKILL.md) §3.

AIE2P is not AIE-ML v2 (`aie2ps`), so don't use the AM027 tables for it. The device is PCI 1022:17f0 rev 0x11, which
the driver runs as `npu5` with the `npu4` tables (MACHINE §0).

## Array and clock

| item | value | source |
|---|---|---|
| geometry | row 0 shim, row 1 memtile, rows 2-5 cores: 32 cores, 8 memtiles | MACHINE §1 |
| clock | **1.81 GHz** in Turbo (pure-register loops; host slope and core timer agree); the driver's top DPM level is 1.8 GHz | MACHINE §1, `tapnext/npu/r7/clk.py` |
| peak | 32 x 512 MAC x 1.8 GHz = 29.5 TMAC/s = 59 TOPS (bfp16; dense int8 is the same rate) | MACHINE §1 |
| programs | one hw context per xclbin; switching costs ~2.7 ms; pyxrt launch floor ~97 us | MACHINE §1, SKILL §5 |
| power modes | Default and Turbo both run the array at DPM level 7 while a context is open; Turbo only turns off clock gating | MACHINE §1 (driver source) |

The early "1.43 GHz" clock was a stalled spin loop read as a clock, and it mispriced every early model
(negative-results [H1](negative-results.md#h1)). Measure the clock with `r7/clk.py` in the same session as the timing
you quote ([measurement.md](measurement.md#clock-and-cpu-load)).

## The compute tile

**Slots.** Each bundle issues at most one instruction per slot: `lda`, `ldb`, `st`, `mv`, `vec`, `alu`, `lng`.
Where an op goes decides an elementwise kernel's floor (MACHINE §2.3):

| slot | holds | result latency |
|---|---|---|
| vec | every fp32-accumulator op: `vmac.f`, `vmul.f`, `vadd.f` | 6 |
| st | **every conversion and every store** (`vconv`, `vfloor`, `vsrs`, `vst*`) | 2-4 |
| mv | integer, bit, shuffle, min/max, `vups`, `vconv.fp32.bf16` | 2 |
| lda / ldb | loads; LUT loads on ldb only; block-vector pops on either | 7 (pop 8) |

**Registers.** 5 accumulators (dm0-dm4, each one 8x8 fp32 block), 12 x registers of 512 bit (the 576-bit `ex`
bfp panels alias them: `ex_n` *is* `x_n`), and 6 `ey` pairs (MACHINE §2, R7.3). Register pressure, not slots,
sets the initiation interval of most table and epilogue loops (R7.3).

**No fp32 vector arithmetic.** fp32 add, mul, compare and convert on vectors are emulated at ~21 bundles per 16 lanes.
bf16 is native at 32 lanes. fp32 exists only inside the MAC accumulator (MACHINE §2,
`moondream/npu/RESULTS.md` phase 6). Everything in [elementwise.md](elementwise.md) follows from this.

**Rounding defaults to FLOOR.** Call `aie::set_rounding(conv_even)` once per core. Every moondream attention kernel
before v7r ran truncating (MD RESULTS phase 9: rms 1.49e-2 → 1.03e-2 once set).

**Register timing.** There are no interlocks. A register written by an op with .td latency *w* is seen by a reader at
stage *r* of an op issued ≥ *w − r* + 1 cycles later; before that the reader gets the OLD value. An mv-slot result reaches
mv-slot readers one cycle early (the mv → mv bypass). Dependent MACs need distance ≥ 3, or 2 through the one-deep
accumulator bypass (R7.1, measured by `tapnext/npu/r11/mk_haz.py`).

**Itinerary ports.** Some ports are shared across slots and are structural hazards, just like slots: DM_ADA (every `vlda` and
every LUT load), QEY_RS (LUT index reads, `vfloor`, `vst x`), and others (R7.2). For any table-lookup loop the
resource bound is max over slots *and* ports. GELU's "8-cycle floor" counted slots only; with ports it is 10-11
(negative-results [B4](negative-results.md#b4)).

## L1: 64 KB in four banks

The four banks are 16 KB each, at 0x70000 + 16 KB·i. The bank rule is exact (R4 §2.2, fitted to 207 placements with zero error):

> A bundle costs max(1, the largest number of its memory accesses that fall in one bank) cycles.

Accesses in different bundles never conflict, and ops on the same slot never co-issue (so two `lda` never collide).
In situ, DMA writes into a bank the core touches in that cycle also cost the core ~0.7 cycles per 32 B write
(α = 0.72; DMAB measured 0.68-0.75; R14.3, R15.1). A full GEMM k-step loses ~140 cycles per k-step to these collisions,
about 1.2 ms per TAPNext step, and no idle bank is left to move them into (R18.7,
[gemm.md](gemm.md#bank-conflicts-and-dma-collisions)).

L1 fills up. The deployed GEMM core used 64,960 of 65,536 B (R15.8), and the EW core ~2.5 KB free (R21.4). The
linker script's `bankN` regions are where `.aie.bankN` sections go, not free space. Free space is the region length minus
its sections ([H25](negative-results.md#h25)).

## Program memory: 16 KB

Text was the binding budget on both TAPNext core types for most of the rounds. The GEMM core sat at 16,256 / 16,384 B
at the pause (HANDOFF round 25). It killed or parked several levers ([G18](negative-results.md#g18)). What it means for
code is in [toolchain.md](toolchain.md#program-memory).

## Locks

Each core tile has 16 locks with 6-bit values (0-63), and a kernel reaches its own lock *n* as id 48 + *n*. A memtile has 64 (MACHINE §2).
`acq #id, rV` with rV < 0 acquires ≥ −rV; rV ≥ 0 is acquire-EQUAL, so "acquire 0" waits for zero and hangs (SKILL §3).
`acq.cond` / `rel.cond` take the condition in **r26 only** and test **bit 0** (R17.6, R19.1). A lock request
holds the lock unit for 4 cycles (MACHINE §2.2).

## DMA, streams and ports

| endpoint | channels | BDs | notes |
|---|---|---|---|
| core tile | 2 S2MM + 2 MM2S (a GEMM core uses 2 inputs: A and B) | 16 | BDs refuse stride 0 |
| memtile | 6 S2MM + 6 MM2S; channels 0-3 reach the neighbour memtile | 48, in two halves of 24 | 512 KB; queue 4 + 1 tasks per channel |
| shim | 2 MM2S + 2 S2MM | 16 | 20-bit steps, 10-bit wraps; ~5 tasks per channel before the CP blocks |

The rates and caps are in [data-movement.md](data-movement.md). In short: memtile → core 7.95 B/cycle per port,
one shim stream ~7 B/cycle (~12.6 GB/s), and an **aggregate** DDR cap of ~55-59 GB/s read and ~50-60 GB/s write, as observed
with the GPU idle.

A GEMM core has exactly two input streams, A (the memtile broadcast to the column's cores) and B (the weight multicast
to the row's cores). Neither can carry per-core data such as a residual tile (R18.5). That one fact ruled out
fusing the residual add into the GEMM epilogue in both projects ([J1](negative-results.md#j1),
`moondream/npu/RESULTS.md` phase 7).

## The command processor (CP)

The runtime sequence (`.bin`) is a stream of CP instructions: BD writes, address patches, queue pushes, maskpolls and
token waits. TAPNext's step is ~51k instructions in 289 sync runs (R15.5). Measured costs: `npu_write32` ~0.06 µs,
`npu_maskpoll` ~0.11 µs (CPRATE, [C3](negative-results.md#c3)), and a post-wait shim fill (5 instructions) ~0.55 µs
(R21.3). The CP issues column by column, so per-column work after a wait skews the columns, and a B multicast then
holds every column to the last one. [fusion.md](fusion.md) covers the consequences.

## Timers and counters

Each module has a 64-bit timer. The core reads it with `read_tm(0x340F8)` **only after** the runtime sequence enables
its processor bus (`npu_maskwrite32` 0x32038 bit 0). Without the enable the core hangs ([G17](negative-results.md#g17)).
There are four performance counters per core (0x31500 / 0x31504 control, 0x31520 + 4n counters; MEMORY_STALL 23,
LOCK_STALL 26). The CP can write a core's data memory, which gives CP-progress stamps (CPST). How to use these is in
[measurement.md](measurement.md#in-kernel-stamps-and-counters).

## What didn't work

The hardware facts above killed these ideas:

- Two block-vector pops from one FIFO in one bundle: a FIFO pops once per bundle, so B over two banks with one FIFO was void (E7'). [A14](negative-results.md#a14) <!-- nr: A14 -->
- Holding Q in `ex0-7` beside x constants: `ex_n` aliases `x_n`, so the loop computed garbage. [D9](negative-results.md#d9) <!-- nr: D9 -->
- A "biased base" form of `vldb.4x64`: it does not exist; LUT loads take an index vector only. [D10](negative-results.md#d10) <!-- nr: D10 -->
- Lock conditions as flag × counter: `acq.cond` tests bit 0, which skipped every even acquire (the KTL hang). [K1](negative-results.md#k1) <!-- nr: K1 -->
- "Scalar results need an extra cycle in a ZOL body": the .td latencies hold (mul 2, add 1). [K2](negative-results.md#k2) <!-- nr: K2 -->
- The residual add in the GEMM epilogue: there is no third core input port. [J1](negative-results.md#j1)
- The early 1.43 GHz clock and 4 B/cycle port constants. [H1](negative-results.md#h1)

## Lessons

- Read MACHINE's tag ([DOC] / [MEAS] / [INF]) before you build on a fact. An [INF] number has misled us more than once
  (the CP's 0.10 µs per BD write was really 0.55 µs, [C8](negative-results.md#c8)).
- Do the arithmetic in bf16 and keep fp32 in the accumulator. Price loops by max(slots, ports, registers), not by op count.
- When the core stalls, check banks first: one bundle with two accesses in one bank costs a cycle, every time.

## Going deeper

- [MACHINE.md](../MACHINE.md) §1-§8 and the R-sections (one per research pass).
- [SKILL](../../.claude/skills/xdna2-npu/SKILL.md) §3 (kernel rules), §5 (pyxrt), §6 (bfp16 codec).
- `common/tools/aie2p_isa_timing.json`: the slot, latency and itinerary of 879 instructions.
