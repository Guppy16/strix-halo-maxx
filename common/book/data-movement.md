# Data movement and DDR

Inside the array, data moves fast: 7.95 B/cycle per memtile → core port. The slow parts are DDR and the DMA
bookkeeping. The NPU's DDR reads and writes each cap at **~55-59 GB/s in aggregate** (observed, GPU idle), and adding channels
does not raise that. Every TAPNext norm and the SSM sit at or near the read cap, and every dependent op boundary
was DDR bytes (a drain tail at the write cap, then the consumer's first tile at the read cap). So the levers are fewer DDR
bytes and keeping data on chip, not more channels. The second half of this chapter is DMA bookkeeping: BD budgets, task
queues, and where a fill must be issued so the command processor neither blocks nor deadlocks.

## Rates

| path | rate | source |
|---|---|---|
| core register ← L1 | 2 load slots × 64 B per cycle (+ pops) | MACHINE §7 |
| memtile → core (A broadcast) | **7.95 B/cycle** per core port | round 10 `r10/rtm.py` |
| memtile → 8 L1 send (K/V multicast) | 580 cycles per 4,608 B object (7.95 B/cycle) | R9.2 |
| one shim → core stream | ~7 B/cycle (~12.6 GB/s) | DDRCAP, R20.2 |
| one shim S2MM write stream | ~6.7 B/cycle (12.1 GB/s) | DDRW, round 20 |
| DDR read, 8+ streams | **~55-59 GB/s aggregate** | DDRCAP `tapnext/npu/r19/ddrcap.py` |
| DDR write, 8 columns | ~50-60 GB/s aggregate (2 columns 24.4, 4 columns 45.3) | DDRW `r20/ddrw.py` |
| reads + writes together | 63-74 GB/s | DDRCAP |
| core MM2S drain of a C object | 7.9 B/cycle | R15.7 |

## The DDR caps

- **The cap is aggregate.** 4 × 1 and 2 × 2 streams give the same rate, and 8 × 1 the same as 8 × 2 (DDRCAP). A second input channel per column
  for the norms ([C13](negative-results.md#c13) <!-- nr: C13 -->) or a second drain channel per column
  ([B10](negative-results.md#b10) <!-- nr: B10 -->, [C16](negative-results.md#c16) <!-- nr: C16 -->) cannot help.
- History: moondream measured "~55-60 GB/s peak" early (MD p2/bw.py). Research rev 15 derived "≥ 84 GB/s" from norm walls × bytes and called
  the old figure a per-channel artefact; DDRCAP restored it ([C12](negative-results.md#c12) <!-- nr: C12 -->). MACHINE R15.3 / R15.4 are
  stale on this point; R20.2 and R21.1 supersede them.
- A per-channel write limit (~4 B/cycle) and token latency were both proposed for the 12 µs drain tail. Both were refuted: one stream writes
  6.7 B/cycle, and the tail is the last 442 KB of C objects at the aggregate write cap ([C14](negative-results.md#c14) <!-- nr: C14 -->,
  [C15](negative-results.md#c15) <!-- nr: C15 -->).
- **Near the cap is not on the cap.** The SSM span read at 54 GB/s, ~95 % of the cap, yet it was compute-bound by ~255 cycles per chunk
  ([M1](negative-results.md#m1)). Measure with a removal twin and input-wait stamps ([measurement.md](measurement.md#twins-perturb-to-find-the-bound)).
- Price a bound over the binding op's own span, with the cap of the same kind ([L3](negative-results.md#l3)).
- GPU load shares this DDR. Every TAPNext number is gpu_busy 0. An LLM on the GPU will slow the read-capped ops (untested; HANDOFF risks).

**What the cap means for a design**: count DDR bytes per op span. TAPNext's step reads 914 MB and writes 676 MB
(`tapnext/research/r21/ddrbudget.py`, R16.5). Its EW ops read at 48-55 GB/s inside their spans, while GEMM steady tiles use only 13-24 GB/s.
The norms' floor is bytes: keeping the panels on chip (NRES) saved −0.65 ms, and a second channel saved nothing.

## Keeping data on chip

The memtile (512 KB per column) is where activations can stay between ops:

- **A slot in the memtile**, loaded once per op and sent to the column's 3 GEMM cores. In TAPNext it is 248,832 B (24 chunks of 10,368 B,
  one lock per chunk). Runtime-driven memtile tasks (`FlowEndpoint.task`, A_RT) let the first tile start after one chunk:
  −2.0 ms (round 10).
- **Producer → memtile directly**: an EW core streams its output panels into the next GEMM's A slot through its free MM2S (NRES, SRES; round 20).
- Capacity binds quickly: a GEMM → GEMM output park needed 124 KB against 112 KB free ([B11](negative-results.md#b11)). A third A region needed
  contiguous space the holes did not have until IRON re-packed the memtile ([C24](negative-results.md#c24)). The model chapters have the details.
- moondream: residency of A in the memtile is the reachable cut of the frame's DDR traffic (A read ~2x: 403 MB vs ~200 inherent).
  It was designed, not built (MD RESULTS phase 12).

## DMA bookkeeping

**BDs.** Shim and core tiles have 16 BDs, memtiles 48 in two halves of 24 per channel group (a static chain gets ~22).
- `BdIteration(size, stride)` lets one BD walk a ring of slots and keeps rings under the budget (SKILL §2).
- A shim BD lives until its TaskGroup finishes. Put each fill in the earliest drain group whose completion proves it was consumed. At
  TAPNext's out_proj, shim 0 held 16-20 live BDs until the attention's fills moved into pass 0's group (P0FREE: peak 20 → 16; R21.5).
- `aiecc --reclaim-runtime-bds` inserts completion maskpolls instead, and those can hang when completion depends on a later push.
- `TaskGroup.finish()` frees *every* BD in the group, waited or not. A group holding only `wait=False` fills must not finish before they
  complete (silent corruption or a hang).

**Field limits** ([machine.md](machine.md#dma-streams-and-ports)):
- Shim BD steps are 20 bits of words. A larger step moves the right bytes but serialises the columns (norm3: 6 ms instead of 0.8;
  [G11](negative-results.md#g11) <!-- nr: G11 -->). Keep the regions one TAP strides across within 4 MB.
- Shim TAP dims hold ≤ 1023 words each. A 4,608 B object needs a fold dimension or per-object tasks.
- Tile and memtile BDs refuse stride 0 ([G10](negative-results.md#g10)).
- A BD takes one acquire and one release, or neither. Release by 0 is the no-op form ([G19](negative-results.md#g19)).
- An EW input fill group must total a multiple of the 768 B ring slot. 1 KB padding left a slot unfilled and hung the step
  ([E12](negative-results.md#e12) <!-- nr: E12 -->).

**Queues.** Each memtile MM2S channel takes **5 tasks without blocking** (1 running + 4 queued). A shim channel behaves the same (~5,
inferred). A 6th push blocks the command processor until that channel's oldest task completes (R20.5; round 23):
- Never push a memtile MM2S task before the same op's fills exist. The blocked push waits on work the CP has not issued
  (SEQPRE v1 +1.1 ms; [C5](negative-results.md#c5) <!-- nr: C5 -->).
- When a column's tile 0 needs a B multicast, push every column's first five tasks before any column's sixth (r23d hung column by column).
- Keep drain groups ≤ 4-5 per column: 7 drain TAPs per column blocked the CP behind them (+0.44 ms; [L6](negative-results.md#l6)).

**Fills before or after waits.**
- Issue everything independent of a producer before its wait (NSEQ: a norm's params and drains before the wait, −0.17 with NRES).
  GEMM drains moved before a wait cost +0.15 instead, cause unexplained ([C7](negative-results.md#c7) <!-- nr: C7 -->).
- A fill that must wait can still be *configured* before the wait and pushed after (CPPRE, −0.09). A post-wait fill costs the CP ~0.55 µs per column,
  and 8 columns skew by ~4.4 µs (R21.3). See [fusion.md](fusion.md#the-command-processor-sequence).
- Lock-governed memtile sends wait in the channel FIFO, so push a split consumer's sends *before* its part waits (OPPRE2 v2 −0.15 vs v1 −0.04).

**Multicast.** A memtile `forward` to N cores is one MM2S channel. A send starts only when every destination has a free L1 slot, so
the destinations run in lockstep. A rare slow unit costs ~3x its own time on the wall ([attention.md](attention.md#lockstep-under-a-multicast)).

## Packet routing

Several PacketFlows can share one shim MM2S channel. That is how a column gets more inbound streams than its 2 shim channels
(GEMM A + EW input, round 2). The IRON 1.4.4 pathfinder behaviours and their patches are in [toolchain.md](toolchain.md#iron-and-mlir-aie).
A core tile's 4 south ports were full in TAPNext (C × 3 + EW output), so a fifth southbound flow must share a packet port.

## What didn't work

- [B10](negative-results.md#b10) <!-- nr: B10 --> C drains over two shim S2MM per column: the write cap is aggregate.
- [C12](negative-results.md#c12) <!-- nr: C12 --> "≥ 84 GB/s": the read cap is ~55-59 GB/s.
- [C13](negative-results.md#c13) <!-- nr: C13 --> a second norm input channel: the 3-input norms already sit at the aggregate cap.
- [C14](negative-results.md#c14) <!-- nr: C14 --> a per-channel write limit as the tail: one stream writes 6.7 B/cycle.
- [C15](negative-results.md#c15) <!-- nr: C15 --> token / CP latency as the tail: the tail is bytes at the write cap.
- [C16](negative-results.md#c16) <!-- nr: C16 --> a two-drain probe: moot once the 1/2/4/8-column series showed an aggregate cap.
- [C5](negative-results.md#c5) <!-- nr: C5 --> memtile sends before the wait: the 5-deep queue blocked the CP (+1.1 ms).
- [C7](negative-results.md#c7) <!-- nr: C7 --> GEMM drains before the wait: +0.15 ms, cause unexplained.
- [G11](negative-results.md#g11) <!-- nr: G11 --> a BD step above 20 bits: correct, but columns serialised.
- [E12](negative-results.md#e12) <!-- nr: E12 --> a 1 KB fill pad: not a slot multiple, hang.
- [L6](negative-results.md#l6) <!-- nr: L6 --> finer drain groups: the shim channel's task queue blocked.

## Lessons

- Count bytes per span against the aggregate cap before pricing a data-movement lever. Channels don't add bandwidth.
- Keep data in the memtile when the next op is on chip, and load the next op's A into the slot half the current op leaves free.
- A DMA push is a CP instruction that can block. Know each channel's queue depth and the order of every wait in the sequence.

## Going deeper

- [MACHINE](../MACHINE.md) §3-§5, R10.2, R15.2-R15.4 (stale on DDR), R16.5-R16.8, R20.2-R20.5, R21.1, R21.5.
- Probes: `tapnext/npu/r19/ddrcap.py`, `r20/ddrw.py`, `r10/rtm.py`; `tapnext/research/r21/ddrbudget.py`, `r26/design8_bd.py`.
- Model chapters: [TAPNext boundaries](../../tapnext/book/boundaries.md), [moondream NPU GEMM](../../moondream/book/npu-gemm.md).
