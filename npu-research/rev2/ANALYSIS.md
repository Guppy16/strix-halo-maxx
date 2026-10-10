# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 2, 2026-10-09. State analysed: round 6, `full12r6e` (76.3 ms per depth-12 step, Turbo, gpu_busy 0).
Revision 1 (round 5; kept in `rev1/`) concluded the GEMMs were bound by each core's A input port. Round 6's twin
builds (P1) refuted that. §0 says why the inference went wrong.

Inputs:
- the host-polled timelines `npu-tapnext/build/tl_full12{m,r6a,r6b,r6e,p1n,p1t}_d12.json`;
- the deployed GEMM kernel's static schedule (`npu-tapnext/r6/mm8o2.s`, `r6/asmstat.py`) and the linked GEMM core elf
  (`build/full12r6e.prj/elfs_main_core_0_2`, `llvm-objdump` in a read-only container);
- the driver, aie-rt, Peano and llvm-aie sources listed in MACHINE.md.

The model is `tools/sim9.py`, the successor of sim8. Tags [MEAS], [INF] and [DOC] mean what they mean in MACHINE.md.

## 0. Corrections, and what the method gets wrong

**What changed** (MACHINE §1, §3):

| constant | revision 1 | now | source |
|---|---|---|---|
| clock | 1.43 GHz | **1.8 GHz** | [DOC] driver DPM table, Default and Turbo; [MEAS] twin and GELU lower bounds 1.75-1.76 GHz |
| memtile -> core stream | 4 B/cycle | **>= 7.6 B/cycle** (13.8 GB/s) | [MEAS] no-MAC twin, sim9 segment fit |
| GEMM k-step | "on the A-port floor" | **compute**: 3,337 cycles fitted vs 2,963 static, data movement 0.753 us | [MEAS] P1, sim9 |
| GEMM inner loop | "at the ldb-slot limit, 27 ldb ops" | **18 ldb ops in 28 bundles**: bound by two accumulator chains, not by ldb (§3) | [DOC] the asm |

**Why the port-bound inference was wrong.** Revision 1 fitted t_ks = 1.857 us. It compared that with an "A-port floor"
of 1.813 us (10,368 B at 4 B/cycle and 1.43 GHz), saw a 2.4% match and called the k-step port-bound. Three faults:
1. **Both constants of the floor were assumptions, and both were wrong in the same direction.**
   - The real clock is 1.26x higher and the real port at least 1.9x wider.
   - So the real floor is ~0.72 us, 0.4x of the assumed one.
   - The match was a coincidence of two errors that compound.
   - The clock was on disk all along (the driver's DPM table). Nobody checked it, because "1.43 GHz" had been
     written down as a measurement.
2. **The competing hypothesis was costed loosely.** The compute side used moondream's 1.59 bundles per MAC (another
   kernel) and an invented 115 cycles of call overhead.
   - The deployed `mm_acc`'s own static schedule says 321 bundles per call: 2,963 cycles per k-step = 1.646 us at
     1.8 GHz.
   - That explains the fitted 1.857 us with 12.6% stall and nothing else.
   - Had the compute hypothesis been costed from the asm, it would have matched as well as the port did.
3. **A single-number match cannot separate two hypotheses that predict the same number.** Revision 1 did propose
   the right discriminating experiment (P1 twins), but wrote the conclusion as if the match had already decided it
   ("t_ks sits on the A-port floor", 32% of the step attributed to "A input port").

**Method rules from now on:**
- Every conversion constant (clock, port width, DDR rate) carries its own source tag, and [DOC] beats an old [MEAS]
  that measured something else (a spin loop with stalls).
- Each hypothesis is costed from the deployed artefact (the asm / elf), not from reference figures of other kernels.
- A fit that matches a bound is reported as "consistent with", and the bound's share is not booked until a
  perturbation experiment confirms it.

## 1. Where the 76.3 ms go (sim9 fit of full12r6e)

The fit runs over blocks 1-10. It uses spans between visible last writes, so ops whose last write the poll cannot see
(the SSM) are fitted jointly with their neighbours. Span rms 9.8 us; mean block 6.342 ms modelled vs 6.346 measured.

| part | per block (ms) | per step (ms) | share | bound [INF unless MEAS] |
|---|---|---|---|---|
| GEMM k-step loops (1,296 tile-k-steps per core x 1.854 us) | 2.40 | 28.8 | 38% | **compute**: static 1.646 us + 12.6% stalls [MEAS P1] |
| SSM (EW cores) | 1.30 | 15.6 | 20% | EW-core compute (ring I/O alone 0.28) |
| attention (GEMM cores) | 1.20 | 14.4 | 19% | compute: exp ~46%, PV ~27%, QK^T ~27% (main agent's static split) |
| per-tile fixed cost (92 tiles x 5.77 us) | 0.53 | 6.4 | 8% | epilogue pass + C drain + acc_zero |
| norms (118 / 121 / 147 / 147 us) | 0.53 | 6.4 | 8% | shim channel (~6.4 GB/s) + ring protocol, on the EW core |
| dependent A-slot loads (7 x 34 us) | 0.24 | 2.9 | 4% | one 248 KB BD through one shim channel |
| GELU tiles (36 x 3.6 us, LUT) | 0.13 | 1.6 | 2% | EW compute in the epilogue (was 16.5 us/tile) |
| final norm + tail | | 0.15 | | |

How the constants moved through round 6 (sim9 fits, us; all compute-side):

| build | t_ks | t_tile | t_gelu | L_dep | attention | SSM | span rms |
|---|---|---|---|---|---|---|---|
| full12m (r5) | 1.835 | 8.63 | 18.3 | 54.8 | 1376 | 1293 | 16.7 |
| full12r6a (-O2 epilogue) | 1.825 | 5.68 | 16.5 | 43.7 | 1355 | 1293 | 16.6 |
| full12r6b (LUT GELU) | 1.843 | 5.75 | 4.1 | 33.2 | 1356 | 1310 | 8.5 |
| full12r6e (attention -O2) | 1.854 | 5.77 | 3.6 | 34.1 | 1203 | 1303 | 9.8 |
| full12p1n (no MACs) | **0.753** | 7.00 | 15.6 | 35.7 | 1297 | 1359 | 6.1 |
| full12p1t (2x MACs) | **3.369** | 5.57 | 16.4 | 42.8 | 1409 | 1298 | 16.2 |

- The constants stay put where the code did not change: t_ks is 1.83-1.85 in every normal build, and t_tile fell
  only with the epilogue rewrite. That supports the law.
- **The fit error is 9.8 us rms per span** (8.5-16.7 across builds), and the block total is within 0.2% in every
  build. The worst spans before the LUT GELU were yx (-55 us); after it, attention (+-40 us).
- The twins fit as well as the base (6.1 / 16.2 us). t_ks moves exactly as compute-bound code must:
  - no MACs: down to the data-movement time;
  - 2x MACs: +1.544 us = 2,779 cycles, against 2,718 static bundles added.

**Utilisation.**
- The GEMM cores are busy ~54 ms of 76: GEMM ops 3.30 ms plus attention 1.20 ms per block.
- They idle during the norms and the SSM: 1.84 ms per block = 22 ms per step.
- The EW cores are busy ~22 ms of 76.
- DDR runs at ~22 GB/s average, ~40% of the read peak.

## 2. The GEMM cost law, refitted

```
T_op = L_dep [producer is the op right before it, or an EW op] + nt * (K/64 * t_ks + t_tile) + g * t_gelu
```

The constants are in §1's table. `python3 tools/sim9.py` reproduces every fit and the twin analysis;
`python3 tools/sim9.py gantt` prints an overlap schedule window (§4).

**Hold-out check (what sim9 cannot yet claim).**
- The depth sweep is linear in the model, so it is not an independent test.
- The twins are independent in one sense: the same law, unrefitted except t_ks, fits them at 6-16 us rms.
- What is still missing is a prediction made before a measurement (P4 gate (c)). The next one: the hand-asm GEMM
  loop's t_ks, predicted in §5 before it is built.

## 3. Anatomy of a GEMM k-step (static schedule of the deployed mm_acc, full12r6e)

**Per k-step on one GEMM core** (objdump of `core_0_2`):
- `acq` B object (lock 0x31);
- 9 x (`jl mm_acc`, `acq`/`rel` of the A ring, ~6 bundles of control in delay slots);
- `rel` B object.

**One mm_acc call** covers 16 rows x 64 columns x k 64 = 128 mmuls:

| piece | bundles | per call |
|---|---|---|
| entry (g_stripe, constants) | 13 | 13 |
| per row block (2): prologue = 8 A pops, accumulator loads, the first column-block pair (12 macs) | 48 | 96 |
| per row block: hardware loop, 3 iterations x 28 bundles (16 macs each) | 84 | 168 |
| per row block: tail (last 4 macs, 8 stores, branch delay slots) | 19 | 38 |
| return | 6 | 6 |
| **total** | | **321 bundles for 128 mmuls = 2.51 bundles per mmul** |

Per k-step: 9 x (321 + 6) + ~20 = **2,963 cycles = 1.646 us at 1.8 GHz**. The fitted t_ks is 3,337 cycles: **12.6%
of the k-step is stalls** that the static schedule does not show.

**The 28-bundle loop, op by op** (`.LBB1_2` in r6/mm8o2.s):

| slot | ops per iteration |
|---|---|
| ldb | 16 `vldb.pop.576` (B) + 2 `vldb.fill.512` = **18** |
| lda | 8 `vlda` (the next accumulator pair's quarters) |
| st | 8 `vst` (the previous pair) |
| mv | 8 `vmov` (loaded quarters into dm1 / dm4) |
| vec | 16 `vmac.f` |

So the round-6 note "28 bundles = exactly its ldb-slot count (27 pops/fills)" is wrong:
- The 8 A pops are hoisted out of the loop into the row-block prologue: A lives in ex0-ex10 for the whole row block.
- The loop holds 18 ldb ops in 28 bundles.

**What sets the 28.** Two accumulator chains, dm0 and dm4, of 8 dependent macs each:
- Peano issues the macs of one chain 3 bundles apart (the stage-3 read / stage-6 write rule).
- With only two chains, the steady pattern is "mac dm0, mac dm4, gap": 2 macs per 3 bundles, so 16 macs need 24
  bundles.
- Plus the chain hand-over at every column-block pair: the first mac reads the freshly loaded dm1, and the stores of
  the old dm0 must drain first.

**Two thirds of the call is not the loop.** 134 of 321 bundles are prologue, tail, entry and return:
- the loop runs only 3 times per row block;
- every row block refills the 8-deep pop pipeline;
- every row block drains the mac pipeline.

**Where the 12.6% stalls likely come from** [INF; P16's counters will tell]. Memory-bank conflicts in bank 3. From
`ldScripts_main_core_0_2.ld.script` (banks of 16 KB from 0x70000):
- In bank 3: both B fifo buffers (0x7C800, 0x7DA00), two of the three A ring slots (0x7EC00, 0x7F080) and the last
  2 KB of the C object.
- During the loop the core pops B from bank 3 (one 64 B access per bundle), while the DMA writes the next A and B
  objects into bank 3.
- The accumulators live in bank 2 (C object rows < 48) and banks 0-1 (`acc_hi`).

## 4. The new ceiling, and EW/GEMM overlap made concrete

### 4.1 Serial fixes

Per block after each fix (depth 12, `sim9.py`; each alone, then combined):

| change | step (ms) | delta |
|---|---|---|
| today (r6e fit) | 76.3 | |
| GEMM loop by hand, today's data flow (loop 28 -> 18, shorter prologue) | 66.6 | -9.6 |
| **GEMM loop by hand, 2x2 register blocking (§5)** | 62.2 | **-14.1** |
| SSM -> 0.5 ms per block | 66.6 | -9.6 |
| attention -> 0.6 ms per block | 69.0 | -7.2 |
| norms folded into the GEMMs (stats pass only, 40 us) | 71.8 | -4.5 |
| t_tile halved (drain overlap, acc_zero folded) | 73.1 | -3.2 |
| L_dep -> 5 us (P5 k-chunk A slot) | 73.8 | -2.4 |

All non-GEMM fixes together:
- with the 18-bundle loop: **39.6 ms**;
- with the 2x2 loop: **35.2 ms** (t_ks 0.947 us, 0.2 us above the measured data-movement floor of 0.753 us).

**So the ~44 ms serial ceiling of revision 1 was too pessimistic.**
- It assumed the GEMM loop could not beat the "port floor" (t_ks 1.28 us at best).
- With the port out of the way, the ceiling is **~35 ms at depth 12** (29 ms at depth 10, 23.5 ms at depth 8).
- Below that, the A port binds again at ~8 B/cycle: t_ks >= 0.72-0.75 us for 144 x 64 tiles.
- Going further needs wider N per core (less A per MAC) or fewer per-tile and per-op fixed costs.

### 4.2 Overlap within a frame: row chunks — not worth it

Two or three row chunks per column, the EW cores running chunk c's norm / SSM while the GEMM cores run chunk c' GEMMs.
sim9 list-schedules both resources (G = 24 GEMM cores, which also run attention; E = 8 EW cores). Attention is a join
over the chunks of its block.

| scenario | serial | 2 chunks (80 + 64 rows) | 3 chunks (48) |
|---|---|---|---|
| today | 76.3 | 76.9 | 77.2 |
| today, if per-tile and dependent-load costs were zero | 66.9 | 60.8 | |
| fixes + 18-bundle loop | 39.6 | 40.1 | 40.4 |
| fixes + 2x2 loop | 35.2 | 35.9 | 36.1 |

It loses for three reasons:
1. Every chunk repeats every N-tile's fixed cost (t_tile) and every dependent A-slot load: +0.77 ms per block today.
2. Attention is a per-block barrier that drains the pipeline.
3. The SSM (1.3 ms) faces only yx and out (~0.3 ms of GEMM work) in its stretch.

Even with free tile boundaries the gain is 6 ms of 67. After the serial fixes, EW work is ~8 ms per step, so there is
little left to hide.

### 4.3 Overlap across frames: two frames in flight (P14) — the fps lever

**Dependencies.**
- Frame t+1's block b needs frame t's block b only through the SSM state.
- It needs its own block b-1 for everything else.

**The concrete schedule: a half-frame phase shift.**
- Dispatch k runs frame k's blocks 6-11 interleaved with frame k+1's blocks 0-5, op by op.
- While the EW cores run one frame's norm or SSM, the GEMM cores run the other frame's GEMMs or attention.
- A window of the sim9 schedule, fixes + 2x2 loop, steady state (`sim9.py gantt`; f = frame, b = block):

  ```
  G: f2 b7 out | f3 b2 yx | f2 b7 up_a, up_b, down_a, down_b | f3 b2 out | f2 b7 qkv, attn, out_proj |
     f3 b2 up_a | f2 b7 mlp0, mlp3_a, mlp3_b | f3 b2 up_b | f2 b8 yx | f3 b2 down_a, down_b | ...
  E: f2 b7 ssm | f3 b2 norm0 | f2 b7 norm1 | f3 b2 ssm | f2 b7 norm2 | f3 b2 norm1 | f2 b7 norm3 | ...
  ```

**Predicted** (6 frames list-scheduled, at most 2 in flight; latency = first op to last op of a frame):

| scenario | depth | serial ms (fps) | 2 in flight: ms per frame (fps) | latency (ms) |
|---|---|---|---|---|
| today | 12 | 76.3 (13.1) | 58.2 (17.2) | 140 |
| today | 10 | 63.6 (15.7) | 48.5 (20.6) | 117 |
| fixes + 18-bundle loop | 12 | 39.6 (25.2) | 31.5 (31.7) | 76 |
| fixes + 18-bundle loop | 10 | 33.1 (30.2) | 26.3 (38.0) | 64 |
| fixes + 2x2 loop | 12 | 35.2 (28.4) | 27.8 (36.0) | 67 |
| fixes + 2x2 loop | 10 | 29.3 (34.1) | 23.2 (43.2) | 56 |
| fixes + 2x2 loop | 8 | 23.5 (42.5) | 18.6 (53.6) | 45 |

- Two frames in flight buy **1.25-1.35x fps for ~2x latency**. Today it would take depth 12 from 13 to 17 fps.
- The GEMM cores become the only bound: 12 x (GEMM ops + attention) per frame.
- It belongs on the fps side of the Pareto front; the latency side stays serial.

**Costs** [INF]:
- Two frames' activations in DDR (~+30 MB) and a runtime sequence of ~2x the ops.
- The SSM state double-buffered.
- One more frame of input latency: frame k+1 must be uploaded before dispatch k.

### 4.4 What else could overlap

- **Attention on 32 cores.** During attention (1.2 ms per block) the EW cores idle. Giving each column's EW core a
  quarter of the heads cuts attention by up to 25% (-3.6 ms today), if its 16 KB program memory has room (P9).
- **The SSM split over two cores per column** (P13 option 2): its 1.3 ms is the longest EW stretch and the main
  bubble in every overlap schedule above.

## 5. The GEMM inner loop: what the ISA allows (guidance for the asm route)

**Facts** (MACHINE §2):
- **Two FIFOs.** (p0, lf0, r24) and (p1, lf1, r25). Pops exist on **both** load slots: `vlda.pop.576` and
  `vldb.pop.576`.
  - A pop delivers one bfp16 block (576 bit) and refills 512 bits from memory.
  - One `fill` per 8 pops makes up the 72 / 64 B difference.
  - Peano puts every pop of mm_acc on ldb, alternating the two streams through lf0. Hand asm can pop A on lda (lf1)
    and B on ldb (lf0) in the same bundle.
- **Accumulators.** 5 (dm0-dm4). `vmac.f dmD, dmS, exA, exB, r` may read dmS and write dmD:
  - load the next accumulator into a staging dm;
  - let the first mac move it into place.

  No `vmov` needed (today: 8 per iteration).
- Accumulator loads are lda-only: there are no ldb forms for bm/dm. Stores are on st.
- **bfp16 mmul shapes.** 8x8x8 and 4x8x16 (the API's 8x8x16 = two 4x8x16 plus an A shuffle). Both are 512 MACs per
  instruction.
  - 4x8x16 needs B as 128 values, so it does not cut loads per MAC.
  - bf16 operands are converted to bfp16 per mmul in the vec slot (`to_v64bfp16ebs8`): strictly worse.
  - The `T` (transposed-B) form is the native one; no load difference.
  - The sparse forms (EX x QEY) double MACs per instruction only for 2:4-sparse weights: an accuracy experiment, not a
    scheduling one.

**Recommended loop: 2x2 register blocking, A and B both streamed.**
- Per pass: 2 row blocks x 2 column blocks.
- 4 accumulators D0-D3 in dm0-dm3; dm4 stages the next pass's loads.
- 8 k-substeps; per substep 4 macs, each A pop feeding 2 macs and each B pop feeding 2 macs:

  ```
  substep q (4 bundles, one mac each, chains round-robin at distance 4 >= 3):
    vmac D0 += A0q*B0q ; vlda.pop A1q (lf1) ; vldb.pop B1q (lf0) ; vlda bmXX  (acc quarter for the next pass)
    vmac D1 += A0q*B1q ; vlda.pop A0q+1     ; vldb.pop B0q+1     ; vst  bmXX  (previous pass's quarter)
    vmac D2 += A1q*B0q ; vlda bmXX          ;                    ; vst  bmXX
    vmac D3 += A1q*B1q ; (fill every 8 pops on each slot)
  ```

- **Per pass (32 mmuls):**

  | slot | ops |
  |---|---|
  | lda | 16 A pops + 2 fills + 16 accumulator quarter loads = **34** |
  | ldb | 16 B pops + 2 fills = 18 |
  | st | 16 |
  | vec | 32 |

  **34 bundles per 32 mmuls = 1.06** (today 2.51 per call, 1.75 inside the loop).
- **Registers.**
  - A and B blocks in flight: pops land 8 cycles after issue.
  - With the exposed pipeline, a destination can be re-targeted before its previous value is consumed, as long as
    the reader issues before the new pop lands.
  - So ~8 of the 12 ex registers suffice; Peano's version spilled because it holds A for the whole k (8 registers).
- **Pipelining.** One pass per column-block pair (4 per call). The next pass's accumulator loads and the previous
  pass's stores run inside the current pass, so there is one prologue and one drain per call (~24 bundles), not per
  row block.
- **Estimate.**
  - ~4 x 34 + 24 = 160 bundles per call, against 321.
  - Per k-step: 9 x (160 + 6) + 20 = 1,514 cycles; x 1.126 stalls = 0.947 us.
  - The data-movement floor (0.753 us measured) is not binding yet. **Predicted t_ks 0.95 us, -14 ms per step**
    (sim9), to be checked against the build (P4 gate c).
- **A cheaper first step (same data flow, Peano-shaped).** Keep A in registers, but run **three or four accumulator
  chains**: 3-4 column blocks per A pass.
  - That needs 3-4 dm plus a stage: possible only with the `vmac dmD, dmS` staging trick, so hand asm again.
  - The loop then sits at the ldb bound: 18 bundles per 16 macs. Estimated t_ks 1.23 us, -9.6 ms.
- **Further, after the loop.**
  - **One call per k-step:** take the A ring locks inside the kernel, as ew8 does. That removes 9 x (13 + 6 + 6)
    cycles of call overhead per k-step (~0.12 us, ~-2 ms).
  - **k = 128 per call:** two k-steps per call with both B objects held (the B fifo is already 2 deep), and the
    memtile sending A in [k-pair][stripe][2 k-steps] order. That halves the accumulator traffic, so lda drops to
    ~26 per 32 mmuls and the vec slot binds at 1.0.
  - **Bank placement (P17):** B fifo, A ring and accumulators in different banks, to remove the 12.6% stalls.
- **Checks.**
  - `p11/gen_gemm.py` + `asmcheck.py` with the stage-3 accumulator rule.
  - Assert which pop each mac reads.
  - Bit-exact against run8 g11 / t1 before any timing.
  - One dispatch with a 3 s timeout first.

## 6. Safe cycle timing on the core

Why the round-6 probe hung:
- `read_tm` reaches tile registers through the core's processor bus.
- Core_Processor_Bus (0x32038) bit 0 `Enable` resets to 0.
- mlir-aie's own test (`test/npu-xrt/tile_mapped_read/aie.mlir`) says: *"Set Core_Processor_Bus register Enable = 1.
  Without this the core will hang on access to the processor bus"*.
- The npu2 example `programming_examples/basic/dma_compression` (write_tm / read_tm of DMA registers, run on AIE2P)
  does exactly that from the IRON runtime sequence:

  ```python
  npu_maskwrite32(column=COL, row=row, address=0x32038, value=0x1, mask=0x1)
  ```

- It also documents a second hazard: tm accesses packed back to back into one bundle stall the core. They separate
  each access with `__builtin_aie2p_sched_barrier()`.

Methods, safest and most useful first:

**1. Core timer with the bus enabled (P2 retried as P15).**
- At the start of the runtime sequence, before any fill: `npu_maskwrite32(column=c, row=r, address=0x32038, value=1,
  mask=1)` for every stamping core.
- In the kernel:

  ```c++
  __builtin_aie2p_sched_barrier();
  uint32_t t = read_tm(0x340F8);  // core module Timer_Low; or 0x140F8, the memory module's timer
  __builtin_aie2p_sched_barrier();
  ```

- Check with `llvm-objdump -d` that each `lda.tm` sits alone, before dispatching.
- The first `read_tm` must come after the core's first lock acquire on data the sequence sends. The write then
  precedes it, because the sequence issues the maskwrite before the fill.
- Risk: the core may run before the sequence; the acquire ordering handles that.
- Test on the one-core kprobe first (3 s timeout), stamps around a known-bundle loop: that also measures the clock
  directly.

**2. Performance counters (the profiler; P16).** Four per core, configured from the runtime sequence with
`npu_write32`, read by the core at op end with `read_tm` (needs method 1's bus enable), stored in the P2 stamp slots:

| counter | start / stop | counts |
|---|---|---|
| cnt0 | INSTR_EVENT_0 / INSTR_EVENT_1 (33 / 34) | cycles inside mm_acc (the markers already exist) |
| cnt1 | LOCK_STALL / LOCK_STALL (26) | lock-stall cycles |
| cnt2 | STREAM_STALL (24) | stream-stall cycles |
| cnt3 | MEMORY_STALL (23) | memory-stall cycles |

- Registers: `Performance_Control0` 0x31500 = (cnt1 stop << 24 | cnt1 start << 16 | cnt0 stop << 8 | cnt0 start),
  `Control1` 0x31504 the same for cnt2/3, counters 0x31520-0x3152C.
- [INF] "start = stop = event" counts that event's cycles, per aie-rt's PerfCounterControlSet usage. Verify on one
  core against the timer.
- Zero cost inside the loop. One `read_tm` per counter per op.
- It answers §3's 12.6% (memory vs stream vs lock stalls) directly.

**3. Control-packet register reads from the host** (IRON 1.4.4 `aie.utils.trace.utils.create_ctrl_pkt`,
`setup.configure_packet_ctrl_flow` / `config_ctrl_pkts_aie`).
- No kernel change and no processor bus.
- Costs a shim MM2S + S2MM pair (packet-switched), one word per transfer, one target per flow.
- Good for a one-column probe or an end-of-run readout of the counters; not for per-op stamps in the full step.

**4. The trace unit** (works today, one-column builds).
- UPSTREAM main adds `aie-insert-trace-flows lateral-routing` (trace to a spare column's shim) and
  `kernel_design.cycles_per_call` (whole-call intervals from the event0 / event1 markers).
- In the 8-column step no column is spare, so per-op profiling of the real step should use 1 + 2.

**Not recommended:** event broadcast to a shim trace unit.
- The shim's two S2MM channels are taken in every column.
- Broadcast lines are shared with the lock and DMA signalling IRON may use.
- It adds routing risk to the full design for what methods 1-2 give per core.

## 7. Data-movement designs from revision 1: what still stands

| item | status |
|---|---|
| 5.1 k-chunk-streamed A slot (P5) | still valid. L_dep is 34 us x 7 per block = 2.9 ms per step, -144 MB DDR. Lower value than thought: -2.4 ms |
| 5.2 fold the norms (P7) | still valid: -4.5 ms, and removes 48 `finish_all` brackets. Also shrinks the EW stretch that two-frame pipelining must hide |
| 5.3 two A streams per column (P6) | **obsolete**: the port is not binding (>= 7.6 B/cycle vs 3.1 needed at today's t_ks). Revisit only once the GEMM loop reaches ~0.75 us per k-step |
| 5.4 overlap | quantified in §4: row chunks lose, two frames in flight win fps at a latency cost |
| §4 critique of activation residency | still stands, with a different reason: neither DDR (~22 GB/s of ~55) nor the memtile -> core port binds. Compute and fixed costs do |
