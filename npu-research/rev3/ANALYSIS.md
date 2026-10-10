# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 3, 2026-10-09 evening. State analysed: round 7's `full12r7g` (71.2 ms per depth-12 dispatch, Turbo, gpu_busy 0)
plus the round-8 in-situ stamps the main agent took while this was written (`g11-r8pa` / `g11-r8pc`, the bank
microbenchmark `mmb`, the timeline `tl_full12r7g_d12.json`). Revision 2 (round 6) is in `rev2/`, revision 1 in `rev1/`.

Tags as in MACHINE.md: [DOC] vendor source or the deployed binary, [MEAS] measured by us, [INF] model or arithmetic.
Cycles convert at **1.81 GHz** [MEAS r7/clk.py].

New inputs this revision:
- `r8/core02_r7{a,g,p}.dis`, `core02_g11-r8p{a,c}.dis`: `llvm-objdump -d -z` of the linked GEMM-core elfs (read-only
  container). Every static count below comes from these.
- `tools/sim10.py` (+ `sim10_r7.out`): the successor of sim9, built on the stamps (§2).
- `r8/prereg_gap.txt`: the gap prediction written BEFORE reading the round-8 stamps (§1 scores it).

## 0. Scorecard: revision 2's predictions against rounds 7-8

| prediction (rev 2) | outcome | verdict |
|---|---|---|
| clock 1.8 GHz | 1.81 GHz [MEAS clk.py] | confirmed |
| `read_tm` hangs because the processor bus is disabled (P15) | 0x32038 enable fixed it | confirmed |
| P17: 12.6% in-call bank stalls in the C++ GEMM loop | 333 cycles in situ vs 321 + 12 of stamps | **refuted** |
| P18: 2x2 asm loop -> -14 ms; first step (4 chains) ~-7 ms extrapolated | first step: -1.9 ms | **mispredicted**, explained in §1 |

**What went wrong in the model, and the rule it gives.** sim9 fitted t_ks = 1.835 us per k-step as one free constant and
called the 11-13% above the static schedule "stalls". The stamps show the C++ k-step runs at its static schedule
(9 x 333 + 401 measured vs 3,383 static in the PROF build: 15 cycles of difference). The 11% was never in the k-step:
it was dependency, coupling and tile-boundary time that the least-squares fit spread over the k-steps. Multiplying
that fitted constant by a bundle ratio then scaled waits that do not scale.
- **Rule:** compose the k-step from the binary (bundles of every function on the path) and fit only what the binary
  cannot know. sim10 does this; its free GEMM parameters are per-op and per-tile terms plus one in-call stall term.

## 1. The -1.9 vs -7 ms gap, quantitatively

All rows are per depth-12 step, 15,552 k-steps per GEMM core (sum of nt x K/64 over the 11 GEMM ops x 12 blocks).

| step | ms | source |
|---|---|---|
| extrapolation used in round 7: 96 cycles saved per call (321 -> 225) | **-7.42** | |
| the asm needs a C++ wrapper: `mm_acc` is 34 bundles around `jl mm_core` (g_stripe update, acc_rows select, call, lr reload, ret + delay slots), replacing the C++ call's own 19 bundles of entry/return. Static call: 259, not 225; saving 62 per call | **+2.63** -> -4.79 | [DOC] objdump `mm_acc` 0x1600-0x16ae |
| in-call memory-bank stalls of `mm_core`: **44 cycles per call** | **+3.40** -> -1.39 | [MEAS] three ways, below |
| `attn_begin` back at minsize to fit program memory (800 vs 1,104 B) | ~+0.1 | [INF] |
| **model** | **~-1.3 to -1.4** | |
| **measured** (step8 time, r7f -> r7g, median of 20) | **-1.91** | [MEAS] |

**The 44 cycles per call, three independent ways:**
1. sim10's joint fit of the r7a and r7g timelines: s = 44.0 cycles per call (robust: 43.8-44.0 whatever is fixed).
2. The main agent's bank microbenchmark (`build/mmb_run.out`, the asm call alone, A / B / C-load / C-store bank
   placements swept): 238 cycles when all four are in different banks, **271 when C loads and stores share a bank**
   (accumulators updated in place, as deployed), 252 when A and B share one, 285 for both, 322 for the worst.
   Applied to the deployed placement of core (0,2) (`ldScripts_main_core_0_2.ld.script`: B fifo 0x7C800 / 0x7DA00 =
   bank 3; A ring 0x77A00 = bank 1, 0x7EC00 / 0x7F080 = bank 3; stripes 0-3 in the C object = bank 2, stripe 4
   straddles banks 3 / 0, stripes 5-8 in `acc_hi` = banks 0-1; stripe j uses A slot j mod 3):
   (271 + 285 + 285 + 271 + ~305 + 285 + 271 + 285 + 285) / 9 = 282.6 -> **+44.6** over the conflict-free 238.
3. The round-8 g11 PROF stamps: `mm_acc` busy region 337 cycles per call against 284 static bundles on its common
   path: +53 (my static path count is +-5).

**What did not absorb anything** (round-8 g11 stamps, asm `pa` vs C++ `pc`, every op):
- Call and lock gaps: between calls inside a k-step 47 vs 43 cycles, at k-step starts 66 vs 62. Static code in both;
  the +4 is the wrapper's longer tail.
- Dependency waits: the first-tile waits are identical in both builds (27-50 us per op start).
- "K = 1536 A reloads": these are not mid-op. Each GEMM op waits once, at its first tile, for the memtile A slot load
  (§2). K = 1536 ops have only 4 tiles per core, so that one wait showed at the p75 of round 7's table.

**Pre-registration score** (`r8/prereg_gap.txt`, written before the stamps were read).
- Best guess H1 ("in-call memory stalls; ring and k-step gaps at static") is confirmed.
- Predicted values against measured:

  | quantity | predicted | measured |
  |---|---|---|
  | cyc/call | 305-325 | 337: right direction, magnitude underestimated by ~15 |
  | gA/call | 45-55 | 47 |
  | gKB/ks | 60-75 | 66 |

## 2. sim10, and the critical path of today's step

**Model** (`tools/sim10.py`):

```
t_ks  = (9 x (CALL + s) + 89 + d) / 1.81 GHz        CALL: C++ 321, asm 259 [DOC]; 89 = IRON k-step code [DOC]
T_gemm = L_dep + nt x (K/64 x t_ks + t_tile + ringx_op) + g x t_gelu
```

- L_dep applies to EVERY GEMM op. The stamps show a first-tile wait in all 11, because the A slot is single-buffered.
- ringx = the measured extra A-ring waits of yx (+5.04 us per tile) and qkv (+1.23), explained in §3, P23.
- Norms, SSM and attention are per-op constants.
- Fitted jointly on full12r7a + full12r7g, blocks 1-10, 320 spans. Shared: GEMM / norm / SSM terms and d. Per build:
  attention, and s for the asm.

**Fit error:**
- **span rms 13.6 us** (r7a 13.2, r7g 13.9);
- block totals 6.189 / 5.963 ms modelled vs 6.192 / 5.960 measured (0.05%).

sim9 reached 9.4-9.8 us, but with t_ks free per build: that freedom is what let it absorb the waits and mispredict P18.
The fitted values:

| L_dep | t_tile | t_gelu | d | s (asm) | norms | SSM | attention r7a / r7g |
|---|---|---|---|---|---|---|---|
| 21-23 us | 7.3-7.6 us | 4.7 us | ~0 (unidentified; stamps: 15) | 44 | 124 / 123 / 148 / 146 us | 1,320 us | 1,052 / 941 us |

**Per block of full12r7g (5.96 ms):**

| part | us/block | ms/step | share | what it is [tag] |
|---|---|---|---|---|
| GEMM MAC schedule (mm_core, 225 bundles x 9 x 1296) | 1,450 | 17.4 | 24% | static [DOC] |
| GEMM in-call bank stalls (44/call) | 284 | 3.4 | 5% | [MEAS 3 ways] |
| GEMM call + IRON code (34 x 9 + 89 = 395 per k-step) | 283 | 3.4 | 5% | static [DOC] |
| GEMM tile boundaries (92 x 7.6 us: C handoff, acc_zero, epilogue) | 700 | 8.4 | 12% | [INF fit]; split not measured (P24) |
| GEMM op starts (11 x ~21 us: A slot load after the previous op's last send) | 251 | 3.0 | 4% | [MEAS stamps 27-50 us in g11; fit 21] |
| GELU epilogues (36 x 4.7 us) | 170 | 2.0 | 3% | [INF fit] |
| yx / qkv row-coupling waits | 55 | 0.7 | 1% | [MEAS r7p stamps] |
| **SSM (EW cores)** | **1,320** | **15.8** | **22%** | [INF fit] |
| **attention (GEMM cores)** | **941** | **11.3** | **16%** | [INF fit] |
| **norms x 4 (EW cores)** | **540** | **6.5** | **9%** | [INF fit] |

**The critical path is the whole op chain.**
- Every op waits for its predecessor. The runtime sequence brackets every EW op with `finish_all`, and GEMM op i+1's A
  slot load starts only after op i's last send. The timeline confirms it: no op's first write precedes the previous
  op's last write by more than its own first tile.
- So every EW op sits on the critical path at 100% of its time: SSM 15.8 ms, norms 6.5 ms, attention 11.3 ms.
- While the EW ops run, the 24 GEMM cores idle: 1.86 ms per block = 22.3 ms per step.
- While GEMM ops and attention run, the 8 EW cores idle: 4.10 ms per block.

## 3. Levers ranked by predicted gain (sim10, each alone, depth 12, from 71.7 ms)

| rank | lever | ms | reasoning |
|---|---|---|---|
| 1 | SSM 1.32 -> 0.5 ms per block (P13) | **-9.8** | 22% of the step, all on the path, GEMM cores idle meanwhile; compute-bound on one EW core per column (ring I/O alone 0.28 ms) |
| 2 | 2x2 register blocking, mm_core 225 -> ~150 (P18 full) | -5.8 | A and B pops each feed 2 MACs: ldb 18 per 32 MACs, the vec slot binds. Needs P21's bank discipline first, or the denser traffic stalls more. Then t_ks ~0.77 us = the A-port floor (~7.5 B/cycle needed vs >= 7.2-7.6 measured in P1) |
| 3 | one call per k-step, stripes pipelined (P20) | -5.0 | removes 9 x 34 wrapper + 89 IRON -> ~40 per k-step, and overlaps each stripe's ~13-cycle ramp and ~20-cycle drain with its neighbours (§4) |
| 4 | fold the norms into the GEMMs (P7) | -4.6 | 4 x ~135 us per block on the path; the shim channel + ring protocol, not compute |
| 5 | tile boundaries halved (P11 via P24) | -4.2 | 7.6 us x 92 tiles per block. The plain epilogue is 0.63 us [MEAS] and acc_zero ~0.33 [INF], so ~6 us is C handoff + loop. Measure before designing (P24) |
| 6 | attention 0.94 -> 0.6 ms per block (P12 / P19) | -4.1 | the exp is at 49 bundles per 64 vs a 16-vldb floor; EW cores idle during it |
| 7 | **bank-aware mm_core** (P21) | **-3.0** | cheapest: the mechanism is measured (§1). The generator forbids same-bank `vlda`/`vst` co-issue on the accumulators, and placement separates A, B, C |
| 8 | A slot chunk-streamed / ping-pong (P5) | -2.4 (up to -4) | removes the op-start slot wait (21 fitted, 27-50 measured) on all 11 ops; also stops reading K = 768 sources twice (-107 MB DDR per step) |
| 9 | balance epilogue kinds across the rows of a column (P23) | -0.7 | yx gives row 0 all GELU tiles and row 2 none; qkv gives Q / K / V to rows 0 / 1 / 2. The A broadcast couples the 3 rows, so the skew becomes waits on the fast rows |

**Combined** (sim10; two frames in flight from §6):

| scenario | depth 12 serial | depth 12, 2 in flight | depth 10 serial |
|---|---|---|---|
| today r7g | 71.7 ms (13.9 fps) | 53.8 ms (18.6 fps), latency 130 ms | 59.8 ms |
| GEMM: P21 + P20 | 63.7 | 47.6 (21.0 fps), latency 116 ms | 53.1 |
| GEMM: P21 + P20 + 2x2 | 59.8 | 44.3 | 49.9 |
| all levers, 4-chain GEMM | 38.0 | 30.2 (33.2 fps), latency 73 ms | 31.7 |
| **all levers + 2x2 GEMM** | **34.1 (29.3 fps)** | **26.9 (37.2 fps), latency 65 ms** | **28.5** |

The serial ceiling stays ~34 ms at depth 12 (rev 2 said 35.2). Below that:
- the A port binds again at ~0.75-0.8 us per k-step;
- the remaining fixed costs are per op and per tile.

## 4. Call and lock overhead: what a lock costs, and how 9 calls become 1

**What the protocol costs** [DOC objdump + llvm-aie TableGen; MEAS r7p stamps]:
- IRON lowers an ObjectFifo / Lock acquire to one `acq #id, rN` and a release to one `rel #id, rN`:
  - id = 48 + the lock number for the core's own tile;
  - the register holds -1 for acquire-greater-equal-1;
  - Peano's `aie2p_locks.h` adds conditional forms (`acquire_cond`, `release_cond`) and pointer forms.
- `AIE2PGenSchedule.td`: ACQ and REL each occupy the LCKREQ unit for 4 cycles. That is one lock request per 4 cycles
  per core; the issue costs one slot.
- When the lock is free, the measured cost is about 0. r7p: 3,398 cycles measured per k-step vs 3,383 static, i.e.
  15 cycles over 20 lock operations (9 A acquires, 9 A releases, B acquire, B release).
- IRON's per-call code, between two `jl mm_acc`, is 8 bundles: the acquire in a delay slot, the next A-slot pointer
  as a constant (the ring index is unrolled), the release, `nopx`, `jl`. Per k-step it adds the B acquire / release and
  the loop counters: **89 bundles per k-step**, identical in every round-7 build.
- The asm path adds the 34-bundle `mm_acc` wrapper per call.
- **Total non-MAC code per k-step: 89 + 9 x 34 = 395 cycles = 14% of today's in-situ k-step** (9 x (259 + 44) + 89 =
  2,816).
- **The cost is the call ABI, not the locks:** `jl` + 5 delay slots, the stack frame, the `lr` reload (7-cycle load
  latency, padded with nops), `ret` + 5 delay slots, and the global `g_stripe` load / store.

**How 9 calls per k-step become 1** (P20):
1. **One asm entry per k-step**, `mm_kstep(A0, A1, A2, B, C)`:
   - It loops over the 9 stripes in a zero-overhead hardware loop.
   - It takes the A ring locks itself: `acq` / `rel` placed in free slots of the schedule (LCKREQ 4-cycle spacing is
     trivially met). Pin the ids with `Lock(tile, init, lock_id=n)` as ew8 does, and stop IRON from touching
     aF / aU.
   - IRON keeps the B acquire / release, once per k-step.
   - The three A ring buffers are passed as arguments: their addresses are fixed (a00_0 0x77A00, a00_1 0x7EC00,
     a00_2 0x7F080) and not contiguous.
   - The stripe's accumulator pointer advances by 4 KB per stripe, with one jump at ACC_LO into `acc_hi`. Either
     pass both bases, or move `acc_hi` so the 9 stripes are contiguous.
2. **Pipeline the stripes:**
   - Today each call has a ~13-cycle ramp before its first MAC and a ~20-cycle store drain after its last.
   - In a loop, stripe j+1's A fill and pops can issue during stripe j's last group. `gen_mm.py` already does this
     between the two row blocks inside a call: A_q is re-popped after its last reader.
   - The steady period becomes 4 groups x 48 = 192 cycles per stripe.
   - Predicted k-step: 9 x 192 + ~40 = ~1,770 cycles static, against 2,420 today.
3. **Larger k per call (k = 128):** not yet. It needs both B objects held, and B L1 is 2 deep: there would be no
   prefetch slack. B depth 3 costs +4.6 KB of L1, which is not free (§5 of MACHINE).
4. **Kernel-driven tile loop, later:**
   - The same entry could take the whole tile (12 or 24 k-steps) and fold `acc_zero` into the first k-step: MACs from
     zero, no accumulator loads.
   - That needs B's lock ids in the kernel, i.e. B as a hand-built ring instead of an ObjectFifo, as A already is.

## 5. Program memory: what fills the 16 KB, and how to free it

full12r7g GEMM core text = 16,176 of 16,384 B [DOC llvm-nm / llvm-size]:

| piece | bytes | notes |
|---|---|---|
| IRON control code `core_0_2` | 3,728 | two copies of the GEMM loop nest (0x080-0x460: 992 B; 0xa60-0xeb0: 1,104 B), because the block's ops are two GEMM segments around attention. The attention control (0x480-0xa60: 1,504 B) has the 3 x 3 Q-copy loop unrolled (9 `jl attn_qcopy`, 928 B) |
| `acc_epilogue` + `bias_pass` + `gelu_lut_n` | 2,912 + 848 + 448 | flag-driven epilogues |
| `mm_core` (asm) | 1,524 | 225 straight-line bundles (6.8 B per bundle) |
| attention: `chunk_k` 1,456, `attn_begin` 800, `attn_end` 784, `chunk_v` 608, `rescale_rb` 560, `aexp_push` 464, `attn_qcopy` 400, `attn_chunk_k` 304, `attn_chunk_v` 256, `rows8` 112, `nk8::exp32` 304 | 6,048 | 37% of the program memory |
| `acc_zero` 336, `mm_acc` 176, `_main_init` 112, `__start` 32 | 656 | |

How to free it, largest first; each is bit-identical by construction (P22):

| change | frees | how |
|---|---|---|
| one GEMM segment instead of two | ~1,000 B | the worker body walks one op table in which attention is an entry: a runtime `if` on the op kind |
| stop unrolling the attention control loops | ~800 B | trip counts from an L1 table, the same trick that stopped the GEMM ops unrolling in round 5 |
| P20 (one call per k-step) | ~400 B per GEMM copy | 9 unrolled calls (496 B) -> 1 |
| `mm_core` as a 2-variant group loop in a hardware loop | ~800 B | the 4 groups are identical modulo A registers and pointers; `gen_mm.py` emits a modulo-scheduled body |
| `rescale_rb` uses the attention exp table instead of `nk8::exp32` | 304 B | accuracy check with r7/expemu.py |

- Together that is **~3.5 KB**. It buys:
  - -DPROF in the full step (the stamps need ~600 B);
  - `attn_begin` back at -O2 (+304 B);
  - room for P20 / P18's asm.
- Not worth it:
  - Overlays: rewriting program memory from the runtime sequence between ops puts KB-sized register writes on the
    critical path and needs the core stopped.
  - Splitting kernels across cores: all 24 GEMM cores run every op, so a core-specific elf would only help if
    attention moved to a subset of cores, which costs more than it frees.
  - minsize on hot epilogues: up to 10x slower, SKILL §3.

## 6. Two frames in flight (P14) re-priced, and the ceilings

Re-priced with sim10's r7g constants and sim9's two-resource list scheduler (G = 24 GEMM cores: GEMM ops +
attention 4.10 ms per block; E = 8 EW cores: SSM 1.32 + norms 0.54 ms per block):

| scenario | depth | serial | 2 in flight (optimistic) | 2 in flight, norms serial with G | latency |
|---|---|---|---|---|---|
| today r7g | 12 | 71.7 ms (13.9 fps) | 53.8 ms (18.6 fps) | 55.3 ms (18.1 fps) | 124-130 ms |
| today r7g | 10 | 59.8 (16.7) | 44.9 (22.2) | 46.1 (21.7) | 102-110 |
| today r7g | 8 | 47.9 (20.9) | 36.1 (27.7) | 37.2 (26.9) | 82-87 |
| P21 + P20 | 12 | 63.7 (15.7) | 47.6 (21.0) | | 116 |
| all levers + 2x2 | 12 | 34.1 (29.3) | 26.9 (37.2) | | 65 |

- **1.30-1.33x fps for ~1.8x latency.** The gain is the SSM hidden behind the other frame's GEMM work. The norms share
  shim MM2S channel 0 with the A-slot loads (the packet-shared channel), so in the conservative column they stay
  serial with the GEMM side.
- P13 (a faster SSM) and P14 buy the same hidden time, so their gains do not add. After P13, P14 is worth ~1.15x.
- **Costs:**
  - a runtime sequence with both frames' ops interleaved and no `finish_all` between them: P3's txnlint first;
  - the SSM state double-buffered, ~30 MB more activations in DDR, and one frame more of input latency.
- On the product's Pareto front it is a pure fps point. The serial build stays the latency point.

## 7. What still stands from revision 2

- Compute, not DDR, bounds the step: DDR ~22 GB/s of ~55. Activation residency would target the wrong limit.
- Row-chunk overlap inside a frame loses (rev 2 §4.2): the per-tile and op-start costs repeat per chunk.
- **Changed:** the k-step "stalls" (rev 2 §3) are gone from the model, replaced by bank stalls specific to the asm
  call (44 per call) and by per-op / per-tile terms.
- **Changed:** the "L_dep only for non-prefetchable ops" assumption is gone: every op pays it.
