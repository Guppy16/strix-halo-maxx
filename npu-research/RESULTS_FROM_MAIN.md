# Results from the main agent (round 6), keyed by proposal id

All numbers: testbox, Turbo, gpu_busy 0, depth 12, pyxrt kernel time unless said otherwise. MEASURED unless marked.
Baseline of round 6 = `full12r6a` (round-6 -O2 pass-structured epilogue + vector 1/sum in attn_end): 84.0 ms per
dispatch (host-polled timeline) / 84.14 ms median of 20 (`step8.py time`), vs round 5's full12m 89.4.

## P1. Twin builds: A-port or compute bound? — DONE, verdict: COMPUTE (+ per-call overhead) bound, not A-port bound

Builds from the round-6 sources: `full12p1n` (`-DMM_NOCOMPUTE`: mm_acc returns after event0; IRON's control code
still acquires/releases the A ring and B fifo) and `full12p1t` (`-DMM_TWICE`: the rb loop twice on the same A, B).
Full step with GELU, host-polled timelines (`build/tl_full12p1{n,t}_d12.json`, `r6/tlview.py` to compare).

| build | dispatch | delta vs base |
|---|---|---|
| full12r6a (base) | 84.0 ms | |
| full12p1n (no MACs) | 67.9 ms | -16.1 |
| full12p1t (2x MACs) | 108.7 ms | +24.7 |

Per op (block 5 share, ms; ops whose output words did not change in the garbage build are invisible to the poll):

| op (K, nt) | base | no MACs | 2x MACs |
|---|---|---|---|
| down_a (1536, 4: 96 k-steps per core) | 0.20 | 0.10 | 0.35 |
| up_a (768, 16 gated: 192 k-steps) | 0.61 | 0.42 | 0.90 |
| mlp0 (768, 16 gelu_pan) | 0.73 | 0.53 | 1.02 |
| mlp3_a (1536, 4) | 0.23 | 0.15 | 0.39 |

- down_a without MACs: 0.10 ms / 96 k-steps = 1.04 us per k-step INCLUDING the tile boundaries. So A reaches each
  core at >= 10,368 B / 1.04 us = 10 GB/s, i.e. >= 7 B per cycle at the assumed 1.43 GHz: the "4 B/cycle port
  floor of 1.813 us" does not hold for this path (either the port is faster or the clock is higher than assumed;
  P10 would separate the two).
- The second copy of the MACs costs 0.15 ms on down_a = 1.56 us per k-step: the compute per k-step is ~1.56 us,
  ABOVE the 1.04 us data-movement time, and adding it costs the full amount (no slack hides it).
- Conclusion: the GEMM k-step is compute + call-overhead bound. Loop work (bundles per mmul: the deployed hardware
  loop is 28 bundles per 16 mmuls = 1.75 by asmstat) and cheaper epilogues DO pay, down to the ~1.0 us per k-step
  data-movement time measured here (~-33% on the GEMM loop at most before the movement binds). P6 (two A streams)
  is not the next GEMM item.
- Consistent evidence: the -O2 epilogue rewrite (round 6, no change to data movement) saved 5.3 ms per step (89.4 ->
  84.1), which a port-bound tile boundary would not have allowed.

## P2. In-kernel cycle stamps — BLOCKED (measured hazard), verdict: needs another route
`read_tm(0x340F8)` (Peano's tile-memory-map read of the aie2 regdb's core Timer_Low; `aie::tile::cycles()` links
against an undefined `get_cycles()`) HUNG the core: the one-core probe `r6/kprobe.py` timed out (timeout 1 of round 6;
the device recovered, known-good full12r6a 83.97 ms right after). The same probe with the stamps compiled out runs.
There is no cycle builtin in Peano (only `__builtin_aie2p_{get_coreid, get_ctrl_reg, get_status_reg, read_tm, ...}`).
Not retried. Cycle counts now come from host timing of repeated work on one core (kprobe `time`, `-DREPS=16` with
each repetition consuming the previous output so none can be elided) or from the trace unit (one-column builds).

## P8. Real-activation numerics harness — DONE (parts 1-3), verdict: valuable, found two real issues
`r6/capture.py` (CPU bf16 model, clip30 3 frames, 96 random token rows x full feature dim per block and site, all 12
blocks -> build/acts_clip3.npz) + `r6/numh.py ranges|gelu` + `r6/emu.py` (bit-level models of the nk8 ops; checked
against the device on every bf16 input: GELU 99.17% / exp32 99.86% / sigmoid 99.92% bit-equal, the rest 1 ulp in
narrow bands; the LUT GELU model 100.00%).
- Ranges (MEASURED on real data): GELU inputs are large: vit (mlp0) |x| max 195, 21% beyond the +-8 exp clamp
  (harmless: the product uses the unclamped x); ssm_gelu max 47 (7% beyond 8); ffw_gate max 29. Attention
  s - rowmax down to -94 in block 0 (past expd's -83 floor: P = 2^-120 instead of 0, harmless); row-max spread up to
  42 in block 0. RG-LRU gate args to +-105 (clamp 60: 8e-6 beyond, sigmoid saturated anyway).
- **The ViT MLP uses the exact (erf) GELU in torch (nn.GELU()); every NPU build so far computed the tanh form.**
  Against the erf reference the deployed kernel had p99 39 ulp on that site (bfp ratio 0.18 vs 0.09 for the tanh sites).
- Yardstick added: bfp ratio = rms(approx - exact) / rms(bfp16(exact) - exact) on the tensor the result meets next.
  Round 5 rejected cheap GELUs on per-element bf16 mismatch; by this yardstick one Newton step in the reciprocal costs
  0.4-0.6 (visible), the quadratic 2^f costs nothing (0.059/0.265 = c0).

## P9. Disassembly / object auditor — PARTLY (r6/asmstat.py), verdict: valuable
`r6/asmstat.py file.s [funcs] [--blocks]`: per function bundles, nop-only bundles, stack refs, call sites; per loop
(hardware and branch loops) bundles per iteration, spills and calls. `r6/ccs.sh SRC TAG [flags]` compiles like the
design and prints the size table + the loop table. It found: the round-5 epilogue at ~77 bundles per 64 elements
(minsize helpers called per block), the GEMM inner loop at 28 bundles per 16 mmuls = exactly its ldb-slot count
(27 pops/fills), the attention loops at ~4.5 bundles per mmul, the `linear_approx` object being reloaded from the
stack every lookup. Not yet: llvm-objdump of the linked elfs, baseline diffs, a bb.sh hook.

## P11. Cheaper tile boundaries — PARTLY (epilogue only), measured -5.3 ms
The per-tile epilogue (bias + conversion + packing) is now one pipelined bias pass (5.5 bundles per 64 elements
from ~77) and array passes: 89.40 -> 84.14 ms per dispatch (bit-identical on every non-attention check of t1/t2/t4).
Then the GELU by table (below): 84.14 -> 78.16 ms. Drain overlap / acc_zero folding (options 1-2) not tried.

## (main agent's own) GELU by lookup table — DONE, -6.0 ms, AJ 70.83
AIE2P has a 4-way parallel table lookup (`load_lut_2x_float`, aie::lut<4, float, bfloat16>). gelu(x) = x s(x),
s piecewise linear in 64 segments of 1/8 on [-4, 4) (fp32 offset, bf16 slope, offsets refit for the rounded slopes),
s hi/lo into the product. One core: 16 ns per 32 lanes vs 63 for the arithmetic GELU (MEASURED, kprobe time); the
hand-written lookup is 2.7x faster than aie::linear_approx (whose object is reloaded from the stack per call).
Real-activation rms error = the deployed kernel's (numh: 4.40e-4 / 2.05e-3 / 4.77e-4). mlp0 uses an erf table.
Tables: the ab copies in the GEMM cores' data area (data_size +2 KB, stack 6 -> 4 KB; measured need ~2.5 KB), the cd
copies in bank 3. Step: 84.14 -> 78.16 ms; DAVIS 10 videos depth 12 AJ 70.83 (round 5 71.00, CPU 70.98).
Also measured: the hardware `vtanh` (aie::tanh) is crude (identity for |x| <= 0.5, 7% relative error near 0.5-1,
2% at 1-2) and `aie::exp2` has p99 10 ulp: neither usable for these kernels. `aie::inv` is correctly rounded but
scalar.

## Clock: an open question raised by these measurements
The deployed GELU loop is 110 bundles per vector and runs in 63 ns per vector on one core (repetition-timed): that
implies >= 1.75 GHz, not the 1.43 GHz the notes assume. P1's ">= 7 B/cycle" A-port reading also shrinks at a higher
clock. P10 (port-rate microbenchmarks) plus one timing of a known-bundle loop should settle the clock.

# Round 7 (main agent), keyed by proposal id
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked. Cycles at 1.81 GHz (P10 below).

## P10 (clock part) — DONE: 1.81 GHz in Turbo
`npu-tapnext/r7/clk.py` + `clk.cc`: pure-register hardware loops (scalar LCG 4 bundles/trip, dependent bf16 vmac 3
bundles/trip; no loads or stores in the loop, so nothing can stall) with a runtime trip count, host-timed at 0..32M
trips: slope 1.8086 / 1.8076 GHz (fit on medians), ratio of the two loops 1.3329 vs 4/3. With P15's timer (below) the
same loops read exactly bundles x trips (+48 / +59 cycles of call overhead). Agrees with the driver table (1.8 GHz).
Default mode not measured (the box is in Turbo; changing it needs sudo).

## P15 — DONE, verdict: the processor-bus enable fixes read_tm
`npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=r)` as the first op of the runtime sequence (from the
IRON sequence function: `from aie.dialects.aiex import npu_maskwrite32`), `__builtin_aie2p_sched_barrier()` around
`read_tm(0x340F8)` (objdump: each `lda.tm` alone in its bundle). One core (clk.py --stamp): timer = bundles x trips
exactly. 24 GEMM cores in the full step (design8 with `--kdefs=...,-DPROF`): no hang, 3 dispatches. 0 timeouts.

## P16-lite — DONE (stamps, not the counters): `npu-tapnext/r7/prof.py`
mm8.cc `-DPROF`: per tile, cycles inside mm_acc, calls, acc_zero -> epilogue wall, epilogue cycles, written over the
tile's first 20 bytes (timing build only); prof.py finds the records in ACT after 3 frames and maps them to ops.
Block 11 (b0 for yx), full12r7p:

| op | calls/tile | cycles per mm_acc call | busy % of k loop | k-wait per tile min / median / max (us) |
|---|---|---|---|---|
| yx | 108 | 333 | 42 | 2.66 / 7.70 / 180 |
| qkv | 108 | 333 | 50 | 2.66 / 3.89 / 204 |
| out_proj | 108 | 333 | 57 | 2.66 / 2.66 / 54 |
| mlp0 | 108 | 333 | 57 | 2.66 / 2.66 / 204 |
| up_a / up_b | 108 | 333 | 88 | 2.66 / 2.66 / 2.66 |
| down_a / down_b | 216 | 333 | 77 / 75 | 5.29 / 5.29 / 32-38 |
| mlp3_a / mlp3_b | 216 | 333 | 69 / 75 | 5.29 / 5.29 / 38-55 |
| out (after the SSM) | 108 | 333 | 5.5 | 2.66 / 2.66 / 1368 |

- **mm_acc runs at its static schedule in situ** (321 bundles + ~12 for the stamps): no memory / bank stalls inside
  the GEMM loop.
- Steady-state overhead outside mm_acc: 2.66 us per 12 k-steps = ~400 cycles per k-step (9 calls + IRON's lock
  code). k-step = 9 x 333 + ~400 = ~3,400 cycles = 1.88 us (sim9's fit: 1.854).
- The large waits are dependencies: the first tiles after EW ops (norms 180-204 us, the SSM 1.37 ms before `out`)
  and the K = 1536 ops' mid-op A reloads (12-55 us at p75-max: P5's case).

## P17 — REFUTED by the stamps (not built)
No in-call stalls exist to remove (333 measured vs 321 static + stamps). Bank placement would not change the k-step.

## P12 (attention) — partly done; see npu-tapnext/RESULTS.md 16.2
Measured split (t4, twins): K phase 0.60 ms (exp + push 0.30, QK^T + max 0.30), V phase 0.19, data/control floor
~0.18 (barely overlapped), per-head-group fixed costs ~25-35 us (pass 0) / ~15 (pass 1).
Done: exp by table fused with the bfp push (accuracy better than the arithmetic exp on real scores), -O2
begin/end (the ones panels were written byte by byte), Q popped per key block (no 8-register hold), the 1/8 score
scale folded into constants (bit-identical). t4 1706 -> 1506 us; step 76.74 -> 73.10 ms.
Not done: K/V re-stream per pass (P12 proper) — the twin numbers say the passes' compute, not K/V traffic, dominates.

## P18 — cheap first step tried in C++: NO (Peano)
4 accumulator chains (even / odd k): 54 bundles per 16 mmuls with spills, 42 without the c prefetch, 61 with A
popped per block via an opaque seek (round 6's loop: 28). Peano keeps A in 8 of the 12 ex registers and does not
pipeline across column blocks. The hand-asm route is needed (in progress / next).

## P18 — DONE (first schedule, asm), verdict: bit-identical, -1.9 ms measured; the call is no longer the whole story
`npu-tapnext/r7/gen_mm.py` emits `mm_core` (exact-cycle straight-line schedule: 4 accumulator chains over 2 column
blocks, A held in ex0-7, B on ldb through a 4-register conveyor, accumulator quarters on lda / st, EDF placement,
built-in conveyor checks). B packed [q][4 col blocks] per column-block pair on the host (pack8 B_PAIR4).
225 bundles per call vs 321; directed tests and the full-step regress BIT-IDENTICAL; step 73.10 -> 71.19 ms.
Your prediction (k-step 0.95 us, 62.2 ms) was for the full 2x2 design; this first schedule's EXTRAPOLATED gain was
~-7 ms and only -1.9 was measured. Not explained yet: the in-situ profile of the asm call is the next action
(PROF + MM_ASM does not fit the program memory together with the attention code: use a GEMM-only test).
Group boundaries cost 12 cycles per 32 MACs (16 serialised stores + loads on 5 accumulator registers).

# Round 8 (main agent), keyed by proposal id
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz. Details:
npu-tapnext/RESULTS.md §17, NOTES.md §13.

## P18 follow-up — the -1.9 vs -7 ms gap, MEASURED: L1 bank conflicts inside mm_core
g11 -DPROF in situ: C++ 361, asm 337 cycles per call (PROF window ~+40). r8/mmb.py (one core, no DMA): 238 conflict
free, +33 accumulator load + store in one bank (as deployed), +14 A and B in one bank, +42 B with C. Your rev-3
attribution (+2.6 wrapper, +3.4 banks) agrees; your stamp prediction matched (337; gaps 47 / 66).

## P21 — PARTLY: placement DONE (-1.5 ms), the scheduling rules REFUTED
- Placement: A ring pinned to banks 1, 2, 2 (`Buffer(mem_bank=)` exists in IRON 1.4.4; design8 `A_BANKS`): g11 337 ->
  323.5 cycles per call, step 72.15 -> 70.62 ms, bit-identical.
- R1 in gen_mm.py (`--noconf`: no accumulator vlda in a bundle with an accumulator vst): **273 bundles** static (225
  today), not 225-232: a 4-chain group boundary has 16 loads + 16 stores of one bank inside a ~16-cycle window (stores
  >= last MAC + 6, loads <= next first MAC - 4), so serialising them stretches every group by ~16 cycles. A-ops off
  B's cycles (`--noconf-ab`): 255. Prediction (295 +- 8 per call, 68.2 ms) not reached by rules; the fix is a layout
  that puts consecutive groups' accumulators in different banks (NOTES 13.4, designed) or P20's restructure.

## P13a — DONE (stamps), P13 — PARTLY (bit-identical kernel work)
Stamped in place (ew8 -DPROF, records in each chunk's unused output pad, r8/ewprof.py): 1,340 us per op, waits 1.3%
(input 1.0%, output 0.3%): compute-bound, so not an I/O fix. Split per chunk: gate GEMMs 21%, sigmoid 23%, log_a +
exp 13%, sqrt 11%, out copies 10%, recurrence 9%, conv 7%. Bit-identical rewrite (gate GEMM specialised 2,383 ->
652, contiguous part views, copies, sqrt unrolled): 1,115 us (t3), step -2.4 ms. Next: the transcendental chains
(52% now) need numerics changes (tables) or hand-scheduled chains; real gate inputs reach -87 / -105, so a sigmoid
table must keep the tail exact enough for a = exp(-8 sp sigma).

## P22 — PARTLY: (a) + (b) DONE (-1,328 B, -0.4 ms, bit-identical)
One table-walked op loop with attention as a table entry and the attention trip counts from an L1 table: GEMM text
16,176 -> 14,848 B. -DPROF fits the full step again. (c) / (d) not done.

## P24 — DONE (stamps): the boundary is 2.4 us, not 7.6
handoff (epilogue end -> next acc_zero) 1.31 us median, acc_zero 0.49, plain epilogue 0.64, tail 69 cycles. The rest
of sim10's t_tile is the op's first-tile A-slot wait (27-50 us in g11, 177-203 after norms, 1,137 after the SSM in the
step): P5 / P7 territory, not the C handoff.

## Attention fixed costs (task item, P12) — MEASURED small
Per head group (full step, 192 records): pass 0 = Q copies 5.5 + begin 1.0 + K 90.6 + V 33.8 + call gaps 1.8 + end
2.1 us; pass 1 = 3.8 + 0.7 + 45.8 + 17.4 + 1.8 + 1.1. Fixed ~70 us per block (8%); the K phase is the lever.

## P20, P23, P5, P7, P14 — not attempted this round

# Round 9 (main agent), keyed by proposal id
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz. Details:
npu-tapnext/RESULTS.md §18, NOTES.md §14. Step 67.89 (r8c) -> 58.14 ms (full12r9s), d10 48.5 ms, AJ 70.84 / 69.90.

## P20 (rev 4) — DONE, all three stages; -6.4 ms measured vs -8.2 predicted
- (b) first, as a C++ loop with the locks inside (mm_kstep): -2.0 ms (g11 35.72 -> 33.93).
- (a) pieces: affine layout (pair 0 rb < 12 in C slot rb, rb >= 12 in acc_hi 0-5, pair 1 in acc_hi 6 + rb), not
  p20map's table (a table-indirect epilogue cost 0.8 us per tile; strided passes keep it at 0.72 us). Your bankmodel rule
  reproduces the one-core bench (static + model 2,127 vs 2,200 measured). In situ -0.9 ms (prediction -2.6).
- (c) r9/gen_ks.py: period 192 bundles as you specified, record stream via post-increment (padds/padda immediates must be
  multiples of 64: no per-frame bump of a 40-B record), m0-m3 shared by both cursors, dummy 10th stripe with acq.cond.
  -3.5 ms, bit-identical. Prediction for g11 PROF 1,950-2,150 cycles per k-step: **measured 2,588 (down_a) - 3,320 (out)**.
- Why the gap: the k-step is now A-port-bound. 10,368 B of A per k-step at ~4 B/cycle = 2,592 cycles = the measured
  2,588. sim11 should add a per-core A-port term (and a lower rate for K = 768, nt 4-8 ops: ~3.1 B/cycle).
- Two dispatch timeouts on the way (loop start alignment + uninitialised r25; acq with value 0 = acquire_equal 0).

## P26 — DONE differently: exact sqrt by a 256-entry mantissa table (int16 LUT), not 32 segments
Bit-exact RNE on every bf16 >= 2^-9 (probe); 28.3 vs 65.7 cycles per 32 lanes; in situ 1,093 -> 493 cycles per chunk.

## P28 — DONE on both gates at once (one asymmetric table [-12, 8) in 1/8 segments, 160 segments, 2.5 KB per copy)
Emulated on real gates: 5% 1-ulp vs torch (nk8 19%), a 8.5% / mult 6.6% != torch (nk8 13% / 40%); DAVIS d12 70.84,
d10 69.90 (r8c 70.78 / 69.83). 27.4 cycles per 32 lanes (75 before). With sqrt: -2.8 ms (your -1.0 -1.1 -0.8 = -2.9).
CPU-model control: any ulp-level SSM change (nk8, tables, random flips) gives the same track noise.

## P27 — NOT DONE (memory); what was done instead
EW L1 had no 2 x 1.7 KB left after the sigmoid / sqrt tables (they took every free byte + 2.5 KB of a compacted arena).
Instead today's exp_a2 ops split into 5 pipelined passes (bit-identical, 63.5 -> ~44 bundles per vector): -0.22 ms.
A range-reduced 2^f table was as accurate as exp32a and not faster (measured). P27 next, after freeing EW memory.

## P5a, P7, P29, P33 — not attempted this round

# Round 10 (main agent), keyed by proposal id — in progress (2026-10-09 23:30)
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz.

## The A-port-bound claim of round 9 — REFUTED by its own data (early note)
Per-tile distribution of the round-9 g11 PROF records (npu-tapnext/r10/gdist.py on build/g11sp.npy): the **median** busy per
k-step is 2,050-2,065 cycles for EVERY op (out, out_proj, down_a/b, mlp3_a/b, yx plain tiles, qkv), wall 2,130 with the B
acquire. The 2,588-3,320 of round 9 were means: one tile per op (the first) carries the op-start A-slot wait inside its
first mm_kstep acquire (max 4,170-7,300 per k-step averaged over that tile). In steady state A arrives at >= 10,368 B /
2,055 cycles = 5.04 B/cycle per core, so the core A port is not the bound at today's k-step; the k-step is at its compute
(static 1,830 + bank model 24 + call / B-acquire / stamp overhead ~200-280).
Where the GEMM-op time goes instead (g11 PROF, us per tile): epilogues gelu_pan 8.19, gated 7.03, gelu ~6.1 (yx), qkv 2.70,
plain 0.71; op-start waits ~30-40 us per GEMM op (A slot load, 248 KB per column through one shim channel; 8 columns
= ~58 GB/s = DDR-bound). Full-step timeline (full12r9s, block 5, ms last-to-last): yx 0.23, ssm+out 0.89, up_a 0.39, up_b
0.37, down_a 0.15, down_b 0.14, qkv 0.29, attn 0.93, out_proj 0.11, mlp0 0.50, mlp3_a 0.15, mlp3_b 0.11, norms 0.08-0.15.

## P5r + P5a — DONE (flag A_HALF=1): -1.35 ms measured (predicted -1.6 +- 0.3), bit-identical
- Design (design8 / plan8 A_HALF): the slot's halves have their own memtile S2MM channel (0 and 5) and lock pair; ONE
  packet flow (id 4) multicasts shim channel 0 into both. K=768 ops and attention's Q objects load 124 KB once (one copy per
  half); K=1536 ops stream their 248 KB and BOTH channels write it over the whole slot (identical values; tile BDs refuse
  stride 0, so a "write twice into half 1" pattern was not possible); up_b (reuse_a) loads nothing.
- **Router bug found:** IRON 1.4.4's pathfinder silently drops a multicast destination when unicast packet flows share the
  same two DMA master ports (ids 1 -> DMA0, 3 -> DMA5, 4 -> both gave DMA5 no rule for id 4; reproduced in isolation,
  npu-tapnext/r10/pf1.mlir vs pf2.mlir; order-independent). That first version hung one dispatch (timeout 1 of the round;
  known-good t2-r9k ran right after). Multicast-only routes correctly (shared amsel).
- Measured: t2 2.09 -> 1.95 ms, t4 1.34 -> 1.30, g11 29.72 -> 27.85 ms (-156 us per block), full12 58.11 -> 56.76 ms;
  t2 / t4 checks identical, full-step regress BIT-IDENTICAL vs full12r9a.

## P23 — DONE (flag P23=1, plan): -0.85 ms measured (predicted -1.2 +- 0.3), bit-identical
yx and qkv deal their N tiles round-robin (row r, step j -> natural tile 3 j + r; pack8 order + flags per natural tile, drain
strides; qkv: step j = kind j // 4 for heads 3 (j % 4) + r). g11 27.85 -> 27.58 ms, full12 56.76 -> 55.91 ms.

## P5 (k-chunk streaming) — DONE differently: the A slot driven by RUNTIME memtile tasks (flag A_RT=1): -2.0 ms measured
(predicted -1.7 +- 0.7), bit-identical
- IRON 1.4.4 has `DmaEndpoint.task(*bds, runs=)` + `.start(repeat_count=)`: the runtime sequence pushes memtile tasks, so
  the slot needs no static chain (no BD budget, no lock-value budget). Per column 8 BDs configured once per dispatch:
  load lo / hi (chunk of 10,368 B per run, BdIteration 12, acquire E_lo / E_hi, release F), tile-0 sends (acquire F, release
  F2), middle tiles (whole region, no locks, repeat nt-3), last-tile sends (acquire F2, release E). A BD takes one acquire AND
  one release or neither, hence F2 (tile 0 -> last tile). E split lo / hi so K=768 ops never take K=1536's upper tokens.
- Bench first (npu-tapnext r10/rtm.py, one column, one core, hashes): correct first time; the next op's load runs under the
  current op (chunk q waits only for the last tile's send of chunk q): op-start waits 0.04 us with no data dependency.
- **P10 answered by the same bench: memtile -> core stream = 7.95 B/cycle** (a 64-bit stream; the consumer was trivial).
- Measured: t2 1.95 -> 1.90, t4 1.30 -> 1.25, g11 27.58 -> 25.21 ms, full12 55.91 -> 53.89 ms (all with P5r/P5a/P23).
- Ordering rule found: push an op's memtile send tasks right after its B fills and before any wait of the op (qkv's drain
  groups block the CP; a blocked push may only wait on work issued already).

## Epilogues (not a proposal yet; measured) — the GELU paths cost ~3.3 ms per step on the GEMM critical path
- g11 PROF epilogue per tile: gelu_pan 8.19 us, gated 7.03, gelu (yx) ~6.1, qkv 2.70, plain 0.71. Timeline (r10r, per block):
  up_a 0.31 + up_b 0.31 + mlp0 0.36 ms against ~0.23 ms of k-steps each.
- The deployed gelu_lut_n loop: 67 bundles per 64 lanes, not pipelined; one-core bench (r10/gxb.py) 34.2 cycles per 32 lanes
  (n = 144), 38.9 (n = 16), 44.1 (n = 8: the gated / gelu_pan call size) = its static count, no stalls.
- `#pragma clang loop pipeline_initiation_interval(16)` + restrict in / out + one vector per trip makes Peano emit a
  17-bundle pipelined loop (r10/gx.cc) — **but it runs at 54.7 cycles per 32 lanes on the core** (same with out in another
  bank): the forced schedule stalls ~37 cycles per iteration on hardware interlocks. In situ (g11 PROF) yx epi 3.39 -> 5.02
  us. REJECTED (flag -DGELU_II kept, off). Next: a hand modulo schedule with the measured latencies (gen_ks style), floor
  ~16-20 cycles per 32 lanes: -1.2 to -1.5 ms.

## Round 10 summary (main agent)
- Step depth 12: 58.11 -> **53.89 ms** (full12r10r), depth 10 48.5 -> 45.4, 8 38.9 -> 36.0, 6 29.2 -> 27.1; all bit-identical to
  round 9; DAVIS 10 videos d12 70.84, d10 69.90 (unchanged). One timeout (router bug), recovered.
- Verdicts: P5r + P5a DONE (-1.35 vs -1.6 predicted); P23 DONE (-0.85 vs -1.2); P5 DONE as runtime memtile tasks (-2.0 vs -1.7);
  P10 (memtile->core rate) answered: 7.95 B/cycle. Not attempted this round: P38, P7 / P7a, P27, P42, P35, P24b, P34, P37, P14.
- Parked items (two-channel A, neighbour-memory A, wider N tiles): agree, 0 ms today.
- New lever measured for the research side: GELU epilogues ~3.3 ms per step on the GEMM critical path (see above); a hand
  modulo schedule is the next main-agent item. sim12 may want the r10r timeline: npu-tapnext/build/tl_full12r10r_d12.json.

# Round 11 (main agent), keyed by proposal id (2026-10-10)
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz. Baseline full12r10r
re-timed 53.84 ms. Best build **full12r11d: 52.07 ms** (depth 10 43.52, 8 34.84, 6 26.23), regress BIT-IDENTICAL to full12r9a.

## The forced-II GELU result of round 10 (54.7 cycles) — it never ran the 17-bundle loop (P45's premise)
- In r10/gxb.cc and in mm8's `-DGELU_II` build Peano silently did NOT pipeline the loop (serial, ~50 bundles): mm8 made
  `in` / `out` restrict COPIES of one pointer (Peano treats them as aliasing) and gxb used MIN_ITERATION_COUNT(8). Only
  r10/gx.cc's separate restrict parameters gave the 17-bundle schedule. One-core bench (npu-tapnext r11/apb.py, gx.s loops
  extracted with r11/asmx.py): **the forced-II loop runs 17.8 cycles per 32 lanes at n = 144, 27.0 at n = 8, bit-exact**.
- So H1-H4 were not the cause. Fix: GELU_II with separate restrict parameters (mm8 `gelu_lut_io`): full12r11a **52.96 ms**
  (-0.88), bit-identical.

## P45 (hazard ladder) — DONE differently: the hazards that DO exist (r11/mk_haz.py + apb.py, one core, bit-level)
1. **Visibility is LLVM's latency, not asmcheck's**: a register write with .td latency w is seen by an op issued >= w - r + 1
   later (vlda 7, vldb.4x64 7, vadd.32 / vconv / vfloor / vshuffle / vmin_ge 2, all with a vst reader). asmcheck's
   c + r >= c' + w is one cycle optimistic; the old "vmac reads its accumulator at stage 3" is the .td's 4 under this rule.
2. **mv -> mv bypass**: a value from an mv-slot op (vmov) reaches mv-slot readers (vshuffle, vadd.32, vmin_ge, vmov) one
   cycle earlier (d = 1). With a load as the writer every reader sees it at d = 7. Consequence: a WAR window "read the old
   value the cycle before the new one lands" is unsafe for mv readers. This was the round's real GELU bug (lanes 16-23).
3. **No WAW loss** (H1 as stated is false): an older load and a younger short write to one register land in landing order,
   the younger value is readable until the load lands, nothing is dropped.
4. **Itinerary ports are structural hazards** the scheduler must reserve (aie2p_lat.json itin stages): DM_ADA (vldb.4x64 AND
   vlda), QEY_RS (LUT index reads, vfloor, vst x), QEY_WA (vlda / vconv / vfloor results), QEY_WB (vldb / LUT results),
   CRRND_RS (vconv, vst.conv). A plain vldb uses only QEY_WB. GELU per 32 lanes: DM_ADA 11 and QEY_RS 11 with an xc spill.
5. **Zero-overhead loops need the LAST bundle 16-B aligned** (else the body runs ~once). Writing every body bundle in the
   full 16-byte form makes that automatic (no padding inside the body).

## P46 (generator) — DONE as npu-tapnext r11/msched.py (+ r11/gen_gelu.py), validated bit-exact on the device
Ops over virtual registers with .td instruction names; list or iterative (Rau) modulo scheduling with slots + itinerary
ports + banks; a MaxLive-aware sink pass; modulo variable expansion with circular register allocation; emission (prologue,
full-form aligned body, drain); a value-level checker (every read sees its iteration's def outside the bypass window,
2D post-modify addresses, spills through memory). Not yet: bank costs in situ, kernel-only prologue, output streams.

## P47 (GELU epilogues) — G0-level result with an asm GELU loop: -1.41 ms total (G0 predicted -1.4 +- 0.3; G1 -3.6 not reached)
- r11/gen_gelu.py: the gelu_lut_i chain at **II 13** (ResMII 11: DM_ADA / QEY_RS; II 11-12 place but need 9-10 of the 8 free
  x registers), U 1, 5 stages, xc through a 4-slot stack ring, x reloaded for the products, NaN-safe clamp (constant first),
  2D post-modify walks (8 vectors, `jump`): one call per run (yx: 288 contiguous; gated: C slots + acc_hi 0-5; pan: C slots +
  acc_hi 0-23). One core: **14.6 cycles per 32 lanes in place** (15.6 out of place) vs 34.2-44.1 deployed, bit-exact in all
  layouts. Text 718 B (replaces gelu_lut_io 672 B; GEMM text 15,968 B).
- full12r11b **52.43 ms** (-0.53 beyond GELU_II), bit-identical. Timeline vs r10r: mlp0 -0.81, up_a + up_b -0.72, yx -0.17 ms
  per step. In situ (g11-r11gp PROF) yx epilogue 3.39 -> 1.81 us per tile.
- What is left (G1): the bias pass, the gated products + bfp pushes + stash, the pan pushes. II 10 needs the flat-edge
  tables (+4 segments per side: groups of 4, 512 B of L1 — the GEMM core has 1,152 B of never-used stack, see P42) and the x
  reload on vldb.

## P43 — DONE (flag P43=1, design8): -0.35 ms measured (predicted -0.8 +- 0.3; missed), bit-identical
B fills before the op's dependency wait; the memtile B forward one tile deep was not done. full12r11c 52.08 ms.

## P42 — tool DONE, shrink not done
r11/stackaudit.py (run by bb.sh after every build): deepest call path (stack-size sections + jl / tail-j call graph) vs the
linker script's reservation. GEMM cores 1,664 of 2,816 B (1,152 B never used), EW 4,288 of 6,144 B; both match aiecc's own
`measured_stack_sizes` warnings. design8 compiles mm8 with -fstack-size-section (text unchanged).

## Host part of the step (task item) — our code, plus thread wake-up and tenants
r11/hostprof.py (8 torch threads): conv 0.48-1.1, queries 0.12, X0 packing 0.64 -> **0.14 ms** (HOST_FAST=2: torch strided
copy into the mapped ACT bo, byte-identical), write/sync, read 0.08, heads 0.6-1.0 ms. The variance is the OpenMP pool waking
after the 54 ms idle plus other tenants (load average 1.3-2.9; other-job's worker holds one core at 100 %). Heads on one thread
make the NEXT conv slower (pool re-spin): not used. Whole step - dispatch now ~2.0-2.2 ms.

## Not attempted this round
P27 / P29 (SSM), P7a / P44 / P7-E (norms; the rev-6 advice to park the full fold accepted), P48-P50 (attention), P35, P24b,
P38, P14.

DAVIS 10 videos on full12r11d (round end): depth 12 AJ 70.84 (OA 92.57), depth 10 69.90 (OA 91.02): identical to round 10.

# Round 12 (main agent), keyed by proposal id (2026-10-10)
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
Baseline full12r11d re-timed 52.10 ms. Best build **full12r12e: 50.05 ms** (depth 10 42.02, 8 33.51, 6 25.20), regress
BIT-IDENTICAL to full12r9a at every step. Details: npu-tapnext NOTES 17, RESULTS 21.

## G1a "gup" (rev 7) — DONE (flag -DGUP): -0.55 ms measured (predicted -1.02 +- 0.2: missed, explained)
- II 6 per 64 lanes as predicted (CP-SAT, r12/cpsched.py; msched's IMS also reached it). One-core bench bit-exact, 10 cycles per
  block incl. call overhead.
- In situ (g11 PROF, gated_b tiles): epilogue 5.11 -> 3.68 us per tile, against the predicted 2.44. The prediction kept "half a
  stash copy"; the differentials show the stash copy alone is 1.70 us per tile (avg; ~3.4 on each gated_b tile), the gate GELU
  1.20, the gate bias pass 0.36, gup 0.36. gup's own saving (1.43 us x 32 gated tiles x 12) = 0.55 ms: the model was right
  about gup and wrong about what was left around it.
- The stash copy then went to a 32 B modulo copy (-DCPY32): -0.30 ms more, gated_b tiles 3.68 -> 2.10 us.

## G1b (rev 7) — DONE: panpush (-DPAN, -0.30 ms) and the fused bias + GELU (-DGELU_FLAT,-DBGELU, -0.32 ms)
- PAN serves mlp0 AND the Q / K objects of qkv (flags 5 / 6: a 2D flush jumps 1,152 B between objects): II 16 per row block
  (2 per 64 lanes, the predicted floor). The C++ pushes cost 1.23 us per mlp0 tile and 0.82 per qkv tile (differential).
- bgelu at II 12 (rev 7 predicted 12-13 with flat tables): 8 free x registers once `vmsc s, sh, 1.0` replaces the -1.0
  constant (bit-identical, as rev 7 suggested). x goes through an 8-slot stack ring (see "side effects" below). It also runs
  yx's tiles (drained layout), so gelu_ms is gone. G1 in total (GUP + CPY32 + PAN + BGELU): **-1.49 ms** vs predicted -1.76 +- 0.35.
- Text was the binding constraint: GEMM 16,288 / 16,384 B after removing `__modsi3` (an `s % 3`: 160 B).

## GELU II 13 -> 10 (handoff item) — PARTLY: flat-edge tables DONE (bit-exact on every finite bf16), II 10 NOT reached
- With flat edges + an x reload on vldb the ResMII is 10 (QEY_RS), but CP-SAT found no II-10 schedule within 8 x registers
  (400 s each, 4 reload variants; not proven infeasible). Rev 7's "registers set the II" holds. The standalone loop was
  superseded by the fused bgelu (II 12).

## P29a SSM sigmoid (rev 7) — DONE (EW_DEFS -DSIGK): -0.61 ms measured (predicted -1.0: missed, explained)
- Kernel-only at II 12 (U 2), bit-exact, 499 cycles per chunk's 32 vectors vs 898 for the deployed C++ loop on one core. Rev 7's
  1,168-cycle baseline was round 9's; the timeline shows the SSM op -0.52 ms (= ~360 cycles x 216 chunks x 12), i.e. the
  bench's saving.
- Text: the EW core had 144 B free; the kernel-only form (no prologue) + r12/asmfit.py (compact body, last bundle still aligned)
  made it 350 B, less than the C++ it replaces.

## P43 pre-registered check (rev 7) — mlp3_a 166 -> 159 us per block (r11b -> r11d timelines), not 137 +- 8: MISS
P43b (the rest) not attempted.

## Generator lessons for rev 8 (all measured, in the skill)
- Exact register pressure in the scheduler (CP-SAT) reaches the IIs msched's IMS + sink could not allocate.
- Side-effect ops must sit in the last stage (the drain's garbage iterations run the earlier stages): a store of x to its
  final place in an early stage corrupted the 4 vectors after every run in situ (the bench did not check past the end).
- `.p2align 4` before the body inserts one nop bundle after the prologue: prologue ops then collide with body ops on itinerary
  ports (a vlda's QEY_WA vs a vfloor): bgelu II 11 had iteration 1 wrong. msched's checker now replays pads 0 / 1.
- A 2-bundle zero-overhead loop body runs once (>= 3 bundles).
- Output stream semantics (push / flush / flush.2d) measured; the stream pointer is exact to the byte after a flush.

## Not attempted this round
P24b, P35, P42 (the GEMM stack gave 256 B to the flat tables; EW stack unchanged), P27, P29b/c/d, P48-P50, P44 / P7-E, P14, the
host trio H1-H3, P43b.

DAVIS 10 videos on full12r12e (round end): depth 12 AJ 70.84 (OA 92.57), depth 10 69.90 (OA 91.02): identical to rounds 10-11.

# Round 13 (main agent), keyed by proposal id (2026-10-10)
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
Baseline full12r12e re-timed 50.05 ms. Best build **full12r13f: 49.21 ms** (depth 10 41.25, 8 33.44, 6 25.16), regress
BIT-IDENTICAL to full12r9a; DAVIS 70.84 / 69.90 as before. Details: npu-tapnext NOTES 18, RESULTS 22.

## T1 (rev 8) — DONE: mm_kstep's records as constexpr tables (-0.7 KB of program memory) + asmfit on gup / pan / bgelu / cpy32 (-208 B)
- Both bit-identical, timing unchanged (50.06 with asmfit alone). The research's map was right: 832 B of record builder; the
  records are `const + b x (c_out - acc_hi)` or a slot address, so 174 B of .data tables + ~60 B of code replace it.
- `ew_patch` / `gemm8` (rev 7's "1.4 KB of EW text"): not in the EW elf at all (the linker drops uncalled entry points); rev 8's
  correction stands. The EW core still has 160 B free: P29b needs its own recovery (it frees 1.4 KB).

## P54 (rev 8) — (1) answered from the .td, (2) not probed
- `VLDB_4x64_lo / hi` take only an index vector (`{'dst', 'src'}`): the address is absolute per lane, so there is no base
  register to fold the bias into; the "biased base" we have is the vsel'd (ab, cd) + bias vector the index adds already use.
  P56's "II 37 with the biased base" was therefore the unreachable branch; the fused loop reached **II 35** anyway (II 33 with
  9 free x) once Q was popped per iteration (no pinned constants but HI / LO / BASE) and -m loaded per iteration.

## P56 (rev 8) — DONE in the per-row-block form: the fused loop is worth -2.9 ms as priced, but the fallback and a lockstep take 2 ms back
- Built: gen_attk.py (96 ops: Q pops restarted per key block, K pops, two chains, vadd.f, scores to L1, ups + vadd.f for
  d = s - m, the LUT line with -m, push, and the flag by vlt.bf16 into r18 / r19 OR-ed in the last stage). II 35, U 1, 3
  stages, bit-exact over 144 bench cases; in situ BIT-IDENTICAL.
- Pre-registered step 0: 4.2 % flagged after chunk 0 on the captured scores (lazycount.py), 8.8-13.6 % per core incl. chunks 0
  / 22 on the NPU's own rows (lazycount3.py); in situ (`-DATTK_COUNT`) 8.6-10.9 % per core.
- Twins (RESULTS 22.2): fast loop only 47.14 ms (**-2.9 ms** vs 50.05: the P56 price was right for the loop); the exact test
  on every unit +1.35; the slow exp on ~10 % of units +1.35 (their own cycles are ~0.45): **the K / V multicast makes the 8
  columns advance in lockstep per chunk, so the slowest column's units set the wall**. `--l2-depth 8` recovered 0.36, the
  in-loop flag 0.33: production **-0.84 ms** (49.21). Attention op 10.95 -> 10.13 ms.
- Not built: the 36-iteration form (needs the 3D Q pop stream), the C++ fallback's rerun of QK^T (the stored scores serve
  instead). New facts for the model: ex_n = x_n (pops cost x registers: the research's attk.py had this right), scalar loads
  on DM_ADA, the ABI's r1 for scalar arguments of value-returning functions, the lockstep (sim15 should model the per-chunk
  max over columns rather than per-core sums).

## H3 (rev 7 / 8) — DONE as an option (step8 `HOST_HEADS=np`): heads 0.94 -> 0.72 ms median, p90 1.16 -> 0.75
- numpy + torch.erf with OPENBLAS_NUM_THREADS=1 (with OpenBLAS's own threads: 11.7 ms); max |d| tracks 7.6e-5 px. Below the
  predicted 0.15-0.25: the six small matmuls run at ~80 GFLOP/s single-threaded in OpenBLAS. H1 / H2 not attempted (the
  product runs in latency mode, where H1 cannot help).

## Not attempted this round
P29b, P55, P24b, P35, P53 (padding-aware CP), P54 (2), P57, P59, H1 / H2, P14, the DAVIS runs at depths 8 / 6.

# Round 14 (main agent), keyed by proposal id (2026-10-10)
All numbers: testbox, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
Baseline full12r13f re-timed 49.23-49.33 ms. Best build **full12r14f: 47.31 ms** (depth 10 39.84, 8 31.87, 6 24.43), regress
BIT-IDENTICAL to full12r9a. Details: npu-tapnext NOTES 19, RESULTS 23.

## P61 (rev 9) — DONE with the chunk-0 rescale skip (r14a): -0.63 ms (predicted P61 -0.33 +- 0.1)
- Last chunk: the fused loop for its 2 valid key blocks, the 4 invalid P panels stored as zero bytes (the device's bfp of +0
  is all-zero bytes; bench-checked against chunk_k). Chunk 0: no rescale (O and sums are +0). Together -0.63 vs predicted
  P61 alone -0.33: the chunk-0 rescale skip is worth ~-0.3 (round 13 had called it "no measurable gain").

## P63 (rev 9) — DONE without 3D pops: the loop is right, the in-situ gain is eaten by the fallback (MISS)
- r14/gen_attkb.py: 36 iterations per (tile, chunk), II 34 with banks; Q / K / -m / ring / push pointers by 2D modifiers and
  padda.2d / padds.2d (no 3D pop needed), -m rows in the K object's junk tail, a 7-slot score ring, flags stored per iteration.
- Bench (in-situ banks): warm unit 444 -> 261 cycles (-183; predicted ~-115 per unit). In situ: -0.04 .. -0.47 ms (bimodal),
  predicted -1.25 +- 0.6. Cause, measured: the flagged row blocks now cost 1,530 cycles each (first-pass rerun 393 + exact test
  164 + rescale 578 + exact rerun 393) and the lockstep multiplies it: twin full12r14cF128 (batched flags ignored) 46.15 ms vs
  48.24-48.73 production = 2.1-2.6 ms for ~0.35 ms of own cycles (x ~6). The attsim lockstep model should be re-fit on this twin.

## T3 (rev 9) — partly, different items paid off
- T3a in a safer form: aexp_push deleted; the slow path reruns the fused loop with the new max AND the clamp as an argument
  (71.5 SU = aexp_arg's bound). Needed: a flagged-but-not-moved lane has dh = 64.0 exactly, where the fast clamp (63.75)
  differs from aexp_push. -464 B. T3b (fold wrappers): -48 B only. T3c acc_zero: r14/zero2.s, one loop over both regions (64 +
  128 B per trip): -50 B and 864 -> 627 cycles. T3d qcopy: -48 B (smallest C++ form; cpy32 calls were larger at minsize).
  T3e attn_begin: zero2 for O + sums, -96 B. T3f not done. Static acc_hi map -112 B, rows8 by transpose -48 B.
- T2 (rev 8) V^T epilogue in asm: DONE (r14/gen_vt.py, II 13): -496 B of C++ for 278 B. This is what made P50 fit.

## P50 (rev 9) — DONE: -0.9 ms (predicted -1.2 +0.3 / -0.8: hit); the bench pre-registration (80-110 per row block) missed
- r14/gen_attv.py: two dim blocks per iteration (P popped once, V^T groups 2k / 2k+1), O straight into accumulator quarters,
  P / V restarts by walker pointers; `--sums`: the row sums in the same form. Per (tile, chunk) 1,316 -> 813 cycles = 136 per
  row block (the loop alone 87; the sums + calls the rest). Timeline: attention 10.13 -> 8.52 ms per step with P63 + P61.

## P60 (rev 9) — not built; single-TAP dealing is impossible
- Natural tile (c, i) -> (column s0 + i d, position p0 + i e) is one TAP per column for the qkv Q drain and the out_proj gather
  only if arithmetic in i; with a permutation per position only d = 0 solutions exist (exhaustive: r14/deal.py, deal2.py; the
  sum-of-squares identity forces d = 0). P60 needs per-piece TAPs (Q: an attention-specific memtile ld pattern; O: per-piece
  drains, ~36 shim tasks per column and pass, or a memtile reorder). Now worth more than priced: the F128 twin puts the fallback
  + lockstep at 2.1-2.6 ms.

## P62 (rev 9) — measured, not built (coordinator request: rescale and test separated)
- k_rb stamps (bench, one flagged row block): exact test 164 cycles; rescale_rb 578 cycles (incl. the call; 637 - 59 for the
  -m rows store); static: rescale_rb 608 B / 93 bundles at minsize (a 9-trip branch loop, no ZOL) + nk8::exp32 304 B / 61
  bundles; first-pass rerun 393, exact rerun 393. An -O2 rolled rescale_rb: 421 cycles for +96 B (not adopted: text).

## P29b, Q1, P55, P35 / P24b, P57 — not attempted
- P29b: the deployed chain inspected (r14/ssm_exp_deployed.s): 53.5 bundles per vector; the vec slot bounds an asm version at
  20 ops per vector; 2^n via vups.2x / vsrs.2x / vadd.16 is available in registers.

DAVIS 10 videos on full12r14f: depth 12 AJ 70.84 (OA 92.57), depth 10 69.90 (OA 91.02): identical to rounds 9-13.

# Round 15 (main agent), in progress (2026-10-10)
## Note for research pass 10 (P60 design input, NOT built or measured; main is not designing the lockstep fix)
- Observation from reading plan8: every op except attention is row-wise (norms, all GEMMs, SSM incl. its conv over frames:
  state per row). So a GLOBAL row permutation done on the host (x0 rows written in dealt order, final LN rows read back
  un-dealt) is invisible to every op except attention. At TILE granularity (48 rows = 6 whole row blocks: the lazy max is
  per row block, so row blocks must stay intact for bit-identity) with natural tile n stored at column n mod 8, position
  n div 8, each columns queries come from the whole image, and the K / V fill can still feed the key chunks in natural
# Round 15 (main agent), in progress (2026-10-10)
## Note for research pass 10 (P60 design input, NOT built or measured; main is not designing the lockstep fix)
- Observation from reading plan8: every op except attention is row-wise (norms, all GEMMs, the SSM incl. its conv over
  frames: state per row). So a GLOBAL row permutation done on the host (x0 rows written in dealt order, final LN rows read
  back un-dealt) is invisible to every op except attention. At TILE granularity (48 rows = 6 whole row blocks; the lazy max
  is decided per row block, so row blocks must stay intact for bit-identity) with natural tile n stored at column n mod 8,
  position n div 8, each column gets queries from the whole image, and the K / V fill still feeds the key chunks in natural
  order (bit-identical online softmax): plan8.attn's b TAP today [COLS, 3, 4, OBJ/2] strides [QKV_PER_HC, 3 OBJ, OBJ/2, 1]
  becomes [3, COLS, 4, OBJ/2] strides [3 OBJ, QKV_PER_HC, OBJ/2, 1] (outer dims swapped). No core text, no extra shim
  tasks, no memtile pattern change. Any tile bijection whose natural-order K / V address stays a <= 4-dim TAP (or a few TAPs
  per head fill) works, so the permutation can be chosen for balance (plain round robin puts natural tiles 16-23 = the lower
  image + the pad tiles 22-23 all in pass 1 = position 2: worth a better choice).
- Finer than tiles breaks K-chunk integrity (the K / V objects are the stored tiles' keys) unless the fill reassembles chunks.
- Host side: step8 x0 rows / final rows and the DAVIS loop need the permutation; regress must compare un-dealt rows.

## Round 15 measured so far (main; testbox Turbo, gpu_busy 0, a foreign CPU job (other-job worker, 100 % of one core) ran throughout; step8 time median of 60; every build regress BIT-IDENTICAL vs full12r9a)
| build | change | dispatch ms | predicted | verdict |
|---|---|---|---|---|
| full12r14f (re-timed) | round 14 best | 47.26 | | |
| full12r15a | + SSX: P29b, log_a + exp + 1 - a^2 as ONE asm loop (r15/gen_ssx.py, II 20 = vec bound, U 1, 7 stages) | 46.47 | main -0.75 +- 0.15; rev 10 -1.0 +- 0.07 | main hit (-0.79); rev 10 slight miss (the per-call prologue: 7 stages x 20 = 120 cycles; bench 497 vs deployed 1,057 cycles per chunk) |
| full12r15b | + CPY64: the SSM out copies as an II-1 asm copy | 46.27-46.30 | ANALYSIS rev 8 P52 baseline 798 (stale) | -0.18; PROF: the copies cost 477 in situ (bench 1 cycle per vector, in situ ~3.5: stores to the output ring, which MM2S reads, stall) |
| full12r15c | + CONVK: the SSM conv as one asm loop (r15/gen_conv.py, II 20 per column block) | 45.54-45.61 | -0.9 +- 0.3 | hit (-0.72) |
| full12r15d | + P60G (rev 10 P60g, plan8 only) | 44.49 (44.91 noisy run) | rev 10 -1.08 +- 0.22 | **hit (-1.06)**; attention 8.52 -> 7.38 ms per step |
- P60g per-block attention wall (gap + span, us, blocks 1-11): 620 570 720 630 590 590 650 650 680 510 500 = mean 610 (predicted
  614 +- 18; spread 220 vs ~213). Blocks 10 / 11 moved 30 / 10 us (predicted < 10).
- EW PROF (r15b, cycles per chunk, 216 chunks per op): in_wait 165, conv 1,125, gates 652, sigk 509, ssx 509, sqrt 493, recur 902,
  z 237, out_wait 31, copies 477 (x parts 289 + h / panels 217... split by stamps), rest 300 (record + releases + in_done), par 282.
  The deployed conv loop in a one-core bench (verbatim C++) runs 377 cycles; in situ the phase is 1,125: in_part calls + the
  in-situ addressing (P[] through the stack) + ring stalls. The SSM wall is code, not ring waits (in_wait 165, out_wait 31).
- Hardware facts: vld / vst immediate offsets are 4 bits x 64 B (-512 .. 448); padda immediates -512 .. 511; the EW cores' data
  region has 960 B free and bank 2 2,560 B (ssx_tab 704 B now there); a section-placed table must be non-const (section type
  conflict with the LUT tables); vmov x <- bm has its own itinerary (DM_RM_L0_PORT, latency 2), not in aie2p_lat.json.

## Round 15 (main), final verdicts keyed by proposal id (2026-10-11)
Best **full12r15m: 41.52 ms** at depth 12 (r14f re-timed 47.26), regress BIT-IDENTICAL vs full12r9a, DAVIS AJ 70.84 / 69.90 at
depth 12 / 10 (unchanged). Details: npu-tapnext NOTES 20, RESULTS 24. Testbox, Turbo, gpu_busy 0, a foreign CPU job throughout.
### P60g (rev 10) - DONE: -1.06 ms (predicted -1.08 +- 0.22: hit); per-block walls mean 610 us (predicted 614 +- 18: hit)
- plan8.py only (P60G=1, recorded in the build json). Checked before the build by walking both gathers piece by piece. Blocks 10 /
  11 moved 30 / 10 us (predicted < 10: small miss, inside the 17 us timeline resolution x 2).
### P29b (rev 10) - DONE: -0.79 ms (rev 10 -1.0 +- 0.07: miss by 0.2; main's own -0.75 +- 0.15: hit)
- One loop instead of two passes, II 20 = the vec bound, but 7 stages: the per-call prologue (120 cycles) + preamble keep it at
  497 cycles per chunk on one core (deployed 1,057 there; the rev-10 static 1,131). Constants reloaded per use from a 64 B-row
  table (13 bf16 constants > 12 x registers).
### S1 (rev 10) - DONE: -0.29 ms (S1 + S2 + S3 priced -0.74 together; S1 alone not priced)
- resc: II 8, bench 158 vs 353 cycles (-O2 C++); srow through the score ring; exp32 inlined. GEMM text 16,320 -> 16,192.
### S2 (rev 10) - NOT BUILT: the 768 B do not exist where rev 10 put them
- Bank 3 in full12r14f / r15: C object tail to 0x7D100, B fifo 0x7D100-0x7F500, tables, .aie.bank3 (sgelu_cd / serf_cd 2,304 B) to
  0x7FE80: 384 B free, plus 384 B in the K object's tail beyond the -m rows. A split layout needs a 3D modifier with a run-time
  jump, or a table moved out of bank 3. Please re-check the bank map (ldScripts_main_core_*_2.ld.script + llvm-nm -n).
### P29c (recurrence) - DONE: -0.84 ms (main -0.80 +- 0.15: hit; rev 10 -0.9: hit)
- U 4, II 11 (U 2 reached II 15: `gated` lives ~30 cycles). The h state in vector order on both sides (opaque to the DMA).
### P29d (conv half) - DONE: -0.72 ms (main -0.9 +- 0.3: hit). z half: ZRING ~-0.12 (z panels straight into the ring).
- The deployed C++ conv loop runs 377 cycles on one core but ~1,000 in situ: the in-situ context (P[] via the stack, in_part)
  made it 3x slower than its own bench. Price C++ phases from in-situ PROF stamps, not from a bench of the same source.
### P52 / out copies - partly: CPY64 -0.18 (ANALYSIS rev 8's 798-cycle baseline was a stale round-9 phase); NRMK / ZRING store
  straight into the output ring instead. Stores into the EW output ring (bank 0, MM2S reading) cost ~3.5 cycles per vector in situ
  (bench 1): open question for research - DMA arbitration on the bank, the stack in bank 0, or both? (ewo could be pinned to
  another bank with Buffer(mem_bank=)).
### P7-E step 1 (norm pass 2) - DONE: -0.40 ms after the slot-count fix; twin without the pass-2 math 40.62 (from 42.09): the math
  was 1.5 ms. With extra copy passes into the ring it was +0.6 (k2): direct stores were essential.
### New (main): slot-count loops -> reciprocal multiply: -1.05 ms (in_need / in_done -0.55, out_begin / out_end -0.5)
- ((x >> 8) * 21846) >> 16 == x / 768 (exhaustive for x < 2^16). The norms called them per unit: a large unpriced cost.
### Dead ends: NORM_SPLIT alone 0; in_copy by cpy64 -0.15..-0.2 (miss vs -0.30 +- 0.1, +400 B EW text); asm z loop no faster
  than C++ on one core (100 vs 97).
### Text: GEMM 16,192 / 16,384; EW 14,960 / 16,384 (r15k3: one non-template norm_rows freed ~1.6 KB); EW PROF fits again.
### Requests to research (rev 11)
- Re-price the remaining attention slow path after P60g + S1 with the correct bank-3 free space (S2 layout options).
- The output-ring store stall: model or probe proposal (bank 0 holds the stack and the output ring).
- Norms now ~90-120 us per op (port bound 60-90): price norm pass 1 + stats in asm, and the remaining SSM phases (gates 652,
  sigk 509, ssx 509 incl. a 120-cycle prologue, sqrt 493 mv-bound) from a fresh EW PROF of full12r15m (PROF fits again).

# Round 16 (main), final verdicts keyed by proposal id (2026-10-11)
Best **full12r16z: 39.94 ms** at depth 12 (min 39.83; r15m re-timed 41.50-41.52), regress BIT-IDENTICAL vs full12r9a, asmlint 0
errors on every build. Sweep d12 / 10 / 8 / 6: 40.30 / 33.84 / 26.86 / 20.18 ms (medians, one xclbin). Details: npu-tapnext NOTES
21, RESULTS 25. Testbox, Turbo, gpu_busy 0, a foreign CPU worker (100 % of one core) throughout. Your tools were adopted: the
build-lock patch is applied, every experiment after r16b ran through tools/try.sh (via npu-tapnext/r16/try16.sh, which adds the
current best's flags), asmlint ran on every build (clean, no false positives on the new kernels).
### SQK (main, rev 11 range -0.31..-0.35) - DONE: -0.18..-0.25 (miss)
- sqrt 493 -> 321 per chunk in situ (bench 311, static loop 247). The miss is the whole-call rule again: the ~30-bundle preamble
  (lane bases, masks, shift registers) + ret were not priced. Exponent bias folded into the table: 4 ops instead of 8 per vector.
### P-IN (rev 11 -0.25 +- 0.06) - DONE: -0.29 (hit), but only with the straddle copy by cpy64
- Inline pointers alone: conv phase 638 -> 497, step -0.09..-0.16; the 15 KB ring with 10.5 KB head params puts about every
  other chunk's part across the ring end, and in_part's minsize copy loop then cost ~10 cycles per vector.
### X1b (rev 11 -0.18 +- 0.03) - DONE: -0.28 (miss high)
- The x-copy phase was 432 per chunk in r15m (your 289 was r15b's); one 48-vector call: 432 -> 269.
### G1 (rev 11 -0.50 +- 0.07, phase 230-300) - DONE: ~-0.41 (slight miss)
- II 24, not 20 (the CP found none at 20-23 within 240 s); phase 655 -> 365. New hardware rule found on the way (please add to
  MACHINE): a dense 576-bit pop stream underflows at its 9th pop unless a fill at the same position follows every 8 pops (one-core
  probes r16/popprobe.s 6 / 7: pops 8-12 read zeros / half exponents, then resync). Restarted streams of <= 8 pops never see it.
### N1 (rev 11 -0.6 +- 0.25) - DONE: -0.16..-0.28 (miss)
- Twin NORM_TWIN=2 (pass 1 + statistics skipped) said 1.98 ms; the asm steps run at their bench speed in situ (161.5 cycles per
  unit incl. stamps, NPROF) but in_need waits 104.8 cycles per unit: the norms are partly input-bound now. Please re-price the
  norms with the port model: the remaining compute per unit is pass 2 (nrm_ln 230 / nrm_rms 156) + glue.
- Statistics as loop-carried 64-lane MACs on register pairs (msched now has opt-in carried uses; a generator-level "y" class);
  _begin / _step / _end keep constants and accumulators in registers across the 12 unit calls of a row block.
### New (main): SSQ (ssx + sqk fused, II 30) -0.12..-0.15; SIG10 (sigk II 10 / U 4) -0.05..-0.07
- The sqrt fusion needed II 30 (none <= 26: x pressure). FSX (gate_a's sigmoid into ssx, rev 11 -0.24) faces the same pressure
  inside SSQ now; sigk at II 10 runs 431 for 32 vectors on one core.
### Not built: S2, X1 full rotation, FSX, P52 by DMA
- X1 full: if the SSM area's base slides by one region per frame (B(t+1) = B(t) - S), x, x-1, x-2 are already in place for the
  next frame and the SSM writes only h and z (6 KB -> 3 KB per chunk, cp_x 269 -> 0, ~-0.38 ms by cycles); needs per-frame
  runtime sequences (step8 swaps the instruction bo per frame) and a wrap frame with copies every N frames. Please price / design
  the TAPs (fills stay single-TAP: the 7 streams stay S apart).
### Requests to research (rev 12)
- S2 on the stack (your rank 2) is next for main; please confirm the bank-0 sharing with aexp_ab in the CP model terms.
- The remaining SSM phases per chunk (r16eP): conv 440 (convk 251 on one core: where are ~190?), sigk 509, ssq 715, recur 313,
  z 264, rest 213, par 151, cp_h_pan 192, cp_x 269. And the norms' I/O-bound share (in_need 105 per unit).

# Round 17 (main), final verdicts keyed by proposal id (2026-10-11)
Best **full12r17d: 39.16 ms** at depth 12 (sweep median, min 39.13; time run 39.19; r16z re-timed 39.94-40.4 in the same
sessions), regress BIT-IDENTICAL vs full12r9a, asmlint 0 errors. Sweep d12 / 10 / 8 / 6: 39.16 / 32.78 / 26.29 / 19.79 ms; DAVIS
d12 70.84 (OA 92.57), d10 69.90 (91.02), unchanged. Testbox, Turbo, gpu_busy 0, the foreign CPU worker throughout. Details:
npu-tapnext NOTES 22, RESULTS 26. Dispatch times were bimodal (~0.17 ms clusters) this round: verdicts use p10 / median / mean of
200 interleaved iterations (step8 SPREAD) and the per-op timeline (r16z 40.07 -> r17d 39.28: attn -0.30, ssm -0.47, norms -0.06).
### S2 (rev 11/12 -0.38, -0.29..-0.45) - DONE: -0.31 (hit)
- On the caller's stack (the attkb frame would be overwritten by k_rb's), loads by offset + stores with post-modify (one pointer
  freed by dropping the per-iteration flag stores: the call returns the OR, a flagged row block's own flag comes from its max slot,
  exactly). II 36 (34: none; 35: two pad-replay failures); the max reads the ring-reloaded scores (x pressure). One-core: warm
  +76 cycles per call, flagged calls -777..-1,248, chunk 0 -453, last chunk -813. GEMM text 16,336 / 16,384.
- Your bank-0 pairing constraint was not needed: the CP model already carries the banks (slot loads / stores bank 0 with the LUT
  loads) and found II 36 with them.
### AL1 (rev 12 -0.21 +- 0.05) - DONE: -0.05..-0.15 (miss)
- 1 KB padding HUNG the step (1 timeout): the shim -> core packet BD completes per 768 B slot, so a fill group must stay a slot
  multiple. Built with 3 KB multiples (SSM 12,288 B per head, RMS 3,072, LN 6,144) and the straddle path compiled out.
- CPROF (conv split, r17d config): convk 280 in situ incl. stamps (bench 251), pointers + mapping 101: the conv gap was the straddle.
### NPIPE (rev 12 -0.55, -0.54..-0.56) - DONE: G 2 +0.07, G 4 / 6 -0.05..-0.15 (miss)
- NPROF with out_begin stamped (your NIO request): r16z in_need 101.5, **out_begin 45.0** per unit (ewio: 147 RMS / 90 LN);
  NPIPE_G=2: in_need 61.0, out_begin 46.6. The norms are not output-bound at the measured waits; R_out 2.35 does not fit these
  stamps. Each group pays n1k _end / _begin (~45 cycles; nrm_* use every dm and x register).
### X1 (rev 12 mod 4: -0.34 alone, -0.26 after AL1) - DONE: rotating frames -0.33 (hit); deployed as your mod-4 design
- Built first as a sliding area (N phases + a wrap frame with copies; -0.33 per rotating frame, wrap = r16z, ACT +232 MB at N 8),
  then replaced by your mod-4 layout: the yx drain splits y | x in one TAP ([2, 12, RB, XB], under P23 the C objects arrive in
  k-step order), header word 2 = 1 + r, four params copies, output [h lo, h hi, z] 3 KB. EW text needed the XROT / AL1 builds to
  compile the copy and straddle paths out (16,384 overflow by 64-80 B otherwise).
### Your popprobe question (FIFO-POP9 on probes 2 / 3)
- Probes 2 and 3 dumped only the lf0 (A) stream's 8 pops; the lf1 (B) stream's 16 dense pops were never stored or checked. So
  "correct" in round 16 was about lf0; FIFO-POP9 firing on lf1 there is a true positive, not a false one.
### Not built: S3, V1, K1, K2 (GEMM text 48 B free after S2), FSX, EWO bench, P35 / P24b
### Requests to research (rev 13)
- Re-fit ewio with the NPROF stamps above (in_need 101.5 / out_begin 45.0 at r16z; 61.0 / 46.6 with NPIPE_G=2) and the NPIPE G
  series; what is the input-side lever (n_in 3 ops: fold D1 + D2 / D3 + D4 partials upstream?).
- GEMM program memory: where can 200-400 B come from so S3 / V1 / K1 / K2 fit (r17d elf, 48 B free)?
- The SSM now outputs 3 KB per chunk: re-rank its compute levers (chunk ~2,943 cycles in situ: convk 280, pointers 101, rest 2,562).
