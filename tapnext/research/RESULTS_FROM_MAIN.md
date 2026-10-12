# Results from the optimisation rounds (round 6), keyed by proposal id

All numbers: test machine, Turbo, gpu_busy 0, depth 12, pyxrt kernel time unless said otherwise. MEASURED unless marked.
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
`r6/capture.py` (CPU bf16 model, the original 30-frame test clip (private, not included), 3 frames, 96 random token rows x full feature dim per block and site, all 12
blocks -> build/acts_davis3.npz) + `r6/numh.py ranges|gelu` + `r6/emu.py` (bit-level models of the nk8 ops; checked
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

## GELU by lookup table (own proposal) — DONE, -6.0 ms, AJ 70.83
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

# Round 7 (optimisation rounds), keyed by proposal id
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked. Cycles at 1.81 GHz (P10 below).

## P10 (clock part) — DONE: 1.81 GHz in Turbo
`tapnext/npu/r7/clk.py` + `clk.cc`: pure-register hardware loops (scalar LCG 4 bundles/trip, dependent bf16 vmac 3
bundles/trip; no loads or stores in the loop, so nothing can stall) with a runtime trip count, host-timed at 0..32M
trips: slope 1.8086 / 1.8076 GHz (fit on medians), ratio of the two loops 1.3329 vs 4/3. With P15's timer (below) the
same loops read exactly bundles x trips (+48 / +59 cycles of call overhead). Agrees with the driver table (1.8 GHz).
Default mode not measured (the box is in Turbo; changing it needs sudo).

## P15 — DONE, verdict: the processor-bus enable fixes read_tm
`npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=r)` as the first op of the runtime sequence (from the
IRON sequence function: `from aie.dialects.aiex import npu_maskwrite32`), `__builtin_aie2p_sched_barrier()` around
`read_tm(0x340F8)` (objdump: each `lda.tm` alone in its bundle). One core (clk.py --stamp): timer = bundles x trips
exactly. 24 GEMM cores in the full step (design8 with `--kdefs=...,-DPROF`): no hang, 3 dispatches. 0 timeouts.

## P16-lite — DONE (stamps, not the counters): `tapnext/npu/r7/prof.py`
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

## P12 (attention) — partly done; see tapnext/npu/RESULTS.md 16.2
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
`tapnext/npu/r7/gen_mm.py` emits `mm_core` (exact-cycle straight-line schedule: 4 accumulator chains over 2 column
blocks, A held in ex0-7, B on ldb through a 4-register conveyor, accumulator quarters on lda / st, EDF placement,
built-in conveyor checks). B packed [q][4 col blocks] per column-block pair on the host (pack8 B_PAIR4).
225 bundles per call vs 321; directed tests and the full-step regress BIT-IDENTICAL; step 73.10 -> 71.19 ms.
The prediction (k-step 0.95 us, 62.2 ms) was for the full 2x2 design; this first schedule's EXTRAPOLATED gain was
~-7 ms and only -1.9 was measured. Not explained yet: the in-situ profile of the asm call is the next action
(PROF + MM_ASM does not fit the program memory together with the attention code: use a GEMM-only test).
Group boundaries cost 12 cycles per 32 MACs (16 serialised stores + loads on 5 accumulator registers).

# Round 8 (optimisation rounds), keyed by proposal id
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz. Details:
tapnext/npu/RESULTS.md §17, NOTES.md §13.

## P18 follow-up — the -1.9 vs -7 ms gap, MEASURED: L1 bank conflicts inside mm_core
g11 -DPROF in situ: C++ 361, asm 337 cycles per call (PROF window ~+40). r8/mmb.py (one core, no DMA): 238 conflict
free, +33 accumulator load + store in one bank (as deployed), +14 A and B in one bank, +42 B with C. The rev-3
attribution (+2.6 wrapper, +3.4 banks) agrees; the stamp prediction matched (337; gaps 47 / 66).

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

# Round 9 (optimisation rounds), keyed by proposal id
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz. Details:
tapnext/npu/RESULTS.md §18, NOTES.md §14. Step 67.89 (r8c) -> 58.14 ms (full12r9s), d10 48.5 ms, AJ 70.84 / 69.90.

## P20 (rev 4) — DONE, all three stages; -6.4 ms measured vs -8.2 predicted
- (b) first, as a C++ loop with the locks inside (mm_kstep): -2.0 ms (g11 35.72 -> 33.93).
- (a) pieces: affine layout (pair 0 rb < 12 in C slot rb, rb >= 12 in acc_hi 0-5, pair 1 in acc_hi 6 + rb), not
  p20map's table (a table-indirect epilogue cost 0.8 us per tile; strided passes keep it at 0.72 us). The bankmodel rule
  reproduces the one-core bench (static + model 2,127 vs 2,200 measured). In situ -0.9 ms (prediction -2.6).
- (c) r9/gen_ks.py: period 192 bundles as specified, record stream via post-increment (padds/padda immediates must be
  multiples of 64: no per-frame bump of a 40-B record), m0-m3 shared by both cursors, dummy 10th stripe with acq.cond.
  -3.5 ms, bit-identical. Prediction for g11 PROF 1,950-2,150 cycles per k-step: **measured 2,588 (down_a) - 3,320 (out)**.
- Why the gap: the k-step is now A-port-bound. 10,368 B of A per k-step at ~4 B/cycle = 2,592 cycles = the measured
  2,588. sim11 should add a per-core A-port term (and a lower rate for K = 768, nt 4-8 ops: ~3.1 B/cycle).
- Two dispatch timeouts on the way (loop start alignment + uninitialised r25; acq with value 0 = acquire_equal 0).

## P26 — DONE differently: exact sqrt by a 256-entry mantissa table (int16 LUT), not 32 segments
Bit-exact RNE on every bf16 >= 2^-9 (probe); 28.3 vs 65.7 cycles per 32 lanes; in situ 1,093 -> 493 cycles per chunk.

## P28 — DONE on both gates at once (one asymmetric table [-12, 8) in 1/8 segments, 160 segments, 2.5 KB per copy)
Emulated on real gates: 5% 1-ulp vs torch (nk8 19%), a 8.5% / mult 6.6% != torch (nk8 13% / 40%); DAVIS d12 70.84,
d10 69.90 (r8c 70.78 / 69.83). 27.4 cycles per 32 lanes (75 before). With sqrt: -2.8 ms (predicted -1.0 -1.1 -0.8 = -2.9).
CPU-model control: any ulp-level SSM change (nk8, tables, random flips) gives the same track noise.

## P27 — NOT DONE (memory); what was done instead
EW L1 had no 2 x 1.7 KB left after the sigmoid / sqrt tables (they took every free byte + 2.5 KB of a compacted arena).
Instead today's exp_a2 ops split into 5 pipelined passes (bit-identical, 63.5 -> ~44 bundles per vector): -0.22 ms.
A range-reduced 2^f table was as accurate as exp32a and not faster (measured). P27 next, after freeing EW memory.

## P5a, P7, P29, P33 — not attempted this round

# Round 10 (optimisation rounds), keyed by proposal id — in progress (2026-10-09)
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz.

## The A-port-bound claim of round 9 — REFUTED by its own data (early note)
Per-tile distribution of the round-9 g11 PROF records (tapnext/npu/r10/gdist.py on build/g11sp.npy): the **median** busy per
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
- **Router behaviour found:** IRON 1.4.4's pathfinder dropped a multicast destination (no error) when unicast packet flows share the
  same two DMA master ports (ids 1 -> DMA0, 3 -> DMA5, 4 -> both gave DMA5 no rule for id 4; reproduced in isolation,
  tapnext/npu/r10/pf1.mlir vs pf2.mlir; order-independent). That first version hung one dispatch (timeout 1 of the round;
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

## Round 10 summary (optimisation rounds)
- Step depth 12: 58.11 -> **53.89 ms** (full12r10r), depth 10 48.5 -> 45.4, 8 38.9 -> 36.0, 6 29.2 -> 27.1; all bit-identical to
  round 9; DAVIS 10 videos d12 70.84, d10 69.90 (unchanged). One timeout (router behaviour above), recovered.
- Verdicts: P5r + P5a DONE (-1.35 vs -1.6 predicted); P23 DONE (-0.85 vs -1.2); P5 DONE as runtime memtile tasks (-2.0 vs -1.7);
  P10 (memtile->core rate) answered: 7.95 B/cycle. Not attempted this round: P38, P7 / P7a, P27, P42, P35, P24b, P34, P37, P14.
- Parked items (two-channel A, neighbour-memory A, wider N tiles): agree, 0 ms today.
- New lever measured for the analysis passes: GELU epilogues ~3.3 ms per step on the GEMM critical path (see above); a hand
  modulo schedule is the next item. sim12 may want the r10r timeline: tapnext/npu/build/tl_full12r10r_d12.json.

# Round 11 (optimisation rounds), keyed by proposal id (2026-10-10)
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked; cycles at 1.81 GHz. Baseline full12r10r
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
4. **Itinerary ports are structural hazards** the scheduler must reserve (aie2p_isa_timing.json itin stages): DM_ADA (vldb.4x64 AND
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

## Host part of the step (task item) — our code, plus thread wake-up and background load
r11/hostprof.py (8 torch threads): conv 0.48-1.1, queries 0.12, X0 packing 0.64 -> **0.14 ms** (HOST_FAST=2: torch strided
copy into the mapped ACT bo, byte-identical), write/sync, read 0.08, heads 0.6-1.0 ms. The variance is the OpenMP pool waking
after the 54 ms idle plus other processes (load average 1.3-2.9; a background worker holds one core at 100 %). Heads on one thread
make the NEXT conv slower (pool re-spin): not used. Whole step - dispatch now ~2.0-2.2 ms.

## Not attempted this round
P27 / P29 (SSM), P7a / P44 / P7-E (norms; the rev-6 advice to park the full fold accepted), P48-P50 (attention), P35, P24b,
P38, P14.

DAVIS 10 videos on full12r11d (round end): depth 12 AJ 70.84 (OA 92.57), depth 10 69.90 (OA 91.02): identical to round 10.

# Round 12 (optimisation rounds), keyed by proposal id (2026-10-10)
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
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

# Round 13 (optimisation rounds), keyed by proposal id (2026-10-10)
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
Baseline full12r12e re-timed 50.05 ms. Best build **full12r13f: 49.21 ms** (depth 10 41.25, 8 33.44, 6 25.16), regress
BIT-IDENTICAL to full12r9a; DAVIS 70.84 / 69.90 as before. Details: npu-tapnext NOTES 18, RESULTS 22.

## T1 (rev 8) — DONE: mm_kstep's records as constexpr tables (-0.7 KB of program memory) + asmfit on gup / pan / bgelu / cpy32 (-208 B)
- Both bit-identical, timing unchanged (50.06 with asmfit alone). The analysis map was right: 832 B of record builder; the
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
  instead). New facts for the model: ex_n = x_n (pops cost x registers: the analysis-pass attk.py had this right), scalar loads
  on DM_ADA, the ABI's r1 for scalar arguments of value-returning functions, the lockstep (sim15 should model the per-chunk
  max over columns rather than per-core sums).

## H3 (rev 7 / 8) — DONE as an option (step8 `HOST_HEADS=np`): heads 0.94 -> 0.72 ms median, p90 1.16 -> 0.75
- numpy + torch.erf with OPENBLAS_NUM_THREADS=1 (with OpenBLAS's own threads: 11.7 ms); max |d| tracks 7.6e-5 px. Below the
  predicted 0.15-0.25: the six small matmuls run at ~80 GFLOP/s single-threaded in OpenBLAS. H1 / H2 not attempted (at
  one frame per dispatch H1 cannot help).

## Not attempted this round
P29b, P55, P24b, P35, P53 (padding-aware CP), P54 (2), P57, P59, H1 / H2, P14, the DAVIS runs at depths 8 / 6.

# Round 14 (optimisation rounds), keyed by proposal id (2026-10-10)
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
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

## P62 (rev 9) — measured, not built (rescale and test measured separately)
- k_rb stamps (bench, one flagged row block): exact test 164 cycles; rescale_rb 578 cycles (incl. the call; 637 - 59 for the
  -m rows store); static: rescale_rb 608 B / 93 bundles at minsize (a 9-trip branch loop, no ZOL) + nk8::exp32 304 B / 61
  bundles; first-pass rerun 393, exact rerun 393. An -O2 rolled rescale_rb: 421 cycles for +96 B (not adopted: text).

## P29b, Q1, P55, P35 / P24b, P57 — not attempted
- P29b: the deployed chain inspected (r14/ssm_exp_deployed.s): 53.5 bundles per vector; the vec slot bounds an asm version at
  20 ops per vector; 2^n via vups.2x / vsrs.2x / vadd.16 is available in registers.

DAVIS 10 videos on full12r14f: depth 12 AJ 70.84 (OA 92.57), depth 10 69.90 (OA 91.02): identical to rounds 9-13.

# Round 15 (optimisation rounds), in progress (2026-10-10)
## Note for analysis pass 10 (P60 design input, NOT built or measured)
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

## Round 15 measured so far (test machine Turbo, gpu_busy 0, a background CPU job (100 % of one core) ran throughout; step8 time median of 60; every build regress BIT-IDENTICAL vs full12r9a)
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
  conflict with the LUT tables); vmov x <- bm has its own itinerary (DM_RM_L0_PORT, latency 2), not in aie2p_isa_timing.json.

## Round 15 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r15m: 41.52 ms** at depth 12 (r14f re-timed 47.26), regress BIT-IDENTICAL vs full12r9a, DAVIS AJ 70.84 / 69.90 at
depth 12 / 10 (unchanged). Details: npu-tapnext NOTES 20, RESULTS 24. Test machine, Turbo, gpu_busy 0, a background CPU job throughout.
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
  jump, or a table moved out of bank 3. To re-check: the bank map (ldScripts_main_core_*_2.ld.script + llvm-nm -n).
### P29c (recurrence) - DONE: -0.84 ms (main -0.80 +- 0.15: hit; rev 10 -0.9: hit)
- U 4, II 11 (U 2 reached II 15: `gated` lives ~30 cycles). The h state in vector order on both sides (opaque to the DMA).
### P29d (conv half) - DONE: -0.72 ms (main -0.9 +- 0.3: hit). z half: ZRING ~-0.12 (z panels straight into the ring).
- The deployed C++ conv loop runs 377 cycles on one core but ~1,000 in situ: the in-situ context (P[] via the stack, in_part)
  made it 3x slower than its own bench. Price C++ phases from in-situ PROF stamps, not from a bench of the same source.
### P52 / out copies - partly: CPY64 -0.18 (ANALYSIS rev 8's 798-cycle baseline was a stale round-9 phase); NRMK / ZRING store
  straight into the output ring instead. Stores into the EW output ring (bank 0, MM2S reading) cost ~3.5 cycles per vector in situ
  (bench 1): open question - DMA arbitration on the bank, the stack in bank 0, or both? (ewo could be pinned to
  another bank with Buffer(mem_bank=)).
### P7-E step 1 (norm pass 2) - DONE: -0.40 ms after the slot-count fix; twin without the pass-2 math 40.62 (from 42.09): the math
  was 1.5 ms. With extra copy passes into the ring it was +0.6 (k2): direct stores were essential.
### New (main): slot-count loops -> reciprocal multiply: -1.05 ms (in_need / in_done -0.55, out_begin / out_end -0.5)
- ((x >> 8) * 21846) >> 16 == x / 768 (exhaustive for x < 2^16). The norms called them per unit: a large unpriced cost.
### Dead ends: NORM_SPLIT alone 0; in_copy by cpy64 -0.15..-0.2 (miss vs -0.30 +- 0.1, +400 B EW text); asm z loop no faster
  than C++ on one core (100 vs 97).
### Text: GEMM 16,192 / 16,384; EW 14,960 / 16,384 (r15k3: one non-template norm_rows freed ~1.6 KB); EW PROF fits again.
### Open questions (rev 11)
- Re-price the remaining attention slow path after P60g + S1 with the correct bank-3 free space (S2 layout options).
- The output-ring store stall: model or probe proposal (bank 0 holds the stack and the output ring).
- Norms now ~90-120 us per op (port bound 60-90): price norm pass 1 + stats in asm, and the remaining SSM phases (gates 652,
  sigk 509, ssx 509 incl. a 120-cycle prologue, sqrt 493 mv-bound) from a fresh EW PROF of full12r15m (PROF fits again).

# Round 16 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r16z: 39.94 ms** at depth 12 (min 39.83; r15m re-timed 41.50-41.52), regress BIT-IDENTICAL vs full12r9a, asmlint 0
errors on every build. Sweep d12 / 10 / 8 / 6: 40.30 / 33.84 / 26.86 / 20.18 ms (medians, one xclbin). Details: npu-tapnext NOTES
21, RESULTS 25. Test machine, Turbo, gpu_busy 0, a background CPU worker (100 % of one core) throughout. The analysis tools were adopted: the
build-lock patch is applied, every experiment after r16b ran through tools/try.sh (via tapnext/npu/r16/try16.sh, which adds the
current best's flags), asmlint ran on every build (clean, no false positives on the new kernels).
### SQK (main, rev 11 range -0.31..-0.35) - DONE: -0.18..-0.25 (miss)
- sqrt 493 -> 321 per chunk in situ (bench 311, static loop 247). The miss is the whole-call rule again: the ~30-bundle preamble
  (lane bases, masks, shift registers) + ret were not priced. Exponent bias folded into the table: 4 ops instead of 8 per vector.
### P-IN (rev 11 -0.25 +- 0.06) - DONE: -0.29 (hit), but only with the straddle copy by cpy64
- Inline pointers alone: conv phase 638 -> 497, step -0.09..-0.16; the 15 KB ring with 10.5 KB head params puts about every
  other chunk's part across the ring end, and in_part's minsize copy loop then cost ~10 cycles per vector.
### X1b (rev 11 -0.18 +- 0.03) - DONE: -0.28 (miss high)
- The x-copy phase was 432 per chunk in r15m (289 was r15b's); one 48-vector call: 432 -> 269.
### G1 (rev 11 -0.50 +- 0.07, phase 230-300) - DONE: ~-0.41 (slight miss)
- II 24, not 20 (the CP found none at 20-23 within 240 s); phase 655 -> 365. New hardware rule found on the way (recorded in
  MACHINE): a dense 576-bit pop stream underflows at its 9th pop unless a fill at the same position follows every 8 pops (one-core
  probes r16/popprobe.s 6 / 7: pops 8-12 read zeros / half exponents, then resync). Restarted streams of <= 8 pops never see it.
### N1 (rev 11 -0.6 +- 0.25) - DONE: -0.16..-0.28 (miss)
- Twin NORM_TWIN=2 (pass 1 + statistics skipped) said 1.98 ms; the asm steps run at their bench speed in situ (161.5 cycles per
  unit incl. stamps, NPROF) but in_need waits 104.8 cycles per unit: the norms are partly input-bound now. To re-price: the
  norms with the port model: the remaining compute per unit is pass 2 (nrm_ln 230 / nrm_rms 156) + glue.
- Statistics as loop-carried 64-lane MACs on register pairs (msched now has opt-in carried uses; a generator-level "y" class);
  _begin / _step / _end keep constants and accumulators in registers across the 12 unit calls of a row block.
### New (main): SSQ (ssx + sqk fused, II 30) -0.12..-0.15; SIG10 (sigk II 10 / U 4) -0.05..-0.07
- The sqrt fusion needed II 30 (none <= 26: x pressure). FSX (gate_a's sigmoid into ssx, rev 11 -0.24) faces the same pressure
  inside SSQ now; sigk at II 10 runs 431 for 32 vectors on one core.
### Not built: S2, X1 full rotation, FSX, P52 by DMA
- X1 full: if the SSM area's base slides by one region per frame (B(t+1) = B(t) - S), x, x-1, x-2 are already in place for the
  next frame and the SSM writes only h and z (6 KB -> 3 KB per chunk, cp_x 269 -> 0, ~-0.38 ms by cycles); needs per-frame
  runtime sequences (step8 swaps the instruction bo per frame) and a wrap frame with copies every N frames. To price / design:
  the TAPs (fills stay single-TAP: the 7 streams stay S apart).
### Open questions (rev 12)
- S2 on the stack (rank 2) is next; the bank-0 sharing with aexp_ab is to be confirmed in CP model terms.
- The remaining SSM phases per chunk (r16eP): conv 440 (convk 251 on one core: where are ~190?), sigk 509, ssq 715, recur 313,
  z 264, rest 213, par 151, cp_h_pan 192, cp_x 269. And the norms' I/O-bound share (in_need 105 per unit).

# Round 17 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r17d: 39.16 ms** at depth 12 (sweep median, min 39.13; time run 39.19; r16z re-timed 39.94-40.4 in the same
sessions), regress BIT-IDENTICAL vs full12r9a, asmlint 0 errors. Sweep d12 / 10 / 8 / 6: 39.16 / 32.78 / 26.29 / 19.79 ms; DAVIS
d12 70.84 (OA 92.57), d10 69.90 (91.02), unchanged. Test machine, Turbo, gpu_busy 0, the background CPU worker throughout. Details:
npu-tapnext NOTES 22, RESULTS 26. Dispatch times were bimodal (~0.17 ms clusters) this round: verdicts use p10 / median / mean of
200 interleaved iterations (step8 SPREAD) and the per-op timeline (r16z 40.07 -> r17d 39.28: attn -0.30, ssm -0.47, norms -0.06).
### S2 (rev 11/12 -0.38, -0.29..-0.45) - DONE: -0.31 (hit)
- On the caller's stack (the attkb frame would be overwritten by k_rb's), loads by offset + stores with post-modify (one pointer
  freed by dropping the per-iteration flag stores: the call returns the OR, a flagged row block's own flag comes from its max slot,
  exactly). II 36 (34: none; 35: two pad-replay failures); the max reads the ring-reloaded scores (x pressure). One-core: warm
  +76 cycles per call, flagged calls -777..-1,248, chunk 0 -453, last chunk -813. GEMM text 16,336 / 16,384.
- The proposed bank-0 pairing constraint was not needed: the CP model already carries the banks (slot loads / stores bank 0 with the LUT
  loads) and found II 36 with them.
### AL1 (rev 12 -0.21 +- 0.05) - DONE: -0.05..-0.15 (miss)
- 1 KB padding HUNG the step (1 timeout): the shim -> core packet BD completes per 768 B slot, so a fill group must stay a slot
  multiple. Built with 3 KB multiples (SSM 12,288 B per head, RMS 3,072, LN 6,144) and the straddle path compiled out.
- CPROF (conv split, r17d config): convk 280 in situ incl. stamps (bench 251), pointers + mapping 101: the conv gap was the straddle.
### NPIPE (rev 12 -0.55, -0.54..-0.56) - DONE: G 2 +0.07, G 4 / 6 -0.05..-0.15 (miss)
- NPROF with out_begin stamped (NIO request): r16z in_need 101.5, **out_begin 45.0** per unit (ewio: 147 RMS / 90 LN);
  NPIPE_G=2: in_need 61.0, out_begin 46.6. The norms are not output-bound at the measured waits; R_out 2.35 does not fit these
  stamps. Each group pays n1k _end / _begin (~45 cycles; nrm_* use every dm and x register).
### X1 (rev 12 mod 4: -0.34 alone, -0.26 after AL1) - DONE: rotating frames -0.33 (hit); deployed as the mod-4 design
- Built first as a sliding area (N phases + a wrap frame with copies; -0.33 per rotating frame, wrap = r16z, ACT +232 MB at N 8),
  then replaced by the mod-4 layout: the yx drain splits y | x in one TAP ([2, 12, RB, XB], under P23 the C objects arrive in
  k-step order), header word 2 = 1 + r, four params copies, output [h lo, h hi, z] 3 KB. EW text needed the XROT / AL1 builds to
  compile the copy and straddle paths out (16,384 overflow by 64-80 B otherwise).
### The popprobe question (FIFO-POP9 on probes 2 / 3)
- Probes 2 and 3 dumped only the lf0 (A) stream's 8 pops; the lf1 (B) stream's 16 dense pops were never stored or checked. So
  "correct" in round 16 was about lf0; FIFO-POP9 firing on lf1 there is a true positive, not a false one.
### Not built: S3, V1, K1, K2 (GEMM text 48 B free after S2), FSX, EWO bench, P35 / P24b
### Open questions (rev 13)
- Re-fit ewio with the NPROF stamps above (in_need 101.5 / out_begin 45.0 at r16z; 61.0 / 46.6 with NPIPE_G=2) and the NPIPE G
  series; what is the input-side lever (n_in 3 ops: fold D1 + D2 / D3 + D4 partials upstream?).
- GEMM program memory: where can 200-400 B come from so S3 / V1 / K1 / K2 fit (r17d elf, 48 B free)?
- The SSM now outputs 3 KB per chunk: re-rank its compute levers (chunk ~2,943 cycles in situ: convk 280, pointers 101, rest 2,562).

# Round 18 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r18m: 35.20 ms** at depth 12 (sweep median, min 34.80; r17d 39.16), d10 29.11, d8 23.30, d6 17.53; regress
BIT-IDENTICAL vs full12r9a, asmlint 0; DAVIS d12 70.84 (OA 92.57), d10 69.90 (91.02), unchanged. Test machine, Turbo, gpu_busy 0, the
background CPU worker throughout. Timing: r18/ab.py (both builds in one process, interleaved blocks, 300-600 kept each). Details:
npu-tapnext NOTES 23, RESULTS 27. Timeouts: 1 (V2T: `add r1, r1, #64` assembled as #-64).
### ROT (rev 14, g2f -3.84 +- 0.4) - DONE as g2a: -3.27 (median) / -3.34 (p10) / -3.35 (mean) vs r18j (slight miss; -3.46 uncalibrated)
- The bench passed for g2f (36/36, warm 1,431 vs pre-registered 1,390-1,405), but g2f in situ was DIFFERENT and varied run to run
  (rms_rel 0.15-0.25, tracks <= 0.13 px). Replacing rel.cond by a plain `rel #52, r26` changed nothing. Cause: `acq.cond #53` sits
  ONE bundle before the A fill (prologue line 31 -> 32; constraint ACQ in [AF-6, AF-1] in solve()); the bench pre-releases all nine
  A slots so the acquire never blocks. Deployed mm_ks has 6 bundles. r18/gen_rot.py (the analysis generator + env ROT_ACQ_GAP / ROT_REL_GAP)
  re-solved with ACQ <= AF - 6 and REL >= A7 + 8 (g2a): OPTIMAL in 19 s, 1,383 static, 0 bank stalls on the r18 placement, bench
  36/36, in situ BIT-IDENTICAL. rel.cond itself works (now deployed).
- KS-P38 around mm_rot (per k-step, 6,240 tiles): TRUE 1,603 (pre-reg 1,610 +- 45, hit), MEMORY_STALL 167.5 (212 +- 40, low edge),
  LOCK 13.4 (13). acc_hi moved to 0x71980 with rot_rec (ksc_check caught it). GEMM text 15,536.
- Note for the bench: mm8_rot.inc includes "build/mm_rot.h", which -I npu-tapnext resolves to tapnext/npu/build/mm_rot.h (the last
  design build's variant) before r19/build: a g2a bench first ran with g2f's border (0/36).
### KSC (rev 13 -270..-290 B) - DONE: -288 B (hit), and -0.14 ms (the per-call check ~16 cycles per k-step; predicted ~0)
### P35 (rev 13 -0.44 +- 0.1) - DONE: -0.10..-0.13 (miss); B locks by id (0 / 1), B buffers by address, even count per tile
### N-IO (rev 13 -0.45..-0.55 / -0.5..-0.7) - C++ form only (NIOC: ring state in locals, helpers inline): -0.17..-0.23
- tcall static -0.55; the miss is waits (pass 1 on input). NIO2 stamps (pre-registered pass-2 unit RMS 325 +- 40, LN 449 +- 60;
  nrm RMS 136-193, LN 210-309): r18d RMS 341.8 / nrm 144.8, LN 428.4 / 229.5 (hit); pass-1 unit RMS 302.0, LN 287.5. After NIOC
  (r18h, ZK off for text): pass-2 unit RMS 302.0, LN 390.7; pass-1 RMS 293.2, LN 248.6. The asm N-IO is not built.
### ZK (rev 13 -0.18..-0.21) - DONE in r18b: -0.14..-0.19 (hit)
### CGF (rev 13 -0.10..-0.13) - DONE: -0.05..-0.06 (miss; tcall static -42 per chunk said -0.06)
### V1 - partly: V2T (attv / attvs once per chunk for both tiles, O jump by d2 / d3): -0.06..-0.08 (main pre-reg -0.06..-0.10, hit)
### E2 DMAB - DONE (measured): same bank +58 (loads) / +59 (stores) per 256-access segment at 8.07 B/cycle; other bank 0
### E3 KS-P38 - DONE (measured): mm_ks MEMORY 173.7, LOCK 13.3, TRUE 2,046.7 vs timer 2,056.7 per k-step
### Not built: S3, K1 (a C++ two-chains prep: -28 per tile-chunk, +96 B: dropped), K2, NS, ssq two chunks, FSX, IV32, g0f / g1f
### Open questions (rev 15)
- ROT g0f / g1f with the acquire gap (and the PWRAP / PADDA probes); a bench mode with a DMA-fed A ring that can see lock timing.
- The B / A bank layout after ROT (168 MEMORY_STALL cycles per k-step left): which hand-built B ring placement, and its price.
- N-IO in asm vs the measured NIO2 numbers after NIOC; a timeline of r18m to re-rank everything after ROT.

# Round 19 (optimisation rounds), in progress (2026-10-11)
## Timeline of full12r18m (MEASURED: test machine Turbo, gpu_busy 0, background CPU worker running; step8 timeline, 3 runs, median; the
## JSONs are tapnext/npu/build/tl_r18m_run{1,2,3}.json; summary by tapnext/npu/r19/tlsum.py; dispatch 34.8-35.1 ms)
span = first..last output word, gap = first word - previous op's last word, share = last - previous last (sums to the dispatch).
model = nt x K/64 k-steps per core x 1,613 cycles (KS-P38 timer per k-step in mm_rot, r18r6K) at 1.81 GHz, no gaps / epilogues.
| kind | ops | span | gap | share | k-steps per core (12 ops) | k-step model | share - model |
|---|---|---|---|---|---|---|---|
| attn | 12 | 5.47 | 1.19 | 6.66 | | | |
| norm | 49 | 4.17 | 0.84 | 5.01 | | | |
| ssm | 12 | 3.37 | 0.13 | 3.50 | | | |
| gemm:mlp0 | 12 | 2.76 | 0.51 | 3.26 | 2,304 | 2.05 | 1.20 |
| gemm:up_a | 12 | 2.34 | 0.58 | 2.93 | 2,304 | 2.05 | 0.87 |
| gemm:up_b | 12 | 2.46 | 0.24 | 2.72 | 2,304 | 2.05 | 0.66 |
| gemm:qkv | 12 | 1.91 | 0.44 | 2.35 | 1,728 | 1.54 | 0.81 |
| gemm:mlp3_a | 12 | 1.08 | 0.65 | 1.73 | 1,152 | 1.03 | 0.71 |
| gemm:yx | 12 | 1.71 | -0.04 | 1.67 | 1,152 | 1.03 | 0.64 |
| gemm:down_a | 12 | 1.11 | 0.18 | 1.29 | 1,152 | 1.03 | 0.27 |
| gemm:out | 12 | 0.56 | 0.48 | 1.04 | 576 | 0.51 | 0.53 |
| gemm:mlp3_b | 12 | 0.95 | 0.09 | 1.03 | 1,152 | 1.03 | 0.00 |
| gemm:down_b | 12 | 0.95 | 0.05 | 1.00 | 1,152 | 1.03 | -0.02 |
| gemm:out_proj | 12 | 0.57 | 0.36 | 0.92 | 576 | 0.51 | 0.41 |
Read: GEMM share 19.94 ms of which the k-step model is 13.37: **6.6 ms is per-op overhead** (gaps 3.3 ms = the op-boundary
bubble: dependent A slot load from DDR + first-tile wait + previous op's drain tail, ~30-55 us per op; span excess 3.3 ms =
epilogues (mlp0 GELU +0.71, up_b +0.41, qkv V^T +0.37, up_a +0.29) and yx +0.68). mlp3_b / down_b (A reused, no epilogue) sit
exactly on the model: the k-step rate is right. vs r17d: attn 6.78 -> 6.66, norms 4.99 -> 5.01, ssm 3.99 -> 3.50, GEMMs 23.6 -> 19.9.
**Open question (rev 15): re-rank with this.** The op-boundary bubbles (~3.3 ms GEMM + 1.19 attn + 0.84 norms) are now as big as any
compute lever: what overlaps the next op's A slot load with the current op's tail (deps allow it only where A is not the
previous op's output), and what the yx span excess is.
## ROT variants (rev 14 rank 3) - DONE: g0a kept (full12r19b)
- Probes (npu-tapnext r19/wrapgen.py + wrapb.py, one core, 128 pops over two passes of a 4,608 B object, the wrap at ROT density):
  PWRAP with m2 = -4544 (pre-registered) WRONG (pass 2 all 64 pops wrong); m2 = -4608 RIGHT: the 2D post-modify (dn2 0) is ADDED to
  the 64 B line step. PADDA (padda [p1], m2 one bundle before the fill) RIGHT; padda in the fill's own bundle WRONG (negative control).
  gen_rot's emit and replay fixed (step = 64 + m2): the replay now flags g0f.
- r18/gen_rot.py ROT_ACQ_GAP=6 ROT_REL_GAP=8 --free, 4 acc_hi placements: g1a OPTIMAL R 73 (20 s), 1,368 static; g0a OPTIMAL R 72
  (86 s), 1,357 static; 0 bank stalls; asmlint 0. Both regress BIT-IDENTICAL.
- A/B vs r18m (20 x 20): g1a -0.104 / -0.111 / -0.079 (p10 / median / mean; pre-reg -0.13 = 15 cycles x 8.59 us); g0a -0.174 /
  -0.183 / -0.152 (pre-reg -0.23 = 26 cycles; rev 14 said g0f -0.28 vs g2f). ~0.75 of the static price lands.
## TPROF: per-tile GEMM stamps of r18m + g0a (MEASURED, npu-tapnext mm8.cc -DTPROF, r19/tprof.py; 6,240 tile records, cycles)
| kind | K | nt | first k-step | k-step | epilogue | hand (prev epilogue end -> mm_ktile entry) |
|---|---|---|---|---|---|---|
| plain (down_a/b, mlp3_a/b, out, out_proj), yx | 768/1536 | 4-8 | 1,487 | 1,576 | 1,310 | **3,010** (p90 3,010) |
| mlp0 (gelu_pan) | 768 | 16 | 1,485 | 1,578 | 4,918 | 3,010 |
| qkv | 768 | 12 | 1,486 | 1,579 | 1,941 (p90 2,794) | 3,010 |
| up_a / up_b (gated, 2 tiles per C object) | 768 | 16 | 1,558 / 1,583 | 1,573 | 3,263 | **682** |
Read: hand = in_b.release (bias) + out_c.release + loop + **out_c.acquire (waits for the previous C object's drain)** + acc_zero.
The second tile of a gated object (no C acquire) shows zero + loop = 682; so the C drain wait is ~2,330 cycles per object: the
core DMA always drains the WHOLE 18,432 B C object (lowered dma_bd len 9216 bf16) at 7.9 B/cycle, and the next tile cannot start
because 12,288 B of its accumulators (rb < 12, pair 0) live in the C object. ~76 C objects per block per core: **~1.2 ms per step
of GEMM cores idle on C drains**, + acc_zero ~600 cycles per tile (~0.29 ms). The steady k-step 1,576 matches the timeline model.
Plan: (1) zero acc_hi's 24,576 B right after out_c.release (overlaps the drain; acc_zero then zeroes only the C part
unless acc_hi is dirty: second tile of an object / after attention): static ~-400 per tpo=1 tile = **-0.16 ms**; (2) release the
C object in parts (a hand-built core-side C BD chain with 2-4 locks, epilogue releasing each part as written; the memtile join
receives the same 18,432 B stream): drain overlaps the epilogue, ~-650 per object with 2 parts = **~-0.33 ms** (more with 4).
**Open question (rev 15): price / design (2)**, and any layout that moves the 12,288 B of accumulators out of the C object (L1 is
~0.5 KB short of everything; acc_hi is 24,576 B), which would hide the whole drain (~-1.2 ms).
## CSPLIT (new, main) - DONE in full12r19d: -0.25 / -0.48 / -0.26 ms (p10 / median / mean vs r19c; pre-reg -0.33)
- The C object drains in two parts: bytes 0..12,287 (the accumulators' C part; that BD releases the C prod lock) then 12,288..18,431
  (that BD releases a new lock 6, init 1, which acc_epilogue / attn_begin take once per object before writing part 2). IRON cannot
  express two BDs for a joined fifo's producer: npu-tapnext r19/csplit.py patches the lowered input_with_addresses.mlir and
  r19/csplit.sh re-runs aiecc on it (unpatched, aiecc on input_with_addresses.mlir reproduces elfs, CDO and .bin byte for byte).
- ZHI (acc_hi zeroed right after the C release) -0.16 / -0.15 / -0.16 (pre-reg -0.15..-0.20).
- What is left of the C wait: part 1 (12,288 B at 7.9 B/cycle = ~1,555 cycles) after each epilogue (~0.8 ms per step).
## Timeline r19d (3 runs; k-step model at TPROF's 1,576): attn 6.69 (span 5.50, gap 1.17), norms 4.88, ssm 3.49, GEMMs ~19.2
- Norm ops per column (span, us): norm0 (RMS, 3 inputs) 95.6, norm1 (RMS, 2) 67.7, norm2 (LN, 3) 98.0, norm3 (LN, 2) 82.7; every norm
  op's gap 16.4-16.7 us (0.79 ms per step). Per unit (216 per op): 3-input ops ~801 / 821 cycles = **3.8 B/cycle of input: input-
  bound** (n1k reads three streams in both cases, so compute is the same); 2-input RMS ~567 (compute + glue). This contradicts the
  rev-13 fit's R_in 5.75; the fit is to be re-checked against these op walls. If R_in ~3.8 holds, the lever for 3-input norms is input
  bandwidth (a second S2MM stream into the EW core), and N-IO / NS only help the 24 two-input norms.
- GEMM epilogues (TPROF, per core): plain 1,310 x 32 tiles, gelu_pan (mlp0) 4,918 x 16, gated (up) 3,263 x 32, qkv 1,941 x 12 per
  block = ~1.65 ms per step.
## KPREP (= K1's prep loop in C++, main) - DONE in full12r19g: -0.13 / -0.09 / -0.15 vs r19f (pre-reg -0.12..-0.18)
- chunk_k_fused's -m rows / -inf slots: the six maxima loaded first, 24 stores straight-line (~40 bundles vs a serial ZOL ~140).
## CPRATE (rev 15 rank 0) - MEASURED on r19g (20 x 20, A/B in one process; 4,800 extra instructions per step on mlp3_a's path)
- npu_write32 x 400 per block: +0.268 / +0.296 / +0.274 ms (p10 / median / mean) -> **t_op(write32) ~0.06 us** (0.056-0.062).
- npu_maskpoll (mask 0, always satisfied) x 400: +0.520 / +0.553 / +0.522 -> **t_op(maskpoll) ~0.11 us**.
- Below the pre-registered 0.13-0.22 (miss on the rate), far above the 0.03 kill line: the CP is on the boundary path. SEQPRE's
  price at these rates (10,515 instructions; mix?) is ~-0.6..-1.0 raw; re-pricing with per-type rates is open (blockwrite / BD writes /
  task pushes not measured).
## SEQPRE (rev 15 rank 1) - MEASURED, mostly a miss (design8 env SEQPRE=bits; every variant regress BIT-IDENTICAL, no hang)
| variant (vs r19g, 30 x 20) | p10 / median / mean ms |
|---|---|
| v1: slot loads + memtile sends + drains before finish_deps (GEMM, attention) | **+1.10 / +1.02 / +1.09** |
| v2: sends back after the A fills (loads + drains before) | +0.11 / +0.13 / +0.15 |
| mode 2: only the drains before | +0.15 / +0.16 / +0.15 |
| mode 1: only the slot loads before (attention: loads + drains) | -0.04 / -0.04 / -0.04 |
- v1's timeline: down_a / down_b / mlp3_b gaps +0.35 / +0.48 / +0.42 ms: a K=1536 op pushes 5 memtile MM2S tasks on the 4-deep
  queue; pushed before the A fills, the blocking push waits on work that cannot progress until after the wait (the round-10 rule).
- Pre-issuing drains costs (+0.15) instead of saving: unexplained (shim S2MM queue / sync semantics?). The CP rate is real
  (CPRATE) but the post-sync instruction count does not set the boundaries. The boundary model needs re-thinking; the next check
  (TPROF `first` on the dependent ops' first tiles) needs ~400 B of GEMM text (16 free... 400 free now).
## DDRCAP (rev 15 rank 0') - MEASURED: the aggregate DDR read cap is ~55-59 GB/s; NIN2 cannot pay
npu-tapnext r19/ddrcap.py: per column NIN shim -> core (row 2) streams (shim MM2S 0 / 1 -> core S2MM 0 / 1, 2 KB objects, the core
only acquires / releases), 4 MB per stream; fills awaited; host time per dispatch (median of 20; ~0.08 ms dispatch overhead inside).
| streams (columns x per column) | MiB read | ms | GB/s raw | GB/s minus 0.08 ms |
|---|---|---|---|---|
| 4 x 1 | 16 | 0.413 | 40.6 | 50 (12.6 per stream = 7 B/cycle: not capped) |
| 2 x 2 | 16 | 0.413 | 40.6 | 50 (same: per-column channel count irrelevant) |
| 8 x 1 | 32 | 0.650 | 51.6 | 59 |
| 4 x 2 | 32 | 0.652 | 51.5 | 59 |
| 8 x 2 | 64 | 1.314 | 51.1 | 54 |
| with an 8 x 256 KB drain: 8 x 1 / 8 x 2 | 32 / 64 | 0.680 / 1.332 | 49 / 50 read, 74 / 63 r+w | |
- One stream alone does ~7 B/cycle; 8+ streams hit an aggregate read ceiling of ~55-59 GB/s (= 3.8-4.1 B/cycle per column over 8
  columns). The 3-input norms read 3,072 B x 8 per 801 cycles = 55.5 GB/s: **they sit at the DDR read cap**, not at one channel's
  rate. MACHINE's old 55-63 GB/s stands for reads; the earlier 84 GB/s (in + out) is not reproduced (r+w with drains 63-74 here).
- So NIN2 is void; N-IO-ASM / NS help only where compute exceeds the DDR floor: LN 2-input (norm3, 693 vs ~540 per unit:
  <= -0.2 ms). The norm floor is bytes: only fewer DDR bytes (on-chip residency, or bit-exact transport savings) move it.
# Round 19 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r19z: 33.88 ms** at depth 12 (sweep median, min 33.86; r18m re-swept in the same session 34.79; A/B in one process
-0.880 / -0.888 / -0.874 vs r18m), d10 28.37, d8 22.74, d6 17.09; regress BIT-IDENTICAL vs full12r9a, asmlint 0; DAVIS d12 70.84
(OA 92.57), d10 69.90 (91.02), unchanged. r19z = r18m + ROT g0a + ZHI + CSPLIT + EPLR + KPREP + SEQPRE=1 (slot loads before the
wait). Test machine, Turbo, gpu_busy 0, the background CPU worker throughout. Details: npu-tapnext NOTES 24, RESULTS 28. Timeouts: 0.
| id | predicted | measured (p10 / median / mean ms) | verdict |
|---|---|---|---|
| PWRAP / PADDA probes | pass 2 = pass 1 at m2 -4544 / padda 1 bundle early | -4544 WRONG, -4608 right (step ADDS); padda right, same-bundle wrong | half hit |
| ROT g1a (rev 14 g1f) | -0.14 (main -0.13) | -0.104 / -0.111 / -0.079 | hit |
| ROT g0a (rev 14 g0f) | -0.28 (main -0.23) | -0.174 / -0.183 / -0.152 | miss (~0.75 lands) |
| GEMM bank layout (rev 14 ranks 4-5) | | not built: the rev 15 capacity proof (B two banks infeasible, A slots <= 15) | dropped |
| ZHI (main) | -0.15..-0.20 | -0.156 / -0.149 / -0.160 | hit |
| CSPLIT (main) | -0.33 | -0.250 / -0.476 / -0.261 | hit |
| EPLR (CSPLIT4's plain share) | -0.06 | -0.100 / -0.119 / -0.096 | better |
| K1 as KPREP (C++ prep only) | -0.12..-0.18 | -0.130 / -0.093 / -0.147 | hit |
| CPRATE | t_op 0.13-0.22 us | write32 0.06, maskpoll 0.11 us | CP real, rate half |
| SEQPRE | -1.4 +- 0.5 | v1 +1.10, v2 +0.13, drains-first +0.15, loads-first -0.04 (kept) | miss |
| DDRCAP | >= 84 GB/s, ceiling ~110-120? | aggregate read cap ~55-59 GB/s; one stream ~7 B/cycle | refutes NIN2 |
| NIN2 + N-IO-ASM | -0.7..-1.5 | not built (DDRCAP: 3-input norms already at the read cap) | void |
| CSPLIT4, LATE-C, DAPRE, S3 / K2, ssq2 / FSX | | not built | next round |
### Open questions (rev 16)
- The op boundaries (~5 ms) are NOT the post-sync CP instruction count (SEQPRE). What else sits between a producer's last drain and
  the dependent op's first output: DMA completion latency, the A slot's first chunk pacing at the DDR cap (all 8 columns load at
  once: 8 x 124-248 KB at 55 GB/s = 18-36 us!), the C drain tail? DDRCAP makes the A-slot load itself a candidate: price it.
- CSPLIT4 with one static sub-part boundary that gated (gup + stash copy rows) and gelu_pan (bgelu3 + pan) can both release early,
  and whether an acquire-only BD (no release) is legal in the lowering.
- Re-rank with the DDR read cap: which ops are DDR-bound now (norms, SSM inputs 7 KB per chunk, the A slot loads)?

# Round 20 (optimisation rounds), early data: the op boundary decomposed (2026-10-11; test machine Turbo, gpu_busy 0, background CPU worker running)
All MEASURED on r19z-based timing builds (npu-tapnext r20/): TPROF records moved into C part 2 (EPLR drains part 1 early) + a core
tile counter (r20t2, text 16,352); the 24 GEMM cores' timers agree within ~1k cycles, so records compare across cores.
**CPST probe** (mm8 -DCPST, design8 CPST_OP / CPST_ADDR, r20c4): core 0_2 spins on an L1 flag word (0x77B40) at b11_mlp3_a's first
tile; the CP writes it right after mlp3_a's dependency wait (npu_write32 to the core's data memory works); every other core's flag is
preset. `r20/binpatch.py` re-orders the lowered .bin (TXN ops) to move that write after the k-th tct of the wait.
### mlp0 -> mlp3_a, block 11 (us after mlp0's last k-step end, max over 24 cores)
| event | us |
|---|---|
| mlp0's last epilogue (gelu_pan 4,910 cyc) ends | 2.7 |
| CP passes tct 1 of 8 (column 0's drain done) | 14.9 |
| CP passes tct 4 / tct 8 | 15.4 / 15.8 (0.13 us per tct) |
| core 0_2 (column 0) first k-step ends | 20.0 |
| first k-step end, columns 0..7 | 20.0 20.8 21.5 22.3 23.1 23.9 24.6 25.5 (0.78 us per column) |
| first tile's k-steps | 2,790 cyc each (steady 1,576): A arrives at the DDR read cap, 13.3 us excess over 24 k-steps |
- **The drain tail is ~12 us**: the last C objects (8 columns x 3 x 18,432 B = 442 KB through the memtile joins, store-and-forward)
  written at the DDR WRITE cap. New probe **DDRW** (r20/ddrw.py, drains only): one column 12.1 GB/s (~7 B/cycle), 2 columns 24.4,
  4 columns 45.3, 8 columns ~50-60 GB/s (slope; 0.06 ms dispatch overhead): **the write cap equals the read cap (~55 GB/s)**.
- **Post-sync CP**: 8 A fills x 4 instructions (blockwrite, patch, maskpoll, write32), then 40 memtile send pushes (maskpoll +
  write32); K=768 ops' first k-steps stagger by 0.47 us per column, K=1536 by 0.78 (their 5 sends per column are pushed after all
  fills). k0 end of column 0 is 4.15 us after the CP pass (fill issue + DDR -> memtile chunk 0 -> core + one paced k-step).
### First-tile A pacing, every GEMM of block 11 (ks per k-step in the op's first tile; steady 1,576)
| op | out | up_a | up_b | down_a | down_b | qkv | out_proj | mlp0 | mlp3_a | mlp3_b | yx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ks1/k | 2,835 | 2,977 | 1,576 | 2,473 | 2,021 | 2,655 | 2,650 | 2,787 | 2,626 | 2,003 | (not dependent in b11 data) |
| excess us | 7.7 | 8.5 | 0 | 11.4 | 5.7 | 6.6 | 6.5 | 7.4 | 13.3 | 5.4 | |
Sum 72 us per block = **0.87 ms per step**: every op whose A slot load starts at its own start runs its first tile at the DDR
read cap (8 x 10,368 B per 2,650-2,980 cycles = 50-57 GB/s), including the non-dependent down_a / down_b / mlp3_b.
### Reading
The boundary is bytes, not instructions: (1) the producer's last outputs written at the DDR cap (~12 us for a GEMM producer),
(2) ~1 us of tcts + ~3-5 us of CP fill / send issue staggered by column, (3) the consumer's A at the read cap through its first tile.
SEQPRE could only touch (2). Next steps: IL = mlp0 dealt round-robin (P23 rr: step j of row r = chunk 3j + r), so mlp3_a's
chunks 0-23 are drained after mlp0's step 7; a split mlp0 drain (steps 0-7 / 8-15) lets mlp3_a's fills start mid-mlp0, half 1
(chunks 12-23) loading into the free half while mlp0 runs, half 0 as soon as mlp0's last sends free it (per-half full locks).
Open question (rev 16): price the norm -> GEMM bypass (EW core output panels straight into memtile c's A slot; norm c's rows = GEMM
column c's A) and any other way to keep the dependent A off DDR; and the drain tail (442 KB at the write cap per GEMM -> X boundary).

# Round 20 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r20z**: A/B in one process vs r19z **-1.99 / -2.07 / -2.06 ms** (p10 / median / mean, 40 x 20); sweep (loaded box, see
below) d12 33.71 / d10 28.30 / d8 22.35 / d6 16.76 vs r19z same session 35.65 / 29.95 / 23.91 / 17.89; quiet-box estimate ~31.8 at
d12. Regress BIT-IDENTICAL vs full12r9a; DAVIS d12 70.84 (OA 92.57), d10 69.90 (91.02), unchanged. Timeouts 0. Later in the session a second
background CPU job (~7.6 cores, no NPU) ran: A/B spreads after that are wider. Details: npu-tapnext NOTES 25, RESULTS 29.
| id | predicted | measured (p10 / median / mean ms) | verdict |
|---|---|---|---|
| DDRW | 3.5-4.5 B/cycle per column (H_WR) or >= 7 (H_TOK) | one stream 12.1 GB/s (~6.7 B/cycle); 4 col 45, 8 col ~50-60 GB/s | aggregate write cap = read cap: tail = 442 KB at ~55 GB/s; neither per-channel H_WR nor H_TOK; DDRW-2ch moot |
| CPST-N | flag at 6-14 us after norm1's last word | up_a: column 0's k0 4.1 us after the CP pass, +0.31 us per column; qkv 9.6 us (norm end not stamped) | partial |
| POPLF1 | 64/64 or a backend error | vldb control, vlda.pop lf1, vlda/vldb mixed, vlda.fill: 128/128 each (r20/poplf1*.py) | E7'' pops feasible |
| MLPRR (= IL) | -0.30..-0.42 | -0.415 / -0.399 / -0.405 (vs r19z); TPROF: mlp3_a k0 1.45 us after mlp0's end, pace 1.1 us | hit |
| DAPRE | -0.10..-0.14 | -0.127 / -0.129 / -0.108; down_a pace 0.5 us | hit |
| NSEQ | -0.08..-0.20 | -0.167 / -0.174 / -0.166 (norms' params + drains before the wait) | hit |
| NRES norm1 | -0.15..-0.28 | without NSEQ -0.113 / -0.057 / -0.091 (columns 1.5 us apart: the CP's per-column norm issue; B lockstep); with NSEQ -0.177 / -0.186 / -0.208 | hit with NSEQ |
| NRES norm0/2/3 | (all four -0.6..-1.1) | -0.690 / -0.651 / -0.641 on top | hit |
| SRES (main: the SSM's z panels the same way) | -0.10..-0.20 | -0.106 / -0.195 / -0.156 | hit |
| AQPRE (main: attention Q in half 1, filled after qkv's Q drain group) | -0.10..-0.20 | -0.087 / -0.106 / -0.127 (40 x 20, loaded); ATPROF: Q copy 28.6 -> 0.4 us after qkv's end | hit |
| AKVPRE (main: qkv step 8 = V heads 0-2 its own group, head group 0's K / V filled after it) | -0.10..-0.14 | -0.241 / -0.134 / -0.168 | hit |
| SENDPRE (main) | -0.05..-0.15 | +0.197 / +0.162 / +0.155 | dead (retired in the analysis) |
| OPPRE (main: out_proj's A in 4 parts after each attention pass-1 head-group drain) | -0.18..-0.3 | +0.117 / +0.205 / +0.184 | dead, cause unmeasured |
| CSPLIT4-lite, LATE-C, S3 / K2, ssq2 / FSX, E7'' | | not built | next round |
### NRES as built (differs from rev 16's design)
- Slot layout: rev 16's rb-major half 0 with new sends broke correctness (the last tile's chunk-wise `last` release let the next op's
  chunk-0 load overwrite rows 0-1 of every chunk; G chunks 31/39/47 wrong). Built instead: the memtile S2MM 5 task scatters the rb-major
  stream into the usual chunk-major half with a 3D TAP `[18 rb][12 q][144 words]` strides `[144, CH, 1]`, and the normal sends stay.
- Route: IRON 1.4.4's pathfinder finds NO route for two packet flows from two DMA channels of one EW tile (even one column alone; from
  one channel it routes). Eout (packet 3) and Pn (packet 5) are both declared from MM2S 0; r20/nresfix.py adds
  `packet_rules(DMA : 1) { rule(31, 5, <same amsel>) }` to each EW switchbox in the lowered MLIR before aiecc re-runs.
- Ring: 2 slots x 576 B in EW bank 0 (stack 6 KB -> 4.75 KB; deepest 3.2 KB). EW text 15,808 -> 16,272 (NRES + SRES).
- The norm's group is pushed, not finished; the GEMM after it has no dependency on it and issues B / sends / drains first.
### Open questions (rev 17)
- E7'' generator design now that POPLF1 passed (A on which FIFO / slot, B line weave, CP feasibility at R = 64, registers, banks).
- The remaining first-tile paces (out_proj 6.8, down_b 6.0, mlp3_b 5.3 us per block) and the GEMM -> norm tails (4 per block, ~12 us
  each at the write cap): price a third A region (B fifos moved off memtiles 0-2?) and anything that shortens the tails.
- OPPRE's +0.18: model what the out_proj A load (1 MB over attention pass 1) costs the attention's K / V stream at the DDR cap.
- C waits (~0.48 ms: hand 1,355 plain / 1,807 others vs 676): CSPLIT4-lite for gelu_pan alone (restructured bgelu3 + pan split at
  rb 12, 1a = [0, 6,912)) vs LATE-C / a zero-load first k-step in ROT records vs a runtime-driven C path with per-op sizes (74 MB junk).

# Round 21 (optimisation rounds), early data (2026-10-11; test machine Turbo, gpu_busy 0; background CPU jobs + CP-SAT running: see the clock note)
- **CSUB** (C part 1 drained in 3 sub-parts [0, 3,456) / [3,456, 6,912) / [6,912, 12,288), each released by the epilogue when written;
  r19/csplit.py --sub 1728,3456): A/B vs r20z -0.135 / -0.133 / -0.131 (p10 / median / mean), predicted -0.12..-0.25. An acquire-only
  BD block is REFUSED by the MLIR verifier ("one use_lock(acquire) and one use_lock(release)"); releasing the prod lock by 0 compiles
  and works (bit-identical, no hang).
- **ZLOAD** (zero-load first k-step, the C-wait floor): no gen_rot change needed. The kp-0 MACs are exactly those with A operand ex0;
  r21/mkzload.py rewrites them to read their config from r6 (= a 4th argument, copied before the preamble's `mova r0, #18`), and
  mm_ktile passes 781 (780 | zero_acc, bit 0 per aie2p_vmult.h) for k == 0. acc_zero / acc_zero_hi keep only their flags.
  Bit-identical. A/B vs CSUB: p10 -0.14..-0.15, median -0.10 (fast mode). **E7'' must stay compatible**: kp-0 MACs on one A
  register and r6 free, so mkzload applies to the E7 kernel too.
- TPROF r21t2 (hands, cycles, from 676-1,807): plain 345, up 75, mlp0 668 (last sub-part 5,376 B at 7.9 B/cycle), V^T 671, Q / K
  1,108 (bias_rows before the first release; CSUB-QK moved the bias into the group loop: A/B -0.02, neutral). Epilogues grew with
  the overlapped drain + extra calls: plain 1,356 -> 1,575, mlp0 4,934 -> 5,495, qkv 1,962 -> 2,123 (TPROF build, incl. stamps).
- CSUB-G (gated tiles in the same group loop): +0.02, dead; frees 256 B of GEMM text if ever needed.
- **POPLF0**: vldb.pop.576 / vldb.fill.512 on lf0 ([p0, lf0, r24]), vlda / vldb alternating on lf0, vlda.fill + vldb pops: 128 / 128
  each (r21/poplf0*). The earlier "lf0 on ldb UNVERIFIED" is now MEASURED.
- **Clock under CPU load (MEASURED, r7/clk.py)**: with a background CPU job (~8 cores) + a 12-worker CP-SAT + another background worker the NPU
  core clock is **1.405 GHz** (not 1.81): r20z 32.3 -> 41.0 ms (x1.27). A/B pairs stay valid; absolute numbers from then on are
  slow-mode. CP-SAT is kept at <= 6 workers during the round-end sweep.
- First-tile paces now (TPROF r21t2, us per block): out_proj 6.7, down_b 4.15, mlp3_b 3.92, mlp0 1.9, up_a 1.5 (sum 20.8).
- Memtile map (r21/mtmap.py on r20z): memtiles 0-2 free 128,000 B in holes of <= 56,320 (B_r fifo 8 x 4,608 there); memtiles 3-7
  free 164,864 with one 141,312 B hole at 382,976. Moving B_r's forward off memtiles 0-2 would leave a 141,312 B hole there too.
- **LATEC** (a variant of the analysis-pass R3 on top of CSUB + ZLOAD: IRON's C acquire = sub-part 1 drained (its BD releases prod), sub-parts
  2 / 3 release new locks 7 / 8, mm_rot acquires them at rb 3 / 5 of a fresh object's first k-step via the record's acquire field
  as a LOCK ID (`acq.cond r26, r4, r26`, Peano's own register-id form)): **hung on the first dispatch** (recovered). Not debugged; hypothesis: the core now waits for its own drain while holding its k-step-0 B object (multicast to the other columns) and A
  stripes. The C++ is r21/latec.diff, csplit.py --latec, r21/mklatec.py.
- ACSUB (attention O releases sub-part t after tile t): neutral (-0.02 / 0.00 / +0.02), reverted.
- Round total so far (r21q1 = CSUB + ZLOAD + CSUB-QK) vs r20z, 30 x 20, mixed clock modes: p10 -0.335 / p25 -0.289 / median -0.293.
- **E7'' integration is built and the analysis-pass r22/mm_e7_r68try.s RUNS in situ (no hang) but is NOT bit-exact** (full12r21e1):
  - Integration side (all in npu-tapnext, try with `E7=1 [E7_V=<v>] r21/try21.sh NAME`): r21/mke7.py renames the mm_e7_<v>.s / .h into
    r21/mm_rot_<v>.s / .h (design8 ROT slot); design8 E7=1: L1 B objects 1056 words (Y 4,224 B), the GEMM stack +1,152 B (so core_data
    starts at 0x70E80 in the IRON build and the KSC addresses are final), tbn / tba in bank 1; mm8 -DE7: aexp_ab in .aie.bank3,
    mm_ktile calls mm_rot(rec, pY, pc, pX, cfg) with pX = KS_X0 / KS_X1 paired with KS_B0 / KS_B1; r21/e7patch.py (via csplit20
    E7PATCH=1) adds the X buffers at 0x70A00 / 0x70C40, lowers the stack back to 2,560 and splits each B S2MM BD:
    [acq prod; Y 1,008 words; rel e7h] -> [acq e7h; X 144 words; rel cons] (an acquire by 0 is REFUSED at CDO generation, so a
    hand-off lock e7h = id 7, init 0). Resulting placement = the specified PLACE exactly (acc_hi 0x71940, rec 0x71784, Y 0x7D100 / 0x7E180,
    .aie.bank3 0x7F200 + 3,584). The `check` on the try kernel: 1,338 cycles per call, 0 stalls, both buffers.
  - aexp_ab in bank 3 alone is fine (debug build r21x2: bit-identical). X buffers swapped: much worse (so the pairing is right).
  - **The error is confined**: block 0's yx output (depth 1, frame 0, vs r21q1, r21/e7dbg.py) differs in 2.2 % of values, ALL in
    row block 0 of every column's 144 rows (37 % of rb 0's values) and only in column blocks cb 1, 2, 3 (pair 0) of EVERY 64-col
    tile; cb 0 and pair 1 are right; rb 1-17 are right. I.e. rb 0's jobs (cb 1-3) started in the PROLOGUE. Suspect: a prologue op
    the replay sees differently from the hardware - e.g. a conveyor (old-value) read across the prologue's `acq.cond #53` (same
    bundle as `lda p0`): the A-slot acquire really STALLS in situ at most tiles' first k-step (and maybe every k-step's prologue),
    and an in-flight pop may land during the stall, so a later MAC sees the new value instead of the old one. (ROT g0a's prologue
    has the same acquire; does it read any register in its old-value window after it?) Or the prologue's lf0 / lf1 restarts
    (`mov p0, p6` / `mov p1, p5` + fills) vs the hardware.
  - Better hypothesis (deterministic, every tile): with ZLOAD the FIRST k-step ignores the loaded accumulators, so only k >= 1
    exercises the accumulator loads. If the replay starts from zero pieces, a kp-0 MAC that reads its dm before the prologue's
    vlda bm quarters landed (or reads the previous call's last job in that dm) is invisible to it, and wrong in situ for every
    k >= 1. Open: replay with non-zero initial pieces and cfg 780 (and 781 for the first call) - rb 0's cb 1-3 jobs first.
  - Timing TWIN (wrong rb-0 outputs, tracks within 0.05 px, so little data-dependent drift): full12r21e1 (E7 r68try) vs r21q1, one
    process, 20 x 20, NPU at 1.405 GHz (CPU load): **-0.438 / -0.458 / -0.399 ms** (p10 / median / mean); ~-0.35 at 1.81 GHz.
  - **UPDATE: the re-fitted r22/mm_e7_r68try.s (one prologue bundle expanded: the .p2align pad) is BIT-IDENTICAL in situ**
    (full12r21e3 = r21q1 + E7 with it; regress vs full12r9a over 4 frames). So the rb-0 / cb 1-3 error was the pad. asmlint: one
    LOCK-REL-GAP warning (rel.cond #52 0 bundles after a vldb.pop on lf0 - with B on lf0 the rule cannot tell A pops from B pops).
  - **E7 r68b (bit-exact) A/B vs r21q1**, one process, 30 x 20, NPU at 1.405 GHz: **-0.385 / -0.282 / -0.285 ms** (p10 / median /
    mean; ~-0.22..-0.30 at 1.81 GHz). Kept as the round's best (full12r21e3). The wrong-output twin (r21e1) said -0.44. GEMM text
    16,320 / 16,384 (64 B free). A faster R drops in with `E7_V=<v> r21/try21.sh NAME` once r21/mke7.py has renamed it.
- Re rev 17:
  - **ACQ0 answered**: aiecc accepts `AcquireGreaterEqual, 0` in the MLIR but FAILS at CDO generation ("does not fit its buffer
    descriptor"); r21/e7patch.py uses a hand-off lock instead (Y BD releases e7h, X BD acquires it; id 7, init 0). Deployed, exact.
  - **KTL**: to be generated on the E7 kernel (now deployed: full12r21e3 = r68b), not on g0z; the C++ (B(0)
    acquire, mm_rot(.., 781), release, mm_rott(rec, B(1), X(1), p3', nks - 1, 780)) is added when the kernel lands. GEMM text has 64 B free;
    256 B can be freed with CSUB-G (+0.02 ms) if the KTL text delta needs it.
  - OPPRE's round-20 patch was not saved (reverted in the working tree); OPPRE-W would need it rebuilt from NOTES 25.4.
  - DAVIS on full12r21e3: d12 AJ 70.84 (OA 92.57), d10 69.90 (91.02): unchanged (bit-identical).
- **E7 R 64 (r22/mm_e7_r64_x8.s)**: the delivered .s was NOT fitted (one prologue bundle short: in situ rb 0 / cb 2-4 wrong, the
  same signature as r68try). The `fit` on e7_r64_x8.json gives the aligned version (r21/mm_rot_e7r64f): BIT-IDENTICAL.
  A/B vs E7 r68b (full12r21e5 vs r21e3), 30 x 20, clock back at 1.81 (CPU load down): +0.041 / -0.035 / -0.080 ms (p10 / median /
  mean): **neutral**, as the replay said (1,244 + 129 bank stalls = 1,373 vs r68b's 1,338 + 0). Kept r68b. An R 64 / 66 schedule
  with hard (or cheaper) bank constraints is the next E7 step; the delivered .s must always be the fitted one.
- r21e3 in this A/B: p10 31.63 / median 32.36 ms at d12 (background CPU job still running, CP-SAT small).

# Round 21 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r21k1** = r20z + CSUB + CSUB-QK + ZLOAD + E7'' (R 68, fitted) + IV32 + K0B. A/B in one process vs r20z, 30 x 20,
quiet box (background job done; CP-SAT at 4 workers): **-0.686 / -0.705 / -0.721 ms** (p10 / median / mean). Same-session sweep:
d12 31.30 / d10 26.22 / d8 21.06 / d6 16.16 vs r20z 32.33 / 26.89 / 21.53 / 16.23. Regress BIT-IDENTICAL vs full12r9a; DAVIS d12
70.84 (OA 92.57), d10 69.90 (91.02) on r21k1, unchanged. Text: GEMM 15,760 / 16,384 (624 B free), EW 16,272. Timeouts: 1 (LATEC).
Details: npu-tapnext NOTES 26, RESULTS 30.
| id | predicted | measured (p10 / median / mean ms) | verdict |
|---|---|---|---|
| CSPLIT4-lite (main: CSUB, 3 sub-parts, every epilogue) | -0.11..-0.17 (rev 16) / main -0.12..-0.25 | -0.135 / -0.133 / -0.131 | hit |
| zero-load first k-step (main: ZLOAD, zero_acc bit) | -0.10..-0.18 (main) | p10 -0.14, median -0.10 (fast clock) | hit |
| LATE-C (main: LATEC) | -0.3 (rev 15) / main ~-0.07 | hung on the first dispatch | dead |
| RTC (runtime C drains) | -0.05..-0.10 | not built (rev 17: not proposed) | - |
| POPLF0 (lf0 on ldb) | UNVERIFIED | 128 / 128 on every form | MEASURED |
| ACQ0 (acquire by 0) | aiecc accepts? | verifier yes, CDO generation NO; hand-off lock used | answered |
| E7'' R 68 (r68b) | rev 16: -0.9 +- 0.3 at R 64 | -0.385 / -0.282 / -0.285 at 1.405 GHz (~-0.22..-0.30 at 1.81) | kept, partial |
| E7'' R 64 (fitted) | replay 1,244 + 129 stalls | +0.041 / -0.035 / -0.080 vs R 68 | neutral |
| IV32 (rev 13) | -200..-300 B text | -560 B GEMM text, time -0.004 | hit (text) |
| K0B (main: chunk 0 through the batched fast pass) | ~-0.03 | -0.032 / -0.010 / -0.006 | small hit |
| CSUB-G, ACSUB (main) | -0.04..-0.08 / 0..-0.05 | +0.02 / neutral | dead |
| 3A, OPPRE-W / -lag, S3 / K2, KTL, ssq2 / FSX | | not built (KTL in flight) | next round |
### Open questions (rev 18)
- **KTL on the E7 kernel** (gen_e7 --aligned): the biggest priced lever; GEMM text now has 624 B free, so its records / glue fit.
  Fitted kernels only (two unfitted ones lost rows 0-7 in situ); mm_ktile's call sequence follows the generator design.
- E7'' at R 64-66 with the bank stalls kept near 0 (R 64 with 129 replay stalls was neutral).
- K2 as a QK-only + running-max variant of attkb for chunk 0 now that K0B batches chunk 0's fast pass (price it on the batched form:
  one 36-iteration call per (tile, head) + 6 exact reruns), and S3 with the text now available.
- The clock drop under CPU load (1.405 GHz) is worth a note in MACHINE: is it the SoC power budget (cTDP) or a DPM policy? It decides
  whether CPU work running beside the NPU has to be limited.

# Round 22 (optimisation rounds), early data (2026-10-11; test machine Turbo, gpu_busy 0; CP-SAT 4 + 8 workers: load 10-16)
- **K2** (main: r22/gen_attkq.py, a QK-only loop with attkb's vmax_lt operands into the slots; II 14, U 1, 474 B; chunk 0 calls it
  instead of the batched fast pass, the per-row-block test then flags every row block as before): BIT-IDENTICAL vs full12r9a, asmlint 0.
  A/B vs r21k1 (30 x 20): **-0.025 / -0.020 / -0.061 ms**, pre-registered -0.05..-0.10 (12 (tile, chunk 0) x ~850 cycles per block
  static). Costs 608 B of GEMM text (16 B left): kept OFF by default (KTL / S3 need the text). So chunk 0's compute is mostly hidden
  (K / V stream start of each head group?). Twins that skip chunk 0's fast pass are useless here: they change the later chunks' flags
  (one twin -0.79 ms, the other +0.23..+0.50).
- **E7'' nx = 10** (n1 30, ny 24: ny must be a multiple of 8 so the lf1 segment starts 64 B aligned; bank 3 61 reads per row block):
  gen_e7 solve, bank3soft, repaired from the nx = 8 solutions, 4 workers, 1,800 s: R 64 excess 4 + 91 modelled stalls, R 65
  excess 2 + 71. No better than nx = 8. The plumbing for any nx is in try21 (E7_X0 / E7_X1 / E7ARGS).
- Observation on gen_e7's objective: one bank-3 excess unit = 1 stall per row block = 18 per call, one `conf` unit = 1 stall per call,
  but the weights are 10,000 vs 100 (5.5x over-weighted). R 64 / 65 were re-solved with E7_W3 = 1,800 (copy r22/gen_e7m.py).
- Reweighted check: E7_W3 = 1,800 (R 64 / 65, hints = the earlier solutions, 1,800 s): R 64 excess 4 + 59, R 65 excess 2 + 43, i.e.
  the solver stays at the hints. R 66 / 67 running (W3 1,800; R 67 hinted from R 68).
- **3A BUILT and kept** (design8 / plan8 env A3=1, r22/try22.sh): the A slot grows to 36 chunks (373,248 B: region 3 = chunks 24-35,
  locks E3 / F3); down_b / mlp3_b take k 0-11 from half 0 and k 12-23 from region 3 (fills k 12-23 first; middle sends one 3D TAP
  [2][32][972], stride 24 chunks). IRON re-packed memtiles 0-2 itself (3,584 B left, C joins still 64 KB aligned) - no l2-depth change
  needed. BIT-IDENTICAL. A/B vs r21k1: -0.076 / -0.080 (30 x 20) and -0.081 / -0.070 (40 x 20) p10 / median; predicted -0.10..-0.14.
- --l2-depth 6 / 4 (B fifo objects per memtile): +0.005 / -0.023 median, both neutral (8 kept).
- **KTL (r22/mm_e7k_r70) HANGS in situ - twice** (both recovered; known-good r21k1 BIT-IDENTICAL after each):
  - integration: r22/mkktl.py (symbol kept, .h + mm_rot_border for pack8), r18/mm8_rot.inc (MM_E7K_NPER: the generator's record() as constexpr,
    table = mm_e7k_order, KTL_J0C = record(0, F_J0) = 0xF40), mm8 mm_ktile: acquire B(0); mm_e7k(rec, pc - 0xF40, 781, 18 nks, Y(b), X(b)).
    ksc_check moved acc_hi to 0x719C0 (the table grew); the generator check at acc_hi 0x719c0, nks 5: 6,376 cycles, 0 stalls.
    The elf's rot_rec (0x71784, 112 words) equals the generator's table() exactly; mm_ktile's disassembly passes p0 = 0x71784, p1 = pc - 0xF40,
    r0 = 780 + (k == 0), r1 = 18 nks, r2 = b, r3 = X(b); the caller keeps its state in r8-r15 only.
  - full12r22k2 (one call per tile): first dispatch TIMEOUT.
  - **full12r22k3 = bisect -DKTL1: the same kernel called once PER K-STEP** (trips 18, the C++ loop acquires B(k), the kernel releases it,
    the in-kernel B acquire is masked by (r27 - 1)): **also TIMEOUT**. So the hang is in a single k-step call, not the k-step boundary.
  - Probe LDA2D (main, r22/lda2d.s + lda2db.*, one core): scalar `lda.2d` with dn 4 / dj 4 / m -16 wraps exactly as the 2D rule says
    for r, p and m destinations (p / m read back 20-bit): NOT the cause.
  - Differences from e7r68b that a one-core DMA-fed bench (KTLP) would separate: the in-kernel B release (`rel.cond #48, r5, r26` with
    r26 = `mov r26, r29` two bundles before), `movs pX, rBase` restarts, the mul masks, no tail after the loop (ret right after le).
    One timeout was left for this phase; KTLP is the next run.

# Round 22 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r22z** = r21k1 + 3A + TXT18 + attkb II 33 + attv II 20 (build r22/try22.sh). A/B vs r21k1 in one process (30 x 20):
**-0.408 / -0.488 / -0.422 ms**. Quiet same-session sweep: d12 30.78 / d10 25.79 / d8 20.71 / d6 15.58 vs r21k1 31.20 / 26.14 / 20.95 /
15.79 (1.806 GHz). Regress BIT-IDENTICAL vs full12r9a. Text: GEMM 15,856 (528 free), EW 15,824 (560 free). Timeouts: 2 (KTL).
| id | predicted | measured (p10 / median / mean ms) | verdict |
|---|---|---|---|
| KTL (r22/mm_e7k_r70) | -0.50..-0.70 | TIMEOUT; also as one call per k-step (KTL1) | open: KTLP |
| E7'' R 64-66 (nx 10, reweighted) | near-zero stalls | CP stays at the hints (excess 4 / 2 + 43..91) | not built |
| K2 | rev 17 part of -0.3..-0.4; main -0.05..-0.10 | -0.025 / -0.020 / -0.061 | small, OFF (608 B text) |
| S3 | -0.05..-0.10 (rev 18) | not built | - |
| 3A | -0.10..-0.14 | -0.076 / -0.080 and -0.081 / -0.070 | hit (low), kept |
| OPPRE-W / -lag | H_CP +0.12..+0.20 | not built (OPPRE patch rebuild is M) | - |
| TXT18 | 0 +- 0.02, text <= 15,890 | -0.072 / -0.076 and -0.076 / -0.072; EW text 15,824 | kept (faster than predicted) |
| NTWIN (MSPLIT-P gate) | walls shift vs grow | -0.951 / -0.920; norm walls unchanged, starts 30-50 us earlier | gate PASSED |
| (main) attkb II 36 -> 34 -> 33 | -0.04..-0.14, -0.04..-0.07 | -0.134 / -0.126; -0.03 / -0.04 | kept |
| (main) attv II 21 -> 20 | <= -0.046 | -0.034 / -0.043 | kept |
| SSM glue, MSPLIT-P, CLKLOAD | | not built / not run | next |
- Resource-bound scan (II / bound): attvs 18 / 11, ssq 30 / 21, n1k_ln 11 / 8, vt 13 / 6, ggk 24 / 22 (RESULTS 31.2): attvs next?

# Round 23 (optimisation rounds), started 2026-10-11 (test machine Turbo, gpu_busy 0)
- Plan: MSPLIT-P (norm0 / norm2 producers down_b / mlp3_b), then SSM glue (tcall gate first), MSPLIT-C, norm3 LN. No KTL work in the optimisation rounds.
- **NPU timing windows** (no KTLP bench runs inside them; docker ps / uptime are checked first and the window is skipped if a
  KTLP container runs). Format: "WINDOW OPEN (what)" / "WINDOW CLOSED".
- **MSPLIT-P built** (full12r23f: down_b + mlp3_b, last tile as row passes rb 0-5 / rb 6-17, each its own C object; E7 kernel takes the
  row-block count in r1; --l2-depth 6 frees the 2 memtile BDs the new A tasks need). BIT-IDENTICAL vs full12r9a, asmlint 0, GEMM text
  16,192. First form hung (r23d, timeout 1, recovered): a column 6th memtile MM2S push blocked on its own tile 0, whose B multicast
  needs the other columns' A sends that the CP had not pushed yet; fixed by pushing every column's first five tasks first.
  **Pre-registered** (vs r22z, same process): -0.18..-0.30 ms (norm0 / norm2 start ~10-13 us earlier per boundary x 24; the NRES
  panel task still waits for pass 2's k 0-11 sends). l2-depth 6 alone (r23l6): 0 +- 0.02 (round 22: +0.005).
- WINDOW OPEN (main A/B msplit1: full12r22z vs full12r23l6 full12r23f)
- WINDOW CLOSED
- WINDOW OPEN (main A/B msplit2: full12r23l6 vs full12r23f)
- WINDOW CLOSED

## Analysis note (pass 19) - KTL root cause and the fixed kernel
- **Cause (MEASURED):** acq.cond / rel.cond test BIT 0 of r26 (r24/relcb.py RELC: rel.cond with r26 = 2 did not release, 3 did).
  KTL's masks were flag x counter (r26 = flag x r27, values up to 18 nks): every A acquire at an even counter was skipped - for an
  even trip count (KTL1's 18) all nine - while all releases ran. Shown without a hang by the lock-free bench (r24/ktlpb.py: each
  lock op an add that sums its condition): A acquires "sum 90" per call, bit 0 of each term 0.
- **Fix:** boolean masks (nez + and), re-solved: **r24/mm_e7k_r70b.s / .h** (deployed placement: X 0x70A00 / 0x70C40, Y 0x7D100 /
  0x7E180; fitted, asmlint 0 / 0, padcheck OK, replay bit-correct for 1-24 k-step tiles at acc_hi 0x719C0, 0 stalls; text 1,536 B).
  **The record order changed: take the new .h** (mkktl.py with both files). Signature and call unchanged:
  mm_e7k(rec, pc - J0c, 781, 18 nks, Y(b), X(b)); the caller acquires B(0) only.
- **KTLP (one core, run between the A/Bs; 1 timeout on a stale-header bench, device recovered):**
  - lock-free twin: 9 / 9 A and nks - 1 / nks B lock ops per tile; pieces bit-identical to the E7 R 68 kernel over 1 and 3
    k-steps, as one call per tile and as one call per k-step;
  - **KTLP-D, real locks, A and B fed by shim DMAs** (r24/ktlpd.py): one call of 3 k-steps completes, 36 / 36 piece hashes equal to
    the E7 reference.
- The replays (r22/gen_e7k.py check, gen_e7.py check) now reject any lock condition other than 0 / 1.
- KTL (r24/mm_e7k_r70b) received: integrating as rank 1 (r23/mm_rot_k70b via r22/mkktl.py)
- KTL r23k (r22z + r24/mm_e7k_r70b): BIT-IDENTICAL, asmlint 0, GEMM text 15,584. Pre-registered vs r22z: -0.45..-0.65 ms.
- WINDOW OPEN (main A/B ktl1: full12r22z vs full12r23k)
- WINDOW CLOSED
- KTL A/B vs r22z (60 x 20): -0.822 / -0.821 / -0.805 ms (p10 / median / mean): HIT (above range). BIT-IDENTICAL. Kept; next: KTL + MSPLIT (needs a KTL kernel with a 6 / 12 row-block period).
- KTL + MSPLIT (r23t: KTL kernel with the record period as an argument, r23/mm_rot_k70p; the passes run on rot_rec with 5 entries patched in place, compile-time checked): BIT-IDENTICAL, asmlint 0, GEMM text 16,256. Pre-registered vs r23k: -0.12..-0.22 ms.
- WINDOW OPEN (main A/B ktlms: full12r23k vs full12r23t)
- WINDOW CLOSED
- KTL + MSPLIT A/B vs r23k (60 x 20): -0.144 / -0.155 / -0.186: HIT, kept (best now full12r23t, ~29.96 median in the A/B window).
- **YXSSM (main, new)**: the SSM starts during yx: yx drains in 4 groups (y + x heads 0-2 / 3-5 / 6-8 / 9-11), SSM head q waits only for the group with x(q); SRES z panels into the A slot half 1, out reads half 1 (yx keeps half 0); l2-depth 5 (one more memtile BD). r23y = r23t + YXSSM: BIT-IDENTICAL. Pre-registered vs r23t: -0.15..-0.35 ms (SSM start 3 of 8 yx steps earlier, ~35 us per block, less DDR contention).
- WINDOW OPEN (main A/B yxssm: full12r23t vs full12r23y)
- WINDOW CLOSED
- YXSSM A/B vs r23t (60 x 20): -0.415 / -0.414 / -0.398: HIT (above range). Best: full12r23y. Open: price YXSSM-R (yx steps interleaved y / x so the SSM starts after step 2 of 8) and whether the SSM / yx DDR overlap caps it.
- YXR (yx steps interleaved y / x, natural groups 0 4 1 5 2 6 3 7; drain groups per step pair): r23z1 = r23y + YXR, BIT-IDENTICAL. Pre-registered vs r23y: -0.15..-0.35 ms.
- WINDOW OPEN (main A/B yxr: full12r23y vs full12r23z1)
- WINDOW CLOSED
- YXR A/B vs r23y (60 x 20): -0.379 / -0.377 / -0.364: HIT (above range). Best: full12r23z1 (29.16 median in window).
- ZCH (SSM z panels into the slot chunk by chunk, out tile 0 follows head by head): r23z2 BIT-IDENTICAL. Pre-registered vs r23z1: -0.03..-0.12 ms.
- WINDOW OPEN (main A/B zch: full12r23z1 vs full12r23z2)
- WINDOW CLOSED
- ZCH A/B vs r23z1: +0.010 / +0.016 / +0.024: neutral, OFF (out is issued after the SSM fills).
- OUTPRE (out issued before the SSM head fills, with ZCH): r23z3 BIT-IDENTICAL. Pre-registered vs r23z1: -0.05..-0.15 ms.
- WINDOW OPEN (main A/B outpre: full12r23z1 vs full12r23z3)
- WINDOW CLOSED
- WINDOW OPEN (main A/B outpre2: full12r23z1 vs full12r23z3)
- WINDOW CLOSED
- OUTPRE A/B vs r23z1: -0.023 / -0.024 / -0.031 (120 x 20): small, kept. SSM glue gate (tcall on the deployed EW core, r23z1): 1,828-1,845 per unit with sigk at 1 trip (~2,115 real); ssq2 + reck2 + ggk2 = -188 -> ~1,930 > 1,880: GATE FAILS without FSX (E16 x-pressure). Not built.
- main: CP-SAT KTL R 68 solve (gen_e7.py --ktl, hint r24/e7k_r70b_n16, 12 workers nice 10, 1,800 s)
- main: TIMEOUT 2 (r23o2 = OPH1 + MSPLIT out / out_proj, l2-depth 3); known-good r23z3 rerun: see next line
- r23o2 hang cause (timeout 2, recovered; r23z3 re-run BIT-IDENTICAL): with OUTPRE, out's split branch pushed "<out>#0" through the
  lookahead (finishing yx's groups early) and the SSM then finished out's "#0", which waits for this SSM's own z: a CP deadlock (no
  hardware cause). Fixed (r23o3: out's groups bypass the lookahead and the SSM skips them; SEQLOG=1 prints every CP group wait).
  r23o3 (MSPLIT out + out_proj, OPH1 = out_proj's A in half 1, l2-depth 3) and r23o4 (out_proj only) BIT-IDENTICAL; timing after the
  CP solve ends. One timeout was left this phase.
- KTL R 68 solve (gen_e7.py --ktl --R 68, hint r70b, 12 workers, 1,800 s): FEASIBLE but 35 modelled bank conflicts per k-step (r70b: 0) vs 36 static cycles saved: net ~0, not emitted. Pre-registered (vs r23z3): r23l3 (l2-depth 3) 0 +- 0.03; r23o1 (OPH1) 0 +- 0.02; vs r23l3: r23o4 (MSPLIT out_proj) -0.03..-0.08, r23o3 (+ out) -0.05..-0.12 (out split may delay the SSM start: its 4th memtile push waits behind yx tasks).
- WINDOW OPEN (main A/B ms768: full12r23z3 vs full12r23l3 full12r23o1 full12r23o4 full12r23o3)
- WINDOW CLOSED
- A/B vs r23z3: l2 3 +0.007 (neutral), OPH1 -0.017 (neutral), r23o4 MSPLIT out_proj -0.08 / -0.10 / -0.12 (HIT, kept: best full12r23o4), r23o3 + out split +0.43 (DEAD: delays the SSM).
- r23z = try23 defaults (= r23o4 rebuilt with depth builds). r23w = MSPLIT incl. out, OUTPRE=0 / ZCH=0: pre-registered vs r23z: -0.02..+0.05 (out split -0.05, OUTPRE lost +0.02).
- WINDOW OPEN (main A/B final1: full12r23z vs full12r23o4 full12r23w)
- WINDOW CLOSED
- WINDOW OPEN (main A/B final2: full12r22z vs full12r23z)
- WINDOW CLOSED
- WINDOW OPEN (main sweeps r23z, r22z)
- WINDOW CLOSED
- WINDOW OPEN (main CLKLOAD, capped at 12 threads: r23/clkload12.sh)
- WINDOW CLOSED

# Round 23 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r23z** = r22z + KTL (r70b, record period as an argument: r23/mm_rot_k70p) + MSPLIT-P (down_b, mlp3_b, out_proj) +
YXSSM + YXR + OUTPRE / ZCH + OPH1 (build r23/try23.sh). A/B vs r22z in one process (60 x 20): **-1.894 / -1.885 / -1.875 ms**.
Quiet same-session sweep (1.801 GHz): d12 28.89 / d10 24.22 / d8 19.40 / d6 14.62 vs r22z 30.81 / 25.79 / 20.66 / 15.54.
BIT-IDENTICAL vs full12r9a. Text: GEMM 16,256 (128 free), EW 15,824 (560 free). Timeouts: 2 (both CP-order deadlocks).
| id | predicted | measured (p10 / median / mean ms) | verdict |
|---|---|---|---|
| KTL (r24/mm_e7k_r70b) | -0.50..-0.70 | -0.822 / -0.821 / -0.805 vs r22z | hit (above) |
| MSPLIT-P | -0.35..-0.6 (rank 2) | down_b + mlp3_b -0.16 / -0.17 / -0.21 (E7), -0.14 / -0.16 / -0.19 on KTL; out_proj -0.08 / -0.10 / -0.12 | kept (below NTWIN's bound) |
| MSPLIT-P on out (norm1) | - | +0.43 (with OUTPRE), +0.14 (without) | dead |
| SSM glue | -0.25..-0.45, gate tcall <= 1,880 | tcall unit 1,828-1,845 (+ sigk trips ~2,115); ssq2 + reck2 + ggk2 -> ~1,930 | gate FAILS without FSX; not built |
| E7'' + KTL R 66-68 | -0.15..-0.3 | R 68 CP feasible, 35 modelled bank conflicts / k-step | not built (bank-clean R 68-69 wanted) |
| MSPLIT-C | -0.15..-0.3 | only yx's first tile fits a plain / gelu pass epilogue (up_a gated, qkv, mlp0 gelu_pan) | not built |
| norm3 LN, OPPRE-W | -0.1..-0.18 / -0.04..-0.10 | - | not built |
| CLKLOAD | power vs thermal vs DPM | 12 spin (~99 W) 1.81; 6 vec (~96-100 W) 1.81; 12 vec (~99 W, Tctl 59-71 C) 1.408 GHz | not thermal, not package power alone |
| (main) YXSSM | -0.15..-0.35 | -0.415 / -0.414 / -0.398 | kept: the SSM overlaps yx |
| (main) YXR | -0.15..-0.35 | -0.379 / -0.377 / -0.364 | kept |
| (main) OUTPRE + ZCH / ZCH alone | -0.05..-0.15 / -0.03..-0.12 | -0.02..-0.03 / +0.01..+0.02 | small kept / off |
| (main) --l2-depth 3 / 6, OPH1 | 0 | +0.007 / -0.019, -0.017 | neutral |
- Open for the analysis passes: (1) YXSSM's timeline: yx wall 94 -> 164 us, SSM 276 -> 292 us (DDR contention while they overlap): is the SSM now
  DDR-bound in the overlap, and what does that do to the SSM glue price? (2) any other EW op whose units are finer than its producer
  (norm rows vs the last tile already done; out -> norm1 failed on the memtile queue)? (3) a bank-clean KTL at R 68 / 69.

# Round 24 (optimisation rounds), started 2026-10-11 (test machine Turbo, gpu_busy 0)
- Base full12r23z. Plan: (1) out -> norm1 split retry with out's pass sends after the SSM head fills; (2) more producer -> EW overlaps
  (YXSSM kind); (3) MSPLIT-C yx, norm3 LN glue; (4) step8 timeline marks for split ops; (5) OPPRE-W. Windows are logged below as before.
- analysis (pass 20): CP-SAT KTL R 69 --hard-banks, 4 workers nice 10, 1,800 s (CPU only, no NPU)
- WINDOW OPEN (main timeline x3: full12r23z, new split-op marks)
- WINDOW CLOSED
- WINDOW OPEN (main timeline x3: full12r23z, marks v2)
- WINDOW CLOSED
- WINDOW OPEN (main timeline DETAIL norms of r23z)
- WINDOW CLOSED
- WINDOW OPEN (main DETAIL timelines of r23z)
- WINDOW CLOSED
- r24a = r23z + MSPLIT out with MSDEFER (out's pass sends pushed by the SSM after yx's last drain group; SEQLOG checked: pushed at b*_ssm after b*_yx): BIT-IDENTICAL, asmlint 0, text unchanged. Pre-registered vs r23z: -0.03..-0.10 ms (norm1 rows 0-5 during out's pass 2; risk: r23w's unexplained +0.11 of the split itself).
- WINDOW OPEN (main A/B msdefer: full12r23z vs full12r24a)
- WINDOW CLOSED
- WINDOW OPEN (main timeline x3 full12r24a)
- WINDOW CLOSED
- WINDOW r24a A/B vs r23z (60 x 20, load 5.6-9.3: CP-SAT 4 workers + a background worker): +0.015 / +0.018 / +0.013: NEUTRAL (miss). The queue block is gone (SSM unchanged) but the split gains nothing: timeline x3 (col 0 marks) out + norm1 end -6.7 us per block after the SSM, up_a's first word +5.5 us later (the norm's late columns: rows 6-17 issued per column after the pass-2 wait). MSPLIT out stays off (MSDEFER kept as an env switch).
- r24b = r23z + YXH (main, new: yx steps head-major y0 x0 y1 | x1 y2 x2 | ..., one drain TAP [12][y|x][18][unit] split in 5 groups; the SSM head 0 waits step 0 only, heads 1-2 step 1): BIT-IDENTICAL, asmlint 0, SEQLOG read before dispatch. Pre-registered vs r23z: -0.04..-0.12 ms (SSM starts one yx step (~12 us) earlier; mixed gelu / plain rows in a step cost ~+1.4 us x 4 steps).
- WINDOW OPEN (main A/B yxh: full12r23z vs full12r24b)
- WINDOW CLOSED
- WINDOW r24b YXH A/B vs r23z (60 x 20, load 5.8-8.6): +0.466 / +0.486 / +0.484: DEAD (timeline next)
- WINDOW OPEN (main timeline x3 full12r24b)
- WINDOW CLOSED
- WINDOW OPEN (main DETAIL yx / ssm, r23z and r24b)
- WINDOW CLOSED
- r24c = YXH tile order with YXR drain groups (bisect of r24b's +0.48). Pre-registered vs r23z: +0.3..+0.5 if the order costs, ~0 if the finer groups do.
- WINDOW OPEN (main A/B yxhbis: full12r23z vs full12r24c full12r24b)
- WINDOW CLOSED
- r24c A/B (40 x 20, load 6.8-10.8): vs r23z -0.00 (+0.011 / -0.004 / -0.002), r24b +0.44 in the same window: YXH's ORDER is free (mixed gelu / plain rows cost nothing), its 7 drain TAPs per column cost +0.44 (INFERRED: a shim S2MM channel queues 5 tasks like the memtile MM2S; 7 yx drains + out's drain blocked the CP before the SSM's fills). r24d = YXH with 4 TAPs (head 0 | 1-2 | 3-5 | 6-11; the SSM waits before heads 0 / 1 / 3 / 6): BIT-IDENTICAL, asmlint 0, SEQLOG read. Pre-registered vs r23z: -0.05..-0.12 ms (SSM one yx step earlier; risk: a stall at head 6 on yx's last steps).
- WINDOW OPEN (main A/B yxh2: full12r23z vs full12r24d)
- WINDOW CLOSED
- r24d A/B vs r23z (60 x 20, load 6.2-10.3): +0.083 / +0.092 / +0.092: MISS (timeline next)
- WINDOW OPEN (main timeline x3 full12r24d)
- WINDOW CLOSED
- analysis: CP-SAT R 69 hard-banks ended (UNKNOWN in 1,800 s); next: R 68 / 69 diagnosis runs, 4 workers nice 10 (CPU only)
- r24d timeline (3 runs): the SSM starts 15 us earlier (gap -158 vs -138) but its wall grows 289 -> 308 us and yx 169 -> 175: the yx / SSM overlap is DDR-capped; YXH DEAD. MSPLIT-C (yx) priced, not built: pass 1 is B-bound (12 weight objects ~4.4 us), saves ~1/3 of a K = 768 first tile (~2.8 us / block, <= -0.035 ms), and needs an NRES ldn split whose E acquire races the previous op's chunk-wise release.
- r24e = r23z + STATS64 (norm statistics, LN and RMS, both 32-row halves as ONE 64-lane chain; replaces ln_stats / rms_rstd / rsqrt_i / mul3 / rowsum8 on the TXT18 path): BIT-IDENTICAL, asmlint 0, EW text 15,824 -> 15,728. tcall: stats per row block LN ~1,050 -> ~471, RMS ~650 -> ~406. Pre-registered vs r23z: -0.04..-0.08 ms (norm3 LN 2-input is compute-bound: ~46 cycles per unit x 216 units x 12; RMS / 3-input norms are input / DDR-bound: ~0).
- WINDOW OPEN (main A/B stats64: full12r23z vs full12r24e)
- WINDOW CLOSED
- r24e A/B vs r23z (60 x 20, load 5.9-8.4): -0.101 / -0.100 / -0.103: HIT (above), kept (best full12r24e).
- WINDOW OPEN (main timeline x3 full12r24e)
- WINDOW CLOSED
- WINDOW OPEN (main DETAIL up_a / qkv / mlp0 of r24e)
- WINDOW CLOSED
- WINDOW OPEN (main TL_ONLY timelines r24e)
- WINDOW CLOSED
- WINDOW OPEN (main TL_ONLY timelines r24e / r23z)
- WINDOW CLOSED
- WINDOW OPEN (main TL_ONLY boundary timelines r24e)
- WINDOW CLOSED
- TL_ONLY fast timelines (3-4 us polls, r24e, 3 runs x blocks 1-11, us): norm0 -> yx first 16.1, norm1 -> up_a 27.9 (a tile pair), norm2 -> qkv 14.2, norm3 -> mlp0 17.6; split producer pass-1 end -> norm first word 11-13 (down_b 29.9 - 16.7, out_proj 10.1 + 0.8); attn -> out_proj first 29.2, out_proj wall 45.9. (The 27 us-poll timeline's norm1 -> up_a shift r23z 24 / r24e 31 was a poll artifact: 28.6 / 27.9 at 3 us.)
- r24f = r24e + OPPRE2 (main: out_proj's A per attention head group during pass 1: pass-1 O drains as 4 head-group groups issued after the K/V fills, pass 0's drain its own group; out_proj waits pass 0 then head group g and fills its A part g into half 1): BIT-IDENTICAL, asmlint 0, SEQLOG read, shim BD peak 16 (limit). Pre-registered vs r24e: -0.10..-0.20 ms (the 1 MB A load leaves the attn -> out_proj boundary; risk: OPPRE's unexplained +0.18 of round 20).
- WINDOW OPEN (main A/B oppre2: full12r24e vs full12r24f)
- WINDOW CLOSED
- r24f A/B vs r24e (60 x 20, load 6.2-11.1): -0.044 / -0.039 / -0.045: below the range (miss low), a gain; timeline next
- WINDOW OPEN (main TL_ONLY attn boundary r24f)
- WINDOW CLOSED
- analysis: R 69 hard-banks with a scaled hint: UNKNOWN (1,500 s, 2 workers); FSX II 30 / 31: none in 700 s each. Now: R 69 soft (2 workers, 1,200 s) + FSX II 32-35 (2 workers, <= 600 s each), CPU only
- r24f timeline (3-4 us polls): attn -> out_proj first 28.3 -> 23.7 us (-4.6), attention wall unchanged: out_proj's sends were pushed after the last head-group wait. r24g = OPPRE2 v2 (sends before the waits): BIT-IDENTICAL, asmlint 0, SEQLOG read. Pre-registered vs r24e: -0.08..-0.16 ms (vs r24f -0.04..-0.12).
- WINDOW OPEN (main A/B oppre2b: full12r24e vs full12r24f full12r24g)
- WINDOW CLOSED
- r24g A/B vs r24e (60 x 20, load 5.8-7.4): -0.161 / -0.150 / -0.153: HIT, kept (best full12r24g); r24f again -0.042 / -0.043 / -0.046.
- WINDOW OPEN (main TL_ONLY attn boundary r24g)
- WINDOW CLOSED
- r24h = r24g + ZDROP (main, new: under SRES the SSM's output ring carries h only (2 KB per chunk, fractional slots by out_begin / out_end) and the drain writes [h lo, h hi]: the 1 KB z part per chunk was stale ring bytes nobody reads, 1.77 MB of DDR writes per block during the DDR-capped yx / SSM (~23 MB in ~336 us)): BIT-IDENTICAL, asmlint 0, EW text 15,728 -> 16,000. Pre-registered vs r24g: -0.10..-0.30 ms.
- WINDOW OPEN (main A/B zdrop: full12r24g vs full12r24h)
- WINDOW CLOSED
- r24h A/B vs r24g (60 x 20, load 5.5-8.9): +0.054 / +0.052 / +0.048: MISS (the SSM is compute-bound per chunk: v1 added two minsize out_begin / out_end calls per chunk). r24i = ZDROP v2 (a fixed 3-chunk slot pattern, acquire 3/3/2 release 2/3/3, inline): BIT-IDENTICAL, asmlint 0, EW text 15,872. Pre-registered vs r24g: -0.05..-0.20 ms.
- WINDOW OPEN (main A/B zdrop2: full12r24g vs full12r24h full12r24i)

## Analysis note (pass 20; corrected - the first version said channel-bound, not DDR: wrong, see below)
- The SSM span is DDR-read-capped. In its 288 us span the SSM reads 13.57 MB and yx (overlapping 137 of its 169 us)
  ~2.0 MB: 15.5 MB = 54 GB/s, at the aggregate read cap 55-59 GB/s (round 19 DDRCAP: 2 streams per column still cap there).
  Writes in the span ~8.2 MB (28 GB/s). Channel 0 is about as close (the SSM's 1,695,744 B per column alone = 2,066 cycles per unit at
  3.8 B / cycle, plus yx's A share); static compute (2,115 per unit, ~252 us) is below both. The first version compared reads + writes over
  the whole 320 us window against norm0's r+w and wrongly cleared DDR.
- Consequences: a compute cut (SSM glue, FSX) is ~0 now (-0..-0.05). What pays is fewer DDR READ bytes inside the SSM span:
  - SSPM: the SSM params are the same for all 8 columns (params_w + q x ps, no c term) and are read 8 times: 12 x 12,288 x 8 =
    1.18 MB per block. Read once and broadcast (one shim channel -> the 8 EW cores' unused S2MM 1): -1.03 MB, ~-19 us per block
    raw, -0.1..-0.2 ms.
  - kb_pad: each params fill is 12,288 B for 10,752 B of data (14 slots): 147 KB per block of padding reads: -0.02..-0.03.
  - ZDROP (writes) pays only if writes take read bandwidth (DRAM read / write turnaround); ZDROP v2 is exactly that test.

## Analysis note (pass 20) - the yx / SSM window is input-channel bound, not aggregate-DDR bound
- Every EW op reads through ONE shim MM2S channel per column: channel 0, packet-shared with the GEMM A loads (MACHINE M10.3), ~3.8 B /
  cycle (M8.7). The SSM reads 1,695,744 B per column per block on it (12 heads x (12,288 params + 129,024 y / x), plan8 r23z), plus
  yx's A (124 KB per column) when they overlap: 247 + 18 us at 3.8 B / cycle. Per SSM unit that is 2,066 cycles of input against
  2,115 static compute (tcall): the two are equal, so in situ (2,416 per unit in r23z, 2,336 in r22t) is their imperfect overlap
  through a 15 KB ring. r24d fits this: starting the SSM earlier put more of yx's A on the same channels and the SSM slowed (+19 us).
- Aggregate DDR in the window: 16.0 MB read + 8.9 MB written in ~320 us = 50 + 28 GB/s, under the >= 84-90 GB/s r+w seen in norm0's
  own span. Writes leave on shim S2MM 1 (EW output), a different channel from the reads: ZDROP's 1.77 MB of writes pays only if
  aggregate DDR binds, which these numbers do not show; adding compute per chunk costs (r24h) because compute is co-binding.
- So a compute cut alone (SSM glue: -185 static per unit) lands only down to the channel (2,066 + yx's share): about half, -0.06..-0.15
  ms. A channel-0 byte cut alone lands only down to compute. Both together pay: SSM glue + the params (147 KB per column per block)
  off channel 0, e.g. into the EW core's unused S2MM 1 from shim MM2S 1 (free on columns 3-7, M10.3): -0.2..-0.35 together (routing
  UNVERIFIED). Details in PROPOSALS rev 20.
- WINDOW CLOSED
- r24i A/B vs r24g (60 x 20, load 6.2-8.6): +0.007 / -0.002 / -0.009: NEUTRAL (r24h again +0.05). ZDROP OFF: the yx / SSM overlap is read-bound, not write-bound (INFERRED: SSM reads ~46 GB/s + yx B ~8 at the read cap).
- WINDOW OPEN (main TL_ONLY + DETAIL out of r24g)
- WINDOW CLOSED
- analysis: R 69 soft (2 w, 1,200 s): no solution at all; FSX II 32 / 33 none in 600 s each (stopped). Now: R 68 soft from the r68 solution (35 conflicts) as hint, 4 workers nice 10, 1,200 s, CPU only
- WINDOW OPEN (main TPROF dump of full12r24tp: timing build, outputs wrong)
- WINDOW CLOSED
- WINDOW OPEN (main SSMIN probe full12r24s = r24g + EW -DSSMIN, analysis rev 20 rank 2)
- WINDOW CLOSED
- **SSMIN (analysis rev 20 rank 2) MEASURED** (full12r24s = r24g + EW -DSSMIN: stamps around the SSM's chunk / params in_need and the
  whole chunk, records in the stale z pad, r24/ssmin.py; 3 frames, all 12 SSM ops x 8 columns, 1.81 GHz): per chunk in_need **52**
  cycles (the call alone is ~44: net wait ~8), params in_need 28 per chunk (~500 per head), whole chunk **2,273** (12 ops within
  2,268-2,282; columns within 49-59 in_need). The SSM is **compute-bound**, not channel-bound: the input ring never runs dry. A compute
  cut lands until the channel floor (2,066 per unit): glue (-185 static) should land nearly whole, ~-0.2..-0.27 ms; a channel-0 byte
  cut alone is worth ~0 today. (SSM span 216 x 2,273 / 1.81 GHz = 271 us + params ~3 = the measured ~290 with the start.)
- SSM glue by chunk pairs (ssq2 / reck2 / ggk2): not built: the pair loop needs ~4.5-5 KB of doubled chunk buffers (g_xc, g_gx, g_ga, g_e2, a second g_tsp, panels); EW L1 has ~2.5 KB free (r24g map: bank 0 128 B after .aie.bank0, bank 2 1,472 B before .aie.bank2 + 960 B after it; banks 1 / 3 full; stack 4,864 with ~3.2 KB deepest). Open: is there a single-chunk form of the glue cut, or L1 to free (the 15 KB input ring holds the 12 KB AL1 params)?
- WINDOW OPEN (main SSMIN skew probe full12r24s2)
- WINDOW CLOSED
- WINDOW OPEN (main SSMIN skew probe full12r24s3 = s2 + SSMPRE)
- WINDOW CLOSED
- **SSM column skew (MEASURED, r24s2 = r24g + SSMIN with the core timer in the record)**: the SSM 8 columns start ~1.6 us apart in
  column order (spread 11-12.8 us per block) and end 8-9.8 us apart: the CP issues each head's two fills (params + chunk) per column
  after the yx wait. Out's tile 0 then waits for the slowest column (its bias via the B multicast: TPROF mid 4.5 us median, max 9.2).
  **SSMPRE** (head 0's params fills for all columns before the yx wait, one fill per column after it; r24s3): start spread 4.1-4.5 us,
  end spread 2.8-3.9 us. A post-wait shim fill costs the CP ~0.55-1 us here, not DEAD_ENDS C8's 0.10 (BDPRE may be worth re-pricing).
- r24j = r24g + SSMPRE: BIT-IDENTICAL, asmlint 0, SEQLOG read (same waits). Pre-registered vs r24g: -0.04..-0.08 ms (the slowest column
  ends ~5.7 us earlier per block).
- WINDOW OPEN (main A/B ssmpre: full12r24g vs full12r24j)
- WINDOW CLOSED
- r24j A/B vs r24g (60 x 20, load 5.9-9.9): -0.069 / -0.073 / -0.085: HIT, kept (best full12r24j).
- analysis: CP-SAT window closed. R 68 soft repaired from the earlier solution (4 w, 1,200 s): still 35 conflicts (prologue 41): no bank-clean R 68-69 found. No more CP-SAT load this pass.
- r24k = r24j + CPPRE (main: post-wait shim fills configured before the wait, pushed after: SSM head-0 chunk fills, the norms input fills incl. both MSPLIT parts; lowered post-sync run per column = maskwrite + push instead of blockwrite patch maskpoll maskwrite push): SEQLOG unchanged. Pre-registered vs r24j: -0.02..-0.08 ms (SSM start spread ~4 -> ~1.5 us; norms DDR-bound: partial).
- WINDOW OPEN (main A/B cppre: full12r24j vs full12r24k)
- WINDOW CLOSED
- r24k A/B vs r24j (60 x 20, load 1.8-4.4): -0.097 / -0.093 / -0.091: HIT (above), kept (best full12r24k). CPPRE on OPPRE2's A parts
  (configured one part ahead or all four): design8's static shim BD peak 20 > 16 at out_proj: built, never dispatched, gated off.
- **Optimisation-round response to analysis pass 20**: SSMIN (MEASURED on r24s, r24s2, r24s3) shows the SSM's input ring never runs dry:
  in_need 52 cycles per chunk of which ~44 is the call (net ~8), params in_need 28 per chunk, whole chunk 2,273 = compute (static ~2,115 +
  ~150). So in r24g the SSM is NOT input- or DDR-bound once running; 54 GB/s in the span can sit near the cap without binding. What
  bound it in the overlap was column skew from the CP (fixed by SSMPRE / CPPRE: -0.16 together). Consequences: SSPM-B (params once) is
  worth ~0 for the SSM itself unless something else in the span (yx's drains) is DDR-starved; the SSM glue IS worth its static price
  (~-0.2), but blocked by EW L1 (~2.5 KB free vs ~5 KB for chunk pairs). Open: re-price with SSMIN; a single-chunk glue form or freed L1
  would be the lever. CPPRE's measured post-wait cost per fill: blockwrite + patch + maskpoll + maskwrite + push ~0.55 us vs push-only.
- WINDOW OPEN (main clock + final sweeps r24k, r23z)
- WINDOW CLOSED
- WINDOW OPEN (main r24z build with depth sequences + its sweep)
- WINDOW CLOSED
- r24z = r24/try24.sh defaults (= r24k) with depth builds: BIT-IDENTICAL, asmlint 0, text GEMM 16,256 EW 15,728. Pre-registered vs r23z: -0.40..-0.45 ms (sum of A/Bs -0.10 -0.15 -0.07 -0.09).
- WINDOW OPEN (main A/B final24: full12r23z vs full12r24z)
- WINDOW CLOSED
- WINDOW OPEN (main final sweep r24z + clock)
- WINDOW CLOSED

# Round 24 (optimisation rounds), final verdicts keyed by proposal id (2026-10-11)
Best **full12r24z** = r23z + STATS64 + OPPRE2 + SSMPRE + CPPRE (build r24/try24.sh). A/B vs r23z in one process (60 x 20):
**-0.476 / -0.483 / -0.508 ms**. Sweep (1.808 GHz, load 1.4-4.1): d12 28.43 / d10 23.85 / d8 19.10 / d6 14.38 vs r23z 28.95 / 24.25 /
19.47 / 14.62. BIT-IDENTICAL vs full12r9a. Text GEMM 16,256, EW 15,728 (656 free). Timeouts: 0.
| id | predicted | measured (p10 / median / mean ms) | verdict |
|---|---|---|---|
| out -> norm1 split (MSDEFER) | -0.03..-0.10 | +0.015 / +0.018 / +0.013 | neutral, dead |
| (main) YXH | -0.04..-0.12 | 7 TAPs +0.47; order alone 0.00; 4 TAPs +0.09 | dead |
| (main) STATS64 (norm3 LN glue, part) | -0.04..-0.08 | -0.101 / -0.100 / -0.103 | kept |
| OPPRE-W -> (main) OPPRE2 | -0.08..-0.16 (v2) | v1 -0.04; v2 -0.161 / -0.150 / -0.153 | kept |
| (main) ZDROP | -0.10..-0.30 / -0.05..-0.20 | +0.05 (calls) / +0.007 / -0.002 / -0.009 | off |
| SSMIN (rev 20 rank 2) | probe | in_need 52 (net ~8), chunk 2,273: compute-bound | done |
| (main) SSMPRE | -0.04..-0.08 | -0.069 / -0.073 / -0.085 | kept |
| (main) CPPRE | -0.02..-0.08 | -0.097 / -0.093 / -0.091 | kept |
| MSPLIT-C yx | rev 18 -0.15..-0.3 | priced <= -0.035 (pass 1 B-bound) | not built |
| nrm_ln 64-lane | - | priced <= -0.04 (st slot binds at II 11) | not built |
| SSM glue (pairs) | - | needs ~5 KB EW L1, ~2.5 KB free (+ stack?) | not built |
| SSPM-B (rev 20 rank 1) | -0.10..-0.20 | SSMIN: SSM not input-starved: ~0 | not built |
| DDRCEIL, CLKLOAD2 | probes | - | not run |
- Open for the analysis passes: (1) re-price the SSM with SSMIN (compute 2,273 per chunk, waits ~8): the glue at its static price and where to find
  ~5 KB of EW L1 (stack deepest C++ 1,792 of 4,864 B; asm frames unaudited); (2) a BD-safe way to configure OPPRE2's A parts early (the
  static shim BD peak reached 20 at out_proj) and an audit of the remaining post-wait shim fills (r24/cprest.py); (3) KTL R 68-69 with
  more workers in a quiet window.

# Round 25 (optimisation rounds), start 2026-10-11
- Base full12r24z (28.43 ms d12). Plan: (1) post-wait fill audit (r24/cprest.py) + CPPRE where the BD budget allows; (2) MSPLIT-C up_a;
  (3) out tile 0 bias wait; (4) C-buffer drain between tiles; (5) SSM glue once EW L1 is found. Every NPU window is logged here as WINDOW OPEN / CLOSED; CP-SAT (<= 12 workers) runs outside them.
- analysis (pass 21): BUILDING full12r25s (= r24z + -DSSMIN) then full12r25g (= r24z + -DSSMIN -DSSM_NO_GATES: a timing
  twin without sigk + ssq, wrong outputs), via r24/try24.sh NOREG=1 NOTIME=1, one after the other. No other builds in npu-tapnext meanwhile
  (shared generated headers, DEAD_ENDS H23); the NPU window (ssmin.py, 2 dispatches) is logged below.
  **SSMCUT pre-registration** (per SSM chunk, mean over columns; r24s: chunk 2,273, in_need 52):
  - H_C (compute-bound, no read cap within reach): the twin's chunk drops by the removed in-situ compute (sigk + ssq ~1,000-1,100)
    to ~1,150-1,350, in_need stays < 100.
  - H_R (the span's reads within ~6 % of the 55-59 GB/s cap, rev 20): the chunk cannot drop below the read floor, ~2,050-2,250
    cycles at today's bytes, and in_need rises to fill the difference (>= 700).
  - In between (chunk 1,350-2,050 with in_need rising): a read floor at that level; it caps the glue at (2,273 - floor).
- WINDOW OPEN (analysis SSMCUT: ssmin.py on full12r25s, full12r25g; one dispatch sequence each)
- WINDOW CLOSED
- **SSMCUT result (analysis, MEASURED, r24/ssmin.py, 12 blocks x 8 columns, 215 chunks each)**: full12r25s (r24z + SSMIN) chunk 2,289,
  in_need 71, params 28 (= r24s). Twin full12r25g (no sigk + ssq, ~1,000 cycles of compute removed): **chunk 2,034, in_need 529**.
  Neither H_C nor H_R as stated: removing ~1,000 cycles bought 255; the rest became input waits. **Both readings hold**: today the SSM
  is compute-bound (2,289 vs an input floor), and the input floor is **~2,030 cycles per chunk** = 7,851 B per column per chunk
  at 3.86 B / cycle, i.e. exactly one shim MM2S channel (M8.7: 3.8) and, over 8 columns, ~56 GB/s (the read cap): the two limits
  coincide, so this run cannot tell channel from aggregate DDR; it does not need to for the prices below. Twin end spread 0.5-0.9 us
  (vs 1.6-2.4): input-paced columns finish together.
  - SSM glue (-185 static per chunk) lands down to ~2,030: -0.15..-0.25 ms (full static price -0.27 minus the approach to the floor).
  - SSPM-B / KPAD: ~0 alone (compute binds); after the glue they lower the floor ~2,030 -> ~1,860-1,880 (params = 683 of 7,851 B per
    chunk) and let FSX-type cuts land: worth -0.05..-0.15 only together with more compute cuts.
- analysis: CP-SAT KTL R 69 --hard-banks, 12 workers nice 10, 1,800 s (CPU only, scalar)
- analysis: CPU-only design run (r26/design8_bd.py, a BD-logging copy of design8 with CPPRE_OP=1, r24z flags; kernel objects regenerated identical in build/), ~5 min; no NPU
- r25a = r24z + P0FREE (the attention Q + pass-0 K / V fills in pass 0 drain group, so out_proj "#p0" wait frees 5 shim BDs) + CPPRE_OP
  (OPPRE2 A parts configured one ahead, part 0 after "#p0"): static shim BD peak 16 (was 20 with CPPRE_OP alone), SEQLOG same waits,
  asmlint 0. Pre-registered vs r24z: -0.01..-0.04 ms (the last part column spread ~3.9 -> ~1.4 us per block).
- WINDOW OPEN (main regress + A/B r24z vs r25a)
- correction: the window above was not open yet: the r25a build overlapped the r26-bd design run and was still building its depth
  phases. r25a changes only the runtime sequence, so every r25a elf (main + phases, after csplit) was checked byte-identical to r24z
  before any dispatch.
- WINDOW OPEN (main: clock, regress r25a, A/B r24z vs r25a; elfs + CDO of r25a byte-identical to r24z)
- analysis: r26-bd (CPU design run) done. The 12-worker CP-SAT is frozen (SIGSTOP) during every WINDOW OPEN and resumed when it closes (r26/cpgate.sh).
- analysis: CP-SAT restarted with a deterministic budget (12 workers nice 10, frozen during the windows by r26/cpgate.sh): R 69 hard banks, then R 68 hard banks
- WINDOW CLOSED
- r25a regress BIT-IDENTICAL vs full12r9a; clock 1.806 GHz (load 13.5: CP-SAT 12 w + a background worker). A/B vs r24z
  (60 x 20, load 13.5 -> 9.0): p10 -0.003 / median +0.009 / mean +0.176 (p90 32.4 vs 32.1: bimodal noise): NEUTRAL, MISS of -0.01..-0.04.
  The last OPPRE2 part's column spread (CP) does not gate out_proj. P0FREE stays as a BD-headroom switch (frees 5 shim-0 BDs from
  out_proj's "#p0" wait on), CPPRE_OP off.
- main: out's tile 0 bias wait (plan item 3) priced ~0 from r24tp (MEASURED, r25/outt0.py on build/tp24.npz): the slowest SSM column
  (7) waits 0.03 us for its bias; columns 0-6 wait 9.2..0.7 us = the column skew, i.e. for column 7's k-step 10. The op ends with
  column 7, so the bias wait is off the critical path; only the SSM end skew (~3 us after SSMPRE + CPPRE) is.
- main: priced, not built: **MSPLIT-C up_a ~0** (rank 4): r24e TL norm1 wall 61.3 us = 503 cycles per unit for 2,048 B of input
  = 4.07 B/cycle per column = ~59 GB/s over 8 columns: norm1 sits at the read cap, and up_a's first pair's B (2 tiles x 13 objects
  x 4,608 B x 3 rows = 360 KB, of which ~70 KB is prefetched today) would be read inside norm1's span (+~5 us), against ~6.3 us of
  pass-1 compute moved; plus the gated pass epilogue text (GEMM 128 B free). The YXH rule applies. Same for qkv / mlp0.
- main: own find, probe first: **RRES** (the residual X' of a norm kept on chip for the next norm). Every norm reads one input
  that an earlier norm wrote (norm1: R0, norm2: R2, norm3: R3, norm0: R5) = 1/2 of the 2-input and 1/3 of the 3-input norms' DDR
  bytes, and the norms sit at the read cap. During the GEMM before each norm one A slot region (124 KB / column, all 8 columns) is
  free: norm0 <- half 1 (mlp3_b holds half 0 + region 3), norm1 <- region 3, norm2 <- half 1, norm3 <- region 3. A prefetch task on
  the existing A load flow (shim MM2S 0 -> memtile S2MM, inside the GEMM's span, which runs at 13-24 GB/s) could park 56 % of the
  residual there and a memtile MM2S -> EW S2MM 1 stream (unused) feed it to the norm. Gate: a timing twin RRTWIN (norms do not read
  their residual at all; outputs wrong; read the norm walls only, TL_ONLY) = the upper bound.
  **Pre-registered (TL_ONLY norm walls, r24z -> twin):** H_IO (input bytes bind): norm1 61 -> 35-42, norm3 71 -> 45-55 (LN compute),
  norm0 89 -> 60-68, norm2 90 -> 60-68 us. H_C (compute binds): unchanged within 3 us. Building full12r25t now (CPU only).
- WINDOW OPEN (main: r25t twin dispatch check, TL_ONLY norms r24z vs r25t)
- WINDOW CLOSED
- **RRTWIN MEASURED** (full12r25t = r24z + RRTWIN: every norm skips its residual input (an earlier norm's X'), params n_in - 1;
  elfs / CDO byte-identical to r24z; one 10-dispatch check first, no hang; TL_ONLY norm0..3 + yx, up_a, qkv, mlp0, 3 runs each,
  load 13.7 -> 7.3, same session). Norm walls r24z -> twin (us per block):
  | norm | inputs | r24z | twin | delta |
  | norm0 (RMS) | 3 -> 2 | 90.1 | 67.0 | -23.1 |
  | norm1 (RMS) | 2 -> 1 | 63.4 | 60.5 | -2.9 |
  | norm2 (LN)  | 3 -> 2 | 91.7 | 73.7 | -18.0 |
  | norm3 (LN)  | 2 -> 1 | 71.7 | 70.3 | -1.4 |
  Verdict vs the pre-registration: H_IO holds for the 3-input norms (inside the 60-68 range), H_C for the 2-input ones (outside the registered
  35-42 / 45-55: mispriced; they sit at a COMPUTE floor of ~480 (RMS) / ~560 (LN) cycles per unit, which coincides with their
  2-input I/O rate (2,048 B / 4.07 B per cycle = 503)). So: (a) RRES pays only on norm0 / norm2: upper bound -41 us per block =
  -0.49 ms per step if the residual came free; with 124 KB of the 221 KB residual per column parked in a free A-slot region,
  ~-0.25..-0.3; (b) the 2-input norms (and the 3-input ones after RRES) are bound by ~500-560 cycles of per-unit COMPUTE: tcall's
  per-unit static (nrm_rms 117 + stats64 406 / 12 + glue) is far below that, so ~250+ cycles per unit are unaccounted. Twin step
  dispatch 27.2 vs 28.4 ms (the attention's data-dependent path changed too: not a price).
- Open questions (pass 21), in this order: (1) where do ~500 (RMS) / ~560 (LN) cycles per norm unit go in situ (NPROF-type
  stamps or tcall of norm_rows_k incl. n1k _step / in_need / out_begin / the NRES panel path)? Every norm is at that floor now; each
  -100 cycles per unit is ~-12 us per norm per block once bytes allow. (2) RRES feasibility next to the SSM glue: both want EW L1
  (a second 1 KB x 2 ring for S2MM 1) and EW text (656 B free). Channels checked in cs_r24z/in.mlir: EW cores use S2MM 0, MM2S 0 / 1
  (S2MM 1 free); memtiles 0-2 use all six S2MM, MM2S 0-2 (MM2S 3-5 free); memtiles 3-7 S2MM 0-3 + 5, MM2S 0-1 (S2MM 4, MM2S 2-5 free).
  The prefetch can ride the A load flow (memtile S2MM 0 tasks); the residual feed needs memtile MM2S -> EW S2MM 1 (northbound route
  unverified). DDR in the preceding GEMM spans: mlp3_b / down_b (J2 said ~66 GB/s r+w there; estimated ~47): to be checked.
- main: building full12r25n (= r24z + EW -DNIO2: the norm-unit stamps, timing build) for question (1) above; per-unit numbers below
- WINDOW OPEN (main: NIO2 stamps full12r25n, 3 frames)
- WINDOW CLOSED
- **NIO2 MEASURED** (full12r25n = r24z + EW -DNIO2 -DNIO2_NONRM: round 18's norm-unit stamps minus the two around the nrm call, which
  overflowed the EW text by 32 B; text EW 16,368; asmlint 0; r25/nio.py, 3 frames, all norm units, 8 columns; load 12.6): cycles per
  unit incl. stamps, mean over columns (columns within +-2 % / monotonic in column for pass 1):
  | mode | pass-2 unit (out acquire + NRES pring acquire + nrm + releases) | pass-1 unit (in_need incl. the producer / DDR wait + n1k step + in_done) |
  | RMS (norm0 3-in, norm1 2-in) | 287 | 720 |
  | LN (norm2 3-in, norm3 2-in)  | 364 | 695 |
  Pass 1 includes the input waits (MSPLIT norms start before their producer ends), so it is not compute. Pass 2 is: nrm_rms static 117 /
  nrm_ln 177 against 287 / 364 in situ: **~170-190 cycles per unit of pass-2 glue or wait** (out ring acquire, the 2-slot NRES panel
  ring's acquire - is MM2S 1 draining 576 B per unit in time? -, the indirect call, ring-wrap copies, releases, 2 stamps). With
  RRTWIN's compute floor (norm1 at 1 input: 507 cycles per unit) pass 2 is ~57 % of a norm unit. Open: decompose pass 2 statically
  (tcall on norm_rows_k of build/full12r24z.prj elfs_main_core_0_5) so the in-situ excess over static = the panel-ring / out-ring
  waits; that names the lever (a deeper panel ring = 1,152 B more EW L1, or batching the per-unit lock work).
- main: plan item 4 priced (MEASURED hands, r25/hands.py on build/tp24.npz block 11, medians per tile, tiles 0..n-2): up_a 6.4,
  up_b 7.1, mlp0 5.2, qkv 3.8, down / mlp3 0.5-1.2 each, yx 3.0 / out 4.3 (skew-inflated) us per block: ~28 us per block, ~0.33 ms per
  step in total. Double-buffered C: impossible (+18 KB GEMM L1, 124 B free). Partial release: the next tile's first k-step (ZLOAD)
  writes rows 0-47 = C part 1 within its first third, so any gain needs IRON's C acquire split into part-wise locks taken inside the
  k-step = LATEC's form, which hung (round 21, cause inferred). <= -0.1..-0.15 ms at L effort and hang risk: parked unless a
  safe lock order is found (e.g. why LATEC hung: the C join's row order?).
- CPU-only job for a gap in the CP queue: attkb at II 32 (resource bound 31; r22 found none at 600-1,800 s with
  2-4 workers; II 33 needed --life 1): `r17/cp_attks.sh ii32 --ii 32 --life 1 --maxsrc sr --time 3600` (12 workers), and the same with
  --stages 4. Worth ~-0.035..-0.065 ms if found (II 34 -> 33 gave -0.035).
- main: **RRES plan** (design / plan side first; the EW kernel part waits for an L1 verdict):
  - only norm0 and norm2 (3-input, byte-bound; norm1 / norm3 are at their compute floor). Residual = stream 2 of both (R5 / R2 sort
    last), so the main ring's TAP for rb < RRB is the 2-stream TAP and for rb >= RRB the 3-stream one (MSPLIT parts: 0-5 | 6-9 | 10-17).
  - park: during mlp3_b (resp. down_b) memtile half 1 is free on every column (mlp3_a / down_a released it; mlp3_b / down_b use half 0
    + region 3; the next half-1 users are out's ldz (SSM) and the attention's Q (AQPRE), both after the norm). A memtile S2MM 0 task on
    the existing A load flow (acquire E[1] by 12, release a new lock RF) + one shim fill per column of R5 / R2 rows rb 0-9 (120 units,
    122,880 of 124,416 B) pushed after the GEMM's own A loads: ~983 KB per block in a k-step-bound GEMM span.
  - feed: a new packet flow memtile MM2S (3 on memtiles 0-2, 2 on 3-7: both free) -> EW S2MM 1 (free), a runtime task per norm
    (acquire RF, release E[1] by 12) pushed before the norm's fills; EW side a 2 x 1 KB ring (locks 6 / 7) on S2MM 1.
  - EW kernel: params header word 2 = RRB; pass 1 takes stream 2 from the R ring for rb < RRB (one lacq / lrel per unit), unit = 2 KB
    on the main ring there. Text ~100-150 B (656 free).
  - price: linear in bytes from RRTWIN: norm0 -23 x 10/18 = -13, norm2 -10 us per block = ~-0.27 ms per step (if the feed keeps up:
    1 KB per ~560-cycle unit, and the park does not slow mlp3_b / down_b).
  - **L1: EW free = 1,472 B (data tail 0x7A640..0x7AC00) + 960 B (bank 2 tail) + 128 B (bank 0): no 1 KB slot pair fits without the
    stack (0x1300 = 4,864 B, deepest C++ 1,792 + unaudited asm frames).** Open: include these 2 KB in the EW L1 audit next to the SSM
    glue's ~5 KB; if both cannot fit, RRES (~-0.27) vs glue (-0.15..-0.25) decides the order.
- main: RRES L1 found without the stack: Worker data_size 27,648 -> 26,240 (.data 0xA00 + .bss 0x5C40 = 26,176 used) and a 2 KB
  IRON Buffer (the R ring) right after core_data; the bank-2 linker region then shrinks 5,120 -> 4,480 B, still >= .aie.bank2's
  4,160. That uses the data tail (1,472) and most of the bank-2 tail (960 -> 320): so the SSM glue's ~5 KB must come from the stack
  (0x1300, deepest C++ 1,792 + asm frames) or the arena. RRES is implemented via env RRES=norm0,norm2, EW -DRRES.
- main: building full12r25r (= r24z + RRES=norm0,norm2, EW -DRRES). Round 25 is the last round: RRES gets one bounded attempt.
- main: **RRES stopped at the build** (round 25 is the last round): full12r25r routes (memtile -> EW S2MM 1 as a
  Flow with compiler-assigned channels; pinning dst_channel=1 is refused) but aiecc fails on (a) "Too many simultaneously active
  buffer descriptors on tile (0,1)" (the park + feed tasks on memtile 0, beside 3A / MSPLIT / NRES tasks) and (b) IRON placed the
  2 KB R ring so that .aie.bank0 / .aie.bank2 overflow by 1,024 / 3,136 B (pin it: mem_bank=2 after core_data, or a linker-aware
  address). Code kept behind RRES= / -DRRES (off by default; r24z plans unchanged, r25/schedchk.py). Price stands at ~-0.27 ms
  (RRTWIN bound -0.49). Final best = full12r24z (no kept change this round).
- main: building full12r25z (= r24/try24.sh defaults from the current checkout, depth builds) to check the best build reproduces byte-identically; its sweep dispatches at the end , then DAVIS d12 + d10 on the best build (~15 min NPU).
- analysis: KTL CP-SAT STOPPED (final pass; R 69 hard banks at 12 workers: no solution; R 68 run cut short). No CP-SAT load now.
- Open question (analysis): CLKLOAD2 (why CPU load slows the NPU): ~4 min, needs a QUIET box (no builds, no other
  CPU load) and the NPU for 5 short r7/clk.py runs under 12 threads each of: none, numpy matmul (AVX-512, L2/L3), 64x64 matmul (L1),
  AVX2 matmul, np.copyto (DRAM). Script r25/clkload2.sh. It runs once the box is quiet after the last build and sweep; WINDOW OPEN / CLOSED are logged.
- WINDOW OPEN (main: DAVIS 10 videos d12 + d10 on full12r24z, ~15 min; r25z sweep done: d12 28.42 / d10 23.83 / d8 19.09 / d6 14.40, 1.808 GHz; r25z bins / elfs / CDO byte-identical to r24z)
- WINDOW CLOSED
- WINDOW CLOSED (DAVIS done): d12 70.84 (92.57), d10 69.90 (91.02) on full12r24z - identical to round 24. Main round 25 ends; final best full12r24z.
- WINDOW OPEN (analysis CLKLOAD2: r25/clkload2.sh, box quiet after round 25 ended; ~5 min CPU load + NPU clock probes)
- WINDOW CLOSED
- WINDOW OPEN (analysis CLKLOAD2 rerun: the first run had a wrong xclbin name, no clock data)
- WINDOW CLOSED
- **CLKLOAD2 result (analysis, MEASURED, r25/clkload2.sh, 12 threads per kind, r7/clk.py median fit; r26/clkload2.out)**: none 1.809 GHz
  (27 W); numpy 512x512 matmul (AVX-512, cache-resident) **1.408** (99 W); the same matmul with OpenBLAS forced to AVX2 **1.408**
  (99 W); 64x64 matmul in L1 **1.584** (99 W, noisy fit); np.copyto on 64 MB arrays (DRAM bandwidth) **1.808** (42 W). With round 23:
  12 spin threads at 99 W and 6 matmul threads at 96-100 W keep 1.81. So DRAM traffic does not slow the NPU, and vector width does
  not matter; what drops it is many threads of floating-point compute with the package at its ~100 W limit (scalar spinning at the
  same power does not). Mechanism (power vs current arbitration in the SMU) UNVERIFIED; the trigger is measured.
