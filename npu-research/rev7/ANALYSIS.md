# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 7, 2026-10-10. State analysed: round 11's `full12r11d` (52.07 ms per depth-12 dispatch, Turbo, gpu_busy 0, AJ 70.84),
the full-step timeline `npu-tapnext/build/tl_full12r11b_d12.json` (r11b: GELU_MS, before P43, 52.43 ms), the PROF image
`build/g11-r11gp.npy`, the r11d depth sweep and `RESULTS_FROM_MAIN.md` round 11. Revision 6 is in `rev6/`. These parts are
unchanged and not repeated here: the norm-fold math and numerics plan (rev6 section 4) and the attention breakdown (rev6
section 5). Tags: [DOC] vendor source or the deployed binary, [MEAS] measured by the main agent, [INF] model or arithmetic,
[GEN] what the main agent's generator (`npu-tapnext/r11/msched.py`, run unmodified on the CPU) produces for an op list.
Cycles at 1.81 GHz.

New this revision (`tools/`, outputs in `r12/`; no device, read-only containers):
- `sim14.py`: sim13's model on the r11b timeline + the g11-r11gp composition; P43 as its measured delta (`r12/sim14_r11b.out`).
- `g1sched.py`: builds the op lists of the fused GELU epilogues, the gated product pass, the SSM sigmoid / recurrence and the
  attention exp, and asks msched.py for a schedule at each II from ResMII up (`r12/g1sched.out`). This replaces my hand slot
  counts: the II it reports includes the itinerary ports, the round-11 visibility rule and register allocation.
- `r12/g11comp_r11gp.out`: g11comp on the round-11 PROF records.

## 0. Scorecard: revision 6 against round 11

| rev-6 prediction | measured | verdict | why |
|---|---|---|---|
| P45 ladder: "L2 (H1) or L4 (H2) stalls" | none of H1-H4 held: the "54.7 cycles" loop was never pipelined | **wrong premise** | I costed hazard hypotheses on round 10's bench without checking that the benched binary was the pipelined loop. The real rules came from the ladder's method (one-core probes), not from its rungs |
| G0 -1.4 +- 0.3 ms | -1.41 (GELU_II -0.88 + asm GELU -0.53) | hit (by another route) | the asm GELU at II 13 gave G0's saving without G0's C++ changes |
| G1 -3.6 +- 0.4 ms in total (II 10) | -1.41 so far; G1 not built | open, **re-priced to -3.2 in total** (-1.76 left) | II 10 is not reachable: ports set ResMII 10 and registers push it to 12-13 [GEN]; the fully fused gated loop does not allocate (section 2) |
| GELU floor 8 cycles per 32 lanes (ldb, mv) | ResMII 10-11 (DM_ADA, QEY_RS); II 13 reached | **wrong** | I counted slots only. DM_ADA (LUT loads + every vlda) and QEY_RS (LUT index reads + vfloor + vst x) are shared ports |
| P43 -0.8 +- 0.3 ms | -0.35 | **miss** | half of the change was not built (below) |
| sim13's GELU deltas per op | timeline r10r -> r11b: mlp0 -68, up_a -29, up_b -31, yx -13 us per block | consistent | together -141 us per block = -1.7 ms, against -1.41 ms measured (53.84 -> 52.43); the rest is noise in ops the change did not touch (norm2 +6, mlp3_a +5) |

### Why P43 missed (-0.35 against -0.8)
P43's -0.8 had three parts (rev 6, sim13): mlp3_a's first-tile B starvation -35 us per block, the four ops after an EW op
(yx, up_a, qkv, mlp0) -7 each, and out_proj -5. Reading `design8.py` (`P43=1`), round 11 built only part of it:
1. `fill_b()` moved before `finish_deps(op)` only. For the four ops after a norm, the EW branch has already run the norm's
   `tg.finish()` before the GEMM's iteration starts, so their B still leaves DDR after the norm: **the -28 us part was never
   tried.**
2. The memtile B forward stayed at `l2_depth 4` + `b_depth 2` objects. Only 6 of mlp3_a's 24 first-tile k-steps can be
   resident before its A arrives; the other 18 still share DDR with the 8-column A slot load.
3. out_proj follows attention, whose K / V fills sit ahead of it on the same B shim channels, so the hoist cannot overtake them.

So the part that was built was priced at mlp3_a's -35 times a partial forward, plus ~0 for out_proj: about -25 to -35 us per
block. **Measured: -29.** The model was right; the proposal was half built. This is unverified per op: there is no r11c / r11d
timeline. Pre-registered: `step8.py timeline --build full12r11d` shows mlp3_a 166 -> 137 +- 8 us, and the after-EW boundaries
unchanged (+-5 us). The rest is now P43b (PROPOSALS), re-priced at **-0.36 ms**. The after-EW part is smaller than rev 6 said,
because B is only ~14 % of the DDR bytes when those ops start (the A slot load dominates), not the whole wait.

Lessons:
1. Bench the binary that is deployed, and cost hypotheses only after that.
2. Price shared ports, not only slots.
3. In each proposal, state which code path each part of the saving depends on, so a partial build can be scored part by part.

## 1. Refit: sim14 on the r11b timeline (`r12/sim14_r11b.out`)

- Depth line through the r11d sweep: **4,310 us per block + 0.37 ms**. Residuals are within 0.05 ms at depths 12, 10, 8 and 6.
- r11b block (blocks 1-11): 4,328 us. The GEMM ops compose (g11-r11gp) to 1,938 us per block, against 2,074 for r10rp. All of
  the -136 us is epilogues, in us per core: yx 27.1 -> 14.5, up_a / up_b 112.5 -> 81.7, mlp0 131.0 -> 68.8.
- Boundary classes: after an EW op +26.7 us, after a dependent GEMM / attention +29.1, independent -11.7, up_b +5.4. Per-op rms
  error 5.8 us. The model gives 52.66 ms for r11b (measured 52.43), and 51.96 with P43 at its measured delta (r11d measured 52.07).

Per block (r11b + P43, 4,299 us; every op is serial):

| part | us | ms per step | share |
|---|---|---|---|
| GEMM k-steps (median 2,052-2,064 cycles) | 1,516 | 18.2 | 35 % |
| attention | 912 | 10.9 | 21 % |
| SSM + its boundary | 796 | 9.6 | 19 % |
| norms (125 / 123 / 152 / 147) | 547 | 6.6 | 13 % |
| GEMM epilogues | 289 | 3.5 | 7 % |
| boundaries (incl. P43's -29) | 106 | 1.3 | 2.5 % |
| tile handoff + acc_zero + first + coupling | 134 | 1.6 | 3 % |

The epilogues are no longer the big lever: 3.5 ms in total, of which G1 can take ~1.8. Attention, the SSM and the norms are
52 % of the block; the k-steps are 35 %.

## 2. G1 at the round-11 rules

### 2.1 What binds a table-lookup loop (GELU, the SSM sigmoid, the attention exp)
Per 32 lanes, from each op's itinerary (aie2p_lat.json via msched.ports) [DOC + GEN]:

| resource | GELU chain (flat-edge table, x reloaded) | notes |
|---|---|---|
| QEY_RS (the x register file's read port) | 8 LUT index reads + 2 vfloor (+1 per `vst x`) = **10** | stores of x registers take it too: spill through `vst.conv` from the accumulator (CRRND_RS) instead |
| DM_ADA (load address port A) | 8 LUT + every vlda = **8 + lda loads** | a plain `vldb` uses only QEY_WB: every non-LUT load in a LUT loop belongs on ldb |
| ldb / QEY_WB | 8 LUT + vldb loads | |
| mv | 2 clamp + 4 shuffles + 1 vmov (+2 index adds) = 7 (9) | |
| st / CRRND_RS | conversions, vfloor, stores: 5-7 | |
| vec | 4-5 | |
| x registers | 12, minus 3-4 pinned constants (clamp lo / hi, 1.0, table base) | **this sets the II reached** |

So the floor of any LUT-line loop is **10 cycles per 32 lanes** (QEY_RS), not rev 6's 8. msched reaches 11-13, because of
x-register pressure (MaxLive 8-9 of the 8-9 free registers) [GEN, `r12/g1sched.out`]:

| loop (per 32 lanes unless stated) | ResMII (binding) | II reached, flat-edge table | + biased LUT base (no index adds) | U / stages | text, explicit / kernel-only |
|---|---|---|---|---|---|
| deployed gelu_ms (bf16 in, xc ring) | 11 (DM_ADA, QEY_RS) | 13 [MEAS 14.6 in place] | | 1 / 5 | 718 B [DOC] |
| gelubf (bf16 in, flat-edge) | 10 (QEY_RS) | 13 | **11** | 2 / 5-6 | 1.2 / 0.6 KB |
| yx: fp32 piece + bias -> GELU -> bf16 | 10 (QEY_RS, DM_ADA, ldb, QEY_WB) | 13 | **12** | 2 / 6 | 1.3 / 0.65 KB |
| pan fused (64 lanes: 2 chains + ups + push) | 20 | 27 (13.5 per 32) | 23 (11.5 per 32) | 1 / 4 | 1.4 / 0.65 KB |
| gated fused (64 lanes: gate chain + up + product + push) | 24 (DM_ADA: 4 accumulator loads per 32) | **no allocation up to II 32** (x pressure 11-13) | none | | |
| gup: up + bias, bf16(v u), ups, push (64 lanes) | 6 (DM_WM) | **6** | | 2 / 6 | 0.75 / 0.45 KB |
| panpush: bf16 -> fp32 -> push (64 lanes) | 2 | **2** | | 2 / 6 | 0.42 / 0.32 KB |

The biased LUT base removes the index adds and is worth 1-2 cycles per 32 lanes. But today's base vector also picks the ab or
cd table copy per lane (the `vsel` with the 0xCCCC mask in gen_gelu), so a probe must come first: does the 4x64 LUT load have
a base operand, or does it read absolute addresses from the index register? If it has none, the bias cannot move and only
the flat-edge column applies.

### 2.2 The G1 design that fits
The fully fused gated loop does not allocate. Per 32 output lanes it keeps the gate chain's ~9 x values plus the up value, and
its 4 accumulator loads per 32 lanes all need DM_ADA (`vldb` cannot load a bm quarter). So split it into passes:

1. **One fused bias + GELU kernel** (the yx row): fp32 pieces + a bias stream in, bf16 out in place, II 12-13, with the two
   table pointers (tanh / erf) and the 2D walks as parameters (as in gelu_ms). It serves yx, the gate half of up_a / up_b, and
   mlp0's GELU.
2. **gup** (II 6 per 64): up piece + bias -> u; v (the GELU'd gate, bf16 in L1) x u -> bf16 -> fp32 -> `vst.push.576.conv`.
   No LUT and 19 ops: the safest first kernel, and the biggest single gain.
3. **panpush** (II 2 per 64): mlp0's bf16 -> bfp panels. Fusing it into kernel 1 (the pan row: 11.5-13.5 per 32) saves only
   ~0.2 us per tile, and costs a third LUT kernel's text.

Per tile (cycles, +5 % bank stalls, ~150 per pass for call + prologue; sim14):

| epilogue kind | tiles per core per block | now (r11gp) | G1 | saving per block | per step |
|---|---|---|---|---|---|
| gated (up_a, up_b) | 32 | 5.11 us | 1.57 us (gate pass 144 x 12.5 + gup 72 x 6 + half a stash copy) | 113 us | **-1.36 ms** |
| pan (mlp0) | 16 | 4.30 us | 2.41 us (fused GELU 288 x 12.5 + panpush 144 x 2) | 30 us | **-0.36 ms** |
| yx GELU tiles | 4 | 2.92 us | 2.18 us | 3 us | -0.04 ms |
| **G1 total** | | | | **146 us** | **-1.76 +- 0.35 ms** (at II 11: -1.83) |
| G1a = gup alone (bias pass + gelu_ms kept) | 32 | 5.11 | 2.44 | 85 us | **-1.02 +- 0.2 ms** |

- **Text** (GEMM core 15,968 / 16,384 B [DOC llvm-size]): kernels 1 + 2 + 3 with explicit prologues are ~1.25 + 0.75 + 0.42 =
  2.4 KB. Freed: gelu_ms 718 B, the flag 2-4 C++ paths of acc_epilogue (~1.2-1.5 KB of its 3,024: the product loop and the pan
  push loop), and 416 B already free: ~2.3-2.6 KB. That is borderline. Build gup and kernel 1 with explicit prologues and
  panpush kernel-only. Kernel-only is safe only if the push stream starts at a scratch block (the leading pushes write
  garbage); otherwise use the explicit form. bias_pcs (752 B) stays for the plain / qkv / v_obj epilogues.
- **L1**: the flat-edge tables add 512 B (two tables x two copies, in whole groups of 4 segments); the x spill ring adds 256 B
  of stack. The GEMM stack reserves 2,816 B and uses 1,664 [MEAS stackaudit]: shrink it from 0xB00 to 0x800 first (P42).
- **Bit-identity**: kernel 1 is bias_pcs + gelu_lut_i, op for op (the flat segments return the clamped edge value exactly).
  gup is today's product + push: the same bf16 rounding of g x u, an exact ups, the same bfp conversion. `vmsc s, sh, 1.0`
  replaces `vmac s, sh, -1.0` (both exact).

### 2.3 What the generator should learn from this
- On these loops the allocator decides the II. msched's list scheduler + sink pass + circular allocation land 2-3 cycles
  above ResMII. Before any new kernel, try:
  - a register-pressure-aware scheduler (stage scheduling after IMS, or swing modulo scheduling);
  - automatic spilling of a long-lived x through `vst.conv` / `vldb` (the ring trick, chosen by the tool);
  - folding constants into scalar or immediate operands where an op allows it.
- Two cheap cuts of ResMII, each to probe on one core:
  - `vfloor.s32.bf16` with a **bm source** (`VFLOOR_..._float_to_int_bm`, whose itinerary holds only QEY_WA). With the clamped
    value in a bm quarter, both vfloors leave QEY_RS (10 -> 8). The clamp writes x, so this costs one vshuffle-to-bm (mv +1).
  - the biased LUT base (above).

## 3. Generator roadmap (msched.py as built)

What msched has [DOC r11/msched.py]:
- ops over virtual registers with .td instruction names; slots, itinerary ports and banks reserved modulo II;
- list scheduling and Rau's iterative modulo scheduling, plus a MaxLive-aware sink pass;
- modulo variable expansion with circular allocation over x and dm (atoms: wl / wh, bm quarters, cml / cmh);
- emission with an explicit prologue, a full-form aligned body and a drain;
- a value-level checker: own-iteration defs, the bypass window, 2D post-modify addresses, spills through memory.

What it lacks: loop-carried dependences, more than one fixed op list per iteration, output streams in the checker, the int16
and int8 LUT forms, pass cutting, and a kernel-only prologue.

### 3.1 P29: the SSM chains
Today, per chunk (8 rows x 64 = 16 vectors of 32), in cycles [MEAS r9 PROF + the round-9 deltas]:

| phase | cycles |
|---|---|
| conv | 1,058 |
| gate GEMMs | 650 |
| sigmoid | 1,168 |
| log_a + exp | 1,170 |
| sqrt | 493 |
| recurrence | 888 |
| z panels | 244 |
| output copies | 798 |
| params | 303 |
| waits | 146 |
| **total** | **~6,900** |

| pass | what msched needs | ResMII / II [GEN where run] | cycles per chunk, now -> P29 |
|---|---|---|---|
| P29a sigmoid (32 vectors, gx and ga) | nothing new (a LUT line without the product) | 10 (QEY_RS) / **12** (11 biased) | 1,168 -> ~450 |
| P29b log_a + exp chain + a, 1 - a^2 (16 vectors) | **int16 vector ops** (`vadd.16`, shifts for 2^n: the VADD_16 / VUPS forms in the json); the magic-number round in the MAC unit; `filter_even` of the int16 view (a vshuffle mode); **pass cutting** (`cut(v, buffer)`: the chain has ~20 vec and ~13 st ops per vector, and needs 2 passes at 5 dm) | vec ~20 -> II ~22 | 1,170 -> ~450 |
| P29c sqrt + recurrence (16 vectors) | the **16-bit LUT form** (`vldb.4x16 / 4x32` with the int8 table: `load_lut_2x_int8`); int16 and / shift; **several input streams** (xc, gx, a, mult, h0 lo / hi, the fp32 state as bm-quarter loads) and **output streams** (h fp32 as 2 bm stores, bf16 hb) | recurrence alone: 10 (vec) / **13** (dm MaxLive 5 of 5) | 1,381 -> ~560 |
| P29d conv + z panels | the per-column tap tables as a 2D-walked stream (the 4 taps reload per vector); the `ups` widen on mv instead of `mul(.., 1)` on vec; a push output stream | vec 8-10 / ~10 | 1,302 -> ~610 |

Also needed:
1. **No loop-carried edges**: every SSM pass is elementwise over the chunk. The recurrence runs across frames, not within the loop.
2. The checker must model the fp32 state's two bm-quarter loads, and the int16 views of x atoms (`cast_to` is the same
   register: a new PART alias, not an op).
3. **A kernel-only or short prologue**: the EW core has 144 B of text free [DOC llvm-size 16,240 / 16,384]. The generated passes
   must replace C++ of about the same size. Replaceable: ew_ssm's loops (~1.5 KB of its 2,752 B), exp_pA-D + loga_p (2,080 B),
   sigmoid_lut_n (384 B), sqrt_lut_n (784 B): ~4.7 KB in all. Four generated kernels at ~0.8-1.2 KB each fit only with the
   short prologue form, and kernel-only where the stores can be redirected.

**Prediction:** P29 a-d save 2,951 cycles per chunk at these IIs, i.e. -4.2 ms. Scaled by 0.85, for what the round-11 GELU lost
between the generator's II and the bench (13 -> 14.6): **-3.6 +- 0.8 ms**. Do P29a and P29b first (-1.0 ms each, independent).
The SSM's input floor (7 KB per chunk through one shim channel at ~6.4 GB/s: ~242 us per block) stays below the new compute
(~470 us per block), so the SSM stays compute-bound.

### 3.2 P48 / P49: the attention K phase
- **P48 exp + push** (one row block = 6 key blocks of 64):
  - the op list: the LUT line, with the row max m as a pinned operand (x for `msc off, slope, m`; -m in a pinned dm for the
    argument MAC), and one push per 64 scores;
  - [GEN]: ResMII 20 per 64 (QEY_RS); msched reaches 31 per 64 with the biased base and nothing without it (x pressure: the
    score s is read by the argument MAC and again by the line);
  - msched needs a pinned dm (drop dm4 from the class) and **two iterations' pushes in order** (the checker should model the
    output stream position);
  - at II 24-31 with only 6 iterations per call, the prologue + drain is a third of the call: ~250 cycles per (rb, chunk)
    against 330 today: **-0.9 +- 0.3 ms**;
  - pipelining across row blocks needs a second 768 B score buffer; the C object has 64 B free [DOC rev 6];
  - rev 6's -2.1 assumed II 18 per 64, which is below ResMII.
- **P49a QK^T + score store** (48 bfp mmul per (rb, chunk)):
  - Q pops on lda (`VLDA_POP_576` holds DM_ADA), K pops on ldb (QEY_WB only);
  - 6 accumulator chains under the measured accumulator rule (distance 2 or >= 3) over 5 dm: two blocks per dm pair, as today;
    stores by `vst.conv`;
  - msched needs the pop ops and the FIFO state registers (gen_ks has them), and the dm-pair constraint;
  - floor 48 + ~12 -> ~70 cycles against 138: **-0.8 ms**.
- **P49b**: the running max on mv as the scores form (`vmax_lt.bf16`, L_WM), rowmax8 by 3 shuffles, rows8 by `vbcstshfl`
  (L_RM): ~50 cycles against ~150: **-1.15 ms**.
- **The K phase as one kernel**: DM_ADA = 96 LUT + 48 Q pops = 144, and ldb = 96 LUT + 48 K pops = 144 per (rb, chunk). The
  floor is ~150, but dm pressure (4 for QK + 3 for the exp) exceeds 5, so the two parts alternate rather than overlap. **P48 +
  P49 -> ~320 cycles per (rb, chunk): -2.9 +- 0.8 ms** (rev 6: -4.5).
- P50 (the V phase in asm, 213 -> ~90 cycles): -1.4 ms.

## 4. Host part (2.0-2.2 ms per frame, our code) [MEAS r11/hostprof.py]

Where it goes:

| piece | ms |
|---|---|
| conv | 0.48-1.1 |
| heads | 0.6-1.0 (on one thread, a steady 0.83) |
| queries | 0.12 |
| packing | 0.14 |
| read | 0.08 |
| launch / sync | the rest |

The spread comes from the OpenMP pool going idle during the 52 ms NPU wait, plus other tenants (load 1.3-2.9). Each idea below
is priced against the whole step (dispatch + host):

| idea | how | predicted | risk |
|---|---|---|---|
| H1 overlap | run heads(t) on a worker thread while dispatch(t+1) runs. Run conv(t+1) before waiting on dispatch(t) when the next frame is already there (at <= 18 fps the camera rarely queues, so in live use only the heads overlap) | throughput -0.6 to -1.0 ms per frame; a frame's latency unchanged | none: no numerics change |
| H2 warm, pinned pool | `torch.set_num_threads(4)`; threads pinned (`OMP_PROC_BIND=close`, `OMP_PLACES` on cores away from other-job's worker); a short busy pre-spin just before a frame is due | conv 0.48-1.1 -> 0.45-0.6 (median and p90) | low; spinning costs CPU time on a shared box |
| H3 lean heads | the heads as numpy fp32 matmuls with pre-transposed weights (48 x 768 -> logits: ~25 MFLOP, ~0.05 ms of math; the 0.8 ms is torch's per-op overhead) | heads 0.83 -> 0.15-0.25 | low; fp32 either way: compare outputs to 1e-6 |
| H4 patch embedding on the NPU | the stride-8 conv as a GEMM op ahead of block 0 (1,024 x 192 x 768 = 151 M MAC: ~9 us on 24 cores); pos-emb added in its epilogue (a full-matrix "bias": one more bf16 load per vector); pixels im2col-packed to bfp16 on the host (393 KB, a reshape) | conv 0.5-1.1 -> ~0.1 (packing only); dispatch +35-50 us (the op, its A load, a boundary) | numerics: bfp16 pixels (8-bit mantissa per block of 8) against torch's bf16 conv, so DAVIS-gated. design7 had an EW-core patch op: `ew_patch` + `gemm8` are still in ew8.o (1.4 KB of EW text): remove them if H4 does not revive them |
| H5 heads on the NPU | the heads' linears on one column's GEMM cores after the final norm; the logits decoded on the host | heads -> ~0.05; dispatch +40-60 us | moderate (new op kinds); only worth it after H3 |

Recommended order: **H3 + H1 + H2** (no numerics change). Host part 2.0-2.2 -> **~0.9 ms of latency, ~0.5 ms per frame in
throughput**. Pre-registered: the whole step at depth 12 goes 54.06 -> 53.0 +- 0.3 ms, with per-frame throughput -> 52.6. H4
only if DAVIS holds: a further -0.4 to -0.9 ms.

## 5. Ceiling and P14 (sim14, depth 12)

Combined, each lever on top of the previous:

| added lever | step ms |
|---|---|
| G1 | 50.20 |
| + P43b | 49.84 |
| + attention K phase (P48 + P49) | 46.97 |
| + P29 (x 0.85) + P52 | 42.59 |
| + P7-E high | 39.27 |
| + P35 + P24b | 38.29 |
| + P50 | 36.88 |
| + P51 | 35.92 |
| + k-step 1,600 (needs L1 and R_A) | **31.6 (31.7 fps)** |

- At that ceiling: depth 10 26.4 ms, depth 8 21.2.
- Without the speculative k-step row: **35.9 ms** at depth 12, 30.0 at depth 10.
- Rev 6 said 30.5 / 34.9. Re-pricing the K phase and G1 at the real port and register rules costs ~2.5 ms; P29, P51 and P52
  recover ~1.4.

**P14** (two frames in flight; per frame >= the busier side + 15 % of the other):

| state | GEMM side / EW side (us per block) | per frame | fps | gain | latency |
|---|---|---|---|---|---|
| today | 2,956 / 1,343 | **38.3 ms** | 26 | x1.36 | ~77 ms |
| after G1 + P43b + attention K | 2,540 / 1,343 | 33.3 ms | 30 | x1.41 | ~67 ms |
| after the SSM and norms | 2,540 / 702 | 32.1 ms | 31 | x1.22 | ~64 ms |
| ceiling | 1,899 / 702 | 24.4 ms | 41 | x1.29 | ~49 ms |

The verdict is unchanged: P14 is the last fps lever and doubles latency. Build it only if fps matters more than latency for
the product.

## 6. What stands from revision 6
- The k-step is compute-bound (median 2,052-2,064 cycles).
- The bank rule and the SSM table numerics.
- The norm-fold analysis and P7a's pre-registered outcome (rev6/ANALYSIS section 4).
- The attention breakdown (rev6 section 5), now with the K-phase prices above.
