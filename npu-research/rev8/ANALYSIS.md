# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 8, 2026-10-10. State analysed: round 12's `full12r12e` (50.05 ms per depth-12 dispatch, Turbo, gpu_busy 0, AJ 70.84),
its timeline `npu-tapnext/build/tl_full12r12e_d12.json`, the r12e elfs (`build/full12r12e.prj/elfs_main_core_*`), the PROF
images `build/g11-r12gc.npy` (GUP + CPY32) and `g11-r12gnp.npy` (PAN_NOPUSH), and `RESULTS_FROM_MAIN.md` round 12. Revision 7
is in `rev7/`. Tags: [DOC] vendor source or the deployed binary, [MEAS] measured by the main agent, [INF] model or arithmetic,
[GEN] what the main agent's scheduler (`npu-tapnext/r11/msched.py`, unmodified) reaches for an op list, [CP] what
`npu-tapnext/r12/cpsched.py` (CP-SAT) reaches, run on the laptop. Cycles at 1.81 GHz.

New this revision (`tools/`, outputs in `r13/`; read-only containers, no device, no builds):
- `tlops.py`: per-op wall per block across timelines (`r13/tlops_r10r_r12e.out`, r10r .. r12e).
- `textmap.py`: program-memory map of an elf from its disassembly: bytes, bundles, nop bundles, full-form bundles and callers
  per function (`r13/textmap_r12e_0_2.out`, `_0_5.out`; disassemblies `r13/r12e_*.dis`, `.nm`, `.size`). Round 13's
  `npu-tapnext/r13/textmap.py` is the per-object view of the asm kernels (preamble / body / tail, compact cost): use its
  numbers for asmfit; mine covers whole elfs including the C++.
- `loops.py`: every zero-overhead loop of an elf with its body bundles (`r13/loops_r12e_0_2.out`, `_0_5.out`): the static
  cost of the deployed loops, the baseline for every re-price below.
- `attk.py`: the attention K-phase op lists (QK^T, exp, fused; variants below) for msched / cpsched (`r13/attk.out`).
- `sim15.py`: sim13's model on the r12e timeline + the g11-r12gc composition (`r13/sim15_r12e.out`).
- `r13/g11comp_r12gc.out`, `r13/g11comp_r12gnp.out`: g11comp on round 12's PROF records.

## 0. Scorecard: revision 7 against round 12

| rev-7 prediction | measured | verdict | why |
|---|---|---|---|
| G1 in total -1.76 +- 0.35 ms | -1.49 | **hit** | with compensating errors (below) |
| G1a (gup) -1.02 +- 0.2 | -0.55 | **miss** | the composition left out the stash copy: priced as "half a stash copy" = 400 cycles; it was 1.70 us = 3,080 cycles per tile on average [MEAS r12a differential]. A guessed component, 7x off |
| G1b -0.74 | PAN -0.30 + BGELU -0.32 = -0.62 | hit (within 0.15) | |
| (not predicted) CPY32 | -0.30 | | the stash copy, once found, was its own lever |
| bgelu II 12-13 | II 12 | hit | CP reached it; II 11 failed the padding check |
| standalone GELU II 10 not reachable ("registers set the II") | no II-10 schedule in 8 x | hit | |
| P29a -1.0 | -0.61 | **miss** | stale baseline: round 9's PROF phase (1,168 cycles) instead of the deployed C++ loop (898 on one core). The saving equalled the bench delta exactly (399 cycles x 216 chunk units x 12) |
| P43 check: mlp3_a 166 -> 137 +- 8 us | 158-161 (r11d .. r12e) | **miss** | wrong mechanism (0.1) |
| sim14 base 51.96 (r11d 52.07) | | hit | sim15 on r12e: 50.14 vs 50.05 |

### 0.1 The P43 miss: the boundary is the A slot load, not B
Per-op walls across six timelines (`r13/tlops_r10r_r12e.out`, us per block): mlp3_a 161 (r10r), 166 (r11b), 158 (r11d), 159,
157, 161 (r12c-e). The 17 us poll quantum shows as a two-level pattern (148 / 165); means over 11 blocks are good to ~3 us. So
P43 moved mlp3_a by ~-8 us at most.

The fit that would have refuted the B-starvation story (sim15 on r12e; composition from g11-r12gc):

| op | follows | A slot per column | boundary + g11 first-tile wait | A at 7.3 GB/s per column |
|---|---|---|---|---|
| out_proj | attention (dependent) | 124 KB | 17.6 + 0.1 us | 17.0 us |
| mlp3_a | mlp0 (dependent) | 248 KB | 15.0 + 22.5 us | 34.0 us |
| down_a, down_b, mlp3_b | an op they do not depend on | 248 KB | -5 to -11 | (loaded during the previous op) |
| yx, up_a, qkv, mlp0 | a norm (EW) | 124 KB | 21-38 | 17.0 + the norm's tail |

Every dependent boundary equals the time to load that op's A slot from DDR at the measured slot rate (8 x 248 KB in ~34 us,
round 10), and the same-shape ops that do not depend on their predecessor have none. B is ~14 % of the bytes at that moment:
hoisting its fills (P43) cannot remove an A-bound wait. The g11 PROF agrees: mlp3_a's first-tile wait grew from 4.2 us
(r11gp) to 22.5 us (r12gc) as the epilogues before it got faster, i.e. its A load lost overlap.

Consequences: **P43b is retired** (its after-EW part was B too); the dependent-boundary lever is **P55** (section 4); **P51**
(norm output straight into the A slot) has the right mechanism for the after-EW boundaries and keeps its price.

### 0.2 Method changes (three misses, one cause: a number not taken from the deployed build)
1. **Price only from the deployed build.** Baselines come from (a) the latest timeline of the deployed build (sim15 refits on
   each round's best build), (b) static loop counts of the deployed elf (`loops.py`: loop bodies x trips + straight-line code),
   (c) one-core benches of the deployed kernel. An older round's PROF phase may be used only after checking that the code it
   measured is unchanged (same loop bundles in the current elf; section 5 does this). Each baseline names its build.
2. **Price a change as a delta of like-for-like numbers** (bench new vs bench old, timeline vs timeline), never as "old in-situ
   phase minus new bench". P29a's error was exactly that mix.
3. **Every component of a composed price is measured or flagged.** A per-tile composition lists each component with its
   source; a guessed component carries +-100 % in the error bar (G1a would then have read -0.55 to -1.0, containing -0.55).
4. **Mechanism check across ops before pricing a boundary fix**: fit each op's boundary against every candidate cause (A bytes,
   B bytes, drain tail, dependency) and require the ops without the cause to show no wait. sim15 prints this table.
5. Pre-registered timeline checks: effects >= 3x the 17 us poll quantum, or a mean over >= 8 blocks; say which.

## 1. Refit: sim15 on r12e (`r13/sim15_r12e.out`)

- Depth line (r12e sweep): **4,153 us per block + 0.32 ms**, residuals <= 0.17 ms. Model base 50.14 ms vs 50.05 measured.
- r11d -> r12e per op (us per block): up_a -41, up_b -41, mlp0 -31, qkv -11, SSM ("out") -41, norm0 -12, attention +3.

| part (per block, 4,152 us, all serial) | us | ms per step | share |
|---|---|---|---|
| GEMM k-steps (2,051-2,066 cycles per k-step) | 1,518 | 18.2 | 37 % |
| attention | 911 | 10.9 | 22 % |
| SSM + its boundary | 755 | 9.1 | 18 % |
| norms | 528 | 6.3 | 13 % |
| tile handoff + acc_zero + first + coupling | 152 | 1.8 | 3.6 % |
| boundaries | 146 | 1.8 | 3.5 % |
| GEMM epilogues | 144 | 1.7 | 3.5 % |

The epilogues are done. The tile handoff (1.3 us per tile: the single C object drains at ~8 B/cycle before the next tile may
start; 80 us per block) is now as large as the epilogues: P57.

## 2. Program memory (task 2)

### 2.1 Where the 16 KB go (r12e; every GEMM core carries the same 16,288 B of text, every EW core 16,224 B)

**GEMM core: 16,288 B (96 free).**

| group | bytes | functions (bytes) |
|---|---|---|
| attention | **6,048** (37 %) | chunk_k 1,456, attn_begin 800, attn_end 784, chunk_v 608, rescale_rb 560, aexp_push 464, attn_qcopy 400, exp32 304 (only rescale_rb calls it), attn_chunk_k 304, attn_chunk_v 256, rows8 112 |
| epilogues | 5,024 | acc_epilogue 2,000 (of it the flag-7 V^T C++ path ~860 and four copies of the callee-saved restore ~190), bgelu 912, bias_pcs 752, gup 448, pan 432, bgelu3 304, cpy32 176 |
| k-step | 2,992 | mm_ks 1,824, **mm_kstep 832** (mm_ks's record builder; it runs once per C object, and each core has one C object (C0_0_buff_0): once per run), acc_zero 336 |
| IRON control | 2,238 | main 2,080 (the table-walked op loop), _main_init, __start |

849 bundles are all-nop (1,842 B): Peano's branch delay slots and alignment, mostly in cold code.

**EW core: 16,224 B (160 free).**

| group | bytes | functions |
|---|---|---|
| SSM | **7,600** | ew_ssm 2,768 (conv, recurrence, z loops inline), exp chain 2,080 (loga_p + exp_pA-D), sqrt_lut_n 784, ssm_tables 480 + bias_tab 160 (tables from the per-head parameters, once per head), gate_gemm 480, sigk 352, out_copy 288 + copy_v 208 |
| norms | 6,752 | norm_rows<LN> 2,624, norm_rows<RMS> 1,968, ln_stats 672, ew_norm 528, rsqrt_i 432, rms_rstd 240, rowsum8 160, mul3 128 |
| ring I/O | 1,296 | in_part 336, out_end 208, in_done 176, in_copy 160, out_begin 160, **__udivsi3 144**, in_need 112 |
| control | 590 | ew_op 256, main 176, _main_init, __start |

### 2.2 Recovery options, ranked (bytes from the map; "net" includes what replaces them)

| rank | option | core | frees | cost / risk | note |
|---|---|---|---|---|---|
| 1 | **mm_kstep's 58 records built at design time** (design8 knows the A ring, C and acc_hi addresses after placement: emit the table as an L1 `Buffer(initial_value=...)`, or pin those buffers' addresses) | GEMM | **~0.70 KB** | S / low; bit-identical (mm_ks unchanged) | cold code |
| 2 | **asmfit on gup / pan / bgelu / cpy32** (33 full-form bundles in the `*_start` bodies + 4 `*_last` blocks) | GEMM | ~0.3 KB | S / none | round 13's textmap gives the exact compact costs |
| 3 | **P29b / P29c / P52 as generated kernel-only passes** (replace the exp chain 2,080 + sqrt 784 + the recurrence loop ~700 + out copies 496 with ~1.6 KB) | EW | **~2.4 KB** net | M / low | and -2.9 ms (section 5) |
| 4 | V^T (flag 7) epilogue in asm (transpose by vshuffle + push, gen_pan style) | GEMM | ~0.5 KB | M / low | faster qkv V tiles too |
| 5 | small C++ fixes: one exit in acc_epilogue (-0.15 KB); `__udivsi3` out of out_begin / out_end by shifts / compares (-0.15 KB, as `__modsi3` in round 12) | both | ~0.3 KB | S / none | |
| 6 | SSM tables precomputed on the host into the parameter stream (ssm_tables + bias_tab go) | EW | ~0.55 KB | M / low | +~2.5 KB parameter DMA per head; the SSM is compute-bound |
| 7 | the attention K phase as P56 (section 3) | GEMM | net ~+0.3 KB (asm ~1.4 KB in, chunk_k's QK^T / max code ~1.1 KB out) | | needs ranks 1-2 first |
| 8 | P7-E: norms as generated port-bound passes (one body for LN and RMS) | EW | ~3 KB | M-L / medium | and -1.5 to -3.1 ms |
| 9 | exp32 replaced by the attention's exp table in rescale_rb | GEMM | 0.3 KB | S / numerics (rescale arguments < -8; table zero below -10) | DAVIS-gated, last resort |
| - | dead entry points | both | **0** | | every linked function has a caller (the linker already drops uncalled entry points) |
| - | moving rarely used code to another core | | ~0 | | no spare core (24 GEMM + 8 EW = 32), and each op kind runs on its own core type. Shrink the cold code in place (ranks 1, 6) |
| - | splitting kernels between cores (per-row GEMM elfs; IRON allows a Worker per core) | GEMM | 0.86 KB on 2 of 3 rows (qkv's V tiles on one row) | M | the binding row keeps the code: useless alone. Splitting attention by phase across rows (P through the neighbour's L1) frees ~2-3 KB per row but imbalances 571 vs 213 cycles per unit |
| - | overlays / program-memory reloads | | | | (a) `npu.load_pdi` exists in mlir-aie 1.4.4 only on the full-ELF path; pyxrt's dispatch bridge refuses it ("cannot supply load_pdi resources"), and it reloads the whole partition (ms): not per block. (b) `npu_blockwrite` into a core's program memory (tile address 0x20000 + offset) from the runtime sequence while the core waits on a lock in resident code: plausible (the CDO loads elfs this way), unmeasured: the control-path write rate, and whether writing program memory while the core fetches elsewhere is safe. Swapping attention <-> epilogue code per block = 2 x 12 x 24 cores x ~3 KB = 1.7 MB of extra sequence per step (the .bin is 1.6 MB today). Probe only (P59), and only if text blocks a > 1 ms lever |

Without numerics changes: GEMM ~1.8 KB (ranks 1, 2, 4, 5), EW ~3.4 KB (3, 5, 6), +3 KB with P7-E. Order: ranks 1 + 2 on the
GEMM core before P56; P29b first on the EW core (it frees text and time together).

## 3. The attention K phase (task 3): P56

### 3.1 Today
In situ (r8 stamps; the attention code is unchanged since r8c: its wall is 908-912 us per block in every timeline r10r..r12e),
per (row block, key chunk) unit, 1,728 units per core per block: K phase **571 cycles** (QK^T + score store 138, max / rowmax8 /
test ~140, aexp_push 49 bundles per 64 x 6 + call ~300), V phase 213. Static check (`r13/loops_r12e_0_2.out`): chunk_k's QK^T
loop is 46 bundles per 2 key blocks (138 per unit), aexp_push 49 per 64: consistent.

### 3.2 What the round-12 rules and the schedulers say (`tools/attk.py`, `r13/attk.out`; one iteration = one key block = 64 scores)

| loop | ResMII (binding) | reached | registers |
|---|---|---|---|
| QK^T: 8 Q pops (lda) + 8 K pops (ldb), two chains (today's even / odd order: bit-identical), vadd.f, 2 vconv | 10 (QEY_WA; vec 9) | **II 10** [GEN, U 1, 4 stages] | x 6, dm 4 |
| exp + push, m pinned in x (rev 7's list, corrected to 2 pinned m registers: rows 0-3 and 4-7 differ) | 20 (QEY_RS) | none to II 48 [GEN]; II 41 with the biased base; none at II 24 [CP, 120 s] | 6 free x |
| exp + push, **m loaded per iteration** (m rows8 on ldb; -m straight into an fp32 half by `vlda.2d.conv.fp32.bf16`; d = ups(s) + (-m) by `vadd.f`) | 20 (DM_ADA, QEY_RS, mv) | **II 30** [GEN, U 4]; **II 26** with the biased LUT base (P54) [GEN]; [CP] none at II 24 in 400 s | x 9, dm 3-4 |
| fused QK^T + exp, m loaded (no score buffer) | 26 (DM_ADA, ldb) | **II 37** with the biased base [GEN, U 2, x 10, dm 5]; none to II 48 without it [GEN]; [CP] none found at 26-28 in 150 s | |

Two facts change the design against rev 7:
1. **The lazy max decouples the exp from the current block's max.** Today m_new = max(rowmax(scores), m_old) and the stored max
   moves only when a row passes m_old + TAU; otherwise P = exp(s - m_old). So P can be computed as soon as a key block's scores
   exist, and a row block needs the slow path only when some lane passed the threshold.
2. **Loading m per iteration frees the 2 pinned x registers and lets one loop run all 36 (row block, key block) iterations of a
   (tile, chunk)**, so the prologue / drain is paid once per 36 iterations, not per row block.

### 3.3 Design (P56, the lazy-fast K loop)
Per (tile, key chunk) one asm call over 36 iterations (rb-major), kernel-only (garbage pushes into the next row block's P slot,
the last row block's into g_srb, scratch); 5 dm in use (QK chains 2-3, exp 2):
- **Q** pops on lda with a 2D post-modify (`VLDA_POP_576_2D`: back to the row block's start 5 times, then +576 B), the FIFO
  refilled per key block (never re-seek a live stream); **K** pops on ldb, contiguous over the chunk, restarted per row block.
- **QK^T**: two chains (even / odd dim panels, today's MAC order), `vadd.f`, `vconv` x 2 -> s (bf16).
- **exp**: -m (the row block's rows8 of -m_old) by `vlda.2d.conv.fp32.bf16` into an fp32 half, `vconv.fp32.bf16` (ups) of s,
  `vadd.f` -> d = s - m_old in fp32 (one rounding, as today's `mac(negm, s, 1.0)`), then today's chain (bf16, clamp, vfloor,
  index add, 4 LUT loads per 16 lanes, shuffles, `vmsc` off - slope m with m loaded on ldb, `vmac` + slope s), one
  `vst.push.576.conv` per 64 (last stage). Bit-identical to aexp_push op for op.
- **The lazy test for free**: the clamp's `vmin_ge.bf16` already writes a lane mask (r16). Lower its bound from 8.9375 SU to
  TAU - 0.25 SU and OR r16 into a per-row-block flag (alu; scalar store per iteration into a 36-byte flag array). A lane at or
  above the bound means "this row block may need an update"; the clamped P it produced is discarded below.
- **After the loop** (C++, ~40 cycles per call): scan the flags; for each flagged row block run today's code for it (QK^T again,
  max, rowmax8, the exact test, rescale_rb if the max moves, aexp_push with the stored max) over its P slot. Every row block's P
  and max are then today's: bit-identical by construction. Chunk 0 of a head always flags (m_old = NEG): call today's chunk_k
  for chunk 0 directly.
- **L1**: rows8 copies of m and -m per row block (2 x 128 B x 6 rb x 2 tiles = 3 KB) are not free (the C object has 64 B).
  Options, to settle before building: (a) one bf16 rows8 copy of m only (1.5 KB) and d by `vmsc` of ups(s) with m and a pinned
  1.0 (x free 8: [GEN] did not schedule it; try CP); (b) the 16-lane g_maxb expanded in the loop by `vbcstshfl` (mv 20 -> 24,
  ResMII 24 < 26: free in the fused loop); (c) find 3 KB in the attention map (the exp table's second copy, g_lcd, lives in the
  C object tail). (b) is the default.
- Text: body U 2 x 37 bundles compact (~0.6 KB) + preamble / drain (~0.4 KB) + C++ driver (~0.3 KB) against chunk_k's QK^T /
  max code (~1.1 KB) that the fallback shares: net ~+0.3 KB. Ranks 1-2 of 2.2 first.

### 3.4 Price (from today's in-situ 571 cycles per unit; [GEN] IIs; per call = 36 iterations = 6 units)

| variant | cycles per call | per unit incl. chunk 0 (+12), updates <= 3 % (+17), banks +5 % | saving per unit | step |
|---|---|---|---|---|
| **P56 fused at II 37** (biased base, P54) | 36 x 37 + 2 x 37 + 70 = 1,476 | ~290 | ~281 | **-3.2 ms** |
| P56 fused at II 30 (if CP finds it) | 1,210 | ~245 | ~326 | -3.7 ms |
| fallback design if P54 fails and CP finds no fused schedule: two loops per row block (QK II 10 into the existing 768 B g_srb, exp II 30) | 6 rb x (6 x 10 + 30 + 15 + 6 x 30 + 60 + 15 + 30) = 2,340 | ~420 | ~150 | -1.7 ms |

**P56: -3.2 +- 0.5 ms with P54's biased base; -1.7 without it.** Not -3.7 until a fused schedule at II ~30 exists. Updates are
unmeasured: step 0 of P56 counts flagged row blocks on the CPU from real scores (r7/expemu's captures); at 10 % the saving
drops by ~0.4 ms. P48 / P49 as separate kernels (rev 7) are superseded. P50 (V phase in asm, 213 -> ~90) stays at -1.4 ms.

## 4. Boundaries: P55 and P57 (new)
- **P55**: mlp3_a's A fills wait for mlp0's whole drain group, then load 248 KB per column (34 us). Split mlp0's drain into g
  groups by output column range (mlp0's N tiles are mlp3_a's K chunks) and let each A-chunk fill of mlp3_a wait only on its
  group: the A load overlaps mlp0's last tiles. The same for out_proj after attention (one K chunk per head, the per-head O
  drains exist). With g = 4: mlp3_a ~37 -> ~12 us, out_proj ~18 -> ~8. **-0.36 +- 0.12 ms.** Needs mlp0's tiles ordered by N
  within each column (check P23's round robin) and per-group task deps in plan8.
- **P57**: release the C object in two halves (two 9 KB sub-objects, two locks): the DMA drains row blocks 0-8 while the
  epilogue does 9-17. Half of the 80 us per block hidden: **~-0.5 ms**. M, medium risk (the pieces map spans both halves).

## 5. SSM re-priced from today's binary (task 4)

Static loop counts of the r12e EW elf (`r13/loops_r12e_0_5.out`) reproduce round 9's in-situ phases, so the code they measured
is unchanged and those phases are today's baselines (the sigmoid excepted: sigk now). Per chunk (cycles; 216 chunk units per
block, 1 cycle per chunk = 1.43 us per step):

| pass | deployed (static: loop bundles x trips + straight-line) | r9 PROF | generated target | saving |
|---|---|---|---|---|
| log_a + exp chain (loga_p 23 x 8, exp_pA 20 x 16, pB 12 x 16, pC 6 x 16, pD 4 x 16, + ~275 straight-line) | **~1,130** | (1,170) | 2 kernel-only passes at II ~22 per vector: 16 x 22 + 2 x ~60 = ~470 | ~660 |
| sqrt (sqrt_lut_n 54 per 64 lanes x 8 + ~80) | ~510 | 493 | int16 LUT pass II ~8 per 32: ~180 | ~330 |
| recurrence (ew_ssm loop 108 per 64 x 8 + ~25) | ~890 | 888 | II 13 per 32 [GEN rev 7]: ~260 | ~630 |
| conv (90 per 64 x 8 + ~330) / z panels (25 x 8 + 44) | 1,058 / 244 | | ~610 together | ~690 |
| out copies | 798 | 798 | P52: ~250 | ~548 |

- **P29b: -0.95 +- 0.2 ms.** Rev 7's -1.0 was right for this pass; the static count confirms its baseline. Pre-registered: a
  one-core bench of the deployed `loga_exp_a2` reads 1,080-1,200 cycles per 16 vectors; the generated passes 420-520.
- **P29c: -1.15 +- 0.3 ms** (960 x 0.85); P29d -0.84; P52 -0.78. All together ~-3.7 ms: the SSM op 755 -> ~430 us per block.

## 6. Ceiling and P14 (sim15, depth 12; `r13/sim15_r12e.out`)

| added lever | step ms |
|---|---|
| P29b | 49.20 |
| + P56 (571 -> 300) | 46.09 |
| + P29c + P29d (x 0.85) + P52 | 43.30 |
| + P55 | 42.94 |
| + P7-E high | 39.84 |
| + P35 + P24b | 38.86 |
| + P50 | 37.45 |
| + P51 | 36.49 |
| + P57 | 36.01 |
| + k-step 2,055 -> 1,600 (needs L1 and R_A) | **31.7** |

- Depth 10 at that ceiling 26.4 ms, depth 8 21.2. Without the speculative k-step row: **36.0 ms** at depth 12, 30.1 at depth 10.
- Same as rev 7 (31.6 / 35.9): round 12 took 2 ms of levers; the re-prices cancel (P43b retired; P55 / P57 new).

**P14** (two frames in flight; per frame >= the busier side + 15 % of the other):

| state | GEMM / EW side (us per block) | per frame | fps | gain | latency |
|---|---|---|---|---|---|
| today | 2,869 / 1,283 | **37.1 ms** | 27.0 | x1.35 | ~74 ms |
| after P29b + P56 | 2,610 / 1,204 | 33.8 ms | 29.6 | x1.36 | ~68 ms |
| after the SSM | 2,610 / 971 | 33.4 ms | 29.9 | x1.30 | ~67 ms |
| ceiling | 1,899 / 713 | 24.4 ms | 41.0 | x1.30 | ~49 ms |

P14 stays the last fps lever and doubles latency.

## 7. What stands from revision 7
- The k-step is compute-bound (2,051-2,066 cycles per k-step in every op, g11-r12gc).
- The LUT-loop floor (10 per 32 lanes) and "registers set the II"; the bank rule; the SSM table numerics.
- The host part (H1-H3) and its price; the norm-fold analysis (rev6 section 4).
