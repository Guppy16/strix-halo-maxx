# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 10, 2026-10-10 (night), after round 14 (`full12r14f`, 47.31 ms at depth 12; `RESULTS_FROM_MAIN.md` round 14 and the
round-15 note). Revision 9 is in `rev9/`, older ones in `rev8/` .. `rev1/`. The analysis behind every number is ANALYSIS rev 10.
Prices are deltas against the deployed r14f at depth 12 from **`tools/sim17.py`** (`r15/sim17_r14f.out`): the attention by
`tools/attsim17.py` (multicast lockstep + head-group barrier, every per-row-block cost a one-core MEASUREMENT, one fitted
constant kappa = 0.6 for the CPU map's flag intensity), the SSM by static counts of the deployed EW elf. **Error bars are no
longer +-100 % defaults**: attention bars = the kappa range 0.5-0.7 that the twins and the per-block walls allow; SSM bars = the
generated target's II range (the deployed side is an exact static count); a cost that cannot be separated names its microbench.

**Headlines.**
- **The lockstep fix is a DMA-only change that can be built now: P60g** (two plan8 TAP functions, each still ONE head-major TAP
  per column, 0 B of core text, no host change, bit-identical): **-1.08 +- 0.22 ms**. Main's round-15 tile permutation is the
  fallback (-0.66 +- 0.13).
- **The slow path**: after P60g, S1 (rescale_rb in asm, round 15) -0.36 and S2 (a running row max in attkb: no first-pass
  rerun, the exact test on one vector) -0.38; alone (no P60g) S1 -0.60, S1 + S2 -1.32.
- **SSM**: every pass of the deployed chain is a latency-bound loop at 2-5x its slot bound (static, r14f elf = r12e's loops):
  P29b -1.0 +- 0.07, P29c -0.9 +- 0.07, P29d -0.9 +- 0.09, sqrt -0.36 +- 0.12, P52 -0.9 +- 0.2: **-4.1 +- 0.5 ms** together.
- FlashAttention-2/3/4: the attention side already has their ideas (lazy rescale, deferred normalisation, software pipelining,
  masked-block skipping); FA4's software exp split transfers only weakly (II is register-bound, not port-bound): parked.
- Ceiling 31.2 ms at depth 12 (35.6 without the k-step row); attention 8.52 -> ~6.0 ms per step.

Common rules (SKILL 3, 9): one dispatch with a short timeout first; the known-good test after any timeout; Turbo / gpu_busy
recorded; one-core bench with guard zones before building in; read the emitted loop; regress BIT-IDENTICAL where marked.

## Ranking (revision 10)

| rank | id | what | cost | risk | expected (depth 12, vs r14f) |
|---|---|---|---|---|---|
| 1 | **P60g** (new) | query half-tiles dealt by the two A GATHERS (attention Q fill, out_proj A fill): G = {0,3,6}+{1,4,7} transpose, pair {2,5} | **S** | low | **-1.08 +- 0.22 ms**; attention 701 -> 614 +- 18 us per block |
| 2 | **P29b** | SSM log_a + exp chain as generated passes (round 15) | M | low | **-1.0 +- 0.07 ms** |
| 3 | **S1 + S2** (new form of P62) | rescale_rb in asm (round 15); attkb keeps a running row max per row block (no first-pass rerun, test on 1 vector) | M | low | after P60g **-0.74 +- 0.14** (S1 -0.36, S2 -0.38); without P60g -1.32 +- 0.22 |
| 4 | **P29c + P29d** | SSM recurrence; conv + z panels as generated passes | M | low | **-0.9 +- 0.07** and **-0.9 +- 0.09 ms** |
| 5 | **P52 + P29s** | SSM outputs in place (no copies); sqrt by the int16 table | M | medium / low | -0.9 +- 0.2 and -0.36 +- 0.12 ms |
| 6 | V1 + K1 + K2 (new) | attv with the row sums folded in; the batched call's -m prep / flag scan in asm; chunk 0 as QK-only + test | M | low | -0.21 +- 0.03, -0.27 +- 0.03, -0.19 +- 0.03 (after 1 + 3) |
| 7 | T4 (new) | GEMM text: S1 frees ~360 B, T3f mm_kstep ~200 B, k_rb's exact path in asm ~400 B | S | low | 0 ms; funds 3 and 6 |
| 8 | P55 | tile-granular A dependency | S-M | low | -0.36 +- 0.12 |
| 9 | P7-E (after P44) | port-bound norm passes | M-L | medium | -1.5 to -3.1 |
| 10 | P35 + P24b, P51, P57 | as rev 8 | | | -0.98, -0.96, -0.5 |
| parked | P60t (main's note) | global host tile permutation + the K / V fill in natural key order | S | low | -0.66 +- 0.13 (P60g is worth more; composing them is worse: -0.76) |
| parked | F1 / F2 (new, FA4) | exp split between the LUT port and a vec-slot polynomial (SSM after P29b; attkb) | M | numerics | F1 -0.1 +- 0.05, F2 <= -0.1 (register-bound): not worth a numerics change |
| parked | P60h (new, FA4 LPT) | adaptive deal from the previous frame's flag counts | L | medium | <= -0.33 beyond P60g (perfect balance = 1-rb round robin -1.41) |
| parked | P66 / P67, H1 / H2, P14, P59, H4 | as rev 9 | | | |
| retired | see the status table | | | | |

Combined (sim17, cumulative from 47.31): attention stack 1 + 3 + 6 -2.54 (44.77) -> SSM -4.1 (40.66) -> P55 (40.30) -> P7-E mid
(38.00) -> P35 + P24b (37.02) -> P51 (36.06) -> P57 (35.56) -> k-step 2,055 -> 1,600 **31.21 ms** (depth 10 ~26.4, depth 8 ~21.6).

## Status of the revision-9 proposals

| id | status | verdict / what is left |
|---|---|---|
| P61 | DONE (+ chunk-0 rescale skip): -0.63 | hit for P61 (-0.33); the chunk-0 skip (~-0.3) was not priced |
| P63 | DONE: loop -183 cycles per unit (bench), in situ -0.04 .. -0.47 | **miss explained** (ANALYSIS 0): the loop was right; sim16 kept r13's slow path (no first-pass rerun, T 66, R 300-450), the real batched slow path costs 1,530 per flagged row block |
| P50 | DONE: -0.9 (hit) | bench pre-registration (80-110 per row block) missed by SCOPE: the loop alone is 87 (hit); the row sums are a second pass (attvs 126 per (tile, chunk) = 21 per rb) and the calls / C++ glue 162 per (tile, chunk) = 27 per rb: 87 + 21 + 27 = 135 vs 136 measured. Lever V1 |
| T3 | partly DONE (a, c, d, e + V^T asm) | T3b / T3f left (T4) |
| P60 | single-TAP rr deal impossible (main's deal.py) | **superseded by P60g**: dealing on the GATHER side with a G6 transpose + pair is single-TAP head-major on both sides |
| P62 | measured (T 164, R 578) | -> S1 / S2 / S3 |
| P64 | parked | the producer does not bind (sim17 agrees) |
| Q1 (l2-depth) | not run | sim17 still has no depth effect at kappa 0.6 (D 8 vs 24: -0.06); the r13 gain stays unexplained, low priority |
| per-column K / V memtile buffering (pass-10 brief) | **retired** | memtiles 0-2 use all 6 S2MM channels (A x2, B, C join x3): no channel for 3 row streams per column; -1.29 alone would not beat P60g anyway |
| more L1 K / V slots | retired (rev 9) | 4.6 KB per slot; -0.75 / -0.89 |
| 2+ TAPs per gather, BD chains for the deal | **retired** | the A slot is loaded in per-head chunks (A_RT BdIteration, one lock per 10,368 B of STREAM): any gather that is not head-major lands pieces in the wrong chunk |

---

## P60g. The query deal on the gather side — **S, risk: low** (rank 1, new)

**Why it works where P60 did not.** Main's deal.py showed that dealing tiles cannot keep the qkv Q DRAIN and the out_proj O
DRAIN single-TAP. P60g leaves both drains (and every core) unchanged: each core still writes its own QKV slots and its own APAN
column. The deal is done by the two consumers' A fills, which are gathers, and only gathers change. The A slot is filled in
per-head chunks (A_RT, a lock per 10,368 B of stream), so each gather must stay ONE head-major TAP: [12 heads][n1][n2][1,728 B].
r15/deal3.py + deal4.py: the sum-of-squares argument leaves only families with size-2 dimensions varying the column, and the
construction below is single-TAP, non-negative and linear on the memtile side for all 8 columns on both gathers (checked piece
by piece in deal4.py).

**The deal** (pieces = half tiles = 3 row blocks = 1,728 B per head; natural piece (n, T, H) = column n, tile T, half H; dealt
column c, slot position p = 2 t + h of the A slot's [tile][3,456] layout):
- group G = {0, 1, 3, 4, 6, 7}: dealt column c = 3 T + e takes half e of tile T from the six natural columns n = 3 j + H
  (j 0-2, H 0-1) at position p = 2 j + H. Every dealt column in G sees six natural columns.
- pair {2, 5}: dealt 2 takes half 0 of every tile of natural 2 (p 0-2) and of natural 5 (p 3-5); dealt 5 the halves 1.

**Code (plan8.py only; byte TAPs as plan8 writes them; OBJ 4,608, QKV_PER_HC 55,296, QKV_PER_H 442,368, PAN 576, H 1,728):**
- `attn()`, A_HALF branch, column c (replaces `[1, 12, 3, 3456] / [0, QKV_PER_H, 3 OBJ, 1]`):
  - c in G: T, e = divmod(c, 3); `tap(qkv + 3 T OBJ + e H, [12, 3, 2, H], [QKV_PER_H, 3 QKV_PER_HC, QKV_PER_HC, 1])`
  - c = 2 / 5: `tap(qkv + 2 QKV_PER_HC + (c == 5) H, [12, 2, 3, H], [QKV_PER_H, 3 QKV_PER_HC, 3 OBJ, 1])`
- `a_fill()` for out_proj (src = r["O"], apan, K 768, A_HALF branch), natural column n (replaces `[1, 12, 1, CHUNK_A] /
  [0, kstride, 0, 1]` at column n), O = src base, col = src["col"], ks = src["kstride"]:
  - n in G: j, H = divmod(n, 3); `tap(O + (2 j + H) H_, [12, 3, 2, H_], [ks, 3 col, col, 1])` (H_ = 1,728)
  - n = 2 / 5: `tap(O + 2 col + 3 (n == 5) H_, [12, 3, 2, H_], [ks, H_, 3 col, 1])`
  Gate it on a flag (`P60G=1`) used only for `r["O"]` -> out_proj and the attention op; every other a_fill unchanged.
- Nothing else: the qkv drains, the K / V fill (keys stay in natural order: bit-identical online softmax), the O drains, the
  memtile chunk tasks, the cores. Strides < 2^20 words (3 col = 228 K words), inner dim 432 words, 4 dims: plan8.tap's asserts
  hold. Shim BDs per op: unchanged (one per column). Host: unchanged.
- Dependencies: the attention fill reads other columns' QKV slots and out_proj reads other columns' O; both already wait for
  their producer op's whole group (`deps` are per region), so nothing changes there. Check the lowered `dma_bd` lengths.

**Price** (sim17, kappa 0.5 / 0.6 / 0.7): -0.87 / **-1.08** / -1.30 ms; with the map's clustering moved by 2-6 columns -1.11
(robust to where the slow rows are). For reference: multi-TAP round robin of 3-rb pieces -1.13, of 1-rb pieces -1.41, main's
tile permutation -0.66, everything balanced and no lockstep (floor) -2.30.
**Pre-registered** (timeline r14f -> P60g, attention wall per block 1-11, us): r14f 717 663 853 750 662 649 799 736 838 531 514;
predicted 572 589 727 637 599 574 676 629 712 525 514 (mean 701 -> 614 +- 18, spread 339 -> ~213; blocks 10 / 11 move < 10 us).
Regress BIT-IDENTICAL; step 47.31 -> 46.2 +- 0.2.

## S1 + S2 + S3. The slow path — **M, risk: low** (rank 3)

Measured per flagged row block (r14 KRB stamps): first-pass rerun FP 393 + exact test T 164 + rescale R 578 (+ -m rows 59) +
exact rerun XR 393 = 1,587.
- **S1** rescale_rb in asm (round 15 is building it): 9 x (4 vlda of an fp32 quarter, hi = srs, lo = mac(a, hi, -1), pr = hi s +
  lo s, 4 vst) pipelined ~8 per trip + exp32 (kept: bit-identical) ~61 + call ~30 = **~170-230** (was 578). Text 560 -> ~200 B.
- **S2** the running row max: today the batched attkb keeps only a 7-slot score ring, so a flagged row block re-runs its 6-trip
  first pass (393) to get its scores back for the exact test. Give attkb one more stream: per iteration `vmax` of the score
  block with the row block's running max slot (load, 1 mv op, store; 6 x 128 B = 768 B of slots. The K object's junk tail has
  only 384 B left after the -m rows (1,152 - 768) and srb + 896 holds the exp-table copy, so this needs a 768 B static: `.aie.bank3` has 2,688 B
  free (ldScripts_main_core_0_2: bank3 0x7F580, 0xA80; bank1 has 192 B), but bank 3 also holds the K objects, so give the CP
  model the bank (the K pops and the max load / store must not share a bundle); initialise the 6 slots to -inf in the -m prep
  loop). The exact test then reads ONE 64-lane vector per row block (rowmax8 of it): FP 393 -> 0,
  T 164 -> ~70. The b / DM_ADA ports are at 27 / 29 of II 34 and the II is register-bound: budget II 34 -> 35 (B +7 per rb).
  Bit-identical (max is exact).
- **S3** the exact test + rescale + the -m row store as one asm call (k_rb's C++ minsize code, 1,056 B, goes): T + R -60.
- Prices (cumulative after P60g; kappa 0.5 / 0.6 / 0.7): S1 -0.31 / **-0.36** / -0.40; + S2 -0.32 / **-0.38** / -0.43; + S3
  -0.05. Without P60g: S1 -0.60 +- 0.10, S1 + S2 -1.32 +- 0.22, S2 alone -0.73 +- 0.12.
- Pre-registered (one core, r13/attkb.py --krb): S1 rescale 170-230 cycles incl. call; S2 warm unflagged 261 -> 261-275 per rb,
  flagged 1,129 -> ~650.

## P29b / P29c / P29d / P29s / P52. The SSM, re-counted on the deployed elf (ranks 2, 4, 5)

The r14f EW elf has exactly r12e's loops (r15/loops_r14f_0_5.out; use `llvm-objdump -z`: plain -d elides runs of 2-byte nops,
which made loga_p look like 11 bundles instead of 23). Per chunk (216 per block; 1 cycle per chunk = 1.43 us per step), static
= loop bundles x trips + straight-line; slot counts from the .td classes (r15/slots.py):

| pass (deployed, static cycles per chunk) | slot use per trip (vec / st / ld / mv) | bound | generated target | saving, ms |
|---|---|---|---|---|
| P29b loga_p 23 x 8 + exp_pA 20 x 16, pB 12 x 16, pC 6 x 16, pD 4 x 16 + 275 calls = **1,131** | loga 4/5/3/0; pA 5/3+2/1/3; pB 6/5/1/0; pC 3/4/3/1; pD 4/3/2/0 | 20 vec ops per vector | 16 x 20-23 + 60-100 = 380-470 | **-0.95 .. -1.08** |
| P29c recurrence 108 x 8 + 25 = **889** | 20 / 19+8 / 18 / - | ~23 per 64 lanes | 210-300 | **-0.84 .. -0.97** |
| P29d conv 90 x 8 + 330, z 25 x 8 + 44 = **1,294** | 20 / 15+14 / 23 / - ; z 1/4/6/2 | ~22 / ~5 | 600-720 | **-0.82 .. -0.99** |
| P29s sqrt_lut_n 54 x 8 + 80 = **512** | 2 / 11+9 / 19 / 26 | mv ~30 per 64 | 180 (int16 table) - 345 (pipelined) | -0.24 .. -0.48 |
| P52 out copies **798** | copy_v 6 bundles per trip (2 ld, 2 st) | | 0 (in place) - 300 | -0.71 .. -1.14 |

- Every loop sits at 2-5x its slot bound because Peano does not pipeline these chains (SKILL 3). Even without fusion, pipelining
  each existing pass at its ResMII (loga 2.5, pA 5, pB 6, pC 4, pD 4 per vector = 21.5) gives ~16 x 21.5 + 5 pass overheads
  ~ 420-460: a C++-only route (constant trips, restrict params, constants as locals) is worth trying first for P29b if text allows.
- P29a's measured saving equalled its bench delta exactly: SSM savings transfer 1:1. The deployed SSM wall (~6,300 cycles per
  chunk on the timeline) is ~1,100 above the static sum of its passes (~5,200 incl. sigk / gate / tables): before P29c / d run
  the EW phase stamps (npu-tapnext r8/ewprof.py, `EW_DEFS=-DPROF`, on r14f) to see whether that gap is ring waits (stays) or
  code (then also shrinks). The input floor is 7 KB per chunk at 4 B/cycle = 1,792 cycles: compute stays binding after all five.
- Pre-registered (one core, apb --cc, 16 vectors): deployed loga_exp chain 1,130-1,170; generated 380-470.
- **F1** (FA4 analogue, parked): after P29b the chain is vec-bound while the LUT port is idle; doing 2^f for half the vectors by
  a 256-entry int16 table (index = f's 8 bits, exponent by int16 add) saves ~6 vec ops per vector on those: ~-0.05 .. -0.15 ms,
  not bit-identical (within 1 bf16 ulp). Not worth a DAVIS round now.

## V1 / K1 / K2 (rank 6, new; uniform per row block, so 1:1)
- **V1**: fold the row sums into attv (P x ones: 6 more mmuls per row block = 1.5 per II-21 iteration; vec is 12 / 21) and call
  attv once per (tile, chunk) with the sums: 136 -> ~110 per rb: -0.21 +- 0.03 ms. Risk: the ones panel costs an x register.
- **K1**: the batched call's per-(tile, chunk) C++ (-m rows prep + flag OR scan, ~46 per rb) into attkb's preamble: -0.27 +- 0.03.
- **K2**: chunk 0 always takes the exact path; a QK-only first pass (no exp, no push) + test + the exact P: 1,005 -> ~600 per rb:
  -0.19 +- 0.03 (needs a second asm entry: text).

## T4. GEMM text (64 B free; `r15/r14f_0_2.nm`) — rank 7
main 2,080 (IRON control), mm_ks 1,824, acc_epilogue 1,456, attn_chunk_k 1,184, k_rb 1,056, attkb 1,050, bgelu 846, bias_pcs
752, attn_end 752, attn_begin 608, rescale_rb 560, attv 502, gup 426, attn_chunk_v 416, attvs 358, mm_kstep 320, exp32 304.
Order: S1 (-360 B), T3f mm_kstep as constant records + tail call (-200 B), S3 replaces k_rb's exact path (-~400 B), then V1 /
K1 / K2 spend it.

## FlashAttention-2 / 3 / 4: what transfers to AIE2P (pass-10 brief; ANALYSIS 5)
| idea | ours | transfer |
|---|---|---|
| FA2: fewer non-matmul ops (normalise once at the end), Q split across workers, skip masked blocks | attn_end divides once; rows across cores; ATTKE zero panels | already done |
| FA3 / FA4: ping-pong softmax of tile A with MMA of tile B; async producer warps | attkb's 3-stage modulo schedule interleaves QK of i + 1 with exp of i; DMA is the async producer | done on one in-order core; cross-core specialisation would only move work |
| FA4: conditional rescale, tau = log2 256 = 8 (x256), whole warp rescales together | TAU = 8 nats (x2,981), whole row block rescales | ours is looser already; their CORRECTION WARPGROUP takes rescales off the critical path: our analogue is P60g + S1-S3 (the rescale off the lockstep) |
| FA4: software exp2 (degree 3, Horner, floor by +2^23+2^22 in round-down) on FMA for 10-25 % of a row; MUFU for the rest | exp by table on the LUT port (attention), by arithmetic on vec (SSM) | **F2** attention: attkb II 34 is register-bound (ResMII 29: DM_ADA 29, b 27, vec 15): moving lanes to vec adds 6 x constants to an x file at maxlive 8 -> <= -0.1 ms, numerics: parked. **F1** SSM: the reverse split (table for some vectors): -0.1, parked |
| FA4: LPT (longest-first) tile scheduling | the deal | P60h adaptive deal: parked (<= -0.33 beyond P60g) |

## Microbenches named for main (cannot be separated statically)
1. EW phase stamps on r14f (`EW_DEFS=-DPROF`, r8/ewprof.py): the ~1,100-cycle gap between the SSM wall and its static passes.
2. If P60g lands outside 614 +- 18 us per block: the per-column K-object wait in situ (KRB_STAMP-style stamps around
   `in_b.acquire` in one_attn, pass 0, one core per column) tells lockstep (waits) from cost (no waits).
3. S2's II: CP-SAT on gen_attkb with the running-max stream (laptop), before any build.
