# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 7, 2026-10-10, after round 11 (`full12r11d`, 52.07 ms at depth 12; `RESULTS_FROM_MAIN.md` round 11). Revision 6 is in
`rev6/PROPOSALS.md`, older ones in `rev5/` .. `rev1/`. The analysis behind every number is in ANALYSIS rev 7. Savings marked
[sim14] come from `tools/sim14.py`: the r11b timeline + the g11-r11gp composition, per-op rms error 5.8 us, base 51.96 ms
against 52.07 measured; output in `r12/sim14_r11b.out`. Loop IIs marked [GEN] are what the main agent's msched.py reaches for
the op list in `tools/g1sched.py` (`r12/g1sched.out`). All figures are for depth 12, each change alone against r11d unless
stated.

**Headlines.**
- A table-lookup loop's floor is **10 cycles per 32 lanes** (the QEY_RS and DM_ADA ports), not 8. msched lands at 11-13,
  because the 12 x registers run out. The fully fused gated epilogue does not allocate at any II: **G1 is two passes**.
- **G1 left: -1.76 +- 0.35 ms.** Its first step, the gated product + push pass (`gup`, II 6 per 64, no LUT), is -1.0 by itself.
- P43 missed because half of it was not built (ANALYSIS 0): **P43b** holds the rest (-0.36).
- The attention K phase re-prices to **-2.9 +- 0.8 ms** (rev 6 -4.5): the exp's ResMII is 20 per 64, and dm pressure keeps
  QK^T and the exp from overlapping.
- The SSM (P29) re-prices up to **-3.6 +- 0.8 ms** with the passes' IIs [GEN]. P52 (SSM outputs written in place) is new: -0.8.
- Host part: H3 lean heads + H1 overlap + H2 a warm, pinned pool takes **2.0-2.2 -> ~0.9 ms** with no numerics change.
- Ceiling 31.6 ms (35.9 without the speculative k-step row); P14 today 38.3 ms per frame.

Common rules (SKILL 3, 9):
- one dispatch with a short timeout first for every new design, and the known-good test after any timeout;
- record Turbo / Default and gpu_busy; report stamp medians and percentiles;
- bench every new loop on one core (apb.py) before building it in, with its U passed to every harness;
- **read the emitted loop of the binary you bench.**

## Ranking (revision 7)

| rank | id | what | cost | risk | expected [sim14] |
|---|---|---|---|---|---|
| 1 | **G1a** (P47) | `gup`: up bias + gate x up + bf16 + ups + bfp push, generated (II 6 per 64), replacing the gated product / push C++ | S | low | **-1.02 +- 0.2 ms** |
| 2 | **H1 + H3 (+ H2)** | heads as lean numpy (H3), heads(t) overlapped with dispatch(t+1) (H1), a pinned warm pool (H2) | S | none | whole step **54.06 -> 53.0 +- 0.3 ms**; throughput 52.6 |
| 3 | **P29a + P29b** | SSM sigmoid pass and the log_a / exp chain, generated (needs int16 ops + pass cutting in msched) | M | low | **-1.0 + -1.0 ms** |
| 4 | **G1b** (P47) | one fused bias + GELU kernel (flat-edge tables; II 12-13) for yx, the gate half and mlp0, + panpush (II 2) | M | medium | **-0.74 ms** more (G1 total -1.76) |
| 5 | **P48 + P49** | attention K phase: exp + push generated, QK^T + max / rows8 in asm | M-L | medium | P48 -0.9, P49a -0.8, P49b -1.15; **-2.9 +- 0.8 together** |
| 6 | P44 -> P7-E | EW norm stamps + EW input rate, then norm passes at their port bound | S then M | low / medium | -1.5 (4 B/cycle) to -3.3 ms |
| 7 | P29c + P29d, **P52** | sqrt + recurrence, conv + z, generated; SSM outputs written in the ring + conv cache by rotation | M | medium | -1.2 / -1.0 / **-0.8** |
| 8 | **P53** | msched: register-pressure-aware scheduling + automatic spill through vst.conv / vldb + kernel-only prologue | M | none (CPU) | II 13 -> 11 on LUT loops: -0.1 (G1) to -0.4 ms (P29a, P48); unblocks the text budgets |
| 9 | **P43b** | B fills before the preceding EW op's finish; the memtile B forward one tile deep (K/64 + 1 objects) | S | low | **-0.36 +- 0.2 ms** |
| 10 | **P54** | one-core probes: vfloor from a bm source (QEY_RS 10 -> 8), and the biased LUT base | S | low | decides II 11 vs 13 on every LUT loop; 0 ms itself |
| 11 | P35 + P24b | B locks inside mm_ks; acc_zero folded | S-M | medium / low | -0.98 together |
| 12 | P50 | attention V phase in asm | M | medium | -1.4 ms |
| 13 | H4 | patch embedding as a GEMM op on the NPU (bfp16 pixels; DAVIS-gated) | M | numerics | host -0.4 to -0.9 ms, dispatch +0.04 |
| 14 | **P51** | the norm writes the next GEMM's A slot directly (EW core -> memtile), no DDR round trip | L | high | **-0.96 ms** (4 after-EW boundaries 27 -> ~8 us) |
| parked | P7 full fold, P7c, P24c, P36-P41, the k-step 1,600 row | see rev 6 | | | |
| last | P14 | two frames in flight | L | medium | 38.3 ms per frame today (x1.36); 24.4 at the ceiling |

Combined (sim14, depth 12, each on the previous):

| added lever | step ms |
|---|---|
| G1 | 50.20 |
| + P43b | 49.84 |
| + P48 + P49 | 46.97 |
| + P29 (x 0.85) + P52 | 42.59 |
| + P7-E high | 39.27 |
| + P35 + P24b | 38.29 |
| + P50 | 36.88 |
| + P51 | 35.92 |
| + k-step 1,600 | **31.6** |

Depth 10 at that ceiling: 26.4 ms. Host part after H1-H3: ~0.5-0.9 ms on top.

## Status of the earlier proposals

| id | status | verdict / what is left |
|---|---|---|
| P45 hazard ladder | **DONE differently** (round 11): H1-H4 moot; the real rules are LLVM visibility, the mv bypass, itinerary ports, aligned loop ends | my premise was wrong (the 54.7-cycle loop was never pipelined). Now in MACHINE R7 |
| P46 generator | **DONE** (msched.py, validated bit-exact) | P53 (allocator) + the P29 / P48 needs in ANALYSIS 3 |
| P47 G0 | **DONE** (-1.41 by GELU_II + asm GELU at II 13; predicted -1.4 +- 0.3) | G1 = G1a + G1b (ranks 1, 4) |
| P43 | **PARTLY** (-0.35; predicted -0.8 +- 0.3) | built: B before `finish_deps`. Not built: before the EW op's finish; the deep B forward -> P43b. Pre-registered check: the r11d timeline shows mlp3_a 166 -> 137 +- 8 us |
| P42 stack | tool DONE (stackaudit) | shrink: GEMM 0xB00 -> 0x800 (frees 768 B for G1b's flat-edge tables), EW 6,144 -> 4,416 (for P27 / P29) |
| P44, P7-E | open | rank 6 |
| P27 | open, **folded into P29b** | `a_om_lut_i` (the a table) is an alternative op list for P29b: 13 vec + 8 LUT per vector against ~20 vec; DAVIS-gated |
| P29 | open, re-priced -3.6 +- 0.8 | ranks 3, 7 |
| P48, P49, P50 | open, re-priced | ranks 5, 12 |
| P35, P24b | open | rank 11 |
| P38 counters | open | useful for the G1 / P29 in-situ checks (bank + DMA stall share) |
| P7 / P7a / P7c | parked (rev 6 section 4 stands) | |
| P3 txnlint | open | worth it before P43b and P51 (sequence order) |
| P14 | last | |

---

## G1a. The gated product + push pass, generated — **S, risk: low** (rank 1)

**Hypothesis.** Of a gated tile's 9,240 cycles [MEAS g11-r11gp 5.11 us], ~5,750 are the C++ product / push / stash loop that
runs after the bias pass (1,290) and gelu_ms (~2,200). As a modulo-scheduled loop it is 19 ops per 64 lanes [GEN: ResMII 6
(DM_WM), II 6, U 2, 6 stages, x MaxLive 6, dm 5].

**Procedure.**
1. Per 64 output lanes (one bfp block):
   - load v (the GELU'd gate, bf16) x2 on ldb;
   - load the up pieces as 2 bm quarters x2 on lda;
   - load the up bias x2 on ldb;
   - 2 bias MACs (`vmac.f y, a, b, 1.0`); `vconv` u;
   - `vmul.f` v x u; `vconv` to bf16; `vconv.fp32.bf16` (ups) into cml / cmh of one dm;
   - `vst.push.576.conv.bfp16ebs8.fp32`.
2. The push output stream state lives in asm: probe it first in apb (one core, a push-only loop against the C++ OutS bytes).
3. The stash of gated_b stays C++ for now.
4. Bench on one core with apb.py (bit-exact against today's product loop), then t1 / t2 / g11, then `regress --ref full12r9a`
   BIT-IDENTICAL.

**Expected (pre-registered):**
- one core: 6.3 +- 0.5 cycles per 64 lanes at n = 72;
- g11 PROF gated epilogue: 5.11 -> 2.44 +- 0.2 us per tile;
- step: **-1.02 +- 0.2 ms**;
- text: +0.75 KB explicit (0.45 kernel-only), against ~0.9 KB of C++ removed.

## H1-H3. Host part — **S, risk: none** (rank 2)

**Procedure.**
- H3: the heads in numpy fp32, with the weights pre-transposed once (or `torch.inference_mode` with a single fused path);
  compare with model.heads to 1e-6.
- H1: a worker thread runs heads(t) + the decode while dispatch(t+1) runs. Step8 returns frame t's tracks one frame late in
  throughput mode, or the caller waits on them in latency mode.
- H2: 4 torch threads pinned away from the tenants' busy cores (`OMP_PLACES`, `OMP_PROC_BIND=close`), and a 1 ms warm-up spin
  before the frame is due. Re-run r11/hostprof.py with the load average recorded.

**Expected:**
- heads 0.83 -> 0.2 +- 0.05 ms;
- conv median 0.5, p90 <= 0.65 ms;
- whole step at depth 12: 54.06 -> **53.0 +- 0.3 ms** (latency); per-frame throughput 52.6.

## P29a + P29b. SSM sigmoid and exp chains, generated — **M, risk: low** (rank 3)

**msched needs** (ANALYSIS 3.1):
- int16 vector ops and int16 views of x atoms;
- the MAC-unit round (magic add) and `filter_even` as a vshuffle mode;
- pass cutting (P29b has ~20 vec + ~13 st ops per vector: 2 passes at 5 dm);
- several streams per loop;
- short prologues (EW text has 144 B free; replace exp_pA-D + loga_p 2,080 B and sigmoid_lut_n 384 B).

**Expected:**
- sigmoid 12 cycles per 32 lanes [GEN] (11 with the biased base): per chunk 1,168 -> 450 +- 60 cycles;
- exp chain 1,170 -> 450 +- 100;
- step **-1.0 and -1.0 ms** (x 0.85 realism: -0.9 each);
- bit-identical (the same op sequences).

## G1b. One fused bias + GELU kernel + panpush — **M, risk: medium** (rank 4)

**Design** (ANALYSIS 2.2): fp32 pieces + a bias stream -> bf16(acc + bias) -> clamp -> floor -> 8 LUT loads -> line -> x
products -> bf16 in place.
- Flat-edge tables, so the s MAC takes x, and x is spilled once through `vst.conv` from the accumulator and reloaded on ldb.
- The table pointers (tanh / erf) and the 2D walks are parameters.
- Used for yx (out to the drained layout), the gate half of up_a / up_b (in place), and mlp0 (in place, then panpush, II 2).
- [GEN] II 13 with flat edges, 12 with the biased base (P54); U 2, 6 stages.
- Text ~1.25 KB explicit: fits after gelu_ms (718 B) and the flag 1-4 C++ paths go (ANALYSIS 2.2).
- L1: +512 B for the flat-edge tables: shrink the GEMM stack first (P42).

**Expected:**
- one core: 13.5 +- 1 cycles per 32 lanes (in place);
- per tile: yx GELU 2.92 -> 2.2 us, the gated gate half (with G1a) -> 1.57 us in total, pan 4.30 -> 2.4 us;
- step **-0.74 ms beyond G1a** (G1 total -1.76 +- 0.35; at II 11 -1.83);
- bit-identical.

## P48 + P49. The attention K phase — **M-L, risk: medium** (rank 5)

- **P48** exp + push per row block, generated: ResMII 20 per 64 [GEN], reached 31 with the biased base and none without, at
  today's x pressure. P53 first, or hold -m in a pinned dm and m as one x.
  Expected ~250 cycles per (rb, chunk) against 330: **-0.9 +- 0.3 ms**.
- **P49a** QK^T in asm (Q pops lda, K pops ldb, 6 chains on 5 dm as pairs): 138 -> ~70, **-0.8**.
- **P49b** the running max during the score formation, rowmax8 by shuffles, rows8 by vbcstshfl: ~150 -> ~50, **-1.15**.
- Together: 571 -> ~320 cycles per (rb, chunk), **-2.9 +- 0.8 ms**; t4 attention bit-identical.
- Overlapping QK^T with the exp inside one kernel is blocked by dm (4 + 3 > 5): do not plan for the ~150-cycle port floor.

## P52. SSM outputs in place, conv cache by rotation — **M, risk: medium** (rank 7, new)

**Hypothesis.** The SSM spends 798 cycles per chunk on output copies (x, x-1, x-2 = input parts 1-3 unchanged; h and z
computed into L1, then copied into the ring) [MEAS r9 PROF].

**Procedure.**
1. Acquire the 8 output slots before the recurrence, and store h (fp32) and the z panels straight into the ring (the 6 KB
   output starts on a slot; handle the wrap as out_copy does).
2. Drop the x / x-1 / x-2 copies: keep the conv cache as 4 per-frame x buffers in DDR, with the runtime sequence picking
   offsets by frame mod 4 (4 .bin variants per depth, swapped like `set_depth`). This also halves the SSM's output DMA.

**Expected:** out copies 798 -> ~250 cycles per chunk: **-0.8 +- 0.3 ms**; bit-identical.

## P53. msched: allocation that reaches ResMII — **M, risk: none** (rank 8, new)

**Hypothesis.** On every LUT loop msched lands 1-3 cycles above ResMII because x MaxLive exceeds the free registers after
its list schedule + sink pass [GEN: yx 13 vs 10, gelubf 13 vs 10, sig 12 vs 10, aexp 31 vs 20 per 64].

**Procedure.**
- swing modulo scheduling or stage scheduling after IMS (minimise lifetimes at a fixed II);
- automatic spill choice: a long x through `vst.conv` from its accumulator source + `vldb` (QEY_WB only), costed by the ports;
- constants as scalars where the op takes one;
- a kernel-only prologue with redirected leading stores;
- check against apb on gelubf.

**Expected:** gelubf 13 -> 11, sig 12 -> 11, aexp per 64 31 -> 24-26. Gains are inside G1b / P29a / P48 (up to -0.4 ms in
total), plus ~0.4-0.6 KB of text per kernel.

## P43b. The rest of P43 — **S, risk: low** (rank 9)

**Procedure.**
1. For a GEMM op that follows an EW op, issue its B fills (in the GEMM's own task group, created early) before the EW op's
   `tg.finish()`.
2. Deepen the memtile B forward so that a whole first tile of B is resident: l2_depth K/64 + 1 = 13 objects (60 KB) for
   K = 768, 25 (115 KB) for K = 1536. Memtiles 0-2 have ~136 KB free; check this against A_HALF's slot.
3. Out_proj after attention: no change (attention's K / V fills share the channels).
4. Run P3 txnlint if it exists. Regress bit-identical.

**Expected (pre-registered):**
- r11d timeline first (no build): mlp3_a 137 +- 8 us;
- after P43b: mlp3_a -> 115-125 us, yx / up_a / qkv / mlp0 boundaries -3 to -5 us each;
- step **-0.36 +- 0.2 ms**.

## P54. Two port probes — **S, risk: low** (rank 10, new)

1. `vfloor.s32.bf16` with a bm-quarter source (`$src[$imm]` form) on the GELU chain, with the clamped value moved to a bm by
   a vshuffle: is the result identical to the w-source form, and does QEY_RS drop to 8 per 32 lanes (II 10 placeable)?
2. The LUT base: does `vldb.4x64` take an absolute address per lane from the index, or a base register? If the latter, point
   the base 16 x BIAS bytes into the table and drop the 2 index adds.

**Expected:** (1) works and frees QEY_RS (a structural fact from the .td; unprobed); (2) 50/50. Together they decide whether LUT
loops reach 11 or stay at 13.

## P51. The norm writes the next GEMM's A slot — **L, risk: high** (rank 14, new)

**Hypothesis.** After each norm the dependent GEMM reloads its A (8 columns x 124-248 KB, the norm's own output) from DDR:
boundaries +26.7 us at yx, up_a, qkv and mlp0 [sim14]. Column c's EW core computes exactly column c's 144 rows, i.e. the
A slot of column c.

**Procedure.**
- The EW core's MM2S streams its normalised panels straight into the memtile A slot (S2MM 0 / 5) instead of to the shim.
- The X' residual still goes to DDR.
- The A slot's runtime tasks then wait on the EW core's lock instead of DDR fills.
- Mind the IRON router bug (multicast + unicast on one channel) and the BD budget.

**Expected:** after-EW boundaries 27 -> ~8 us: **-0.96 ms**; DDR traffic -12 MB per step. An L-sized change; do it after the
vector kernels.

## Unchanged from rev 6 (see rev6/PROPOSALS.md)
P44 / P7-E (procedure and pre-registration), P35, P24b, P50, P7 / P7a / P7c, P36-P41, P14.
