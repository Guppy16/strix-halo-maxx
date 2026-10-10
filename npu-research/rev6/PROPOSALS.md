# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 6, 2026-10-10, after round 10 (`full12r10r`, 53.89 ms at depth 12; `RESULTS_FROM_MAIN.md` round 10). Revision 5 is in
`rev5/PROPOSALS.md`, older ones in `rev4/` .. `rev1/`. Analysis behind every number: ANALYSIS rev 6. Savings marked [sim13] come
from `tools/sim13.py` (the r10r timeline + the g11-r10rp composition, per-op rms error 5.2 us, step 53.81 vs 53.83 ms; output
`r11/sim13_r10r.out`), depth 12, each change alone against r10r unless said.

**Headlines.**
- The GELU-path epilogues cost 384 us per block (4.6 ms per step), not ~3.3: the bias pass, the gated products / pushes /
  stash and 18-36 calls of 8 vectors per tile sit beside the GELU loop. A fused, modulo-scheduled epilogue is worth
  **-3.3 to -3.7 ms** almost independent of its II (ANALYSIS 2).
- The GELU loop's resource floor is **8 cycles per 32 lanes** (ldb: 8 LUT loads; mv 8 with the index adds); registers make
  II 10 the realistic target. The forced-II failure (54.7 cycles) has four candidate hazards; a 5-rung bench ladder (P45)
  names it before anyone hand-schedules.
- mlp3_a's first tile is **B-starved** (49 us): the sequence issues every dependent op's B fills after the dependency wait (P43).
- The norm fold needs residual sums, X' panels and row stats that rev 4 / 5 priced as free; re-priced at -4.6 ms for an L+
  change with L1 / text blockers. Making the **EW norm kernels reach their port bound (P7-E)** gets -1.5 to -3.3 ms for M.
- Attention's K phase is 571 cycles per (row block, key chunk): 55 % exp + push, 24 % QK^T, 21 % max / broadcast / control.

Common rules (SKILL 3, 9): one dispatch with a short timeout first for every new design and the known-good test after any
timeout; record Turbo / Default and gpu_busy; report stamp medians and percentiles; **bench every new loop on one core
before building it in**.

## Ranking (revision 6)

| rank | id | what | cost | risk | expected [sim13] |
|---|---|---|---|---|---|
| 1 | **P45** | GELU hazard bench ladder (L1-L5, one core) | S | low | names the forced-II hazard; gates P47's II. 0 ms itself |
| 2 | **P47** (G0 -> G1) | GELU epilogues: one call per tile + gated pass (C++), then fused bias + GELU (+ product + push) modulo-scheduled in asm | M (G0 S-M) | medium | **G0 -1.4 +- 0.3; G1 -3.6 +- 0.4 ms total** (II 10: -3.74, II 16: -3.33) |
| 3 | **P43** | B fills of a GEMM issued before its dependency wait; memtile B forward one tile deep | S | low | **-0.8 +- 0.3 ms**; mlp3_a 161 -> ~125 us |
| 4 | **P44** | EW norm stamps (per row block: input wait / compute / output wait) + an EW input-port rate probe | S | low | decides P7-E low (-1.5) vs high (-3.3); 0 ms itself |
| 5 | **P48** (A1) | attention exp + push modulo-scheduled per row block (the P47 machinery) | M | medium | **-2.1 +- 0.5 ms** |
| 6 | **P7-E** | EW norm passes at the port bound (generator-built); P7-P (two input channels) if P44 shows a 4 B/cycle port | M | medium | **-1.5 (4 B/cycle) to -3.3 ms (8 B/cycle or P7-P)** |
| 7 | **P46** | the modulo-schedule generator (tool) | M-L | none (CPU) | enables P47 G1, P48-P49, P29, P7-E |
| 8 | P27 + P29 | SSM: a by table, then the chains modulo-scheduled | S-M / L | low | -0.7 / -3.5 ms |
| 9 | **P49** (A2 + A3) | QK^T + score store in asm with the running max fused; rows8 by vbcstshfl; then one K-phase kernel with A1 | M-L | medium | -0.8 / -1.3 alone; **K phase as one kernel -4.5 +- 1.2 ms** (with A1) |
| 10 | P35, P24b | B locks inside mm_ks; acc_zero folded | S-M / S | medium / low | -0.44 / -0.54 ms |
| 11 | P38 | stall counters in the PROF record | S | low | explains the 201 in-situ cycles; 0 ms itself |
| 12 | **P50** (A4) | attention V phase in asm | M | medium | -1.6 ms |
| 13 | P7a | the fold's numerics emulation (no device) | M | none | go / no-go per site and block; pre-registered outcome in P7 |
| parked | P7 full fold, P7c, P24c | fold (L+, L1 / text blockers), residual feature permutation, bias as accumulator init | | | -4.6 +- 1.5 (full fold); see P7 |
| last | P14 | two frames in flight | L | medium | today 40.2 ms per frame (x1.34); x1.44 after P47 + P43 + attention K |
| parked | P36, P39, P40, P41 | deep-K visits, neighbour-memory A, A over 2 S2MM, a 4th GEMM row | L | high | the k-step 1,600 row: -4.3 ms, needs L1 and R_A >= 6.5 |

Combined (sim13, depth 12, each on the previous): P47 G1 (II 12) 50.2 -> P43 49.4 -> attention K (P48 + P49) 44.9 -> P27 + P29
40.7 -> P7-E high / P7-P 37.5 -> P35 + P24b 36.5 -> P50 34.9 ms (28.7 fps) -> k-step 1,600 **30.5 ms** (needs L1 and R_A).
Depth 10 at that ceiling 25.5 ms. P14 at the ceiling ~23 ms per frame (latency ~46 ms).

## Status of the earlier proposals

| id | status | verdict / what is left |
|---|---|---|
| P1 twin builds | DONE | compute-bound |
| P2 / P15 stamps | DONE | P38 adds counters |
| P3 txnlint | open | useful before P43 (fill order) and P14 |
| P5 A slot (P5r, P5a, chunks) | **DONE** (round 10): -1.35 (P5r + P5a; predicted -1.6 +- 0.3) and -2.0 (runtime memtile tasks; predicted -1.7 +- 0.7) | independent ops' A now loads under the previous op (boundary -6 to -11 us); dependent ops pay +20-38 us (P43) |
| P7 norm fold | open, **re-priced** (rev 6): -4.6 +- 1.5 for an L+ change; P7-E first | ANALYSIS 4 |
| P8 numerics harness | DONE | P7a extends its capture |
| P10 ports | **DONE**: clock 1.81 GHz; memtile -> core stream 7.95 B/cycle (r10/rtm.py) | the EW input port is not measured: P44 |
| P12 / P19 attention | open, **split** into P48 (exp), P49 (QK + max), P50 (V) | ANALYSIS 5 |
| P13 SSM | PARTLY | P27, P29 left |
| P14 | open, re-priced | last |
| P18 2x2 | blocked by L1 | |
| P20 | DONE (round 9) | |
| P22 program memory | (a) + (b) DONE | GEMM text 15,840 / 16,384: P47 must fit ~2.5 KB after removing the C++ GELU paths |
| **P23** row balance | **DONE** (round 10): -0.85 (predicted -1.2 +- 0.3, missed by 0.05) | the qkv coupling (80 us) came from PROF walls; the full step had ~15 |
| P24b | open | |
| P27 | open | unblocked (rev 5) |
| P29 | open | via P46 |
| P30 bank linter | open | P46's check() includes it |
| P33 timeline | **DONE** by the main agent (r9s and r10r timelines) | sim13 is built on it |
| P34 A-rate twin | open, low priority | only matters for the k-step 1,600 row |
| P35 | open | -0.44 |
| P36-P41 | parked | |
| P38 counters | open | |
| P42 EW stack | open | needed for P27's second copy |

---

## P45. GELU hazard bench ladder — **S, risk: low** (rank 1)

**Hypothesis.** `gelu_ii16` ran 54.7 cycles per 32 lanes for 17 static bundles; the deployed loop runs at its static count. One
of four differences stalls the core (ANALYSIS 2.5): H1 LUT-load destinations re-used while a vldb.4x64 to them is in flight;
H2 vmov into accumulator quarters of a dm a vmac is using; H3 an ordinary vldb mixed between the 4x64 lookups; H4 vmov chains
whose latency the schedule took as 1.

**Procedure.** Hand-written asm loops (zero-overhead loop rules of SKILL 3), one core, timer around n = 144 trips x reps,
gxb.py style; tables placed as deployed (ab bank 0, cd bank 3):
- L1: 8 vldb.4x64 (lo / hi pairs, indices from a fixed register) into 8 distinct w registers, II 8;
- L2: L1 with the 8 destinations re-used every trip and each read by a vmov 7 cycles after its load;
- L3: L1 + one ordinary `vldb x, [p], #64` per trip, II 9;
- L4: II 6: `vmac.f dm1, dm1, x0, x1` with a `vmov bmlh1, x2` 3 cycles before each mac reads dm1 (values garbage);
- L5: the P47 schedule at II 16, 12, 10 (from P46 or by hand).

**Expected (pre-registered):** L1 8.0 +- 0.2 cycles per trip; L3 9.0 +- 0.2. Of L2 / L4 I expect **L2 (H1) or L4 (H2) to stall,
not both**, with H1 the more likely (the LUT path is the only thing the deployed loop never pipelines). If L1-L4 all run at
their static counts, H4 (or a combination) is it: then L5 at II 16 with every vmov latency taken as 2 must run at 16.

---

## P47. GELU epilogues: G0 (C++) then G1 (fused, modulo-scheduled) — **M, risk: medium** (rank 2)

**Hypothesis.** Per tile today: yx GELU 10,990 cycles (38.7 per 32 lanes), gelu_pan 14,824 (51.5), gated 12,724 (88.4 per
output vector). The GELU loop is a third to a half of it; the bias pass, the gated products / pushes / stash copies and the
calls of 8 vectors are the rest.

**Procedure.**
- **G0 (S-M, C++, bit-identical):** one `gelu_lut_n` call per tile over the three piece segments (pass n = 144 / 288 and the
  segment pointers; 44.1 -> 34.2 per 32 lanes), and the gated product / push pass as an array pass (constants as locals,
  separate in / out pointers, all loads before stores, two row blocks per trip; SKILL "Peano software-pipelines simple loops").
- **G1 (M, asm via P46 or by hand, bit-identical):** one call per tile and variant (ANALYSIS 2.4): fp32 pieces + bias stream
  in, bias mac, clamp, floor index, 8 LUT loads, 4 shuffles, s = off + slope x, sh / sl split, x reloaded for the products,
  output bf16 (yx) / bfp push (pan) / gate x up product + push (gated). II 10 (yx, pan), 11 (gated); MVE unroll 2;
  kernel-only loop with the first S - 1 stores redirected; flat-edge table segments so the s mac takes the unclamped x.
  Text: ~2.5 KB for the three after removing flags 1-4 from acc_epilogue and gelu_lut_n (llvm-size check).
- Directed tests t1 / t2 / g11 bit-identical, then `step8.py regress --ref full12r10r` BIT-IDENTICAL.

**Expected (pre-registered):**
- one-core bench: G1 yx 10.9 +- 1.5 cycles per 32 lanes (incl. call), pan 11.4 +- 1.5, gated 12.0 +- 1.5 per output vector;
- g11 PROF epilogue per tile: yx GELU tile 10,990 -> 3,100-3,600; gelu_pan 14,824 -> 3,200-3,800; gated 12,724 -> 1,800-2,400;
- step: **G0 -1.4 +- 0.3 ms; G1 -3.6 +- 0.4 ms in total** (53.9 -> ~50.3). If P45 forces II 16: -3.3.

---

## P43. B before the dependency wait — **S, risk: low** (rank 3)

**Hypothesis.** design8's sequence runs `finish_deps(op)` before the op's B fills. A dependent op's B then leaves DDR together
with its A slot load (8 columns x 124-248 KB) and loses: in g11-r10rp mlp3_a's first tile waits 49 us on B (2,324 cycles per
k-step outside mm_ks, 33 K cycles before the first k-step; busy per k-step normal). In the full step mlp3_a is 48.6 us slower than
mlp3_b and the post-EW / post-attention ops carry +20-38 us boundaries.

**Procedure.** In design8's sequence, for each GEMM op issue its B fills (and the memtile B tasks, if any) before
`finish_deps(op)` and, for ops after an EW op, before the EW op's `tg.finish()`. Give the memtile B forward a depth of one tile
(K / 64 + 1 objects: 13 x 4,608 = 60 KB for K = 768, 25 x 4,608 = 115 KB for K = 1536; memtiles 0-2 have ~136 KB free) so the
first tile's B is resident. Mind the group rule (a fill sits in a group finished after it completed) and the shim BD pool;
run P3 txnlint if it exists. Regress bit-identical (no data change).

**Expected (pre-registered):** g11 mlp3_a first-tile B wait 49 us -> < 5 us; full step mlp3_a 161 -> 120-130 us; yx / up_a /
qkv / mlp0 / out_proj boundaries -5 to -10 us each; **step -0.8 +- 0.3 ms**.

---

## P44. EW norm stamps and the EW input rate — **S, risk: low** (rank 4)

**Hypothesis.** The norms run 123-147 us at 2.2-3.4 B/cycle of input. Either the EW kernel is the bound (then a modulo-scheduled
norm reaches the port) or the EW core's single input channel at ~4 B/cycle is (then P7-P, a second input channel).

**Procedure.** (1) ew8 `-DPROF` records for norm ops, as ewprof.py does for the SSM: per row block the cycles waiting for input,
computing, waiting for output slots. (2) rtm.py-style one-core probe: shim -> EW core packet stream into a ring, trivial
consumer: B/cycle.

**Expected (pre-registered):** input stream >= 7 B/cycle (same switch as P10's 7.95); norm1 / norm3 compute-bound (input wait <
20 %), norm0 within 25 % of the input port at 4 B/cycle only if the stream is 4 B/cycle. Decision: >= 7 B/cycle -> P7-E high
(-3.3 ms); ~4 -> P7-E low (-1.5) + P7-P.

---

## P48 (A1). Attention exp + push, modulo-scheduled — **M, risk: medium** (rank 5)

**Hypothesis.** aexp_push is 29 + 6 x 49 + 7 bundles per row block (~330 of the K phase's 571 cycles per (rb, chunk)); its floor is
16 LUT loads per 64 scores (96 per call).

**Procedure.** P46 kernel (or by hand): per row block, the 6 key blocks' 64 scores: d = bf16(s - m) clamped, index, 8 LUT loads per
32 lanes, line, bfp push of 64; the 3-term line `off - slope m + slope s` as today (bit-identical). II ~18 per 64 scores, the
row max operand as a pinned broadcast register. The score buffer stays one row block (no L1 for six).

**Expected:** ~150 cycles per call (from ~330); K phase 136 -> ~93 us per head; **step -2.1 +- 0.5 ms**; t4 attention bit-identical.

---

## P7-E (rank 6). EW norms at their port bound — **M, risk: medium**

**Procedure.** After P44: rebuild norm_rows / ln_stats as generator loops (P46): pass 1 sums the n_in inputs into X' (bf16, as
today) and accumulates sum / sum of squares per row (lane butterflies at the row end); pass 2 normalises, writes X' and the
panels. If P44 shows a 4 B/cycle input: P7-P, the residual input on the EW core's second S2MM (shim MM2S 1 packet-shared with B
on columns 0-2; check the router-bug rule: no multicast on those channels).

**Expected:** 4 norms 542 -> ~420 us per block (port 4 B/cycle, **-1.5 ms**) or ~270 us (8 B/cycle or P7-P, **-3.3 ms**);
numerics bit-identical to today's EW norms (same op order).

---

## P46. The modulo-schedule generator — **M-L, risk: none (CPU)** (rank 7)

Specification in ANALYSIS 3 (Machine from aie2p_lat.json + measured rules; Loop with streams, constants, LUT macro, bank tags;
iterative modulo scheduling with ResMII / RecMII / RegMII; register allocation over the unrolled kernel; gen_ks-style
check(); bit-level emulate(); kernel-only or explicit prologue; report with predicted cycles and text bytes). First targets:
P47 G1 (validates against P45), P48, then P29 and P7-E.
**Expected:** its predicted cycles per call within 10 % of the one-core bench for every kernel it emits (that is the acceptance
test of the tool).

---

## P49 (A2 + A3). QK^T, max and broadcasts in asm — **M-L, risk: medium** (rank 9)

**Procedure.** QK^T for one (rb, chunk): 48 mmul with Q pops on lda and K pops on ldb, 6 accumulator chains (one per key block,
8 dim-block macs each, distance >= 3), each block's add + bf16 conversion + store, and its running max (vmax on mv) as it is
formed; rowmax8 + the lazy-max test as today; rows8 by `vbcstshfl` (one mv op per 8 rows) instead of 8 scalar broadcasts.
Bit-identical (same MAC order per block). Then fuse with P48 into one K-phase kernel per row block: lda Q + K pops (96), ldb
LUT (96).

**Expected:** QK + store 138 -> ~68 cycles, max / broadcast / control ~150 -> ~40: **-0.8 and -1.3 ms alone**; the fused K-phase
kernel at ~180 cycles per (rb, chunk) with P48: **-4.5 +- 1.2 ms** in total.

---

## P50 (A4). Attention V phase in asm — **M, risk: medium** (rank 12)

PV (48 mmul) + the ones panels (6) + 8 O blocks loaded / stored per row block: 213 cycles per (rb, chunk) today, floor ~60.
**Expected: -1.6 +- 0.5 ms.**

---

## P7 (rev 6). The norm fold — **L+, parked behind P7-E; P7a may run any time (no device)**

Math, dataflow, numerics risks and the emulation design are in ANALYSIS 4. Summary:
- RMS: z = r (x W') + b with W' = diag(1 + s) W. LN: z = r (x W' - mu c) + b', c = column sums of bfp16(W'), b' = beta W + b.
  r, u = -r mu, c applied in hi / lo bf16 in the epilogue (RMS 5, LN 8 vec ops per 32 lanes).
- Producer side: residual parts through the A broadcast (~91 us per block), a second drain per tile for the panels and the
  stats partial (~21), centred per-tile stats (~6); consumer: 12 stats slots appended to tile 0's A send, a 1,152 B L1 buffer,
  the scale (+0.1 / +0.6 us per tile). **+161 us per block added, 542 removed: -4.6 +- 1.5 ms.**
- Blockers: GEMM L1 (932 B free), GEMM text (544 B free, shared with P47), the A-path residual protocol, the double drain.
- **P7a** (no device): capture X' per site / block, emulate T / F / F+P / E, ratio rms(z_F - z_E) / rms(z_T - z_E); go if median
  <= 1.10 and every block <= 1.25; 1.25-2 -> only with P7c; > 2 -> that site stays on the EW core. Pre-registered: norm0 passes;
  norm1 fails block 0 (18 small-g channels); norm2 / norm3 fail blocks 0-1 without P7c.
- **Measured now (weight side, `tools/p7w.py`, real weights):** bfp16(diag(g) W) vs g x bfp16(W): error ratio median 1.013 /
  1.020 / 1.023 / 1.016 (norm0-3), worst block 1.086: the weight side is not the risk; the small-g channels on the activation
  side are (`r11/p7w_smallg.out`).
- **P7c** (with P7a): one global permutation of the residual's 768 features that puts the small-g / large-x channels into their
  own 8-blocks (weights permuted on the host; the patch embedding and the heads take it at the ends).

## P14 (re-priced). Two frames in flight — **L, risk: medium** (last)

sim13 list bound: today 40.2 ms per frame (x1.34, latency ~80 ms); after P47 + P43 + the attention K levers 31.3 ms (x1.44: the
GEMM side shrinks first, so more EW work hides); after the SSM / norm levers x1.24; at the ceiling 23.2 ms per frame (43 fps,
latency ~46 ms). Shared shim channel 0 (A loads + EW inputs) is a new contention point to check. Still the last fps lever.

## Also small (no rank change)
- **P24c** (parked): bias as the accumulator's initial value (the first k-step loads a bias block instead of zeros): removes the
  bias loads / macs from every epilogue (gated becomes mv / st-bound instead of lda-bound). Not bit-identical (fp32 order);
  DAVIS-gated.
- **Biased LUT pointer probe** (with P45): a LUT base register pointing 16 x BIAS bytes into the table, negative indices: if it
  works, every LUT loop loses 2 mv ops per 32 lanes (GELU, attention exp, SSM sigmoid).
- Tools: P30's bank linter becomes P46's check(); P3 txnlint before P43 / P14.
