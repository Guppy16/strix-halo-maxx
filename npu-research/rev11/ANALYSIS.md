# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 11, 2026-10-10 (late). State analysed: round 15's deployed `full12r15m` (41.52 ms per depth-12 dispatch, Turbo,
gpu_busy 0, BIT-IDENTICAL to full12r9a), its elfs disassembled with `-z` (`r16/r15m_0_2.dis`, `r16/r15m_0_5.dis`), round 15's
EW PROF (r15b) and one-core benches (RESULTS_FROM_MAIN round 15). Revision 10 is in `rev10/`. Tools: `tools/asmlint.py`,
`tools/asmlint_hist.py`, `tools/try.sh`, `tools/sim18.py` (output `r16/sim18.out`). Written by the coordinating session from
research pass 11's hand-back.

## 0. Scorecard (rev 10 vs round 15)
- **P60g**: −1.06 vs −1.08 ± 0.22 (hit); per-block attention wall 610 vs 614 ± 18 µs (hit).
- **P29b**: −0.79 vs −1.0 ± 0.07 (miss): the loop was priced, not the call (7 stages × II 20 of prologue + preamble).
- **S1**: −0.29. Rev 10 priced S1+S2+S3 only together (−0.74); sim18 fits the in-situ rescale at R = 270 cycles (rev 10: 200).
- **S2**: not built. Rev 10 read bank 3's linker REGION (0xA80 = 2,688 B, the space for `.aie.bank3` sections) as free space;
  sgelu_cd + serf_cd fill 2,304, leaving 384 B. Fixed in §3.
- **P29c** hit (−0.84 vs −0.9). **P29d** conv hit (−0.72). **P52** miss (−0.18 vs −0.71..−1.14): stale baseline, then §4.
- **Unpriced**: main's div768 (−1.05 ms), minsize slot-count helpers called once per unit. Research had no tool that counts
  calls per unit of work; T-CALL (PROPOSALS rank 9) is that tool.

### 0.1 Method changes
- Whole-call static counts only: function body + call/ret delay slots (5 each) + loop setup + prologue + drain + glue.
- The deployed elf is the source of truth: every count here is from `llvm-objdump -d -z` of r15m.
- Peano's own code is the calibration set for the instruction model (asmlint "calib"): where the `.td` model disagrees with
  what the compiler schedules, the model is suspect, not the compiler.

## 1. asmlint against history (`tools/asmlint_hist.py`: ALL PASS)

| artefact (r16/hist/) | what happened | rule that fires |
|---|---|---|
| r11_g7.s, r11_p2.s | r11 GELU probes before the end-alignment rule | ZOL-END-ALIGN |
| r11_t1.s | first GELU_MS attempts (c + r ≥ c' + w timing, then the mv bypass) | LAT-UNLANDED, LAT-MV-BYPASS |
| cpy32_u2.s | r12 gen_copy U 2: 2-bundle body ran once | ZOL-MIN-BODY |
| bgelu_bg11_pad.s | r12 bgelu II 11: `.p2align` pad shifted a prologue vlda's QEY_WA onto a body vfloor | PORT (QEY_WA_L0_PORT) |
| bgelu_bg13.s / bg12 (fixed) | r12: x stored to its final place in an early stage / via a stack ring | STAGE-SIDE-EFFECT / none |
| attk_b_p4push.s | r13: push on p4 (does not assemble) | PUSH-P2 (+ ASSEMBLE) |
| attk_klda_exq.s | r13: Q held in ex0–7 beside x constants | EX-ALIAS (31 reads) |
| attk_d_dmada.s / attk_e (fixed) | r13: `lda p0, [sp]` co-issued with vldb.4x64 | PORT (DM_ADA) / none |
| attk_i_r0.s / attk r13f (fixed) | r13: value-returning attk read its trip count from r0 | ABI-R0 / none |
| attkb_b3_fa1.s / r2 (fixed) | r14: flag OR read r4 one cycle on, inside the loop body | SCALAR-PLUS1 / none |
| zero2_v1.s (reconstructed) / zero2 | r14: six 2-byte nops after `add.nc lc` | ZOL-SETUP (52 B to le) / none (132 B) |
| nrm_ln_nosave.s (reconstructed) | r15 rule: p6/p7 written without save | ABI-CALLEE-SAVED |
| full12r15m elfs | deployed | 0 errors, 0 warnings (469 calib findings in Peano code) |

Known false positive (warning): STAGE-SIDE-EFFECT on attk/attkb score stores (garbage lands in the ring's slack by design).
Not testable from artefacts: ACQ-EQUAL, IMM-RANGE (the assembler aborts on the latter itself).

## 2. What the r15m elfs teach about the rules (calibration against Peano)
- **Loop setup is measured in BYTES to the loop END.** The bad zero2 (six 2-byte nops) and the deployed one (six full-form
  bundles) both have 10 bundles from the lc write to `le`; only the bytes differ (52 vs 132). Peano writes lc 2 bundles
  (4–16 B) before `ls` when the body is long (ew_ssm, bias_pcs, sqrt_lut_n), so the distance to `ls` is not the rule. Smallest
  deployed distance 80 B (cpy32); Peano's smallest 96 B. Fits instruction fetch running ahead of issue.
- **Peano's loops start with compact bundles** (10/12 B first bundles in bias_pcs, gate_gemm, norm_rows_k, ew_ssm); every
  deployed loop ends with a 16 B bundle. "First bundle in full form" is a convention, not a hardware rule.
- **Peano reads mv-slot results in the bypass cycle** (46 cases in r15m): asmlint flags this only in generated code, where
  msched forbids the window.
- **Peano reads scalar registers while a newer load is in flight** (conveyor reads, e.g. `sel.nez` reading r0 one cycle after
  `lda r0`): legal; flagged only in generated code.
- **alu→alu at +1**: Peano does it 1,258 times in straight-line and branch-loop code; the deployed generated resc and reck do
  alu→mv at +1 inside loop bodies. The one failure (attkb FO→FA) was alu→alu inside a loop body; the rule is kept narrow
  (probe: PROPOSALS microbench 4).
- **Scalar `lda` latency is not in the `.td`** (no itinerary); Peano consumes its result 4–6 cycles later (pointer users 4–5).

## 3. S2 re-priced with the real bank map
GEMM core memory (r15m, 0x70000–0x7FFFF):
- **Bank 0:** stack 0x70000–0x709FF; sgelu_ab, serf_ab, aexp_ab (0x70A00–0x717FF); ks_tab; acc_hi from 0x718C0.
- **Bank 1:** acc_hi to 0x778C0, globals, tbk/tbt/tbc, a00_0 from 0x77B80 (into bank 2). 192 B free (0x77AC0).
- **Bank 2:** a00_1, a00_2, the C object from 0x78900.
- **Bank 3:** C tail to 0x7D100 (4,352 B); B fifo 2 × 4,608 (0x7D100–0x7F500); tbn/tba; `.aie.bank3` = sgelu_cd + serf_cd
  (0x7F580–0x7FE80). 384 B free, plus 384 B in the K object's tail beyond the −m rows.
- **Stack:** deepest path 832 B + 128 margin of 2,560 (stackaudit r15m); attention path main 128 + attn_chunk_k 384 + k_rb 192
  + attkb 64. 1,728 B free.

Options for the 768 B: (a) split 384 K tail + 384 bank 3 with a 2D jump (fragile, the K tail moves with the object);
(b) move serf_cd next to serf_ab (+8 cycles per 32 lanes on every erf GELU, r9); (c) **chosen: 768 B on attkb's own stack
frame** (bank 0). Bank 0 also takes aexp_ab's 16 LUT loads per iteration (and Q if q_buf sits there): ≤ 26 of 34 cycles, a CP
constraint, not a stall.

sim18 (P60g + measured S1): S2 at II 35 −0.33 / −0.38 / −0.44 ms (κ 0.5 / 0.6 / 0.7); II 36 with T 100 −0.29; II 34 −0.45.
Then S3 −0.06, V1 −0.21, K1 −0.27, K2 −0.19: attention stack −1.11 (−1.04..−1.17). No slow path at all: −0.93.

## 4. The output-ring "store stall": refuted by a static whole-call count
Round 15's claim: stores into the EW output ring (bank 0, read by MM2S) cost ~3.5 cycles per 64 B vector in situ vs 1 bench.
Static count of one `out_copy(ring, off, src, 16)` in r15b: out_copy 20 bundles to the `jl` + 5 delay slots; cpy64 3 setup +
7 full-form prologue + 4 trips × 4 + ret + 5 = 32; back in out_copy ge + jnz + 5, 7 restores + ret + 5 = 20. **Total 77 cycles
for 16 stores.** The x-parts phase = 3 calls + loop glue (P[] from the stack) ≈ 290 = **the measured 289**. The h copy (32
vectors, one call, sometimes a wrap split) 93–170 + glue = the measured 217. No room for a store stall: the 3.5 cycles per
vector was ~60 cycles of fixed call cost divided by 16 vectors.
Mechanisms checked: DMA arbitration (MM2S reads at 4 B/cycle: ≤ one bank access per 8–16 cycles, ≤ 0.1–0.2 cycles per stored
vector); the stack in bank 0 (cpy64 touches no stack); lock timing (out_wait 31 per chunk). NRMK's +0.6 ms with copies was the
same per-call cost (2 out_copy calls per unit). Consequence: fewer calls (X1, X1b, P-IN, G1), not other banks. Microbench 2 if
doubted (predicted equal within 0.2 cycles per vector).

## 5. EW re-rank (static counts of r15m per SSM chunk; 1 cycle = 1.43 µs per step)

| pass | in situ (r15b PROF) | static now (r15m) | bound / target | lever | ms |
|---|---|---|---|---|---|
| gates: 8 xc pushes + 2 × gate_gemm | 652 | 2 × (24 + 4 × 57 + 12) + ~60 ≈ 590 | vec 20 per np: 8 × 20 + ~90 | G1 | −0.50 ± 0.07 |
| sqrt_lut_n | 493 | 54 × 8 + 80 = 512 (mv 26 per trip) | II 13, U 2 | SQK (round 16) | −0.31..−0.35 |
| sigmoid (sigk, gx + ga) | 509 | 16 × 24 + ~60 + call | ResMII 8 per vector, II 12 | FSX (ga into ssx) | −0.24 ± 0.09 |
| ssx (log_a, exp, 1 − a²) | 509 | 16 × 20 + 120 prologue + ~50 | vec 20 | in FSX (fewer stages −0.08) | |
| x-part copies | 289 | 3 × 77 + glue | 1 call | X1 / X1b | −0.28 / −0.18 |
| in_part × 7 (PH 0) | ~245 | 7 × ~25–35 | inline | P-IN | −0.25 ± 0.06 |
| conv (convk) / recurrence (reck) | 1,125 / 902 before | 54 + 8 × 20 / 81 + 4 × 44 | at their II | done | |
| in_wait | 165 | | input port | stays | |

Norm ops (~58 per step, ~6.0 ms after NRMK). Per LN row block (18 per core): pass 1 ~2,690 (a 23-bundle loop with 15 alu ops of
ring arithmetic per cb), stats ~900, pass 2 ~3,240 ⇒ ~68 µs compute per op vs ~140 measured (RMS ~56 vs ~100). The gap is I/O
and waits: input 256 × n_in cycles per unit, output 1.6 KB per unit = 400 cycles, 9 KB rings. N1 (pass 1 + stats fused in asm,
~10 cycles per cb) removes 17 µs of compute per op: −0.6 ± 0.25 ms depending on the port-bound share (an EW PROF decides).

## 6. Lockstep, FlashAttention, Q1: unchanged from rev 10 (§2–§5 there).

## 7. Program memory (r15m)
- **GEMM** 16,192 / 16,384 (192 free): S2 needs ~40 B; S3 frees ~400 (k_rb); K1/K2 as rev 10.
- **EW** 14,960 / 16,384 (1,424 free): G1 −480 B C++ for ~400 B asm; N1 −500 B C++ for ~350 B asm; P-IN saves ~200; X1b ~0.

## 8. Ceiling (deltas applied to the measured 41.52)

| added | ms |
|---|---|
| G1 | 41.02 |
| attention stack (S2, S3, V1, K1, K2) | 39.91 |
| N1, SQK, X1, P-IN, FSX | 38.21 |
| P55, P35 + P24b, P51, P57 (rev 8/9 prices) | 35.41 |
| k-step 2,055 → 1,600 | **31.06** |

Depth scaling from the r15m sweep (41.52 / 35.03 / 27.86 / 20.96): ~3.43 ms per block + ~0.4; depth 10 ≈ 25.9, depth 8 ≈ 20.8.
