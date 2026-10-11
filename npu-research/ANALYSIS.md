# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 15, 2026-10-11, research pass 15. State: full12r18m (35.20 ms sweep, depth 12, Turbo, gpu_busy 0, BIT-IDENTICAL) and
main's round-19 builds r19b (+ ROT g0a), r19c (+ ZHI), r19d (+ CSPLIT), r19 EPLR (≈ 34.5 by A/B). Data: timelines r18m / r19d,
TPROF, KS-P38, NIO2, the deployed full12r19b .bin / elfs. Tools: r20/ (gen_rot, collide, tilefit, cpseq, rotdb), asmlint rev 15.
Revisions 13 / 14 in rev13/, rev14/ (PROPOSALS) and §5.1 of rev 14. Text from research pass 15's hand-back, written by the
coordinating session.

## 0. Scorecard
- ROT (rev 14, g2f −3.84 ± 0.4): g2f raced in situ (acq.cond 1 bundle before the A fill); as g2a −3.27 (slight miss). KS-P38 in
  situ: TRUE 1,603 (pre-reg 1,610 ± 45, hit), MEMORY 167.5 (212 ± 40, low edge), LOCK 13.4 (13, hit).
- g0f / g1f (rev 14: −0.28 / −0.14 vs g2f): main's g0a −0.18 / g1a −0.11: k-step static cuts land at ~0.75.
- P35 −0.10 vs −0.44 (miss); CGF −0.06 vs −0.10..−0.13 (miss vs research, hit vs tcall); NIOC −0.20 vs static −0.55: the norms'
  pass 1 is input-bound at ONE shim channel (§4), which rev 13's fitted R_in 5.75 hid.
- TPROF: the pass-15 draft pre-registered hand / epi after main had measured them; not counted.

## 1. Where the time goes (r18m timeline, main; tilefit)
attention 6.7 (gap 1.2), norms 5.0 (gaps 0.8), SSM 3.5, GEMM 19.9 = k-steps 13.4–13.7 + per-op overhead ~6.3:
dependent start-up ~3.3 (33–38 µs per op whose A is the previous op's output; up_a ~50; ops with A ready early: none), per tile
~2.2 (TPROF: epilogues ~1.65 ms per step; hand 3,010 = C-drain wait 2,330 + zero / loop; since then ZHI −0.15, CSPLIT −0.25,
EPLR −0.12), residual ~0.8. Op boundaries ≈ 5 ms per step in all.

## 2. Op boundaries: the command processor (hypothesis, unmeasured)
The .bin has ~51k CP instructions per dispatch in 289 sync runs. After each sync the CP executes 92–144 instructions before the
dependent op's A path is armed (shim A fills → memtile slot loads → memtile sends); drains, memtile tasks and, after norms (whose
group is finished at once), even the B fills sit behind it. t_op ≈ 0.17 µs (0.13–0.22) + the A-bound first tile + ~3 µs of DMA
latency reproduces yx / out / qkv / out_proj / mlp0 / mlp3_a and the norms within ~5 µs (up_a +20). SEQPRE issues all work that
does not read the producer's output before the wait (36–40 instructions left after it): −1.37..−2.31 ms (cpseq), ~−1.4 realised.
CPRATE (N dummy writes at one boundary) measures t_op first.

## 3. GEMM
- k-step (TPROF): 1,576 in situ = mm_rot g0a 1,357 static + DMA collisions ~168 + LOCK ~13 + glue. Time-resolved collision
  model: α 0.72 (= DMAB) reproduces 167.5; A 75, B 92. B's is structural (popped bank busy 0.97); B objects in two banks are
  infeasible (23 KB of k-step-cold data needed, 14.6 KB exist); the best feasible A-slot layout saves ≤ 15 cycles: dropped.
  ROT's 72 + g is the single B FIFO's ldb slot and bank together: E7' void; E7'' (B on both FIFOs, A merged into lf0, lines
  woven over two banks) would reach R = 64 (−0.9 ms), feasibility open (R2).
- Per tile: the accumulators cannot leave the C object (+12,288 B needed, 576 B free). After CSPLIT / EPLR: CSPLIT4 (rb-major
  gated / gelu_pan / qkv epilogues + part 1 in 4 sub-parts) −0.3..−0.35; LATE-C (C holds pair 0 of rb 6–17, zero-load first
  k-step, C acquire inside mm_rot at rb 6) −0.3 more.
- down_a, the one non-dependent op still paying (0.52 vs down_b 0.26): DAPRE −0.2..−0.3 if a K = 768 A needs only one half of
  the A slot.

## 4. Norms (r19d walls, main)
Per column: norm0 95.6 / norm1 67.7 / norm2 98.0 / norm3 82.7 µs; per unit 801 / 567 / 821 / 693 cycles. Same compute for 2 and
3 inputs → 3-input ops are input-bound at 3.8 B/cycle = one shim MM2S channel (Ein shares MM2S 0 with Ain). Model (216 units per
op, compute from the n2 walls): today 4.21 ms + gaps; N-IO-ASM alone −0.26 (2-input ops hit 539 cycles of input), NIN2 alone
(second input stream) −0.52, both −1.57; with a 100 GB/s DDR cap −0.68. NIN2: shim MM2S 1 is free on columns 3–7 and is B's on
0–2 (B becomes a hand-built packet ring), EW S2MM 1 is free. DDRCAP measures the ceiling (≥ 84 GB/s in + out today). SEQPRE takes
~0.3 of the 0.8 ms of norm gaps.

## 5. SSM: unchanged (3.5 ms; ssq two chunks −0.11, FSX −0.07, rev 13).

## 6. Attention: 6.7 ms incl. a 1.2 ms gap (SEQPRE part); S3 / K1 / K2 ~−0.5 at rev 10/11 prices.

## 7. Ranking and ceiling (from r19d ≈ 34.6)

| rank | item | price | gate |
|---|---|---|---|
| 0 | CPRATE probe (400 dummy CP writes after mlp3_a's sync) | +0.62 / +0.82 / +1.06 ms at t_op 0.13 / 0.17 / 0.22 µs | none |
| 0' | DDRCAP probe (aggregate DDR ceiling, 2 shim channels per column) | — | none |
| 1 | SEQPRE | −1.4 ± 0.5 | CPRATE; bit-identical |
| 2 | NIN2 + N-IO-ASM | −0.7..−1.5 | DDRCAP |
| 3 | CSPLIT4 | −0.3..−0.35 | — |
| 4 | attention S3 / K1 / K2 | ~−0.5 | — |
| 5 | DAPRE | −0.2..−0.3 | — |

| added | ms |
|---|---|
| SEQPRE | 33.2 |
| NIN2 + N-IO-ASM | 32.2 |
| CSPLIT4 | 31.8 |
| attention S3 / K1 / K2 | 31.3 |
| DAPRE + SSM | 30.9 |
| LATE-C | ~30.5 |
| E7'' + DDR-free dependent A | ~28.6 |

Stacked prices below 33 carry ±1 ms.
