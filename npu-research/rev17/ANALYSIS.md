# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 17, 2026-10-11, research pass 17. State: full12r21k1 (31.30 ms sweep at d12, quiet box, MEASURED by main; -0.70 vs r20z in
one process; BIT-IDENTICAL) + 3A (round 22, -0.08). Data: the deployed GEMM-core disassembly (r22/full12r20z_0_2.dis, tcall),
main's round-20/21/22 results (RESULTS_FROM_MAIN, NOTES 25-26, RESULTS 29-30), r21/ddrbudget, main's r21/mtmap.py, the r22/ CP and
replay runs. Revision 16 is in rev16/.

## 0. Scorecard
- Rev 16 prices: MLPRR (-0.40), DAPRE (-0.13), NSEQ (-0.17), NRES (-0.84 with all norms), CSPLIT4-lite (= CSUB -0.13) hit; SENDPRE
  (+0.16) dead as rev 16 had retired it; OPPRE (+0.18, main's) dead; DDRW refuted both tail hypotheses (aggregate write cap).
- Rev 17 (this pass, consumed live by main): E7'' built from gen_e7 at R 68: -0.28 at 1.405 GHz (~-0.22..-0.30 at 1.81). My
  standalone static price was -19 cycles per k-step (~-0.12 ms): the in-situ gain was 1.4-1.9x the static one [INF: the B DMA writes
  left the busiest bank]. R 64 neutral, as the replay said (1,244 + 129 stalls vs 1,338). 3A re-priced -0.07..-0.10: -0.08 (hit).
  K2 (rev 16's S3 / K2 -0.3..-0.4) measured -0.02: miss (chunk 0 hidden behind the K / V stream start).
- Corrected in this pass: the first rev-17 draft priced KTL on ROT; one row block's pops span ~85 cycles > R, so a single-FIFO
  loop across k-steps is impossible (DEAD_ENDS A28). KTL is built on E7'' with segment-aligned streams.

## 1. Where the time goes (r21k1, ms per step)
| part | ms | how known |
|---|---|---|
| GEMM k-steps | ~12.6 | in-situ k-step ~1,545 (1,576 - E7's ~30) x 14,930 |
| of which per-call overhead | ~1.0 | 114 (E7 R 68 call minus body) + 13 (mm_ktile loop) cycles per k-step: DOC tcall |
| GEMM epilogues | ~1.6 | CSUB moved drain collisions into them (+219 / +561 cycles per plain / gelu_pan tile) |
| C hands | ~0.2 | 345 / 668 cycles (TPROF r21t2) |
| attention | ~6.5 | timeline; K2 / K0B small |
| norms | ~4.6 | DDR read cap |
| SSM | ~3.6 | DDR-near |
| first-tile paces | ~0.25 -> ~0.17 with 3A | tpgap |
| GEMM -> norm tails | ~0.58 | 4 x ~12 us per block at the write cap |
| other boundary (CP issue, EW) | ~1.0 | timeline means |

## 2. The k-step after E7''
- Static: 18 R + call overhead. E7 R 68 call 1,338 = 1,224 + 114; mm_ktile glue 13. The A-port floor is 1,304 per k-step (10,368 B
  at 7.95 B/cycle), so a lower R pays only while the in-situ k-step (static + ~190 collisions / locks) stays above ~1,300-1,450.
- Bank 3 is E7'''s bound: with a 576 B X segment it takes 56 panels + 7 fills = 63 line reads per row block. R 64 had no conflict-free
  schedule (CP: the bank-3 group alone makes it hard; dropping it is feasible in 300 s); R 68 is conflict-free.
- Residue matching (a stream instance may serve two row blocks) is what makes R 64-68 schedulable; the absolute form (each instance
  one row block) is infeasible at R 64 (r22/diag64_absolute.out).

## 3. KTL (one loop per tile)
- The call overhead (127 cycles per k-step) goes; a tile pays 76 once. mm_e7k_r70: 76 + 1,260 x nks, replay-verified for 1-25 k-steps.
- Requirement: consecutive k-steps read different B buffers, so every stream instance must serve one row block (gen_e7 --aligned,
  lf1 split L | N); feasible at R 70 (not found at R 68 in 5 min).
- Price: -83..-87 static per k-step vs today -> -0.50..-0.70 ms (0.75-1.0 landing). After it the in-situ k-step (~1,460) is ~12 %
  above the A-port floor: little k-step headroom remains (R 66-68 with KTL: -0.15..-0.3).

## 4. Boundaries
- Paces: out_proj 6.7 (dependent: OPPRE-lag only), down_b / mlp3_b handled by 3A (-0.08), mlp0 1.9, up_a 1.5 us per block.
- Tails: the producer's last tile at the aggregate write cap; every row's last 64 columns are in it. No lever above ~0.1 ms
  (NTAIL-PF <= 0.07, NTAIL-M -0.1..-0.25 for L).
- OPPRE: pass 1 is ~23 GB/s with the A load (cap ~55): not DDR. H_CP (waits in the in-order CP ahead of K / V pushes) vs H_SHIM;
  OPPRE-W decides.

## 5. Norms, SSM, attention
- Unchanged in kind: norms at the DDR read cap (only fewer bytes move them), SSM near it (ssq2 + FSX -0.18 needs EW text),
  attention ~6.5 with S3 re-priced to -0.05..-0.10 after K2's -0.02.

## 6. C waits
- CSUB + ZLOAD done (hands 345 / 668); LATE-C hangs (a core waiting on its own drain while holding multicast inputs); RTC not worth
  its CP pushes. Residual ~0.2 ms.

## 7. Ranking and ceiling (from r21k1 31.30 + 3A = 31.22)
| rank | item | price | gate |
|---|---|---|---|
| 1 | KTL (mm_e7k_r70) | -0.50..-0.70 | in situ bit-identical; TPROF steady k-step -60..-85 |
| 2 | E7'' + KTL at R 66-68 | -0.15..-0.3 | replay 0 stalls; A-port floor |
| 3 | SSM ssq2 + FSX | -0.18 | EW text |
| 4 | S3 | -0.05..-0.10 | text; flagged non-chunk-0 units |
| 5 | OPPRE-W -> OPPRE-lag | -0.04..-0.10 | H_CP |
Ceiling ~30.1 ms (EXTRAPOLATED); floor 27.5 +- 1 (k-steps at the A-port floor ~11.5, epilogues 1.6, attention ~6.3, EW DDR floor
6.6, boundaries ~0.75). Stacked prices below 31 carry +-1 ms.
