# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 18, 2026-10-11, research pass 18. State: full12r21k1 (31.30 ms sweep at d12, quiet, MEASURED) + 3A (round 22, -0.08) ~= 31.22;
KTL in main's round 22. Data: tl_full12r21k1_d12.json (12 blocks, r21/tlops21.py), TPROF build/r21/tp_r21t2.npz (r20/tpgap.py), the
deployed EW core (r23/full12r21k1_0_5.dis, tcall), the EW C++ compiled with the deployed flags (r23/ewsize.sh), r21 ddrbudget. Rev 17 in
rev17/.

## 0. Scorecard
- Rev 17: E7'' R 68 kept (-0.28 at 1.405 GHz); R 64 neutral as the replay said; 3A re-priced -0.07..-0.10, built -0.08 (hit); K2 -0.02
  (miss: chunk 0 hidden). KTL: replay-verified kernel delivered, main integrating.
- This pass prices with static counts only (tcall, compiled sizes); no device runs.

## 1. The ledger (ms per step, r21k1)
| part | now | floor | bound |
|---|---|---|---|
| GEMM k-steps | 12.74 | 11.0-11.9 | A port 1,304 / k-step + ~150 DMA collisions (A writes 93, B writes ~80; no idle bank: capacity) |
| epilogues + hands | ~1.8 | ~1.7 | GELU LUT (mlp0 gelu_pan 5,495 / tile), CSUB drain collisions |
| GEMM-side boundaries | ~2.4 | ~1.0 | 5 GEMM<->EW sandwiches ~83 us / block over the EW spans (TPROF k0 - previous GEMM end - EW span - first-unit time: up_a 19, qkv 16, mlp0 22, out 32 (yx->SSM->out), norm0 ~19), first-tile paces 0.25 ms (out_proj 6.7 us / block), attention edges |
| attention | 6.13 | ~5.9 | compute (pass 0 ~73 us, pass 1 ~40 us per head group) |
| norms | 4.52 | ~4.35 | DDR read cap (3-input norm0 / norm2 801 / 821 cycles per unit; norm1 2-input RMS at ~52 GB/s); norm3 (LN 2-in, 693 / unit) compute |
| SSM | 3.59 | 2.86 | compute: ~2,340 / unit in situ (static ~2,115 incl. sigk's trips; glue 740-757) vs DDR floor 2,010 / unit (13.6 MB / block at 57 GB/s) |
| total | 31.2 | ~27.0 +- 0.7 | |

## 2. GEMM k-step
- After KTL (in situ ~1,460) the k-step is ~10 % above the A-port floor; R 66 with KTL adds -0.15..-0.3. The rest is DMA collisions:
  A writes into banks 1 / 2 (busy ~0.4) ~93, B into bank 3 (0.9) ~70 + bank 0 ~12 per k-step. Moving either needs an idle bank, which
  the GEMM core's L1 (~576 B free) cannot give (DEAD_ENDS A13, A24, A25, J6).

## 3. Boundaries
- GEMM<->GEMM boundaries are mostly gone (IL, DAPRE, 3A, AQPRE, AKVPRE). What remains are the EW sandwiches: producer tail (its last
  tile written at the write cap, ~12 us) + the EW op's whole-row dependency + the consumer's k-outer first tile (k-step 0 needs every
  row's unit q = 0).
- MSPLIT (new): row-half tiles at those boundaries. Producer side: the last tile as rows 0-9 / 10-17 with separate drain groups, the
  EW op starts on rows 0-9: saves half a tile + half the tail (16.7 us K 1536, 11.5 us K 768): 68 us / block raw. Consumer side: the
  first tile on rows 0-9 as the norm passes row 9: 27 us / block raw. Separating probe NTWIN (norm0 / norm2 skip the last drain group;
  per-op walls).
- out_proj (dependent on attention): OPPRE-W, rev 17.

## 4. EW ops
- Norms: at the DDR read cap except norm3; fewer bytes would move them, but every byte lever found is blocked (RADD / KSUM: no third
  GEMM input port; EW-PRESUM: no DDR headroom in the second K half; KCONCAT: 497 KB A slot; DEAD_ENDS J1-J3). norm3's LN glue <= -0.18.
- SSM: compute-bound by glue (tcall): convk 228 (68 glue), ggk 282 (90), sigk (45), ssq 674 (194), reck 268 (92), zk 74 (34), in_need /
  in_done ~60, C++ ~180. ssq2 / FSX / reck2 / ggk2 can land until the DDR floor (2,010 / unit): -0.25..-0.45 (cap -0.47). YRES / SSPM
  (DDR byte cuts) pay only after that (J4).
- EW text: TXT18 frees 400 B of the 16,272 (r23/txt18.out), enough for ssq2 + FSX.

## 5. Attention
- ~6.1 ms; S3 -0.05..-0.1 after K2's miss; no structural lever priced this pass.

## 6. Ranking and ceiling (base 31.22)
| rank | item | price |
|---|---|---|
| 1 | KTL | -0.50..-0.70 |
| 2 | TXT18 | 0 (text) |
| 3 | SSM glue | -0.25..-0.45 |
| 4 | MSPLIT-P (NTWIN first) | -0.35..-0.6 |
| 5 | E7'' + KTL R 66 | -0.15..-0.3 |
| 6 | MSPLIT-C | -0.15..-0.3 |
| 7 | norm3 LN | -0.1..-0.18 |
| 8 | OPPRE-W -> lag | -0.04..-0.10 |
Ceiling ~29.2 (EXTRAPOLATED); floor ~27.0 +- 0.7. Between them, unpriced: k-step collisions ~0.9 (capacity), boundary residue
after MSPLIT ~0.6, SSM glue beyond four kernels ~0.4, attention ~0.2.
