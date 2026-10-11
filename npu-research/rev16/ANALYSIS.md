# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 16, 2026-10-11, research pass 16. State: full12r19z (33.88 ms sweep, depth 12, Turbo, gpu_busy 0, BIT-IDENTICAL).
Data: timelines r18m / r19d (3 runs each), main's round-20 TPROF dump of r19z (build/r20/tp_r20t2.npz) and CPST probe
(tp_r20c4, c4k1, c4k4), the deployed full12r19z.bin. Tools: r21/ (tlhist21, bexcess, ddrbudget, cpsig, cpdump, bduse,
bndmodel). Revision 15 in rev15/ (PROPOSALS); text from research pass 16's hand-back, written by the coordinating session.

## 0. Scorecard
- SEQPRE (rev 15 rank 1, -1.4 +- 0.5): miss (best -0.04). CPRATE: the CP is real but cheap (write32 0.06, maskpoll 0.11 us;
  BD write + patch ~0.10 us INFERRED from CPST). The rev-15 boundary model fitted the totals with a free t_op that absorbed the
  12 us producer tail; it was not causal. Lesson kept: name the probe that separates a hypothesis before ranking it (main's
  CPST, a CP-written flag stamped by a core, was that probe).
- DDRCAP refuted rev 15's ">= 84 GB/s": aggregate read cap ~55-59 GB/s; NIN2 void.
- Hits: PADDA, g1a, CSPLIT, ZHI, EPLR (better), KPREP. Misses: g0a (~0.75 of static), PWRAP's offset (-4608: the 2D step adds
  to the 64 B line step).
- Rev 16 pre-check: the DDR-cap pace model predicts 7.0 / 14.0 us of first-tile pace (K = 768 / 1536, A only); measured (tpgap,
  r20t2) 6.5-8.5 / 13.3. Hit (post hoc: main measured it first; not counted as a pre-registration).

## 1. Where the time goes (r19z; timelines r19d, TPROF r20t2)
GEMM k-steps ~13.0 ms (1,576 cycles each), GEMM epilogues ~1.65, C part-1 waits (hand 1,371 plain / 1,808 gated-first,
gelu_pan, qkv, gelu; floor 676), attention ~6.7, norms ~4.9 (DDR-read-bound), SSM ~3.35, and op boundaries ~3.0-3.3 ms:

| part | ms per step | how known |
|---|---|---|
| first-tile pace (A at the DDR read cap) | 0.87 | MEASURED (tpgap: 72 us per block) |
| producer tail + token wait (~13 us x ~8 dependent boundaries) | ~1.3 | MEASURED once (CPST, mlp0 -> mlp3_a), extrapolated |
| CP issue to the first k-step (~4 us + 0.5-0.85 us per column) | ~0.4 | MEASURED once (CPST) + model |
| norm / SSM boundaries (gap means 8.6-11 us; ~4 us of it is real first-rb work) | ~0.6 | timeline means |

The rev-15 "5 ms" counted each GEMM's first-tile compute inside its gap. Host-poll medians are quantised at ~16.5 us: the norms'
"16.5 us gap every op" is an artefact (means 8.6-11.0, r21/tlhist21.py).

## 2. The op boundary (mlp0 -> mlp3_a, microseconds after mlp0's last epilogue end)

| part | us | source |
|---|---|---|
| tail: last C drain + completion token -> the CP passes the first tct | 12.2 | MEASURED (CPST k = 1: 14.88 after the last k-step end, epilogue 2.71) |
| the 8-tct run | ~1.0 | MEASURED (CPST k = 1 / 4 / 8: 14.88 / 15.37 / 15.84) |
| 8 A fills + sends -> column 0's first k-step end | 4.15 | MEASURED (CPST core 0_2: 19.99) |
| each further column (K = 1536: 5 send pushes x 0.17 us) | +0.85 | MEASURED (median core +2.7, model +3.0) |
| first-tile pace: 24 chunks x 8 columns at ~57 GB/s vs 24 k-steps of compute | 13.3 | MEASURED (tpgap; model 14.0) |

Independent ops (A preloaded during the previous op's last tile) start 1.2-1.5 us after the previous epilogue; down_b / mlp3_b
still pace 5.4-5.7 us (their 2 MB loads only as the previous K = 1536 op frees chunks), down_a 11.4 (stream order: half 0 first,
blocked by up_b).
Tail alternatives: H_WR = the C join's last 55 KB per column through one shim S2MM at ~4 B/cycle (~7.6 us) + core drain + DDR
latency; H_TOK = ~6 us of token / CP latency after a full-speed drain. For H_WR: down_a / mlp3_a spans are 15 us longer than
down_b / mlp3_b, whose final drains do not overlap a 2 MB A preload. DDRW (main) decides: 3.5-4.5 B/cycle per column if H_WR,
>= 7 if H_TOK; DDRW-2ch separates a per-channel from a per-column write limit; CPST-N (CPST_OP = up_a or qkv, k = 1 / 8) the
norm tail and column skew; CPBD (optional) the BD-write cost.

## 3. DDR (r21/ddrbudget.py: plan8 TAPs vs r19d shares)
Step: 914 MB read, 676 MB written. In-span read rates: norm0 / norm2 (3 inputs) ~55 GB/s (the cap), norm1 ~49, norm3 ~44, SSM
48.5 (DDR floor ~20 us per head vs ~23.3 measured), GEMM steady 13-24 (writes 15-28). Weights: ~230 MB per step. APAN junk
tails: 74 MB of writes per step. EW ops' read floor 6.6 ms vs ~8.1 ms of EW share. Write cap unknown (>= 28 GB/s alongside
the read cap, DDRCAP).

## 4. GEMM
- k-step 1,576 in situ (ROT g0a); E7'' (B on both FIFOs) the only k-step lever left: -0.9 +- 0.3 if `vlda.pop.576` on lf1 works
  (POPLF1 probe) and CP-SAT finds R = 64.
- Per tile: hand 1,371 (plain, after EPLR) / 1,808 (gated-first, gelu_pan, qkv, gelu) / 676 (gated second = the floor).
  CSPLIT4-lite (-0.11..-0.17), LATE-C (-0.08..-0.12).
- Boundaries: MLPRR (mlp0 dealt rr, two drain groups, mlp3_a waits on the first; -0.30..-0.42), DAPRE (down_a's half 1
  first; -0.10..-0.14), NRES (norm panels straight into the A slot; -0.6..-1.1). DAPRE and MLPRR need per-half F locks: the
  A_RT loads release one F for both halves.
- Memtile residency for GEMM -> GEMM (MRES) is infeasible on memtiles 0-2 (124 KB park, 112 KB free in two holes); attention
  -> out_proj gathers across columns (P60G).

## 5. Norms and SSM
Norms read at the DDR cap (3 inputs) or near it (2 inputs); only fewer DDR bytes move them (NRES removes the panel writes, -36 %
of norm writes, unpriced). NSEQ (params + the next GEMM's B / slot loads before the sync) -0.08..-0.2. SSM: ssq2 + FSX -0.18
lands under the DDR margin (~3.5 us per head); beyond ~-0.5 ms the SSM hits its DDR floor (~2.9 ms per step).

## 6. Attention
~6.7 ms; S3 / K2 -0.3..-0.4 (K1 done as KPREP). Its boundary (Q fills + 12 K / V fills before the sends) is ~3 us longer than a
GEMM's; ATTORD (sends before the pass-0 B fills, B fills head-major) -0.03..-0.05.

## 7. Ranking and ceiling (from r19z 33.88)

| rank | item | price | gate / pre-registered check |
|---|---|---|---|
| 1 | MLPRR | -0.30..-0.42 | per-half F locks; mlp3_a W 24.0 -> <= 2.5 us, pace 13.3 -> <= 2; CPST passes the wait before mlp0's last tile |
| 2 | NRES | -0.6..-1.1 (norm1 -> up_a first: -0.15..-0.28) | routing EW (c,5) MM2S 1 -> memtile S2MM 5; up_a W 106.5 -> 85-92 us, pace 8.5 -> <= 1 |
| 3 | DAPRE | -0.10..-0.14 | per-half F locks; down_a pace 11.4 -> <= 1.5 us |
| 4 | CSPLIT4-lite | -0.11..-0.17 | acquire-only BD in the lowering; hand 1,808 -> 1,000-1,150 cycles |
| 5 | NSEQ | -0.08..-0.20 | post-norm first-tile k-step 2,787-2,977 -> ~2,650 cycles |
| 6-10 | SSM ssq2 + FSX -0.18; attention S3 / K2 -0.3..-0.4; dual C drain -0.3 or 0 (DDRW-2ch); LATE-C -0.1; E7'' -0.9 (POPLF1) | | |

| added | ms |
|---|---|
| MLPRR, DAPRE, NSEQ, CSPLIT4-lite | 33.1 |
| SSM, attention, LATE-C | 32.5 |
| NRES | 31.6 |
| dual drain (if per channel), E7'' (if feasible) | ~30.4 |

Floor ~28 +- 1 ms (k-steps 12.1-13.0, epilogues ~1.4, attention ~5.5, EW DDR floor 6.6, boundaries ~1). Stacked prices below
33 carry +-1 ms.
