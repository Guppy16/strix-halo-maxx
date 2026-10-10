# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 13, 2026-10-11. State analysed: round 17's deployed `full12r17d` (39.16 ms at depth 12, Turbo, gpu_busy 0, BIT-IDENTICAL
to full12r9a) and main's round-18 `full12r18a` (HRING + PTRG, −0.34 A/B) / `full12r18b` (+ ZK, −0.44 A/B vs r17d: ~38.72).
Disassemblies (`-z`) of r17d (GEMM 0_2, EW 0_5) and r18a (EW) in `r18/`; round-17 NPROF (ewp_nb0 / nb2) and CPROF (ewp_dc) records;
timelines tl_full12r16z / tl_full12r17d. Revision 12 is in `rev12/`. Tools: `tools/ewio.py` (+ fit13 / levers13), `tools/ksplan.py`
(new), `tools/tcall.py`, `tools/textmap.py`. Text from research pass 13's hand-back, written by the coordinating session.

## 0. Scorecard (rev 12 vs round 17)
- S2 −0.31 vs −0.38 (−0.29..−0.45): hit. X1 mod 4 −0.33 vs −0.34 alone: hit.
- AL1 −0.05..−0.15 vs −0.21 ± 0.05: miss. CPROF confirms the straddle is gone (conv = convk 280 + pointers 101); half the static
  price did not appear (unexplained; candidates: fewer straddles in situ, the +98 KB per column of padded params). 1 KB padding hung:
  an EW fill group must be a 768 B slot multiple.
- NPIPE ~−0.1 (G4-6), +0.07 (G2) vs −0.55: **miss from a wrong bound** (section 1).
- NIO stamps: out_begin 45.0 gross (~5 net) vs the pre-registered 147 / 90: the measurement that refuted R_out 2.35.
- FIFO-POP9 on popprobes 2 / 3: true positive (round 16 had checked only lf0).

## 1. The EW I/O refit (ewio fit13, r18/ewio_fit13.out)
Stamps bracket calls: in_need's interval contains ~44 cycles of static call + stamp, out_begin's ~40. Net waits per unit: r16z
in_need ~57.5, out_begin ~5; NPIPE_G=2 ~17 / ~6.6. Model changes: in-situ per-unit costs (pass 1 +17 from the NPROF n1k stamp;
pass 2 = static + s2 x the one-core nrm bench excess 19 / 33), NPIPE groups of G with 45 cycles per group, a per-op fixed latency,
and the r16z timeline walls per kind (RMS n2 83.8, RMS n3 103.1, LN n2 100.6, LN n3 115.4 µs) as targets.
Best fit: R_in 5.75 (5.5-6.0), R_out >= 3.5 (any value to 8 fits equally), s2 = 4 (3-5) = pass 2 +76 (RMS) / +132 (LN) per unit
in situ. Reproduced: in_need 51 / out_begin 0 (net ~57 / ~5), G2 in_need 11 (~17), r15i −0.54 (−0.53), NRMK −0.45 (−0.37..−0.42),
N1K −0.22 (−0.16..−0.28), walls rms 3.9 µs with F ≈ 5 µs. Misses: n3 − n2 wall 10.7 µs vs 15-19 measured; NPIPE G6 −0.21 / G2 −0.13
vs −0.05..−0.15 / +0.07 (NPIPE's own glue is not modelled, ~0.1-0.2 ms).
**Why rev 12's R_out 2.35 passed five checks**: the three deltas (r15i, NRMK, N1K) are compute cuts that realised less than static;
"pass 2 waits for the output port" (R_out 2.35, s2 1) and "pass 2 runs slower in situ than static" (R_out >= 3.5, s2 ~4) predict the
same deltas (−0.48 / −0.42 / −0.18 vs −0.54 / −0.45 / −0.22). Rev 12 fixed s2 at the bench value by assumption, so the fit put the
missing pass-2 time on R_out. The n_in invariance (r15h) and NPROF in_need are insensitive to R_out. Only a stamp on the hypothesised
wait could break the degeneracy; round 17 added it.

## 2. The norms after NPIPE (r17d: 4.74-4.87 ms per step)
Compute-bound. Per step: static compute ~3.7 ms, pass-2 in-situ excess ~0.61, pass-1 in-situ ~0.10, I/O waits ~0.07 (~12 per unit),
per-op fixed ~0.25. Levers (ewio levers13 at the fit, vs r17d's NPIPE G6): FOLD (n_in 3 -> 2 by folding D1 + D2 / D3 + D4 upstream)
−0.03 (−0.05 on the serial order): retired. NPIPE G2 +0.08, G12 +0.21: keep G6. NKEEP (n1k state kept across nrm) ~0. Compute:
NG1 (pass-1 glue −80 per unit) −0.44, NG2 (pass-2 glue −100) −0.56, NS (stats in asm −700 per rb) −0.34, together −1.22. Per unit
today: pass 1 = ~117 cycles of glue (in_need 41, ring pointers ~36, in_done 40) around a 140-cycle n1k call; pass 2 ~170 of glue
(out_begin 37, pointers / wrap ~30, nrm call glue 45 / 73, out_end 38, loop) around nrm. N-IO1 / N-IO2 (the glue into the asm steps)
are the norm items. Where the fitted pass-2 excess sits is unmeasured: NIO2 stamps (nrm call, whole pass-2 unit, whole pass-1 unit).

## 3. The SSM after X1, HRING, PTRG, ZK
Output 3 KB per chunk at >= 3.5 B/cycle (<= 880 cycles) and input 7 KB at 5.75 (1,247) are far under the chunk: compute-bound, every
cycle 1:1 (1 cycle per chunk = 1.43 µs per step; main's HRING + PTRG measured −0.34 for a static −0.17..−0.32, ZK −0.17 for −0.18).
r18b chunk static ~1,854 + sigk's 8 trips: ssq 680 (glue 194), sigk 371 (glue 45), ggk 288 (glue 90) + the xc -> bfp push, reck 274
(glue 92), convk 226 (glue 60), z ~100, in_need + in_done 81. In situ ~+250-350 per chunk above static (r17d timeline). Levers: CGF
(convk pushes bfp panels: −70..−90 per chunk, −0.10..−0.13), ssq on two chunks per call (−75, −0.11), FSX (−50, −0.07), params
in_copy by DMA (−25, −0.04).

## 4. GEMM program memory (r17d 16,336 / 16,384)
textmap: main 2,080 (204 nop bundles = delay slots + 16 B branch-target alignment), mm_ks 1,824, acc_epilogue 1,456, attn_chunk_k
1,344, k_rb 944, attkb 768, attn_end 752, bias_pcs 752, bgelu 688, attn_begin 608, mm_kstep 320, ... Candidates: KSC −270..−290
(mm_kstep's once-per-run record builder + per-call cache check; all 24 cores share C 0x78900 and A slots 0x77B80 / 0x78000 /
0x78480), IV32 −200..−300 (12 i64 loop counters in IRON's lowered control code), P35 −80..−120 (main's k-loop moves into mm_ks),
END_MIN (measure), EXIT −64..−120 (main's dead restore block; needs a true while(1) worker).

## 5. The GEMM k-step (2,122 = 2,055 in mm_ks + 67 between calls)
Static (ksplan --dis): per 32 MACs 32 MAC bundles + 12.75 swap (C store / reload with all 4 accumulators finishing together; MACs
stop) + 3 ldb-only (36 B ops per 32 MACs on ldb) = 47.75; 9 trips x 4 groups + glue 96 = 1,815 (tcall 1,824). The swap is 459 cycles
per k-step, the largest static item; dm4 is unused. In situ +231: not the bank rule (24 modelled: consecutive groups alternate pieces
in different banks), not the A port (needs 5.0 B/cycle; memtile -> core 7.95). UNVERIFIED: DMA writes colliding with core accesses
(B prefetch 144 writes x bank-3 busy 0.75 + A 324 x ~0.35 = 221). Between calls 67 (static 51: mm_kstep 26 + main's loop 25).
Bounds without the swap, per 8 MACs: vec 8, bank 3 (B pops + fill) 9, load slots 7.1, st 4; per k-step A port 1,304, B 580.
ROT (5th accumulator rotating through the swap, jobs of 8 MACs, 2-3 chains): II 9 per job (bank-3 bound), static ~1,356; registers
ex 8 A + 4 B = 12, dm <= 5; C load (job j+2) and store (job j-2) always in opposite pieces = different banks. Through the collision
model: k-step ~1,720 (−3.5 ms), with P35 ~1,670 (−3.9). Then: layout (B prefetch outside the popped bank −101, A ring away from C
pieces −120: −1.9 ms if the core yields), B over two banks (−144: −1.2 ms), A port floor 1,304. Order: E1 CP feasibility of ROT
(offline), E2 DMAB bench, E3 KS-P38 / A-only twin, E4 P35, E5 ROT, E6 layout, E7 B split. 1 cycle per k-step = 8.59 µs per step.

## 6. Attention
Unchanged from rev 12 except S2 (done, −0.31). S3 −0.06, V1 −0.21, K1 −0.27, K2 −0.19 are rev 10/11 prices on r15m (±30 %; re-price
with sim18 on r17d); they wait on GEMM text (KSC).

## 7. Ceiling (from r18b ~38.72)

| added | ms |
|---|---|
| KSC + S3 / V1 / K1 / K2 | 37.99 |
| P35 | 37.55 |
| N-IO1 + N-IO2 | 36.55 |
| NS, CGF | 36.1 |
| DMAB layout (EW) | ~35.8 |
| P55, P51, P57 (rev 8-10 prices) | ~34.0 |
| ROT (k-step 2,122 -> ~1,670) | **~30.5** |
| B over two banks | ~28.9 (then the A port binds) |

Depth 10 / 8 scale by the r17d sweep ratios 32.78 / 39.16 and 26.29 / 39.16.
