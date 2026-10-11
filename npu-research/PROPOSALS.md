# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 18 is archived in `rev18/`. This is revision 19, 2026-10-11, research pass 19, written during main's round 23 (MSPLIT-P and the SSM glue).

- **Base:** full12r22z, 30.78 ms at d12 (quiet sweep, MEASURED).
- **This pass:**
  - Found the KTL hang (MEASURED).
  - Delivered a fixed KTL kernel that passes a one-core bench with real locks and DMA-fed A / B.
  - Priced attvs II and S3 (both not worth it).
- **NPU use (one core, between main's A/Bs):**
  - MULLAT and RELC probes;
  - the lock-free KTLP-NL bench (5 dispatches);
  - the DMA-fed KTLP-D (1 dispatch);
  - 1 timeout: my KTLP-NL build had a stale record header (DEAD_ENDS K3). The device recovered; the known-good probe passed after it.
- **CP-SAT:** one 900 s KTL re-solve at 4 workers (nice).

## Headlines (rev 19)

1. **KTL's hang: acq.cond / rel.cond test bit 0 of r26.**
   - RELC (r24/relcb.py), MEASURED: r26 = 2 → no release, 3 → release.
   - KTL masked its acquires as flag × row-block counter, so every A acquire at an even counter was skipped. With an even trip count
     (KTL1's 18) that was all nine, while every release ran: the lock counts drifted and the step hung.
   - The replay modelled "non-zero" and passed. It now rejects any condition other than 0 / 1.
2. **The fixed kernel `r24/mm_e7k_r70b.s / .h`:**
   - Masks are booleans (nez + and). The schedule was re-solved at R 70; fitted, asmlint 0 / 0.
   - The replay is bit-correct for 1–24 k-step tiles at acc_hi 0x719C0. 1,536 B of text.
   - **KTLP-NL** (lock-free twin: each lock op becomes an add of its condition):
     - 9 / 9 A and nks − 1 / nks B lock ops per tile;
     - pieces bit-identical to E7 R 68 over 1 and 3 k-steps, both as one call per tile and as one call per k-step.
   - **KTLP-D** (real locks, A and B fed by shim DMAs): one call of 3 k-steps completes, 36 / 36 pieces equal to the reference.
   - The record order changed: main must take the new .h. Signature and caller are unchanged.
3. **Two refuted suspects:**
   - Scalar mul / add latency in a ZOL body is as the .td says (MULLAT).
   - lda.2d was already cleared by main.
4. **Not worth building:**
   - attvs II 18 → 11 is ≤ −0.02 ms: 6 iterations per call, so the loop pays (6 + S − 1) × II.
   - S3 on the new attkb / attv is −0.01..−0.04 ms. K2's −0.02 already said chunk 0 is hidden.

## Ranking (rev 19; base full12r22z 30.78)

| rank | id | what | cost | price | gate / check |
|---|---|---|---|---|---|
| 1 | **KTL** (r24/mm_e7k_r70b) | one loop per tile on the E7 kernel | S (main's integration exists: swap the .s / .h) | **−0.50..−0.70** | BIT-IDENTICAL; TPROF steady k-step −60..−85 cycles. The kernel passed KTLP-NL and KTLP-D |
| 2 | MSPLIT-P (main, round 23) | rev 18 | L | −0.35..−0.6 (NTWIN bound −0.92) | rev 18 |
| 3 | SSM glue (main, round 23) | rev 18 | M | −0.25..−0.45 | tcall per unit ≤ 1,880 |
| 4 | E7'' + KTL at R 66–68 | aligned + KTL with boolean masks at lower R (`gen_e7.py --ktl --R 68 --hint r24/e7k_r70b_n16.json`) | S (CP) | −0.15..−0.3 | replay 0 stalls; KTLP-NL counts |
| 5 | MSPLIT-C | rev 18 | L | −0.15..−0.3 | after MSPLIT-P |
| 6 | norm3 LN glue | rev 18 | M | −0.1..−0.18 | NIO2 stamps |
| 7 | OPPRE-W → lag | rev 17 | S | −0.04..−0.10 | H_CP |
| — | attvs II, S3 | — | — | ≤ −0.02 / −0.01..−0.04 | not proposed (DEAD_ENDS K4, K5) |

**Ceiling:** 30.78 → KTL 30.2 → MSPLIT-P 29.7 → SSM 29.35 → R 66 29.15 → MSPLIT-C 28.95 → norm3 + OPPRE-lag **~28.8** (EXTRAPOLATED).
**Floor ≈ 27.0 ± 0.7** (rev 18).

## KTL integration (main)

- Take `r24/mm_e7k_r70b.s` and `.h` together. mkktl.py's symbol and border handling are unchanged; the record table is the new
  header's `mm_e7k_order` with rec_k() as before.
- mm_ktile: acquire B(0); `mm_e7k(rec, pc − record(0, F_J0), 781, 18 nks, Y(b), X(b))`; no per-k-step loop.
- First dispatch with the 3 s timeout. If it hangs, KTLP-D on the deployed placement is the next step: `r24/ktlpd.py` builds for any
  placement in two passes.

## Benches and probes (rev 19, `r24/`)

| tool | what |
|---|---|
| `mullat_gen.py`, `mullatb.*` | MULLAT: scalar mul / add result visibility to `st` at 1–3 bundles, in a ZOL body and straight (MEASURED: mul 2, add 1) |
| `relc.s / relc.cc / relcb.py` | RELC: rel.cond with r26 = 0 / 2 / 1 / 3 via a DMA-observable release that cannot hang (MEASURED: bit 0) |
| `mkvar.py` | lock-free instrumented twins: lock ops → adds of r26 into r8–r11, logged after the loop (KTL), or into r28 (E7) |
| `ktlpb.py / ktlpb.cc` | KTLP-NL: one core, static A / B, the E7 reference vs KTL per k-step vs KTL per tile; hashes + lock sums (`--kh` = the kernel's own header) |
| `ktlpd.py / ktlpd.cc` | KTLP-D: real locks (ids 0 / 1 / 4 / 5), A and B fed by shim DMAs, two-pass build so the toggles match the bench's buffers |
| `padcheck.py` | asserts no `.p2align` pad between a (modified) kernel's prologue and body |
| `mm_e7k_r70b.*`, `e7k_r70b_n16.json` | the fixed KTL kernel (deployed placement) and its schedule |
| `r22/gen_e7.py`, `r22/gen_e7k.py` | boolean KTL masks (NZA / NZB ops); the replays reject lock conditions other than 0 / 1; `--place` for emit / fit |
