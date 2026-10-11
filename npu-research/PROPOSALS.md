# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 16, 2026-10-11, research pass 16, written during main's round 20.

- **Base:** `full12r19z` (33.88 ms sweep median at depth 12; r18m + ROT g0a + ZHI + CSPLIT + EPLR + KPREP + SEQPRE=1).
- **Data:**
  - main's r18m / r19d timelines (3 runs each);
  - main's round-20 TPROF dump of r19z (`build/r20/tp_r20t2.npz`: per tile, with core timers and tile counters);
  - main's CPST probe (`tp_r20c4*`: a flag the CP writes at mlp3_a's dependency wait, stamped by core 0_2);
  - the deployed `full12r19z.bin`.
- **Archive:** revision 15 is in `rev15/`; older ones are in `rev14/` .. `rev1/`.
- **Prices:** deltas at depth 12, Turbo, 1.81 GHz.
  - "us per block" × 12 = µs per step.
  - k-step static cuts land at ~0.75 (round 19). Removed waits are priced at 0.75–1.0.

## Headlines (rev 16)

1. **The op boundary is three things in series, and the CP instruction count is not one of them.** For a GEMM whose A is the previous op's output (mlp0 → mlp3_a, all µs measured):

   | part | µs | source |
   |---|---|---|
   | producer's last epilogue end → CP passes the first token wait (**tail**) | 12.2 | MEASURED, CPST |
   | the 8-tct run | ~1.0 | MEASURED, CPST k = 1 / 4 / 8 |
   | 8 A fills + the memtile sends up to column c, then chunk 0 and one k-step (**issue**) | 4.2 (column 0); +0.85 per column for K = 1536 (+0.5 for K = 768) | MEASURED, CPST: column 0 at 4.15, median core +2.7 |
   | first tile paced by the A load at the DDR read cap (**pace**) | 13.3 (K = 1536), 6.5–8.5 (K = 768) | MEASURED, tpgap |

   - The cap model predicts the pace for A alone: 14.0 / 7.0 µs. That is a hit.
   - From the CPST numbers, a BD write + patch costs **~0.10 µs** (INFERRED). So hoisting BD writes saves ≤ 0.1 ms per step (BDPRE: dropped).
2. **Budget per step (r19z):**

   | part | ms | how |
   |---|---|---|
   | pace | **0.87** | measured: tpgap sum, 72 µs per block |
   | tail + tct | **~1.3** | ~13 µs × ~8 dependent boundaries per block |
   | issue | ~0.4 | |
   | norm / SSM boundaries | ~0.6 | timeline means |
   | **total** | **~3.0–3.3** | not 5 ms |

   - "5 ms" counted the first tile's compute inside the GEMM gaps.
   - The norms' "16.5 µs gap every op" is a median of host polls quantised at 16.5 µs. The means are 8.6–11.0 µs (`r21/tlhist21.py`).
3. **Main's hypothesis (A loads at the DDR cap) explains the pace: 0.87 ms, ~30 %.** The largest single part is the **tail** (~40 %).
   - Candidates for the tail:
     - **H_WR:** the C drain's last 55 KB per column through one shim S2MM at ~4 B/cycle (32-bit) is ~7.6 µs, plus the core drain and DDR latency.
     - **H_TOK:** the token / CP wake-up latency, ~6 µs on top of a full-speed drain.
   - Evidence for H_WR: down_a's and mlp3_a's spans are 15 µs longer than down_b's and mlp3_b's. Their final drains run while the next op's 2 MB A preload saturates the reads. A fixed token latency would not stretch with DDR load.
   - **The separating probe: DDRW** (main has built it).
4. **DDR traffic per step (`r21/ddrbudget.py`): 914 MB read, 676 MB written.**
   - **DDR-bound now:**
     - every EW op: norms 49–55 GB/s in span, SSM 48.5 GB/s in span;
     - the dependent GEMMs' first tiles.
   - GEMM steady tiles read 13–24 GB/s and write 15–28.
   - The EW ops' read floor is 6.6 ms per step against ~8.1 ms of EW share.
   - 74 MB per step of GEMM writes are the junk tails of APAN objects (gated / gelu_pan).
5. **The DDR-free dependent A path is viable for norm → GEMM only (NRES), with zero extra memtile memory:**
   - The norm's panels can stream straight into the GEMM's A-slot half 0, which is idle while the norm runs.
   - GEMM → GEMM APAN residency (MRES) needs a 124 KB park that memtiles 0–2 do not have (112 KB in two holes).
   - attention → out_proj gathers across columns (P60G).
   - **MLPRR** gets most of MRES's gain for mlp0 → mlp3_a through DDR: rr-deal mlp0 so that mlp3_a's 24 chunks are written in mlp0's first half.

## Ranking (rev 16; base full12r19z 33.88)

| rank | id | what | cost | expected (d12) | separating probe / pre-registered check |
|---|---|---|---|---|---|
| 1 | **MLPRR** | mlp0 dealt rr (P23 for gelu_pan); its drains in two groups (steps 0–7 = M chunks 0–23, steps 8–15); mlp3_a waits on group 1 only; mlp3_a's fills half 1 (chunks 12–23) first; **per-half F locks** (Flo / Fhi) | M | **−0.30..−0.42** (35 µs per block × 0.75–1.0) | tpgap: mlp3_a W 24.0 → ≤ 2.5 µs, pace 13.3 → ≤ 2; CPST_OP=mlp3_a: the CP passes the wait before mlp0's last tile; timeline mlp3_a gap 52 → 10–18 µs; BIT-IDENTICAL |
| 2 | **NRES** | norm → GEMM A on-chip: the norm writes X' (1 KB per unit) to its DDR ring and the panel (576 B) to a second ring; EW MM2S 1 → memtile S2MM 5 → A-slot half 0 laid out rb-major (one contiguous BD, F += 12 at the end); the GEMM's sends read chunk q with stride 6,912; no A fill, no CP wait for the norm | L | **−0.6..−1.1** (all 4 norms; 18–29 µs per boundary) | first norm1 → up_a only: up_a W (TPROF, from out's end) 106.5 → 85–92, pace 8.5 → ≤ 1; A/B −0.15..−0.28 |
| 3 | **DAPRE** | down_a's A as two fills (half 1 = chunks 12–23 first; it loads during up_b, half 1 being free under K = 768) + ld[1] before ld[0] + **per-half F locks** | S–M | **−0.10..−0.14** | tpgap down_a pace 11.4 → ≤ 1.5 µs |
| 4 | **CSPLIT4-lite** | a static sub-part boundary (9,216 B) inside part 1 for gated / gelu_pan (+ qkv): an acquire-only BD 1a, then 1b releases the C prod lock | M | **−0.11..−0.17** | TPROF hand: gated-first / gelu_pan / qkv 1,808 → 1,000–1,150 cycles (floor 676) |
| 5 | **NSEQ** | norms: params fill (and the NEXT GEMM's B fill + slot loads) before the norm's / previous op's sync; norm input fills for all 8 columns before any drain | S | **−0.08..−0.20** | tpgap: post-norm GEMMs' first-tile ks/k 2,787–2,977 → ~2,650 (B prefetched); k0 end −2..−4 µs each |
| 6 | SSM ssq2 + FSX | rev 13 | M | −0.18 (fits under the SSM's DDR margin, ~3.5 µs per head) | rev 13 checks; SSM per-head span −1.2 µs |
| 7 | attention S3 / K2 | rev 10 / 11 prices (K1 done as KPREP) | M | −0.3..−0.4 (±30 %) | rev 11 checks |
| 8 | dual C drain | C drains over shim S2MM 0 + 1 per column (S2MM 1 is EW output on shims 0–2: packet-shared) | M–L | −0.3 if H_WR is per channel, 0 if per column | gate: DDRW-2ch |
| 9 | LATE-C | rev 15 R3 | L | −0.08..−0.12 (EPLR took most) | plain hand 1,371 → ~800 |
| 10 | E7'' | B on both FIFOs (rev 15 R2) | L | −0.9 ± 0.3 if feasible | gate: POPLF1 (below) |
| — | BDPRE | hoist the post-sync BD writes (`r21` analysis) | S | ≤ −0.1 | dropped: BD write + patch ~0.10 µs (CPST) |
| — | SENDPRE-dep | memtile sends before the wait for dependent ops | S | ~0 | dropped: the first tile is DDR-paced after the wait anyway |
| — | MRES | GEMM → GEMM APAN in memtile | L | (−0.4) | infeasible on memtiles 0–2 (124 KB park, 112 KB free); MLPRR instead |

**Ceiling from 33.88, stacked:**

| step | ms |
|---|---|
| MLPRR −0.36 | 33.5 |
| DAPRE −0.12 | 33.4 |
| NSEQ −0.14 | 33.25 |
| CSPLIT4-lite −0.14 | 33.1 |
| SSM −0.18 | 32.9 |
| attention −0.35 | 32.6 |
| LATE-C −0.1 | 32.5 |
| NRES −0.85 | **31.6** |

- **Conditional:**
  - dual drain (H_WR per channel) −0.3 → 31.3;
  - E7'' −0.9 → **30.4**.
- **Floors:**
  - GEMM k-steps 13.0 ms (12.1 with E7'');
  - epilogues ~1.4;
  - attention ~5.5;
  - EW ops' DDR read floor 6.6;
  - boundaries after all levers ~1.0.
- So the **floor is ≈ 28 ± 1 ms**. Stacked prices below 33 carry ±1 ms.

## Probes (all S unless noted; the first two are main's and are already built)

| id | what | separates | pre-registered |
|---|---|---|---|
| **DDRW** (main, `r20/ddrw.py`) | drains only, c = 1 / 2 / 4 / 8 columns, n = 32 / 512 / 2,048 objects of 2 KB | H_WR vs H_TOK for the 12 µs tail | per-column write rate **3.5–4.5 B/cycle** (6.3–8 GB/s) if H_WR (then 55 KB = 6.8–8.7 µs, the tail's bulk); ≥ 7 B/cycle → H_TOK (~6 µs of token / CP latency). Aggregate 8 columns: ≤ 2 × the per-column rate × 8 or a cap ~55–60 |
| **DDRW-2ch** | the same with 2 drains per column (S2MM 0 + 1) | per-channel vs per-column write limit (gates rank 8) | per channel: 2× per column; per column: ~1× |
| **CPST-N** | CPST_OP = up_a (after norm1) and qkv (after norm2), k = 1 / 8 | the norm's tail + column skew | flag after k = 8 at 6–14 µs after norm1's column-0 last word |
| **POPLF1** | one-core asm probe (wrapb-style): 64 × `vlda.pop.576` on lf1, plus the same on `vldb` interleaved | E7'' feasibility (MACHINE R15.10 UNVERIFIED) | assembles; 64 / 64 pops correct, or a fatal "error in backend" |
| **CPBD** (optional) | `r21`: N dummy blockwrite + patch to shim BD 15 after mlp3_a's sync (.bin-only, xclbin unchanged) | confirms BD-write cost | +12 × N × 0.10 µs (N = 100 → +0.12 ms); kill BDPRE if < 0.2 µs |

## MLPRR — design (rank 1)

**Today:**
- mlp0 (K = 768, nt 16, gelu_pan) writes M chunk q = j + 16 r (row r, step j).
- mlp3_a's A = chunks 0–23, which are row 0's 16 objects + row 1's first 8. The last of them lands at mlp0's end.
- So mlp3_a waits for mlp0's whole drain group, then for the tail and the CP, and then streams 2 MB at the cap.

**Change:**
1. **plan8:**
   - Allow `rr` for gelu_pan.
   - The drain TAP becomes `[16, 3, 1, WIN]` with strides `[3 kstride, kstride, 0, 1]`, so chunk q = 3 j + r. pack8's rr dealing is generic.
   - Split the drains in two per column: steps 0–7 (chunks 0–23) and 8–15 (24–47).
   - Writer keys per 24-chunk window: (M, 0) → group 1, (M, 24) → group 2.
2. **design8:**
   - mlp0's two drain groups go to `pending` separately, as qkv's groups already do.
   - mlp3_a's `finish_deps` waits on group 1 only. That group is complete at mlp0's midpoint plus a tail.
   - mlp3_a's A fill becomes two fills per column (chunks 12–23 → half 1, then 0–11 → half 0), with `ld[1]` pushed before `ld[0]`.
3. **Locks:** the A_RT chunk loads release ONE lock F for both halves, and the t0 send acquires F per chunk.
   - Half 1 loaded first would let t0[0] send half-0 chunk 0 before it is loaded (wrong data).
   - So F becomes Flo / Fhi (`chunk(h, E[h], F[h])`, t0[h] acquires F[h]). F2 stays shared: the sends are in k order.
   - The same lock change serves DAPRE.

**Timing (measured inputs):**
- Half 1 (1 MB) loads during mlp0's second half (~110 µs).
- Half 0's 12 chunks load as mlp0's last tile frees them. At the cap that is 17.5 µs, and mlp3_a's k-step 11 comes 25 µs after that tile starts, so there is no stall.
- So mlp3_a starts ~1.3 µs after mlp0's last epilogue, like mlp3_b does today. That removes W 24.0 − 1.3 and pace 13.3 − ~1, about 35 µs per block.
- mlp3_b is unchanged: it depends on group 2 and loads during mlp3_a's last tile. norm0 waits for mlp3_b as before.

**Bit-identical by construction:** the same N-tiles and epilogues run, on other cores and in another order.

## NRES — design (rank 2)

- **EW (ew8.cc norm pass 2):** write X' (1 KB per unit) into the output ring (→ shim, drain TAP `[RB, 12, 1, XB]` into the Q16 X' parts).
  - Write the panel (576 B) into a new 4-slot ring P (2,304 B; the EW bank-2 region tail has 2,560 B).
  - The EW core's MM2S 1 (unused) drains P.
  - EW text: ~+150 B (576 B free); the out_begin / out_end pair is duplicated for P.
- **Route:** EW (c, 5) MM2S 1 → memtile c S2MM 5.
  - Both ends are free in r19z's lowered MLIR (checked).
    - EW core (0, 5): only S2MM 0 and MM2S 0 are started.
    - Memtile 0: S2MM 1 (B), 2–4 (C join) and runtime S2MM 0 (A ld).
    - Memtile 3: S2MM 1–3 and runtime S2MM 0.
  - It is a circuit flow through rows 4..2, and its routing is the main risk.
- **Memtile:** for an NRES op, a task on S2MM 5 with one BD.
  - It acquires Elo (12), writes 216 × 576 B contiguously (rb-major: unit (rb, q) at rb × 6,912 + q × 576) and releases Flo by 12.
  - The op's t0 / last sends read chunk q as `[18, 576 B]` at stride 6,912 (new send tasks for NRES ops; mid sends whole).
- **CP:** an NRES GEMM pushes no A fill and no ld.
  - Its B fill, NRES ld task, sends and drains go before the norm's input fills. The norm is no longer `finish()`ed before the GEMM.
  - The next norm's `finish_all` keeps the DDR ordering of X'.
- **Saves per boundary:** the norm tail + sync + issue (12–23 µs minus ~1.5 µs of on-chip hand-off) + the pace (6.6–8.5 µs).
  - 72–116 µs per block → −0.86..−1.39 raw, **−0.6..−1.1** realised.
  - Also 4 × 1 MB fewer reads (A) and 4 × 1 MB fewer writes (panels) per block. The norms are DDR-bound, so their own time may drop too (unpriced).
- **Order:** norm1 → up_a alone first. It needs one route, one ring, one op flag.

## DAPRE (rank 3)

- down_a (K = 1536) loads during up_b's last tile in stream order: half 0 first (blocked until up_b frees each chunk), then half 1.
- Measured pace 11.4 µs.
- **Change:**
  - Two shim fills: chunks 12–23 first. They load early into half 1, which is free under K = 768 ops: A_RT starts `ld[h]` for h < K / 768 only.
  - Then chunks 0–11.
  - `ld[1]` is pushed before `ld[0]`, and the per-half F locks (MLPRR item 3) are used.
- Half 0's 12 chunks at the cap take 17.5 µs against down_a's k-step 11 at ~25 µs after up_b's last tile starts, so the pace is ~0.
- Price: −0.11..−0.14 ms. up_b's drain tail shortens too, but it is not on the GEMM cores' path.

## CSPLIT4-lite (rank 4; main's request)

- **Acquire-only BD:** legal in hardware. The core and memtile BD layouts have independent lock_acq_enable / lock_rel_value fields, and a release value of 0 is no release (aie-rt layout via txn_trace `_LOCKS`).
  - In csplit.py emit BD 1a with an acquire only.
  - Check the lowered BD words with `txn_trace`, or decode the core's DMA BD from the CDO: lock_rel_value 0.
- **Sub-part boundary:** 9,216 B is APAN rows 0–15 (16 × 576), and for qkv the Q / K / V object bytes 0–9,215.
  - BD 1a [0, 9,216) acquires CS_CONS (no release).
  - BD 1b [9,216, 12,288) acquires CS_CONS and releases the C prod lock.
  - BD 2 stays as CSPLIT's.
  - The epilogue releases CS_CONS after the 1a bytes are final, and again at the part-1 end.
- **Price:** today's hand is 1,808 for gelu_pan / qkv / gated-first / gelu tiles (TPROF r20t2), floor 676.
  - With 1a draining under the rest of the epilogue, ~1,050 remain.
  - −750 × ~44–48 tiles per block per core gives 18–20 µs → **−0.11..−0.17** after realisation.

## NSEQ (rank 5)

1. A norm's params fill (from W, independent) and its drain BD are issued before the producer's sync; only the 8 input fills come after it, all before any drain.
   - Column 7's input then starts ~2.2 µs after the sync instead of ~6.8.
   - The 2-input norms are compute-bound per column, so this shortens the op's last column.
2. The GEMM after a norm issues its B fill and slot loads before the norm's sync. This needs the norm pushed to `pending` instead of `finish()`ed at once: rev 15 SEQPRE item 2, never built.
   - Then B is prefetched into the memtile fifo (8 objects + 2 in L1), and the first tile's pace drops from A+B (2,787–2,977) to A only (~2,650): ~1.8 µs per op.
   - It also takes 3 BD groups + 8 pushes (~2.2 µs) off the post-sync path. Most of that is hidden by the pace.
3. Do not move the GEMM drains: round 19's "drains first" cost +0.15, cause unknown.

## Re-pricing requested by main (rev 15 items on r19z)

| item | rev 15 | rev 16 | why |
|---|---|---|---|
| CSPLIT4 / -lite | −0.3..−0.35 | −0.11..−0.17 | measured hands: gated-first / gelu_pan / qkv at 1,808, floor 676; ~44–48 such tiles per block per core |
| LATE-C | −0.3 | −0.08..−0.12 | plain hand is 1,371 after EPLR, so at most ~700 cycles × 32 tiles is left |
| DAPRE | −0.2..−0.3 | −0.10..−0.14 | measured pace 11.4 µs; up_b's tail is off the critical path |
| S3 / K2 | ~−0.5 with K1 | −0.3..−0.4 | K1 is done (KPREP −0.13); GEMM text 400 B free |
| ssq2 / FSX | −0.11 / −0.07 | −0.18 together | the SSM reads at 48.5 GB/s in span; its DDR floor is ~20 µs per head against ~23.3 measured, so ≤ ~3.5 µs per head of compute cuts land |
| E7'' | −0.9, feasibility open | unchanged; POPLF1 first | no new evidence |

## Refuted / retired (rev 16)

- **The CP-boundary model (rev 15) and SEQPRE:** the boundary is tail + DDR pace.
  - Post-sync CP work is ~2.5 µs to column 0 and +0.5–0.85 µs per further column, not 92–144 × 0.17 µs.
  - The rev-15 fit absorbed the 12 µs tail into t_op.
- **BDPRE:** BD write + patch is ~0.10 µs.
- **SENDPRE-dep:** after the wait the first tile is DDR-paced, so earlier sends do not help.
- **MRES on memtiles 0–2:** 124 KB park against 112 KB free.

## Tools (rev 16, `r21/`)

| tool | what |
|---|---|
| `tlhist21.py` | per-op gap mean ± SE and a histogram in poll quanta (blocks 1–11): shows the 16.5 µs quantisation of medians |
| `tlops21.py` | per-op gap / span mean and median |
| `bexcess.py` | per-op excess = gap + span − nt × T_tile (TPROF tile parameters per build) |
| `ddrbudget.py` | DDR bytes per op (plan8 TAPs) against the timeline share: per-op GB/s, per-step totals |
| `cpsig.py` | one line per CP segment of a .bin: pushes in order (tile, channel, length), tct run lengths (npu-iron) |
| `cpdump.py` | a segment's instruction kinds with push targets |
| `bduse.py` | shim BD ids written / pushed per tile (r19z: BDs 14–15 never used on any shim, 13 also on shims 3–7) |
| `bndmodel.py` | the boundary decomposition against main's TPROF npz (tpgap-compatible) and the CPST anchors: pace measured vs the cap model, issue model per column |
