# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 4, 2026-10-09 night, after round 8 (`full12r8c`, 67.89 ms at depth 12; `RESULTS_FROM_MAIN.md` round 8).
Revision 3 is in `rev3/PROPOSALS.md`, revision 2 in `rev2/`, revision 1 in `rev1/`.

Savings marked [sim11] come from `tools/sim11.py`:
- depth 12, each change alone, against the r8c composition (ANALYSIS rev 4 §2);
- the k-step is fitted at 2,899 cycles = 1.60 us.

Every entry gives the hypothesis, the procedure, the expected outcome (written before the build), the cost (S / M / L),
the device risk, and a **Result** field for the main agent.

Common rules (SKILL §3, §9):
- one dispatch with a short timeout first for every new design, and the known-good test after any timeout;
- record Turbo / Default and gpu_busy with every number; cycles at 1.81 GHz.

## Ranking (revision 4)

| rank | id | what | cost | risk | expected |
|---|---|---|---|---|---|
| 1 | **P20** (rev 4) | `mm_kstep`: one asm call per k-step, A-ring locks inside, stripes in a hardware loop, conflict-free accumulator pieces. Staged a / b / c | M-L | medium | (a) -2.6, (b) -5.7, (c) **-8.2 +- 1.5 ms** [sim11] |
| 2 | **P26** | SSM: exact sqrt by table (Q1) | S | low | bit-exact on every bf16; -0.8 ms |
| 3 | **P27** | SSM: log_a fold + a by grid table + a2 / om (L + A1), one fused pass | S-M | low | a >= 0.95 exact; -1.3 ms |
| 4 | **P28** | SSM: sigmoid by one symmetric table (S1), gate_x first, then gate_a | S-M | low-medium (DAVIS) | -1.0 then -1.1 ms |
| 5 | P7 (rev 4) | fold the norms: P7a numerics emulation first, then the epilogue fold | L | medium (LN numerics) | -4.6 to -6.2 ms |
| 6 | P5 (rev 4) | A slot: K = 768 loaded once (P5a), then chunk streaming + ping-pong | M | medium | P5a -1.3; full -3.9 ms (-3.6 after P20) |
| 7 | P12 / P19 | attention K phase (0.93 -> 0.6 ms per block) | M | medium | -4.0 ms |
| 8 | P18 (full) | 2x2 register blocking inside P20's body | L | medium | k-step 1,940 -> ~1,500, -3.8 ms after P20; then the A port (~1,364 cycles per k-step) binds |
| 9 | P29 | SSM chains hand-scheduled (N0 bit-identical, or N2 = the tables) | L | low | -5.2 (N0) / -5.8 (N2) ms instead of P26-P28's -4.2 |
| 10 | P23 | balance epilogue kinds across a column's rows (yx, qkv) | S | low | -0.7 ms |
| 11 | P24b | acc_zero folded into the first k-step (P20's first stripe MACs from zero) + C object double-buffered | S-M | low | -0.5 and -0.9 ms |
| 12 | P14 | two frames in flight | L | medium | 1.15-1.30x fps (ANALYSIS rev 3 §6), after P3 |
| - | P30, P32, P33, P3, P4, P9, P10 | tools: bank linter in the build, DMA bank bench, r8c timeline + sim11 span fit, txnlint, ... | S-M | none-low | see the end |

Combined (sim11):
- P20 + SSM tables + P5 + P7 + attention + the small ones: **41.3 ms** at depth 12;
- with the 2x2 GEMM and hand-scheduled SSM chains: **35.7 ms** (28 fps); depth 10 29.8 ms.

## Status of the earlier proposals

| id | status | verdict / what is left |
|---|---|---|
| P1 twin builds | DONE | compute-bound |
| P2 cycle stamps | DONE via P15 | stamps everywhere (gprof, ewprof) |
| P3 txnlint | open | before P5 full and P14 |
| P4 command-processor simulator | open | sim11 composes ops; the queue / DMA level is unbuilt |
| P5 A slot | open, **rewritten** (rev 4) | the waits are measured: 27-50 us every op |
| P6 two A streams | REFUTED for now | revisit after P18 full |
| P7 norm fold | open, **rewritten** (rev 4) | P7a numerics first |
| P8 numerics harness | DONE (parts 1-3) | `tools/ssmnum.py` adds the RG-LRU chain |
| P9 object auditor | PARTLY | `tools/bundles.py`, `bankmodel.py`; baseline diffs open |
| P10 port microbenchmarks | clock DONE | the memtile -> core A rate matters after P18 full |
| P11 tile boundaries | PARTLY (epilogue pass) | the rest is P24b |
| P12 attention | PARTLY | K phase is 66% of 0.93 ms |
| P13 SSM | PARTLY (round 8 rewrite -2.4 ms, bit-identical) | the transcendental part is P26-P29 |
| P13a SSM stamps | DONE | compute-bound, waits 1.3% |
| P14 two frames in flight | open | |
| P15 core timer | DONE | |
| P16 counters | not needed | stamps + mmb answered the stall question |
| P17 bank placement (C++ loop) | REFUTED | |
| P18 asm GEMM | first step DONE | full 2x2 after P20 |
| P19 attention on 32 cores | open | EW program memory (592 B free) is the blocker |
| P20 one call per k-step | open, **rewritten** (rev 4) | the full spec is ANALYSIS rev 4 §3 |
| **P21** bank-aware asm call | placement DONE (-1.5 ms); **rules REFUTED** | the rule is exact (`bankmodel.py`); a conflict costs what its serialisation costs, so rules cannot win in dense windows. The layout part moved into P20 (a) |
| P22 program memory | (a) + (b) DONE (-1.3 KB) | (c) superseded by P20 (one stripe body); (d) open |
| P23 row balance | open | unchanged |
| P24 tile-boundary stamps | DONE | 1.31 us handoff + 0.49 acc_zero; the rest of sim10's 7.6 us was op-start wait and k-step |
| P25 `tools/bundles.py` | DONE | |

---

## P20 (rev 4). `mm_kstep`: one call per k-step with conflict-free accumulator pieces — **M-L, risk: medium**

**Hypothesis** (ANALYSIS rev 4 §1, §3; MACHINE §2.2):
- Today's k-step costs 2,899 cycles. Static code is 2,420 of that: 9 calls of 259 (225 mm_core + a 34-bundle wrapper),
  plus 89 of IRON code.
- The remainder is bank stalls (306), DMA collisions (~108) and ~65 more.
- The bank stalls are the in-place accumulators: C loads and stores co-issue in one bank, 33 per call. No schedule of the
  4-chain group structure avoids them (§1.2); a layout does.
- One call per k-step also removes:
  - the wrapper and IRON's per-call code (378 cycles);
  - each call's ramp and drain beyond one group boundary (9 x 33).

**Procedure** (each stage measured alone):
- **(a) Layout in today's call.**
  - `gen_mm.py` takes 4 piece pointers (load and store cursors per group), with the map from `tools/p20map.py` (36
    addresses in k-step order; `acc_piece(rb, pair)` for the epilogues).
  - mm8.cc: `acc_rows` / `rb_bf` / qkv packing / bias_pass / stash go through `acc_piece`.
  - `bankmodel.py` on the new .s with the r8c placement must print <= 4 stall cycles per k-step.
  - regress BIT-IDENTICAL (same MAC order).
- **(b) One call per k-step, stripes not yet pipelined.**
  - design8: `Lock(..., lock_id=4)` for aF and `lock_id=5` for aU (the ids IRON allocates today, so attention's IRON
    acquires are unchanged).
  - Worker: `b = in_b.acquire(1); mm_kstep(ptab, b); in_b.release(1)`.
  - gen_mm.py emits the 9 stripes in a hardware loop:
    - per stripe `acq #53, r4` (r4 = -1) before the stripe's first A access;
    - `rel #52, r5` (r5 = 1) >= 6 cycles after its last A pop;
    - the A slot address and the 4 piece addresses come from `ptab` (5 words per stripe, plus a dummy record).
  - Registers as ANALYSIS rev 4 §3.2.
- **(c) Pipeline the stripe boundary** (the 4th group boundary of the body):
  - the next stripe's first piece loads, `acq`, A fill + rb-0 pops and the B rewind overlap the last piece's stores,
    exactly as gen_mm.py already does at the row-block boundary;
  - the 9th iteration prefetches the dummy record, its `acq` made conditional (`acq.cond`, r26) or the stripe peeled
    (avoid: +1.3 KB).
- Each stage: one dispatch with a 3 s timeout, directed tests t1 / t2 / g11 bit-exact, regress BIT-IDENTICAL, the step
  time, g11 -DPROF.

**Expected** (pre-registered):

| stage | static cycles per k-step | in situ per k-step | step at depth 12 |
|---|---|---|---|
| today | 2,420 | 2,899 (fit) / 3,352 (PROF) | 67.9 ms |
| (a) | 2,420 | ~2,595 (calls 283 -> ~250) | **65.3 +- 0.7 ms** |
| (b) | ~2,060 (9 x 225 + 25 + loop) | ~2,230 | **62.2 +- 1.0 ms** |
| (c) | ~1,796 | **~1,940 (1,850-2,100)** | **59.7 +- 1.5 ms** |

- Program memory: mm_kstep ~1.7 KB replaces ~2.1 KB (mm_core, mm_acc, 9 IRON call sites): GEMM text ~14.45 KB.
- If (c) lands above 2,100 cycles per k-step: the DMA term grew with the denser schedule (run P32), or the `acq` stalls
  (stamp before / after it).

**Risk.**
- Lock misuse hangs the core: 3 s timeout, assert in the generator that `rel` follows the last pop's memory cycle.
- `acq` / `rel` / `lda` in a hardware loop's last bundles: check the assembler, or rotate the body.
- Epilogue map bugs: `step8.py blocks` localises them.

**Result:**

---

## P26. Exact bf16 sqrt by table (SSM mult) — **S, risk: low**

**Hypothesis.**
- mult = RN(sqrt(om)) for bf16 om depends only on om's 7 mantissa bits and its exponent parity.
- A PWL table on m' = 1.m x 2^(e & 1) in [1, 4) with 32 segments (512 B per copy; fp32 offset, bf16 slope, minimax
  offsets) is **exact on every positive bf16** (`tools/ssmnum.py`, exhaustive).
- The result's exponent is (e - 127) >> 1 added in int16; om = 0 -> 0.
- Today's `sqrt32_i` (seed + 2 Newton) takes 68 cycles per vector (stamps: 1,093 per chunk) and is 1 ulp off in narrow
  bands.

**Procedure.**
1. Table generator (r6/lutgen.py style): exhaustive check against `np.sqrt` RNE over all 2^15 positive bf16.
2. `sqrt_n` -> the table:
   - index = ((bits & 0xFF) ^ 0x80) >> 3, then << 4 as a byte offset;
   - m' bits = (bits & 0x7F) | (odd exponent ? 0x3F80 : 0x4000);
   - one `load_lut_2x_float` pair per 32 lanes, one MAC, a conversion, the exponent add, a zero mask.
3. t3 with cmpact: mult must be bit-identical to the reference RNE. It will differ from today's kernel where today's
   kernel is wrong: report how many.

**Expected:**
- 68 -> ~32 cycles per vector (Peano; slot floor 14 on mv);
- SSM -69 us per block, **step -0.8 +- 0.3 ms**;
- mult equals torch's RNE on 100% of elements (today: 99.x%).

**Result:**

---

## P27. a, a2, om in one pass: log_a fold + grid table (A1) — **S-M, risk: low**

**Hypothesis.**
- log_a = RN(sigma * c) with c = -8 sp per channel (precomputed) is bit-identical to torch's two roundings.
- a = bf(exp(log_a)) can be read from a table indexed by the bf16 bits of -log_a:
  - segment = 16 consecutive bf16 values (s = 4): 106 segments for |log_a| in [2^-10, 9.5], 1.7 KB per copy;
  - below 2^-10 the result is 1.
- The table is exact for every a >= 0.95 (exhaustive), and its absolute-error ratio over the real data is 1.008 against
  torch's own rounding.
- a2 = RN((fp32 a)^2) (hi / lo split, 2 MACs) and om = 1 - a2 follow in the same pass.
- s = 3 (3.4 KB per copy) gives ratio 1.001 if memory allows.

**Procedure.**
1. Generator + exhaustive check over every bf16 log_a in [-19, 0]. Report mismatches by a range.
2. One pass per chunk:
   - load sigma_a and c;
   - mul + conversion -> log_a;
   - int16 index ops (`(bits & 0x7FFF) - B0`, clamp, `& 0xFFF0`) and 2 x 16-lane lookups;
   - MAC -> a (store);
   - split, a2, om (store om for P26).
3. cmpact on t3 against the reference; then regress (expect small diffs), DAVIS 10 videos.

**Expected** (`ssmnum.py`, exact sigma upstream):
- a >= 0.95: 0 mismatches; a overall ~8% 1-ulp differences, all below 0.95 (max |da| 3.9e-3);
- mult 1.4% 1-ulp; a = 1 agreement 100%;
- cycles 94 -> ~38 per vector; SSM -115 us per block, **step -1.3 +- 0.4 ms**;
- DAVIS AJ within +-0.15 of 70.78.

**Result:**

---

## P28. Sigmoid by one symmetric table (S1) — **S-M, risk: low (gate_x) / medium (gate_a)**

**Hypothesis.**
- T on [-10, 0) in 80 segments of 1/8 (1.25 KB per copy); sigma(x) = T(-|x|), or 1 - T(-|x|) for x > 0; 0 below -10.
- The tail is exact in effect: gate_a < -9.18 gives (a, mult) = (1, 0) for every channel (sp <= 1.18), and gate_x <
  -10 contributes <= 4.5e-5 |xc|.
- On real gates: 4.5% of sigma values 1 ulp off; rms error equal to bf16 rounding's (ratio 1.002).
- Downstream of gate_a: a >= 0.95 gets 0.40% 1-ulp flips with bias -4e-6; a = 1 agreement 100%.

**Procedure.** gate_x first (it only scales the input): t3 cmpact, regress, DAVIS. Then gate_a, separately: DAVIS.
Memory: S1 + A1 + Q1 = 3.45 KB per copy, 6.9 KB with two copies; EW has ~6 KB. Single-copy the one that does not fit
(both LUT pointers equal; expect +8 cycles per 32 lanes on it).

**Expected:**
- 79 -> ~32 cycles per vector per gate;
- gate_x **-1.0 +- 0.3 ms**, gate_a -1.1 +- 0.3 ms;
- DAVIS within +-0.15 AJ for gate_x; gate_a within +-0.3 (if worse, keep gate_a arithmetic: N1-x).

**Result:**

---

## P29. Hand-scheduled SSM chains — **L, risk: low**

**Hypothesis.**
- Peano runs the chains at 3-7x their slot floor (MACHINE §2.3): 321 cycles per channel vector against a floor of 73
  (vec slot) for today's math.
- A generator in the style of gen_mm.py, modulo-scheduling 4-6 vectors in flight, reaches ~1.3x the floor.
- Two variants:
  - N0: today's math, bit-identical, -5.2 ms in total;
  - N2: P26-P28's tables, mv-bound floor ~55, -5.8 ms in total.
- Only worth it after P26-P28, if the SSM is still on the critical path (P14 hides it).

**Expected:** N0 ~95 cycles per channel vector (1,520 per chunk); N2 ~70 (1,120).

**Result:**

---

## P7 (rev 4). Fold the norms into the GEMMs — **L, risk: medium**

**Hypothesis.**
- The 4 norms cost 541 us per block on the critical path (EW cores; GEMM cores idle).
- The producing GEMMs (out, down_b, out_proj, mlp3_b) already write the residual stream. Their epilogues can also write:
  - the bfp16 A panels of the un-normalised x;
  - per-row partial sums (Σx, Σx²) per N tile.
- The consumer computes x W' (gamma folded into W' on the host) and applies the row statistics in its epilogue:
  out = (C - mu s) / sigma + b', with s = column sums of W' and b' = beta W + b.

**Procedure.**
- **P7a (no device):** emulate on `acts_clip3` per norm site:
  - bfp16 of x vs of x_hat (the deployed path);
  - the cancellation C - mu s in fp32;
  - judge by the bfp ratio of the consumer's output. RMS sites are expected fine; LN sites depend on |mu| / sigma per row.
- Then build per site, starting with the RMS sites.

**Expected:** -4.6 ms (norms at 40 us each) to **-6.2 ms** (no norm ops, +0.3 us of epilogue per tile) [sim11].
A per-site bfp ratio <= 1.1 to go ahead.

**Result:**

---

## P5 (rev 4). The A slot: load K = 768 once, then stream in k-chunks — **M, risk: medium**

**Hypothesis.**
- Measured waits: 27-34 us after a GEMM op (the 248 KB slot at ~7.5 GB/s), 50 for mlp3_a, ~55 after an EW op (incl.
  ~22 of sync). 484 us per block in all.
- **P5a:** K = 768 ops fill both halves with the same 124 KB; sending from one half halves the load, -16 us on 7 ops.
- **P5 full:**
  - `Bd(iteration=...)` k-chunk loads with a lock release per chunk, so the first k-step starts after ~5 us;
  - up_b, down_b and mlp3_b (A ready before the previous op ends) loaded into the free half during the previous op.

**Expected:**
- P5a **-1.3 +- 0.4 ms** and -107 MB of DDR per step;
- full **-3.9 +- 1.0 ms** (-3.6 with P20: the first tile then outruns a K = 768 load by ~3.6 us);
- g11 first-tile waits: 33 -> <= 8 us (GEMM after GEMM), 2-3 us for up_b / down_b / mlp3_b.

**Risk.** Memtile BDs (static chain near 22) and lock values <= 63. Run P3 txnlint first.

**Result:**

---

## P23, P24b, P12 / P19, P14

Unchanged from rev 3 apart from sim11's numbers:
- P23: -0.7 ms.
- P24b: acc_zero folded into the first k-step -0.5 ms (P20 makes it nearly free: the first stripe's first k-step MACs
  from zero, with no piece loads); C object double-buffered -0.9 ms.
- P12 / P19: -4.0 ms for 0.6 ms of attention per block.
- P14: build only after P3.

**Result:**

---

## Tools

- **P30 bank linter** (S, no device): run `tools/bankmodel.py` in the build on every asm kernel with its placement
  (input_with_addresses.mlir + llvm-nm), and print the stall cycles per call. A layout or placement regression then
  shows at build time. Extend the classifier to the attention asm and to LUT loads (2 addresses per vldb.4x).
- **P31 rotating accumulator (dm4)** (L, not now): with a spare 5th accumulator preloaded, a group boundary drops from 13
  to ~9 cycles (st-slot bound: 12 stores before the 4th new chain): -144 cycles per k-step, -1.2 ms. The register
  rotation has period 5 groups against 4 per stripe, so the body would unroll to 5 stripes (~1,000 bundles): blocked by
  program memory unless P18's restructure changes the group shape.
- **P32 DMA bank bench** (S): mmb with the A and B DMA streaming into the core's banks (a fill loop in the runtime
  sequence) at the placements of P20. It splits the ~12 cycles per call of DMA collisions by bank and says whether the
  core or the DMA yields (MACHINE §9 Q6).
- **P33 r8c timeline** (S): `step8.py timeline` of full12r8c. sim11 then fits spans again (today it has one constant
  fitted to one number).
- P3 txnlint, P4 command-processor simulator, P9 baseline diffs, P10 port rate: as rev 3.
