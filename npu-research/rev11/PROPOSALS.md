# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 11, 2026-10-10 (late), after round 15 (`full12r15m`, 41.52 ms at depth 12; RESULTS_FROM_MAIN round 15). Revision 10
is in `rev10/`, older ones in `rev9/` .. `rev1/`. The analysis behind every number is ANALYSIS rev 11 (draft copy:
`r16/ANALYSIS_rev11.md`). Prices are deltas against r15m at depth 12.

**Method (rev 11).** Every price is a WHOLE-CALL static count of the r15m elfs (`r16/r15m_0_2.dis`, `r16/r15m_0_5.dis`, made
with `-z`) or a one-core measurement, converted at 1 cycle per SSM chunk = 1.43 µs per step (216 chunks x 12 blocks at 1.81 GHz);
norm costs per op (18 row blocks per core) at 1.81 GHz, times the ~58 norm ops per step. The attention levers come from `tools/sim18.py` (attsim17 with
P60g applied and S1's in-situ rescale cost fitted to its MEASURED -0.29: R = 270, not the 200 rev 10 assumed); bars = kappa
0.5-0.7 plus the II range. No ±100 % defaults; where a cost cannot be separated statically the deciding microbench is named.

**Headlines.**
- **New tools for the implementer** (use them every build): `tools/asmlint.py` (offline hazard linter, 22 rules; flags 18/18
  historic bugs, deployed r15m clean: `tools/asmlint_hist.py`), `tools/try.sh` (one-command build + lint + regress + time,
  <= 6-line summary), `tools/asmfit_tmp.patch` (the parallel-build race: a build lock in design8 + private temp dirs in asmfit).
- **S2 fits after all**: the 768 B of running-max slots go on attkb's **stack frame** (bank 0; the GEMM core's deepest path uses
  832 of 2,560 B: 1,728 B free). No split layout, no LUT move. **-0.38 ms (-0.29 .. -0.45)**.
- **The "output-ring store stall" does not exist** (static whole-call count): out_copy(16 vectors) costs 77 cycles in r15b's
  code, of which 16 are stores; 3 calls + the loop glue = 289 = the measured phase. The ~3.5 cycles per vector was a per-call
  overhead divided by vectors. Levers follow from that: fewer calls (X1, P-IN), not other banks.
- **EW re-rank** (static): gates (2 x gate_gemm at 57 bundles per 20-vec-op trip) is the biggest remaining SSM pass -> G1 -0.50;
  norm pass 1 + stats fused in asm (N1) -0.6 +- 0.25; the per-chunk call glue (in_part x 7: P-IN -0.25; x-part copies: X1
  -0.28); sigk + ssx fused (FSX) -0.24 +- 0.09. SQK (round 16, in progress) -0.31 .. -0.35.
- Ceiling: **31.0 ms** at depth 12 (35.4 without the k-step row); depth 10 ~25.9, depth 8 ~20.8.

## Ranking (revision 11; vs full12r15m 41.52 ms)

| rank | id | what | cost | risk | expected (depth 12) | pre-registered whole-call price |
|---|---|---|---|---|---|---|
| 1 | **G1** (new) | both gate GEMMs + the xc -> bfp panel push as ONE generated loop over 8 (gemm, np) iterations, II 20 = the vec bound (16 mmul + 4 epilogue ops per np) | M | low (bit-identical: same MAC order, same two-rounding epilogue) | **-0.50 +- 0.07** | PROF phase gates (PH 1 -> 2) 652 -> 230-300 per chunk (preamble ~30, prologue (S-1) x 20 ~60, 8 x 20, ret ~10) |
| 2 | **S2** (rev 10, new layout) | running row-block max in attkb, slots on attkb's stack (`paddxm [sp], #768`, bank 0), initialised -inf in the -m prep; exact test on one vector | M | low (bit-identical) | **-0.38 (-0.29 .. -0.45)** | attkb one-core warm unit 261 -> 261-275 per rb (II 34 -> 34-36); flagged rb 1,129 -> ~650 (k_rb --krb stamps) |
| 3 | **N1** (new) | norm pass 1 + the statistics loop fused, in asm: per cb 6 loads, 6 macs + 4 stat macs, 2 stores (vec-bound 10 / cb), same accumulation order (bit-identical) | M | low-medium (the ring wrap per stream: copy the rare straddling part as in_part does) | **-0.6 +- 0.25** | per unit 224 + 40 (stats share) -> 120 cycles; per op 18 rb x 12 units: -31k cycles = -17 µs of compute per op; the bar is the I/O-bound share (EW PROF of r15m decides) |
| 4 | **SQK** (main, round 16) | sqrt as an asm loop (II 13, U 2) | M | low | -0.31 .. -0.35 (main -0.29) | 493 -> 230-270 per chunk |
| 5 | **X1** (new) | rotate the conv state instead of copying it: the SSM writes only this frame's x; x-1 / x-2 are read next frame from the regions of the two previous frames (plan8 addresses by frame mod 3; three runtime sequences per depth, step8 swaps the instruction bo per frame like set_depth) | M | medium (host + plan only; bit-identical) | **-0.28 +- 0.03**, and 2 KB less SSM output per chunk | x copies (PH 8 -> 13) 289 -> ~95 per chunk (one 16-vector out_copy) |
| 5b | X1b (fallback) | the three x parts in one out_copy call (P[1..3] contiguous in the arena) | S | low | -0.18 +- 0.03 | 289 -> ~150 |
| 6 | **P-IN** (new) | in_part's common path inline: P[p] = ring + ((g_in_b + p KB) mod IN_B) with one straddle test per chunk; the copy path only for a part that straddles | S (C++) | low | **-0.25 +- 0.06** | PH 0 (7 noinline minsize calls, ~25-35 each incl. jl / ret delay slots) ~245 -> ~40 |
| 7 | **FSX** (new) | gate_a's sigmoid moved into ssx's loop (ldb 11 + 8 = 19 of II 20; vec stays 20): sigk keeps only gate_x (16 vectors) | M | medium (register pressure: ssx already reloads 12 of 13 constants) | **-0.24 +- 0.09** | sigk 509 -> ~260; ssx 509 -> 509-560 (II 20-23) |
| 8 | S3, V1, K1, K2 (rev 10) | attention glue after S2 | M | low | -0.73 together (-0.06 / -0.21 / -0.27 / -0.19) | as rev 10 |
| 9 | T-CALL (new tool) | a static "call census": calls per unit of work x whole-call cycles from the elf + the op schedule | S | none | finds the next div768-class cost (round 15's -1.05 ms nobody priced) | |
| 10 | P55, P35 + P24b, P51, P57, k-step | as rev 10 | | | -0.36, -0.98, -0.96, -0.5, -4.35 (ceiling row) | |
| parked | P60t, F1 / F2, P60h, P66 / P67, H1 / H2, P14, P59, H4 | as rev 10 | | | | |

Combined (cumulative from 41.52): G1 -0.50 (41.02) -> S2 + S3 + V1 + K1 + K2 -1.11 (39.91) -> N1 -0.6 (39.31) -> SQK -0.33
(38.98) -> X1 -0.28 (38.70) -> P-IN -0.25 (38.45) -> FSX -0.24 (38.21) -> P55 (37.85) -> P35 + P24b (36.87) -> P51 (35.91) ->
P57 (35.41) -> k-step 2,055 -> 1,600 **31.06 ms** (depth 10 ~25.9, depth 8 ~20.8; r15m sweep fit 3.43 ms per block + 0.4).

## Status of the revision-10 proposals (round 15 verdicts)

| id | predicted | measured | verdict / lesson |
|---|---|---|---|
| P60g | -1.08 +- 0.22; 614 +- 18 µs per block | -1.06; 610 µs | **hit** (both) |
| P29b | -1.0 +- 0.07 | -0.79 | **miss**: the loop was priced, not the call (7-stage prologue 120 + preamble). Rev 11 prices whole calls only |
| S1 | (only S1 + S2 + S3 priced: -0.74) | -0.29 | in-situ rescale ~270 cycles, not 200 (sim18 fit) |
| S2 | -0.38 | not built | **rev 10 error**: 2,688 B is bank 3's linker REGION for .aie.bank3 sections, of which sgelu_cd + serf_cd take 2,304: 384 free. Rev 11: on the stack (rank 2) |
| P29c | -0.9 | -0.84 | hit |
| P29d | -0.82 .. -0.99 | -0.72 (conv) + ~-0.12 (ZRING) | hit for the conv half |
| P52 | -0.71 .. -1.14 | -0.18 (CPY64) | **miss**: the 798 baseline was stale (main) and the residual 477 is call overhead, not a store stall (this pass) |
| P7-E step 1 | (part of -2.3 +- 0.8) | -0.40 | consistent |
| div768 (main's find) | not priced | -1.05 | **research blind spot**: minsize helper calls per unit; T-CALL proposed |
| per-column K / V buffers, multi-TAP gathers | retired (rev 10) | | unchanged |
| Q1 (l2-depth) | not run | | still unexplained, low priority |

## The new tools (this pass) — use them for every asm change

### asmlint (`tools/asmlint.py`, rules: `--rules`)
Offline linter over built elfs / objects / generated .s / `-z` disassemblies; in npu-iron (CPU only):
`docker run --rm --network none -v $HOME/git/app/.scratch:/work npu-iron python3 /work/npu-research/tools/asmlint.py --prj
/work/npu-tapnext/build/full12NAME.prj` (or `... build/r15/ssx.s` for a generator's output before building).
- Checked statically: ZOL end / start alignment, minimum body (3), setup distance to the loop END in bytes (<= 52 B error,
  < 80 B warning), last body bundle full form, register visibility by the LLVM rule incl. landing order and the mv bypass,
  unlanded reads, scalar conveyor reads, alu -> alu at +1 inside a ZOL body, itinerary port collisions (incl. scalar `lda`'s
  DM_ADA and vmov x <- bm), ex / x aliasing (landing order), ABI r0 (value-returning functions), callee-saved r8-r15 / p6 / p7
  and lr, .td immediate types, push / flush on p2, raw bytes in .text, non-last-stage side effects in epilogue-less loops
  (stack-ring pointers exempt), acquire-equal.
- Provenance: Peano's functions are the calibration set (the compiler knows the true latencies): intent rules (which value a
  read means) report there as "calib", never as errors; universal rules (ZOL, ports, ABI) must hold for Peano too, and do.
- Not checkable statically (listed by `--rules`): bank conflicts, garbage stores onto live data, FIFO restart byte positions,
  lock protocols, kernel-only negative-iteration bounds, budgets (stackaudit / llvm-size do those).
- `tools/asmlint_hist.py`: 18 historic artefacts in `r16/hist/` (16 real, 2 reconstructions marked) + the deployed r15m elfs;
  ALL PASS (table in ANALYSIS rev 11 §1).

### try.sh (`tools/try.sh NAME [--ew-add ..] [--k-add ..] [--env K=V] [--sweep] [--print-only]`)
Builds full12NAME (+ 10 / 8 / 6 with --sweep) from r15m's flags plus the changes (refuses an existing name: the jit cache ignores
sources), runs asmlint + llvm-size, `step8.py regress --ref full12r9a`, `step8.py time` (60 + 20 warm-up) after a gpu_busy check,
and prints <= 6 lines. Mount root = `ROOT` (one line). Only `--print-only` was run by research; the implementer validates it.

### The parallel-build race (`tools/asmfit_tmp.patch`, apply with `patch -p1` in npu-tapnext)
The race is wider than asmfit's `<name>_fit.s`: design8's asm_object / ew_object write fixed names (build/mm8asm.o, ew8cpp.o, the
generated `<k>.s`, and the `<k>.h` that mm8.cc includes as `"build/<k>.h"`), and aiecc links those objects later in the same
process. Per-build object directories would silently compile against a stale `build/<k>.h` (a quoted include searches the
including file's directory first), so the patch takes an exclusive `flock` on build/.design8.lock for the whole design8 run
(parallel builds queue instead of corrupting each other) and gives asmfit a private temp dir per call (+ `-z`).

## Microbenches named (cannot be separated statically)
1. **EW PROF of full12r15m** (fits again): the norm phases (pass 1, stats, pass 2, in_need / out_begin waits) decide N1's bar
   and the I/O-bound share; the SSM phases confirm G1 / X1 / P-IN's static baselines (pre-registered above).
2. **Output ring, if anyone doubts the static count**: one-core cpyb with the destination in bank 0 while an MM2S DMA drains it
   vs idle, 16-vector calls: predicted equal within 0.2 cycles per vector (DMA reads at most one bank access per 8 cycles at
   4 B/cycle). In situ, a stamp pair around ONE out_copy(16): predicted 77-90 cycles (r15b code), 95-120 (r15m code: copy_run).
3. **ZOL setup**: zero2 with k full-form nop bundles (k = 0 .. 6) and with 2-byte nops, body 3 bundles (r13/attkb.py --zero):
   the bytes-to-le limit between the measured 52 (bad) and 80 (good).
4. **SCALAR-PLUS1 scope**: r11/mk_haz-style probe of `or rA, ..` -> `or rB, rB, rA` at +1 inside a ZOL body vs straight-line.

## G1 — the gate GEMMs in one generated loop (rank 1)
gate_gemm today (r15m, 81 bundles): a 57-bundle ZOL body per np (4 trips) with vec 20 / lda 14 / ldb 14 / st 12 ops: 2.85x its
vec bound, + ~24 bundles outside + call (~12); called twice per chunk after an 8-push C++ panel loop. One generated loop over
the 8 (gemm, np) pairs: A panels (g_xcp, 8 pops) restarted per np (as attkb restarts Q), B (wx / wa) popped straight, two acc
chains (8 macs each, dependent macs 2 apart through the bypass), the two-rounding epilogue (4 vec ops) and the bf16 stores
into g_gx / g_ga in the last stage; the 8 xc pushes go in its preamble (or stay C++). II 20 (vec) — CP-SAT on the laptop.
Text: -480 B C++ for ~400 B asm (EW has 1,424 B free). Bit-identical. Pre-registered above.

## S2 — the running max on the stack (rank 2)
attkb's preamble: `paddxm [sp], #768` (and `paddxm [sp], #-768` before ret), slot pointer = sp - 768 (a stack ring as bgelu
does); the -m prep loop stores -inf into the 6 slots (or attkb's preamble does with zero2-style stores). Per iteration: vlda
slot (DM_ADA: 29 -> 30 of II 34), vmax_lt (mv), vst (st) — the slot pointer steps 0 x5 then +128 by a 2D modifier like K's.
The exact test reads one vector per row block. Bank: the stack is bank 0, which also holds aexp_ab (the exp table's first copy:
16 vldb.4x64 per iteration touch it) and possibly Q (acc_hi spans banks 0-1): give cpsched the bank of each stream; at most
26 of 34 cycles touch bank 0. Stack: deepest GEMM path 832 B + margin 128 of 2,560 (stackaudit r15m; attkb's own frame 64)
-> 768 more fits. sim18: II 35 -0.38 (-0.33 .. -0.44), II 36 + T 100 -0.29, II 34 -0.45.

## N1 — norm pass 1 + statistics (rank 3)
Static per row block (r15m, LN; RMS in brackets): pass 1 12 units x (8 x 23-bundle ZOL trips + ~40 glue incl. in_need /
in_done) = 2,690; stats 96 x 5 (4) + 2 ln_stats + 4 rowsum8 ~ 900 (600); pass 2 12 x (nrm_ln 230 [nrm_rms 156] + ~40) = 3,240
(2,350); total ~6,800 (5,600) = 68 (56) µs per op of compute at 18 row blocks per core, against ~140 (~100) µs measured: the
rest is I/O (input 256 n_in cycles per unit, output 1.6 KB per unit = 400 cycles at 4 B/cycle, the 9 KB rings) and waits.
N1 fuses the stats accumulation into pass 1 (same order: q-major, cb-minor, so bit-identical) and runs it as a modulo loop at
~10 cycles per cb: per unit 224 + 40 -> ~120. Compute-bound ops gain fully, port-bound ops less: -0.6 +- 0.25 ms.

## X1 / X1b / P-IN — the SSM's per-chunk call glue (ranks 5, 6)
In r15b's code one out_copy(16 vectors) = out_copy 20 bundles + 5 delay slots, cpy64 3 + 7 (prologue) + 16 + 6 (ret), back 7 +
13 = 77 cycles; r15m (copy_run between them) ~120. The x parts phase = 3 calls + loop glue = 289 (measured) — fully static.
- X1 removes two of the three by rotating the conv state across frames (plan8 + three runtime sequences, step8 picks by frame
  mod 3; the data are identical: bit-identical). Also 2 KB less output per chunk (the SSM output port carries 8 KB per chunk today).
- X1b (S) lays P[1..3] contiguously so one 48-vector call does all three.
- P-IN: 7 in_part calls per chunk (minsize, noinline: jl + 5 + ret + 5 per call) -> inline arithmetic.

## FSX — gate_a's sigmoid inside ssx (rank 7)
sigk: 24-bundle body per 2 vectors (II 12 / vector, ResMII 8: ldb 8, mv 8), 32 vectors (gx, ga) = 509 in situ; ssx: vec-bound
20 / vector, 16 vectors (ga) + a 120-cycle prologue = 509. ga's sigmoid needs 8 ldb + 8 mv per vector; ssx uses ldb 11 / mv 5
of 20: both fit under II 20 if registers allow (ssx holds 1 of 13 constants resident). sigk then runs 16 vectors (~260).

## T-CALL — a static call census (rank 9, tool)
Round 15's largest single gain (div768, -1.05 ms) was a minsize helper called per unit that no model priced. A tool that walks
each EW / GEMM op's control path in the elf (jl graph from asmlint's parser), multiplies whole-call static cycles by the call
count per unit (from the op table: units per op, chunks per head) and lists the top-20 call sites per step would have found it.
Build it on asmlint's parser (research can do this offline next pass).
