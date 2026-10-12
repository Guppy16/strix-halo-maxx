
# Round 12 (optimisation round), keyed by proposal id (2026-10-10)
All numbers: test machine, Turbo, gpu_busy 0, pyxrt kernel time, MEASURED unless marked, median of 60 dispatches; cycles at 1.81 GHz.
Baseline full12r11d re-timed 52.10 ms. Best build **full12r12e: 50.05 ms** (depth 10 42.02, 8 33.51, 6 25.20), regress
BIT-IDENTICAL to full12r9a at every step. Details: npu-tapnext NOTES 17, RESULTS 21.

## G1a "gup" (rev 7) — DONE (flag -DGUP): -0.55 ms measured (predicted -1.02 +- 0.2: missed, explained)
- II 6 per 64 lanes as predicted (CP-SAT, r12/cpsched.py; msched's IMS also reached it). One-core bench bit-exact, 10 cycles per
  block incl. call overhead.
- In situ (g11 PROF, gated_b tiles): epilogue 5.11 -> 3.68 us per tile, against the predicted 2.44. The prediction kept "half a
  stash copy"; the differentials show the stash copy alone is 1.70 us per tile (avg; ~3.4 on each gated_b tile), the gate GELU
  1.20, the gate bias pass 0.36, gup 0.36. gup's own saving (1.43 us x 32 gated tiles x 12) = 0.55 ms: the model was right
  about gup and wrong about what was left around it.
- The stash copy then went to a 32 B modulo copy (-DCPY32): -0.30 ms more, gated_b tiles 3.68 -> 2.10 us.

## G1b (rev 7) — DONE: panpush (-DPAN, -0.30 ms) and the fused bias + GELU (-DGELU_FLAT,-DBGELU, -0.32 ms)
- PAN serves mlp0 AND the Q / K objects of qkv (flags 5 / 6: a 2D flush jumps 1,152 B between objects): II 16 per row block
  (2 per 64 lanes, the predicted floor). The C++ pushes cost 1.23 us per mlp0 tile and 0.82 per qkv tile (differential).
- bgelu at II 12 (rev 7 predicted 12-13 with flat tables): 8 free x registers once `vmsc s, sh, 1.0` replaces the -1.0
  constant (bit-identical, as rev 7 suggested). x goes through an 8-slot stack ring (see "side effects" below). It also runs
  yx's tiles (drained layout), so gelu_ms is gone. G1 in total (GUP + CPY32 + PAN + BGELU): **-1.49 ms** vs predicted -1.76 +- 0.35.
- Text was the binding constraint: GEMM 16,288 / 16,384 B after removing `__modsi3` (an `s % 3`: 160 B).

## GELU II 13 -> 10 (handoff item) — PARTLY: flat-edge tables DONE (bit-exact on every finite bf16), II 10 NOT reached
- With flat edges + an x reload on vldb the ResMII is 10 (QEY_RS), but CP-SAT found no II-10 schedule within 8 x registers
  (400 s each, 4 reload variants; not proven infeasible). Rev 7's "registers set the II" holds. The standalone loop was
  superseded by the fused bgelu (II 12).

## P29a SSM sigmoid (rev 7) — DONE (EW_DEFS -DSIGK): -0.61 ms measured (predicted -1.0: missed, explained)
- Kernel-only at II 12 (U 2), bit-exact, 499 cycles per chunk's 32 vectors vs 898 for the deployed C++ loop on one core. Rev 7's
  1,168-cycle baseline was round 9's; the timeline shows the SSM op -0.52 ms (= ~360 cycles x 216 chunks x 12), i.e. the
  bench's saving.
- Text: the EW core had 144 B free; the kernel-only form (no prologue) + r12/asmfit.py (compact body, last bundle still aligned)
  made it 350 B, less than the C++ it replaces.

## P43 pre-registered check (rev 7) — mlp3_a 166 -> 159 us per block (r11b -> r11d timelines), not 137 +- 8: MISS
P43b (the rest) not attempted.

## Generator lessons for rev 8 (all measured, in the skill)
- Exact register pressure in the scheduler (CP-SAT) reaches the IIs msched's IMS + sink could not allocate.
- Side-effect ops must sit in the last stage (the drain's garbage iterations run the earlier stages): a store of x to its
  final place in an early stage corrupted the 4 vectors after every run in situ (the bench did not check past the end).
- `.p2align 4` before the body inserts one nop bundle after the prologue: prologue ops then collide with body ops on itinerary
  ports (a vlda's QEY_WA vs a vfloor): bgelu II 11 had iteration 1 wrong. msched's checker now replays pads 0 / 1.
- A 2-bundle zero-overhead loop body runs once (>= 3 bundles).
- Output stream semantics (push / flush / flush.2d) measured; the stream pointer is exact to the byte after a flush.

## Not attempted this round
P24b, P35, P42 (the GEMM stack gave 256 B to the flat tables; EW stack unchanged), P27, P29b/c/d, P48-P50, P44 / P7-E, P14, the
host trio H1-H3, P43b.

DAVIS 10 videos on full12r12e (round end): depth 12 AJ 70.84 (OA 92.57), depth 10 69.90 (OA 91.02): identical to rounds 10-11.
