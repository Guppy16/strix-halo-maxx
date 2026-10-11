# moondream NPU elementwise: GELU, LayerNorm, LayerNorm + residual

The tower's elementwise work is small in arithmetic and large in time. The first attempt, GELU in fp32 in the fc1 epilogue, would have cost
**~120 ms per frame** because AIE2P has no native fp32 vector arithmetic (MEASURED per op, extrapolated x27; [`p6`](../npu/p6/)). Rewriting everything in bf16
with fp32 only inside the MAC unit brought it to the numbers below: **LayerNorm 116 us and LayerNorm + residual 146 us, both bit-exact**
to the bf16 reference, and **GELU +0.56 ms per fc1 (~15 ms per frame)** at 0.998 % one-ulp flips (MEASURED, [`p8`](../npu/p8/), MD RESULTS phase 8).
GELU is the largest remaining piece of the elementwise budget: its schedule runs at 492 bundles per 64 elements against a slot bound of ~114
and was not hand-scheduled (phases 11-12). The general rules are in [elementwise](../../common/book/elementwise.md); this chapter is the moondream instance.

## Phase 6: the fp32 plan fails

Hypothesis: fuse tanh-GELU into fc1's `acc_store` in fp32 (the accumulator is fp32 anyway). It built and was correct (0.0226 % of outputs one bf16 ulp off,
rms at the bf16 floor), but fc1 took **5.1 ms against 0.56 ms** without GELU (MEASURED, `d8_fc1g`). The fp32 vector ops are emulated at ~21 bundles per
16-lane op, ~1.3 cycles per element per op; a 40-op GELU is ~55 cycles per element. The int8 dequant epilogue of phase 4 (3 ops) paid 6 cycles per element for the same
reason. Getting even this to build took four changes: no inlining into the 12x12 block loop (16 KB program memory), 16-lane vectors (64-lane temporaries
spilled 5.5 KB of stack), two chains per out-of-line call, a larger worker stack. The fp32 LayerNorm (`p6/ln.py`, `ln.cc`) was 246 us per pass, ~13.5 ms per frame.
Softmax with an fp32 exp per score was priced at ~80 ms per frame. Conclusion: elementwise math on this NPU is bf16 (native 32 lanes) with fp32 only in reductions.

## Phase 7-8: bf16 kernels and the right GELU form

**Accuracy decides the form first.** A GPU emulation (`bf16emu.py`, [numerics](numerics.md)) rounded every op to bf16 and measured per-op error: LayerNorm 3.7-4.4e-3, softmax
4.2-7.6e-3, and tanh-GELU in the form `1 - 2/(exp(2u)+1)` **1.4-2.1e-2**, ten times the others, from cancellation. The sigmoid form `x / (1 + exp(-2u))` is
the same function at 0.5-4.5e-3 (MEASURED, `moondream/gpu/BF16_ELEMENTWISE.md`). At the task level, with only GELU rounded to bf16, the tanh form gave 14 false positives (exact path: 8), 10 decision flips and a maximum margin shift of 2.22; the
sigmoid form gave 13, 5 flips and a shift of 0.97 ([numerics](numerics.md)).

**GELU kernels** (fc1 729x1152x4304, `p7/` then `p8/`):

| variant | vs the sigmoid reference | bundles per 64 elements | fc1 us (without GELU 560) |
|---|---|---|---|
| fp32, emulated | 0.0226 % one-ulp | ~3,600 (1,806 per 32) | 5,100 |
| bf16 tanh form, bf16 Horner | 29.96 % one-ulp, rms 3.55e-3 | 420 | 1,168 (contended) |
| sigmoid form, bf16 Horner exp, bf16 Newton reciprocal | 38.8 % one-ulp, rms 3.8e-3 | 392 | 948 |
| **sigmoid form, split-operand exp and reciprocal** | **0.998 % one-ulp, rms 1.5e-4, max 3.9e-4** | 492 | **1,117** |

The split trick: take an fp32 value in the accumulator as `hi + lo` (two bf16 vectors from one MAC); then every product is three MACs (hi x hi, hi x lo, lo x hi)
accumulated in fp32, which gives ~2^-16 precision through a degree-4 Horner and two Newton steps using only native MAC-unit ops. It costs ~1.3 bundles per element and
is shared with the attention softmax (`p8/expsplit.h`). Result: +0.56 ms per fc1, **~15 ms per frame** (27 x 0.56, MEASURED per op). The native exp2/tanh are too coarse
to replace it ([F7](../../common/book/negative-results.md#f7) <!-- nr: F7 -->).

**LayerNorm** (`p7/ln.cc`, `ln.py`, 768x1152 over 32 cores): a column's shim feeds a memtile that splits to four cores (objects of 4 rows) and joins back; gamma/beta come once.
fp32 mean and variance run through the bf16 MAC unit (`x * 1`, `d * d` with d = bf16(x - bf16(mean))), the rstd is a scalar bit-trick plus three Newton steps (no libm), then
bf16 rounding after each of (x - mean), x rstd, x gamma, + beta. **Bit-exact against the reference, 116 us uncontended** (142 us under GPU load), 55 per frame ~6.4 ms.

**LayerNorm + residual** (`p7/ln_res.py`): y = LN(x + delta) and x' = x + delta from one pass, so the residual add costs nothing extra. Objects are one row of [x | delta], gathered per column by one
4-D shim pattern (sizes [24, 4, 2, 1152]); two-row objects exceed the shim's 10-bit wrap and decompose into more than 16 BDs. Both outputs bit-exact, **146 us**, 54 per frame
~8 ms. Fusing the residual into the GEMM epilogues was not built: a core has two input channels (A and B), and a residual tile would need a third stream or 166 KB per column
in the memtile.

## Phase 11-12: why GELU stays at 492 bundles

The slot bound of the phase-8 epilogue (`gelu_b2`) is ~114 vector-slot ops per 64 elements (52 `vmac.f`, 16 `vmul.f`, 20 `vsub.f`...) and ~110 in the store slot (55 fp32-to-bf16
conversions). Peano emits 492 bundles with 94 (phase 11) or 74 (phase 12 table) spills and 154 nops: 4.3x over the bound, ~11 of the 15 ms (MEASURED, MD RESULTS phase 11-12).
Phase 12 tried to give the scheduler independent work: the same arithmetic as three stage loops with L1 staging gave bit-identical output but **1,177 us** rolled and 471 / 469 / 481 bundles
fully unrolled with 2 / 4 / 8 vectors (70 / 122 / 302 spills). Peano issues about one op per bundle (467 non-nop ops in 470 bundles): the loop is allocation-bound, so each extra chain becomes spills
([B12](../../common/book/negative-results.md#b12) <!-- nr: B12 -->, [elementwise](../../common/book/elementwise.md#gelu)). A hand schedule needs a list scheduler with register allocation over ~80-op chains
(12 x registers allow ~3 chains) and the exact Peano forms for the fp32-to-int emulations; estimated 150-200 bundles per 64 elements, GELU 15 -> 5-6 ms per frame. Not started.

## What worked

- The sigmoid form with split-operand exp: 38.8 % -> 0.998 % flips for 169 us more per fc1.
- Doing the reductions in the MAC unit (x * 1, d * d) so LN stays bf16 and bit-exact at 116 us, half the fp32 version.
- Emulating the exact bf16 op chain on the GPU first, then asking the kernel to match it to the bit.
- A one-row-object gather for LN + residual: residual adds for free.

## What didn't work

- [B12](../../common/book/negative-results.md#b12) <!-- nr: B12 --> staged GELU chains for Peano: no gain, allocation-bound. Lesson: the asm route, or fewer ops.
- [F7](../../common/book/negative-results.md#f7) <!-- nr: F7 --> hardware exp2, tanh and reciprocal for the GELU and softmax: too coarse.
- [F8](../../common/book/negative-results.md#f8) <!-- nr: F8 --> the 8-lane bf16 max (also hit in phase 8 attention).
- [N17](../../common/book/negative-results.md#n17) <!-- nr: N17 --> fp32 elementwise on AIE2P: emulated, 120 ms per frame for GELU alone, 13.5 ms for LN. A hardware fact.
- [N18](../../common/book/negative-results.md#n18) <!-- nr: N18 --> GELU in the tanh form in bf16: cancellation, 1.4-2.1e-2 per call; the same function in the sigmoid form is 5-10x better.
- [N19](../../common/book/negative-results.md#n19) <!-- nr: N19 --> A plain bf16 Horner exp and Newton reciprocal: 38.8 % one-ulp flips, rms 3.8e-3; the split-operand version is needed.
- [N6](../../common/book/negative-results.md#n6) <!-- nr: N6 --> A residual add in the GEMM epilogue: no third input stream; not built.
- [N20](../../common/book/negative-results.md#n20) <!-- nr: N20 --> Fusing GELU into the block loop (program memory) and 64-lane fp32 temporaries (5.5 KB of stack): did not link or spilled.
- [N21](../../common/book/negative-results.md#n21) <!-- nr: N21 --> The GELU hand schedule is open: the bound is ~114 bundles, the compiler gets 492.

## Lessons

- Count what the unit does natively before porting an op. fp32 vector math is emulated, so the plan changed in one measurement (5.1 vs 0.56 ms).
- Choose the numerically stable form of a function (sigmoid, not 1 - 2/(e+1)) before spending effort on kernel accuracy.
- Split a value into hi and lo bf16 and accumulate three products in fp32 when you need more than bf16 precision from MAC-only hardware.
- Compare the compiler's bundle count with the slot bound. A 4x gap at one op per bundle means allocation, and independent chains do not fix it.

## Going deeper

- `moondream/npu/NOTES.md` phases 6-8 and 11-12; code `p6/ln.{cc,py}`, `p7/ln_res.py`, `p8/expsplit.h`, `p8/mm_f32acc.cc` (`-DGELU_BF16`), `p12/mm_f32acc.cc` (`-DGELU_STAGED`), `p12/fnsize.py`.
- Emulations: `moondream/gpu/bf16emu.py`, `bf16emu_check.py`, `BF16_ELEMENTWISE.md`.
- General: [elementwise](../../common/book/elementwise.md#the-bf16--fp32-rules), [machine](../../common/book/machine.md), [toolchain](../../common/book/toolchain.md#peano-llvm-aie).
