# moondream NPU attention: v2 to v15

Fused attention for moondream's tower (16 heads x 729 x 72, non-causal) went from **3.78 ms per block in phase 8 to 0.675 ms in phase 9**
(MEASURED, 8 columns, Turbo; 102 -> 18.2 ms per frame by x27 arithmetic, [`p9`](../npu/p9/), MD RESULTS phase 9). Three things bought the 5.6x: a one-pass
online softmax with bfp16 operands, finding that every earlier kernel had run in the **floor rounding mode**, and an exact exp built from
the MAC unit and a Horner cubic. Phase 10 then proved a hand-scheduled score block bit-exact and found it did not matter. The kernel sits at the
bfp16-operand accuracy floor (rms 9.8e-3 against a host floor of 8.1e-3). This chapter is the history behind the general
[attention](../../common/book/attention.md) chapter; the table of versions is the main content.

## What was priced and what was measured

Phase 7 priced a straightforward QK^T, softmax, PV at ~1 ms per block, ~27 ms per frame, from bundle counts (MD RESULTS phase 7). Phase 8
built it (`p8/attn.cc`) and measured 3.78 ms: **102 ms per frame**, 3.8x over the price. The kernel ran ~14k cycles per 32-key chunk for ~6k cycles of counted
vector work: 9.9 KB of spills, 288 register transposes and 576 block loads per chunk. The path to that number was itself a list of toolchain traps
(MD NOTES phase 8): the first dispatch timed out because IRON's `acquire(n)` holds n objects in total (two `acquire(1)` against one
`release(2)` drifted the locks); a vectorised reduction was wrong because of a shuffle lane mapping (rms 2.9e-2, found by dumping the row maxima); there is no
fp32 64-lane `aie::transpose`; the 8-lane bf16 `aie::max` returns wrong values; Peano aborts (assertion) on vectors assigned under runtime conditions.

## Phase 9: the build trail

Per block, Turbo, rms against the `bf16emu` attention reference (MEASURED, `p9/attn{4..15}.{cc,py}`, `runattn8.py`):

| version | ms per block | rms | what it taught |
|---|---|---|---|
| v2 (phase 8, bf16 API mmul, two passes) | 3.82 | 6.2e-3 | floor rounding; P normalised before PV |
| v3 one pass, 48x48 | 3.18 | 1.0e-2 | |
| v4 bfp16 operands, P converted on chip | 2.04 | 1.49e-2 | NaN until panel groups were padded to 64 B; then P truncation bias |
| v4b row sums from a ones panel in PV | 1.82 | 1.13e-2 | sum of exactly the P the mmul multiplies |
| v5 exp in three stage loops | 1.43 | | ~98 bundles per 32-lane exp |
| v6 magic-constant exp, 3 K / 2 V streams | 1.61 | worse | two FIFO load units; more live streams spill |
| v7 Q and P row blocks in block-vector registers | 0.88 | 1.40e-2 | conversions were truncating everywhere |
| **v7r `set_rounding(conv_even)`** | 0.885 | **1.03e-2** | every earlier kernel ran with floor; the v6 exp was exact all along |
| v8 K/V broadcast through the memtile (33.8 -> 9.7 MB per block) | 0.955 | | not DMA-bound: the kernel-less floor is 0.29 |
| v9 4 chains, bf16-only running max | 0.90 | 9.6e-3 | best accuracy |
| v10/v11 K and V on separate fifos | 0.82 | | |
| v13 K object and V^T object, one held at a time | 0.845 | 9.6e-3 | DMA overlapped, gaps 1,002 -> 36 cycles |
| v14 native `aie::exp2`, `aie::inv` | 0.51 | 2.7e-2 | exp2 is a ~4-bit approximation |
| **v15 v14's infrastructure with v9's exact exp** | **0.675** | **9.8e-3** | 6.5e-3 vs the bfp-operand reference |

The final design: bfp16ebs8 Q, K (natural layout) and V^T panels packed on the host, QK^T into fp32, a bf16 running max, exp as a
magic-add integer split plus a Horner cubic, P converted to bfp16 on chip, PV with a ones column giving the row sums, native bf16 reciprocal
for 1/sum (costs 2e-4), 48 queries per core and K/V broadcast to the four cores of a column in lockstep. Differential variants at 0.675 ms:
exp 0.24, scores ~0.2, PV ~0.13, rest ~0.1; the DMA floor (0.29) is hidden under compute.

## Three findings that carried over

- **Rounding mode.** The core's conversions default to floor and apply to every bf16/bfp16 conversion. ln.cc and the GEMM set round-to-even; no
  attention kernel did, so v2 to v7 were all in floor mode (MEASURED, p9). Setting it did not change speed, it changed rms 1.40e-2 -> 1.03e-2 and
  made the exact exp exact ([machine](../../common/book/machine.md)). Phase 8's 6.2e-3 vs v15's 9.8e-3 is therefore not like for like.
- **Trace before guessing.** A hardware trace (`Program.enable_trace`, a one-column build; the trace unit cannot be routed beside 8 broadcast
  columns) showed ~1,000 cycles of lock stall between chunks: IRON gives a forwarded L1 fifo at most 3 free slots, and the paired `acquire(2)` left one,
  so there was no prefetch. The cure was to split the kernel per object (v13), not to deepen the fifo (depth 6 still lowers to 3). Inside chunks vector issue was
  only ~25 % of cycles and stalls under 5 %: the rest is the dependent-chain schedule. It also showed the item boundary (scalar begin/end/copy) at 21.6k cycles,
  25 % of an item, cut to 5.5k in v14.
- **Attention here is compute-bound.** v8 cut K/V traffic 3.5x and gained nothing ([D2](../../common/book/negative-results.md#d2) <!-- nr: D2 --> cites this as the
  precedent for TAPNext). See [attention](../../common/book/attention.md#is-it-compute--or-data-bound).

## Phase 10 and after: the schedule, not the instructions

Phase 10 proved `clang++ -S` output reassembles byte-identically with a three-line dialect fix (`##sym`, `.word`, `event`) and links through IRON,
built a latency table of 879 instructions (now `common/tools/aie2p_isa_timing.json`) with a checker (0 violations on Peano's output), and wrote a
**hand-scheduled score block** (`p10/gen_scores.py`, 162 bundles per row block, bit-exact). The K phase stayed at 6,730 cycles per chunk and
0.672 vs 0.671 ms per block (MEASURED): Peano's own score section was already at the mmul chain bound. The cost is the exp (~440 cycles per row block,
bound ~13 vector ops per 32 lanes) and the softmax bookkeeping, so the exp block is the target (~2.6k of the 6.7k K-phase cycles; the bound
suggests ~0.13 ms per block). Phases 11 and 12 spent their budget on the GEMM and GELU and never reached it, so **0.675 ms per block, 18.2 ms per frame**
is where attention stopped; the target was 10 ms ([overview](overview.md#the-tower-budget)). The earlier "scores 0.2 ms" differential that suggested the scores mattered had
also removed the P stream and pops.

## Numerics at the task level

On 462 labelled frames the bfp16 attention operands cost 1 false positive at 729 tokens (97.6 % against 97.8 % with bf16 attention) and none at
pool 16 (99.6 % against 99.4 % for the GPU path); the flips are tv frames the GPU already scores within +-1.25 of zero
(MEASURED, `moondream/gpu/BF16_ELEMENTWISE.md` "Follow-up"; [numerics](numerics.md)).

## What worked

- One-pass online softmax with a ones-panel row sum and bfp16 operands: 3.78 -> 1.82 ms by v4b.
- Setting the rounding mode and keeping the exp exact: accuracy at the floor, and the base for v15.
- Hardware trace on a one-column build; splitting the kernel per fifo object to get prefetch: 0.90 -> 0.845 ms.
- A bit-exact reassembly of Peano's output and a latency checker: the asm route is open for the next attempt.

## What didn't work

- [D22](../../common/book/negative-results.md#d22) <!-- nr: D22 --> v6's three K and two V streams (1.61 vs 1.43 ms, spills), v12's `seek_byte(0)` per row block (NaN), v14's native exp2 (0.51 ms but 2.7e-2),
  v15s's stage-major exp (wrong and no faster). Lesson: two FIFO load units, and an exact exp is not optional.
- [F7](../../common/book/negative-results.md#f7) <!-- nr: F7 --> the elementary-function unit: exp2 ~4 bits, `inv` ~1 ulp (fine for 1/sum, not for P).
- [F8](../../common/book/negative-results.md#f8) <!-- nr: F8 --> the 8-lane bf16 max returned wrong values; use 16 or 32 lanes.
- [G5](../../common/book/negative-results.md#g5) <!-- nr: G5 --> re-seeking a block-vector stream (v12): the fill pointer is not where you think.
- [G7](../../common/book/negative-results.md#g7) <!-- nr: G7 --> vectors assigned under a runtime condition make Peano abort ("Bad PHI" assertion); apply masks unconditionally.
- [D2](../../common/book/negative-results.md#d2) <!-- nr: D2 --> v8's K/V broadcast: fewer bytes, no gain.
- [N12](../../common/book/negative-results.md#n12) <!-- nr: N12 --> the hand-scheduled score block matched Peano (162 vs ~160 cycles) and changed nothing; the exp block is the target. [A23](../../common/book/negative-results.md#a23) <!-- nr: A23 --> is the GEMM's first hand schedule.
- [N8](../../common/book/negative-results.md#n8) <!-- nr: N8 --> Floor rounding as the default for v2 to v7: every number before v7r is in the wrong mode.
- [N9](../../common/book/negative-results.md#n9) <!-- nr: N9 --> Block-vector stream alignment: a seek or start not on a 64 B boundary gives NaN (v4); panel groups are padded.
- [N10](../../common/book/negative-results.md#n10) <!-- nr: N10 --> A forwarded fifo with `acquire(2)`: no prefetch because of the 3-slot cap; deeper fifos lower to 3 as well.
- [N15](../../common/book/negative-results.md#n15) <!-- nr: N15 --> The first reductions (shuffle lane mapping, no fp32 64-lane transpose, scalar `__divsf3` calls): plan on 16 and 32 lanes in bf16 and MAC-unit sums.
- [N11](../../common/book/negative-results.md#n11) <!-- nr: N11 --> Estimating from structure (27 vs 102 ms per frame).
- [N13](../../common/book/negative-results.md#n13) <!-- nr: N13 --> Injecting the standalone `event` encoding as `.byte` into a bundle stream hung the core twice (both of phase 10's timeouts); assemble only the hand `.s` and merge with `ld.lld -r`.
- [N14](../../common/book/negative-results.md#n14) <!-- nr: N14 --> Tracing eight broadcast columns: the trace unit cannot be routed; use one column.
- [N16](../../common/book/negative-results.md#n16) <!-- nr: N16 --> Calling the asm route closed after phase 5's failed inline-asm attempt; phase 10 reassembled `clang -S` output byte-identically.

## Lessons

- Fix the numerics baseline (rounding mode, reference) before comparing versions. Seven kernels were compared in the wrong mode.
- Price a fused kernel from a measured prototype, not from bundle counts: the 3.8x miss was spills and per-block structure.
- A trace that shows vector issue at 25 % points to the schedule; check the instruction count before rewriting in asm, and check which part of the loop the time is in.
- A differential variant that removes more than one thing gives a misleading cost for any of them.

## Going deeper

- `moondream/npu/NOTES.md` phases 8-10 (build trail and "Learned"), `RESULTS.md` phases 8-10; code `p8/attn.cc`, `p9/attn15.{cc,py}`, `p9/runattn8.py`, `p9/tracesum.py`, `p10/gen_scores.py`, `p10/asmcheck.py`.
- General: [attention](../../common/book/attention.md), [elementwise](../../common/book/elementwise.md#exp), [toolchain](../../common/book/toolchain.md#the-asm-route).
