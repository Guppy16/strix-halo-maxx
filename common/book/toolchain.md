# Toolchain and hand-written assembly

Designs are written in Python with IRON (mlir-aie 1.4.4), kernels in C++ compiled by Peano (llvm-aie), and the hot
loops in generated assembly. The pattern that paid off was: IRON for the data movement, C++ for cold code, and a
generator plus CP-SAT for every loop that must beat Peano. The asm route took the TAPNext GEMM call from 321 to 225 bundles
(round 7) and later to a modulo-scheduled k-step loop (ROT −3.3 ms, round 18). Every generated kernel passes a value-level
replay and `asmlint` before it runs. The rules themselves are in the
[xdna2-npu skill](../../.claude/skills/xdna2-npu/SKILL.md) §2-§4. This chapter explains why the toolchain is built the way it is.

## IRON and mlir-aie

- A design is an `@iron.jit` function. `Worker`s run C++ kernels; `ObjectFifo`, `forward`, `split` and `join` move data;
  `TensorAccessPattern` (TAP) drives shim DMAs from the runtime sequence (SKILL §2).
- When IRON cannot express a shape, **patch the lowered MLIR**. aiecc rebuilds a design byte for byte from its own
  `input_with_addresses.mlir`, so a patched copy (an extra lock, a split BD chain, a switchbox rule) rebuilds in
  seconds (`tapnext/npu/r19/csplit.py`, round 19, d5a9ee2). The rule that comes with it: never dispatch the
  unpatched xclbin of a kernel that takes the new lock. It hangs.
- IRON 1.4.4 behaviours we hit, with their workarounds:

| behaviour | symptom | workaround | ref |
|---|---|---|---|
| multicast + unicast packet flows on shared DMA ports | a destination silently dropped, hang | multicast-only routes | [G8](negative-results.md#g8) <!-- nr: G8 --> |
| two packet flows from two DMA channels of one tile onto one port | no route found | declare both from MM2S 0, patch the switchbox (`r20/nresfix.py`) | [G9](negative-results.md#g9) <!-- nr: G9 --> |
| acquire-only BD block | verifier refuses | release by 0 (a no-op) | [G19](negative-results.md#g19) <!-- nr: G19 --> |
| jit cache key excludes source files | an edited kernel reuses the old build | a new build name after every edit | [H22](negative-results.md#h22) |
| worker loop counters lowered to i64 | ~6 extra scalar ops per loop | `scf.for` with i32 bounds (IV32: −560 B of text) | [M5.5](../MACHINE.md#m5-5), e7afce3 |
| `acquire(n)` means hold *n* in total | `acquire(1)` twice then `release(2)` drifts the locks | acquire one object per call | SKILL §2 |

## Peano (llvm-aie)

Observed Peano behaviour: short, simple loops pipeline; long dependency chains do not. What we measured:

- **It does not software-pipeline long chains.** exp, sigmoid and GELU leave every MAC-latency slot as a nop. Two independent
  chains interleave (exp 73 → 38, sigmoid 140 → 77, GELU 199 → 111 bundles per vector). Four chains spill and gain nothing
  ([G4](negative-results.md#g4) <!-- nr: G4 -->). Splitting a chain into short passes through L1 lets Peano pipeline each pass
  (SSM exp: 63.5 → ~44 bundles; round 9).
- **On long chains it issues ~1 op per bundle**, whatever ILP you expose. Register allocation binds (moondream GELU 492
  bundles per 64 elements against a slot bound of ~114; [B12](negative-results.md#b12)).
- **`-O2`, not `-Oz`/`-Os`.** At -Oz the EW object was *larger* and 2.4x slower; minsize on hot code cost up to 10x
  ([G1](negative-results.md#g1) <!-- nr: G1 -->). Use minsize only on cold functions.
- **A forced II can silently give the serial loop.** With restrict copies of one pointer, Peano emitted ~50 unpipelined
  bundles, and round 10 blamed "interlocks". With separate restrict parameters the loop ran at 17.8 cycles, −0.88 ms
  ([B2](negative-results.md#b2) <!-- nr: B2 -->, [B3](negative-results.md#b3) <!-- nr: B3 -->). Read the emitted loop of the binary you deploy.
- The same C++ can be serial in situ and pipelined in a one-core bench: the SSM z loop was 27 bundles per 64 lanes inside `ew_ssm`
  and 97 cycles for all 8 in the bench ([E15](negative-results.md#e15), [M3.4](../MACHINE.md#m3-4)).
- Compiler errors: vectors assigned under runtime conditions or `aie::select` give "Bad PHI" ([G7](negative-results.md#g7) <!-- nr: G7 -->);
  inline sched barriers in a pipelined loop make the compiler fail ([G6](negative-results.md#g6) <!-- nr: G6 -->). Mask arithmetically and stamp
  through a noinline call.
- Never re-seek a block-vector stream after popping from it; build a fresh stream object per block
  ([G5](negative-results.md#g5) <!-- nr: G5 -->).

## The asm route

`clang++ -S` emits Peano's schedule. `moondream/npu/p10/asmfix.py` fixes three dialect gaps, and `clang -c` reassembles it
byte-identically (moondream phase 10). A hand `.s` is assembled alone and merged with the C++ object by `ld.lld -r`, then linked
with IRON `Kernel(...)`. Never inject raw `.byte` bundles: that hung the core twice (SKILL §4).

**The machine model** is `common/tools/aie2p_isa_timing.json`: 879 instructions with slot, operand latencies and itinerary ports,
generated from llvm-aie's TableGen by `common/tools/gen_isa_timing.py`. Apply the visibility rule from
[machine.md](machine.md#the-compute-tile) on top. This one table replaced three copies (3976925).

**Hardware rules a generator must encode** (each one cost at least one wrong kernel or one timeout):

| rule | ref |
|---|---|
| ZOL: start and last body bundle 16 B aligned; body ≥ 3 bundles; ≥ 80 B from the last lc/ls/le write to `le` | [G13](negative-results.md#g13) <!-- nr: G13 -->, [M3.5](../MACHINE.md#m3-5) |
| `.p2align 4` before a body inserts a nop bundle, so prologue ops run a cycle early; fit the prologue to end aligned | [G14](negative-results.md#g14) <!-- nr: G14 --> |
| a dense 576-bit FIFO stream needs a fill every 8 pops | [G15](negative-results.md#g15) <!-- nr: G15 --> |
| `add rX, rY, #imm` is 7-bit signed and wraps silently | [G12](negative-results.md#g12) <!-- nr: G12 --> |
| `acq` value 0 = acquire-equal-0 (hang); a function returning a value takes its first scalar argument in r1 | [G16](negative-results.md#g16) <!-- nr: G16 --> |
| lock conditions 0 / 1 only (bit 0 is tested) | [K1](negative-results.md#k1) |
| side-effect ops only in the last stage (drain iterations write past the buffer) | SKILL §3 (round 12) |
| an acquire needs ~6 bundles before the fill it guards (1 raced in situ) | [A17](negative-results.md#a17) |
| a kernel `.s` and its record header must come from the same schedule | [K3](negative-results.md#k3) <!-- nr: K3 --> |

## Generators and modulo scheduling

The progression shows what each tool added:

1. **`r7/gen_mm.py`** (round 7): a list-scheduled GEMM call with 4 accumulator chains over 2 column blocks. 225 vs 321 bundles,
   bit-identical, −1.9 ms (L1 bank conflicts ate the rest of the predicted −7 ms; [A7](negative-results.md#a7)).
2. **`r9/gen_ks.py`** (round 9): the whole k-step as one zero-overhead loop, modulo-scheduled, with a checker over the unrolled
   stream (every MAC's operands, addresses, lock order). −3.5 ms.
3. **`r11/msched.py`** (round 11): a general modulo scheduler over virtual registers, with ports from the itinerary, iterative modulo
   scheduling, modulo variable expansion and a value-level checker. GELU at II 13, bit-exact.
4. **`r12/cpsched.py`** (round 12): CP-SAT (OR-tools) with one-hot issue cycles; slots, ports and banks modulo II; exact steady-state
   MaxLive per register class. It found II 6 / 12 / 16 where list scheduling failed. Solve with `nice -n 10`, ≤ 12 workers.
5. **`gen_rot` / `gen_e7` / `gen_e7k`** (research passes 14-19): the GEMM k-step as a CP model with banks per cycle,
   a whole-call value-level replay on any L1 placement, and `fit` (prologue alignment). See
   [gemm.md](gemm.md#the-k-step-from-c-to-ktl) for what they bought (ROT −3.3, E7'' −0.28, KTL −0.82 ms).

CP-SAT limits: "no schedule found" within a time budget is not a proof. Two levers stay open for that reason: KTL at R 68-69
bank-clean ([L1](negative-results.md#l1)) and FSX at II 30-33 ([L2](negative-results.md#l2)). Re-solving old schedules with more
time paid off once: attkb II 36 → 33, worth ~0.065 ms per II at depth 12 (round 22, 5a75321, 2049c58).

## asmlint, tcall and fitting

- **`common/tools/asmlint.py`** checks 22 static rules (FIFO-POP9, ZOL alignment and distance, LOCK-ACQ-GAP, IMM-SRC on the
  source text, conveyor reads, …). It flags every historic case (23) in `tapnext/research/r16/hist/`. Run it on every
  build (`tapnext/npu/r24/lint.sh`); the best build has 0 errors.
- **`common/tools/tcall.py`** gives whole-call static cycles from a `-z` disassembly: loops, delay slots, callees, glue.
  It reproduced the norm row blocks within 3 %. Price with it, not with the loop body ([H8](negative-results.md#h8)).
- **`asmfit`** expands the cheapest subset of compact bundles to the full 16-byte form so the last body bundle lands
  aligned (−192 B on a 24-bundle body). Run `fit` on every delivered modulo kernel: an unfitted E7'' kernel ran without
  a hang and silently lost rows 0-7 (round 21).
- Disassemble with `llvm-objdump -d -z`. Without `-z`, nop runs print as `...` ([H24](negative-results.md#h24)).

## Program memory

16 KB per core holds the kernels *and* IRON's per-op control code. What worked:

- One table-driven op loop instead of unrolled ops (12 unrolled GEMM ops took ~12 KB). Small constant trip counts come from
  an L1 table so LLVM cannot unroll them (−1.3 KB, round 8). With this the elfs are the same at every depth, so one xclbin
  serves depths 12 / 10 / 8 / 6 by swapping the instruction buffer.
- Records as `constexpr` tables instead of run-time builders (−0.7 KB, round 13); `asmfit`; 2-stage schedules for short loops;
  tail calls; i32 loop bounds.
- `llvm-size` after every kernel edit. Build the default configuration after every flag-guarded edit (6f7b7d7).

## What didn't work

- Forced-II GELU "slower in situ": the loop had never been pipelined. **Revived** with separate restrict parameters (−0.88). [B2](negative-results.md#b2)
- Hazard hypotheses H1-H4 for that loop: their premise was wrong. [B3](negative-results.md#b3)
- -Oz / minsize on hot code: slower, and not smaller for vector code. [G1](negative-results.md#g1)
- `AIE_PREPARE_FOR_POSTPIPELINING`: 100 vs 65 bundles. [G2](negative-results.md#g2) <!-- nr: G2 -->
- `aie::linear_approx`: 2.7x slower than a hand lookup (its object is reloaded every compute). [G3](negative-results.md#g3) <!-- nr: G3 -->
- More than two chains in C++: spills. [G4](negative-results.md#g4)
- 4-chain GEMM loops in C++: 42-61 bundles vs 28 deployed; Peano holds A in 8 of 12 registers. [A6](negative-results.md#a6) <!-- nr: A6 -->
- Shapes IRON or the BDs cannot express: stride-0 tile BDs, padd immediates that aren't multiples of 64, two producer BDs. [G10](negative-results.md#g10) <!-- nr: G10 -->
- The 16 KB text budget parked S3, K1, K1C, rolled rescale and PROF builds. [G18](negative-results.md#g18) <!-- nr: G18 -->
- In-kernel `read_tm` hung until the bus enable was found (dead for one round, then resolved). [G17](negative-results.md#g17)
- Parallel builds in one directory raced on the generated asm. [H23](negative-results.md#h23) <!-- nr: H23 -->
- Linker regions and dropped entry points read as free space. [H25](negative-results.md#h25) <!-- nr: H25 -->
- `llvm-objdump` without `-z` hid nop runs. [H24](negative-results.md#h24) <!-- nr: H24 -->

## Lessons

- Bench the binary you deploy, and read its disassembly first.
- Encode every hardware rule in the generator *and* in asmlint. A rule that only lives in a note gets broken again.
- Price a kernel by static cycles + replayed bank stalls of the whole call, then × ~0.75 for k-step cuts
  ([gemm.md](gemm.md#pricing-a-k-step-change)).

## Going deeper

- [SKILL](../../.claude/skills/xdna2-npu/SKILL.md) §2-§4 (all rules) and §10 (tool locations per round).
- `common/tools/`: asmlint.py, asmlint_hist.py, tcall.py, try.sh (build + lint + regress + time in one command).
- Model chapters: [TAPNext GEMM](../../tapnext/book/gemm.md), [TAPNext attention](../../tapnext/book/attention.md),
  [moondream NPU GEMM](../../moondream/book/npu-gemm.md).
