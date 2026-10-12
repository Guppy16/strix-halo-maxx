# TAPNext: GEMM

The 11 GEMM ops of a block are **compute-bound** on 24 cores, and almost all of the 269 → 28.4 ms came from making their inner loop,
their epilogues and their C drain cheaper, in that order. A core runs **1,296 k-steps per block** (MODEL, `tapnext/research/r25/ledger25.py`;
ANALYSIS rev 20) on a 144 × 64 tile with a 64-deep K slice. The k-step went from ~2,900 cycles in situ (C++ `mm_acc`, round 6) to ~1,447
(KTL, round 23) against an A-port floor of 1,304 (MODEL, [M8.3](../../common/MACHINE.md#m8-3)). The epilogues went from ~77 bundles per 64 elements (round 5) to hand-scheduled
loops that cost ~1.6 ms per step (r21 timeline), and the C drain from ~2,330 cycles of waiting per tile to a few hundred. What is left is
the DMA/bank collisions of the k-step (~1.2 ms per step, MODEL) and a GEMM L1 and text budget that is full to within 124 B and 128 B.
The mechanisms are in [common/gemm](../../common/book/gemm.md); this chapter is the TAPNext instance.

## The ops and their tiles

| op | K | N tiles per core | k-steps per tile | k-steps per core and block |
|---|---|---|---|---|
| yx | 768 | 8 | 12 | 96 |
| out, out_proj | 768 | 4 | 12 | 48 each |
| up_a, up_b | 768 | 16 | 12 | 192 each |
| down_a, down_b, mlp3_a, mlp3_b | 1,536 | 4 | 24 | 96 each |
| qkv | 768 | 12 | 12 | 144 |
| mlp0 | 768 | 16 | 12 | 192 |

The sum is 1,296. Per k-step a core consumes **10,368 B of A** (144 rows × 64 K as bfp16: nine stripes of 1,152 B through a 3-stripe ring) from
the memtile slot, one 64 × 64 weight tile of 4,608 B through the B fifo, and accumulates 144 × 64 fp32 (36 KB). K = 3072 ops run as two
K = 1536 halves because the A slot (24 chunks, 248,832 B) cannot hold 48 chunks (497 KB of a 512 KB memtile; [J3](../../common/book/negative-results.md#j3)).
The price is a third norm input, see [norms.md](norms.md).

- **Twins say compute-bound**: no-MAC −16.1 ms, 2x-MAC +24.7 ms of 84.0 (round 6, MEASURED; [A1](../../common/book/negative-results.md#a1) refuted the A-port story).
- **Accumulator pieces** (round 9, `MM_PIECES`): the 36 KB of accumulators are 36 pieces of 1 KB (one row block of 8 rows × 32 columns of fp32; 18 row blocks × 2 column pairs). 12 live in the C object
  (12,288 B) and 24 in `acc_hi` (24 KB, now at 0x71A00). The map is affine: pair 0 of row block rb < 12 in C slot rb, pair 0 of rb ≥ 12 in `acc_hi` 0-5, pair 1 in
  `acc_hi` 6 + rb. The epilogues are strided passes over pieces. A bank-aware runtime map made the k-step faster (323 vs 333 cycles per stripe) but the epilogue
  slower (0.64 → 1.45 µs per tile), so the affine map won (NOTES 14.1, 14.3; [A8](../../common/book/negative-results.md#a8)).

## The k-step in TAPNext

The generation table is in [common/gemm](../../common/book/gemm.md#the-k-step-from-c-to-ktl). What the TAPNext layout added:

- **Packing follows the kernel.** Each kernel generation needed the host to permute the weight tiles (`pack8.py`): `B_PAIR4` for the 4-chain call, `B_ROT` for
  ROT, and the E7'' split of B over two FIFOs. Only the GEMM weights are permuted. The first full-step run permuted the SSM and patch parameter tiles too
  and the SSM output was garbage (NOTES 12.1; `step8.py blocks` found it in one run).
- **Layout is part of the schedule.** The A ring is pinned away from the B fifo (`A_BANKS=1,2,2`, −1.5 ms, round 8). E7'' puts the X half of B at 0x70A00 / 0x70C40 in bank 0 and
  the Y half (4,224 B) in bank 3, with the GEMM stack +1,152 B so `core_data` starts at 0x70E80 (NOTES 26.1). `ksc_check` in `design8.py` fails a build whose layout moved and names
  the new `KS_ACC_HI`: the k-step records are build-time constants (KSC, `c731fc3`, −288 B text).
- **One call per tile.** `mm_ktile` (P35, `5bb071e`) runs all of a tile's k-steps in one call, with B locks taken by id. KTL (`516d3a9`) then made that one modulo loop of
  `76 + 1,260 × nks` cycles; the gap between k-steps was waits, not code ([A21](../../common/book/negative-results.md#a21)).
- **In situ vs static.** In situ ROT ran 1,603 cycles per k-step with 168 memory stalls (round 18; KS-P38 counters), against 1,383 static. The excess is DMA writes into the bank the core
  is popping (DMAB), now ~140 cycles per k-step or ~1.2 ms per step (MODEL, `tapnext/research/r20/collide.py`). That is the largest unpriced GEMM item at the pause, and it is blocked by L1
  capacity (see the budgets below; PROPOSALS rev 21 rank 8).

## Epilogue kinds

The kind is a flag word in each N tile's bias object, so one table-driven loop serves every op. The epilogue turns fp32 pieces into the next op's operands:

| kind | ops | what it does | hand-scheduled as |
|---|---|---|---|
| plain | out, out_proj, down_*, mlp3_* | bias add, bf16, drained in the Q16 layout | `bias_pcs` strided passes (44 bundles per 8 blocks) |
| GELU | yx (y half, 12 of 24 tiles) | bias + LUT GELU, bf16 | `bgelu` (II 12, flat-edge tables), `GELU_MS` before it (II 13) |
| gated pair | up_a, up_b | `gelu(gate + b) · (up + b)` on a tile pair = one k-step chunk of down's A, as bfp panels; the second tile of a pair copies the first's stash | `GUP` (II 6 per 64 lanes), `CPY32` (II 1 copy) |
| GELU to panels | mlp0 | bias + GELU, then bfp pushes in mlp3's A layout | `bgelu3` + `PAN` (II 16 per row block of 8 pushes) |
| Q / K objects | qkv | bf16 to bfp panels in attention's object layout | `PAN` flags 5 / 6 |
| V^T objects | qkv | 8 × 8 transposes into V^T panels | `VTK` (II 13) |
| attention O | attn | O as bfp chunks of out_proj's A | `attn_end` |

Measured steps (each bit-identical; commits in the NOTES round sections): epilogue as passes −5.3 ms (round 6), LUT GELU −6.0, `GELU_II` fixed −0.88 and `GELU_MS` −0.53 (round 11),
GUP −0.55, CPY32 −0.30, PAN −0.30, BGELU −0.32 (round 12). In situ a gated_b tile's epilogue fell 5.11 → 3.68 → 2.10 µs and the mlp0 panel pushes cost 1.23 µs per tile (NOTES 17.3). `bgelu` sits at its
port bound (ldb 11 / DM_ADA ~12), so there is no further epilogue lever (NOTES 26.4; [B5](../../common/book/negative-results.md#b5)).

## The C path

The C object (18,432 B) also holds 12,288 B of the accumulators, so a tile could not start until the whole object had drained at 7.9 B/cycle: ~2,330 cycles per tile, ~1.2 ms per step (round 19 TPROF, MEASURED).
The cures, in order, all bit-identical and measured in one-process A/B:

| step | idea | A/B ms | commit |
|---|---|---|---|
| ZHI | zero `acc_hi` right after the C release, inside the drain wait | −0.15 | `9133756` |
| CSPLIT | drain bytes 0-12,287 first (releases the next tile), 12,288-18,431 during it; the lowered MLIR is patched (`r19/csplit.py`, lock CS_Z), an unpatched build hangs | −0.25 | `d5a9ee2` |
| EPLR | plain epilogues release part 1 as they finish it | −0.12 | `d6761bd` |
| CSUB | part 1 in three sub-parts [0, 3,456) [3,456, 6,912) [6,912, 12,288) released one by one (release-by-0 BDs) | −0.13 | `5c9fa57` |
| ZLOAD | the first k-step's MACs use the `zero_acc` config bit; no zeroing stores | −0.13 | `d09a01a` |

TPROF hand-offs (epilogue end to the next tile's entry), plain / gated / mlp0 tiles: 1,355 / 676 / 1,807 cycles before CSUB, 345 / 75 / 668 after (r21t2). About 28 µs per block is still hand-off (up_a 6.4, up_b 7.1, mlp0 5.2,
qkv 3.8; `r25/hands.py`). A double-buffered C needs 18 KB of GEMM L1; LATEC (acquire sub-parts 2 and 3 inside the first k-step) hung the step, probably a core waiting for its own drain while holding a multicast
input (`50640be`, [B9](../../common/book/negative-results.md#b9)). The residual add cannot join the epilogue ([J1](../../common/book/negative-results.md#j1)).

## MSPLIT-P: the last tile as two row passes

For down_b, mlp3_b and out_proj (the ops whose consumers are norms) the last N tile runs as two row passes, rb 0-5 and rb 6-17, each with its own C object, so the norm after it starts on rows 0-5 while the cores
compute the rest (NOTES 28.1; the overlap idea is in [boundaries.md](boundaries.md)). The pieces:
- the core table carries `nks | (nt + 1) << 8`; `mm_ktile` counts objects and picks the pass; KTL takes the row-block period as an argument (`r23/mm_rot_k70p`);
- the passes' own record tables (464 B) do not fit the GEMM L1, so they run on `rot_rec` with five entries patched in place, and a `static_assert` compares the patched table with `ktl_rec`'s pass tables (it caught pass 1's head jump);
- the memtile gets tasks `pa` (rows 0-5 of every chunk, lock-free) and `lastb` (rows 6-17 with the release); the sixth task of a column is pushed only after all columns have five (pushed column by column it hung, r23d).

Result: −0.16..−0.21 ms on down_b + mlp3_b (`dd3677b`, pre-registered −0.18..−0.30), −0.14..−0.19 on KTL (`7edc4ab`), out_proj −0.08..−0.12 (`a20ceff`). That is near the static price and far under the NTWIN bound of −0.92 ms (norms
not waiting for their producers at all): each pass pays its own call and drain, and the norms take their DDR share. MSPLIT on `out` and MSPLIT-C (the GEMM after a norm) are dead or ~0: see [boundaries.md](boundaries.md#what-didnt-work).

## L1 and text budgets at the pause

| resource | state | evidence |
|---|---|---|
| GEMM program memory | 16,256 of 16,384 B (128 free); attention was ~6.9 KB of it at round 13 | HANDOFF round 25; MACHINE [M6.2](../../common/MACHINE.md#m6-2) |
| GEMM L1 | data region 124 B free; bank 3 full (a +384 B `data_size` overflowed it); stack 2,560 B, deepest path 1,856 B | NOTES 28.3; MACHINE [M6.3](../../common/MACHINE.md#m6-3) |
| EW program memory | 15,728 of 16,384 B (656 free) | HANDOFF round 24 |
| EW L1 | stack 4,864 B (deepest 1,792), arena 22,016 B, input ring 20 × 768 B; research found 5.4-6.9 KB for the SSM glue (not applied) | ANALYSIS rev 21 §3; MACHINE [M7.2](../../common/MACHINE.md#m7-2) |
| memtile (0-2) | A slot grown to 36 chunks; 3,584 B left after IRON re-packed it | NOTES 27.1 |

Budgets decided what could be built, more often than any schedule did:
- Text levers are time levers: `TXT18` (EW, −400 B) also gave −0.07 ms, `IV32` freed 560 B of GEMM text, `KSC` 288 B. K2 (a QK-only chunk-0 pass) is bit-identical and −0.02 ms for 608 B, so it stays off (`5e3d6ba`).
- Every layout lever needed an idle bank or free L1 the core does not have (J6, A24, A25 below); the GEMM core is the reason the largest remaining item is unreachable.

## What worked

- Moving the loop to hand-scheduled asm, one generation after another, each bound by the one before: C++ → `gen_mm` (−1.9) → `mm_kstep` (−2.0) → `mm_ks` (−3.5) → ROT (−3.3) → g0a (−0.18) → E7'' (−0.28 at 1.405 GHz) → KTL (−0.82). (NOTES 12-28)
- Giving the k-step what it needs by layout, not by scheduling rules (A ring pinned off the B bank −1.5; X and Y halves of B in two banks).
- Epilogues as array passes and then modulo-scheduled loops: about −9 ms in rounds 6-12, with the GELU table in the epilogue.
- Releasing the C object early and in parts (ZHI, CSPLIT, EPLR, CSUB, ZLOAD): −0.78 ms together.
- Row passes on the last tile where a norm follows (MSPLIT-P −0.27 ms in all).

## What didn't work

L1, text and tile-shape limits killed these (all in [negative-results](../../common/book/negative-results.md)):
- [A9](../../common/book/negative-results.md#a9) <!-- nr: A9 --> A-slot bank pinning with the piece layouts did not link (bank sections overflow); no number recorded. Lesson: write down the section budget before trying a layout variant.
- [A10](../../common/book/negative-results.md#a10) <!-- nr: A10 --> deep-K accumulator visits, A from the neighbour core, a fourth GEMM row: parked on L1 and EW blockers, then superseded by ROT. Revisit only if the k-step needs more A.
- [A16](../../common/book/negative-results.md#a16) <!-- nr: A16 --> moving the accumulators out of the C object needs +12,288 B of L1; 576 B are free. ZHI / CSPLIT / EPLR attacked the wait instead.
- [A24](../../common/book/negative-results.md#a24) <!-- nr: A24 --> shrinking the GEMM stack to free bank-0 room: the deepest path is 1,856 B, so at most ~512 B can go. Name the bytes before proposing an L1 lever.
- [A25](../../common/book/negative-results.md#a25) <!-- nr: A25 --> moving the GELU tables' bank-0 copies to bank 3: never built; moving `aexp_ab` instead is open.
- [J3](../../common/book/negative-results.md#j3) <!-- nr: J3 --> K = 3072 as one op: 48 chunks = 497 KB of the 512 KB memtile with this chunking (3A already took it to 36).
- [J6](../../common/book/negative-results.md#j6) <!-- nr: J6 --> alternating the popped B bank per k-step needs a second 4,224 B Y buffer in bank 0, which has no room.
- [L4](../../common/book/negative-results.md#l4) <!-- nr: L4 --> `out`'s tiles 1-3 following the SSM heads like tile 0: four 18 KB C objects (72 KB) exceed the 64 KB L1, and the 1.77 MB drain would land after the last head anyway.
- [B13](../../common/book/negative-results.md#b13) <!-- nr: B13 --> gated tiles in the CSUB group loop: +0.02 ms against −0.04..−0.08 predicted; kept only as a −256 B text lever (`4ed0218`).
- [B14](../../common/book/negative-results.md#b14) <!-- nr: B14 --> runtime-driven C drains with per-op lengths (74 MB of junk writes per step): the junk sits in part 2 and tiles are not write-bound, so ~−0.05..−0.1 ms for ~17 runtime pushes per block on the in-order CP.

Measured against the C path or the k-step (details in [common/gemm](../../common/book/gemm.md#what-didnt-work)):
- [A8](../../common/book/negative-results.md#a8) <!-- nr: A8 --> bank-aware runtime piece map: +0.8 µs per tile in the epilogue.
- [A12](../../common/book/negative-results.md#a12) <!-- nr: A12 --> B prefetch out of the popped bank; [A15](../../common/book/negative-results.md#a15) <!-- nr: A15 --> A slots in the least busy bank: no idle bank, and bank 0 has no room.
- [A27](../../common/book/negative-results.md#a27) <!-- nr: A27 --> and [L1](../../common/book/negative-results.md#l1) <!-- nr: L1 --> E7'' at R 64 (neutral in situ) and KTL at R 68-69 bank-clean (no schedule in the CP budgets). Both are `unfinished`: more CP workers in a quiet window, or a placement where A slots and pieces never share a bank.
- [B7](../../common/book/negative-results.md#b7) <!-- nr: B7 --> hoisting the B fills for B-starved first tiles: −0.35 against −0.8 predicted, because the dependent boundary is the A slot load from DDR (B is 14 % of the bytes).
- [J5](../../common/book/negative-results.md#j5) <!-- nr: J5 --> GELU on the idle EW cores instead of in the epilogue: ~11 MB per block on top of ~42 GB/s, over the DDR cap.
- [B9](../../common/book/negative-results.md#b9) <!-- nr: B9 --> LATE-C / LATEC: the build hung. [J1](../../common/book/negative-results.md#j1) <!-- nr: J1 --> residual add in the epilogue: no third core input port.

## Lessons

- Attack the largest static item of the whole call, re-solve, replay; every generation moved the bound (chains → accumulator swap → B bank → call prologue → DMA collisions and the A port).
- In this core the real budget is L1 and text, not cycles. Put bytes and banks in every proposal before pricing it.
- Release the C object early and in parts; never wait for your own drain while holding a multicast input.
- Static price × 0.75 for k-step cuts ([A18](../../common/book/negative-results.md#a18)); a gap measured as waits is not code.

## Going deeper

- [`tapnext/npu/NOTES.md`](../npu/NOTES.md) 12 (round 7), 14 (9), 17 (12), 23-24 (18-19), 26 (21), 28 (23); RESULTS 17-18, 28, 30, 32.
- Generators: `tapnext/npu/r7/gen_mm.py`, `r9/gen_ks.py`, `r12/gen_gup.py`, `r12/gen_pan.py`, `r12/gen_bgelu.py`, `r14/gen_vt.py`, `r18/gen_rot.py`, `tapnext/research/r22/gen_e7.py` (`--ktl`); C path: `r19/csplit.py`, `r21/mkzload.py`; MSPLIT-P: `r23/mm_rot_k70p`.
- Probes: `r19/tprof.py` (tile stamps), `r18/ksp38.py`, `tapnext/research/r20/collide.py`, `r25/hands.py`.
- [MACHINE](../../common/MACHINE.md) [M4.4](../../common/MACHINE.md#m4-4), [M6.3](../../common/MACHINE.md#m6-3), [M8.3](../../common/MACHINE.md#m8-3), [M15.1](../../common/MACHINE.md#m15-1). Next: [attention.md](attention.md) (the other user of these cores), [boundaries.md](boundaries.md).
