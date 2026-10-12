# PROPOSALS.md — state at pause: the ranked queue for the TAPNext++ NPU port

Earlier revisions: `git log --follow -p tapnext/research/PROPOSALS.md`, or `git show eed3870:npu-research/revN/<file>`. This is revision 21, 2026-10-11, research pass 21, the last pass before the NPU work pauses.

- **Base:** full12r24z, 28.43 ms at d12 (quiet sweep, MEASURED; r25z rebuilt byte-identical, 28.42); DAVIS d12 AJ 70.84 (OA 92.57),
  d10 69.90 (91.02), bit-identical to full12r9a.
- **This pass (NPU: two short research windows, no timeout):**
  - SSMCUT settled the SSM bound: compute-bound by ~255 cycles per chunk above an input floor of ~2,030;
  - CLKLOAD2 answered why CPU load slows the NPU;
  - EW L1 for the SSM glue found (~5.4-6.9 KB);
  - the out_proj BD peak broken down (main's P0FREE fixed it; one step more proposed);
  - DEAD_ENDS C8 revived, M1-M2 added; KTL CP-SAT stopped (no bank-clean R 68-69 found).

## Headlines (rev 21)

1. **SSMCUT (MEASURED)**: r25s (r24z + SSMIN) chunk 2,289, in_need 71; twin r25g (sigk + ssq removed, ~1,000 cycles) chunk 2,034,
   in_need 529. The SSM is compute-bound today (main's SSMIN) and has an input floor of ~2,030 cycles per chunk (7,851 B per column
   per chunk = 3.86 B / cycle: one shim MM2S channel, and ~56 GB/s over 8 columns, the read cap; the two coincide). Rev 20's
   "read-capped, glue ~0" was wrong (DEAD_ENDS M1). Pre-registration: H_C (1,150-1,350) and H_R (>= 2,050) both missed; the
   registered "in between" band held.
2. **CLKLOAD2 (MEASURED, 12 threads each)**: none 1.809 GHz; matmul AVX-512 1.408; matmul AVX2 1.408; 64 x 64 matmul in L1 1.584;
   DRAM copy 1.808 (42 W). Round 23: 12 spin threads (99 W) and 6 matmul threads keep 1.81. **Many threads of floating-point compute
   with the package at its ~100 W limit slow the NPU by ~22 %; memory traffic and scalar load do not; vector width does not matter.**
   For an app beside the NPU: cap BLAS / OpenMP threads (OMP_NUM_THREADS / OPENBLAS_NUM_THREADS <= 6) for CPU math that runs while the
   NPU does, or keep the package under ~95 W. Mechanism UNVERIFIED (SMU power vs current arbitration).
3. **EW L1 for the glue (~5 KB wanted):** stack 0x1300 -> 0x800 frees 2,816 B in bank 0 (deepest path 1,792 B; asm frames 64 B in ggk /
   convk / nrm_ln, none on that path); free tails 1,472 + 960 + 128 B; norm-only scratch (g_panels 1,024 + g_nacc 512) unioned with
   SSM-only buffers: 1,536 B. Total 5,376 B (6,912 with the union); the input ring at 16 slots would add 3,072 B (norms A/B first).
4. **BD headroom:** at out_proj shim 0 held the attention group's Q + 8 K / V fills until that group finished. Main's P0FREE (Q +
   pass-0 K / V in "#p0") took the static peak 20 -> 16 with CPPRE_OP. P1FREE (pass-1 K / V of head group g in "#h{g}") frees one more
   BD per part wait. Rule: a fill rides in the earliest drain group whose completion proves it was consumed.
5. **C8 revived** (0.55 us per post-wait fill, not 0.10); C3 / C2 / H14 stand.

## The queue at pause (base full12r24z 28.43 ms)

| rank | id | what | cost | price (ms) | gate / check |
|---|---|---|---|---|---|
| 1 | **SSM glue** | ssq2 + reck2 + ggk2 (chunk pairs), with EWSTACK's L1 | M | **-0.15..-0.25** | tcall per chunk; SSMIN chunk >= ~2,030 after (waits rise when it reaches the floor) |
| 2 | EWSTACK | EW `stack_size` 0x1300 -> 0x800 (+2,816 B bank 0); union norm-only scratch (+1,536 B) | S | 0 (enables 1) | aiecc measured requirement 1,792 B; regress bit-identical |
| 3 | MSPLIT-C up_a / qkv / mlp0 | the GEMM's first tile on rows 0-5 while the norm writes 6-17 (row-pass epilogues) | M | -0.05..-0.10 / -0.03..-0.08 each | TL_ONLY gaps 27.9 / 14.2 / 17.6 us |
| 4 | SSPM-B + KPAD + FSX | after 1: params read once (8 columns read the same 12,288 B per head) and unpadded; floor ~2,030 -> ~1,870; then FSX's compute can land | L | -0.05..-0.15 together | an SSMCUT twin after 1 |
| 5 | P1FREE | pass-1 K / V fills in "#h{g}" | S | ~0 (BD room for more CPPRE) | static peak <= 15 |
| 6 | KTL R 68 / 69 bank-clean | no schedule found (R 69 hard: 4 / 12 workers; R 68 soft: 35 conflicts plateau); needs a placement where A slots and accumulator pieces never share a bank | L | -0.2..-0.3 / -0.1..-0.15 | replay 0 stalls |
| 7 | input ring 16 slots | +3,072 B bank 3 if 1 needs more L1 | S | 0..+0.03 risk | norms A/B |
| 8 | k-step DMA collisions (~140 cycles per k-step, ~1.2 ms per step) | the largest unpriced item; blocked by GEMM L1 capacity (no idle bank) | — | — | DEAD_ENDS A13 / A24 / A25 / J6 |

**Ceiling of the queue:** 28.43 -> glue 28.23 -> MSPLIT-C 28.05 -> SSPM-B + FSX 27.95 -> KTL R 68 ~27.7 (EXTRAPOLATED).
**Floor:** ~25.2 ± 0.9 ms (EXTRAPOLATED; rev 20's ledger less round 24's boundary gains).

## Probes and tools (rev 21, `r26/`)

| tool | what |
|---|---|
| SSMCUT (`r24/ssmin.py` on try24 builds with `--ew-add -DSSMIN [--ew-add -DSSM_NO_GATES]`) | distance to the SSM's input floor; `ssmcut_r25s.out`, `ssmcut_r25g.out` |
| `r25/clkload2.sh` (+ `r26/clkload2.out`) | NPU clock under 12 threads of none / matmul AVX-512 / AVX2 / L1 matmul / DRAM copy |
| `design8_bd.py` (+ `bdop.log`) | design8 copy printing live shim-0 BD tasks per group at a threshold |
| `ew_r24z.dis`, `ew_r24z.stack`, `ew_funcs.txt` | EW core disassembly, stack sizes, function map |
| `cpgate.sh` | freezes the CP-SAT solvers during the NPU measurement windows |
| `r22/gen_e7.py` | `E7_DTIME`: a deterministic CP budget |
