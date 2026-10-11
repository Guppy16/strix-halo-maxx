# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 18, 2026-10-11, research pass 18, written during main's round 22 (KTL integration in flight).

- **Base:** full12r21k1, 31.30 ms at d12 (quiet sweep, MEASURED), + 3A (−0.08, round 22) ≈ **31.22**.
- **Data:**
  - Timeline `tl_full12r21k1_d12.json` (12 blocks, per-op gap / span means; `r21/tlops21.py`).
  - TPROF `build/r21/tp_r21t2.npz` (`r20/tpgap.py`: first-k-step times after the previous GEMM end).
  - The deployed EW core (`r23/full12r21k1_0_5.dis`, tcall).
  - The EW C++ compiled with the deployed flags (`r23/ewsize.sh`).
  - r21 `ddrbudget` bytes.
- **Archive:** revision 17 is in `rev17/`.
- **Prices:** depth 12, 1.81 GHz.
  - One cycle per GEMM k-step = 8.25 µs per step.
  - One cycle per SSM unit (18 units × 12 heads per block) = 1.43 µs per step.
  - One cycle per norm unit (216 per column per op) = 1.43 µs per step per norm.
- **CPU:** CP-SAT none this pass; the EW compiles ran at nice 10, single-threaded.

## Headlines (rev 18)

1. **The time ledger at r21k1 (ms per step).** Floors are INFERRED from measured rates; levers are priced below.

   | part | now | floor | what bounds it | levers |
   |---|---|---|---|---|
   | GEMM k-steps | 12.74 (1,545 cycles in situ × 14,930) | ~11.0–11.9 | A port 1,304 cycles per k-step + ~150 of DMA collisions (no free bank: capacity) | KTL −0.6, E7'' R 66 −0.2 |
   | GEMM epilogues + C hands | ~1.8 | ~1.7 | GELU by LUT (mlp0 gelu_pan 5,495 cycles per tile), CSUB's drain collisions | none above 0.1 |
   | GEMM-side boundaries (paces, GEMM↔EW sandwiches, attention edges) | ~2.4 | ~1.0 | producer tail (last tile at the write cap) + the norm's whole-row dependency + the consumer's k-outer first tile | **MSPLIT** −0.35..−0.6 (P) + −0.15..−0.3 (C), OPPRE-lag |
   | attention | 6.13 | ~5.9 | its own compute (pass 0 ~73 µs, pass 1 ~40 µs per head group) | S3 −0.05..−0.1 |
   | norms | 4.52 | ~4.35 | DDR read cap (norm0 / 2 / 1); norm3 (LN, 2 inputs) compute (693 vs ~567 cycles per unit) | norm3 LN glue ≤ −0.18 |
   | SSM | 3.59 | **2.86** | compute: ~2,340 cycles per unit in situ (static 2,115, of which **~750 is call glue**) vs the DDR floor 2,010 | **SSM glue family** −0.25..−0.45 (needs EW text: TXT18) |
   | **total** | **31.2** | **~27.0 ± 0.7** | | |

2. **EW text: TXT18 frees 400 B (`r23/mk_txt18.py`, measured with the deployed flags).** EW C++ goes 11,056 → 10,656 B; the elf goes
   16,272 → ~15,872 (512 B free).
   - Function pointers for the ln / rms kernel twins (one call site each).
   - A 2-trip statistics loop.
   - One wrap-aware ring copy for in_copy / out_copy.
   - Same calls in the same order: bit-identical by construction. That is enough for ssq2 + FSX.
3. **SSM is compute-bound, by glue.**
   - tcall: one SSM unit is 1,828 static (sigk at 1 trip; ~2,115 with its real trips), of which **740–757 is glue**. The pieces:
     ssq 194, reck 92, ggk 90, convk 68, sigk 45, zk 34, and the C++ loop.
   - The DDR floor is 2,010 cycles per unit (13.6 MB per block at 57 GB/s), so up to ~330 cycles per unit can land: −0.47 ms at most.
4. **New structural lever: MSPLIT (row-half tiles at the EW boundaries).**
   - The five GEMM ↔ EW sandwiches cost ~83 µs per block over the EW spans (TPROF k0 times minus the EW span, minus first-unit time).
     Each one is the producer's last tile and its write-cap tail, plus a norm / SSM that needs whole rows.
   - Producer side: the producer's last tile run as two row halves (rb 0–9, 10–17) with separate drain groups. The norm starts on
     rows 0–9 while half 1 computes. This saves half a tile + half the tail per boundary: ~68 µs per block raw.
   - Consumer side: the consumer's first tile on rows 0–9 as soon as the norm passes row 9 (~27 µs raw).
5. **Refuted by model this pass (DEAD_ENDS I-section):**
   - Residual add / K-split sum in the GEMM epilogue (RADD / KSUM): a GEMM core has no third input port.
   - yx → SSM on chip (YRES) and SSM params multicast (SSPM): they save DDR, but the SSM is compute-bound.
   - GELU moved to the idle EW cores: the DDR during mlp0 is already ~42 GB/s.
   - mlp3 as one K = 3072 op: a 497 KB A slot.
6. **CLKLOAD is designed, not run:** `r23/clkload.sh` (needs the NPU and a quiet box). Package power (amdgpu hwmon `PPT`, 28.0 W at
   load ~9) and Tctl (78 °C) are readable without sudo, so power, thermal and DPM can be told apart.

## Ranking (rev 18; base 31.22)

| rank | id | what | cost | static / whole-call price | expected (d12) | separating probe / pre-registered check |
|---|---|---|---|---|---|---|
| 1 | **KTL** (main, in flight) | mm_e7k: one loop per tile | M | 76 + 1,260 × nks per tile vs 1,351 × nks | −0.50..−0.70 | rev 17 checks |
| 2 | **TXT18** | EW text: kernel twins by pointer, stats loop, shared ring copy (`r23/mk_txt18.py`, -DTXT18) | S | EW C++ −400 B (11,056 → 10,656, MEASURED); per unit +0..2 cycles (indirect call, one select) | 0 ± 0.02 (text lever) | BIT-IDENTICAL; elf text ≤ 15,890 |
| 3 | **SSM glue** (needs 2) | ssq2 (2 chunks per call: −97 per unit), FSX (sigk folded into ssq: −45 glue − the sigk pass), reck2 / ggk2 (2 chunks per call: −46 / −45), one C++ loop trip per 2 units | M | per-unit static 2,115 → ≤ 1,880 (tcall) | **−0.25..−0.45** (DDR floor caps it at −0.47) | tcall per-unit static before any build; in situ SSM span 283 → 245–255 µs per block |
| 4 | **MSPLIT-P** | the last tile of out / down_b / out_proj / mlp3_b / yx as two row halves with separate drain groups; norm / SSM deps on half 0 | L | −(half tile + half tail): 16.7 µs (K 1536) / 11.5 µs (K 768) per boundary, 68 µs per block raw | **−0.35..−0.6** | **NTWIN first** (below); then norm0 / norm2 first output 10–17 µs earlier relative to the producer's last k-step (TPROF + timeline) |
| 5 | E7'' + KTL R 66 | aligned + KTL schedules at lower R | S (CP) | −36..−72 static per k-step | −0.15..−0.3 | replay 0 stalls |
| 6 | MSPLIT-C | the consumer's first tile on rows 0–9 (NRES chunk locks per half) | L | −half tile (5.5 µs) per boundary, 27 µs per block raw | −0.15..−0.3 | after MSPLIT-P |
| 7 | norm3 LN | LN pass-2 / NPIPE glue for the 2-input LN norm (the only compute-bound norm) | M | 693 → ~600 cycles per unit | −0.1..−0.18 | NIO2 stamps on norm3 first |
| 8 | OPPRE-W → OPPRE-lag | rev 17 | S | — | −0.04..−0.10 | rev 17 |

**Ceiling:** 31.22 → KTL 30.6 → SSM glue 30.25 → MSPLIT-P 29.8 → R 66 29.6 → MSPLIT-C 29.4 → norm3 + OPPRE-lag **~29.2**
(EXTRAPOLATED).

**Floor ≈ 27.0 ± 0.7.** It keeps the A port + collisions in the k-step, GELU in the epilogues, the attention's compute, the norms' DDR cap and
the SSM's DDR floor.

**What lies between 29.2 and 27.0:**
- k-steps ~0.9 ms: DMA collisions with no free bank (capacity, DEAD_ENDS A13 / A25).
- GEMM-side boundaries ~0.6: the halves' tails and handoffs.
- SSM ~0.4: glue beyond the four kernel changes.
- Attention ~0.2.
- No priced lever closes these.

## TXT18 — the patch (rank 2)

- `python3 npu-research/r23/mk_txt18.py npu-tapnext/ew8.cc npu-tapnext/ew8.cc.new`: everything is behind `#ifdef TXT18`; without it,
  the source compiles to the same 11,056 B (checked).
- **A.** `norm_rows_k` picks `n1k_{ln,rms}_{begin,step,end}` and `nrm_{ln,rms}` once per op as function pointers.
  - nrm_rms is called through nrm_ln's 6-pointer type, with p4 = xdst (its 5th pointer argument) and p5 unused.
  - norm_rows_k 3,504 → 3,296 B.
- **B.** The per-row-block statistics as a 2-trip loop over the halves: same ln_stats / rms_rstd calls, same order.
- **C.** `in_copy` / `out_copy` share one `wrap_copy(ring, ring_b, p, lin, n, to_ring)` (minsize): the same copy_run calls in the same
  order.
  - out_copy runs per unit only on a ring wrap or for mode 3 (ln_out): +~10 cycles per call there (~0.01 ms per step).
- **Measured:** `r23/txt18.out`, via `npu-research/r23/ewsize.sh [SRC] [-D..]` (any EW variant, deployed flags, per-function sizes).

## SSM glue family — design notes (rank 3)

- **Today's unit:** in_need 22–35, convk 228 (glue 68), ggk 282 (90), sigk 85 at 1 trip (glue 45), ssq 674 (194), reck 268 (92),
  zk 74 (34), in_done 33, C++ ~180 (tcall `--func ew_ssm`, `r23/full12r21k1_0_5.dis`).
- **ssq2:** two units per ssq call (prologue / drain once per two): −97 per unit. Needs double g_ga / g_tsp buffers in EW L1: check the
  EW free L1 first (stack deepest ~3.2 KB of 4.75 KB).
- **FSX:** gate_a's sigmoid inside ssq. The x pressure held it at II 30 (DEAD_ENDS E16); price only after a CP schedule fits 12 x
  registers.
- **reck2 / ggk2:** two units per call (−46 / −45 per unit). These are the cheapest after ssq2.
- **Cap:** 2,010 cycles per unit (the SSM's DDR read floor). Beyond ~−0.47 ms nothing lands.

## MSPLIT — design and the separating probe (ranks 4 / 6)

- **Producer half tile (P):**
  - The producer's last N tile per core runs as two passes: rows 0–9 (stripes 0–4) and rows 10–17 (stripes 5–8), all k-steps each.
  - **A:** the memtile sends per k-step [0, 5,760) then [5,760, 10,368) of each chunk; one 2D send TAP per half (chunks are rb-major).
  - **Kernel:** the KTL kernel with a 10- / 8-row-block record table and trips = 10 / 8 × nks (records are data; text ~0).
  - **Epilogue:** the epilogue and the C drain of a half (rows 0–79 = C bytes [0, 10,240); CSUB's sub-part split moves to 10,240).
  - **Deps:** the EW consumer depends on the half-0 drain group for row blocks 0–9 and on half 1 for the rest. The norm's input fills
    split at rb 10 (its fills are rb-major per column).
- **Consumer half tile (C):** NRES scatters the norm's panels chunk-major. Per-half chunk locks (rows 0–9 / 10–17 of each chunk), and the
  consumer's first tile in two row halves.
- **NTWIN (S, separating probe):**
  - A timing twin where norm0 and norm2 skip the wait on the producer's last drain group (data wrong).
  - Read per-op walls from `step8.py timeline`, never the step delta: a wrong residual changes the attention's lazy-max work (H17).
  - Pre-registered: if the producer side is real, norm0 / norm2 start 20–33 µs earlier relative to the producer's last k-step and
    their walls do not grow. If the norms' walls grow by the same amount (DDR contention with the producer's last tile), MSPLIT-P is
    worth ≤ −0.15: re-rank.

## Probes

| id | what | separates / gates | pre-registered |
|---|---|---|---|
| **NTWIN** | norm0 / norm2 skip the producer's last drain group (twin) | MSPLIT-P's overlap is real (not DDR-bound) | norms start 20–33 µs earlier, walls unchanged → MSPLIT-P −0.35..−0.6; walls grow → ≤ −0.15 |
| **SSM-tcall** | tcall per-unit static of each SSM variant before building | the SSM glue price | ≤ 1,880 static per unit for the four changes |
| **CLKLOAD** | `r23/clkload.sh`: r7/clk.py vs 0–20 threads of scalar spin vs numpy matmul, logging amdgpu PPT and k10temp Tctl (quiet box only; main runs it) | power budget vs thermal vs DPM | power: the clock tracks PPT (vector load drops it at fewer threads); thermal: it tracks Tctl with a lag; DPM: it tracks neither |
| OPPRE-W | rev 17 | H_CP vs H_SHIM | rev 17 |

## Tools (rev 18, `r23/`)

| tool | what |
|---|---|
| `ewsize.sh` | compile any ew8.cc variant with the deployed EW flags (CPU, npu-iron): total C++ text + the largest functions; `DIS=path` also writes the disassembly |
| `mk_txt18.py` | the TXT18 patch generator (A / B / C behind -DTXT18), `txt18.out` its measurement |
| `clkload.sh` | the CLKLOAD probe (not run) |
| `full12r21k1_0_5.dis` | the deployed EW core (tcall: ew_ssm per-unit static and glue) |
