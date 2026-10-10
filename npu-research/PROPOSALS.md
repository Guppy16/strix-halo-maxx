# PROPOSALS.md — ranked experiments and tools for the TAPNext++ NPU port

Revision 14, 2026-10-11, research pass 14 (one job: E1, make ROT buildable), during main's round 18 (best **full12r18h**:
r17d + HRING PTRG ZK KSC NIOC V2T CGF, median **38.40 ms** at depth 12, BIT-IDENTICAL; then r18j = + P35, A/B −0.10 median over
30 rounds; GEMM text 15,904 / 16,384). Revision 13
is in `rev13/` (its sections T, S, the norm and attention items stand unless named here), older ones in `rev12/` .. `rev1/`.
Prices: deltas at depth 12, Turbo, 1.81 GHz, whole-call (tcall static + the DMA-collision model, now calibrated on main's
round-18 KS-P38 counters). 1 cycle per GEMM k-step = 8.59 µs per step.

## Headlines (rev 14)
- **ROT is ready to build.** `r19/gen_rot.py` is one CP-SAT model: one row block of 8 jobs, with the slots, the itinerary ports,
  the .td latencies under the visibility rule, the register classes and old-value windows, the B stream's fill / pop / restart
  pattern, ZOL stages and the bank rule on the r17d placement. It emits mm_rot, and a value-level replay of the whole call
  checks it (setup, prologue, 18 trips, FIFO streams byte by byte, every MAC's operands, locks, records, final memory). It also
  fits the prologue so the `.p2align` before the body adds no pad bundle, asserted on the disassembly. asmlint passes with
  0 errors and 0 warnings.
- **Is II 9 feasible? Yes, but only with a gap-free B restart.** The binding resource is the B bank: per row block there are
  64 pops + 8 fills = 72 line reads, all in bank 3, plus the g bundles the B stream needs to restart, so R ≥ 72 + g. CP-SAT
  proves R = 72 + g OPTIMAL for each g:

  | g | restart mechanism | status | R | mm_rot whole call |
  |---|---|---|---|---|
  | 2 | `mov p1, p5` (deployed) | verified | 74 | 1,384 |
  | 1 | `padda [p1], m2` | UNVERIFIED | 73 | 1,368 |
  | 0 | `vldb.pop.576.2d` wrap in the last pop | UNVERIFIED | 72 | 1,351 |

  - Deployed mm_ks is 1,824. II 9 itself (one job per 9 cycles) needs g = 0.
  - Registers are not binding: ex holds A 8 + B 4 = 12, dm 4 of 5, MACs 64 per R.
  - Bank conflicts: 0 on the r17d placement and on four r18h-ROT placements (records in bank 0, acc_hi 0x71900..0x71A00),
    all solved together. mm_ks has 24.
  - With g > 0, identical job templates are INFEASIBLE (a structural proof, confirmed by CP). The exact cover of the B pass
    wraps each job's pops by R mod 9 and forces its 8 pops into consecutive cycles. A dependent chain (distance ≥ 3) cannot
    follow that with 4 B registers. Per-job issue cycles fix it, because the loop body is one row block.
- **Recommended build: g2f**, which uses only deployed mechanisms (plus the never-used `rel.cond`; see the risks).
  - mm_rot whole call: 1,384 static (tcall).
  - Text: 1,486 B (mm_ks is 1,824, so −338 B).
  - Records: 94 words (+36, +144 B of L1 as KSC constexpr data).
  - k-step price (Δ against the r18h k-step; P35 does not change it, it only touches the gap between calls):

    | model | k-step | Δ step |
    |---|---|---|
    | calibrated (KS-P38) | 1,676 | **−3.84 ms** |
    | uncalibrated collision model | 1,720 | −3.46 ms |

  - Pre-registered in situ: mm_rot ≈ 1,610 ± 45 per k-step under KS-P38 (MEMORY ~212, LOCK ~13). g0f would be −0.27 ms
    more once its probe passes (g1f −0.14).
- **Bit-exact by construction**:
  - Every accumulator (rb, cb) gets exactly the deployed operation sequence: load its 4 fp32 quarters, 8 `vmac.f` (config
    780) in k-panel order 0..7 with A(rb, kp) and B(cb, kp), store the quarters.
  - B is only permuted, in whole 72 B bfp panels (each 8-value block carries its own exponent).
  - vmac has no state across accumulators: no rounding-mode dependence for the fp32 accumulate, and srFPFlags is status only.
  - The replay checks every MAC's operands. The one-core bench `r19/rotb.py` compares all 36 pieces after 1 and 4 k-steps.
- **E2 and E3 are done (main, round 18)**:
  - DMAB: the core yields about one cycle per 32 B DMA write into the bank it is using (+58–59 vs the model's 65).
  - KS-P38: MEMORY_STALL 173.7, LOCK_STALL 13.3 per k-step (pre-registered 150–230 / ≤ 30: hit). The collision model holds
    at ×0.79–0.9.
  - Consequence for ROT's layout levers: B is popped every cycle, so the B prefetch collides about 1:1 (~110 cycles after
    calibration). There is no bank without core traffic to move it to (ROT busy: bank 0 0.24, 1 0.45, 2 0.37), so E6-B is void
    under ROT.

## ROT ready — build recipe (main, rank 1)

| item | file | what |
|---|---|---|
| schedule | `r19/rot_g2f.json` | CP solution (times per op, F, E, records order, B order, fit) |
| kernel | `r19/mm_rot_g2f.s` (+ `.dis`) | `extern "C" void mm_rot(const int32_t *rec /*p0*/, const bfp16ebs8 *pB /*p1*/, float *piece0 /*p2*/)`; mm_ks's signature, locks (acq.cond #53 / rel.cond #52), pieces |
| header | `r19/mm_rot_g2f.h` | `MM_ROT_NREC` 94, `mm_rot_order[][2]` (rb, field), `mm_rot_border[64]` (stream position → pair4 panel index) |
| C++ | `r19/mm8_rot.inc` | `rot_record()` (constexpr), `rot_kstep()`: KSC-style build-time records under `-DKSC` (same KS_C / KS_ACC_HI / KS_A0..2), else a run-time builder |
| bench | `r19/rotb.py`, `r19/rotb.cc` | one core, `ks` (deployed mm_kstep → mm_ks) vs `rot`; xclbins already built in `r19/build/rotb_g2f_{ks,rot}` |
| price | `r19/price.py` (`price.out`) | whole-call + collision model (raw and KS-P38-calibrated) |

Steps:
1. **Bench** (one dispatch each, short timeout):
   `docker run ... --device /dev/accel/accel0 npu-iron python3 /work/npu-research/r19/rotb.py run --v g2f --calls 1,4`.
   Rebuild with `rotb.py build --v g2f` (CPU only; keys the IRON cache on the object md5).
   - Pre-registered: **36/36 pieces identical after 1 and 4 k-steps**.
   - rot, --calls 4 (warm records): last call 1,390–1,405 cycles. That is mm_rot on the bench's placement, 1,383 static +
     0 bank stalls (`gen_rot.py check --place` with `rotb.py place`), plus rot_kstep's compare and call.
   - --calls 1 also times the cold record build.
   - ks: about 1,824 + glue + its own stalls (B sits in bank 1 in this bench), so it is not a deployment number.
   - Optional `-DROTB_LOCKS` reads both lock values after the calls (expect 0 / 0; the register offsets are unverified).
2. **mm8.cc**: `#ifdef MM_ROT #include "build/mm_rot.inc"`. Put it after `piece()` and `acc_hi`, i.e. after the KSC block.
   - **With P35 (r18j)**, the only call site is mm_ktile's loop: `mm_ks(ks_rec.r, b, pc);` → `mm_rot(rot_rec.r, b, pc);`. Both
     B buffers are in bank 3 as the schedule assumes, and mm_rot takes B by pointer, so P35's alternation works unchanged.
   - Without P35: in mm_kstep, `#ifdef MM_ROT  rot_kstep(a0, a1, a2, pB_in, c_out); #else` (the MM_KS / KSC path) `#endif`.
   - Keep `-DMM_PIECES` (pieces unchanged) and `-DKSC` (records become .rodata).
   - mm_ks and its records drop out: text ≈ 15,904 − 1,824 + 1,486 ≈ **15,566**.
   - L1 data: rot_rec 376 B vs ks_rec 232 B, so **+144 B**. Check that the KSC post-link check also covers rot_rec.
3. **design8**: `rot = "-DMM_ROT" in flags`. Copy `mm_rot_g2f.s / .h / mm8_rot.inc` into `build/` as `mm_rot.s / .h` and
   `mm_rot.inc` (or run `gen_rot.py fit` there: no ortools needed, Peano only). Then assemble, and merge the object like
   mm_ks.o. Write the B order (`border` from the JSON) into `build/<name>.json`.
4. **pack8 B_ROT** (host): in `w_tile_words`, after the pair4 transpose:
   `v = np.ascontiguousarray(v).reshape(64, 72)[B_ROT]`, with `B_ROT = [pair4 index of each stream position]`
   (= `mm_rot_border`).
   - run8 / step8 set `P.B_ROT` from the build json when its kdefs contain `-DMM_ROT`, as they do for B_PAIR4.
   - Only pack_gemm's weight tiles change; the EW parameter tiles keep their order.
5. Lint (`asmlint --prj`), then **one in-situ dispatch** with a short timeout, then `step8 regress` BIT-IDENTICAL vs
   full12r9a, then A/B vs r18h.
   - Pre-registered: **−3.84 ± 0.4 ms** (−3.46 if the uncalibrated collision model is right).
   - KS-P38 build: TRUE per k-step in mm_rot ≈ 1,610 ± 45, MEMORY_STALL ≈ 212 ± 40, LOCK ≈ 13. Between calls 67 unchanged.
   - The A port needs 10,368 / 1,610 = 6.4 B/cycle. DMAB measured 7.3–7.45 under collisions, so A does not bind.
6. Risks:
   - **(a) `rel.cond #52, r5, r26`.** It is in the .td (the same family as the verified `acq.cond`) but no deployed code uses
     it. A misfire releases 18 slots per k-step instead of 9: corruption or a hang in situ, nothing in the bench without
     ROTB_LOCKS.
   - **(b) The prologue's B stream starts mid-pass** at B + 576 i0 (`movxm m3, #2880 / padda [p1], m3` in setup). This is a
     fresh stream at a 64 B line boundary, the standard start.
   - **(c)** The schedule is conflict-free for the r17d / r18h placement (0x78900 C, 0x718C0 acc_hi, slots 0x77B80 / 0x78000 /
     0x78480, B 0x7D100 / 0x7E300, records 0x71800 or 0x778C8, acc_hi 0x718C0..0x71A00). If KSC's check reports a moved
     layout, re-run `gen_rot.py check --place` for the new addresses. Correctness is placement-independent; only stalls
     change (a re-solve with `--place` takes ~25 s).
7. **After E5, faster variants** (no rebuild of the method, just `--v`):
   - g0f (1,351, −0.28 ms more) after probe PWRAP.
   - g1f (1,368, −0.14 ms more) after probe PADDA.
   - Both are fitted, replayed, linted and priced in r19/.

**Probes for the faster variants** (popprobe style, one core, after npu-tapnext r16/popprobe.s):
- **PWRAP:** two passes over a 4,608 B B object (8 × (fill + 8 pops)). The 64th pop is
  `vldb.pop.576.2d exN, [p1, lf1, r25, d2]`, with m2 = −4544, dn2 = 0, dj2 = 64, dc2 = 0 and `mov r25, #0` in the same bundle.
  The next fill follows in the next bundle. Store every popped panel:
  - pass 2 must equal pass 1;
  - variant m2 = −4608 in case the 2D step adds to the line step;
  - pre-registered: m2 = −4544 works.
- **PADDA:** the same with a plain last pop, `padda [p1], m2` (m2 = −4608) one bundle later, and the fill one bundle after
  that. Pre-registered: pass 2 = pass 1.

## Ranking (rev 14; base full12r18h, median 38.40 ms)

| rank | id | what | cost | risk | expected (depth 12) | pre-registered check |
|---|---|---|---|---|---|---|
| 1 | **ROT (E5, g2f)** | the recipe above | M (all files exist) | medium (rel.cond, new B order) | **−3.84 ± 0.4** (−3.46 uncalibrated) | bench 36/36 identical, rot 1,390–1,405 per warm call; regress BIT-IDENTICAL; KS-P38 TRUE in mm_rot ~1,610 ± 45 |
| done | P35 (main r18j) | mm_ktile: a tile's k-steps in one call, B locks by id | | | measured −0.10 (priced −0.44: miss, the gap is mostly not IRON's control code) | ROT plugs into mm_ktile (step 2) |
| 3 | PWRAP / PADDA probes, then g0f / g1f | probes above, then `--v g0f` | S | low | −0.28 / −0.14 on top of g2f | probe pass 2 = pass 1 |
| 4 | E7' (B over two banks, ROT) | split each B object over two banks (half the pops each): bank bound 36 + g per row block, so the MAC bound (64) decides; re-solve gen_rot with a second B stream | L | high (L1: 4.6 KB more in another bank) | static −10..−15 % of mm_rot (~−1.2 ms) + fewer B collisions | gen_rot solve with the split first (research) |
| 5 | A slots into ROT's least-busy bank (E6') | ROT bank busy: b0 0.24, b1 0.45, b2 0.37; A DMA collisions ~0.79 × 324 × 0.40 = 102 per k-step; slots in bank 0 → ~62 | M (L1 moves) | medium | ~−0.34 | re-solve with `--place`; KS-P38 MEMORY −40 |
| — | rev 13 items not touched here | N-IO rest (NIOC done), NS, attention S3 / V1 / K1 / K2 (text: ROT frees 338 B more), ssq two chunks, P55 / P51 / P57 | | | as rev 13 | |

Ceiling from 38.40: P35 −0.10 (done) → ~38.3; ROT −3.84 → ~34.5; g0f −0.28 → ~34.2; rev 13's remaining compute levers (attention
~−0.5, NS −0.34, ssq −0.11, P55 / P51 / P57 ~−1.8 at rev 8-10 prices) → ~31.1; E7' and the A layout ~−1.5 more → ~29.6. The A port
floor (1,304 per k-step at 7.95 B/cycle, ~1,420 at the 7.3 B/cycle DMAB measured under collisions) is then ~200 cycles away
from mm_rot's in-situ ~1,610. (Totals below 34 are rev 8-13 prices stacked: ±1 ms.)

## Status of the revision-13 k-step plan

| # | rev 13 | status (rev 14) |
|---|---|---|
| E1 | ROT CP feasibility (research) | **done** (this pass): II 9 with g = 0, R = 74 optimal with the deployed restart; kernels, bench, recipe |
| E2 | DMAB bench | **done** (main r18): the core yields ~1 cycle per 32 B DMA write in its bank (+58–59 vs 65) |
| E3 | KS-P38 counters | **done** (main r18): MEMORY 173.7, LOCK 13.3 per k-step (hit) |
| E4 | P35 | **done** (main r18j): −0.10 (miss, −0.44 priced) |
| E5 | ROT | **ready** (rank 1) |
| E6 | layout: B prefetch out of the popped bank | **void under ROT** (B is popped every cycle and no bank is idle); A slots: rank 5 |
| E7 | B over two banks | rank 4, needs a gen_rot re-solve with two B streams first |

## Tools (rev 14)
- `r19/gen_rot.py`:
  - `solve`: CP-SAT, `--free` gives per-job cycles, `--place` (repeatable) keeps the schedule conflict-free on several
    placements.
  - `fit`: emit + prologue alignment, then assemble in npu-iron and replay.
  - `check`: whole-call replay; `--place` gives correctness and the stalls on any L1 placement.
  - The replay is placement-independent for correctness. It counts bank stalls, per-bank busy fractions, MACs and stores.
- `r19/build_all.sh [--solve] g2f g1f g0f`: solve, fit, link (ld.lld so tcall sees the ZOL), lint, tcall, and the replay on
  the five placements.
- `r19/price.py`: tcall + per-bank busy → ksplan's collision model, raw and KS-P38-calibrated (`price.out`).
- `r19/rotb.py` + `rotb.cc` (`build` / `run` / `place`): one-core bench. `mm8_rot.inc`: the C++ side for mm8.cc.
- Kept from rev 13: `tools/ksplan.py` (variants table), `tools/tcall.py`, `tools/asmlint.py` (all four mm_rot variants: 0
  errors, 0 warnings).
