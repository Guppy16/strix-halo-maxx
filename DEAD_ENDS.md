# DEAD_ENDS.md — ideas tried and refuted on the TAPNext++ NPU port (check before proposing)

Last updated 2026-10-11 (research pass 20; 184 entries + section R). Each entry: **ID — name** (round / rev), what it was; measured result; why it failed; lesson. Tags: [MEAS] measured on the device, [MODEL] refuted by a model or CP proof, [METHOD] a methodological dead end. "Conditional" = dead under today's design, may revive if X changes (say what X is).

Source keys: RFM = `npu-research/RESULTS_FROM_MAIN.md` (round N section); NOTES / RESULTS / HANDOFF = `npu-tapnext/*.md` (§ = section number); revN/P, revN/A = `npu-research/revN/PROPOSALS.md` / `ANALYSIS.md` (rev 16 = the top-level files); MACHINE Rx.y = `npu-research/MACHINE.md` revision additions; SKILL = `~/.claude/skills/xdna2-npu/SKILL.md` (laptop); MD = `.scratch/npu-moondream/{NOTES,RESULTS}.md` (pre-TAPNext, same hardware). ms = depth-12 step delta, Turbo, 1.81 GHz, unless stated. Negative = faster.

Not listed: proposals that were built and hit or merely missed their price but were kept (ZHI, CSPLIT, EPLR, P60g, ROT g2a, NRES, …), unless their stated mechanism was refuted.

## Index
| # | id | area | verdict (one line) |
|---|---|---|---|
| A1 | A-PORT (rev 1) | GEMM | k-step "on the 4 B/cycle A-port floor": twins say compute-bound (no-MAC −16.1, 2x-MAC +24.7 ms) |
| A2 | P6 / P40 | GEMM | two A streams per core: memtile→core port not binding (7.95 B/cycle); conditional on a k-step needing > ~7.5 B/cycle |
| A3 | A-PORT-R9 | GEMM | round 9 "A-port-bound at ~4 B/cycle" was a mean incl. the op-start wait; medians 2,050–2,065, A ≥ 5.04 B/cycle |
| A4 | P17 | GEMM | bank placement for the C++ mm_acc: 333 in situ = 321 static + stamps, no stalls to remove |
| A5 | P21-RULES | GEMM | scheduling rules vs bank conflicts: R1 273 / A-off-B 255 bundles vs 225; serialising costs what stalling costs |
| A6 | CPP-4CHAIN | GEMM | C++ 4-chain GEMM loops under Peano: 42–61 bundles per 16 mmuls vs 28 |
| A7 | P18-EXTRAP | GEMM | first asm schedule: −7 ms extrapolated, −1.9 measured (L1 bank conflicts) |
| A8 | PIECEMAP-GREEDY | GEMM | bank-aware runtime piece map: k-step −10 cycles/stripe, epilogue +0.8 µs/tile; affine map won |
| A9 | A_BANKS-PIECES | GEMM | A-slot bank pinning together with pieces: does not link (bank sections overflow) |
| A10 | LAYER-II | GEMM | P36 / P39 / P41 deep-K visits, neighbour A, 4th GEMM row: parked (L1 / EW blockers), "0 ms today" |
| A11 | ROT-TEMPLATES | GEMM | identical ROT job templates for g > 0: infeasible (CP proof) |
| A12 | E6-B | GEMM | B prefetch out of the popped bank: void under ROT (B bank busy 0.97, no idle bank) |
| A13 | B-2BANK | GEMM | B objects in two banks: needs 23 KB k-step-cold data, core has 14.6 KB |
| A14 | E7' | GEMM | B over two banks with ONE FIFO: void (single lf1 FIFO + ldb slot bind at 72 + g); E7'' is the live successor |
| A15 | ASLOT | GEMM | A slots into the least-busy bank: ≤ 15 ± 10 cycles (~0.1 ms) and no bank-0 room; dropped |
| A16 | ACC-OUT | GEMM | accumulators out of the C object: +12,288 B needed, 576 B free |
| A17 | g2f | GEMM | ROT with acq.cond 1 bundle before the A fill: raced in situ; bench passed (pre-released locks) |
| A18 | STATIC-PRICE | GEMM | k-step static cuts priced 1:1 (g0f −0.28, g1f −0.14): ~0.75 lands (g0a −0.18, g1a −0.11) |
| A19 | g1a | GEMM | padda restart variant: −0.11, superseded by g0a −0.18 |
| A20 | PWRAP-4544 | GEMM | pre-registered wrap offset −4544: pass 2 wrong; −4608 (2D step adds to the line step) |
| A21 | P35-PREMISE | GEMM | "67-cycle gap between k-step calls is code": −0.10 vs −0.44; gap mostly waits (kept) |
| A22 | MD-KLOOP | GEMM (moondream) | C++ k-loop variants (pipeline hints, unroll, rho 2, fp32 C prefetch): all slower or no gain |
| A23 | MD-HANDSCHED | GEMM (moondream) | early hand schedule 1.83 correct / 1.32 broken bundles per MAC: "not worth more" |
| B1 | P24 | tiles | "C handoff 3–6 µs per tile": measured 1.31 + acc_zero 0.49 µs; sim10's 7.6 µs held op-start waits |
| B2 | GELU_II | epilogue | forced-II GELU "54.7 cycles, slower in situ": Peano never pipelined it; with separate restrict params 17.8 (−0.88 ms) |
| B3 | P45-H1..H4 | epilogue | hazard hypotheses H1–H4 for the forced-II result: none held (wrong premise) |
| B4 | GELU-FLOOR8 | epilogue | GELU floor 8 cycles / 32 lanes (slots only): wrong, shared ports give ResMII 10–11 |
| B5 | GELU-II10 | epilogue | standalone GELU at II 10 (no 8-register schedule); bgelu II 11 (padding check) |
| B6 | RECIP-FREE-GELU | epilogue | reciprocal-free / polynomial GELU: 26–37 % bf16 mismatches; not built |
| B7 | P43 / P43b | tiles | B-fill hoist for "B-starved" first tiles: −0.35 vs −0.8; boundary is the A slot load; P43b retired |
| B8 | CSPLIT4 | C path | progressive part-1 release: re-priced −0.3..−0.35 → −0.11..−0.17 (lite); not built; conditional |
| B9 | LATE-C | C path | conditional C acquire inside mm_rot: −0.3 → −0.08..−0.12 after EPLR; not built; conditional |
| B10 | DUALDRAIN | C path | C drains over two shim S2MM per column: 0 (the write cap is aggregate) |
| B11 | MRES | C path | GEMM → GEMM APAN residency in memtile: 124 KB park vs 112 KB free; infeasible |
| B12 | MD-GELU-STAGED | epilogue (moondream) | staged / unrolled GELU chains: rolled slower (1177 vs 1108 µs), unrolled no gain |
| C1 | ROWCHUNK | boundaries | overlap EW and GEMM by row chunks: 76.3 → 76.9 / 77.2 ms (model) |
| C2 | CPBND-FIT | boundaries | rev-15 CP-instruction boundary model (t_op 0.17 µs × 92–144 instr.): fit not causal |
| C3 | CPRATE-CAUSE | boundaries | CP rate as the cause of boundaries: rate real but small (write32 0.06, maskpoll 0.11 µs) |
| C4 | SEQPRE | boundaries | pre-issue independent work before the wait: −1.4 ± 0.5 predicted, best −0.04 |
| C5 | SEQPRE-v1 | boundaries | memtile sends (+ loads + drains) before the wait: +1.10 ms (4-deep queue blocks the CP) |
| C6 | SEQPRE-v2 | boundaries | sends after the fills, loads + drains before: +0.13 ms |
| C7 | DRAINS-FIRST | boundaries | only the drains before the wait: +0.15 ms, cause unexplained |
| C8 | BDPRE | boundaries | hoist post-sync BD writes: BD write + patch ~0.10 µs ⇒ ≤ 0.1 ms; dropped |
| C9 | SENDPRE-dep | boundaries | memtile sends before the wait for dependent ops: ~0 (first tile DDR-paced anyway); dropped |
| C10 | SENDPRE | boundaries | memtile sends before the wait after EW ops: +0.16 ms |
| C11 | OPPRE | boundaries | out_proj's A in 4 parts during attention pass 1: +0.18 ms, cause unmeasured |
| C12 | DDR84 | DDR | "≥ 84 GB/s DDR": aggregate read cap ~55–59 GB/s (DDRCAP) |
| C13 | NIN2 | DDR / norms | second EW input channel: void, 3-input norms already at the aggregate read cap |
| C14 | H_WR | DDR | per-channel shim write limit ~4 B/cycle as the 12 µs tail: one stream writes ~6.7 B/cycle |
| C15 | H_TOK | DDR | ~6 µs token / CP latency as the tail: refuted; tail = 442 KB at the aggregate write cap |
| C16 | DDRW-2ch | DDR | per-channel vs per-column write probe: moot (aggregate cap) |
| C17 | NRES-RBMAJOR | boundaries | rb-major A-slot half for NRES: G chunks 31 / 39 / 47 wrong (chunk-wise release) |
| C18 | NRES-ALONE | boundaries | NRES norm1 without NSEQ: −0.06..−0.11 vs −0.15..−0.28 (column stagger) |
| C19 | P60-1TAP | attention data | single-TAP round-robin query deal: impossible except d = 0 (exhaustive + identity) |
| C20 | GATHER-MULTITAP | attention data | 2+ TAPs per gather / BD-chain deal: pieces land in the wrong A_RT chunk |
| C21 | KV-MEMBUF | attention data | per-column K / V memtile buffering: no free S2MM channel on memtiles 0–2 |
| C22 | P60t | attention data | host tile permutation: −0.66 alone, composed with P60g −0.76 (worse than P60g) |
| C23 | P60h | attention data | adaptive deal from last frame's flags: ≤ −0.33 beyond P60g; parked |
| D1 | P12-RESTREAM | attention | re-stream K / V per pass: twins say pass compute dominates; not done |
| D2 | ATT-DDR | attention | attention DDR-bound: no K-phase compute → 43.4 ms (from 50.05): compute-bound |
| D3 | EAGER-RESCALE | attention | eager online-softmax rescale: cost as much as the exp; lazy TAU 8 instead |
| D4 | SM-BF16 | attention | rounding s − m to bf16 before exp: up to 2–3 % error at |s − m| ~ 10 |
| D5 | EXP-TABLE-RANGE | attention | exp table h = 1/2 or range ending ≥ −7: worse on real scores |
| D6 | EXP-CARRY | attention | carrying looked-up exp lines across trips: 65 bundles with spills (vs 49) |
| D7 | PV-POP | attention | P popped in chunk_v: 67 vs 42 bundles per pair |
| D8 | SPLIT-QK-EXP | attention | separate QK / exp asm loops: aexp no II ≤ 28; superseded by the fused loop |
| D9 | Q-IN-EX | attention | fused K loop holding Q in ex0–7: garbage (ex_n aliases x_n) |
| D10 | P54-BIASBASE | attention | "biased base" form of vldb.4x64: does not exist |
| D11 | P56-FALLBACK | attention | fused loop priced −3.2: production −0.84 (exact test everywhere, chunk 22, lockstep) |
| D12 | P63 / 3D-POPS | attention | batched K loop "needs 3D pops": 2D modifiers suffice; gain eaten by the fallback (−0.04..−0.47 vs −1.25) |
| D13 | CHUNK0-SKIP | attention | chunk-0 rescale skip "no measurable gain" (r13) — later ~−0.3 (r14): sources disagree |
| D14 | EXACT-TEST-CHEAPER | attention | cheaper exact test: < 0.1 ms left once tested only on flagged units; retired |
| D15 | FLAG-RARER | attention | make the lazy-max flag rarer without numerics: flag-not-move 0.055 %; retired |
| D16 | DEFER-SLOW | attention | defer slow rows outside the lockstep: no L1 for V copies, moot after P60; retired |
| D17 | KV-SLOTS | attention | more L1 K / V slots: 4.6 KB each, moot after P60; retired |
| D18 | AEXP-ASM | attention | asm aexp: superseded (attk with the new max computes the same P, −464 B) |
| D19 | P64 | attention | 3,456 B K / V objects: −0.04 (producer not binding); parked |
| D20 | L2D12 | attention | --l2-depth 12: 49.58 vs 49.54 ms at 8, no more |
| D21 | ATT-TEXT | attention | K1C, rolled rescale_rb, 2-stage attkb, kernel-only attv, cpy32 Q copy: text / schedule dead ends |
| D22 | MD-STREAMS | attention (moondream) | 3 K / 2 V streams, memtile K/V broadcast, re-seek, stage-major exp, native exp2 |
| E1 | OUT64 | norms | noinline per-64 B ring helper: norms 3x slower than their DMA bound |
| E2 | ROUT235 | norms | "norms output-port bound, R_out 2.35": out_begin wait ~5 net vs 147 / 90 predicted |
| E3 | NPIPE | norms | pass-2 / pass-1 interleave: G2 +0.07, G4–6 −0.05..−0.15 vs −0.55 |
| E4 | NG / N-IO-ASM / NS | norms | norm glue / stats in asm: retired, revived, then capped ≤ 0.2 ms (LN 2-input only) by the DDR cap; conditional |
| E5 | RIN575 | norms | fitted R_in 5.75 (and rev 15's "one shim channel at 3.8"): both refuted; aggregate DDR cap |
| E6 | FOLD | norms | n_in 3 → 2 by summing upstream: retired at −0.03 under a refuted fit; conditional |
| E7 | EWO / OSLOT | norms | output-slot bench / bigger slots: retired, R_out not binding |
| E8 | STORE-STALL | SSM | "~3.5 cycles per store into the EW output ring": per-call cost, no stall |
| E9 | NORM_SPLIT | norms | statistics as a second pass: 0 ms |
| E10 | NRMK-COPIES | norms | X' / panel copies into the output ring: +0.6 ms |
| E11 | N1-PRICE | norms | norm pass 1 + stats in asm priced −0.6 ± 0.25: −0.16..−0.28 (glue + input waits stayed) |
| E12 | AL1-1K | EW I/O | params padded to 1 KB: hang; 3 KB: −0.05..−0.15 vs −0.21 |
| E13 | SIG-SYM | SSM numerics | symmetric LUT sigmoid [0, 8): 1 − s loses small-sigmoid accuracy (|da|/(1−a) 1.6e3) |
| E14 | EXP-2F | SSM | range-reduced 2^f table (not faster), direct PWL exp on log_a (29 % / 17 % mismatches), log_a fused into pass A |
| E15 | ZASM-BENCH | SSM | "asm z loop no faster than C++" (one-core bench) — wrong in situ: ZK −0.17 |
| E16 | X-PRESSURE | SSM | sqrt fused into ssx below II 27, FSX inside SSQ: x-register pressure |
| E17 | X1-SLIDE | SSM | sliding SSM area + wrap frame: superseded by mod-4 rotation (ACT +232 MB at N 8) |
| E18 | P7-FOLD | norms | norm folded into GEMM prologue / epilogue: parked (L1 / text); conditional |
| F1 | HEAD-1T | host | heads on one thread: next conv slower (OpenMP pool re-spin) |
| F2 | OPENBLAS-THREADS | host | numpy heads with OpenBLAS's own threads beside torch: 11.7 ms |
| F3 | H1 / H2 | host | host overlap levers: H1 cannot help in latency mode; not attempted |
| F4 | BF16C-64K | numerics (moondream) | bf16 C between 64-K chunks: 96.6 % vs 98.0 % (−1.4 points) |
| F5 | HADAMARD | numerics (moondream) | 64-wide Hadamard along K with bfp: 96.1 % (hurts) |
| F6 | INT8 | numerics (moondream) | per-token int8 vision: 89.2 %; int8 mmul gives no extra MACs |
| F7 | HW-PRIMS | numerics | vtanh, aie::exp2, aie::inv: inaccurate or scalar |
| F8 | MAX8 | numerics | 8-lane bf16 aie::max: wrong values |
| F9 | NEWTON-RECIP | numerics | one Newton step in the reciprocal: bfp ratio 0.4–0.6 (visible) |
| F10 | P24c | numerics | bias as the accumulator's initial value: not bit-identical; parked |
| F11 | P66 / P67 | numerics | seed max with q·k / TAU 16: not bit-identical, +1 KB L1; parked |
| F12 | FA4-F1F2 | numerics | software exp split (FA4): ≤ −0.1 ms, register-bound; not worth a DAVIS round |
| G1 | OZ-MINSIZE | Peano | -Oz / -Os / minsize on hot code: up to 2.4x–10x slower, not smaller |
| G2 | POSTPIPE | Peano | AIE_PREPARE_FOR_POSTPIPELINING: 100 vs 65 bundles |
| G3 | LINEAR_APPROX | Peano | aie::linear_approx: 2.7x slower than a hand lookup |
| G4 | PEANO-CHAINS | Peano | four interleaved chains spill; gain nothing over two |
| G5 | RESEEK | Peano | re-seeking a block-vector stream after a pop: garbage / NaN |
| G6 | SCHED-BARRIER | Peano | sched barriers inline in a pipelined loop crash clang |
| G7 | PEANO-SELECT | Peano | vectors assigned under runtime conditions / aie::select: "Bad PHI" crash |
| G8 | IRON-MCAST | IRON | multicast + unicast packet flows on shared ports: destination dropped, hang |
| G9 | IRON-2CH | IRON | two packet flows from two DMA channels of one tile: no route |
| G10 | IRON-BD | IRON | stride-0 tile BDs, two producer BDs, non-64 padd immediates: not expressible |
| G11 | BDSTEP20 | DMA | BD step > 20 bits: correct but columns serialised (norm3 6 ms) |
| G12 | ADDIMM7 | asm | add #imm is 7-bit signed: #64 → #−64 silently (hang) |
| G13 | ZOL-ALIGN | asm | unaligned ZOL last bundle / 2-bundle body: runs once |
| G14 | P2ALIGN-PAD | asm | .p2align pad bundle: prologue/body port collision (bgelu II 11 rejected) |
| G15 | FIFO-POP9 | asm | > 8 dense 576-bit pops without a fill: underflow |
| G16 | LOCK-ABI-HANGS | asm | acq value 0, uninitialised r25, Peano's r1 argument: hangs |
| G17 | READ_TM | probes | core timer read hung the core (resolved by the bus enable) |
| G18 | TEXT-16K | Peano | 16 KB program memory killed or parked several levers |
| H1 | CONSTANTS | method | 1.43 GHz clock and 4 B/cycle port: wrong constants that compounded |
| H2 | ONE-NUMBER | method | one number matching a bound cannot separate two hypotheses |
| H3 | FREE-CONST | method | fitted free constants (sim9 t_ks, sim10 t_tile) absorbed unseparated time |
| H4 | MEAN-VS-MEDIAN | method | k-step means include the op-start wait (round 9 → 10) |
| H5 | PROF-COUPLING | method | row coupling read from PROF walls: qkv −76 µs error |
| H6 | STALE-BASE | method | baselines from older rounds (P29a, P52, X1b) |
| H7 | GUESSED-COMPONENT | method | a guessed component 7x off (G1a stash copy) |
| H8 | LOOP-VS-CALL | method | loop-only prices vs whole call (P50 87 → 136, P29b, SQK, N1) |
| H9 | BENCH-NOT-DEPLOYED | method | benched a binary that was not the claimed loop (forced II) |
| H10 | BENCH-CPP | method | C++ bench pipelines, in situ serial (conv 377 vs ~1,000; z loop) |
| H11 | PRERELEASED-LOCKS | method | bench pre-released locks (g2f) |
| H12 | BENCH-COVERAGE | method | benches not checking past the end / the second FIFO / the right header |
| H13 | EW-FITS | method | rev 12 / 13 EW fits with a nuisance parameter (R_out 2.35, R_in 5.75) |
| H14 | CP-FIT | method | rev 15 boundary fit with a free t_op |
| H15 | FIVE-MS | method | "5 ms of boundaries" counted first-tile compute: ~3.0–3.3 ms |
| H16 | POLL-QUANT | method | host-poll medians quantised at 16.5 µs |
| H17 | TWINS-STEP | method | twins priced from step deltas: −1.98 twin vs −0.16..−0.28 real |
| H18 | KAPPA | method | CPU lazy-max map overstated in-situ flags (κ 0.6); "lockstep x6" |
| H19 | RARE-MEAN | method | rare per-unit costs priced by their mean (lockstep multiplies them) |
| H20 | BIMODAL-60 | method | single 60-iteration medians under foreign load are bimodal |
| H21 | POSTHOC | method | post-hoc "hits" not counted as pre-registrations |
| H22 | FALSE-NULL | method | stale jit cache / #undef-ed flags: same elf, false nulls |
| H23 | BUILD-RACE | method | parallel builds in one dir; shared generated headers |
| H24 | OBJDUMP-Z | method | llvm-objdump without -z hides nop runs |
| H25 | REGION-AS-FREE | method | linker regions / dropped entry points read as recoverable space |
| H26 | NUMERICS-30F | method | SSM numerics judged by 30-frame max deviations |
| H27 | HALF-BUILT | method | a half-built proposal scored as a whole (P43) |
| H28 | STAMP-CONTAINS-CALL | method | stamp intervals include the call's own cycles |
| L1 | KTL-R68-CP | GEMM k-step | KTL at R 68-69 without bank conflicts: not found within 4 CP workers (bank 3 at 63 / R) |
| L2 | FSX-CP | SSM | FSX: no schedule at II 30-33 (ResMII 30); worth <= -0.05 in a read-capped SSM span |
| L3 | WINDOW-AVERAGE | method | DDR headroom read from a window average (r+w over 320 us) while the SSM's own span sits at the read cap |
| L4 | OUT-TILES-ON-HEADS | GEMM | out's tiles 1-3 following the SSM heads: 4 resident C tiles per core |
| L5 | MSPLIT-OUT | boundaries | out in two row passes for norm1: +0.43 / +0.14 / neutral (main r23, r24a) |
| L6 | YXH | SSM / yx | yx head-major with finer drain groups: +0.48 (7 TAPs), +0.09 (4 TAPs; the SSM slows) |
| K1 | KTL-MULMASK | GEMM k-step | lock condition flag x counter: acq.cond tests bit 0, acquires skipped: the KTL hang |
| K2 | MUL-LATENCY | asm | scalar mul / add latency in a ZOL body as the .td says: not the KTL cause |
| K3 | STALE-HEADER | method | kernel run with another schedule's record header: a timeout |
| K4 | ATTVS-II | attention | attvs II 18 -> 11: 6 iterations per call, <= -0.02 ms |
| K5 | S3 | attention | S3 on the new attention loops: -0.01..-0.04 ms |
| J1 | RADD / KSUM | GEMM / norms | residual add or K-split sum in the GEMM epilogue: no third core input port |
| J2 | EW-PRESUM | norms | EW presum during the second K half: no DDR headroom there |
| J3 | KCONCAT | GEMM | mlp3 / down as K = 3072: A slot 497 KB |
| J4 | YRES / SSPM | SSM | yx on chip / params multicast: SSM compute-bound, ~0 today (conditional) |
| J5 | GELU-EW | epilogue | GELU on the EW cores: DDR over the cap during mlp0 |
| J6 | B-ALT | GEMM L1 | alternate the popped B bank: no room for a second Y buffer in bank 0 |
| A24 | E7L-STACK | GEMM L1 | shrink the GEMM stack for bank-0 room: ≤ ~512 B (deepest 1,856 + 64) |
| A25 | E7L-LUT | GEMM L1 | GELU LUT bank-0 copies into bank 3: both copies in one bank, +8 cycles / 32 lanes |
| A26 | E7-ABS | GEMM k-step | rb 0's pops from rb 0's stream instance: CP-infeasible at R = 64 |
| A27 | E7-R64-NX8 | GEMM k-step | E7'' R = 64 with a 576 B X segment: bank 3 at 63/64, no schedule found |
| A28 | ROT-KTL | GEMM k-step | one modulo loop per tile with one B FIFO: instances straddle row blocks (needs E7'' + aligned) |
| B13 | CSUB-G | C path | gated tiles in the CSUB group loop: +0.02 ms (round 21) |
| B14 | RTC | C path | runtime C drains with per-op lengths: ~−0.05..−0.1 for L effort + CP pushes |
| C24 | 3A-HOLES | memtile | third A region in today's memtile 0–2 holes: largest hole 56 KB < 124 KB |
| C25 | NTAIL-PF | boundaries | norm inputs prefetched during the drain tail: ≤ 0.07 ms |
| G19 | ACQ-ONLY-BD | IRON / MLIR | acquire-only BD block refused by the verifier; release by 0 works |
| H29 | RESIDUE-PROLOGUE | method | per-call prologue left out of a k-step price (gen_e7 R = 68: 1,338 vs 1,357) |

## A. GEMM k-step and L1

**A1 — A-PORT: "the k-step sits on the 4 B/cycle A-port floor"** (rev 1, round 5) [MEAS]. Rev 1 fitted t_ks 1.857 µs against an A-port floor of 1.813 µs (10,368 B at 4 B/cycle, 1.43 GHz) and booked 32 % of the step on "A input port". Measured (RFM round 6 P1, twins): no-MAC build 67.9 ms (−16.1), 2x-MAC 108.7 (+24.7) vs 84.0; down_a without MACs 1.04 µs per k-step ⇒ A ≥ 7 B/cycle. Why: both constants of the floor were wrong (H1) and a one-number match decided nothing (H2). Lesson: book a bound only after a perturbation (twin) confirms it.

**A2 — P6 / P40: two A streams per core** (rev 1–2; P40 rev 5) [MEAS]. Refuted rev 2 (rev2/P status table): the memtile→core port is not binding (≥ 7.6 B/cycle available, 3.1 used); round 10 measured 7.95 B/cycle (RFM round 10 P10, r10/rtm.py). P40 (A over the core's 2 S2MM) also needs out-of-order BDs that IRON 1.4.4 lacks. Conditional: revives only if the k-step needs > ~7.5 B/cycle of A (A-port floor 1,304 cycles per k-step at 7.95; rev 13/14 note "beyond ~1,450 the k-step needs A on two channels").

**A3 — A-PORT-R9: "after P20 the k-step is A-port-bound at ~4 B/cycle"** (round 9) [MEAS]. Round 9 read 2,588 (down_a) – 3,320 (out) cycles per k-step = 10,368 B at ~4 B/cycle. Round 10 (RFM, r10/gdist.py): per-tile MEDIAN busy 2,050–2,065 for every op; steady A ≥ 5.04 B/cycle. Why: the 2,588 was a mean; each op's first tile carried the op-start A-slot wait (4,170–7,300 per k-step). See H4. MACHINE rev 5 corrected.

**A4 — P17: L1 bank placement for the C++ GEMM loop** (rev 2) [MEAS]. Predicted 12.6 % in-call bank stalls. Stamps (RFM round 7 P16-lite): mm_acc 333 cycles per call in situ = 321 static + ~12 for the stamps. Not built. Lesson: the fitted "stalls" were dependency / tile-boundary time spread by a least-squares fit (H3). (The asm call later DID stall on banks: P21 placement −1.5 ms.)

**A5 — P21-RULES: scheduling rules to avoid bank conflicts in the asm call** (rev 3) [MEAS static]. Predicted 225–232 bundles, 295 ± 8 cycles per call, 68.2 ms. Built in gen_mm.py (RFM round 8): R1 (no acc vlda with an acc vst in one bundle) 273 bundles; A-ops off B's cycles 255; neither kept. Why: a 4-chain group boundary has 16 loads + 16 stores of one bank inside ~16 cycles; serialising costs the same cycle as the stall. Lesson: count slot occupancy in the conflict window before pricing a rule; fix layout instead (P20a).

**A6 — CPP-4CHAIN: 4-accumulator-chain GEMM loops in C++** (round 6–7, P18 "cheap first step") [MEAS static]. 54 bundles per 16 mmuls with spills, 42 without the C prefetch, 61 with A popped per block via an opaque seek, vs 28 deployed (RFM round 7 P18; NOTES 12.3). Round 6: two column blocks per A pass 75 per 32 mmuls with spills; half-k passes 26 / 35 / 48 per 4 / 8 / 16. Why: Peano keeps A in 8 of 12 ex registers and does not pipeline across column blocks; 5 dm registers for 4 accumulators + loads. Lesson: the asm route (gen_mm.py 225 vs 321 per call).

**A7 — P18-EXTRAP: first asm GEMM schedule extrapolated at −7 ms** (round 7) [MEAS]. Measured −1.9 ms (73.10 → 71.19). Round 8 (r8/mmb.py): 238 cycles conflict-free, +33 with accumulator load + store in one bank (as deployed), +14 A and B in one bank, +42 B with C; in situ 337 per call. Why: L1 bank conflicts not in the extrapolation. Lesson: bench the call on the deployed placement before extrapolating.

**A8 — PIECEMAP-GREEDY: bank-aware runtime accumulator piece map** (round 9, P20a / p20map) [MEAS]. k-step faster (323 vs 333 cycles per stripe, down_a; qkv 317 vs 342) but forced a table-indirect epilogue: plain epilogue 0.64 → 1.45 µs per tile (~0.8 µs worse); affine layout + strided passes (0.72 µs) won (RFM round 9; RESULTS 18.4; NOTES 14.3). Lesson: a piece map is paid for in every epilogue.

**A9 — A_BANKS-PIECES: A-slot bank pinning combined with the piece layout variants** (round 9) [MEAS build]. Did not link: bank sections overflow (NOTES 14.3). Number not recorded.

**A10 — LAYER-II: P36 deep-K accumulator visits, P39 A from the neighbour core, P41 a 4th GEMM row** (rev 5–8, parked) [MODEL]. Blockers: P36 +8 KB L1 in a full core; P39 neighbour locks in asm, unmeasured 3-core bank behaviour; P41 needs the EW work gone (rev5/P "Parked layer-ii"). Main round 10: "agree, 0 ms today". Superseded in practice by ROT (k-step 2,055 → ~1,576 in situ). Conditional: only if the k-step again needs more A or MACs per byte than one column delivers.

**A11 — ROT-TEMPLATES: identical job templates at 9-cycle offsets** (rev 14) [MODEL, CP proof]. With a B restart gap g > 0 (g ≢ 0 mod 9) identical job timings are INFEASIBLE: the B pass wraps each job's pops by R mod 9 and forces its 8 pops into consecutive cycles, which a dependent MAC chain (distance ≥ 3, 4 B registers) cannot follow (rev14/P headline; MACHINE R14.1). Fix used: per-job issue cycles (the loop body is one row block).

**A12 — E6 / E6-B: move the B prefetch out of the popped bank** (rev 13 E6; rev 14) [MODEL + MEAS]. Void under ROT: B is popped every cycle (bank-3 busy 0.97) and no bank is idle to move it to (ROT busy b0 0.24, b1 0.45, b2 0.37); the B prefetch collides ~1:1 (~110 cycles after calibration) (rev14/P headline, status table). Lesson: layout levers need an idle bank; check per-bank busy first.

**A13 — B-2BANK: B objects in two banks** (rev 14 ranks 4–5; rev 15) [MODEL, capacity proof]. A bank holding a popped B object can hold nothing the k-step touches; two such banks need 2 × (16 − 4.5) = 23 KB of k-step-cold data, the GEMM core has 14.6 KB (stack 2.5, tables 5.6, C tail 6, misc 0.5). Partial version (B1 in bank 0 with 5 pieces): ~125 cycles per k-step of internal conflicts for ~70 saved (rev15/P "GEMM L1 layout"). Dropped round 19 (RFM round 19 table). Remaining B-collision floor ~92 cycles per k-step "has no lever".

**A14 — E7': B over two banks with one stream** (rev 14 rank 4; rev 15) [MODEL]. Void: ROT's R = 72 + g is the single B stream — the 72 line reads per row block all issue from ONE FIFO (lf1), a FIFO cannot pop twice in a bundle, so the ldb slot and bank 3 both bind at 72 + g (rev15/P; MACHINE R15.10). Successor E7'' (B on BOTH FIFOs, −0.9 ± 0.3 if CP-SAT finds R = 64): POPLF1 passed round 20 (vlda.pop.576 on lf1 128/128), not built. Conditional on E7'' feasibility, not on E7'.

**A15 — ASLOT: A slots into the least-busy bank** (rev 14 rank 5; rev 15) [MODEL calibrated]. collide.py (α 0.72, reproduces KS-P38's 167.5 MEMORY_STALL): best feasible assignment (slots in banks 1 / 2 / 0) −15 ± 10 cycles per k-step ≈ −0.1 ms, and needs bank-0 room the 28 KB data region does not leave; the bound (A slots in a piece-free bank) −48 cycles is forbidden by capacity (36 KB of pieces cannot fit two banks). Dropped round 19. Why: every row block's two pieces lie in two of banks 0–2.

**A16 — ACC-OUT: move the 12,288 B of accumulators out of the C object** (main's round-19 request; rev 15) [MODEL, capacity]. Core 0_2 L1 is 64,960 of 65,536 B (576 B free); moving the accumulators out needs +12,288 B while C keeps its 18,432 B; reducible items ~3.4 KB; tables (5.6 KB) are needed (rev15/P "The C object"). Dropped; ZHI / CSPLIT / EPLR attacked the C wait instead (−0.5 ms together).

**A17 — g2f: ROT with acq.cond one bundle before the A fill** (rev 14 rank 1; round 18) [MEAS]. One-core bench passed (36/36, warm 1,431 vs pre-registered 1,390–1,405); in situ the regress was DIFFERENT and varied run to run (rms_rel 0.15–0.25, tracks ≤ 0.13 px); a plain `rel` changed nothing. Why: acq.cond #53 sat 1 bundle before the A fill (constraint ACQ in [AF−6, AF−1]); the bench pre-releases all nine A slots so the acquire never blocks (H11). Fixed as g2a (ACQ ≤ AF − 6, REL ≥ A7 + 8): −3.27 ms (RFM round 18 ROT). Lesson: lock gaps are hard floors in the generator (asmlint LOCK-ACQ-GAP).

**A18 — STATIC-PRICE: k-step static cuts priced 1:1** (rev 14 g0f −0.28 / g1f −0.14 vs g2f; main −0.23 / −0.13) [MEAS]. Round 19 A/B vs r18m: g0a −0.174 / −0.183 / −0.152, g1a −0.104 / −0.111 / −0.079 (p10 / median / mean) for 26 and 15 static cycles (8.59 µs per step per cycle): ~0.75 of the static price lands (RFM round 19; rev15/P prices). Lesson: price k-step cycle cuts × 0.75.

**A19 — g1a: ROT with the padda restart** (round 19) [MEAS]. −0.111 median (OPTIMAL R 73, 1,368 static) vs g0a (B wraps inside the last pop) −0.183 (R 72, 1,357). Not kept; g0a deployed (full12r19b). Both bit-identical.

**A20 — PWRAP-4544: pre-registered pop.2d wrap offset** (rev 14 / 15 pre-registration; round 19) [MEAS]. m2 = −4544 gave pass 2 all 64 pops wrong (64/128); m2 = −4608 right: the 2D post-modify (dn 0) is ADDED to the 64 B line step (r19/wrapgen.py + wrapb.py). PADDA in the fill's own bundle (negative control) 64/128; one bundle before 128/128. gen_rot's emit and replay fixed (step = 64 + m2). Lesson: probe every unverified addressing mode with a negative control.

**A21 — P35-PREMISE: "the gap between mm_ks calls (67 cycles per k-step) is code"** (rev 13 rank 2, −0.44 ± 0.1) [MEAS]. Built (B locks by id, a tile's k-steps per call): 30 rounds −0.11 / −0.10 / −0.08 (RESULTS 27.1). Why: the k-step gap is mostly waits, not code (NOTES 23.2). Kept, but the price model's premise is dead: do not re-price call-overhead removal in the k-loop as 1:1.

**A22 — MD-KLOOP: Peano k-loop variants (moondream)** [MEAS]. Inner block loop 68–70 bundles per 32 MACs: pipeline(disable) + min count 68, II hint 40 ignored (70), unroll 2 (140 per 2 blocks), full unroll spills (worse) — "DISCARD all". rho (A sub-tiles per k-step) 2: 2x slower (unexplained; bank conflicts?). fp32 C prefetch of the next block needs 6 dm registers (AIE2P has dm0–dm4): spills, -DNO_PREFETCH better (MD NOTES lines 80–88). Lesson: these limits carried over to TAPNext (A6).

**A23 — MD-HANDSCHED: moondream's first hand-scheduled GEMM loop** (moondream) [MEAS]. 1.83 bundles per MAC correct / 1.32 broken by the stage-3 rule vs the deployed 1.59 — "not worth more" (SKILL §7). Superseded by the TAPNext asm generators (gen_mm → gen_ks → gen_rot) once the hazard rules were measured (round 11).

## B. GEMM tiles, epilogues, C path

**B1 — P24: "the C handoff costs 3–6 µs per tile"** (rev 3) [MEAS stamps]. Round 8 stamps (RFM round 8 P24; RESULTS 17.2): handoff (epilogue end → next acc_zero) 1.31 µs median, acc_zero 0.49, plain epilogue 0.64, tail 69 cycles. Why: sim10's fitted 7.6 µs t_tile held each op's first-tile A-slot wait (27–50 µs in g11, 177–203 after norms, 1,137 after the SSM). Lesson: H3. (Round 19 TPROF later found a real C-drain wait of ~2,330 cycles per object once the k-step was faster — a different mechanism, fixed by ZHI / CSPLIT / EPLR.)

**B2 — GELU_II: forced initiation interval on the C++ GELU loop** (round 10) [MEAS]. `#pragma clang loop pipeline_initiation_interval(16)`: "17-bundle loop" ran 54.7 cycles per 32 lanes (deployed 34.2); in situ yx epilogue 3.39 → 5.02 µs; rejected (RFM round 10; NOTES 15.3 still lists it as a dead end). Round 11 correction: Peano had silently NOT pipelined it (mm8 made in / out restrict copies of one pointer; gxb used MIN_ITERATION_COUNT(8)); with separate restrict parameters 17.8 cycles at n = 144, −0.88 ms (full12r11a). So the dead end was a build artefact; the idea worked. Lesson: read the emitted loop and bench the binary you deploy (H9).

**B3 — P45-H1..H4: hazard hypotheses for the 54.7-cycle loop** (rev 6) [MEAS]. None held: the loop was never pipelined (B2); rev 7 scorecard "wrong premise". H1 (WAW loss) is false outright: an older load and a younger write land in landing order, nothing dropped (RFM round 11 P45 item 3). The real rules came from the ladder's method (one-core probes): LLVM latency visibility, the mv → mv bypass, itinerary ports.

**B4 — GELU-FLOOR8: "GELU floor 8 cycles per 32 lanes (ldb, mv)"** (rev 6) [MEAS]. Wrong: counted slots only; DM_ADA (LUT loads + every vlda) and QEY_RS (LUT index reads + vfloor + vst x) are shared ports: ResMII 10–11; II 13 reached (rev7/A scorecard). Lesson: price shared itinerary ports, not only slots.

**B5 — GELU-II10 / bgelu II 11** (round 12) [MODEL, CP]. Standalone GELU with flat-edge tables: ResMII 10 but CP-SAT found no II-10 schedule within 8 x registers (400 s each, 4 reload variants; NOT proven infeasible). bgelu at II 11 placed but failed the .p2align padding check (G14); II 12 deployed (NOTES 17.4). Conditional: a padding-aware CP model (P53, never attempted) might recover II 11.

**B6 — RECIP-FREE-GELU: reciprocal-free / polynomial GELU** (round 5) [MEAS emulation]. gelu = relu(x) − |x| s with s by a polynomial in u = exp(−|t|), or h(|x|) as one polynomial: 26–37 % bf16 mismatches, up to 4–6 ulp, rms 2.3–4.5e-3 vs today's 15 % / 2 ulp; a polynomial in |x| on [0, 4.5] is ill-conditioned in bf16 Horner (1.6e-2) (NOTES 10.2). Not built. The LUT GELU (round 6, −6.0 ms) replaced the idea.

**B7 — P43 / P43b: hoist the B fills for "B-starved" first tiles** (rev 6, −0.8 ± 0.3; round 11) [MEAS]. Built part (P43): −0.35. Pre-registered check mlp3_a 166 → 137 ± 8 µs per block: measured 158–161 (MISS). Why: the dependent boundary is the A slot load from DDR (8 × 248 KB in ~34 µs), B is ~14 % of the bytes; same-shape non-dependent ops show no wait (rev8/A §0.1). P43b retired (rev 8). Lessons: mechanism check across ops before pricing a boundary fix; score a partial build part by part (H27).

**B8 — CSPLIT4: progressive release of C part 1 in 4 sub-parts** (rev 15 rank 3, −0.3..−0.35) [MEAS TPROF + MODEL]. Re-priced rev 16 as CSPLIT4-lite −0.11..−0.17 (one static sub-part boundary at 9,216 B): measured hands 1,808 cycles (gelu_pan / qkv / gated-first / gelu) vs floor 676; EPLR had taken the plain share (−0.12). Not built through round 20 (needs every epilogue rb-major for rb < 12 and an acquire-only BD whose lowering is UNVERIFIED). Conditional: worth building only as -lite, or if epilogues change.

**B9 — LATE-C: C acquire inside mm_rot at rb 6 + zero-load first k-step** (rev 15 R3 / rank 7, −0.3) [MODEL]. Re-priced rev 16 to −0.08..−0.12: plain hand 1,371 after EPLR leaves at most ~700 cycles × 32 tiles. Not built. Conditional: revives if the C wait grows again (e.g. a faster epilogue path) — needs gen_rot CACQ + CP feasibility at R = 72.

**B10 — DUALDRAIN: C drains over shim S2MM 0 + 1 per column** (rev 16 rank 8: −0.3 if the write limit is per channel, 0 if per column) [MEAS]. DDRW (RFM round 20): one stream 12.1 GB/s (~6.7 B/cycle), 2 columns 24.4, 4 columns 45.3, 8 columns ~50–60 GB/s: the write cap is AGGREGATE (= the read cap ~55 GB/s), so a second channel per column adds nothing. Dead.

**B11 — MRES: GEMM → GEMM APAN output resident in the memtile** (rev 16, ~−0.4) [MODEL, capacity]. Infeasible on memtiles 0–2: a 124 KB park against 112 KB free in two holes (rev 16 P/A §4). MLPRR (mlp0 dealt round-robin, mlp3_a waits for the first drain group) got most of it through DDR: −0.40 (RFM round 20).

**B12 — MD-GELU-STAGED: exposing independent GELU chains to Peano** (moondream phase 12) [MEAS]. Same arithmetic as three stage loops with L1 staging: rolled (GV 8) 1177 µs vs 1108 reference (slower); fully unrolled GV 2 / 4 / 8: 471 / 469 / 481 bundles per 64 elements vs 492 (70 / 122 / 302 spills): no gain, not run (MD NOTES ~l.695–702; MD RESULTS l.619). Why: Peano issues ~1 op per bundle on long chains (register-allocation bound). Lesson: the asm route or a table.

## C. Op boundaries, command processor, DDR

**C1 — ROWCHUNK: overlap EW and GEMM work by splitting a frame into row chunks** (rev 2 §4.2) [MODEL, sim9 list schedule]. 76.3 ms serial → 76.9 (2 chunks) / 77.2 (3 chunks); with all fixes 35.2 → 35.9 / 36.1. Why: every chunk repeats each N-tile's fixed cost and every dependent A-slot load (+0.77 ms per block then); attention is a per-block barrier; the SSM faces only ~0.3 ms of GEMM work. Not built. Conditional: only if per-tile and dependent-load costs approach zero (the model's 66.9 → 60.8 row).

**C2 — CPBND-FIT: "the op boundary is command-processor-bound"** (rev 15 headline 2, "strong circumstantial case, NOT measured") [MODEL → MEAS refuted]. Model: 92–144 CP instructions after each token wait × t_op 0.17 µs (0.13–0.22) + A-bound first tile + ~3 µs DMA latency reproduced yx / out / qkv / out_proj / mlp0 / mlp3_a / norm boundaries within ~5 µs. Measured: CPRATE t_op 0.06 / 0.11 µs (C3); CPST (round 20): the CP passes the first tct 12.2 µs after mlp0's last epilogue; post-sync CP work ~2.5 µs to column 0 (+0.5–0.85 per column). Why: the free t_op absorbed the 12 µs producer tail (rev16/A §0). Lesson: name the probe that separates hypotheses before ranking (H14).

**C3 — CPRATE-CAUSE: CP instruction rate as the cause of the boundaries** (rev 15 rank 0 probe) [MEAS]. CPRATE=400 extra instructions per block on mlp3_a's path (r19g, 20 × 20 A/B): npu_write32 +0.268 / +0.296 / +0.274 ms → 0.056–0.062 µs each; always-satisfied npu_maskpoll +0.520 / +0.553 / +0.522 → ~0.11 µs (pre-registered 0.13–0.22; kill line 0.03) (RFM round 19 CPRATE). "CP real, rate half"; with SEQPRE's null (C4) the CP count does not set the boundaries.

**C4 — SEQPRE: issue the dependent op's independent work before the producer wait** (rev 15 rank 1, −1.4 ± 0.5; raw −1.37 / −1.79 / −2.31 at t_op 0.13 / 0.17 / 0.22) [MEAS]. Round 19 variants (design8 SEQPRE=bits; all bit-identical, no hang): v1 +1.10, v2 +0.13, drains-only +0.15, loads-only −0.04 (kept as SEQPRE=1). Why: the boundary is bytes (producer tail at the DDR write cap, consumer's first tile at the read cap), not post-sync instructions (RFM round 20 "Reading"). Retired rev 16.

**C5 — SEQPRE-v1: memtile slot loads + memtile sends + drains before finish_deps** (round 19) [MEAS]. +1.098 / +1.015 / +1.091 ms (vs r19g, 30 × 20). Timeline: down_a / down_b / mlp3_b gaps +0.35 / +0.48 / +0.42 ms. Why: a K = 1536 op pushes 5 memtile MM2S tasks onto the 4-deep queue; pushed before the A fills, the blocking push waits on work that cannot progress until after the wait (the round-10 ordering rule). Lesson: never push a memtile MM2S task before the same op's fills.

**C6 — SEQPRE-v2: sends after the A fills, loads + drains before** (round 19) [MEAS]. +0.109 / +0.131 / +0.145 ms. Dead (RESULTS 28.1).

**C7 — DRAINS-FIRST: only the drains before the wait** (round 19, SEQPRE mode 2) [MEAS]. +0.150 / +0.158 / +0.151 ms. Cause unexplained (shim S2MM queue / sync semantics suspected; RFM round 19). NSEQ (round 20) moved norms' params + drains before the wait and hit (−0.17), so the penalty is specific to GEMM drains; rev 16 NSEQ item 3: "Do not move the GEMM drains".

**C8 — BDPRE: hoist the post-sync BD writes** (rev 16, `r21` analysis) [INF from CPST]. BD write (blockwrite) + address patch ~0.10 µs of CP time (MACHINE R16.2) ⇒ ≤ −0.1 ms per step. Dropped. The confirming probe CPBD (dummy blockwrites after mlp3_a's sync) was optional and not run.

**C9 — SENDPRE-dep: memtile sends before the wait for DEPENDENT ops** (rev 16) [MODEL]. ~0: after the wait the first tile is DDR-paced anyway (pace 6.5–8.5 / 13.3 µs). Dropped.

**C10 — SENDPRE: memtile sends before the wait after EW ops** (main, round 20, predicted −0.05..−0.15) [MEAS]. +0.197 / +0.162 / +0.155 ms (full12r20s1) — "dead (as you retired it)" (RFM round 20; NOTES 25.4). Cause not separately measured; consistent with C5 / C7 (anything pushed early on the memtile / shim queues can delay the critical path).

**C11 — OPPRE: out_proj's A loaded in 4 parts during attention pass 1** (main, round 20, predicted −0.18..−0.3) [MEAS]. +0.117 / +0.205 / +0.184 ms (full12r20o1). Pass 17 (rev17/P): the aggregate-DDR reading is refuted by the bytes — attention pass 1 reads ~2.65 MB of K / V in ~160 µs (~17 GB/s) and the 1 MB A adds ~6 GB/s, ~23 GB/s against the ~55 cap. Remaining hypotheses: H_CP (the per-head-group drain waits and their extra task groups / BD reclamation sit in the in-order CP stream and delay the K / V pushes behind them; 3 waits × ~5 µs tail ≈ 15 µs per block = +0.18) and H_SHIM (shims 0–2 carry the K / V on MM2S 1 and now an A part on MM2S 0 at once). Separating probe: OPPRE-W (the same waits, 64 B fills) and OPPRE-37 (the A parts only on shims 3–7). The patch was not committed: commit dead experiments so their order can be read. Lesson: a CP wait is a barrier for every push behind it.

**C12 — DDR84: "the NPU sustains ≥ 84 GB/s DDR read + write"** (rev 15 R15.4; DDRCAP pre-registered ≥ 84, ceiling ~110–120?) [MEAS]. DDRCAP (r19/ddrcap.py): one stream ~7 B/cycle; 4 × 1 and 2 × 2 streams 50 GB/s (minus dispatch), 8 × 1 / 4 × 2 59, 8 × 2 54 GB/s; with drains 63–74 GB/s read + write. Aggregate read cap ~55–59 GB/s. Why: rev 15 derived 84 from norm walls × bytes (8 × 4,672 B per 801 cycles) and called MACHINE §5's 55–63 GB/s "8 × the single-channel rate" — DDRCAP restored the old figure. Note: MACHINE R15.4 still states ≥ 84 (stale).

**C13 — NIN2: the norms' inputs over a second shim channel into the EW core's free S2MM 1** (rev 15 rank 2, with N-IO-ASM −0.7..−1.5) [MEAS]. Void (RFM round 19 DDRCAP): 3-input norms read 3,072 B × 8 per 801 cycles = 55.5 GB/s, already at the aggregate read cap; channel count per column is irrelevant (4 × 1 = 2 × 2). Not built. Lesson: the norm floor is bytes; only fewer DDR bytes move it (NRES −0.65 for the panels).

**C14 — H_WR: the 12 µs producer tail is a per-channel shim S2MM write limit (~4 B/cycle)** (rev 16 headline 3) [MEAS]. DDRW: one column writes 12.1 GB/s (~6.7 B/cycle), not 3.5–4.5 (RFM round 20 table). Refuted. Rev 16's evidence (down_a / mlp3_a spans 15 µs longer than down_b / mlp3_b) is explained by the 2 MB A preload sharing the aggregate cap.

**C15 — H_TOK: the tail is ~6 µs of token / CP wake-up latency** (rev 16) [MEAS]. Refuted by DDRW too: the tail is the last 442 KB of C objects (8 columns × 3 × 18,432 B through store-and-forward memtile joins) written at the aggregate write cap (~55 GB/s ≈ 8 µs + latency). Neither H_WR nor H_TOK.

**C16 — DDRW-2ch: two drains per column to separate per-channel from per-column write limits** (rev 16 probe gating B10) [MEAS by implication]. Moot: the 1 / 2 / 4 / 8-column DDRW series already shows an aggregate cap; not run.

**C17 — NRES-RBMAJOR: NRES with an rb-major A-slot half 0 and new sends** (rev 16 rank 2 design; round 20) [MEAS]. Broke correctness: the last tile's chunk-wise `last` release let the next op's chunk-0 load overwrite rows 0–1 of every chunk (G chunks 31 / 39 / 47 wrong). Built instead: memtile S2MM 5 scatters the rb-major stream into the usual chunk-major half with a 3D TAP `[18 rb][12 q][144 words]`, normal sends kept (RFM round 20 "NRES as built"). Lesson: keep the A slot layout the sends and lock releases expect.

**C18 — NRES-ALONE: NRES norm1 → up_a without NSEQ** (round 20, predicted −0.15..−0.28) [MEAS]. 40 × 20: −0.113 / −0.057 / −0.091 (miss): columns start 1.5 µs apart (the CP's per-column norm issue) and the B multicast holds every column to the last one. With NSEQ −0.177 / −0.186 / −0.208 (hit). Lesson: an on-chip bypass still pays the CP's serial per-column issue unless the sequence is reordered too.

**C19 — P60-1TAP: deal natural query tiles round-robin with one TAP per column** (rev 9 P60; round 14) [MODEL, exhaustive + proof]. Natural tile (c, i) → (column s0 + i d, position p0 + i e) is one TAP for the qkv Q drain and the out_proj gather only if arithmetic in i; with a permutation per position only d = 0 solutions exist (r14/deal.py, deal2.py; a sum-of-squares identity forces d = 0) (RFM round 14 P60). Replaced by P60g (deal on the gather side): −1.06 ms.

**C20 — GATHER-MULTITAP: 2+ TAPs per gather / BD chains for the deal** (rev 10) [DOC]. Retired: the A slot is loaded in per-head 10,368 B stream chunks with one lock each (A_RT BdIteration); any gather that is not head-major lands pieces in the wrong chunk (rev10/P status table).

**C21 — KV-MEMBUF: per-column K / V buffering in the memtile** (pass-10 brief; rev 10) [DOC]. Retired: memtiles 0–2 use all 6 S2MM channels (A × 2, B, C join × 3), no channel for 3 row streams per column; −1.29 alone would not beat P60g.

**C22 — P60t: global host tile permutation (main's round-15 note)** (rev 10) [MODEL, sim17 κ 0.5 / 0.6 / 0.7]. −0.53 / −0.66 / −0.79 alone; composed with P60g −0.76, worse than P60g alone (−0.87 / −1.08 / −1.30) (rev10/A §3 lever table; rev10/P parked). Parked. Conditional: only if P60g is undone.

**C23 — P60h: adaptive deal from the previous frame's flag counts** (rev 10, FA4 LPT analogue) [MODEL]. ≤ −0.33 beyond P60g (perfect balance = 1-rb round robin −1.41). Parked; L effort.

## D. Attention

**D1 — P12-RESTREAM: re-stream K / V per pass** (rev 2 P12; round 7) [MEAS twins]. Not done: t4 twins put the K phase at 0.60 ms (exp + push 0.30, QK^T + max 0.30), V 0.19, data / control ~0.18; "the passes' compute, not K/V traffic, dominates" (RFM round 7 P12). Fixed costs per head group ~70 µs per block (8 %) (RFM round 8).

**D2 — ATT-DDR: "attention is DDR-bound"** (round 13) [MEAS twin]. A twin with no K-phase compute at all ran 43.4 ms (base ~50 ms): attention is compute-bound, not DDR-bound (NOTES 18.4). Moondream precedent: v8's memtile K / V broadcast (9.7 MB per block instead of 33.8) gave 0.955 vs 0.885 ms, nokernel 0.29 (MD NOTES table) — also not DMA-bound.

**D3 — EAGER-RESCALE: rescale O / sums on every row-max increase** (round 5) [MEAS]. Rescaling cost 0.42 ms per block, as much as the exp; lazy max (move only when the row max passed the stored one by TAU = 8, P ≤ e^8) took attention 1.77 → 1.35 ms per block (NOTES 10.1; SKILL §3).

**D4 — SM-BF16: round s − m to bf16 before the exp** (round 4–5) [MEAS]. Up to 2–3 % error at |s − m| ~ 10; take z = (s − m) log2e in fp32 straight from s and m (NOTES 10.1; SKILL §3).

**D5 — EXP-TABLE-RANGE: exp table with h = 1/2 or a range ending at −7 or higher** (round 7) [MEAS emulation on real scores, r7/expemu.py]. Worse than the deployed 80 segments of 1/4 on [−11, 9) (NOTES 12.1). Magnitude not recorded.

**D6 — EXP-CARRY: carry the looked-up exp lines across loop trips** (round 7) [MEAS static]. 65 bundles with spills vs 49 per 64 scores (NOTES 12.3; RESULTS 16).

**D7 — PV-POP: pop P per block in chunk_v instead of holding it** (round 7) [MEAS static]. 67 vs 42 bundles per pair (popping the held operand helped QK^T 70 → 53 but hurt PV) (NOTES 12.3; SKILL §3). Lesson: measure both directions.

**D8 — SPLIT-QK-EXP: separate asm QK and exp loops** (round 13) [MODEL CP]. attqk II 11 (−27 cycles per row block); aexp: no II ≤ 28 with 6 free x registers. Superseded by the fused lazy-fast K loop (II 35) (NOTES 18.5).

**D9 — Q-IN-EX: fused K loop holding Q in ex0–7** (round 13) [MEAS]. Computed garbage: ex_n aliases x_n, pops are x registers (NOTES 18.4–18.5). Also `--klda` (K pops on unit A) unneeded once known. Q is popped per key block instead.

**D10 — P54-BIASBASE: a base-register ("biased base") form of vldb.4x64** (rev 8 P54(1)) [DOC .td]. Refuted: `VLDB_4x64_lo / hi` take only an index vector (absolute address per lane); rev 8 read a slot name as an operand. P56's "II 37 with the biased base" was the unreachable branch; the fused loop reached II 35 anyway (RFM round 13; rev9/A scorecard).

**D11 — P56-FALLBACK: fused K loop priced −3.2 ± 0.5 with "fallback updates ≤ 3 %, +17 cycles per unit"** (rev 8) [MEAS]. Fast-loop-only twin −2.9 ms (the loop price was right); production −0.84 (49.21). Unpriced: the exact test on every unit +1.35 ms; chunk 22 (invalid key blocks) forced slow (4.35 % of units); the K / V multicast lockstep: the slow exp on ~10 % of units +1.35 for ~0.45 ms of own cycles (RFM round 13; rev9/A). Lesson: H19.

**D12 — P63 / 3D-POPS: the 36-iteration batched K loop "needs the 3D Q pop stream"** (round 13 not built; rev 9 P63, −1.25 ± 0.6) [MEAS]. Built round 14 with 2D modifiers and padda.2d / padds.2d — no 3D pop needed. Bench warm unit 444 → 261 cycles (−183 vs ~−115 predicted); in situ −0.04..−0.47 (bimodal): each flagged row block cost 1,530 cycles (rerun 393 + exact test 164 + rescale 578 + exact rerun 393) and the lockstep multiplied it (twin F128 46.15 vs 48.24–48.73 production) (RFM round 14). Kept as the base for S1 / S2 / P60g; the "3D pops" premise and the price were dead.

**D13 — CHUNK0-SKIP: skip the chunk-0 rescale (O and sums are zero)** (round 13) [MEAS — sources disagree]. Round 13: "no measurable gain, reverted for the 16 B of text" (NOTES 18.5). Round 14: built together with P61, −0.63 vs P61's −0.33 alone, "the chunk-0 rescale skip is worth ~−0.3 (round 13 had called it no measurable gain)" (RFM round 14). Not a dead end; listed so nobody re-kills it from NOTES 18.5.

**D14 — EXACT-TEST-CHEAPER** (HANDOFF 2; rev 9) [MEAS]. Retired: after the in-loop flag the exact test runs only on flagged units (10.8 %): < 0.1 ms left (rev9/P status).

**D15 — FLAG-RARER: make the lazy-max flag rarer without a numerics change** (rev 9) [MEAS map]. Retired: flag-not-move is 0.055 %; only numerics changes (F11) reduce it.

**D16 — DEFER-SLOW: defer slow rows outside the lockstep** (rev 9) [MODEL]. −0.80 alone, but needs a V copy per deferred chunk (no L1) and P60 leaves ~0.06 ms of lockstep. Retired.

**D17 — KV-SLOTS: more L1 K / V slots** (rev 9–10) [MODEL]. −0.66 alone (rev 9; rev 10: 3 / 4 slots −0.75 / −0.89), 4.6 KB per slot that L1 does not have; moot after P60g. Retired.

**D18 — AEXP-ASM: asm aexp (gen_aexp, HANDOFF 2b)** (rev 9) [MODEL]. Superseded by P62b: attk run with the new max computes the same P at the same cost and deletes 464 B (rev9/P; RFM round 14 T3a).

**D19 — P64: 3,456 B K / V objects** (rev 9) [MODEL]. −0.04: the producer is not binding (sim17 agrees). Parked. (P65, chunk-0 QK-only loop −0.13, parked for needing a second asm loop.)

**D20 — L2D12: `--l2-depth 12` for the memtile K / V forward** (round 13) [MEAS]. full12r13e12 49.58 ms vs 49.54 at depth 8 (4 → 8 gave −0.36) (RESULTS 22.1): no more. Why 8 helped is still open (rev 9 Q1).

**D21 — ATT-TEXT: attention variants killed by program memory or scheduling** [MEAS / CP].
- K1C (prep loop, two row blocks per trip): tcall −28 per (tile, chunk) for +96 B: dropped (round 18, RESULTS 27.1).
- −O2 rolled rescale_rb: 421 vs 578 cycles for +96 B: not adopted (round 14 P62).
- attkb in 2 stages (for text): no schedule up to II 42; kernel-only attv: the negative iterations' last-stage O stores land before the O tiles; cpy32 for the Q copy / exp-table copy: larger at minsize (NOTES 19.4).
- −O2 attn_begin 1,104 B vs 800 minsize: the GEMM elf overflowed by 96 B (round 7, NOTES 12.3).

**D22 — MD-STREAMS: moondream attention variants** (moondream) [MEAS]. v6: 3 K / 2 V streams + magic exp 1.61 ms vs 1.43 (the core has two FIFO load units; 3–4 live streams spill); v12: one stream per row block with `seek_byte(0)` → NaN (dropped); v14: native `aie::exp2` 0.51 ms but rms 2.7e-2 (ramp 8 % rms, 23 % max); v15s: stage-major exp wrong and no faster (MD NOTES attention table). v15 (exact exp) 0.675 ms was kept.

## E. Norms / SSM / elementwise

**E1 — OUT64: a noinline "append 64 B to the output ring" helper** (round 4–5) [MEAS]. ~300 calls per row block with lock bookkeeping each: norms 0.31–0.36 ms per op, 3x slower than their DMA bound; slot reservation per unit + direct ring stores: 0.10–0.13 ms (NOTES 10.1; SKILL §3). Lesson: per-call overhead dominates small-grain I/O.

**E2 — ROUT235: "the norms are output-port bound, R_out 2.35 B/cycle"** (rev 12 headline) [MEAS stamps]. ewio fit on four measurements gave R_out 2.35 (2.3–2.45) and pre-registered out_begin waits 147 (RMS) / 90 (LN) per unit. NIO stamps (round 17, r16z): out_begin 45.0 gross ≈ 5 net (stamp interval contains ~40 cycles of call; MACHINE R13.2); with NPIPE_G 2 46.6. Refuted (rev13/A §1). Why: H13 (s2 fixed at the bench value, the fit put the missing pass-2 time on R_out).

**E3 — NPIPE: interleave norm pass 2 of rb with pass 1 of rb + 1** (rev 12 rank 1, −0.55 (−0.54..−0.56)) [MEAS]. G 2: +0.07; G 3 / 4 / 6: p10 39.89 / 39.82 / 39.80 vs r16z p10 39.87–39.91 (G 4–6 −0.05..−0.15 by main’s reading) (RESULTS 26); with NIOC G 4 / G 12 +0.06..+0.13 (round 18, "dead"). NPROF: in_need 101.5 → 61 per unit, out_begin 45 → 47. Why: built on E2's wrong bound; each group pays n1k _end / _begin (~45 cycles; nrm_* use every dm and x register). G 6 kept in r17d.

**E4 — NG1 / NG2 / NS / N-IO-ASM: norm glue and statistics in asm** (rev 12 parked → rev 13 revived → round 19 capped) [MODEL + MEAS]. Rev 12 retired them to parked under E2 (NG1 ~0 after NPIPE, NG2 −0.09, NS ~−0.02). Rev 13 revived them (NG1 −0.44, NG2 −0.56, NS −0.34; N-IO1 −0.45..−0.55, N-IO2 −0.5..−0.7). The C++ form NIOC measured −0.17..−0.23 vs tcall static −0.55 (pass 1 waits on input; RESULTS 27.1). DDRCAP (round 19): the 3-input and RMS 2-input norms sit at the DDR read cap; only LN 2-input (norm3, 693 vs ~540 cycles per unit) is compute-bound: ≤ −0.2 ms. Conditional: revives where a norm becomes compute-bound again (after byte cuts such as NRES lower its DDR floor).

**E5 — RIN575: fitted EW input rate R_in 5.75 B/cycle** (rev 13 fit13) [MEAS]. Round 19 walls (r19d): 3-input norms 801 / 821 cycles per 3,072 B unit = 3.84 / 3.74 B/cycle (input-bound), 2-input 567 / 693 (RFM round 19). The fit's free per-op constant F and fitted s2 absorbed the n3 input time; its one miss was this signal (n3 − n2 wall 10.7 µs modelled vs 15–19 measured) (rev15/P norm plan). Rev 15's replacement explanation — ONE shim MM2S channel at ~3.8 B/cycle — was refuted in turn by DDRCAP (one stream ~7 B/cycle; 4 × 1 = 2 × 2 streams): the bound is the aggregate DDR read cap (8 × 3,072 B per 801 cycles = 55.5 GB/s). MACHINE R15.3 still says "one shim MM2S channel" (stale).

**E6 — FOLD: n_in 3 → 2 by summing D1 + D2 / D3 + D4 upstream** (rev 13) [MODEL, under a refuted fit]. −0.03 ms with the rev-13 ewio model (norms compute-bound after NPIPE G6, input waits ~12 per unit): retired. Conditional: under the DDR-cap reading (E5) the 3-input norms are bound by input bytes, so fewer input streams is exactly the lever — re-price with the cap before treating as dead.

**E7 — EWO / OSLOT: bench the EW output path; 1,600 B output slots** (rev 12 rank 6, OSLOT −0.3 if per-slot overhead) [MEAS by E2]. Retired rev 13: R_out is not binding. Not built.

**E8 — STORE-STALL: "stores into the EW output ring cost ~3.5 cycles per vector in situ (bench 1)"** (round 15 claim, open question to research) [MODEL, static whole-call count]. rev11/A §4: one out_copy(ring, off, src, 16) = 77 cycles for 16 stores (out_copy 20 + 5, cpy64 32, return 20); the x-parts phase = 3 calls + glue ≈ 290 = measured 289; DMA arbitration ≤ 0.1–0.2 cycles per stored vector; no stack traffic in cpy64. Refuted: fixed call cost / 16 vectors. Lesson: fewer calls (X1, X1b, P-IN, G1), not other banks.

**E9 — NORM_SPLIT: statistics as a second pass over the stage** (round 15) [MEAS]. full12r15j 42.13 vs r15i 42.09: 0 (pass 1 was not the norms' bound); kept inside NRMK only (RESULTS 24.1; NOTES 20.4).

**E10 — NRMK-COPIES: norm pass 2 with X' / panel copy passes into the output ring** (round 15) [MEAS]. full12r15k2 42.67 = +0.6 vs r15i; with direct ring stores / pushes (r15k3) −0.40. Why: per-call cost of out_copy (E8).

**E11 — N1-PRICE: norm pass 1 + statistics in asm priced −0.6 ± 0.25** (rev 11) [MEAS]. −0.16..−0.28 (r16f → r16h). Why: rev 11 priced the 23-bundle C++ body against a 10-cycle asm cb and assumed the per-unit glue would go; it stayed (in_need 28–41, pointer setup 36, in_done 39, loop 13) and in_need waits 104.8 cycles per unit (norms partly input-bound) (rev12/A §0). Whole-call static −0.32 was a hit. The twin that "confirmed" −1.98: H17.

**E12 — AL1-1K: pad every EW op's params fill to a 1 KB multiple** (rev 12 rank 3, −0.21 ± 0.05) [MEAS]. 1 KB padding (11,264 B per SSM head) HUNG the step (timeout): the shim → core packet BD completes per 768 B slot, so a fill group must total a slot multiple. Built with 3 KB multiples: −0.05..−0.15 (p10 39.82, mean 39.87); half the static price did not appear (unexplained; candidates: fewer straddles in situ, +98 KB per column of padded params) (RFM round 17; rev13/A §0).

**E13 — SIG-SYM: symmetric LUT sigmoid on [0, 8) for the RG-LRU** (round 6) [MEAS emulation]. 5.1 % vs 18.6 % bf16 mismatches, but 1 − s loses the relative accuracy of small sigmoids and gate_a feeds a = exp(−8 sp σ) near 1: |da| / (1 − a) 1.6e3 vs nk8 1.4e-2. Rejected; a direct [−16, 8) table needs 3 KB per copy and no EW bank had room (NOTES 11.2; RESULTS 15.5). Revived differently in round 9 (P28: one asymmetric [−12, 8) table, −2.8 ms with sqrt).

**E14 — EXP-2F: cheaper SSM exp variants** (round 9) [MEAS]. Range-reduced 2^f table: as accurate as nk8 / exp32a, not faster (61.5 vs 60.7 cycles per 32 lanes); direct PWL exp tables on log_a: 29 % / 17 % a mismatches (rejected); log_a fused into pass A: 50 bundles per vector, not pipelined (NOTES 14.3; RESULTS 18). The chain went to asm later (SSX, −0.79).

**E15 — ZASM-BENCH: "the asm z loop is no faster than C++"** (round 15) [MEAS — sources disagree]. One-core bench: asm 100 vs C++ 97 cycles for 8 z panels → dropped (RFM round 15 dead ends). Round 18: inside ew_ssm Peano compiled the same C++ serially (27 bundles per 64 lanes, zol @0x1610 in r18a); ZK with r15/gen_zk.py's loop −0.14..−0.19 (−0.17 A/B) (rev13/P; RESULTS 27.1). The dead end was a bench artefact (H10).

**E16 — X-PRESSURE: sqrt fused into ssx below II 27; FSX inside SSQ** (round 16; rev 11–16) [MODEL CP]. No schedule ≤ 26 (x pressure): SSQ at II 30 (−0.12..−0.15). FSX (gate_a's sigmoid into ssx, rev 11 −0.24 ± 0.09) faces the same pressure; rev 13 re-priced −0.07 (sigk's glue 45) and parked; rev 16 "ssq2 + FSX −0.18" unbuilt (RFM round 16). Conditional: needs a schedule that fits 12 x registers.

**E17 — X1-SLIDE: sliding SSM area + wrap frame** (round 16–17) [MEAS]. Built: −0.33 per rotating frame; the wrap frame (copies) = r16z speed; ACT +232 MB at N 8. Superseded by the mod-4 rotation (no wrap frame, single-TAP fills), same −0.33 (RFM round 17 X1). Not a failure; do not rebuild the sliding form.

**E18 — P7-FOLD: fold the norms into the GEMM prologue / epilogue** (rev 6, −4.6 ± 1.5) [MODEL]. Parked behind P7-E (main round 11 accepted): blockers GEMM L1 (932 B free), GEMM text (544 B), an A-path residual protocol, a double drain; activation-side small-g channels are the numerics risk (P7a weight side fine: error ratio 1.013–1.023) (rev6/P P7). Conditional: L1 and text are fuller now (576 B / ~400 B free); NRES already removed the panel round trip for norm → GEMM boundaries.

## F. Host, packing, numerics

**F1 — HEAD-1T: the output heads on one thread** (round 11, HEAD_THREADS=1) [MEAS]. Heads on one thread made the NEXT conv slower (OpenMP pool re-spin after the idle): not used (RFM round 11 host part; NOTES 16.3). Number not recorded.

**F2 — OPENBLAS-THREADS: numpy heads with OpenBLAS's default threads** (round 13, H3) [MEAS]. 11.7 ms beside torch's pool; with OPENBLAS_NUM_THREADS=1 0.72 ms median (deployed as HOST_HEADS=np) (RFM round 13 H3; NOTES 18.4).

**F3 — H1 / H2: host overlap levers** (rev 7–8) [MODEL]. Not attempted: the product runs in latency mode, where H1 cannot help (RFM round 13). H2 parked since rev 9.

**F4 — BF16C-64K: bf16 C between 64-K chunks (mlir-aie ATB style)** (moondream phase 1, 493 frames, bfpemu) [MEAS emulation]. Frame accuracy: HLin (GPU bf16) 98.0 %, AccLin (bf16 acc every 64 K) 97.8, BfpLin (bfp both, bf16 acc / 64) 96.6 (−1.4 points), BfpAcc32Lin (fp32 acc over all K) 97.6 (MD RESULTS precision table). Lesson: keep the accumulator fp32 across the whole K (TAPNext does).

**F5 — HADAMARD: 64-wide Hadamard rotation along K with bfp (per-8 shared exponents)** (moondream phase 1) [MEAS emulation]. BfpRotLin 96.1 % vs BfpLin 96.6 %: hurts; dropped (MD RESULTS l.92–95).

**F6 — INT8: int8 vision / int8 mmul** (moondream) [MEAS]. Per-token int8 vision dropped frame accuracy to 89.2 % (vs 98.0 %) (MD NOTES l.64). The AIE2P dense int8 mmul is 8x8x8 = 512 MACs per instruction, the same rate as bfp16 (the 1,024-MAC form needs a sparse B): int8 buys nothing in MACs and 11 % in bytes (MD RESULTS ~l.305).

**F7 — HW-PRIMS: hardware vtanh / aie::exp2 / aie::inv** (round 6; moondream v14) [MEAS on every bf16 input]. vtanh is the identity for |x| ≤ 0.5, 7 % off near 0.5–1, 2 % at 1–2 (RESULTS 15.4 row: 1.73 % / 1.21, 19 at 0.5); aie::exp2 p99 10 ulp (RESULTS 15.4: 53.3 % / 10.3; moondream: a ~4-bit interpolant, 8 % rms on a ramp); aie::inv correctly rounded but scalar (32 lane extracts) (RFM round 6; RESULTS 15.4; SKILL §3). Use the LUT / nk8 routes.

**F8 — MAX8: 8-lane bf16 aie::max** (moondream / round 5) [MEAS]. Returns wrong values; use 16 / 32 lanes (SKILL §3). Number not recorded.

**F9 — NEWTON-RECIP: one Newton step in the reciprocal (cheap GELU / sigmoid)** (round 5–6) [MEAS real activations]. bfp ratio 0.4–0.6 = visible on the next tensor; the quadratic 2^f costs nothing (0.059 / 0.265 = nk8) (RFM round 6 P8; RESULTS 15.5). Note: round 5 had rejected cheap GELUs on per-element bf16 mismatch; the bfp-ratio yardstick (P8) is the criterion now.

**F10 — P24c: bias as the accumulator's initial value** (rev 6, parked) [MODEL]. Removes bias loads / MACs from every epilogue but changes the fp32 order: not bit-identical, DAVIS-gated (rev6/P "Also small"). Not built. Conditional: only if a DAVIS round is accepted for it.

**F11 — P66 / P67: fewer lazy-max flags by numerics** (rev 9, parked) [MODEL]. P66 seeds the stored max with q_i · k_i (flags after chunk 0 2.26 → 0.27 %; A load and ring traffic 2x ~0.4 ms unmeasured): ~−0.3 ± 0.4 after P60 + P61. P67 TAU 8 → 16 nats needs the exp table to +17 (+1 KB of L1 that is not there). Not bit-identical (rev9/P). Conditional on a DAVIS gate and L1.

**F12 — FA4-F1F2: FlashAttention-4 style software exp split** (rev 10) [MODEL]. F1 (SSM, 2^f for half the vectors by a 256-entry int16 table) −0.05..−0.15 (−0.1 ± 0.05); F2 (attention) ≤ −0.1: attkb's II 34 is register-bound, not port-bound; both within 1 bf16 ulp, not bit-identical: "not worth a DAVIS round" (rev10/P). Also FA4 LPT → P60h (C23).

## G. Toolchain / Peano / IRON traps that killed an approach

**G1 — OZ-MINSIZE: -Oz / -Os / minsize on hot code** (round 4–6) [MEAS]. EW object at -Oz LARGER than -O2 and 2.4x slower (non-inlined AIE-API helpers spill every live vector) (NOTES 9.2); -Os / -Oz ~5 % on the GEMM loop; the round-5 minsize epilogue ran ~77 bundles per 64 elements vs 5.5 for a noinline -O2 pass ("up to 10x") (SKILL §3; RFM round 6 P11, −5.3 ms). Use minsize only on cold code; it does not shrink vector-heavy code either (V^T epilogue).

**G2 — POSTPIPE: `AIE_PREPARE_FOR_POSTPIPELINING` on the exp loop** (round 6) [MEAS static]. 100 bundles per trip vs 65 (NOTES 11.2).

**G3 — LINEAR_APPROX: aie::linear_approx for table lookups** (round 6) [MEAS]. 2.7x slower than the hand-written `load_lut_2x_float` lookup: its object is reloaded from the stack on every compute (RFM round 6 GELU LUT).

**G4 — PEANO-CHAINS: more than two interleaved dependency chains in C++** (round 4–5) [MEAS static]. Two chains: exp 73 → 38, sigmoid 140 → 77, gelu 199 → 111 bundles per vector; four chains spill and gain nothing (NOTES 9.2; SKILL §3). An -O2 loop with 4 fp32 accumulator chains spills every iteration (5 dm registers) (NOTES 20.1).

**G5 — RESEEK: re-seeking a block-vector input stream after popping from it** (round 7; moondream v12) [MEAS]. Garbage (t4 O = inf); moondream: NaN (fill pointer not where assumed after a partial 64 B word). Construct a fresh stream object per block (NOTES 12.3; MD NOTES).

**G6 — SCHED-BARRIER: read_tm stamps (sched barriers) inline in ew8's pipelined chunk loop** (round 8) [MEAS]. Crash clang (AIE Region assertion); stamp through a noinline function (NOTES 13.3).

**G7 — PEANO-SELECT: vectors assigned under runtime conditions / aie::select** [MEAS]. Peano crashes ("Bad PHI"); mask arithmetically (SKILL §3). Also `#pragma unroll` over a loop with a conditional vector reference stays rolled: unroll by macro.

**G8 — IRON-MCAST: unicast + multicast packet ids sharing two DMA master ports (P5a first version)** (round 10) [MEAS]. IRON 1.4.4's pathfinder silently drops a multicast destination (ids 1 → DMA0, 3 → DMA5, 4 → both: DMA5 had no rule for id 4); one dispatch hung (r10/pf1.mlir vs pf2.mlir) (RFM round 10). Fix: multicast-only routes.

**G9 — IRON-2CH: two packet flows from two DMA channels of one tile merging onto one port** (round 20, NRES) [MEAS]. Pathfinder finds NO route (even one column; from one channel it routes). Workaround: declare both from MM2S 0 and patch the lowered switchbox (`packet_rules(DMA : 1)`, r20/nresfix.py); never run the unpatched xclbin (RFM round 20; SKILL §3).

**G10 — IRON-BD: shapes IRON / the BDs cannot express** [MEAS / DOC].
- Tile BDs refuse stride 0: P5a's "write twice into half 1" was impossible; K = 1536 ops write the whole slot over both channels instead (round 10).
- padds / padda immediates must be multiples of 64: no per-frame bump of a 40 B record (round 9, P20c).
- Two BDs for a joined fifo's producer (CSPLIT): not expressible in IRON; patch the lowered MLIR and re-run aiecc (round 19). An acquire-only BD is legal in hardware; its lowering is UNVERIFIED (blocks CSPLIT4).

**G11 — BDSTEP20: a BD step larger than 20 bits (words)** (round 4) [MEAS]. norm3 took 6 ms: its 2-input fill had a 10.5 MB step; the transfer was correct but the columns ran one after another. R4 now follows R3 (NOTES 9.2).

**G12 — ADDIMM7: `add rX, rY, #imm` is 7-bit signed** (round 18) [MEAS]. `#64` assembled silently as `#−64`: V2T hung once (1 timeout). asmlint IMM-SRC checks the source (RFM round 18).

**G13 — ZOL-ALIGN: zero-overhead loops** (round 11–12) [MEAS]. A body whose LAST bundle is not 16-B aligned runs ~once (copy probes "ran at 1 cycle per vector"); a 2-bundle body runs once (needs ≥ 3 bundles). Write body bundles in the full 16-byte form (RFM round 11 P45 item 5; round 12 lessons).

**G14 — P2ALIGN-PAD: the `.p2align 4` before the body inserts a nop bundle** (round 12) [MEAS]. Prologue ops then collide with body ops on itinerary ports (a vlda's QEY_WA vs a vfloor): bgelu II 11 had iteration 1 wrong → rejected, II 12 deployed. msched.check replays pads 0 / 1; asmfit fits the prologue (RFM round 12).

**G15 — FIFO-POP9: more than 8 dense 576-bit pops on one FIFO without a fill** (round 16) [MEAS]. Pops 8–12 read zeros / half exponents, then resync (popprobes 6 / 7); the first ggk failed this way (G1 built at II 24). Restarted streams of ≤ 8 pops never see it; asmlint rule (RFM round 16; round 17: true positive on popprobes 2 / 3).

**G16 — LOCK-ABI-HANGS** (rounds 9, 13) [MEAS]. `acq` with value 0 = acquire_equal 0 (a hang); loop start not 16 B aligned + uninitialised r25: 2 timeouts in round 9 (RESULTS 18.4). Peano's ABI with a return value reserves r0, the first scalar argument comes in r1: a trip count read from r0 ran 5 trips, or 0 → hang (2 timeouts in round 13; NOTES 18.4). Also: scalar `lda pN, [sp, #imm]` uses DM_ADA and co-issued with vldb.4x64 loaded garbage; scalar alu results read at +1 give the OLD value (an OR at FO + 1 missed the last flag).

**G17 — READ_TM: in-kernel cycle stamps via read_tm(0x340F8)** (round 6, P2 "BLOCKED") [MEAS]. Hung the core (1 timeout); no cycle builtin in Peano. Resolved in round 7 (P15): enable the processor bus (npu_maskwrite32 0x32038) and fence each read with sched_barrier. Listed as a trap: the approach was dead for one round.

**G18 — TEXT-16K: 16 KB program memory** [MEAS]. Killed or parked: S3 / V1 / K1 / K2 in round 17 (GEMM text 48 B free after S2), K1C (+96 B), the rolled rescale_rb (+96 B), PROF + MM_ASM together (round 7), the EW PROF build in round 12 (784 B over), the asm z loop and copy paths until XROT / AL1 compiled paths out (EW 16,384 overflow by 64–80 B, round 17). IRON lowers worker loop counters to i64 (~6 extra scalar ops per loop; the i32 lever IV32 is UNVERIFIED). Check `llvm-size` after every kernel change.

## H. Methodological dead ends

**H1 — CONSTANTS: the "1.43 GHz" clock and the "4 B/cycle" A port** (rev 1, round 5) [METHOD]. The clock came from moondream phase-2c's spin loop (a 364-bundle call in 0.261 µs, a lower bound read as the clock; 1.408 GHz is also DPM level 5's H-clock); the driver's DPM table said 1.8 GHz all along. Measured: 1.8086 / 1.8076 GHz (r7/clk.py, RFM round 7 P10); memtile → core 7.95 B/cycle (round 10). The two errors compounded: the real A floor was ~0.72 µs, 0.4x the assumed 1.813 (rev2/A §0; MACHINE §1). Lesson: every conversion constant carries a source tag; [DOC] beats an old [MEAS] that measured something else.

**H2 — ONE-NUMBER: a single number matching a bound** (rev 1; round 9) [METHOD]. Rev 1's t_ks 1.857 vs floor 1.813 (2.4 % match) "decided" the port bound; the compute hypothesis was costed loosely (moondream's 1.59 bundles per MAC, an invented 115-cycle overhead) though mm_acc's own 321 bundles (1.646 µs at 1.8 GHz) matched as well (rev2/A §0). Round 9's 2,588 = 10,368 / 4 was likewise a coincidence (rev5/A §0). Lesson: report "consistent with" and run the perturbation first.

**H3 — FREE-CONST: fitted free constants absorb what the stamps have not separated** (rev 2–4) [METHOD]. sim9 fitted t_ks 1.835 µs and called 11–13 % above static "stalls"; stamps: 9 × 333 + 401 vs 3,383 static (15 cycles apart) — the excess was dependency / coupling / tile-boundary time (rev3/A §0). sim10's fitted t_tile 7.6 µs held op-start waits and ~9 % of k-step time (rev4/A §0; B1). Lesson: compose from the binary; fix every term that has a stamp; fit at most one.

**H4 — MEAN-VS-MEDIAN: in-situ k-step means** (round 9 → 10) [METHOD]. Means 2,588–3,320 cycles per k-step vs medians 2,050–2,065: one tile per op (the first) carried the op-start wait (4,170–7,300 per k-step averaged over it) and moved the mean 25–60 % (rev5/A §0; A3). Lesson: report median and distribution, never only the mean.

**H5 — PROF-COUPLING: row coupling read from PROF per-column max walls** (rev 5, P23) [METHOD]. sim12 priced qkv's coupling at 80 µs per block from the g11 PROF wall (337 µs); the non-PROF timeline had 284 (only 21 above composition): qkv predicted −107.5 vs measured −31.2 µs per block (−76); P23 −0.85 vs −1.2 (rev6/A §0). Lesson: coupling and op starts from the full-step timeline.

**H6 — STALE-BASE: baselines from an older round** [METHOD]. P29a −0.61 vs −1.0 (round 9's 1,168-cycle PROF phase, the deployed C++ was 898); P52 −0.18 vs −0.71..−1.14 (round 9's 798); X1b −0.28 vs −0.18 (r15b's 289, r15m had 432); P5r up_b's round-9 records incomplete (rev6/8/11/12 scorecards). Lesson (rev8/A §0.2): price only from the deployed build; deltas like-for-like.

**H7 — GUESSED-COMPONENT: an unmeasured component inside a composed price** (rev 7 G1a) [METHOD]. "Half a stash copy" = 400 cycles; measured 1.70 µs = 3,080 cycles per tile (7x off): G1a −0.55 vs −1.02 ± 0.2 (rev8/A §0). Lesson: every component measured or flagged with a wide bar.

**H8 — LOOP-VS-CALL: pricing the loop, not the whole call** [METHOD]. P50: bench pre-registration 80–110 cycles per row block, measured 136 = loop 87 + sums pass 21 + calls / glue 27 (rev10/A §0); P29b −0.79 vs −1.0 (7 stages × II 20 = 120 cycles of prologue + preamble; 497 per chunk); SQK −0.18..−0.25 vs −0.31..−0.35 (~30-bundle preamble + ret); N1 (E11). Lesson (rev 11): whole-call static counts (tcall): body + call / ret delay slots + loop setup + prologue + drain + glue.

**H9 — BENCH-NOT-DEPLOYED: benching a binary without checking it is the claimed loop** (round 10, rev 6) [METHOD]. The "17-bundle" forced-II GELU was serial in both the bench and mm8 (B2); rev 6 then costed hazard hypotheses on it (B3). Lesson: disassemble the benched object; bench the binary you deploy.

**H10 — BENCH-CPP: a one-core bench of C++ is not the in-situ code** (rounds 15, 18) [METHOD]. The deployed conv loop ran 377 cycles on one core, ~1,000–1,125 in situ (P[] via the stack, in_part, ring stalls; 3x); the z loop pipelined in the bench (97 for 8) and was serial in ew_ssm (27 bundles per 64 lanes) — which made E15 a false dead end (RFM round 15 P29d; MACHINE R13.8). Lesson: price C++ phases from in-situ stamps / the deployed disassembly.

**H11 — PRERELEASED-LOCKS: a bench that pre-releases every lock** (round 18 g2f) [METHOD]. The ROT bench pre-released all nine A slots, so an acquire 1 bundle before the fill never blocked: bench 36/36, in situ raced (A17). Tool since: r20/rotdb.py (DMA-fed A ring), asmlint LOCK-ACQ-GAP.

**H12 — BENCH-COVERAGE: benches that do not look where the bug is** [METHOD]. bgelu stored x in an early stage: the drain corrupted the 4 vectors after every run in situ; the bench did not check past the end (round 12). popprobes 2 / 3 dumped only lf0's 8 pops, lf1's 16 dense pops unchecked: "correct" in round 16, FIFO-POP9 true positive in round 17. A scalar flag OR at +1 missed the last iteration: bench passed, in-situ regress did not (round 14). mm8_rot.inc includes "build/mm_rot.h" → the g2a bench first ran with g2f's border (0/36) (round 18). The mm_ks bench (2,200) ran slower than in situ (2,055): its buffers shared banks (rev5/A §0). Lesson: guard zones both sides, every stream, the header you mean.

**H13 — EW-FITS: rev 12 / 13 EW I/O fits with a nuisance parameter** [METHOD]. Rev 12 fixed the pass-2 in-situ cost s2 at the bench value and the fit put the missing time on R_out (2.35; passed five checks); "R_out ≥ 3.5 with s2 ≈ 4" fitted the same deltas (r15i −0.54, NRMK −0.45, N1K −0.22 vs −0.48 / −0.42 / −0.18) (rev13/A §1). Rev 13's R_in 5.75 had a free per-op constant F and s2 that absorbed the 3-input time (E5). Refuted by stamps on the hypothesised wait (out_begin ~5 net) and by walls (3.8 B/cycle). Lesson: before pricing levers off a fitted bound, stamp the wait itself.

**H14 — CP-FIT: rev 15's boundary model with a free t_op** [METHOD]. Fitted t_op ≈ 0.17 µs per CP instruction reproduced six op boundaries within ~5 µs, but t_op absorbed the 12 µs producer tail (C2); measured 0.06 / 0.11 µs (CPRATE) and CPST put 12.2 µs before the first tct. Lesson: name the probe that separates the hypotheses before ranking a lever on a fit (CPST was it).

**H15 — FIVE-MS: "op boundaries cost ~5 ms per step"** (rev 15 headline 1) [METHOD]. The GEMM timeline "gap" contains the op's whole first tile; decomposed (rev 16): pace 0.87 + tail / tct ~1.3 + issue ~0.4 + norm / SSM ~0.6 = ~3.0–3.3 ms.

**H16 — POLL-QUANT: medians of host-poll timelines** (rounds 19–20) [METHOD]. Polls come every ~16.5–17.5 µs, so per-op median gaps are 0 / 16.5 / 33 µs: "every norm gap 16.5 µs" (r19d) is an artefact; means over blocks 1–11 are 8.6–11.0 µs (r21/tlhist21.py; MACHINE R16.4). Earlier: the 17 µs quantum showed as a two-level 148 / 165 pattern for mlp3_a (rev8/A §0.1). Lesson: means over ≥ 8 blocks or effects ≥ 3x the quantum.

**H17 — TWINS-STEP: twins priced from step deltas** (round 16; rev 12) [METHOD]. NORM_TWIN=2 removed 1.81 ms of static compute and saved 1.98; NORM_TWIN=4 removed 1.03 and saved 1.53–1.61; the asm N1 then gave −0.16..−0.28 (RFM round 16; rev12/A §0). Inferred (UNVERIFIED) cause: wrong outputs change downstream data-dependent work (the attention lazy-max fallback, ~0.9 ms at full rate). Also r14cF2 (50.33) unusable: skipping chunk 0's exact path flagged every later row block. Lesson: read a twin's per-op walls, never the step.

**H18 — KAPPA: the CPU lazy-max map overstated in-situ flags; "lockstep x6"** (rev 9–10) [METHOD]. In situ ~1.36 % of row blocks in chunks 1–21 move the max vs 2.26 % in the CPU map; raw map predicts r14c − F128 +3.77 ms vs +2.08..+2.53 measured; κ 0.6 fits (slopes 0.98 / 1.05, slow units 9.9 % vs 8.6–10.9 %) (rev10/A §1). sim16's "lockstep x6" came from a too-hot map with fitted T 66 / R 300 compensating (T + 0.1 R = 96 vs 222 measured) (rev10/A §0). Lesson: one fitted constant (κ), every per-row-block cost a one-core measurement.

**H19 — RARE-MEAN: rare per-unit costs priced by their mean** (rev 8–9) [METHOD]. Under the K / V multicast the 8 columns advance in lockstep per chunk; ~10 % slow units cost ~3x their own time on the wall (P56: +1.35 ms for ~0.45 own; P63: 2.1–2.6 ms for ~0.35) (D11, D12). Lesson (rev9/A §0.1): price non-uniform costs by the max over columns per K / V object and validate per block (attblk corr 0.93–0.97).

**H20 — BIMODAL-60: single 60-iteration medians under foreign CPU load** (rounds 14–20) [METHOD]. With a foreign worker (100 % of one core) dispatch times split into two clusters ~0.17 ms apart; a 60-iteration median lands in either (P63 −0.04..−0.47 "bimodal"; round 17 NOTES 22.3). Since round 18: both builds in ONE process, interleaved blocks (r18/ab.py), p10 / median / mean of 300–600 each; a second 7.6-core foreign job in round 20 widened spreads again.

**H21 — POSTHOC: hits that were not pre-registrations** (rev 15–16) [METHOD]. Rev 15's draft pre-registered hand 1,900–2,700 / plain epilogue 1,350–1,600 after main had measured 3,010 / 1,310; rev 16's DDR-cap pace model (7.0 / 14.0 µs vs 6.5–8.5 / 13.3) came after main's measurement. Neither counted (rev15/P C-object note; rev16/A §0).

**H22 — FALSE-NULL: stale jit cache and #undef-ed flags** (rounds 15–16) [METHOD]. IRON's jit cache key excludes source files: a kernel edit under the same build name reused the old build (a PROF phase read 0) (NOTES 20.1). `-DPIN` was #undef-ed by an included header: full12r16b 41.38 "dead build (same elf)" (RESULTS 25.1). Lesson: new build name after every source edit; check the elf md5 / llvm-nm sizes changed before timing; long specific flag names.

**H23 — BUILD-RACE: parallel builds and shared generated headers** (rounds 13, 16, 18) [METHOD]. Two design builds in one build dir race on the generated asm (asmfit temp files; spurious "bundle not 16 B aligned") (NOTES 18.4); per-build object dirs do not fix it because mm8.cc includes "build/<k>.h", resolved next to mm8.cc first (MACHINE R11.9). Build sequentially / under the build lock (tools/try.sh, round 16).

**H24 — OBJDUMP-Z: `llvm-objdump` without -z** (rev 10) [METHOD]. Runs of 2-byte nops (zero bytes) are elided as `...`: loga_p showed 11 bundles instead of 23 (rev10/A §0.1). Always disassemble with -z (tools/loops.py, bundles.py assume it).

**H25 — REGION-AS-FREE: reading linker regions or uncalled code as recoverable space** (rev 7, rev 10) [METHOD]. Rev 10 read bank 3's `.aie.bank3` REGION (0xA80 = 2,688 B) as free; sgelu_cd + serf_cd fill 2,304, leaving 384 B: S2 not buildable as designed in round 15 (built later on the stack, −0.31) (rev11/A §0). Rev 7's "1.4 KB of EW text in ew_patch / gemm8": not in the elf, the linker drops uncalled entry points (RFM round 13 T1). Lesson: free = region length minus the sections in it; read `llvm-nm -n` of the linked elf.

**H26 — NUMERICS-30F: judging SSM numerics by 30-frame max deviations** (round 9) [METHOD]. Any ulp-level SSM change (deployed nk8, tables, random 1-ulp flips of 5 % / 18 % of gates) gives the same chaotic track noise (~0.015 px mean, 0.1–0.2 px max over 30 frames) (RESULTS 18; NOTES 14.2). Judge by DAVIS (AJ 70.84 / 69.90 unchanged since round 9).

**H27 — HALF-BUILT: a partial build scored against the whole proposal** (rev 6–7 P43) [METHOD]. Round 11 built only `fill_b()` before `finish_deps` for mlp3_a; the after-EW part (−28 µs) and the deeper B forward were never tried; the miss (−0.35 vs −0.8) then read as a model error (rev7/A "Why P43 missed"). Lesson: state which code path each part of a saving depends on.

**H28 — STAMP-CONTAINS-CALL: stamp intervals include the call** (rev 13) [METHOD]. NPROF in_need 101.5 gross ≈ 57.5 net (35-cycle path + delay slots + stamp ≈ 44); out_begin 45.0 gross ≈ 5 net (MACHINE R13.2). Comparing gross stamps with modelled waits mis-sized every EW wait before rev 13.

## Where the sources disagree (resolve before citing)
- **D13 chunk-0 rescale skip:** NOTES 18.5 "no measurable gain, reverted" (round 13) vs RFM round 14 "worth ~−0.3".
- **E15 asm z loop:** RFM round 15 dead end ("no faster than C++, 100 vs 97") vs round 18 ZK −0.17 (in situ the C++ was serial).
- **B2 GELU_II:** NOTES 15.3 still lists "GELU_II slower in situ" as a dead end; RFM round 11 shows it never pipelined and the fixed form gave −0.88.
- **C12 / E5 DDR:** MACHINE R15.4 (≥ 84 GB/s) and R15.3 (norm input = one shim MM2S channel at 3.8 B/cycle) are stale against DDRCAP (aggregate read cap ~55–59 GB/s, one stream ~7 B/cycle). Rev 15 called MACHINE §5's 55–63 GB/s "8 × the single-channel rate"; DDRCAP restored it. SKILL §7 / MD keep "~55–60 peak" and "~4 B/cycle per column with all 8 loading", consistent with DDRCAP.
- **D12 3D pops:** RFM round 13 says the 36-iteration form "needs the 3D Q pop stream"; round 14 built it with 2D modifiers.
- **C4 SEQPRE price:** rev 15 raw −1.37..−2.31 at t_op 0.13–0.22; at CPRATE's measured rates main re-priced ~−0.6..−1.0 raw; measured best −0.04.
- **H17 norm twin:** RFM round 16 "twin said 1.98 ms"; RESULTS 25.1 r16tw2 38.20 vs r16f 40.18 / r16h 39.96 gives 1.76–1.98 depending on the base.
- **C17 NRES:** rev 16's rb-major design vs the built chunk-major 3D-TAP scatter (the design was wrong, not the idea).

## I. Added by research pass 17 (2026-10-11)

**A24 — E7L-STACK: free GEMM bank-0 space by shrinking the stack** (pass 17) [MEAS stackaudit]. full12r20z GEMM cores: deepest path 1,856 B
+ attkb's own 64 B frame against a 2,560 B stack (0xA00): at most ~512 B can go (2,048 keeps the audit's 128 B margin). Lesson: L1 levers
that need bank-0 room must name the bytes; the GEMM core has ~576 B free in total.

**A25 — E7L-LUT: move the GELU tables' bank-0 copies to bank 3** (pass 17) [MODEL]. sgelu_ab / serf_ab (2 × 1,152 B) are the bank-0 halves
of the parallel LUT pairs whose other halves (sgelu_cd / serf_cd) live in bank 3; both copies in one bank cost +1 cycle per 4x load
(+8 per 32 lanes, round 9 lprobe) on every GELU epilogue. Not built. aexp_ab (1,280 B) can move instead: its partner g_lcd is in bank 2
and attkb never co-issues a LUT load with a bank-3 access (r22/full12r20z_0_2.dis).

**A26 — E7-ABS: serve row block 0's B pops from row block 0's own stream instance** (pass 17, gen_e7 --absolute) [MODEL, CP]. Meant to
shorten the prologue; INFEASIBLE at R = 64 even with banks, B registers and ports dropped (r22/diag64_absolute.out), UNKNOWN at R = 68
in 150 s: one row block's 64 pops span ~85 cycles, one FIFO instance at most R − 3. Residue matching (+ the previous period's segments
in the prologue) is the working form; the longer prologue is what KTL amortises.

**A27 — E7-R64-NX8: E7'' at R = 64 with a 576 B X segment** (pass 17; main round 21) [MODEL + MEAS]. Bank 3 must take 63 line reads in 64 cycles (56 panels + 7 fills). No conflict-free schedule in 25 + 7 min; with the bank-3 cap as a cost CP-SAT returned 4 excess residues + 55 soft conflicts: replay 1,244 static + 129 stalls = 1,373 vs R 68's 1,338 + 0. Main measured it in situ (full12r21e5 vs r21e3): +0.04 / −0.04 / −0.08 ms, neutral, as the replay said. Lesson: price by static + replayed stalls; the A-port floor (1,304 per k-step) also caps what R < 68 can buy.
panels + 7 fills); no schedule found in 25 + 7 min (r22/solve_r64.log, diag64.out) while every other constraint group is satisfiable
(dropping only the bank-3 cap: FEASIBLE in 300 s). Not a proof; see rev17/P for nx = 12 / R = 65.

**A28 — ROT-KTL: one modulo loop over a tile's k-steps with ONE B FIFO** (pass 17, retracted from the first rev-17 draft) [MODEL]. Consecutive k-steps read different B buffers, so no stream instance may serve two row blocks across a k-step boundary; one row block's 64 pops span ~85 cycles (8 jobs + a 21-cycle chain) > R = 72, so a single FIFO's consecutive instances would overlap. KTL needs E7'''s two FIFOs with every segment instance serving one row block (gen_e7 --aligned / --ktl; lf1 split L | N): CP-feasible at R = 70 (r22/solve_k_r70.log). Lesson: a loop-carried buffer switch needs segment-aligned streams; check what one instance serves before pricing a fused loop.

**B13 — CSUB-G: gated tiles in the CSUB group loop** (main, round 21) [MEAS]. +0.027 / +0.021 / +0.023 ms (predicted −0.04..−0.08), bit-
identical; kept only as a text lever (−256 B).

**B14 — RTC: runtime-driven core C drains with per-op lengths** (pass 17) [MODEL]. Drops 74 MB of APAN junk writes per step, but the junk
sits in part 2 (and the last 1,920 B of part 1) and the GEMM steady tiles are not write-bound: ~−0.05..−0.1 ms for 24 cores × ~17 runtime
pushes per block on the in-order CP (4-deep queues). Not proposed.

**C24 — 3A-HOLES: a third A region in today's memtile 0–2 holes** (pass 17) [DOC mtmap]. 128,000 B free but in 5 holes (largest 56,320);
12 chunks of 10,368 B need 124,416 contiguous (A_RT BdIteration). Needs the B fifo objects and C joins repacked (or l2 depth 7), see
rev17/P 3A.

**C25 — NTAIL-PF: prefetch a norm's independent inputs during the producer's drain tail** (pass 17) [MODEL]. The EW ring holds ~5 units:
≤ ~10 KB per column can load early (≤ ~1.5 µs per boundary, ≤ 0.07 ms per step) and it competes with the drain at the shared DDR cap.

**G19 — ACQ-ONLY-BD: a DMA BD block with only an acquire** (main, round 21) [MEAS]. The MLIR verifier refuses it; a release by 0 is the
working form (csplit.py --sub). The hardware fields are independent (MACHINE R16.8).

**H29 — RESIDUE-PROLOGUE: per-call prologue left out of a k-step price** (pass 17) [METHOD]. A modulo k-step's whole call is 18 R + setup
+ prologue + tail; at R = 68 gen_e7's first schedules carried ~100 cycles of non-body (ROT g0a: 61) and the per-call saving shrank from
72 × 18 to ~20 cycles. Pre-register the whole call (tcall) and the mm_ktile glue (13 cycles per k-step), not R.

## R. Revived — once recorded as dead, later shown to work (do not cite them as dead)
- **B2 GELU_II** (−0.88 ms once the loop really pipelined), **E15 ZASM-BENCH** (ZK −0.17 in situ), **D13 CHUNK0-SKIP** (~−0.3 in round 14),
  **C17 NRES** (the idea, built chunk-major: −0.84 ms with norm1), **A13 B-2BANK** (void with one FIFO; E7'' splits B over two banks with
  two FIFOs and is live), **E6 FOLD** (retired under a refuted fit: re-price under the DDR-cap reading).
- **J4 YRES / SSPM** (pass 20): "the SSM is compute-bound, byte cuts ~0" no longer holds after YXSSM: the SSM's span is at the DDR
  read cap, so read bytes in it pay and compute cuts do not. SSPM (params read once, broadcast) is PROPOSALS rev 20 rank 1.

## J. Added by research pass 18 (2026-10-11)

**J1 — RADD / KSUM: the residual add (or the K-split partial sum) in the producing GEMM's epilogue** (pass 18) [MODEL]. Would turn the
3-input norms (norm0 = x + mlp3_a + mlp3_b, norm2 = x + down_a + down_b) into 1–2-input ones (−0.5 ms of DDR-bound norm time) and stay
bit-identical (same fp32 order). Blocked: each GEMM core has two input ports (A from the memtile broadcast, B the multicast weights); the
residual / partial tile is per core (its rows × its N columns), so neither stream can carry it, and there is no third S2MM. Lesson: check
the per-core input ports before moving an operand into a GEMM.

**J2 — EW-PRESUM: the EW core adds x + d1 while the second K half runs** (pass 18) [MODEL]. Moves 5.3 MB per op (read x, d1; write
x + d1) into mlp3_b / down_b, which already run at ~66 GB/s read + write (r21 ddrbudget): no DDR headroom there.

**J3 — KCONCAT: mlp3 / down as one K = 3072 op** (pass 18) [DOC]. Removes the K split (and the third norm input), but the A slot would be
48 chunks = 497 KB of a 512 KB memtile (3A already took it to 36).

**J4 — YRES / SSPM: yx's output to the SSM on chip; the SSM params multicast once instead of 8 column fills** (pass 18) [MODEL,
conditional]. Save 3.5 MB / 1.0 MB of DDR per block, but the SSM is compute-bound (in situ ~2,340 cycles per unit vs a DDR floor of
2,010): ~0 today. Revive once the SSM glue work brings it to its DDR floor.

**J5 — GELU-EW: mlp0's GELU + panel packing on the idle EW cores** (pass 18) [MODEL]. The EW would read the pre-activations and write the
panels (~11 MB per block) during mlp0, which already uses ~42 GB/s: over the cap. GELU stays in the GEMM epilogue (~0.55 ms per step,
LUT-bound).

**J6 — B-ALT: alternate the popped B bank per k-step so the DMA writes the idle one** (pass 18) [MODEL]. Would cut E7'''s B write
collisions (~70 cycles per k-step in bank 3 at 0.9 busy), but both banks would need a whole 4,032 B Y buffer; bank 0 has no room (A24 /
A25).

## K. Added by research pass 19 (2026-10-11)

**K1 — KTL-MULMASK: lock conditions as flag x counter** (rev 17 KTL, research's own) [MEAS RELC + KTLP-NL]. KTL masked its A / B
acquires with r26 = flag x r27 (the remaining-row-block counter) so the trip after the last would not acquire. acq.cond / rel.cond test
BIT 0 of r26 (RELC: r26 = 2 did not release, 3 did), so every A acquire at an even counter was skipped (all of them for an even trip
count, e.g. KTL1's 18) while every release ran: the locks drifted and the step hung in situ (main round 22, twice; KTL1 too). The
replay modelled "non-zero" and passed. Fixed with boolean masks (nez + and; r24/mm_e7k_r70b): lock-free bench 9 / 9 A and
nks - 1 / nks B per tile, pieces bit-identical against E7. Lesson: a lock condition is one bit; the replays now reject any other
value.

**K2 — MUL-LATENCY: "scalar results need one more cycle inside a ZOL body"** (pass 19 suspect for K1) [MEAS MULLAT]. mul -> st at 2
bundles and add -> st at 1 bundle see the new value in a ZOL body and in straight code. Not the KTL cause.

**K3 — STALE-HEADER: a generated kernel run with another schedule's record header** (pass 19, research's bench) [MEAS]. The record
loads read fields out of order (pointers as flags, flags as jumps): a timeout in the lock-free bench (device recovered). Lesson: the
kernel .s and its .h travel together (r24/ktlpb.py asserts --kh).

**K4 — ATTVS-II: attvs from II 18 towards its resource bound 11** (pass 19) [MODEL]. One call is 6 iterations, so the loop pays
(6 + S - 1) x II: II 18 / 2 stages = 126 cycles, II 11 / 4 stages = 99 (more stages, MVE text). -27 cycles per call x ~1,280 calls
per step = <= -0.02 ms. Not proposed.

**K5 — S3 on the new attkb / attv** (pass 19) [MODEL]. ~220 flagged row blocks per core per step (chunk 0's 144 + ~2.3 % after it)
x -60 cycles, even with a x6 lockstep amplification: -0.01..-0.04 ms. K2 (-0.02 measured) already said chunk 0 is hidden. Not
proposed except as a text lever (~400 B of k_rb C++).

## L. Added by research pass 20 (2026-10-11)

**L1 — KTL-R68-CP: KTL at R 68-69 with no bank conflicts** (pass 20) [MODEL CP, conditional]. r22/gen_e7.py --ktl: R 69 --hard-banks
1,800 s at 4 workers (r70b hint) and 1,500 s at 2 workers (hint scaled to R 69): no solution; R 69 soft 1,200 s at 2 workers: no
solution at all; main's R 68 soft (12 workers, 1,800 s): 35 modelled conflicts per k-step against 36 cycles saved; repaired from it
at 4 workers for 1,200 s: still 35 (prologue 45 -> 41), a plateau. Bank 3 holds 63
accesses per row block (Y 27, L 18, N 18), i.e. 91-93 % of the bundles at R 68-69; the A pops' bank (1 / 2) and the accumulator
pieces' banks both change with the row block, so an A pop's bundle can rarely hold an accumulator op. Revive with more CP workers in
a quiet window, or with a placement whose A slots and accumulator pieces never share a bank.

**L2 — FSX-CP: gate_a's sigmoid inside ssq** (pass 20) [MODEL CP]. r25/gen_fsx.py: 109 ops, ResMII 30 (bank 2: 30, mv 29, ldb 26 with
eight constants on lda); cpsched found no schedule at II 30, 31 (700 s each) or 32, 33 (600 s each), 2 workers. Even at II 30 FSX
removes only compute (~160 static cycles per SSM unit), and the SSM span is at the DDR read cap: <= -0.05 ms. E16 stands.

**L3 — WINDOW-AVERAGE: DDR headroom read from a window average** (pass 20, research's own note to main at 15:05, corrected at
15:20) [METHOD]. Reads + writes over the whole yx / SSM window (24.9 MB in ~320 us = 78 GB/s) compared with norm0's 90 GB/s r+w said
"headroom: the SSM is channel-bound". Inside the SSM's own 288 us span the reads are 15.5 MB = 54 GB/s, at the measured read cap
(55-59, round 19 DDRCAP). Lesson: price a bound over the span of the op that binds, with the cap of the same kind (read vs r+w).

**L4 — OUT-TILES-ON-HEADS: out's tiles 1-3 following the SSM heads like tile 0** (pass 20) [MODEL]. k-outer across tiles needs every
tile's C resident at once: four 18 KB C objects per core in a 64 KB L1 that has ~576 B free. And the whole 1.77 MB drain would then
land after the last head (~35-50 us at the drain rate): no gain over today's 52 us tail.

**L5 — MSPLIT-OUT: out in two row passes so norm1 starts on rows 0-5** (main rounds 23-24) [MEAS main]. +0.43 with OUTPRE (it
delayed the SSM's fills), +0.14 without, neutral with MSDEFER (r24a: norm1 ends 6.7 us earlier, up_a's first word 5.5 us later).

**L6 — YXH: yx steps head-major with finer drain groups** (main round 24) [MEAS main]. 7 drain TAPs per column: +0.48 (a shim S2MM
channel queues ~5 tasks); 4 TAPs: +0.09 (the SSM starts 15 us earlier but its wall grows 289 -> 308 us: more of yx's reads inside the
read-capped SSM span). The order alone is free (r24c).
