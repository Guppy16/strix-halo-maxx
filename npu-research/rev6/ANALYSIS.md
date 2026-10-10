# ANALYSIS.md — where the TAPNext++ step's time goes, what bounds it, and what is left

Revision 6, 2026-10-10. State analysed: round 10's `full12r10r` (53.89 ms per depth-12 dispatch, Turbo, gpu_busy 0), its
full-step host-polled timeline (`npu-tapnext/build/tl_full12r10r_d12.json`), the round-9 timeline `tl_full12r9s_d12.json`, the
round-10 GEMM PROF image `build/g11-r10rp.npy` and the r10r depth sweep. Revision 5 is in `rev5/`. Tags: [DOC] vendor source
or the deployed binary, [MEAS] measured by the main agent, [INF] model or arithmetic. Cycles at **1.81 GHz** [MEAS].

New tools this revision (`tools/`, none needs the device; outputs in `r11/`):
- `sim13.py`: the block composed from the measured r10r timeline (per-op walls) + the g11-r10rp per-core GEMM composition +
  3 fitted boundary constants; prices the rev-6 levers, the ceiling and P14.
- `p7w.py`: the weight side of the norm fold on the real weights (bfp16 of diag(g) W against today's bfp16(W)).
- Re-runs of `g11comp.py` / `profdec.py` on g11-r10rp (`r11/g11comp_r10rp.out`, `r11/profdec_r10rp.out`).

## 0. Scorecard: revision 5 against round 10

| rev-5 prediction | measured | verdict | why |
|---|---|---|---|
| P5r + P5a -1.6 +- 0.3 ms | -1.35 | on the range's lower edge | up_b's round-9 wait was overstated (its r9 records were incomplete): up_b gained 18 us per block, sim12 said 31 |
| P23 -1.2 +- 0.3 ms | -0.85 | miss by 0.05 | sim12 priced qkv's row coupling at 80 us per block from the g11 PROF per-column max wall (337 us). The non-PROF full step never had it: r9s qkv was 284 us, only 21 us above its per-core composition. **Coupling must be read from the timeline, not from PROF walls** |
| P5 (streaming) -1.7 +- 0.7 ms | -2.0 (as runtime memtile tasks) | hit | |
| P5 + P23 combined -4.74 ms | -4.25 dispatch / -4.32 block | +0.4 ms (10 %) | per-op errors were large but cancelled (below) |
| first-tile waits after P5: ~3 us (K 768), ~6 (K 1536) | g11-r10rp: 0.03-0.2 us (mlp3_a 4.2) | better than predicted | the runtime tasks load the next op's A *under the current op*; sim12 assumed the load starts at op start |
| sim12 norms 541 us per block (r8 stamps) | r9s timeline 542 | hit | |
| sim12's fitted SSM + syncs 873 us per block | r9s: SSM + its boundary 791, the other 82 sit in the EW -> GEMM boundaries | consistent | P33 answered by the r9s timeline |

Per op, r9s -> r10r (timeline means of blocks 1-11, us) against sim12's per-op deltas (P5 load model + P23 coupling):

| op | predicted | measured | error | cause |
|---|---|---|---|---|
| yx | -48.9 | -43.1 | -5.8 | |
| out | -26.5 | -29.9 | +3.4 | |
| up_a | -26.0 | -29.3 | +3.3 | |
| up_b | -31.0 | -18.1 | -12.9 | r9 wait overstated |
| down_a / down_b / mlp3_b | -17.1 / -23.1 / -22.5 | -35.1 / -32.7 / -38.2 | +18 / +10 / +16 | A of an independent op now loads under the previous op (no DDR-paced first tile) |
| qkv | -107.5 | -31.2 | **-76** | PROF-wall coupling artefact (above) |
| out_proj | -25.8 | -33.6 | +7.8 | |
| mlp0 | -26.9 | -26.2 | -0.7 | |
| mlp3_a | -39.6 | -13.6 | **-26** | its first tile is B-starved (section 1.3) |
| attention | 0 | -28.4 | +28 | Q objects now leave DDR once (A_HALF) and the slot is runtime-driven; not modelled |

Lessons: (1) price row coupling and op starts from the full-step timeline; (2) model *which* load overlaps *which* op
(A_RT made "independent next op" loads free); (3) one op (mlp3_a) hid a different bound (B) behind the A story.

## 1. Refit: sim13 on the r10r timeline

### 1.1 Method and fit error (`r11/sim13_r10r.out`)
- Per op: mean last-to-last wall over blocks 1-11 of the timeline. Block = 4,466 us (median block diff 4,470).
- Depth line through the sweep: **4,483 us per block + 0.21 ms**; residuals -0.18 / +0.31 / -0.10 / -0.04 ms (d12 / 10 / 8 / 6).
- GEMM ops: per-core composition from g11-r10rp (steady k-steps 2,051-2,065 cycles, gaps, first wait, coupling, epilogue,
  handoff, acc_zero). The composition sums to 2,074 us per block = 24.89 ms x 12 against the measured non-PROF g11-r10r
  **25.21 ms (-1.3 %)**.
- Boundary = timeline wall - composition, fitted with 3 class constants: after an EW op **+28.7 us** (yx +28.9, up_a +21.1,
  qkv +38.2, mlp0 +26.7), after a dependent GEMM / attention **+26.7** (out_proj +19.7, mlp3_a +33.8), independent **-9.3**
  (down_a -6.1, down_b -10.9, mlp3_b -10.9: the next op starts under the previous op's last drain); up_b (reuses up_a's
  slot) +5.6.
- **Fit: per-op rms error 5.2 us; block 4,466 vs 4,466; step 53.81 vs 53.83 ms.** The class constants carry the boundary
  model; everything else is measured.

### 1.2 Per-block breakdown (r10r, all serial: one op at a time on the critical path)

| part | us per block | ms per step | share |
|---|---|---|---|
| GEMM k-steps (2,055 + 67 cycles each, 1,296 per core) | 1,515 | 18.2 | 33.9 % |
| attention (wall incl. its Q copies and boundary) | 910 | 10.9 | 20.4 % |
| SSM + its boundary (out wall - out composition) | 795 | 9.5 | 17.8 % |
| norms (4 EW ops: 126 / 123 / 146 / 147) | 542 | 6.5 | 12.1 % |
| GEMM epilogues (yx 27, up_a 113, up_b 113, mlp0 131, qkv 25, 5 plain 15) | 425 | 5.1 | 9.5 % |
| op boundaries | 146 | 1.8 | 3.3 % |
| tile handoff + acc_zero + first + coupling | 133 | 1.6 | 3.0 % |

**Critical path**: the whole block (no op overlaps another except the drain-under-next-op of independent GEMMs). The
GELU-path epilogues (up_a, up_b, mlp0, yx: 384 us) are now a bigger lever than the norms' rev-5 fold estimate, and
attention + SSM are 38 % of the block.

### 1.3 New finding: dependent GEMMs start with their B, not their A, late
- g11-r10rp, mlp3_a (the only g11 op that depends on the op right before it): every core's **first tile** waits outside
  mm_ks 2,324 cycles per k-step (B acquire; steady 69) plus 33 K cycles before its first k-step; A is on time (busy 2,059
  per k-step). That is ~49 us of B waiting, and the timeline shows mlp3_a 48.6 us slower than mlp3_b.
- Cause [DOC design8.py, sequence]: for every GEMM op the runtime sequence runs `finish_deps(op)` (await the producer's
  drains) *before* it issues the op's B fills. So a dependent op's B (3 rows x 110 KB for K = 1536) leaves DDR only after
  the producer finished, together with 8 columns x 248 KB of A: the first tile is DDR-bound and B loses the arbitration.
- The same ordering applies to every op after an EW op (yx, up_a, qkv, mlp0) and after attention (out_proj): their
  boundaries are +20-38 us.
- Lever **P43**: issue a GEMM's B fills (they depend on nothing) before `finish_deps` / before the preceding EW op's finish,
  and deepen the memtile B forward so a whole first tile of B (55-110 KB; memtile has ~136 KB free) is resident.

## 2. GELU epilogues: the hand modulo schedule

### 2.1 Today (g11-r10rp per tile; deployed asm `npu-tapnext/build/r10/gx.s`)

| epilogue | cycles per tile | what | cycles per 32 output lanes |
|---|---|---|---|
| plain (bias_to_c) | 1,290 | 288 vectors acc + bias -> bf16 | 4.5 |
| yx GELU tile (flag 1) | 10,990 | bias 1,290 + gelu_lut_n(n = 288) at 34.2 | **38.7** |
| gelu_pan (mlp0, flag 4) | 14,824 | bias 1,290 + 36 calls of n = 8 at 44.1 (12,700) + 72 bfp pushes (~830) | **51.5** |
| gated (up_a / up_b, flags 2/3) | 12,724 | bias 1,290 + 18 calls of n = 8 on the gate half (6,350) + products, 72 pushes and (gated_b) the 288 B stash copies (5,084) | **88.4** per output vector |

Per core per block: yx 4 GELU tiles, up_a 16, up_b 16, mlp0 16 gated / pan tiles: 384 us = 4.6 ms per step on the critical
path (the GELU loop itself ~246 us; the rest is the bias pass, the gated products / pushes / stash and call overheads).

### 2.2 Slot usage and the minimum initiation interval
Slot classes from `aie2p_lat.json` (llvm-aie .td): **ldb**: vldb.4x64 (LUT, latency 7); **lda**: vlda (7); **st**: vfloor.s32.bf16,
vconv.bf16.fp32, vst.conv, vst.push.544.conv (bfp, 64 lanes); **mv**: vmin_ge / vmax_lt / vadd.32 / vshuffle /
vconv.fp32.bf16 (ups) / vmov x (all latency 2); **vec**: vmac.f / vmsc.f / vmul.f (accumulator read at stage 3 [MEAS], result 6).

Ops per 32 output lanes, fused form (bias add folded into the same loop, one call per tile):

| variant | lda | ldb | st | mv | vec | ResMII | binding |
|---|---|---|---|---|---|---|---|
| yx (bias + tanh GELU, bf16 out in place) | 4 acc quarters + 1 bias = 5 | 8 LUT | conv x, 2 vfloor, conv sh, conv sl, vst.conv = 6 | 2 clamp + 4 shuffle + 2 index add = 8 (6 with a biased table pointer) | bias mac, s mac, msc sl, mul x sh, mac = 5 | **8** | ldb, mv |
| gelu_pan (bias + erf GELU, bfp panels) | 5 | 8 | 6 + conv to bf16 + 0.5 push = 7.5 | 8 + 1 ups = 9 (7) | 5 | **8-9** | mv / ldb |
| gated (2 x bias, GELU on gate, gate x up, bfp) | 8 acc quarters + 2 bias = 10 | 8 | 7 + conv up + conv product + 0.5 push = 8.5 | 9 (7) | 7 | **10** | lda |

- The LUT path's floor is the ldb slot: 4 vldb.4x64 per 16 lanes (8 per 32), no alternative unit (the 4x16 / 4x32 forms
  need more loads for the same 64-bit entries).
- mv is the second wall: drop the two `vadd.32` index adds by pointing the LUT base registers 16 x BIAS bytes into the
  table (needs a probe that a negative index offset works; else keep them and accept mv = 8).
- gated is lda-bound (two fp32 accumulators per output vector). Folding the bias into the accumulator init (P24c: the
  first k-step loads a bias block instead of zeros) would remove the bias loads and macs, but changes the fp32 summation
  order: not bit-identical.

### 2.3 Latency and register pressure (one iteration, yx form; times in cycles from the first acc load)
acc quarters land 7-10 -> bias mac 10 (result 16) -> conv x 16 -> clamp 18 / 20 -> vfloor 22 / 23 -> (index add 24 / 25)
-> 8 LUT loads 24-31 (land 31-38) -> 4 shuffles 32-39 -> s mac 40 (46) -> conv sh 46 -> msc sl and mul x sh 48 (54) ->
conv sl 54 -> final mac 56 (62) -> vst.conv 62. **Latency ~63 cycles: 6-7 stages at II 10.**

Live values (x = 512-bit vector registers, 12 in the file; dm = 1,024-bit accumulators, 5 [DOC Peano asm: dm0-dm4]):
- constants: clamp lo / hi, -1.0 (or the msc form), the index bias (unless the pointer is biased): 3-4 x;
- x is needed at 18-20 (clamp) and again at 48-57 (both products): **reload it** from its in-place copy (store the bf16 x
  at 16: st +1; vlda at 41: lda +1) instead of holding it 40 cycles (that alone would take 4-5 x at II 8-10);
- index (2 x per iteration, ~8 cycles), LUT results (4 x for ~3 cycles), slope (1 x, ~6), sh / sl (~1): ~5 x at II 10;
- accumulators: acc + bias (written 7-10, read 13, result 16), sacc (offsets, ~5), s (~5), sl (~1), x sh (~6), final (~1):
  **~3.9 dm-equivalents at II 8, ~3.1 at II 10.**
- So: RegMII ~9 (x) and ~8-9 (dm); with modulo variable expansion (kernel unrolled 2) **II 10 is feasible, II 8 is not
  with 12 x / 5 dm.**
- The flat-edge trick: give the table one flat segment beyond each clamp edge (slope 0, offset = today's edge value of
  off + slope * xc, exact in fp32) and the s mac can take the unclamped x (s = off + 0 * x): bit-identical, frees xc after the
  vfloor. 2 x 16 B x 2 copies x 2 tables = 128 B of L1.

### 2.4 The schedule to build (rev-6 spec for the main agent)
1. One asm call per tile and per variant, kernel-only loop (zero-overhead, 16-B aligned start, full six-slot first / last
   bundles), II **10** for yx / pan, **11** for gated (12 as the fallback for all three). Two output vectors per kernel
   trip (MVE unroll 2) so register names rotate statically.
2. Inputs straight from the fp32 pieces (no separate bias pass): the three piece segments (pair 0 rb < 12 in C slots,
   rb >= 12 in acc_hi 0-5, pair 1 in acc_hi 6+) as three trip ranges with pointer jumps from a 3-entry record (gen_ks's
   record stream). The bias pattern (8 vectors per piece, the same for every piece of a pair) is a second stream on lda.
3. Banks: the LUT loads touch bank 0 (sgelu_ab / serf_ab at 0x70B00 / 0x70F00) and bank 3 (cd copies at 0x7F580) in 8 of 10
   bundles. acc_hi pieces 0-9 sit in bank 0, the C object's last 4,352 B in bank 3 [DOC readelf full12r10r core 0,2]. Put
   every st to bank 3 and as many lda as fit in the 2 LUT-free bundles; the rest of the bank-0 piece loads cost +1 each
   (bank rule): ~+0.8 cycles per vector on average.
4. Bit-identity: same op sequence as `gelu_lut_i` (clamp, floor index, offset + slope xc, hi / lo split, x sh + x sl), same
   roundings; gelu_pan keeps the bf16 rounding before the bfp push (conv + ups + push); gated rounds the GELU and the
   product to bf16 as today. The stash copy of gated_b goes away only if the panel layout changes (optional; it is ~300
   cycles per tile once the loop is pipelined).
5. Prologue: kernel-only with S - 1 extra trips; the extra trips' loads read inside the C object / acc_hi (harmless), their
   stores must not land on unread input: direct the first S - 1 stores to a scratch area through the record stream, or emit
   the prologue explicitly with its stores removed. Text budget: GEMM core 544 B free + what the C++ paths free (flags 1-4
   code in acc_epilogue ~1.5 KB of its 3,168 B + gelu_lut_n 448 B): **~2.5 KB for the three kernels**; an explicit prologue
   costs ~1 KB per variant, so kernel-only is required for all three, or one shared yx / pan kernel with two store modes.

### 2.5 The forced-II result (17 static, 54.7 measured) and how to avoid repeating it
The rejected `gelu_ii16` loop differs from the deployed (static-speed) loop in four ways; each is a hypothesis for its
~38 stall cycles per iteration:
- H1: LUT-load destination registers re-used while a vldb.4x64 to them is in flight (`vldb.4x64.lo wl2, wh9; vmov wl2, wh4`
  in one bundle) - a pending-load scoreboard on the LUT path;
- H2: vmov into accumulator quarters (bmlh3, bmll3) of a dm that a vmac is reading / writing;
- H3: the input vector loaded by an ordinary vldb on the ldb port between the 4x64 lookups (the deployed loop loads on lda);
- H4: the mv slot at 17 of 17 bundles with vmov chains (x8 <- x10 <- ...) whose latency the schedule may take as 1.
gen_ks's conveyors (reads of an old value while a new load is in flight) run bit-correct and near static speed, which argues
against a general load-use interlock, so H1 (LUT-specific), H2 and H3 are the candidates. **Bench ladder before the hand
schedule (one core, gxb.py style, timer, n = 144 x reps)** - P45:
- L1: 8 vldb.4x64 per trip into distinct registers, nothing else, II 8: predicted 8.0 cycles per trip;
- L2: L1 with destinations re-used every trip (read by a vmov 7 cycles after the load): 8.0 if H1 is false;
- L3: L1 + one ordinary vldb per trip (9 ldb ops, II 9): 9.0 if H3 is false;
- L4: an II-6 loop of vmac.f on dm1 with a vmov into a quarter of dm1 three cycles before each mac's read: 6.0 if H2 is false;
- L5: the hand schedule at II 16, then 12, then 10.
The first rung that runs slower than its static count names the hazard; the generator then gets that rule as a constraint
(as gen_ks got the stage-3 rule).

### 2.6 Predicted cycles and saving (sim13)

| variant | today per 32 lanes | II 10 (incl. ~250 cycles per call, bank stalls in the +0.5 / +1) | per tile today -> II 10 |
|---|---|---|---|
| yx | 38.7 | 10.9 | 10,990 -> ~3,130 |
| gelu_pan | 51.5 | 11.4 | 14,824 -> ~3,270 |
| gated | 88.4 | 12.0 | 12,724 -> ~1,830 |

- **G1 (asm, fused): -3.74 ms at II 10, -3.61 at II 12, -3.33 at II 16** (sim13). Most of the gain is the bias pass, the gated
  products / pushes / stash and the per-call overhead, so the result is not very sensitive to the II.
- **G0 (C++ only, a cheaper first step): -1.4 ms**: one GELU call per tile (n = 144 / 288 instead of 18-36 calls of 8: 44.1 ->
  34.2 per 32 lanes) and the gated product / push pass written as a pipelinable array pass (5,084 -> ~1,300 cycles per tile).
- Plain / qkv / v_obj epilogues are unchanged (bias_pcs stays).

## 3. A modulo-schedule generator for GELU, the SSM chains and the attention exp (P46)

gen_ks proved the method on one loop. A general generator would serve G1, the SSM passes (P29), the attention exp (A1) and the
norm kernels (P7-E). Specification (python, emits Peano-assemblable `.s` + a `.h` with constants and record layouts):

```
M = Machine.from_td("p10/aie2p_lat.json", acc_read_stage=3, regs=dict(x=12, dm=5, p=8, m=8, r=32))
M.rule(...)                                         # hazards found by benches (e.g. P45's result)
L = Loop(M, "gelu_pan", ii=10, unroll=2)           # unroll = modulo variable expansion factor (auto: ceil(max life / II))
acc = L.stream("p0", step=128, segments="rec")       # pointer + post-increment; segment jumps from a record stream
out = L.ostream("p2", kind="push_bfp")               # block-vector output stream (no calls while live: kernel-only loop)
lo, hi, m1 = L.const_bf16(-4.0), L.const_bf16(3.984375), L.const_bf16(-1.0)   # pinned physical registers
tab = L.lut(ab="serf_ab", cd="serf_cd", entry=16, banks=(0, 3))
a = L.load_acc(acc, quarters=4); b = L.load(bias_stream)
x = L.conv_bf16(L.mac(a, b, L.ones)) ...            # dataflow graph of typed ops (each op = one .td instruction or a macro)
off, slope = L.lut_line(tab, xc, shift=3, bias=32)  # macro: 2 vfloor, (2 vadd), 8 vldb.4x64, 4 vshuffle
L.push(out, L.ups(L.conv_bf16(final)))
L.banks(piece=lambda seg: (2, 0, 1)[seg], tab=(0, 3)) # bank tags per access for the bank rule
L.schedule()      # iterative modulo scheduling (Rau): ResMII per slot, RecMII from loop-carried edges, RegMII; slots modulo II;
                  # bank conflicts as soft costs (the exact rule: max accesses to one bank per bundle)
L.allocate()      # register allocation over the unrolled kernel (rotating names), fails loudly with the live-range table
L.check()         # gen_ks-style: simulate the unrolled stream (prologue, n trips, epilogue) with the .td latencies + rules;
                  # assert every read sees its intended producer; print bank stalls and cycles per trip
L.emulate(inputs) # bit-level numpy semantics of every op (bf16 RNE / FLOOR conversions, fp32 MAC, bfp16 encode_hw)
                  # against the C++ reference (r6/emu.py models) before any device run
L.emit("gelu_pan.s", prologue="kernel_only" | "explicit", text_budget=900)
L.report()        # II, MII per slot, stages, registers, text bytes, predicted cycles per call for n = 8 .. 288
```
Requirements from the SSM chains (P29, `ssm9.h`): int16 vector ops (2^n assembly by shift / add), the 16-bit LUT form
(`load_lut_2x_int8`, sqrt), several input / output streams per loop (gx, ga, x, h0 lo / hi, outputs h, z), fp32 state as
hi / lo bf16 pairs or accumulator loads / stores, and **pass splitting**: `L.cut(v, buffer)` ends a chain in L1 so one long
chain becomes two loops when RegMII would exceed ResMII (round 9's 5-pass exp was the manual version). From attention (A1):
an output stream per row block and a per-row-block broadcast operand (the row max). From the norms (P7-E): reductions across
lanes (the butterfly) as a loop-carried RecMII edge. Build order: the GELU variants first (they validate the machine model
against P45), then the attention exp (same LUT macro), then the SSM passes.

## 4. The norm fold (P7): math, placement, numerics, emulation design, price

### 4.1 Math (per row i of x, 768 features; consumer z = y W + b, W: 768 x N)
- **RMSNorm** (TAPNext: y = x r (1 + s), r = (mean(x^2) + 1e-6)^(-1/2)): with g = 1 + s and W' = diag(g) W,
  **z_ij = r_i C_ij + b_j, C = x W'**. Per-row scale after the GEMM; g folded into the weights on the host.
- **LayerNorm** (torch: y = (x - mu) / sqrt(var + 1e-6) * gamma + beta, var biased): with W' = diag(gamma) W,
  c_j = sum_k W'_kj (column sums), b' = beta W + b,
  **z_ij = r_i (C_ij - mu_i c_j) + b'_j = r_i C_ij + u_i c_j + b'_j, u_i = -r_i mu_i** (a rank-1 correction).
- c must be the column sums of the **quantised** weights the MAC unit uses (c = 1^T bfp16(W')), computed on the host in fp64:
  then C - mu c = (bfp16(x) - mu 1^T) bfp16(W') up to fp32 accumulation, and the mean subtraction is consistent with what the
  GEMM summed. b' in fp64 on the host, rounded to bf16 like today's bias.
- In the MAC unit (no fp32 vector multiply): split C -> C_h + C_l (2 conversions, 1 msc), r -> r_h + r_l, u -> u_h + u_l,
  c -> c_h + c_l (bf16 hi / lo): z = C_h r_h + C_l r_h + C_h r_l [+ u_h c_h + u_l c_h + u_h c_l] + b' x 1, then one
  conversion. RMS: 2 st + 5 vec per 32 lanes; LN: 2 st + 8 vec. vs today's bias pass 1 + 1: **+0.1 us (RMS) / +0.6 us (LN)
  per consumer tile** at full vec-slot use.

### 4.2 Where the row statistics are computed, and the dataflow that rev 4 / 5 left out
Rev 4 / 5 priced the fold as "+0.3 us per tile". The C object and the B broadcast make the producer side much costlier:
- **Residual sums.** Today the norm op sums the parts (norm0: D3 + D4 + R5, norm1: R1 + R0, norm2: D1 + D2 + R2, norm3:
  R4 + R3). Without the norm op the producer GEMM (out, down_b, out_proj, mlp3_b) must add them in its epilogue. The parts are
  per-column data, so they cannot ride B (broadcast across columns); they must come through the A broadcast, which sends all 3
  rows' tiles (55 KB per tile step, 1/3 useful): ~3.8 us per tile step and source at 7.95 B/cycle. 6 source-ops x 4 tiles =
  **~91 us per block**.
- **Panels of X'.** The consumer needs bfp panels of the un-normalised X' (10,368 B per tile) besides the bf16 X' (18,432 B):
  the C object (18,432 B) holds one. Drain it twice per tile (bf16, then panels + the stats partial into an APAN-style chunk
  with a junk tail): +1.32 us handoff per producer tile, **~21 us per block**.
- **Row statistics.** Producer epilogue: per tile and row, centred partials (n_t = 64, mean_t, M2_t) of the bf16-rounded X'
  (fp32; centring avoids the E[x^2] - mu^2 cancellation at LN sites), 1,152 B per tile in the second drain's tail (~0.4 us per
  tile). Consumer: the 12 partials of its rows arrive as 12 extra A-ring slots appended to tile 0's send BD (they fit: 1,152 B
  = one ring slot), tile 0's epilogue combines them (Chan's parallel formula), computes r (rsqrt by the EW's table / Newton
  method) and keeps (r_h, r_l, u_h, u_l) as bf16 in a **1,152 B L1 buffer** for the op's other tiles (up_b reuses up_a's).
  ~1.5 us per consumer op.
- Total added ~**161 us per block**; removed: the 4 norm walls (542). Boundaries: EW -> GEMM (+28.7) becomes dependent GEMM ->
  GEMM (+26.7): no change.
- **Blockers:** GEMM-core L1 (~932 B free, the buffer needs 1,152: a GEMM-core stack audit first), GEMM text (544 B free; the
  fold's epilogue modes need ~1-1.5 KB, competing with G1), the A-path residual protocol (new send BDs) and the double drain
  (C fifo protocol). L+.

### 4.3 Numerics risks per site
| risk | where | size |
|---|---|---|
| A side: bfp16(x) instead of bfp16(norm(x) g). A channel with large x and **small g** sets its 8-block's exponent and flushes its 7 block-mates; today g shrinks it first | every site; worst where g is tiny | `r11/p7w_smallg.out`: channels with g below 0.05 per block: norm0 none; norm1 18 in block 0, 1 in blocks 1, 2, 7, 10, 11 (g down to 3e-4); norm2 110 / 20 / 4 in blocks 0-2; norm3 52 / 28 / 11 in blocks 0-2. If these are massive-activation channels (likely in early ViT blocks) the fold fails there |
| A side, LN: bfp16 keeps x's relative precision, not (x - mu)'s: error ~ 2^-8.5 mu / sigma relative | norm2, norm3 | needs real activations (P7a) |
| W side: bfp16(diag(g) W) along K | every site | **measured on the real weights (`r11/p7w.out`): error ratio fold / today median 1.013 / 1.020 / 1.023 / 1.016 (norm0-3), worst block 1.086**: harmless |
| cancellation C - mu c (fp32) | LN | harmless below mu / sigma ~ 1e3; c from the quantised weights is required |
| stats: E[x^2] - mu^2 in fp32 | LN | centred partials + Chan combination |
| r applied as bf16 only | all | 0.4 % row-scale error: use the hi / lo form (above) |
| bit-identity lost | all | judge by the bfp ratio, the CPU-model track noise and DAVIS |

Mitigation **P7c**: the residual stream's feature order is internal to the trunk (norms are permutation-invariant; every
GEMM that reads or writes the residual can take permuted weights on the host; the SSM / GELU act on GEMM outputs, not on the
residual; the patch embedding and the final norm / heads take the permutation at the ends). A global permutation that groups
the small-g / large-x channels into their own 8-blocks removes the block-mate flush. Choose it from P7a's captured statistics.

### 4.4 P7a, the emulation the main agent runs (no device; CPU numpy + the torch CPU model)
1. **Capture** (extend `r6/capture.py`): for each block and norm site, the pre-norm residual X' as torch computes it (bf16), 256
   random rows + the 48 query rows, 3 frames of clip30 and 2 DAVIS videos; plus per-channel max |x| over all rows (outlier
   map).
2. **Paths** per site, in numpy with the device's numerics (`pack7.encode_hw` for device conversions, the host codec for host
   operands, fp32 accumulate, bf16 RNE at each conversion):
   - T (today): y = the EW norm's bf16 output (emu of ew8's norm), A = bfp16(y), z_T = bf16(A bfp16(W) + b);
   - F (fold): A = bfp16(X'), B = bfp16(diag(g) W), C = A B (fp32), stats from X' as in 4.2, z_F = bf16(r C [+ u c] + b') with
     the hi / lo arithmetic of 4.1;
   - F+P: F after the P7c permutation;
   - E (exact): fp64 norm and fp64 W.
3. **Metric**: the P8 yardstick on the consumer output, ratio = rms(z_F - z_E) / rms(z_T - z_E) per site and block, plus
   per-row ratios (to catch single rows with mu / sigma >> 1) and the list of channels whose 8-block-mates lose > 2 bits.
4. **Pass thresholds** (pre-registered): median over blocks <= 1.10 and every block <= 1.25 -> fold that site; 1.25-2 in a
   block -> fold only with P7c, re-check; > 2 -> keep that site's norm on the EW core. Then the CPU model with emulated
   numerics (`r9/cpuemu.py`): track noise over 30 frames no larger than the ulp-level control's; then DAVIS 10 videos on the
   device build: AJ d12 >= 70.6, d10 >= 69.7 (today 70.84 / 69.90).
5. **Pre-registered outcome:** norm0 passes everywhere (no small-g channels, g 0.12-1.7); norm1 passes except block 0
   (18 small-g channels) and possibly blocks 7 / 10 / 11; norm2 and norm3 fail in blocks 0-1 without P7c and pass from block 3
   on; P7c brings the failing blocks under 1.25 if the large-x channels are few (< 8 per site).

### 4.5 Price (sim13) and the cheaper alternative
- **P7-full: -4.6 ms +- 1.5** (all 4 sites folded, +161 us of added work), L+, blocked by L1 / text. Per-site partial folds
  scale with the sites that pass P7a.
- **P7-E (new): the EW norm kernel itself.** The norms run at 2.2 B/cycle input (norm1, n_in 2: 442 KB in ~106 us of
  streaming) to 3.4 (norm0, n_in 3). If the EW input port is 4 B/cycle (the design-7 assumption), norm0 is near its bound and
  the others are not: modulo-scheduled norm passes give **-1.5 ms**. If it is ~8 B/cycle like the memtile -> core stream (P10),
  every norm is kernel-bound and the output (345 KB) binds at ~70 us: **-3.3 ms** (P7-P, two input channels, reaches the same
  with a 4 B/cycle port). P44 (norm stamps + an EW input-rate probe, S) decides which.
- So the fold's increment over P7-E-high is only ~-1.3 ms for an L+ change: **P7-E first, the fold parked**.

## 5. Attention: where the K phase goes

Per core and block: 4 heads x 3 query tiles (48 rows) x 1,072 keys; per head 18 row blocks x 24 key chunks = 432
(rb, chunk) units. Round-8 stamps: K phase 136.4 us per head (both passes) = **571 cycles per (rb, chunk)**; V phase 51.2 us
(213 cycles); fixed ~70 us per block. From the deployed asm (`r6/asmstat.py` on build/r10/mm8.s):

| piece of one (rb, chunk) | static bundles | share | floor |
|---|---|---|---|
| QK^T: 3 hardware-loop trips of 46 bundles (16 mmul each: 48 mmul) | 138 | 24 % | 48 (Q and K pops on lda + ldb, 1 mmul per cycle) |
| exp + bfp push (aexp_push: 29 + 6 x 49 + 7) | ~330 | 55 % | 6 x 16 = 96 (16 LUT loads per 64 scores on ldb) |
| score stores, 6-block max, rowmax8, lazy-max test, rows8 (scalar broadcasts, 26 bundles), call overheads | ~150 | 21 % | ~30 |

Options (predictions alone, sim13; 432 x 4 units per core per block):
- **A1** exp + push as a modulo-scheduled per-row-block call (the GELU generator; II ~18 per 64 scores, ~150 cycles per call
  incl. fill): **-2.1 ms**. Pipelining across row blocks would need a 4.6 KB score buffer; the C object is full (18,368 of
  18,432 B) and acc_hi too.
- **A2** QK^T + score conversion / store in asm (Q and K pops split over lda / ldb, 6 accumulator chains): 138 -> ~68: **-0.8 ms**.
- **A3** running max fused into A2's output path (vmax on the mv slot as scores are formed), rows8 by `vbcstshfl` instead of
  8 scalar broadcasts: ~150 -> ~40: **-1.3 ms**.
- **A1 + A2 + A3 as one K-phase kernel**: lda carries Q + K pops (96), ldb the LUT (96): floor 96, predicted ~180 cycles per
  (rb, chunk): **-4.5 ms**.
- **A4** V phase (PV + ones panels, 54 mmul, O load / store) in asm: 213 -> ~70: **-1.6 ms**.
- The two passes' K / V re-streaming is not a cost (pass 1's per-query time equals pass 0's).

## 6. Ceiling and P14, re-priced on the refit (sim13, depth 12)

Combined, each step on top of the previous: G1 (II 12) 50.2 -> + P43 49.4 -> + A1-A3 44.9 -> + P27 + P29 40.7 -> + P7-E high
/ P7-P 37.5 -> + P35 + P24b 36.5 -> + A4 34.9 -> + k-step 1,600 (needs L1 and R_A >= 6.5) **30.5 ms (32.8 fps)**; depth 10
25.5 ms, depth 8 20.4. Without the speculative k-step: 34.9 ms at depth 12. Ceiling block: k-steps 1,117, SSM 447, attention
400, norms 270, epilogues 124, tile overheads 88, boundaries 78 us. Still fully serial.

**P14** (two frames in flight; list bound: per frame >= busier side + 15 % of the other):
- today: GEMM side 3,129, EW side (norms + SSM) 1,337 us per block -> **40.2 ms per frame (x1.34), latency ~80 ms**;
- after G1 + P43 + A1-A3 (GEMM side shrinks first): **31.3 ms per frame (x1.44)**, latency ~63 ms;
- after the SSM and norm work too: 30.2 ms (x1.24); at the ceiling 23.2 ms per frame (43 fps, x1.32), latency ~46 ms.
- P14's value now *rises* while the GEMM-side levers land and falls when the EW-side ones do. It doubles latency and shares
  shim channel 0 (A loads and EW inputs) between the two frames. Still the last fps lever; build it only once the serial
  levers have landed, and only if fps beats latency for the product.

## 7. What still stands from revision 5
- The A-port verdict (not binding at 2,055 cycles; the k-step is compute), the bank rule, the SSM table numerics, P38's design.
- Compose from the binary, fit at most a few constants, pre-register; read distributions, not means; price coupling and op
  starts from the full-step timeline.
