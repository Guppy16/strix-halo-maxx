
## 5. The end-to-end prototype (npu_step.py) and what the full port still needs
Done tonight: the step runs on real frames with every trunk GEMM and every attention executed by the NPU kernels
(tracks within 0.03 px of the all-CPU bf16 step; DAVIS in RESULTS.md). It is a correctness vehicle: one dispatch
per op across six xclbins (~2 ms each of context switch + DMA setup), activations packed to bfp16 on the host
(3.5 s/step with the numpy codec) and the elementwise work on the CPU.
Facts that shaped it: design6 cannot express a per-step program with data dependencies between ops (its ops are a
static list fed from one A buffer), so the fused step needs a new runtime sequence: per block, GEMM -> elementwise
kernel -> GEMM ... with activations in DDR and the elementwise kernels writing bfp panels (the attention-v15 push
mechanism) in the GEMM's A order; the 48 query rows need either the 1280-row program (core program memory +
L1: 3.6 KB over each) or a CPU thread overlapped with the NPU (bf16 CPU: 32-50 ms/step, too slow to hide fully).
Kernels still to write for 1072 x 768 (all have moondream precedents except the RG-LRU step): RMSNorm (x2),
LayerNorm (x2), GELU (tanh form, phase-8 sigmoid epilogue), gated products, residuals, conv1d over the 3-step
cache, RG-LRU (sigmoid gates from the 12 block-diagonal 64x64 linears, exp, sqrt, fp32 state), the attention
input re-layout (qkv output -> per-head Q/K natural + V^T panels), the K-chunk partial sums, bfp packing.
Numbers to keep in mind: NPU kernels ~37 ms/step measured for GEMMs + attention (DDR-contended), the rest
extrapolated at ~35 ms now (~25 after the queued GELU/LN work), dispatch ~0.1 ms per program; GPU 22.6 ms.

## 6. Session 2: packet-switched shim channels (MEASURED) and the design7 plan
`pkt_test.py` mode 2: two `PacketFlow`s on shim (0,0) MM2S channel 0 - pkt 1 into a memtile static slot chain
(then circuit to core (0,2)), pkt 2 straight into a hand-built 2-deep ring on core (0,3); both routes bit-correct,
dispatch 58-90 us. `ObjectFifo(packet=True).prod(channel=0)` on the same channel is REFUSED by IRON ("pinned MM2S
DMA channel 0 already in use"): a shared shim channel means PacketFlows + hand-built core rings. The shim allows at
most 16 simultaneously active runtime BDs (design6's lookahead=2 task groups stay under it).

### design7 (the fused step; see HANDOFF.md for the build order)
Row mapping: 1152 rows = 8 columns x m=144 (1024 image + up to 128 query slots; rho=9 stripes of 16, a_ring=3).
Cores: rows 0-2 = GEMM (design6's A slot broadcast + B row broadcast + C join; N split 3-way: nt = N/192 per core,
ffw_up as two N=3072 ops so 2 x nt <= 63); row 3 = one elementwise (EW) core per column, input = PacketFlow on the
shim's A channel (pkt id per column), 8-row objects on a hand-built ring, output = circuit fifo core -> shim S2MM ch1
(C drains use ch0 with c_streams=1). Memtile channels per column: S2MM A, B(fwd, mems 0-2), 3 x C join; MM2S A,
B, C. Shim: MM2S ch0 = A loads + EW inputs (packets), ch1 = B (shims 0-2); S2MM ch0 = C, ch1 = EW out.
Layouts: every activation in DDR is the GEMM's blocked C layout ([row-block][col-block][8][8] per 64-wide tile),
so a DDR "row" of 8 blocks is 8 real rows x 64 cols; an 8x8 block is exactly one bfp panel (8 values along the
columns share an exponent). Epilogues emit the NEXT op's operands directly: bias (per N-tile, appended to the B
stream), gate/up pairs ([32 gate | 32 up] interleaved weight columns; gelu_tanh(g+bg) * (u+bu) -> bfp panels; a
core's consecutive tile pair = one full k-step chunk of the K=3072 consumer's A layout, 10,368 B, drained linearly),
mlp0 gelu -> bfp, qkv -> the attention's Q/K/V^T objects (144 tokens per column = 3 x 48-token tiles; V^T = 8x8
transposes; objects padded to 4608 B = 4 A sub-tiles or one B object), attention O -> bfp k-step chunks (head h =
k-step h of out_proj's A). Attention runs on the GEMM cores: Q objects through the A chain (4 sub-tiles each,
attn_copyq), K/V chunks through the B fifo (row r takes heads r, r+3, r+6, r+9), O through C. K=3072 GEMMs
(ffw_down, mlp down) run in two row-halves (stripes 0-4, 5-8) with the whole K cached in the slot (80 x 3072 bfp =
277 KB): A read once, fp32 accumulation over all of K, no partial sums; their outputs are two drains per op.
EW ops on the row-3 core: RMSNorm / LayerNorm (+ residual + bias of the producing GEMM) over blocked 8-row chunks
(row sums per 8-lane group, rstd broadcast per group) -> bfp panels scattered into the consumer's A layout by the
drain TAP (12 x 576 B per 8-row object); the SSM in 4 head-groups of 3 heads (gate weights 27.6 KB per group via the
EW input; x cache = the previous two frames' YX x-halves, rotated by using 3 instruction streams (.bin) per
xclbin; h fp32 state in 3 rotating buffers; output objects [h 8x192 fp32][z 3 x 576 B] drained linearly, the
consumers gather with 4-dim TAPs). Biases: in the B stream per N-tile (one 4.6 KB object per tile, +8% B bytes).
Budget (extrapolated): GEMM ~29 ms (24 cores, idle DDR), EW ~7 (port-bound: 4 B/cycle per core, SSM 1.3 MB per
column per block), attention ~2, bubbles ~2 => ~40 ms depth 12; DDR ~700 MB/step.

## 7. Session 2 (continued): design7 built and verified through the first elementwise chain
What exists now (`design7.py`): the GEMM frame on core rows 0-2 (m 144, rho 9, a_ring 3, bias epilogue from a
bias object appended to each N-tile's B stream, blocked C tiles), the A slot fed by a PacketFlow on shim channel 0,
and the elementwise (EW) core on row 3 of every column: input = PacketFlow (pkt 2) on the same shim channel into a
32-slot x 768 B ring (one S2MM BD with BdIteration; locks pinned to ids 0/1), output = circuit Flow core -> shim
S2MM channel 1 from a 32-slot ring (one MM2S BD with BdIteration; locks 2/3). The kernels (`ew7.cc`) drive both
rings with lock intrinsics (`acquire_greater_equal(id + 48, n)` / `release`), wrap-aware addressing per 64 B vector
and a per-slot acquire/release on the output, so a kernel call = a whole op (params + 18 chunks). Schedule entries:
`[K, nt]` GEMM from the host A buffer, `[K, nt, "E"]` GEMM whose A is the preceding EW op's output region in E
(InOut), `["rmsnorm"]` EW op reading params + chunks from X. The runtime sequence finishes every pending group
before an EW op and after it (data dependency through DDR); the EW drain scatters 18 x 12 x 576 B panels into the
A layout with one 4-dim TAP per column.
Learned: Tile endpoints are matched by object identity (make the tiles once); `ObjectFifo(packet=True)` cannot be
pinned to a channel a PacketFlow uses (two PacketFlows can share it); a shim holds <= 16 active runtime BDs;
BdIteration works on core DMAs; filling a GEMM from an `InOut` buffer that a drain wrote earlier in the same
sequence works; the bfp conversion's +128 exponent bump (RESULTS 11).

### Decisions for the next steps (not yet built)
1. One activation buffer: every device-produced activation lives in ONE `InOut` buffer `ACT` with a memory plan
   (offsets per block: X, YX, SSM out, partials, panels, qkv objects, ...), host inputs (frame, params) in X and
   weights/biases/params in W. Then a single 4-dim fill can interleave two regions chunk by chunk (e.g. r and X for
   the residual op: sizes [18, 2, 3072 words], strides [3072, region distance, 1]) and no second input ring is
   needed. Three instruction streams (.bin) rotate the frame-indexed regions (x cache, h state).
2. Layout L for every bf16 activation: per column [row-block 18][k-step N/64][col-block 8][8][8]. Change the C
   drain TAP to sizes [nt, 3, 18, 512] strides [512, nt*512, (N/64)*512, 1] so an 8-row chunk is 12,288 B
   contiguous and k-step pieces are 1024 B; `pack7.unblock` must follow.
3. EW ops to write (all in `ew7.cc`, same ring protocol): `resid_rmsnorm` (in per chunk: r, [p2 for the chunked
   K=3072 ops], X; out per chunk: 12 x [X' 1024 B | panel 576 B] = 19,200 B to the combined layout
   [q][rb][X' | panel]; the next GEMM gathers panels with sizes [2, 12, 18, 144] strides [0, 7200, 400, 1] offset
   256), `resid_ln` (LayerNorm, gamma + beta = 4 param slots; start with ln.cc's bf16-emu numerics, upgrade to the
   hi/lo MAC form if DAVIS says so), `ssm` (4 head-groups of 3 heads: 36 slots of gate weights, then per chunk y,
   x, x_prev1, x_prev2 (3 KB each, column slices via the fill TAP) and h (6 KB fp32); conv1d 4 taps, block-diagonal
   gate GEMMs as 8x8 mmuls, sigmoid via exp32, sqrt via rsqrt16 * x, h fp32 out + z = bf16(h) * y as panels; output
   per chunk [h 6144 B][z 1728 B + 576 pad] = 11 slots), the qkv -> attention-object and O -> panel epilogues
   belong to the GEMM cores (mm7.cc variants selected per op).
4. Per-op bubbles: split each op's slot load into two BDs (h0, h1) with their own `full` locks so the cores start
   after the first half (K=768 ops: 19 us instead of 38); BD count = 7 x 2 + 4 x 1 + 2 sends = 20 <= 22. Prefetch
   the next op's B (already queued by the lookahead). Then the memtile self-copy for the K=768 doubling (a spare
   MM2S + S2MM channel on the memtile, a loopback route, copy BD gated by a `copy_go` lock the K=768 loads release).
5. Attention on the GEMM cores: Q objects through the A chain as 4 sub-tiles (4608 B), K/V through the B fifo
   (4608 B objects: 3584 B used), O as bfp k-step chunks of out_proj's A through C; core row r takes heads
   r, r + 3, r + 6, r + 9. L1 is the constraint (the GEMM worker has ~5 KB spare): alias the attention's statics
   onto acc_hi and cut its stack.

## 8. Round 3 (2026-10-09): design8 — the whole step in one program
### 8.1 Architecture (what differs from design7)
One `InOut` buffer `ACT` for every device-side activation, state and host input (frame patches, pos-emb, query
tokens), `W` for weights and parameter objects. Layouts (plan8.py): **Q16** per column `[k-step][row-block 18]
[1600 B = 1024 B bf16 (8 rows x 64 cols, 8x8 blocked) | 576 B bfp panels]` — the GEMM C drain writes the 1024 B
parts linearly per tile, a norm op emits `X' | panel` per k-step (one 3-dim drain), a GEMM's A gathers the panels
with one 4-dim TAP, a residual norm gathers 1-3 streams with one 4-dim TAP (regions at equal distances; the sum is
commutative so inputs are sorted by address). **APAN** regions for epilogue-made k-step chunks (18,432 B stride =
the C object window, so the junk tail of every object lands in padding). **QKV** `[head][col][12 slots x 4608]`
= Q_i K_i V_i per query/key chunk i. **SSM area** per block `[YX (2 S)][C0][C1][C2][Hlo][Hhi][Z]` (7 input streams
S apart; the kernel emits x -> C0, C0 -> C1, C1 -> C2 as a shift register, so the conv cache needs no per-frame
instruction variants; h fp32 in two 4-row halves; z panels at unit offset 0).
GEMM cores: one table-driven loop (`[k-steps, C objects, tiles per object]` per op in L1 buffers) because 16 KB of
program memory cannot hold 12 unrolled ops; epilogue selected by a flag word in each N-tile's bias object (plain,
GELU, gated pair -> chunk, GELU -> panels, Q/K objects, V^T objects); the accumulator split at 48 rows so the C
object's tail hosts the gated stash / V^T temp. Attention on the GEMM cores: Q objects through the A slot (sent
once per pass; each core copies its heads' tiles into acc_hi), K/V through the B fifo one object at a time (two
P buffers keep prefetch), two passes (query tiles {0,1}, {2}) so the online-softmax state fits; O as bfp k-step
chunks of out_proj's A with a 25,344 B k-step stride.
EW cores: `ew_norm(mode)` with n_in from a header slot, `ew_patch` (12 k-step GEMMs from bf16 patches), `ew_ssm`
(head-major: per head a 14-slot parameter object, 18 chunks of 7 KB). Patch embedding as an EW op keeps the 24
GEMM cores on 12 identical block passes (`pre` / `body` x reps / `post` op phases).
Numerics (nk8.h): exp32 with a hi/lo argument, reciprocal = elementary-unit seed + two fp32 Newton corrections,
GELU = x * sigmoid(2u) with u in fp32 (15% of bf16 roundings flip, <= 2 ulp), sqrt by int16 seed + 2 Newton steps.
### 8.2 Decisions
- Dependencies explicit: `plan8` records the last writer of every region; the sequence finishes exactly the
  producer's group before a consumer (the lookahead pipeline alone passed t2 only by timing).
- The qkv epilogue drains per tile (12 per column) and the attention fills per head (4 per row per pass) because
  a 4608 B object exceeds the 1023-word inner TAP dim.
- Half-slot sends: K=768 tiles consume one half, K=1536 both (`nsend = nt` / `2 nt`); the attention 2.
- `--reclaim-runtime-bds` for the 206-op build (IRON's static BD count disagreed with ours: peak 14 of 16);
  `--lookahead 1` there. Both cost time and need a proper fix.
### 8.3 Learned (see the skill file)
finish() frees unwaited fills; output streams vs calls and alignment; NULL static pointers across entry points;
16 KB program memory; stale schedules; the half-slot send count; V^T panels share exponents over keys (the host
reference must encode the transposed matrix); attn_tn's `j <= NB` PV loop touched a 9th dim block (its 1.7e-2
rms error).
### 8.4 Where the 269 ms go and what to change
GEMM frame 62 (35 in design7 + GELU epilogues + per-op deps bubbles), SSM 80, attention 17, norms ~18 (0.2-0.25 ms
of port time + a pipeline drain each), patch + final ~2, the rest = ~200 serialising sync points at ~0.1-0.3 ms
and the 7.5 MB instruction stream. Beyond fixing the SSM kernel: fewer, larger DMA tasks; EW ops overlapped with
the next GEMM (double-buffered regions so `finish_all` is not needed); the GELU epilogue in asm.

## 9. Round 4 (2026-10-09): correct with the real weights, 269 -> 107 ms
### 9.1 Correctness
- Attention: round 3's late fix (g_srb 384 -> 768 B in mm8.cc's attention map; the running maxima aliased the
  scores of key blocks 3-5) verified on `t4`: O rms_rel 7.2e-3, corr 1.000.
- `step8.py` runs the step with the real weights: W packed from the safetensors per plan op (yx = [y; x] rows,
  ffw_up interleaved [32 gate | 32 up] then halves for up_a/up_b, ffw_down / mlp.3 split at K 1536 with the bias
  on the first half, in_proj, norms as (scale + 1) / gamma, beta, SSM per-head params with `first` patched in W
  per frame), ACT stays on the device across frames (the SSM areas are the recurrent state; `reset()` zeroes them).
- The GELU bug (nk8 gelu32 multiplied the clamped x) was invisible to every random-data test; `step8.py blocks`
  (depth-1 build, every region vs the torch block) found it in one run. Keep that test in the loop.
- The patch embedding moved to the host: torch's bf16 conv on the CPU takes 0.2 ms, gives torch's exact numerics,
  frees 0.7 KB of EW program memory and 1.2 ms of device time. X0 = [conv + pos-emb; query tokens; 0] is written
  each frame (2.8 MB, Q16 X' parts); block 0's norm0 has one input.
### 9.2 Speed: what the host-polled timeline showed
The timeline (`step8.py timeline`) polls, while the dispatch runs, the first and last word every op writes in
column 0 (64 B `sync(FROM_DEVICE)` per probe: host_only bos are CPU-cached, mapped reads alone never change) and
assigns the k-th change of an address to the k-th op that writes it. ~17 us resolution, no trace hardware, the
real program. `DETAIL=<op>` watches every object / every column of one op. It found, in order:
1. norm3 took 6 ms: its 2-input fill had a 10.5 MB step (R3 and R4 far apart). The BD step field is 20 bits
   (words); a larger step still transferred correctly but the columns ran one after another. R4 now follows R3.
2. -Oz on the EW core: 2.4x slower than -O2 and not smaller. -O2 for both kernels.
3. SSM 7.2 ms/block: rewritten (per-head broadcast tables instead of `rep4` calls in the loops, transcendentals as
   array loops, slot-wise output instead of `out64` per 64 B), then the reciprocal: `aie::inv` is scalar. 1.45 ms.
4. Two-chain helpers (exp, sigmoid, sqrt, gelu): Peano schedules one chain per basic block with nops in every
   MAC-latency slot; two independent chains interleave (loop bundles per vector: exp 73 -> 38, sigmoid 140 -> 77,
   gelu 199 -> 111); four spill. Applied to the attention's exp, the GELU epilogues and the SSM.
### 9.3 The floor of this architecture (estimated, see HANDOFF 4)
Per block today 8.9 ms: GEMM ops 4.4 (incl. GELU epilogues ~1.2 and gaps), norms 1.36, SSM 1.45, attention 1.75.
The DDR round trips (~1.5 GB per step) put a ~40 ms floor under the design; fixing every inefficiency named in
HANDOFF 3 should land at ~55-65 ms. The activation-stationary redesign (weights streamed, activations in the
memtiles, norms folded into GEMM prologue/epilogue) is the route to ~20-25 ms.

## 10. Round 5 (2026-10-09): 106.8 -> 89.4 ms, depth chosen at run time
### 10.1 Changes (each measured with the timeline tool; RESULTS 14)
1. **Norms** (ew8.cc `norm_rows<LN>`): 0.31-0.36 -> 0.10-0.13 ms each. The old kernel wrote its output through `out64`
   (a noinline call with lock bookkeeping per 64 B, ~300 calls per row block), branched on mode / n_in inside the
   inner loops and did the row statistics in emulated fp32. Now: one loop body per mode (template), the missing
   input streams re-read stream 0 with weight 0 (no n_in branches), outputs stored straight into the ring
   (`out_begin` / `out_end` reserve and commit slots per k-step), statistics on 32 lanes in the MAC unit: a row-sum
   butterfly (`rowsum8`: interleave_unzip/zip + accumulator adds, native), RMS mean / LN variance with three-part
   bf16 splits (`split3`, `mul3`), var = (768 q - s^2) / 768^2 with exact products, rsqrt by bit-trick seed + one
   bf16 and two hi/lo Newton steps, nb = -mean rstd from three-part factors. The norms are DDR-bound now (~57 GB/s
   for the 3-input ones). eps travels in the params header (word 1): the encoder norm uses torch's 1e-5 (round 4
   hard-coded 1e-6 for every norm).
2. **One EW entry, depth-independent cores**: LLVM fully unrolled the EW worker's 12-block loop (60 calls, 2 KB of
   the 16 KB program memory). The EW core now calls `ew_op` once per iteration of IRON's endless worker loop;
   `ew_op` reads the op kind from header word 3 of the op's first params slot (pack8.EW_KIND). The GEMM cores run one
   block per worker iteration. The cores' code no longer depends on the depth: the elfs and CDO of full{12,10,8,6,1}m
   are byte-identical, only the runtime sequence (.bin) differs. plan8.full_step lays out ACT and W for 12 blocks at
   every depth, so one packed W serves every depth. `Step8.set_depth(d)` swaps the instruction bo (no rebuild, no new
   hw context; call reset() after raising the depth). EW elf 16,272 -> 15,216 B.
3. **Attention**: lazy online max (the stored row max moves only when a row's max passed it by TAU = 8, so P <= e^8;
   O and the sums are fp32 and P is bfp16, so the scale is free): rescaling cost 0.42 ms per block. P's exp is
   `nk8::expd_i`: z = (s - m) log2e taken in fp32 straight from s and m (round 4 rounded s - m to bf16 first),
   quadratic 2^f, 2^n added to the exponent bits; ~30 instead of 38 bundles per vector. 1.77 -> 1.35 ms per block;
   t4 O rms_rel 7.16e-3 (round 4 7.2e-3).
4. **Two ops in flight** (`--lookahead 2`): shim BD peak still 14 (qkv); outputs bit-identical; ~0.1 ms per block.
   The remaining ~0.1 ms gap per dependent op is the A-slot load: the slot is single-buffered and its load waits for
   the previous op's last send. A split load would start the first send after half the slot (load BD 1: acquire
   empty nsend_prev - 1, release full 1; BD 2: acquire 1, release nsend - 1; nsend is always even so the previous
   op's last send reads half 1): ~25 us per op, not built.
5. **SSM**: exp(2 log_a) = a^2 from the fp32 a (`exp32a_i`, `exp_a2_n`), with torch's bf16 roundings of e2 and
   1 - e2: one exp and one array pass less. 1.45 -> 1.38 ms per block. The two sigmoids per element (~0.37 ms per
   block) are the rest of the transcendental cost.
### 10.2 What did not work / was not built
- **Reciprocal-free GELU** (gelu = relu(x) - |x| s, s = u/(1+u), u = exp(-|t|), s by a polynomial in u; or h(|x|) as
  one polynomial): numpy emulation of the exact op sequence (r5/gelufit*.py) gave 26-37% bf16 mismatches, up to
  4-6 ulp, rms 2.3-4.5e-3 against today's 15% / 2 ulp. Every 8-bit operand rounding costs 2^-9 and buying it back
  needs hi/lo splits; a polynomial in |x| on [0, 4.5] is ill-conditioned in bf16 Horner (1.6e-2). Not built.
- The GELU chain is 0.77 ms per block (`-DNO_GELU`: 89.5 -> 80.3 ms per step); the epilogues without it still cost
  ~0.2 ms per op over the GEMM rate (mlp0 0.46, up_a 0.43 vs ~0.25): `acc_epilogue` is compiled `minsize`.
### 10.3 Floor and redesign
REDESIGN.md: the vector work (GELU, epilogues, attention, SSM: ~4.1 ms per block) dominates, not the DDR round trips
(~1.5 ms per block: gaps + norms). The activation-stationary design is estimated at ~62 ms with today's kernels and
~44 ms with 2x faster vector kernels; GPU parity needs the GEMM loop near the MAC bound too.

## 11. Round 6 (2026-10-09): 89.4 -> 76.3 ms, tooling, the LUT GELU
### 11.1 Changes (each measured; RESULTS 15)
1. **Epilogue as passes** (mm8.cc `bias_pass`, `acc_epilogue`): round 5's `acc_epilogue` was `minsize` and called
   `blk_src` / `blk_bias` / `gelu32x2` per 64-element block with spills: ~77 bundles per block before any GELU
   (r6/asmstat.py). Now one noinline bias pass (row block unrolled over its 8 column blocks, constant bias offsets,
   pointer increments: 44 bundles per 8 blocks) writes bf16 either in the drained layout (flags 0/1) or compactly into
   the first half of each row block's own fp32 area (`rb_bf`, flags 2-6; keeps the gated stash and every packed output
   behind the reads), then the GELU runs as an array pass and the packing pass reads bf16. Bit-identical outputs.
   89.40 -> 84.14 ms.
2. **attn_end without the scalar unit**: the ones panels are ones in all 8 dims, so every lane of a row's sum holds the
   sum and 1/sum is a 64-lane vector reciprocal (`nk8::recipa_i`, the old recip1p_i body generalised) in the row
   layout; round 5 ran aie::inv on bf16(sum) (scalar, 1,264 B of code) and gathered 8 lanes. It also uses the fp32
   sum: t4 O rms_rel 7.16e-3 -> 6.99e-3. Not bit-identical to round 5 (tokens differ at the level of NPU-vs-CPU noise).
3. **GELU by lookup table** (mm8.cc `gelu_lut_i` / `gelu_lut_n`, tables gelu_lut.h from r6/lutgen.py): AIE2P's 4-way
   parallel lookup (`load_lut_2x_float`, aie::lut<4, float, bfloat16> layout: (slope, offset) float pairs, groups of 4
   entries stored twice, a second copy in another bank; index = a byte offset, 16 B per segment). gelu(x) = x s(x),
   s piecewise linear in 64 segments of 1/8 on [-4, 4) (bf16 slope rounded, fp32 offset refit), s hi/lo into the
   product. The ViT MLP (gelu_pan, mlp0) now uses the erf form torch's nn.GELU() computes (every earlier build used the
   tanh form there). One core: 16 ns per 32 lanes vs 63 (r6/kprobe.py time). Real-activation rms error = round 5's.
   The ab copies live in the GEMM cores' data area (data_size 25,088 -> 27,136, stack 6 -> 4 KB: measured need
   ~2.5 KB = attn_chunk_k 1,664 + rescale_rb 768), the cd copies in bank 3 (`.aie.bank3` section). The hand-written
   lookup is 2.7x faster than aie::linear_approx, whose object is reloaded from the stack on every compute.
   84.14 -> 78.16 ms; DAVIS AJ 70.83.
4. **Attention loops at -O2**: chunk_k / chunk_v lost `minsize` (PV loop 56 -> 42 bundles per 12 mmuls) and the P exp
   is one in-place array pass per row block over all 6 key blocks (`expd_n`, constant trip count; invalid blocks
   zeroed after) instead of a noinline expd32x2 call per 64 scores (88 -> 65 bundles per 64 scores). Bit-identical.
   78.16 -> 76.34 ms.
### 11.2 What did not work / was not kept (measured)
- **GEMM inner loop variants** (P1 says the GEMM is compute-bound, so the loop matters): the round-5 hardware loop is
  28 bundles per 16 mmuls and exactly ldb-slot bound (8 A pops + 16 B pops + 3 fills: every block-vector pop issues on
  the one ldb slot). Two column blocks per A pass (B repacked [n/32][k/8][4 panels]): 75 bundles per 32 mmuls with
  spills (only 5 dm registers for 4 accumulators + loads). Half-k passes holding 2/4/8 A blocks in registers: 26 per
  4 / 35 per 8 / 48 per 16 mmuls. The compiler hoists A into 8 of the 12 ex registers and then has 4 left for B pops
  in flight (pops land at +8). Beating 28 needs a hand schedule that reuses a pop's destination register before the
  previous value is consumed (exposed pipeline): the asm route. Not tried this round.
- **Core timer** `read_tm(0x340F8)` hung the core (timeout 1 of the round, device recovered). No cycle builtin exists.
- **Hardware tanh** (`aie::tanh`, vtanh) and `aie::exp2`: too inaccurate (vtanh is the identity for |x| <= 0.5 and
  2-7% off up to |x| = 2; exp2 p99 10 ulp). `aie::inv` is correctly rounded but scalar.
- **LUT sigmoid for the RG-LRU** (symmetric table on [0, 8), 1 KB per copy, fits the EW cores' banks 2/3): far fewer
  bf16 mismatches overall (5% vs 18.6%) but 1 - s loses the relative accuracy of small sigmoids, and gate_a feeds
  a = exp(-8 sp sigma) near 1: |da| / (1 - a) reaches 1.6e3 (nk8: 1.4e-2). A direct [-16, 8) table needs 3 KB per copy
  and the EW cores have no second bank with room. Not built. (gate_x alone could take the table; small gain.)
- `AIE_PREPARE_FOR_POSTPIPELINING` on the exp loop: 100 bundles per trip instead of 65 (static count).
### 11.3 Tools added (r6/)
`asmstat.py` (bundles / nops / spills / calls per loop of a .s), `ccs.sh` (compile like the design + size table +
loop table), `tlview.py` (timeline JSON -> per-op roofline + diff of two builds + HTML Gantt), `step8.py regress`
(bit-level regression of the final tokens and tracks between builds) and `timeline` JSON output, `kprobe.py/.cc`
(every bf16 input through candidate primitives on one core: accuracy vs exact; `time`: repetition-timed cost),
`emu.py` (bit-level numpy models of the nk8 ops, 99.2-99.9% bit-equal to the device), `capture.py` (real activations
at every nonlinearity, CPU model), `numh.py` (ranges vs kernel limits; GELU / sigmoid candidates with the bfp-ratio
yardstick), `lutgen.py` (LUT tables in the aie::lut layout).

## 12. Round 7 (2026-10-09): 76.3 -> 71.2 ms, the clock, the core timer, the asm GEMM call
### 12.1 Changes (each measured; RESULTS 16)
1. **Attention exp by table** (mm8.cc `aexp_arg` / `aexp_val` / `aexp_push`, table exp_lut.h from r7/explut.py):
   80 segments of 1/4 on [-11, 9), the 4 bottom ones zero (P = 0 below e^-10 of the stored max), bf16 slope (relative
   least squares) + fp32 offset (minimax relative error for the rounded slope). p = (off - slope m) + slope s with
   s, m bf16 (exact products, no hi/lo split of d); P converted to bfp16 straight from the fp32 accumulator (fused
   with the push: round 6 rounded P to bf16, stored it and ran a separate push pass). Emulated first on real scores
   (r7/expemu.py, 5 blocks, lazy max replayed): softmax-weight row-norm error 0.91-0.96x round 6's expd_i; h = 1/2
   or a range ending at -7 or higher was worse. Two hand stages in the loop (next block's index argument while this
   block looks up and pushes): 49 bundles per 64 scores incl. the push (round 6: 65 + the push pass).
   Memory: master copy in the data area (stack 4 -> 3 KB: deepest chain measured 2.24 KB; data_size 27,136 ->
   28,160), second copy written by attn_begin into the C object's unused tail (bank 3).
2. **attn_begin / attn_end**: round 6 wrote the 432 B ones panels byte by byte (17 bundles per byte, ~4 us per head
   group) and zeroed O with a divided index; now 6 bfp pushes of a 1.0 vector, O zeroed only for the pass's tiles
   (minsize, 4 stores per trip: program memory), attn_end at -O2 (j loop 82 -> 32 bundles).
3. **QK^T with Q popped per key block** on the other load slot (lda: Q, ldb: K; round 6 held Q in 8 registers and
   spilled), a constant trip count (invalid key blocks masked to -1024 SU after, max from the stored scores):
   ~70 -> 53 bundles per pair of key blocks. The same MAC order: bit-identical.
4. **Unscaled scores**: the 1/8 is a power of 2, so the scores stay q.k and the 1/8 moves into the table (slopes x 1/8,
   index shift -3), TAU / NEG / the mask (x 8) and the rescale argument (x 1/8, exact): 53 -> 46 bundles, bit-identical.
5. **The GEMM call in hand-scheduled asm** (r7/gen_mm.py -> mm_core.s; design8 builds mm8.cc + mm_core.s merged by
   ld.lld -r and links it with `Kernel(...)` when the kdefs hold -DMM_ASM). 4 accumulator chains (a group = row block x
   pair of 16-column blocks = 32 MACs, one per cycle, each chain 4 apart) instead of 2 at distance 3; the B panels of
   a pair in pop order [q][4 col blocks] (pack8 `B_PAIR4`, a host permutation inside each weight object; GEMM weights
   only: the SSM / patch parameter tiles keep the old order - the first full-step run permuted those too and the SSM
   output was garbage, `step8.py blocks` found it in one run). 225 bundles per call vs 321, 0 asmcheck violations,
   BIT-IDENTICAL (t1 / t2 / t4 / g11 directed tests as round 6; regress vs full12r7a). 73.10 -> 71.19 ms.
   Group starts every 48 cycles (32 MACs + 16 serialised stores / loads at the boundary on 5 accumulator registers).
6. **Tools**: r7/clk.py + clk.cc (clock probe; `--stamp` = P15 timer test), r7/prof.py + mm8.cc `-DPROF` + design8's
   bus enables (per-tile mm_acc cycles / calls / k-loop wall / epilogue from the core timer), r7/t4var.sh (build +
   time a t4 variant), r7/expemu.py (attention exp candidates on real scores), r7/explut.py, r7/gen_mm.py,
   `step8.py timeline` DETAIL=bN_attn (every head's O drain time), the attention twins (-DATTN_KTWICE / VTWICE /
   ETWICE) and stubs.
### 12.2 Measured facts
- **Clock 1.81 GHz (Turbo)**: two pure-register loops, host-timed slope and the core timer agree (RESULTS 16.1).
- **read_tm works once the processor bus is enabled** from the runtime sequence (npu_maskwrite32 0x32038 bit 0 per
  core) and each read is fenced with __builtin_aie2p_sched_barrier(): exact cycle counts, no hang (24 cores).
- **mm_acc runs at its static schedule in the real step** (333 cycles per call measured = 321 + the stamps): no bank
  stalls (P17 refuted). ~400 cycles per k-step go to the 9 calls + IRON's lock code. GEMM tiles wait mostly on
  dependencies: the first tiles after EW ops (180-204 us after norms, 1.37 ms before `out` = the SSM) and the
  K = 1536 ops' mid-op A reloads.
- Attention (t4 twins, r7b): K phase 0.60 ms (exp + push 0.30), V phase 0.19, data/control ~0.18 not overlapped,
  per head group 25-35 us (pass 0) / 15 us (pass 1) of fixed cost.
### 12.3 Dead ends (measured)
- C++ GEMM loops with 4 chains: 54 bundles per 16 MACs (spills), 42 without the accumulator prefetch, 61 with A popped
  per block via an opaque seek: Peano keeps A in 8 of 12 ex registers and does not pipeline across blocks.
- Carrying the looked-up exp lines across loop trips (65 bundles, spills); chunk_v with P popped (67 vs 42 per pair).
- **Re-seeking a block-vector input stream after popping from it returned garbage** (t4 O = inf); a fresh stream object
  per block works.
- -O2 attn_begin is 1,104 B (minsize 800 B); with the asm call the GEMM elf overflowed program memory by 96 B.

## 13. Round 8 (2026-10-09): 72.2 -> 67.9 ms, the asm gap explained (L1 banks), the SSM and attention stamped
### 13.1 Changes (each measured; RESULTS 17)
1. **A ring pinned away from the B fifo** (design8 `A_BANKS=1,2,2` -> `Buffer(mem_bank=)`; the allocator then moves
   the C object to 35072): -1.5 ms, bit-identical. Build every round-8 step with A_BANKS=1,2,2.
2. **SSM kernel (ew8.cc), bit-identical**: `gate_gemm` (gemm8 with nt = 1 and constant trip counts: 1,190 -> ~300
   cycles per call), `in_part` (contiguous 1 KB views of the chunk's 7 input parts; the one a ring wrap splits is
   copied to g_ptmp), `out_copy` / `copy_v` (pointer-bump copy loops, at most one wrap: indexed addressing cost 8
   bundles per vector pair), conv / recurrence as 8 trips of 2 halves, `sqrt32_i`'s Newton loop unrolled. EW
   data_size 26,112 -> 27,648 (g_ptmp, g_panels padded to 1 KB so the pad copies with the panels). -2.4 ms.
3. **One table-walked op loop on the GEMM cores** (design8 `ONE_SEG=1`, default): attention is an entry of the
   op table (tbl_a = 1, tbl_n = 0 skips the GEMM nest), the attention loops take 3 / 4 / 24 / 2 from tbl_c (constant
   trip counts let LLVM unroll the 3 x 3 Q-copy loop into 9 calls). Text 16,176 -> 14,848 B; -0.4 ms. -DPROF now
   fits the full step.
4. **Tools** (r8/): `mmb.py` + `mmb.cc` (one-core GEMM-call bench: operands at chosen L1 offsets, timer-read cycles
   per call, every bank assignment; `gen_mm.py --split` = mm_core_split with separate load / store pointers),
   `gprof.py` (g11 / any run8 test from `PROF_OUT=X.npy`, and `--step BUILD`: per-op cycles per call, gaps inside /
   at k-step starts, first wait, tile handoff, acc_zero, plus the attention head-group table), `ewprof.py` (SSM
   phase cycles from records ew8 -DPROF writes into each chunk's unused output pad: the step stays correct),
   `cmpact.py` (bit-level ACT compare of two run8 tests: SSM changes checked on t3 against `build/t3base.npy`).
   run8.py `PROF_OUT`; r8.sh passes it; ew8 `-DEW_NO_NORM` (profiling SSM-only tests within program memory).
   gen_mm.py `--noconf` / `--noconf-ab` (P21 rules as scheduler constraints; both lose, RESULTS 17.1).
### 13.2 Measured facts
- **L1 banks are 16 KB (4 per core); two accesses to one bank in one bundle stall the core.** Same-slot ops never
  co-issue, so A pops and accumulator loads (both lda) do not conflict. The asm call (3 memory ops per bundle)
  stalls where the Peano loop (fewer co-issued accesses) did not: C load + store in place +33 cycles per call, A + B
  in one bank +14, B with C +42. That is the round-7 gap (-1.9 vs -7 ms): in situ the call ran ~290 of 225.
- GEMM tile boundary: handoff 1.31 us, acc_zero 0.49 us, plain epilogue 0.64 us; op-start A-slot wait 27-50 us.
- SSM: compute-bound (waits 1.3%); after round 8 the transcendentals are 52% of it (sigmoid 2,541 + exp 1,506 +
  sqrt 1,093 cycles per chunk of 512 values: ~80 cycles per 32-lane vector each, latency-bound 2-chain Peano code).
- Attention per head group: fixed costs ~10 us (pass 0) / ~7 us (pass 1) of 135 / 71: the K phase is the bulk.
### 13.3 Dead ends (measured)
- P21 as scheduling rules: R1 (no accumulator load + store in one bundle) 273 bundles static; A ops kept off B's
  cycles 255. A layout where consecutive groups' accumulators live in different banks keeps 225 conflict-free
  (designed, not built: 36 pieces of 1 KB, NOTES 13.4).
- Peano: sched barriers (read_tm stamps) inline in ew8's pipelined chunk loop crash clang (AIE Region assertion):
  stamp through a noinline function.
### 13.4 Designed, not built: conflict-free accumulators for the asm call
Groups run (rb0, pair0), (rb0, pair1), (rb1, pair0), (rb1, pair1); at each boundary the next group's loads overlap the
previous group's stores. Give each (row block, pair) its own 1 KB piece with consecutive pieces in different banks:
stripes 0-5 put their pair-1 pieces in the C object (slot = global row block: bank 2) and pair-0 pieces in acc_hi;
stripes 6-8 alternate bank 0 / bank 1 inside acc_hi (acc_hi = 24 pieces at 0x71900: 9 in bank 0, 1 straddling, 14 in
bank 1). mm_core takes 4 piece pointers (p2-p5 as load cursors, reset from r-registers as store cursors; pB saved in
r5: `mov rX, pY` / `mov pY, rX` assemble). The epilogues read pieces through a (row block, pair) map; the in-place
orders stay safe (a C slot holds a piece of row block <= its own index). Expected 283 -> ~250 cycles per call in
situ (-2.5..-3 ms); do it inside P20's generator rather than twice.

## 14. Round 9 (2026-10-09/10): 67.9 -> 58.1 ms, SSM tables, the k-step in pipelined asm, the GEMM now A-port-bound
### 14.1 Changes (each measured; RESULTS 18). All switchable by flags; the r8c build still reproduces.
1. **SSM sigmoid + sqrt by table** (EW_DEFS=-DSSM_LUT; r9/ssm9.h, tables r9/ssm_lut.h from r9/ssmlut.py): sigmoid = line
   per 1/8 segment on [-12, 8) (x clamped: x < -12 gives a = 1, mult = 0 exactly as torch since sp <= 1.18), exact bf16
   sqrt from a 256-entry mantissa table (int16 LUT, exponent halved in int16, w = 0 -> 0 via min(r, 2^60 w)). New
   numerics, more accurate than nk8 on every metric; DAVIS 70.84 / 69.90 (d12 / d10). -2.8 ms.
2. **Tables in two banks**: SSM arena compacted (hb over gx, 1 - e2 over the dead param vector, tba into the freed slot),
   arena 24 KB -> 21.5 KB, the sigmoid table's second copy in .data (bank 1). -0.35 ms, bit-identical.
3. **exp_a2 as five pipelined passes** (-DSSM_EXP3): the same ops split so Peano software-pipelines each loop (11.5 + 20 +
   12 + 6 + 4 bundles per vector against 63.5). Constants must be locals in the pass (through `const ExpC &` Peano
   reloaded them inside the loop). Needed 4 ring helpers -> minsize (program memory). -0.22 ms, bit-identical.
4. **mm_kstep** (kdefs -DMM_KSTEP): one call per k-step; the A ring's locks pinned (design8: aF = lock 4, aU = 5) and taken in
   the kernel. -2.0 ms, bit-identical.
5. **Accumulator pieces** (-DMM_PIECES; NOTES 13.4 / research rev 4): 36 x 1 KB pieces, affine layout (pair 0 of rb < 12 in
   C slot rb; pair 0 of rb >= 12 in acc_hi 0..5; pair 1 in acc_hi 6 + rb), epilogues rewritten as strided passes over
   pieces (bias_pcs, two pieces per trip; rb_bf2 halves). -0.9 ms (greedy table map, slow epilogue) / affine kept.
6. **mm_ks** (-DMM_KS; r9/gen_ks.py): the whole k-step in asm, a zero-overhead loop over 9 stripes, modulo-scheduled at 192
   bundles per stripe (the stripe boundary overlapped), record stream through one post-incremented pointer (A slot, 4
   cursor jumps, acq flag per stripe; order in the generated build/mm_ks.h), a dummy 10th stripe (acq.cond skipped),
   "stripe -1"'s spilled stores harmlessly rewrite piece 0. The generator checks every MAC's A / B / accumulator operands,
   cursor addresses, record-register timing and lock order on the unrolled 9-stripe stream. -3.5 ms, bit-identical.
   GEMM text 15,840 B; GEMM stack 0xC00 -> 0xB00 (+256 B data for the 232 B record table).
### 14.2 Measured facts
- ~~The GEMM k-step is A-port-bound now~~ **CORRECTED in round 10**: the 2,588 (down_a) / 3,300 (out) cycles per k-step were
  MEANS over tiles. Per-tile medians (r10/gdist.py on build/g11sp.npy) are 2,050-2,065 for every op: steady k-steps run at
  their compute (static 1,830 + banks 24 + call/stamp overhead). Each op's first tile carries the A-slot load wait (28-34 us)
  inside its first in-kernel acquire; K = 768 ops with few tiles show it most. A arrives at >= 5 B/cycle per core. The
  ~4 B/cycle that is real: the DDR -> memtile slot load per column (8 x 248 KB in ~34 us = DDR read peak).
- LUT lookups (load_lut_2x_float / _int8): both copies in one bank work, +1 cycle per 4x load (+8 per 32 lanes);
  bfloat16_to_int is vfloor. A 16-bit table is entries doubled in runs of 16 (lut_inv.h layout), index = 4 B per entry.
- Peano: separate in / out pointers and one vector per trip let it software-pipeline a table loop (24 vs 27 bundles);
  loads-before-stores in source order avoid "every load waits for the previous store" (aliasing); 2 pieces per trip.
- acq with a positive value is acquire_EQUAL (Peano: acquire_greater_equal(n) = acq -n); acq.cond #id, rV, r26 runs when
  r26 != 0. Zero-overhead loops: start 16 B aligned (.p2align 4 pads with nops), first / last bundle in full 16-B form.
- .bss after an initialised .data in a core's data area is zero-filled (probed). IRON jit caches by CompileTime args:
  put the flags in a CompileTime key or builds silently reuse the first. IRON Locks only in worker args need
  rt.add_lock() so their Tile resolves.
- The SSM numerics: any ulp-level change (deployed nk8, tables, random flips) gives the same track noise on the CPU
  model (~0.015 px mean, 0.1-0.2 px max over 30 frames): judge SSM numerics by DAVIS, not by 30-frame max deviations.
### 14.3 Dead ends (measured)
- exp by a range-reduced 2^f table: as accurate as nk8, not faster (61.5 vs 60.7). Direct PWL exp on log_a: 17-29% a
  mismatches. Fusing log_a into pass A: 50 bundles per vector (not pipelined).
- Greedy runtime piece map (bank-aware incl. A slots): faster k-step (323 vs 333 cycles per stripe) but forced a
  table-indirect epilogue (0.64 -> 1.45 us per tile); the affine layout + strided epilogue won overall.
- A_BANKS variants with pieces do not link (bank sections overflow).

## 15. Round 10 (2026-10-10): 58.1 -> 53.9 ms, the A slot fixed at the source (loads once, runtime-driven, streamed)
### 15.1 Changes (each measured; RESULTS 19). All behind flags; full12r9s still reproduces without them.
1. **A_HALF (plan + design8)**: the memtile slot's halves get their own S2MM channel (0 and 5) and lock pair, both fed by ONE
   packet flow (id 4) multicast from shim channel 0. K=768 ops and attention's Q objects load 124 KB once; K=1536 ops
   stream 248 KB that both channels write over the whole slot (identical values); `reuse_a` (up_b) loads nothing.
   -1.35 ms. Superseded by A_RT for the memtile side (A_RT keeps the A_HALF plan: single fills and reuse_a).
2. **P23**: yx and qkv deal N tiles round-robin over the 3 rows (row r, step j -> tile 3 j + r); pack8 orders weights / flags
   by natural tile, drains stride 3 k-steps; qkv's step j is kind j // 4 for heads 3 (j % 4) + r. -0.85 ms.
3. **A_RT**: the memtile A slot is driven by runtime-sequence tasks (IRON `FlowEndpoint.task`): per column 8 BDs configured
   once per dispatch: load lo / hi (one 10,368 B k-step chunk per run, BdIteration 12; acquire E_lo / E_hi, release F),
   tile-0 sends (acquire F, release F2), middle tiles (whole region, no locks, repeat nt - 3), last-tile sends (acquire F2,
   release E). Per op the sequence starts the loads after the A fill and pushes the sends right after the B fills. The
   first tile starts after one chunk; the next op's load of chunk q waits only for the last tile's send of chunk q.
   -2.0 ms. Bench first: r10/rtm.py.
4. step8 / run8 apply a build's plan flags from build/<name>.json (a_half, a_rt, p23); run8 asserts the sched is current.
### 15.2 Measured facts
- The steady k-step is 2,055 cycles in mm_ks + ~70 between calls (per-tile median); see NOTES 14.2's correction.
- Memtile -> core stream 7.95 B/cycle. With all 8 columns loading their slot from DDR the load runs ~4 B/cycle per column.
- A BD needs both a lock acquire and a release, or neither. Tile / memtile BDs refuse stride 0. A full memtile queue
  (4 tasks) blocks the command processor: push an op's tasks before any of its waits.
- IRON's pathfinder drops a multicast destination when unicast flows share the same DMA master ports (r10/pf1.mlir).
- The GELU-by-table loop is not stall-free when Peano is forced to pipeline it (17 static -> 54.7 measured cycles / 32
  lanes); the deployed 2-vector loop runs at its static count (34.2).
### 15.3 Dead ends (measured)
- Unicast + multicast packet ids for the slot halves (router bug; hung one dispatch).
- GELU_II (forced initiation interval): slower in situ.

## 16. Round 11 (2026-10-10): 53.9 -> 52.1 ms, a validated modulo-schedule generator, the real register-timing rules
### 16.1 Changes (each measured; RESULTS 20). All behind flags; full12r10r still reproduces without them.
1. **GELU_II made real** (mm8 `-DGELU_II`, `gelu_lut_io` with separate restrict parameters; `GELU_II_MIN` 4): Peano's
   17-bundle forced-II loop, 17.8 cycles per 32 lanes on one core. -0.88 ms. Superseded by GELU_MS.
2. **GELU_MS** (kdefs `-DGELU_MS`; design8 asm_object runs r11/gen_gelu.py -> build/gelu_ms.s/.h and links it): the GELU chain
   as a hand modulo schedule, II 13, one call per run walking pieces with 2D post-modify (`jump` 64 / 576). Bit-identical.
   -0.53 ms more (gated / pan / yx epilogues).
3. **P43** (design8 env `P43=1`, recorded in build/<name>.json): GEMM B fills before the dependency wait. -0.35 ms.
4. design8 compiles mm8 with `-fstack-size-section`; bb.sh runs r11/stackaudit.py after every build and fails on overflow.
5. step8 `HOST_FAST` (default 2): X0 packed by torch straight into the mapped ACT bo (0.64 -> 0.14 ms host); `HEAD_THREADS`
   (default 0, off: one thread for the heads slows the next conv).
### 16.2 Measured facts (tools in r11/)
- Register visibility = LLVM latency: seen by an op issued >= w - r + 1 after the writer (mk_haz.py). The mv -> mv bypass:
  mv readers see an mv writer's value one cycle early. Older loads and younger short writes to one register land in landing
  order, nothing lost. Itinerary ports (DM_ADA shared by vldb.4x64 + vlda; QEY_RS by LUT index / vfloor / vst x) are
  structural hazards. The last bundle of a zero-overhead loop must be 16-B aligned.
- Peano's pipeline_initiation_interval pragma silently gives the serial loop with restrict copies of one pointer (and
  in r10/gxb.cc with MIN_ITERATION_COUNT(8)): always read the emitted loop, and bench the binary you deploy.
- Host part of the step: conv 0.48-1.1, packing 0.64 (now 0.14), heads 0.6-1.0 ms; variance = OpenMP wake-up + tenants.
- Stacks (aiecc's measured_stack_sizes agrees): GEMM 1,664 of 2,816 B, EW 4,288 of 6,144 B.
### 16.3 Dead ends (measured)
- First GELU_MS attempts: c + r >= c' + w timing (one cycle optimistic) -> garbage; then the mv bypass -> lanes 16-23 wrong.
- A loop body without an aligned last bundle runs once (the copy probes ran at 1 cycle per vector).
- One-thread heads (HEAD_THREADS=1).
- Timeout 1 this round: a tap probe run with the wrong U (the loop ran past its buffer); known-good probe ran right after.

## 17. Round 12 (2026-10-10): 52.1 -> 50.05 ms, the epilogues fused (G1), the SSM sigmoid in asm, an exact scheduler
### 17.1 Changes (each measured; RESULTS 21). All behind flags; full12r11d still reproduces without them.
1. **GUP** (kdefs `-DGUP`, needs GELU_MS + MM_PIECES): gated_a / gated_b's up half + bias, gate x up, bf16, bfp push as one
   asm loop (r12/gen_gup.py, CP schedule r12/gup.json, II 6 per 64 lanes). -0.55 ms.
2. **CPY32** (`-DCPY32`): gated_b's stash copy (18 x 288 B, 320 -> 576 B strides) as a 32 B modulo copy (r12/gen_copy.py,
   II 1, U 3). The C++ 16 B loop cost ~6,100 cycles per gated_b tile, the asm ~430 in situ. -0.30 ms.
3. **PAN** (`-DPAN`): the bf16 -> bfp panel pushes of mlp0 (flag 4) and the Q / K objects (flags 5 / 6) as one loop per
   segment (r12/gen_pan.py, II 16 per row block of 8 pushes; a 2D flush jumps between Q / K objects). -0.30 ms.
4. **GELU_FLAT + BGELU** (`-DGELU_FLAT,-DBGELU`): flat-edge tables (r12/lutflat.py -> gelu_lut_flat.h: 4 flat segments
   per side, bias 36, +512 B: GEMM stack 0xB00 -> 0xA00, data +256) and bias + GELU fused (r12/gen_bgelu.py, II 12, x through
   an 8-slot stack ring) for yx (drained layout), the gate half and mlp0 (bgelu3 helper: three segments). gelu_ms is not
   linked any more. Text 16,288 / 16,384 B (`s % 3` in mm_kstep's record setup replaced by compares: -160 B of __modsi3). -0.32 ms.
5. **SIGK** (EW_DEFS `-DSIGK`): the SSM gate sigmoid as a KERNEL-ONLY asm loop (r12/gen_sig.py, II 12, U 2, backwards over
   g_gx..g_ga so the 4 garbage stores + 1 over-read land in g_hout, scratch then) with a compacted body (r12/asmfit.py:
   350 B, less than the 384 B C++ loop). design8 builds the EW object itself then (ew_object; ExternalFunction cannot link
   asm). -0.61 ms (SSM op -0.52, norms -0.18 in the timeline).
Best: **full12r12e** = r11d + `-DGUP,-DCPY32,-DPAN,-DGELU_FLAT,-DBGELU` + EW_DEFS `-DSIGK`: 50.05 ms, BIT-IDENTICAL.
### 17.2 Tools (r12/)
- **cpsched.py**: CP-SAT (OR-tools) modulo scheduling: one-hot issue cycles, slots + itinerary ports + banks modulo II, the
  EXACT steady-state register pressure per class (live(v, c) booleans summed per residue), `maxdist` (a stream's ops in
  order inside one II), `last_stage` (side-effect ops), `max_life` (bounds U), the schedule pinned to start at 0. Runs on
  the laptop (`uv run --no-project --with ortools python ...`); generators save the schedule as JSON and replay it with
  `--times` (msched `Loop.place`), so builds on testbox need no solver. It found II 6 (gup), 16 (pan), 12 (bgelu, sigk)
  where msched's IMS + sink pass did not allocate.
- msched (r11/msched.py) additions: `place()`, per-op stream offsets `f(i, k)`, `MIN_BODY` 3, `check()` replays every
  prologue -> body padding (PADS 0 / 1) INCLUDING itinerary ports along the real timeline, the one-cycle bypass window only
  for mv -> mv pairs, kernel-only emission + checking (`kernel_only`, `extra_trips()`, `neg_steps` per stream), `body_full`.
- **asmfit.py**: compact loop bodies: assemble, read bundle sizes, expand the cheapest subset to the full form so the last
  bundle stays 16 B aligned (subset sum mod 16), verify. -192 B on sigk.
- Generators: gen_gup, gen_pan, gen_copy, gen_bgelu, gen_sig (+ gen_gelu `--flat --rel --cp --times`); one-core benches
  gupb / panb / cpyb / bgb / sigb (via r11/apb.py `--cc`; apb `--flat --sweep`; the jit key now includes the C++ mtime);
  gen_pushprobe + pushcheck (output-stream semantics); lutflat.
### 17.3 Measured facts
- Output streams (`vst.push.576.conv.bfp16ebs8.fp32` / `vst.flush.512.conv` on [p2, sf, r26]): a flush after every push,
  every second push or only at the end all give the codec's bytes; after a flush the pointer sits at the exact byte (288 B =
  4 blocks, 32 B granular writes), and `vst.flush.512.conv.2d [p2, sf, r26, dN]` adds its 2D post-modify to that (strides
  320 / 576 / 144-B groups verified). Back-to-back pushes need no spacing.
- A zero-overhead loop body of 2 bundles runs once; 3 loop (gen_copy U 2 vs U 3).
- Ops with side effects belong in the LAST stage: the drain's garbage iterations run the earlier stages (bgelu's x store,
  first written to the output, corrupted the 4 vectors after every run: the gated_b stash and pair 1's accumulators).
- `.p2align 4` before the body inserts ONE executable nop bundle (bgelu: 12 B) after the prologue; ops issued in the
  prologue then run one cycle "early" against the body: a vlda's QEY_WA port use collided with a body vfloor (bgelu II 11:
  iteration 1 wrong). msched.check now replays pads 0 / 1 with port reservations along the real timeline.
- 2D post-modify with dj = m = -64 (a backwards walk) misbehaved in the first sigk build; plain `vlda [p], #-64` /
  `vst.conv [p], #-64` post-modify works (the prologue build was bit-exact with them). The first kernel-only build was
  shifted by 3 vectors because only the store pointer was offset; the loads of negative iterations need it too.
- P43's pre-registered timeline check: mlp3_a 166 (r11b) -> 159 us (r11d) per block, not 137 +- 8.
- In situ (g11 PROF, gated_b tiles; gated_a's records are never drained): epilogue 5.11 (r11) -> 3.68 (GUP) -> 2.10 us
  (CPY32). Differentials of r12a: stash copy 1.70 / gate GELU 1.20 / gate bias 0.36 / gup 0.36 us per tile; the panel
  pushes cost 1.23 us per mlp0 tile and 0.82 per qkv tile (PAN_NOPUSH).
- The EW core's PROF build no longer fits (784 B over): profile the SSM by timeline differences now.
### 17.4 Dead ends / not done
- GELU II 13 -> 10 as a standalone loop: with flat-edge tables ResMII is 10 but no 8-register schedule at II 10 (CP-SAT, 400 s,
  4 reload variants; not proven infeasible); the fused bgelu at II 12 replaced the standalone loop instead.
- bgelu at II 11 (CP found one) is rejected by the padding check; a padding-aware CP model would recover it.
- Mistyped rsync put copies of r11/* into the workspace root: moved to build/r12/stray_rsync/ (safe to delete).

## 18. Round 13 (2026-10-10): 50.05 -> 49.21 ms, the attention K phase as one lazy-fast loop, text recovered twice over
### 18.1 Changes (each measured; RESULTS 22). All behind flags; full12r12e reproduces without them.
1. **ASM_FIT=1** (design8 env): r12/asmfit.py on gup / pan / bgelu / cpy32 (-16 / -128 / -64 / 0 B): GEMM text 16,288 -> 16,080 B,
   timing unchanged (full12r13a 50.06), BIT-IDENTICAL.
2. **T1** (mm8.cc, always on with MM_KS): mm_kstep's 58 k-step records as `constexpr` tables (every record is a constant plus
   b x (c_out - acc_hi), or an A slot address): the run-time builder went from 832 B of program memory to ~60 B + 174 B of
   .data (int16 / int8 tables). `minsize` on the old builder saved nothing (scalar code). BIT-IDENTICAL.
3. **ATTKF** (kdefs `-DATTKF`): the attention K phase per row block as ONE generated asm loop (r13/gen_attk.py, CP schedule
   r13/attk.json, II 35, U 1, 3 stages, 96 ops): per key block the 8 Q pops (the stream restarted per iteration: pointer
   reloaded from the stack, FIFO position reset) and 8 K pops, the two mmul chains in today's order, vadd.f, the scores to
   g_srb (bf16, reloaded for the line), d = s + (-m_old) by ups + vadd.f (-m rows8 in g_mbuf), the clamp / floor / LUT line
   with -m, one push per 64, and a flag (vlt.bf16 into r18 / r19, OR-ed into r17 in the last stage, returned in r0: any lane
   with d >= 63.75 SU = TAU - 0.25). The C++ driver (chunk_k_fused) runs today's exact test, rescale and aexp_push over the
   stored scores only for flagged row blocks or chunks with invalid key blocks; chunk 0 always flags (m_old = NEG). The
   C++ chunk_k's QK^T loop is gone (text). The C object was re-laid (scores, -m rows, the exp table copy first: bank 2;
   the K objects alone in bank 3) - no measurable effect, kept. BIT-IDENTICAL (regress, 144-case one-core bench).
4. **--l2-depth 8** (design8 arg, was 4): the memtile B forward holds 8 K / V objects per core row: -0.36 ms (the K / V stream
   is multicast to the 8 columns, so per chunk every column waits for the slowest; deeper buffering decouples them). 12: same.
5. **HOST_HEADS=np** (step8 env): the heads as numpy fp32 + torch.erf (H3): torch 0.94 -> numpy 0.72 ms median (p90 1.16 ->
   0.75) with OPENBLAS_NUM_THREADS=1 (with OpenBLAS threads it was 11.7 ms: thread thrash); max |d| tracks 7.6e-5 px. Not the
   default yet (the step's host part is measured with torch heads in RESULTS 22).
Best: **full12r13f** = r12e flags + ASM_FIT=1 + `-DATTKF` + `--l2-depth 8`: 49.21 ms, BIT-IDENTICAL to full12r9a. GEMM text
16,384 / 16,384 B (0 free), EW 16,224.
### 18.2 What the K-phase numbers say (in situ, timing twins `-DATTK_FORCE=bits`, RESULTS 22.3)
- The fast loop alone (no test, no fallback): 47.14 ms = -2.9 ms, as research rev 8 priced P56. Production: 49.21.
- The flagged units cost it back: ~10 % of units (in-situ count, `-DATTK_COUNT`: 8.6-10.9 % per core = chunk 0 + chunk 22 +
  1-2 % moved maxima) take +400-900 cycles each (the fast pass wasted, then today's exp), and a multicast K / V stream makes
  every column wait for the slowest one per chunk: a 10 % event rate costs ~3x its own time on the op's wall.
- The exact test over the stored scores (max tree + rowmax8 + compare + rows8 prep) cost 1.35 ms when run for every unit
  (r13c); the in-loop flag removed it from the unflagged 90 % (-0.33 ms).
- Next levers (HANDOFF 4): no redundant fast pass on chunk 0 / the invalid chunk (a QK-only variant or the flag known in
  advance), a cheaper slow exp (asm), the 36-iteration form (one call per (tile, chunk): -115 cycles per unit of prologue /
  preamble, needs the 3D Q pop stream), and text for any of it.
### 18.3 Tools (r13/)
gen_attk.py (+ attk.json; --qkonly / --klda / --noflag / --hi71 / --drain debug variants), gen_attqk.py + gen_aexp.py (the
split loops: attqk bit-exact at II 11 but only -27 cycles per row block vs the C++; aexp found no schedule with 6 free x),
attkb.cc / attkb.py (one-core bench of chunk_k vs the new driver on random Q / K / state: hashes of P / maxima / scores, O and
sum checksums, per-block equality, dumps, skip modes incl. "ignore the flag"), lazycount.py / lazycount2.py (fallback rate on
the captured scores), lazycount3.py (the same on the NPU's own 1152 rows from the CPU model, per core), attkcount.py (+ mm8
`-DATTK_COUNT`: in-situ counts written over the attention output), textmap.py, push.sh. msched: ex_n = x_n (PARTS "ex");
cpsched: `min_t`, `hint`. apb.py: `GC`, `NW` / data_size / stack overrides (the bench holds both K paths + acc_hi).
### 18.4 Measured facts
- **ex_n aliases x_n**: a loop holding Q in ex0-7 beside x constants / temporaries computed garbage; pops are x registers.
- **A scalar `lda pN, [sp, #imm]` uses DM_ADA**: co-issued with a vldb.4x64 it loaded garbage (iteration 2 on wrong).
- **Peano's ABI with a return value: r0 reserved, the first scalar argument in r1** (the trip count read from r0 ran 5 trips,
  or 0 -> a hang: 2 timeouts this round, both recovered). Read the call site in `clang -S`.
- A drain's garbage store landing on data a later stage still reads (the -m rows): panel 5 of every row block wrong until the
  loads were ordered before it (maxdist).
- vconv.bf16.fp32 from cmh encodes distinctly and works; `vlt.bf16` targets any rN; `or` is an alu-slot op.
- The r12 kernels' asmfit: -208 B; the EW core had no dead entry points (ew_patch / gemm8 are already dropped by the linker:
  research rev 7's 1.4 KB does not exist).
- Two design builds in parallel in one build dir race on the generated asm (asmfit temp files).
- Attention is compute-bound, not DDR-bound (no K-phase compute at all: 43.4 ms).
- numpy on testbox: OpenBLAS DYNAMIC_ARCH, 48 x 768 x 256 fp32 in 117 us single-threaded; with its default threads beside
  torch's the heads took 11.7 ms.
### 18.5 Dead ends / not done
- The split QK / exp loops (attqk II 11 -27 cycles per row block; aexp: no II <= 28 with 6 free x; the fused loop superseded both).
- A fused loop with Q held in ex0-7 (the alias); K pops on unit A (`--klda`, unneeded once the alias was known).
- Skipping the chunk-0 rescale (O is zero): no measurable gain, reverted for the 16 B of text.
- P29b, the padding-aware solver, H1 / H2, P54's vfloor-from-bm probe: not attempted. P54(1): the .td has no base-register
  form of `vldb.4x64` (the index is an absolute address); the "biased base" is the vsel'd (ab, cd) + bias vector we already use.

## 19. Round 14 (2026-10-10): 49.21 -> 47.31 ms, the K loop batched per (tile, chunk), the V phase and the V^T epilogue in asm
### 19.1 Changes (each measured, RESULTS 23; all behind kdefs, BIT-IDENTICAL to full12r9a)
1. **ATTKE** (r14a, 48.70): chunk 0 skips its rescale (O and the sums are +0 there: +0 x finite = +0); the last chunk (nkb = 2)
   runs the fused loop for its 2 valid key blocks and stores the 4 invalid P panels as zero bytes (today's exp of -1024 SU is
   +0, the device's bfp block of zeros is all-zero bytes) unless flagged. Also: acc_hi's attention map is static (q_buf / erb /
   o_rows, -112 B), slow_rb split out.
2. **ATTKB** (r14/gen_attkb.py, schedule r14/attkb.json: II 34, 3 stages, banks of the in-situ layout): one call per (tile,
   chunk) over 36 (row block, key block) iterations. Q returns to its row block by padda.2d (-576 x5, then 0), K to the object
   start by padds.2d (0 x5, -3456), the -m rows (6 x 128 B, written by C++ into the K object's junk tail at +3456: this core
   holds the object) step by 2D loads (vlda.conv.fp32.bf16 for the fp32 add: no ups op), the scores go to a 7-slot ring
   (896 B: a 6-trip call keeps its scores in slots 0-5, the drain's garbage store lands in slot 6), the P pushes skip the 16 B
   pad by flush.2d, and every iteration stores its flag (st r4, [p6], #4) plus their OR after the loop. The clamp / flag
   bound is an argument (r1: 63.75 fast, 71.5 exact). Driver (mm8.cc chunk_k_fused, -DATTKB): full chunks after chunk 0 batched,
   then for a flagged row block k_rb = the 6-trip loop again (scores back in the ring) + exact test + rescale + the loop with
   the new max and clamp 71.5 (= aexp_push op for op: aexp_push and rows8 are gone, -464 B). Chunk 0 and the last chunk run
   k_rb per row block. Bench (in-situ banks): warm unit 444 -> 261 cycles, cold 933 -> 1,038, last chunk 321 -> 358.
3. **KFOLD** (wrappers inline the phase bodies, -48 B), **ZASM** (r14/zero2.s: acc_zero's two fills as ONE loop, 64 B of the C
   object + 128 B of acc_hi per trip, 627 cycles vs ~864; attn_begin zeroes O tile 0 + the sums and O tile 1 with it too).
4. **ATTV** (r14/gen_attv.py: O += P V two dim blocks per iteration, 24 iterations per (tile, chunk), II 21, 2 stages; `--sums`
   the row sums, II 18) and **VTK** (r14/gen_vt.py: the qkv V^T epilogue, II 13, -496 B of C++ for a 278 B kernel, which paid
   for ATTV's text). V phase per (tile, chunk) 1,316 -> 813 cycles (bench --vphase, hashes equal).
Best: **full12r14f** = r13f kdefs + `-DATTKE,-DATTKB,-DKFOLD,-DZASM,-DATTV,-DVTK`: 47.31 ms (d10 39.84, d8 31.87, d6 24.43).
GEMM text 16,320 / 16,384.
### 19.2 What the numbers say
- The batched K loop saved 183 cycles per warm unit on one core but only 0.04-0.47 ms in situ (bimodal timings): its flagged
  row blocks now cost first pass 393 + exact test 164 + rescale 578 + exact rerun 393 = ~1,530 cycles, and the K / V multicast
  lockstep multiplies that: the twin with the batched chunks' flags ignored (full12r14cF128, outputs wrong) runs 46.15 ms, i.e.
  the fallback costs ~2.1-2.6 ms of wall for ~0.35 ms of its own cycles (x ~6).
- Single-TAP query dealing (research P60) is impossible: tile (c, i) -> (s0 + i d, p0 + i e) must be arithmetic in i for one
  TAP and a permutation per position; only column permutations satisfy both (r14/deal.py, deal2.py; sum of squares). P60 needs
  multi-TAP DMA (attention-specific memtile ld / send patterns, per-piece O drains or a memtile reorder).
### 19.3 Measured facts / rules (also in the skill)
- A zero-overhead loop whose `add.nc lc` is followed by 6 two-byte nops before the body ran its body ONCE (r14/zero2.s); 6
  full-form 16 B nop bundles work (in generated kernels the prologue provides the distance).
- A FIFO pop stream restarted after 6 pops of 72 B (432 B, not a multiple of 64) needs an explicit pointer (walker register +
  mov) and position reset; continuing the pointer with -448 or -512 corrections gave wrong data. Round 13's restarts ended on
  576 B (a line multiple).
- Scalar alu results (or r4) read one cycle later give the OLD value: the summary OR at FO + 1 missed the last iteration's flag
  (the in-situ regress caught it; the bench did not). Give scalar consumers after-edges >= 2; msched cannot check scalars.
- A walker pointer copy (mov p0, p4) must come >= 2 cycles after the walker's 2D step of the previous iteration (maxdist): the
  first pair of every row block read the old P slot without it.
- The K object's junk tail (B fifo objects are 4,608 B, K uses 3,456) is usable scratch while the core holds the object.
- Peano keeps up to 11 callee-saved registers live across calls in small wrappers (416 B for two calls in a tile loop).
- k_rb's exact test 164 cycles, rescale_rb 578 (608 B / 93 bundles at minsize + exp32 61 bundles; -O2 rolled 421 for +96 B).
### 19.4 Dead ends / not done
- attkb in 2 stages (text): no schedule up to II 42. Kernel-only attv: the negative iterations' last-stage O stores would land
  before the O tiles (g_lcd / acc_hi's neighbours). cpy32 for the Q copy / the exp table copy: larger at minsize.
- P29b (SSM log_a + exp chain in asm): the deployed passes were inspected (r14/ssm_exp_deployed.s: 53.5 bundles per vector,
  vec-slot bound 20 per vector; vups.2x / vsrs.2x / vadd.16 for 2^n); not built.

## 20. Round 15 (2026-10-10/11): 47.31 -> 41.52 ms, the SSM / norm passes in asm, the query deal (P60g), the slot-count loops
### 20.1 Changes (each regress BIT-IDENTICAL vs full12r9a; RESULTS 24)
1. **SSX** (r15/gen_ssx.py, ssx.json, EW_DEFS -DSSX; research P29b): log_a + exp + 1 - a^2 as ONE modulo loop, op for op loga_p +
   exp_pA..D (vmac x, -1 -> vmsc x, 1.0 and ones x -M0 -> vmsc ones, M0: exact). 13 bf16 constants > 12 x registers: only 1.0 lives
   in x11, the others are reloaded per use (ldb has room; II 20 = the vec bound) from ssx_tab (11 x 64 B rows, .aie.bank2, non-const),
   1.0 fp32 for the c0 accumulator by vlda.conv.fp32.bf16 from the 1.0 row. U 1, 7 stages. -2,080 B of passes for 990 B.
2. **CPY64** (r15/gen_cpy64.py): an II-1 copy (vlda + vst per cycle, U 4, full-form prologue = the loop setup distance) for the SSM
   out copies, in_copy (per-head params, with copy_run's <= 3-vector remainder and one ring-wrap split) and, with NRMK, the norms.
3. **CONVK** (r15/gen_conv.py, convk.json): the SSM conv1d, one column block (2 vectors) per iteration, II 20 = vec bound; the taps
   + bias re-laid out by ssm_tables as [cb][w0 w1 w2 w3 b] (g_tct, the same 2,560 B) so one pointer + immediates + one padda serve
   all five.
4. **P60G** (plan8.py, research rev 10 P60g, recorded as p60g in build/<name>.json and applied by step8 / run8): the query half
   tiles dealt across the columns in the attention Q gather and out_proj's A gather only (composition checked piece by piece
   before the build: every natural half tile returns to its column in natural order).
5. **RESCK** (r15/gen_resc.py, kdefs -DRESCK; research S1): rescale_rb's O / row-sum update in asm (II 8, 4 stages, a 2D jump to
   the row sums computed in the preamble); exp32 inline (-16 B: it was the only caller), srow through the score ring's slot 0
   (only k_rb's order allows it: #error without ATTKB). GEMM text 16,320 -> 16,192.
6. **RECK** (r15/gen_rec.py, reck4.json: U 4, II 11; research P29c): the RG-LRU recurrence; the h state is kept in VECTOR order
   ([cb][half] 128 B per vector) on both sides (only the SSM reads it back: the DMA moves it opaquely); one base pointer + dj
   offsets for mult / xc / gx / a, h0 by a 2D jump between the input ring's parts.
7. **div768** (ew8.cc, always on): in_need / in_done / out_begin / out_end counted ring slots with minsize subtraction loops
   (called per norm unit and per SSM chunk): -1.05 ms together. ((x >> 8) * 21846) >> 16 == x / 768 for 0 <= x < 65536.
8. **NRMK** (r15/gen_nrm.py, nrm_rms.json / nrm_ln.json; research P7-E step 1): the norm pass-2 math (rep4 = vldb.128 + 4 vshift
   by 16 B as Peano does it; RMS z = bf(bf(x r) sc); LN t = nb + x r + x rl, split, be + th ga + tl ga) with X' stored and the
   panels pushed STRAIGHT into the output ring (scratch + copy only when the part would wrap the ring end); one norm_rows for
   both modes (pass 1 sums, the statistics as a second loop over the stage in the same order). EW text 15,776 -> 14,960.
9. **ZRING** (ew8.cc): the SSM's output slots acquired before the z loop, the z panels pushed straight into the ring's Z part (never
   wraps: chunks on 1 KB boundaries of the 9 KB ring; the 448 B pad is never read), no g_panels copy: ~-0.12 ms. (An asm z loop,
   r15/gen_zk.py, ran 100 cycles against the C++ 97 on one core: not used.)
Best: **full12r15m** = r14f + P60G=1, kdefs + -DRESCK, EW_DEFS + "-DSSX -DCPY64 -DCONVK -DRECK -DNRMK -DZRING": 41.52 ms.
### 20.2 What the numbers say
- The SSM wall was code, not ring waits (EW PROF r15b: in_wait 165, out_wait 31 of ~5,640 cycles per chunk). The deployed C++
  conv loop runs 377 cycles on one core but ~1,000 in situ (P[] through the stack, in_part); generated loops run near their
  bench numbers in situ (SSX -0.79 ms = 560 cycles x 216 x 12 chunks at 1.43 us per cycle per chunk).
- Stores into the EW output ring (bank 0, read by MM2S) run ~3.5 cycles per 64 B vector in situ against 1 on the bench (both copy
  groups, whatever the source bank). Extra passes over the output ring cost more than they save (NRMK with X' / panel copies was
  +0.6 ms; with direct stores -0.40).
- The norms are compute-bound (RMS ~100 us, LN ~140 us per op whatever n_in; twin NORM_TWIN=1 without the pass-2 math 40.62 ms).
### 20.3 Measured facts / rules (also in the skill)
- vld / vst immediate offsets are 4 bits x 64 B (-512 .. 448); padda immediates -512 .. 511.
- A section-placed table (.aie.bankN) must be non-const: a const one conflicts with the LUT tables' section type.
- vmov x <- bm has its own itinerary (DM_RM_L0_PORT, latency 2): aie2p_lat.json's generic VMOV_alu_mv_mv_x has none.
- 8 bfp pushes of 72 B (576 B, a multiple of 32) + one flush write nothing past the 576 B (bench-checked): direct pushes into a
  ring unit are safe.
- A value live over the whole chain (x stored in the last stage) needs U >= its lifetime / II; a reload by a second load stream
  is cheaper in text (LN norm II 17 -> 13).
- The CP schedule must pass msched.check inside the II search: one LN schedule broke only with the .p2align pad (a port collision).
- Callee-saved p6 / p7 in a leaf asm function: paddxm [sp], #64; st p6, [sp, #-64] ... lda p6, [sp, #-64] before ret.
- IRON's jit cache key does not include source files: a kernel edit under the same build name and flags reuses the old build
  (a PROF phase that read 0). Use a new build name after every source edit.
- An -O2 C++ loop with 4 fp32 accumulator chains spills every iteration (a 32-lane MAC writes a whole dm; 5 dm registers).
### 20.4 Dead ends / not done
- NORM_SPLIT alone (statistics as a second pass over the stage): no gain (pass 1 was not the norms' bound); kept inside NRMK.
- in_copy by cpy64: -0.15..-0.2 ms for +400 B of EW text (kept).
- S2 (running row-block max in attkb): not built; bank 3 has 384 B free in this map (+384 B in the K object's tail), not 2,688.
- SSM sqrt in asm: mv-bound (~14 mv ops per vector after v = b - 16256, k7 = srs(v, 1) & 0xFF80): ~-0.29 ms, not built.

## 21. Round 16 (2026-10-10/11): 41.52 -> ~40.0 ms, the SSM's sqrt / gates / input glue and the norm pass 1 in asm
### 21.1 Changes (each regress BIT-IDENTICAL vs full12r9a; RESULTS 25; one commit each in the repo holding npu-tapnext/)
1. **SQK** (r16/gen_sqk.py, sqk.json, sqkb): exact bf16 sqrt, II 13 (the mv bound), U 2. Peano's LUT index path (band, ups by
   2, two vmov out of the accumulator, + lane bases, 8 vldb.4x16, three shuffles) with the exponent bias folded into the table:
   t' = t - ((64 - p) << 7), r = t' + ((b & 0xFF00) >> 1) (band + ups + srs + add instead of three ups / srs pairs). r9/ssm_lut.h
   takes SQK_TAB under -DSQK (r9/ssmlut.py regenerates it identically). Bit-exact on all 32,768 non-negative bf16. 493 -> 321.
2. **INPART_INLINE** (research P-IN): the 7 part pointers inline, one straddle test per chunk, the straddling part by in_copy /
   cpy64. The first try used the flag PIN, which some header #undef-s: the build silently compiled the old path (same elf md5).
3. **XONE** (research X1b): x / x-1 / x-2 by one 48-vector out_copy when consecutive in the ring. cp_x 432 -> 269.
4. **GGK** (r16/gen_ggk.py, ggk2.json, ggkb, ggkref): both gate GEMMs as one loop over the 8 (gate, np) pairs, II 24 (vec 20),
   A (xc's panels, now a bank-2 static g_xcpb) restarted per np, B (Wx | Wa) one pop stream WITH A FILL BEFORE EVERY 8TH POP
   (21.3), the epilogue as vlda.conv bias + vmac bf16(acc) 1 (the same rounded sum as mul + mac). gates 655 -> 365.
5. **SSQ** (r16/gen_ssq.py, ssq.json, ssqb): ssx + sqk in one loop: no schedule up to II 26 (x pressure), II 30 / U 2 / 6
   stages; 15-row constant table (the lane-base row written by the preamble). 832 -> 715.
6. **SIG10** (r12/gen_sig.py --umax, r16/sig10.json): sigk at II 10 / U 4 (round 12 had capped U at 2 -> II 12). 499 -> 431 (bench).
7. **N1K** (r16/gen_n1k.py, n1k_ln.json / n1k_rms.json, n1kb; research N1): norm pass 1 + statistics per unit; the statistics
   are loop-carried 64-lane MACs on register pairs (y4 = x8:x9, y5 = 1.0 = x10:x11; a generator-level "y" class); _begin / _step
   / _end entries keep constants and accumulators in registers across a row block's 12 steps. LN II 11, RMS II 10.
Tools: r16/nrun.sh, b16.sh, try16.sh (research tools/try.sh on the current best), commit.sh, popprobe.s + popb.py (FIFO probes),
NPROF (norm-step stamps without the SSM phase stamps), timing twins NORM_TWIN=2 / 4. Research's build-lock patch applied.
msched / cpsched: loop-carried uses (Op cuses, U 1, initial value from the preamble; opt-in, every deployed generator replays
byte-identical); gen_sig --umax.
### 21.2 What the numbers say
- EW PROF (cycles per SSM chunk) r15m -> r16eP: conv 638 -> 440, gates 651 -> 365, ssx + sqrt 1,001 -> 745, cp_x 432 -> 269,
  SSM op 532 -> 426 us (sigk 509 -> ~440 with SIG10, not stamped).
- The norms: twin NORM_TWIN=2 (pass 1 + statistics skipped) 40.18 -> 38.20 ms, but N1K gained only ~0.2: in situ the n1k step
  costs 161.5 cycles per unit (= its bench) and in_need waits 104.8 per unit: the norms are partly input-bound now, so removing
  compute in a twin overstates what faster compute buys.
- One-core benches matched in situ for every generated loop this round (sqk 311 / 321, ggk 310 / 365 incl. the push loop,
  ssq 701 / 715, n1k 139 / 141 without stamps).
### 21.3 Measured facts / rules (also in the skill)
- **A dense 576-bit FIFO pop stream underflows at its 9th pop** (r16/popprobe.s probes 6 / 7): a pop brings one 64 B line and
  consumes 72 B, so without an extra `vldb.fill.512 [p, lf, r]` at the same position (no pointer / position reset) after every 8
  pops, pops 8-12 read zeros / half exponents, then the stream resyncs. Peano's gate_gemm inserts that fill; restarts every <= 8
  pops (attv, attkb, the ggk A stream) never hit it. The first ggk lost the kk 4-6 terms in every np: four schedules (slack,
  pop exclusion, no bias, no epilogue MACs) failed identically; taps of ex1 / ex4 at the MAC's cycle and the probes found it.
- A 2D modifier is PERIODIC (dn steps of dj, then m, then again): for one ring-end jump inside k pieces, set dn = k - 1 (here 7)
  and start the counter dc at k - n1, not dn = n1 - 1 (which jumps every n1 pieces).
- `#ifdef PIN` was false with -DPIN on the command line: some included header #undef-s PIN. After a new flag, check that the elf
  changed (md5 / llvm-nm sizes) before timing it.
- 64-lane bf16 MACs (`vmac.f dm, dm, yA, yB, r` with y_n = x_2n : x_2n+1) halve the dm registers of per-lane fp32 accumulations
  (bit-identical per lane); a 32-lane MAC writes a whole dm.
- vconv to x_n landing in the same cycle as a pop's lda result is a QEY_WA port collision (modeled; probe 1 reproduces it).
### 21.4 Dead ends / not done
- sqrt fused into ssx at U 1 / 2 below II 27 (x pressure); II 30 kept (SSQ).
- The first ggk debugging hypotheses (pop register windows, 2D bias modifier, A / B co-issue, epilogue MACs): all refuted by
  probes; a pop-window allocation guard was added to msched and reverted (never isolated as a cause).
- Not built: X1 full rotation (cp_x 269 -> 0: the sliding SSM area base needs per-frame runtime sequences), S2, FSX, P52 by DMA.

## 22. Round 17 (2026-10-10/11): 39.94 -> 39.16 ms, S2, X1 by rotation (mod 4), AL1, NPIPE
### 22.1 Changes (each regress BIT-IDENTICAL vs full12r9a; RESULTS 26; commits 16998cd, 26ccb0a, d3c671d in strix-halo-maxx)
1. **S2** (mm8 -DATTKS2MAX, r14/gen_attkb.py --s2, schedule r17/attks.json): every attkb iteration loads its row block's max slot
   ([p6, #0] / [p6, #64], the caller's stack, bank 0), vmax_lt(slot, reloaded score) in the C++ test's operand order, stores it back
   (vst #64 then vst.2d d7: 0 x5, then +128); the stores in the last stage. The per-iteration flag stores are gone: attkb returns the
   flags' OR (int32_t: trips / hi move to r1 / r2) and a flagged row block's own flag is recomputed from its slot (bf16(max s - m) >
   63.75 in some lane: monotone, exactly the loop's per-score test). k_rb takes the slot and skips the first-pass rerun. II 36
   (34: none in 240 s; 35: failed the pad replay twice); x pressure needed cpsched `reserve` 1 and the max on the ring-reloaded scores.
2. **X1** (plan8 XROT=4, ew8 -DXROT; research rev 12 design): conv history R0..R3 at B + S .. B + 4 S, x(t) -> R[t mod 4] by splitting
   the yx drain ([2, 12, RB, XB], strides [x_at - base, KSTEP16, UNIT, 1]: under P23 the C objects arrive in k-step order), the kernel
   maps tap x - k to R[(r - k) mod 4] (header word 2 = 1 + r, four params copies in W), output [h lo, h hi, z] (3 KB) at B + 5 S. 4
   runtime sequences per depth (build/full<d>NAME_p<k>), step8 rotates the instruction bo per dispatch (Step8.select / phase), reset()
   starts at phase 0. Built first as a sliding area (N phases, a wrap frame with copies, (N + 7) S per column): same rotating-frame
   gain, replaced (no wrap frame, no memory growth).
3. **AL1** (plan8 AL1=1, ew8 -DAL1): params fills padded to 3 KB multiples (SSM 12,288 B per head, RMS 3,072, LN 6,144); the
   straddle path compiled out.
4. **NPIPE** (ew8 -DNPIPE, NPIPE_G=6): norm pass 2 of rb interleaved with pass 1 of rb + 1 in groups of G units; the n1k accumulators
   go through g_nacc around each group (_end / _begin).
Tools: r17/try17.sh (r16z base), r17/xtry.sh (XROT phase builds + regress + time + sweep), r17/xrotsim.py (state flow check),
r17/cp_attks.sh (CP on testbox), step8 SPREAD / PHASES lines, NPROF out_begin stamp (record word 19), CPROF (conv split), attkb.py
--hashes, cpsched `forbid` + gen_attkb re-solve on pad port collisions.
### 22.2 What the numbers say
- S2 hit (-0.31). X1 hit (-0.33 per rotating frame; the mod-4 form has no wrap frame). AL1 and NPIPE missed: AL1 -0.05..-0.15
  (-0.21 predicted), NPIPE ~-0.1 at G 4-6 and nothing at G 2 (-0.55 predicted).
- NPROF: in_need 101.5 -> 61 per unit with NPIPE, out_begin 45 -> 47: the norms are not output-bound (ewio predicted out_begin
  147 / 90); the extra _end / _begin per group (~45 cycles) eats much of what the overlap gives at G 2.
- The SSM conv gap is closed: convk runs 250-280 in situ; the 440 was the straddle copy (~47 % of chunks, AL1) + pointers + stamps.
### 22.3 Measured facts / rules (also in the skill)
- **An EW input fill group must total a multiple of the 768 B ring slot**: the core's packet BD completes per slot; 1 KB-padded params
  (11,264 B per SSM head) left the last slot unfilled -> the step hung (TIMEOUT, recovered; known-good bench fine after).
- A modulo loop can carry state through memory: load the previous iteration's store by offset, store with post-modify, maxdist
  (load, last store) <= II - 4 (data land +4, pointer +3), stores in the last stage (drain garbage iterations must not write).
- cpsched solutions can fail msched.check only with the .p2align pad: re-solve with the colliding op's time forbidden (gen_attkb does
  it for port collisions); allocation failing at maxlive = budget: `reserve` 1 register in the CP bound.
- Under the foreign CPU worker dispatch times split into two clusters ~0.17 ms apart; one 60-iteration median can land in either.
  Compare p10 / median / mean of 200 interleaved iterations.
- A GEMM drain can split its output into two regions with one TAP when the C objects arrive in a linear order: [2, half, RB, XB].
### 22.4 Dead ends / not done
- NPIPE_G 2 (+0.07), 1 KB AL1 (hang). S3 / V1 / K1 / K2: GEMM text 48 B free after S2 (not attempted). FSX, P52 by DMA, EWO bench.

## 23. Round 18 (2026-10-11): 39.16 -> 35.20 ms (sweep d12), ROT, P35, KSC, the SSM and norm glue
### 23.1 Changes (each regress BIT-IDENTICAL vs full12r9a; RESULTS 27; commits 52ffad1 .. 1ffff40 in strix-halo-maxx)
1. **HRING + PTRG** (ew8): reck writes h into the output ring (lacq before reck; the 2 KB cross the 9 KB ring end in one chunk of
   three: reck's second h store is `vst.2d` on d5 whose counter the caller starts at 16 - n1; r15/gen_rec.py --ring, msched check
   dc0); the chunk's part pointers from g_in_b by offset, taps from the rotation (no stack array / remap loop).
2. **ZK**: r15/gen_zk.py's z loop (round 15 dropped it against a one-core C++ bench that pipelined; in situ the C++ was serial).
3. **KSC**: mm_ks records as compile-time data from -DKS_* addresses; design8 ksc_check() verifies acc_hi (llvm-nm) and the A slot /
   C / B buffers + B locks (lowered MLIR) on all 24 cores after the link and fails the build otherwise (acc_hi moves with .data size).
4. **NIOC**: the NPIPE norm loop keeps the ring state in locals and inlines the four ring helpers.
5. **V2T**: attv for both tiles per call (O jump by d2 / d3 every 24 iterations), attvs once per chunk.
6. **CGF**: convk pushes xc's bfp panels (x1 on p6, push on p2 = g_xcpb_k).
7. **P35**: mm_ktile runs a tile's k-steps; B cons / prod locks by id, B buffers by address; acc_epilogue records the bias
   buffer so the next tile starts at the other one (each tile consumes an even count: IRON's own index stays right).
8. **ROT** (research pass 14 E5): mm_rot (rotating accumulator, 1,383 static vs mm_ks 1,824); pack8 B_ROT panel order; g2a =
   research's g2f re-solved with acq.cond 6 bundles before the A fill (r18/gen_rot.py ROT_ACQ_GAP=6 ROT_REL_GAP=8).
Tools: r18/ab.py (one-process A/B), r18/try18.sh (current best's flags), r18/dmab.*, r18/ksp38.py (+ mm8 KSP38), ew8 NIO2 +
r8/ewprof NIO2 lines, r18/fsize.sh, r18/rotbench.py / rotlocks.py.
### 23.2 What the numbers say
- The GEMM k-step dominated: ROT -3.3 ms alone. In situ mm_rot runs 1,603 cycles per k-step with 168 of memory stalls: the DMA
  collisions (DMAB) remain the next GEMM lever (B into bank 3 where it pops, A slots 1 / 2 in bank 2 with the C object).
- P35 and NIOC missed their static prices: the k-step gap and the norm pass 1 are mostly waits, not code.
### 23.3 Facts / rules (also in the skill)
- `add rX, rY, #imm` is 7-bit signed: #64 assembled silently as #-64 (V2T hung once).
- A lock acquire needs distance before the dependent load: acq.cond ONE bundle before the A fill raced in situ (bench with
  pre-released locks passed); 6 bundles (as mm_ks) is bit-identical.
- DMA writes into the bank the core is using cost the core ~1 cycle per 32 B.
- Core performance counters: Performance_Control0/1 (0x31500 / 0x31504) with start = stop = event counts occurrences; the core reads
  0x31520 + 4n by read_tm after the bus enable (TRUE = cycles validates). Lock values at 0x1F000 + 16 id read 0 from the core.
- One-core benches share headers through build/: mm8_rot.inc includes build/mm_rot.h (the bench picked another variant's border).
### 23.4 Dead ends / not done
- K1C (two row blocks per prep trip: -28 per tile-chunk for +96 B), NPIPE_G 4 / 12, N-IO in asm, S3 / K1 / K2, ssq two chunks per
  call, FSX, the B / A bank layout (needs a hand-built B ring), ROT g0f / g1f (probes PWRAP / PADDA).
