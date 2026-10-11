---
name: xdna2-npu
description: Build, run, verify and time hand-written kernels on testbox's XDNA2 NPU (Strix Halo, AIE2P) with IRON/mlir-aie + Peano, incl. the asm route, the bfp16 codec, safety rules and the moondream/TAPNext precedents.
---

# XDNA2 NPU on testbox — how to work

The NPU is the 32-core AIE2P array (8 columns x 4 rows, memtile per column) in testbox's Strix Halo. Everything
below was learned over the moondream phases 1-12 and the TAPNext++ port (2026-10-07..09). Progress state lives in
the workspaces' NOTES.md / RESULTS.md / HANDOFF.md, not here.

## 1. Environment
- **Repo (2026-10-10):** the NPU work lives in the git repo `~/git/app/strix-halo-maxx/` on testbox; `.scratch/npu-{tapnext,research,moondream}`
  are absolute symlinks into it. Every `docker run` that mounts `-v $HOME/git/app/.scratch:/work` must ALSO mount
  `-v $HOME/git/app/strix-halo-maxx:$HOME/git/app/strix-halo-maxx`, or `/work/npu-*` is a dangling link inside the container.
  Commit locally (one commit per kept change or failed experiment, measured ms + predicted vs measured); never push.
- **All compute on testbox (user, round 17)**: generators, CP-SAT (`nice -n 10`, <= 12 workers), sims and tools run there; edit on
  testbox (source of truth); the laptop only for ssh / reading and the end-of-round mirror rsync.

- `ssh testbox`. Repo `~/git/app`; scratch work under `~/git/app/.scratch/<project>/` (never in /tmp, never in the
  app package). The laptop is only where code is written: edit locally, `scp` to testbox, run there.
- **Docker only for NPU work.** Images on testbox:
  - `npu-iron`: Ubuntu 25.10, XRT 2.21 (pyxrt at `/opt/xilinx/xrt/python`), mlir-aie 1.4.4 IRON in
    `/opt/ironenv` (python 3.13: `/opt/ironenv/bin/python`), Peano/llvm-aie at
    `/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin`, numpy, ml_dtypes, `aie.utils.bfp`.
  - `npu-iron-torch`: the same + torch 2.14 CPU + safetensors (for prototypes that run a torch model on the CPU
    around NPU kernels). Make new images with `docker commit` after a `pip install` inside a container.
- Run pattern (the only way to reach the device):
  `docker run --rm --name <yours> --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v <host dir>:/work <image> bash -c "..."`
  Builds (CPU only) drop `--device`. Mount the parent of your workspace as `/work` so the design's include path
  `/work/npu-moondream/mlir-aie/aie_kernels` (the `aie_kernels` headers checkout lives in the moondream workspace)
  resolves; adjust `INC` in the design if you mount differently.
- Python inside the container is 3.13: a module written for 3.14's lazy annotations (app's `model.py`) needs
  `-> "Cls"` quoting or `exec` with a text patch.
- Host driver amdxdna 2.21 (Jan 2026) works with mlir-aie 1.4.4. **Core clock 1.81 GHz in Turbo** (measured round 7,
  npu-tapnext/r7/clk.py: pure-register loops, host-timed slope and the core timer agree; the driver's top DPM level is
  1.8 GHz; the old "1.43" was a stalled spin loop). `xrt-smi examine -r all` (with --ulimit memlock) shows the mode;
  the user sets it with sudo - say which mode every number was taken in.

## 2. Building a program (IRON route)

- A design is a python file with `@iron.jit(aiecc_flags=["--dynamic-objFifos"])` function taking `In`/`Out`
  buffers and `CompileTime[...]` parameters; `Worker`s run C++ kernels compiled by `ExternalFunction(name,
  object_file_name=..., source_file=..., arg_types=[...], compile_flags=[...])`; `.object_file.bind(name, args)`
  exposes more entry points of the same object. `Kernel(name, "<prebuilt .o>", arg_types, stack_size_override=)`
  links a hand-built object instead (Worker `stack_size` must exceed the override plus aiecc's libcall estimate).
- Build: `design.specialize(**params).compile(xclbin_path=..., inst_path=...)` (see the `b.sh` scripts: they run
  `python3 /work/<ws>/<design>.py --out /work/<ws>/build/<name> <args>` and log to `<name>.log`, `rc=` on the last
  line). Lowered MLIR for inspection: `build/<name>.prj/input_with_addresses.mlir` (fifo depths, lock inits, BDs,
  runtime-sequence DMA lengths).
- Data movement vocabulary: `ObjectFifo(type, depth)`, `.prod(tile=shim)`, `.cons()` (each call = a consumer
  endpoint: several = broadcast), `.cons().forward(tile=mem, obj_type=, depth=)` (memtile hop, one MM2S channel,
  multicast), `.cons().split(offsets, tile=mem, obj_types=, names=, depths=)`, `.prod().join(...)`,
  `TensorAccessPattern(shape, offset, sizes[4], strides[4])` for shim DMAs (`fill`/`drain` in the runtime
  sequence; `TaskGroup` + `wait=False/True`). Explicit `Buffer`/`Lock`/`Bd`/`TileDma` chains for static memtile
  rings (design6).
- Budgets to check BEFORE building: memtile 512 KB, 6 MM2S + 6 S2MM channels, 48 BDs in two halves of 24 per
  channel group (a static chain gets ~22); core L1 64 KB (all fifo objects x depth + buffers + statics + stack);
  core program memory 16 KB (.text overflow = too many unrolled ops/calls: fewer ops, `BLKU=1`, `-Os`); locks are
  6-bit (values <= 63); shim BD dims 10-bit (fold long transfers: `tap1d`); objects >= 64 B aligned; a core has
  2 S2MM + 1 MM2S ports (~4 B/cycle each) and reads only its own / N / S / W neighbour L1.
- Packet-switched routes (IRON 1.4.4): `PacketFlow(pkt_id, src, dst, src_channel=, dst_channel=)` + `.fill(buf,
  tap=)`; several PacketFlows may share one shim MM2S channel (e.g. the GEMM A loads into a memtile chain and an
  elementwise core's private ring), which is how a column gets more inbound streams than the shim's 2 channels.
  `ObjectFifo(packet=True).prod(channel=)` pinned to a channel a PacketFlow uses is REFUSED - share only between
  PacketFlows with hand-built rings. `Bd(iteration=BdIteration(size, stride))` makes one BD walk a ring of slots
  (works on core DMAs too; stride in elements of the buffer's dtype), which keeps rings under the 16 BDs of a core.
  Hand-built rings driven from C++: `Lock(tile, init, lock_id=n)` pins the id and the kernel uses
  `acquire_greater_equal(buf, id + 48, n)` / `release(buf, id + 48, n)` (Peano's aie2p_locks.h; +48 = this tile;
  pointer forms tie the lock to the buffer). A shim holds at most 16 simultaneously active runtime BDs: `finish()`
  task groups (design6's lookahead=2). `Tile` endpoints match by object identity: make each Tile once and reuse it.
  **Shim BD steps are 20-bit (words)**: a TAP step >= 2^20 words (4 MB) still moves the right bytes, but the
  transfer then runs column after column (measured: a 2-input norm fill with a 10.5 MB step took 8 x 0.75 ms
  instead of 0.8 ms). Keep regions that one TAP strides across within 4 MB (assert it in the planner).
  A fill may read an `InOut` buffer that an earlier drain of the same sequence wrote (dependent ops in one program).
- **Memtile DMAs driven from the runtime sequence (round 10, npu-tapnext r10/rtm.py, design8 A_RT)**: IRON 1.4.4's
  `FlowEndpoint.task(Bd(...), runs=n)` configures a mem/core-tile task's BDs inside the sequence; `task.start(repeat_count=)`
  pushes it on the channel queue (4 deep; a full queue blocks the command processor), `free()` at the end. Configure each
  pattern ONCE and start it per op: no static-chain BD budget, per-op repeat counts, and chunked loads (`BdIteration(size,
  stride)` + `runs=size`: one chunk per run with its own lock acquire / release). A BD takes one acquire AND one release, or
  neither (a lone acquire is refused). Push an op's tasks before any wait of that op: a blocked push may only wait on
  work already issued. Locks used only by tasks need `rt.add_lock`.
- **Packet router bug (IRON 1.4.4 pathfinder)**: a multicast packet flow to two DMA channels of one tile is routed
  correctly alone, but when unicast flows also target those channels it silently drops a destination (no rule; the
  dispatch hangs). Check `masterset`/`rule` lines in input_with_addresses.mlir for every multicast (npu-tapnext
  r10/pf1.mlir reproduces it with `aie-opt --aie-create-pathfinder-flows`). Tile / memtile BDs refuse stride 0.
- Runtime tasks (IRON 1.4.4, design8): `TaskGroup.finish()` awaits the group's `wait=True` transfers and then FREES
  every transfer's BD, waited or not: a group that holds only `wait=False` fills must not be finished before those
  fills completed (the BD gets re-programmed mid-flight: silent corruption or a hang). Put fills in the group whose
  drains complete after them, or mark them `wait=True`. Consumers of a device-written region must finish the
  producer's group first (`deps` in plan8): the lookahead pipeline alone is a race that only passes by timing.
  Shim DMA TAPs: 4 dims, every size <= 1023 (words), so a 4608 B object (1152 words) needs a fold dim: layouts
  where objects of one transfer are strided need per-object or per-tile tasks (the qkv drain = 12 per column;
  the attention K/V fill = one per head). The shim BD pool is 16; `aiecc --reclaim-runtime-bds` lifts the static
  check (used for the 206-op step; semantics unverified). Schedules are compile-time: a stale .sched after a plan
  change (e.g. parameter slot counts) hangs the EW core silently (assert them in the design).
- **Worker loops get unrolled; use IRON's endless worker loop instead** (round 5): LLVM fully unrolled an EW worker's
  `range_(12)` block loop into 60 calls (2 KB of the 16 KB program memory). IRON already wraps every worker body in an
  endless loop: make the body ONE op (or one block of ops) and let data drive it. An EW core that reads its op kind
  from the op's params header (design8 `ew_op`) has control code independent of the op list; GEMM cores that run one
  block per iteration are independent of the depth. Then the elfs and CDO are byte-identical across depths, one
  xclbin serves every depth and only the runtime sequence (.bin) differs: switch depth at run time by swapping the
  instruction bo (no new hw context). Lay out memory for the maximum depth so W/ACT offsets match across .bins.
- IRON semantics that bit us: `acquire(n)` HOLDS n objects in total (`acquire(1)` twice vs `release(2)` drifts
  locks -> deadlock); a forwarded (broadcast) L1 fifo lowers to at most 3 free slots whatever depth you ask, so a
  kernel that acquires pairs gets no prefetch (acquire one object per call); the runtime sequence's lookahead and
  memtile object sizes must match the host packing exactly (check the lowered `dma_bd` lengths against your
  buffers before dispatching); `aiecc` with `--dynamic-objFifos` is required for the loop forms used here.

## 3. Kernels (C++ with the AIE API) — rules

- **No native fp32 vector arithmetic**: 8/16/32-lane fp32 add/mul/min/max/compare/convert are emulated (~21
  bundles per 16 lanes). Do elementwise math in bf16 (32 lanes native) and integer work in int16 (one register);
  fp32 only inside the MAC accumulator (`accfloat`) and for reductions. "Fake fp32" = hi+lo bf16 pairs multiplied
  in the MAC unit (phase-8 `expsplit.h`: `split_acc`, `fma_split`, `exp_acc`, `recip_b`) ~2^-16.
- **Rounding mode defaults to FLOOR**: `aie::set_rounding(aie::rounding_mode::conv_even)` once per core
  (`crrnd`); without it every bf16/bfp conversion truncates (the whole phase-8 attention ran that way).
- MMUL: `aie::mmul<8,8,8,bfp16ebs8>` = 512 MACs/instr (dense int8 is the same rate); operands are
  `block_vector<bfp16ebs8,64>` popped from `block_vector_restrict_input_buffer_stream` (seek/pop; a stream must
  start 64 B-aligned, so pad panel groups: 9 panels = 648 -> 704 B, 6 panels = 432 -> 448 B, 8 panels = 576 B
  is aligned by itself); B is given as `aie::op_transpose(B)` with B stored `[n][k]`; results via
  `acc.to_vector<bfloat16>()` or pushed as bfp with `block_vector_output_buffer_stream::push(acc.to_vector<bfp16ebs8>())`
  (the on-chip fp32->bfp conversion follows the rounding mode). The AIE-API bf16 mmul converts to bfp internally.
- The core has two FIFO load units: at most two block-vector streams live at a time (a third spills its state).
- **Register timing (round 11, measured, npu-tapnext r11/mk_haz.py)**: a write with .td latency w is seen by an operand read
  at .td stage r of an op issued **>= w - r + 1** cycles later (LLVM's latency; vlda / vldb / vldb.4x64 7, vadd.32 / vconv /
  vfloor / vshuffle / vmin_ge 2); until then readers see the OLD value (no interlocks). The older "c + r >= c' + w" model (and
  p10/asmcheck.py) is one cycle optimistic; its "vmac reads its accumulator at stage 3" is the .td's 4 under this rule
  (dependent macs >= 3 apart, or back-to-back at distance 2 through the one-deep accumulator bypass, round 7; a loaded
  accumulator quarter must land a cycle before the mac reads it). **mv -> mv bypass**: a value from an mv-slot op (vmov, vshuffle, vadd.32, vmin/vmax) reaches
  mv-slot readers one cycle EARLY: never let an mv reader rely on the old value in the cycle before an mv write lands. An older
  load and a younger short write to one register land in landing order (nothing is dropped). Pops land at +8 (gen_ks).
  `vsrs` +4, `vups` +3; pointer moves (`movs`) need 2-3 cycles before a load uses them.
- **Itinerary ports are structural hazards (round 11)**: aie2p_lat.json `itin` stages (SimpleCycle = one stage, PrefixCycle =
  same stage as the next, EmptyCycles<n>) name shared ports: DM_ADA (vldb.4x64 LUT loads AND vlda), QEY_RS (LUT index reads,
  vfloor, vst x), QEY_WA (vlda / vconv / vfloor results), QEY_WB (vldb / LUT results), CRRND_RS (vconv, vst.conv), L_RM / DM_WM
  (vshuffle into bm). A plain vldb uses only QEY_WB. Reserve them modulo II like slots (r11/msched.py does).
- Elementary unit: `aie::inv` (bf16) is good for 1/sum; `aie::exp2` is a ~4-bit approximation (8% rms) - do
  not use it for softmax/GELU. Exact exp: z = x log2e via bf16 hi/lo MACs, n = round(z) by adding 1.5*2^23 in
  the MAC unit (fp32 ulp 1; low 16 bits of the result = n), f = z - n, 2^f cubic on f, f^2, f^3, 2^n assembled
  as int16 `(n << 7) + 0x3F80` (phase-9 v9/v15 `exp32`). 8-lane bf16 `aie::max` returns wrong values: use 16/32
  lanes. Peano crashes on vectors assigned under runtime conditions and on `aie::select` ("Bad PHI"): mask
  arithmetically. `aie::add(acc, acc)` and `aie::filter_even` are native.
- Peano schedules long elementwise chains at ~1 op/bundle (register-allocation-bound; more chains = spills) and
  its mmul loops are near the chain bound already: measure the slot/latency bound against a hardware trace
  BEFORE hand-scheduling anything.
- **Block-vector output streams and function calls (design8, 2026-10-09)**: never call a non-inlined function while
  an `OutS` stream is live (between its first push and the last): compute the vectors first, then push. A stream's
  destination must be truly 64 B-aligned (a stack array with `alignas(64)` is not): a 288 B stack stream came out
  shifted by 32 B. Static pointers set in one kernel entry and used in another entry called earlier are NULL on the
  first call (attn_qcopy wrote 10 KB through a NULL pointer into the stack: a silent hang); initialise lazily.
- Program memory is 16 KB per core (`program (RX) : LENGTH = 0x4000` in the .ld script). The elf text = the
  kernels + IRON's per-op control code: 12 unrolled GEMM ops cost ~12 KB, so drive repeated ops from an L1 table
  (`Buffer(initial_value=np.int32[...])`, `range_(to_index(tbl[i]))` with `arith.IndexCastOp`) and keep ops looped.
  `-Os`/`-Oz` cost ~5% on the GEMM loop; `__attribute__((noinline, minsize))` on exp/gelu/attention helpers buys
  ~4 KB; entry points the worker never calls ARE dropped by the linker (a t3 build's EW elf held only ew_ssm).
  `Worker(data_size=)` reserves the kernel's statics when IRON's placement leaves too little.
  Measure a core's real stack need with `-fstack-size-section` + `llvm-readelf --stack-sizes` (the GEMM cores need
  ~2.5 KB of their former 6 KB; round 6 gave the freed 2 KB to LUT tables via data_size).
- **-O2, not -Oz/-Os, by default**: design8's EW object at -Oz was LARGER than at -O2 and 2.4x slower (non-inlined
  AIE-API helpers such as `concat` become calls that spill every live vector). Use `minsize` on cold functions.
  Measure size with `llvm-size`/`llvm-nm --size-sort -S` on a standalone `clang++ -c` (npu-tapnext/build/asm/cc.sh).
- **`aie::inv` is scalar** (32 lane extracts + scalar inverses per bf16 vector): for 1/x in vector code use an int16
  bit-trick seed (`0x7EF3 - bits(bf16 x)`, |err| < 5.3%) + two Newton corrections in hi/lo bf16 MACs (nk8 recip1p_i).
- **Reductions and exact constants without fp32 vector math** (round 5, ew8.cc): sum the 8 lanes of each row of a
  [4 rows][8] fp32 accumulator with a butterfly, `auto [e, o] = aie::interleave_unzip(v, v, st); a = aie::add(a,
  A32(aie::interleave_zip(o, e, st).first))` for st = 4, 2, 1 (native, ~40 bundles; every lane ends up with its row's
  sum, i.e. already broadcast). `reduce_add` on fp32 vectors is emulated. For fp32-accurate products in the MAC unit
  split an accumulator into three bf16 parts (`h = bf(a); m = bf(a - h); l = bf(a - h - m)`, exact) and the constant
  likewise; six MACs give ~2^-20. var = (n q - s^2) / n^2 with exact bf16 products keeps the fp32 code's accuracy.
- **Per-call overhead dominates small-grain I/O**: a noinline "append 64 B to the ring" helper (lock bookkeeping per
  call) made design8's norms 3x slower than their DMA bound. Reserve the output slots per larger unit and store
  vectors straight into the ring; keep mode / count branches out of inner loops (one loop body per mode via a
  template; missing input streams can re-read stream 0 with weight 0).
- **Online softmax with fp32 O and bfp16 P: update the row max lazily** (only when a row's max passed the stored one
  by TAU = 8; P <= e^8): eager rescaling on every increase cost as much as the exp. Take exp(s - m)'s argument in
  fp32 straight from s and m (rounding s - m to bf16 first costs up to 2-3% at |s - m| ~ 10).
- **Peano does not software-pipeline long dependency chains** (exp, sigmoid, GELU: every MAC-latency slot is a nop).
  Two independent chains in one basic block interleave: loop bundles per vector exp 73 -> 38, sigmoid 140 -> 77,
  gelu 199 -> 111 (nk8 `*_n`, `exp32x2`, `gelu32x2`); four chains spill and gain nothing. Count bundles with
  npu-tapnext/build/asm/loopcount.py before and after.
- **Attention exp by table (round 7)**: 80 segments of 1/4 on [-11, 9) (zero below -10), line in the s domain
  (off - slope m) + slope s with bf16 s, m (no hi/lo split), P to bfp16 straight from fp32: as accurate as the
  arithmetic exp on real scores (r7/expemu.py), 49 bundles per 64 incl. the push. Each 16-lane lookup is 4 vldb on
  the one ldb slot (16 per 64 lanes = the floor); a second table copy can live in a scratch area the kernel rewrites
  per op (attn_begin copies it into the C object's tail).
- **Table lookups (round 6): AIE2P has a 4-way parallel LUT load** (`load_lut_2x_float(ab, cd, idx, c0, c1)`, 16
  lanes; the aie::lut<4, float, bfloat16> layout: (slope, offset) float pairs, groups of 4 entries stored twice in a
  row, a second copy of the table in ANOTHER bank). Write the lookup by hand as mlir-aie's
  `aie_kernels/common/activations.h lut_segments_acc` does: the index is a BYTE offset, 16 B per segment, so for
  segments of 2^-k: `bfloat16_to_int(x, k + 4) + 16 * bias`; then offset (fp32) + slope (bf16) * x in one mac.
  `aie::linear_approx` keeps its object in memory and reloads it every compute (2.7x slower). Place the copies with
  `__attribute__((section(".aie.bankN")))` (the linker script maps .aie.bank0-3 onto each bank's FREE space after
  IRON's buffers: check `ldScripts_*.ld.script`) or in the data area (raise Worker `data_size`). A 64-segment
  piecewise-linear s(x) with gelu = x s(x) (s hi/lo into the product) is 3.9x faster than the arithmetic GELU at the
  same real-data rms error (`npu-tapnext/mm8.cc gelu_lut_i`, `r6/lutgen.py`). Linear tables have ABSOLUTE error:
  for anything whose small values need relative accuracy (sigmoid feeding a = exp(-8 sp sigma)), check the tail.
- Hardware `aie::tanh` (vtanh) is crude (identity for |x| <= 0.5, 2-7% off up to |x| = 2) and `aie::exp2` has p99
  10 ulp; `aie::inv` is correctly rounded but scalar. Characterise any primitive on every bf16 input first
  (`npu-tapnext/r6/kprobe.py`).
- **Core timer (round 7, npu-research P15)**: `read_tm(0x340F8)` hangs the core UNLESS its processor bus is enabled
  first: in the IRON runtime sequence function, `from aie.dialects.aiex import npu_maskwrite32;
  npu_maskwrite32(address=0x32038, value=1, mask=1, column=c, row=r)` per core before the fills that release it, and
  each read fenced with `__builtin_aie2p_sched_barrier()` (check `lda.tm` sits alone in its bundle). Then cycle counts
  are exact (a known loop reads bundles x trips + ~50). npu-tapnext `mm8.cc -DPROF` + `r7/prof.py` = per-tile
  in-situ cycle profile of the GEMM cores (records written over each tile's first words; timing builds only).
- **`minsize` on hot code costs up to 10x**: the round-5 epilogue (minsize, a helper call per 64-element block) ran
  ~77 bundles per block; a noinline -O2 pass with the row block unrolled runs 5.5. Keep per-element work in loops
  inside one function (array passes), out of per-block calls; constant trip counts pipeline, variable ones often don't.
- **Block-vector pops can issue on both load slots** (`vlda.pop.576 .. [p0, lf0, r24]`, `vldb.pop.576 .. [p1, lf1,
  r25]`; Peano picks lda for an `aie_dm_resource::a` stream when two streams are live). The C++ GEMM loop (28 bundles
  per 16 mmuls) is bound by its 2 accumulator chains at distance 3, not by ldb. C++ 4-chain variants are SLOWER under
  Peano (A held in 8 of 12 ex registers, no pipelining across column blocks): the asm route
  (npu-tapnext/r7/gen_mm.py: 4 chains over 2 column blocks, B repacked [q][4 col blocks] on the host, 225 vs 321
  bundles per call, bit-identical). Popping the "held" operand instead of holding it helped the attention QK^T
  (70 -> 53 bundles) but hurt PV (42 -> 67): measure both.
- **L1 banks (round 8, npu-tapnext r8/mmb.py): a core's 64 KB are 4 banks of 16 KB (0x70000 + 16 KB i); two
  accesses to one bank in ONE bundle stall the core** (ops of the same slot never co-issue, so lda pops + lda loads
  are free). Peano's loops rarely co-issue 3 memory ops; hand asm does: the asm GEMM call ran ~290 cycles in situ for
  225 static (accumulator load + store in one bank +33, A + B one bank +14, B with C +42). Pin buffers with IRON
  `Buffer(..., mem_bank=n)` (1.4.4 has it; ObjectFifo buffers then move around them) and check
  `input_with_addresses.mlir` / `llvm-nm -n` for every asm kernel; measure with a one-core bench that calls the
  kernel between timer reads with operands at chosen offsets. Scheduling rules that forbid co-issue cost more
  bundles than the stalls when the accesses cluster (16 loads + 16 stores in a ~16-cycle group boundary).
- **Never re-seek a block-vector input stream after popping from it** (round 7: t4 output inf); construct a fresh
  stream object per block (a `#pragma unroll` loop over two fresh streams compiles to the same code).
- Peano program memory: `minsize` is cheap in size but a branch loop costs ~12-17 bundles per trip (hand-unroll 4
  stores per trip); `-O2` grew attn_begin 800 -> 1,104 B; never blanket-remove `#pragma clang loop unroll(disable)`
  (two of them kept the epilogue inside 16 KB). Check `llvm-size` of the linked elf after every kernel change.
- **Stamping anything (round 8)**: the same bus enable works for the EW cores (row 5). Write timing records into
  bytes the op outputs but nobody reads (the SSM chunk's output pad) and the step stays correct; or over the first
  output words (timing builds only); find them by a magic word in the ACT image (npu-tapnext r8/gprof.py, ewprof.py).
  `__builtin_aie2p_sched_barrier()` inlined into a pipelined Peano loop crashed clang (AIE Region assertion): call a
  noinline stamp function there.
- **Control code size (round 8)**: two GEMM segments around attention = two copies of the loop nest. Walk ONE op
  table in which attention is an entry (a trip count of 0 / 1 from the table replaces an `if`), and take small
  constant trip counts (3 x 3 loops) from an L1 table so LLVM cannot unroll them: -1.3 KB, bit-identical.
- Peano copy / elementwise loops with indexed addressing (`p + 32 * i`) cost ~8 bundles per vector pair; bump the
  pointers (`s += 64`) and keep wrap checks out of the loop (copy in at most two straight segments).
- **LUT lookups (round 9, npu-tapnext r9/lprobe.py)**: `bfloat16_to_int(x, k)` is vfloor. A table's two copies in ONE bank
  work (+1 cycle per 4x load, +8 per 32 lanes). 16-bit tables (`load_lut_2x_int8`, index 4 B per entry) store runs of 16
  entries twice (aie_kernels vision/lut_inv.h): an exact bf16 sqrt is 256 mantissa entries + an int16 exponent halving.
  A table sigmoid needs a RELATIVE-accurate tail only as far as its consumer cares (TAPNext's gate_a: x > -12).
- **`#pragma clang loop pipeline_initiation_interval(N)` can silently give the SERIAL loop** (round 11 correction of round 10):
  with restrict copies of one pointer (Peano treats them as aliasing) or, in r10/gxb.cc, MIN_ITERATION_COUNT(8), Peano emitted
  ~50 unpipelined bundles; round 10 benched that and blamed "interlocks". With separate restrict PARAMETERS the 17-bundle
  forced-II GELU loop runs 17.8 cycles per 32 lanes, bit-exact. Read the emitted loop of the binary you deploy, then bench it.
- **Peano software-pipelines simple loops**: one long chain per vector is latency-bound (exp 63 bundles per 32 lanes with 2
  chains); the same ops split into short passes through L1 pipeline (exp: 5 passes, ~44 bundles). Rules: separate in /
  out pointers, constant trip counts, constants as LOCALS (a `const ExpC &` parameter made it reload every constant in
  the loop: aliasing), all loads of an iteration before its stores in source order, two independent pieces per trip.
- **Locks in asm / intrinsics**: `acq #id, rV` with rV < 0 is acquire_greater_equal(-rV); rV >= 0 is acquire_EQUAL (a
  "no-op" acq with 0 waits for the lock to be 0: hang). Conditional: `acq.cond #id, rV, r26` runs when r26 != 0.
  Locks of this tile are 48 + id; pin ids with `Lock(..., lock_id=)`; locks only a worker uses need `rt.add_lock()`.
- **Zero-overhead loops in hand asm (round 9, r9/gen_ks.py)**: `movxm ls, #start; movxm le, #last_bundle; add.nc lc, rN, #0`
  well before (lc = trips); the start 16-B aligned (`.p2align 4` pads with executable nops), first and last bundle written in
  the full six-slot form (nopa; nopb; nops; nopx; nopm; nopv where idle), the `le` label before the LAST bundle, and **the
  last bundle 16-B aligned too** (round 11: otherwise the body runs about once). Writing EVERY body bundle in the full 16-byte
  form makes both automatic without padding inside the body (padding would shift a modulo schedule). Initialise every
  FIFO register (r24 / r25 with the fills) and pointer the body uses. padda / padds immediates are multiples of 64; a
  table walked by `lda rX, [p], #4` post-increment (any r / p / m register) needs no bump.
- **The generator (round 11, npu-tapnext r11/msched.py)**: ops over virtual registers tagged with .td instruction names
  (ports from the itinerary), list or iterative (Rau) modulo scheduling, a MaxLive-aware sink pass, modulo variable expansion
  with circular register allocation, emission (prologue, full-form body, drain) and a value-level checker (each read sees
  its own iteration's def outside the bypass window; 2D post-modify addresses; spills through memory). r11/gen_gelu.py is the
  worked example (GELU at II 13, bit-exact; register pressure, not slots, set the II: 12 x / 5 dm). Bench every generated
  loop with r11/apb.py (one core, bit-exact compare, `--u` per probe: a wrong U runs the loop past its buffer = a timeout).
  Debug a wrong kernel with `--tap REG` (store an intermediate) and hazard probes (r11/mk_haz.py) before trusting the model.
  2D post-modify (`vlda.2d x, [p], dN`, `vst.2d`): dN = (m, dn, dj, dc): p += dj while dc < dn (dc++), else p += m (dc = 0).
  A leaf asm function may take stack with `paddxm [sp], #N` (grows up; space at old sp) and give it back before `ret`.
- **Modulo-scheduled asm kernels**: build one iteration's ops at stripe-relative times with resources reserved modulo the
  period, then emit prologue (stage -1 of iteration 0), the body, epilogue (stage +1 of the last); run the extra stage
  ops of a dummy iteration harmlessly (reads only, acq.cond skipped). Check the UNROLLED stream: every MAC's operands
  (the pop / load it sees), every address, register-reload timing and lock order (gen_ks.py `check`).
- **Exact scheduling with register pressure (round 12, npu-tapnext r12/cpsched.py)**: on LUT / epilogue loops the allocator,
  not the slots, decides the II; msched's IMS + sink pass lands 2-3 above ResMII or fails to allocate. A CP-SAT model (one-hot
  issue cycles; slots, itinerary ports, banks modulo II; the exact steady-state MaxLive per register class; `maxdist` to keep
  a stream's ops of one iteration in order inside one II; `last_stage`; `max_life` to bound the unroll U) found II 6 / 12 / 16
  where msched did not. OR-tools is not on testbox: solve on the laptop (`uv run --no-project --with ortools python ...`), save
  the schedule as JSON, replay it in the build (`msched.Loop.place`). `vmsc s, sh, 1.0` instead of `vmac s, sh, -1.0` frees a
  constant register (exact, bit-identical).
- **Side-effect ops only in the LAST stage** (round 12): the drain runs the early stages of garbage iterations past the end,
  so any store / push / flush in an earlier stage writes past the buffer (bgelu stored x to its final place early and corrupted
  the 4 vectors after every run, in situ only: the bench did not check past the end). Spill through a ring on the stack
  instead, and make one-core benches check guard zones on both sides.
- **`.p2align 4` before the loop body inserts ONE executable nop bundle after the prologue** (round 12): ops issued in the
  prologue then run a cycle early against the body. Both register lifetimes and ITINERARY PORTS can break (bgelu II 11: a
  prologue vlda's QEY_WA write collided with a body vfloor; only iteration 1 wrong). msched.check replays pads 0 and 1 with port
  reservations along the real timeline; reject schedules that fail either.
- **Zero-overhead loop body >= 3 bundles** (round 12, r12/gen_copy.py): a 2-bundle body ran once.
- **Kernel-only loops** (no prologue: the body runs ceil((S - 1) / U) extra trips over negative iterations; msched
  `kernel_only`): saves the prologue's text (the EW core had 144 B free). Every stream with negative-iteration accesses (loads
  too, msched reports `neg_steps`) must start that many steps early (round 12: only the store was offset -> output shifted by
  3 vectors); walk the buffer in the direction that puts the garbage stores in dead memory; initialise the index registers of
  LUT loads to the table base so garbage lookups stay in the table. Plain `vlda x, [p], #-64` post-modify walks backwards fine.
- **Compact bodies (round 12, r12/asmfit.py)**: only the first and last body bundles must be 16 B aligned; assemble once, read
  the bundle sizes, expand the cheapest subset to the full form so the last bundle lands aligned (subset sum mod 16), verify.
  -192 B on a 24-bundle body. Text is the binding budget on both core types now (GEMM 16,288, EW 16,224 of 16,384 B).
- **Block-vector output streams in asm (round 12, r12/gen_pushprobe.py)**: `vst.push.576.conv.bfp16ebs8.fp32 dmN, [p2, sf,
  r26]` (r26 = 0 at the start) pushes 64 fp32 lanes as one bfp block (72 B); `vst.flush.512.conv [p2, sf, r26]` after every
  push, every second push or only at the end all give the codec's bytes, and afterwards the pointer is at the exact byte (32 B
  granular writes). `vst.flush.512.conv.2d [p2, sf, r26, dN]` adds its 2D post-modify to that position: one modulo loop can
  write strided rows (every 4th flush +32 or +288) or jump between objects. Back-to-back pushes need no spacing.
  `vlda.2d.conv.fp32.bf16 cml/cmh` loads bf16 straight into an fp32 accumulator half (lda only; DM_WM port).
- Calling convention detail: pointer arguments take p0, p1, ... in order of the POINTER arguments (a 5th pointer is p4); a 6th
  is p5. **A function that returns a value has r0 reserved for it and its first scalar argument in r1** (round 13: `int32_t
  attk(...)` got its trip count in r1 while the loop read r0: 5 trips, then a hang when r0 held 0). Read the call site in
  `clang -S` output (the argument moves sit in the `jl` delay slots) for any new signature, and copy scalar arguments to a
  safe register (r7) before the preamble's own `mova r1, #828`. The push stream's pointer must be p2 (`[p2, sf, r26]`).
- **ex_n ALIASES x_n (round 13)**: a bfp panel register (`vlda.pop.576 exN`) is the x register of the same number (Peano spills
  live x registers around pop loops). A loop that held Q in ex0-7 beside x constants computed garbage. In msched the "ex" part
  belongs to class x (`PARTS["ex"] = ("x", ("l", "h"))`); pops and x temporaries share the 12 registers.
- **Scalar loads use the A data port (round 13)**: `lda pN, [sp, #imm]` co-issued with a `vldb.4x64` LUT load (both DM_ADA)
  loaded garbage into the pointer (the fused attention loop's Q stream read junk from iteration 2 on). Give every scalar
  load an itinerary in the generator (`LDA_dmv_lda_q_spill` has DM_ADA at stage 1); the checker cannot see scalar registers.
- Scalar registers in a generated kernel: list them all in one place. r2 / r3 hold the shuffle modes (24 / 5) of the LUT
  line, r1 = 828 (bf16 mac config), r16 = the clamp masks, r24 / r25 = FIFO positions, r26 = the push position; the bfp
  mmul config (780) and `vadd.f` config (60) went to r5 / r6 after a generator reused r3 for both (every score wrong).
- **A drain's garbage store can land on data a later stage still reads (round 13)**: the fused K loop's iteration-6 garbage
  score store hit the -m rows that iteration 5's line loaded in the same trip: panel 5 of every row block wrong. Order the
  reads before the garbage store with a maxdist, or give the slack region nothing live.
- A bf16 compare can target any scalar register (`vlt.bf16 r20, x, x`, mv slot, latency 2) and `or r22, r22, r20` is an
  alu-slot op: a per-iteration flag accumulated in the LAST stage (garbage iterations run the compare, never the OR; the
  OR reads before the next iteration's compare lands: maxdist II + 1) and returned in r0.
- A FIFO pop stream can be restarted every iteration of a modulo loop: reload its pointer (`lda p0, [sp, #-64]` from a
  preamble store), `mov r24, #0` (mv) and the fill, ordered after the previous iteration's last pop (maxdist); the same
  576 B were popped per key block that way.
- **Bench a new asm kernel on one core first** (npu-tapnext r9/ksb.py: locks pre-armed by the kernel, hash of all outputs
  against the reference path): it found two hang causes after they had cost two multi-core timeouts.
- **Judge in-situ k-step rates by the per-tile MEDIAN, never the mean** (round 10 correction): round 9 read a 2,588-cycle mean
  k-step as "A port ~4 B/cycle binds"; the per-tile distribution (npu-tapnext r10/gdist.py) shows steady k-steps at
  2,050-2,065 cycles for every op, with each op's first tile carrying the A-slot load wait (28-34 us) inside its first
  in-kernel acquire. A arrives at >= 5 B/cycle per core in steady state. What runs at ~4 B/cycle per column is the
  DDR -> memtile slot load (8 x 248 KB in ~34 us = the DDR read peak).
- IRON's jit caches compiled designs by CompileTime arguments: builds that differ only in compile flags reuse the first
  build unless the flags are in a CompileTime key.
- **Program memory, round 13**: a per-core table that a kernel builds once at run time (mm_kstep's 58 k-step records, 832 B of
  code) is better built as `constexpr` tables at compile time (the records were `const + b x (c_out - acc_hi)` or a slot
  address): ~60 B of code + 174 B of .data (.rodata counts against the data region, not program memory; `minsize` on the
  run-time builder saved nothing). A one-core bench that includes both the C++ and the asm paths of mm8.cc needs `-ffunction-
  sections` + `ld.lld -r --gc-sections -u apb` and `minsize` on its own code to fit (r13/attkb.py, apb.GC).
- **Two design builds in parallel in one build dir race** on the generated asm (asmfit's `<name>_fit.s`): "the last body
  bundle is not 16 B aligned" from a build that is fine alone. Build sequentially.
- **The lazy-fast attention K phase (round 13, `-DATTKF`)**: one asm loop per row block does QK^T, the scores to L1, exp(s -
  stored max) and the bfp push, and returns a flag (any lane with s - m >= TAU - 0.25); the C++ runs today's exact test and
  path only for flagged row blocks (~10 % of units: chunk 0, the chunk with invalid key blocks, the ~1-2 % whose max moves).
  Bit-identical by construction. Measured in situ: the fast loop alone would be -2.9 ms, the flagged units' rerun + the
  column lockstep below leave -0.5 ms; with the deeper K / V buffer -0.84 ms (50.05 -> 49.21). The fallback's cost, not the
  loop, is what to attack next (a cheaper slow path, or no redundant fast pass on chunk 0 / invalid chunks).
- **The attention's K / V stream is multicast to the 8 columns (round 13)**: per chunk every column waits for the slowest
  one, so a rare slow unit (the lazy-max fallback, ~10 % of units) costs ~3x its own time on the op's wall. A deeper memtile
  forward (`--l2-depth 8`, 2 -> 8 K / V objects per core row) decoupled the columns: -0.36 ms; 12 gave no more.
  **Mechanism (lowered MLIR, npu-research rev 9)**: a `forward` to N cores is ONE memtile MM2S channel multicast; a send waits for
  a free L1 slot in every destination (2 slots: a column runs at most one object ahead). The memtile depth only decouples the
  shim side, so why l2-depth 8 helped is still open (research PROPOSALS Q1). Price any non-uniform per-unit cost behind a
  multicast as the max over the destinations per object (npu-research `tools/attsim.py`, calibrated on the round-13 twins, per-
  block corr 0.97), never as its mean; a cost every column pays at once (chunk 0 / 22) passes 1:1.
- **The lazy-max fallback is spatially clustered (rev 9, CPU map `npu-research/r14/lazymap.py`; [model, not yet measured])**:
  flagged = really moved (flag-not-move 0.055 %), 2.3 % of row blocks after chunk 0, mostly in columns 5-6 (lower image rows).
  Dealing query row blocks to the columns round robin is bit-identical (rows are independent) and removes most of the lockstep
  (research P60). A row's self-score q_i . k_i is nearly always its max (a numerics-changing seed option).
  **Measured round 14**: with the batched K loop, the twin that ignores the batched chunks' flags (full12r14cF128) ran 2.1-2.6
  ms faster than production for ~0.35 ms of the fallback's own cycles (x ~6). P60 cannot be done with one TAP per column
  (r14/deal.py: only column permutations are arithmetic in the tile index); it needs per-piece DMA patterns.
  **Round 15**: research rev 10 reads the cost as ~1.3 ms multicast + ~0.8 ms head-group barrier (not "x 6"); its P60g deals HALF
  tiles in the two consumer GATHERS only (attention Q fill, out_proj A fill: one head-major 4-dim TAP per column each, drains /
  cores / host unchanged): measured -1.06 ms (predicted -1.08 +- 0.22), bit-identical (plan8 P60G=1).
- **The lockstep re-calibrated (npu-research rev 10, sim17; [model, checked three ways])**: with the round-14 one-core costs, the
  CPU lazy-max map overstates the in-situ flags ~1 / 0.6 (kappa 0.6 reproduces the F128 twin, the per-block walls at slope
  0.98-1.05, and the in-situ count). Of the ~2.3 ms the flagged row blocks cost r14f, only ~0.22 ms is their own cycles; ~1.3 ms
  is the K / V multicast lockstep and ~0.8 ms the head-group barrier waiting for the busiest core. So balancing the rows across
  columns beats decoupling the columns. [DOC lowered MLIR] design8's memtiles 0-2 use all 6 S2MM channels (A slot x2, B, C join
  x3): there is no channel for extra per-column streams.
- **Gathers into an A_RT slot must arrive chunk-major (head-major)** [DOC design8 `chunk()`]: the slot load is a BdIteration of
  10,368 B stream chunks with a lock per chunk, so a fill split into several TAPs (piece-major) lands pieces in the wrong chunk.
  A row deal therefore has to be ONE 4-dim TAP per column: [12 heads][n1][n2][bytes] affine with non-negative strides. Families
  that move the source column only along size-2 dims satisfy both gathers; research P60g (a 6-column transpose of two triples +
  a half-split pair) is single-TAP on both the attention Q fill and out_proj's A fill [algebra verified, r15/deal4.py; not yet
  built].
- **Disassemble with `llvm-objdump -d -z`**: without -z, runs of 2-byte nops (all-zero bytes) print as `...` and loop bodies look
  up to half as long (the SSM's loga_p: 11 instead of 23 bundles). tools/loops.py and bundles.py assume -z.
- **Pre-register the whole call, not the loop** (round 14 P50: loop 87 cycles per row block as predicted, but the second asm pass
  for the row sums (21) and the calls / C++ glue (27) made it 136).
- **Zero-overhead loop setup distance (round 14, r14/zero2.s)**: with 6 two-byte `nop`s between `add.nc lc` and the body the loop
  ran ONCE; 6 full-form (16 B, six-slot) nop bundles work. Generated kernels get the distance from their prologue; a bare loop
  needs the full-form gap (the same for a second loop set up right after the first).
- **FIFO pop streams restarted mid-line (round 14, r14/gen_attv.py)**: after a fill + 6 pops of 72 B (432 B, not a multiple of 64)
  the pointer is NOT start + 448 (nor + 512): restart with an explicit pointer (a walker register stepped by padds.2d, copied by
  `mov p0, p4`) plus a position reset (`mov r24, #0`) before the fill. Restarts after 576 B (9 lines) may continue. The walker
  copy must come >= 2 cycles after the walker's previous step (a maxdist), else each period's first iteration reads the old one.
- **Scalar results are read 2 cycles on**: an alu `or r4, ...` read one cycle later gives the OLD value (a call-wide flag OR
  missed the last iteration: the bench passed, the in-situ regress did not). msched cannot check scalars: `after` edges >= 2.
- **Pointer-only 2D steps**: `padda.2d` / `paddb.2d` / `padds.2d [p], dN` (lda / ldb / st slot) step a pointer by a 2D modifier
  without a memory access; 3D modifiers pair dN with dN+4. A modulo loop over (row block, key block) needs no 3D pops.
- **A held B-fifo object's junk tail is scratch**: K objects use 3,456 of 4,608 B; while the core holds the object the tail can
  carry per-call data (the -m rows of r14/gen_attkb.py); the DMA rewrites the whole object next time.
- `vlda.conv.fp32.bf16` (bf16 -> fp32 in the load, DM_WM port) replaces a load + ups in modulo loops (bit-exact); `vlda bmXX` /
  `vst bmXX` move fp32 accumulator quarters 64 B at a time (O straight into a MAC's accumulator, r14/gen_attv.py).
- `zero2(R, R + 64 t, t)` (r14/zero2.s: 64 B at a and 128 B at b per trip) zeroes any 192 t-byte region at one store per cycle.
- **Text levers (round 14)**: 2-stage schedules (`max_len = 2 II - 1`) halve the prologue (attv -150 B for +5 cycles per call;
  impossible for long chains: attkb found nothing up to II 42); `max_life = II` forces U = 1; straight-line calls instead of
  tile loops in wrappers (Peano kept 11 callee-saved registers: 416 B for 2 calls); static addresses instead of pointer
  globals; rows8 = `aie::transpose(concat(v16 x4), 8, 8)` (vshuffle modes 52 / 53); a V^T epilogue in asm (-496 B C++ for
  278 B). minsize does not shrink vector-heavy code (the V^T epilogue at minsize was larger). Measure with r14/csize.sh.
- **Round 15 (npu-tapnext r15/)**: vld / vst immediate offsets are 4 bits x 64 B (-512 .. 448; padda #imm the same: c10s_step64,
  multiples of 64 in [-512, 448] - the assembler's "[-512, 511]" message is only the containing range): point the base at
  table + 512 to reach 16 rows. A table in `.aie.bankN` must be non-const (a const one is a section type conflict with the LUT
  tables). `vmov x, bmLL` has its own itinerary (DM_RM_L0_PORT, latency 2): set Op.ports by hand (aie2p_lat.json's generic
  VMOV_alu_mv_mv_x has none). More bf16 constants than x registers: keep the most-used one resident and reload the others per
  use from a 64 B-row table (spare ldb slots; fp32 1.0 for an accumulator input by vlda.conv.fp32.bf16 from the bf16 1.0 row).
- **Extra copy passes into the EW output ring cost ~60 cycles PER CALL** (research rev 11, static whole-call count; corrects round
  15's "~3.5 cycles per stored vector"): one out_copy(16 vectors) = 77 cycles in r15b of which 16 are the stores (out_copy glue,
  cpy64 setup + 7-bundle prologue + drain, jl / ret delay slots); 3 calls + glue = the measured 289. No bank-0 store stall is left
  to explain [model; one-core DMA-drain bench named in research PROPOSALS rev 11]. Still: store / push results straight into the
  ring, and make copies few and long.
  8 bfp pushes of 72 B (576 B, a 32 B multiple) + one flush write nothing past the 576 B (bench-checked).
- rep4 (8 bf16 x 4) = `vldb.128 w.wl, [p], #16` + 4 x `vshift d, prev, w, r(=16)` (any register as the first source); bit-exact.
- minsize subtraction loops that count ring slots (in_need / in_done / out_begin / out_end) cost ~1 ms per step: x / 768 =
  ((x >> 8) * 21846) >> 16 for 0 <= x < 65536 (exhaustive).
- A value live across the whole chain (stored in the last stage) needs U >= lifetime / II; a second load stream that reloads it
  costs less text. Run msched.check inside the CP II search (a schedule can fail only with the .p2align pad). Callee-saved p6 / p7
  in a leaf kernel: `paddxm [sp], #64; st p6, [sp, #-64]` ... `lda p6, [sp, #-64]` before ret.
- **IRON's jit cache key does not include source files**: after editing mm8.cc / ew8.cc / a generator, build under a NEW name.
  A bench can key on the kernel object instead: `specialize(key=f"{out}_{md5(obj)[:8]}")` (pass 14 rotb.py; the stale elf kept
  an old placement otherwise).
- **tcall / asmlint on a generated .s: link it first** (pass 14): in an unlinked .o `movxm ls / le` print `#0x0` (relocations),
  so tcall sees no ZOL trip count; `ld.lld x.o -e func -o x.elf` then `llvm-objdump -d -z` resolves them. A label without a
  leading `.` (and not `<func>_start / _last`) starts a new "function" in their .dis parser.
- `mova` takes an 11-bit signed immediate ([-1024, 1023], assembler error otherwise): larger constants by `movxm`.
- **Pinning a static in a one-core bench** (pass 14 rotb.cc): `__attribute__((section(".aie.bank3")))` lands in the linker script's
  bank3 region (the space after IRON's buffers, ~5 KB there); Worker `data_size` must shrink by the pinned bytes or aiecc fails
  "section '.aie.bank3' will not fit in region 'bank3'". A zero array put in `.data.*` lands at the data region's start (bank 0).
- An internal state layout only the kernel itself reads back (the SSM's h) may be changed freely: the DMA moves it opaquely.
- A C++ loop with 4 fp32 accumulator chains (-O2) spills every iteration: a 32-lane bf16 MAC writes a whole dm (5 dm registers).
- **Lint every generated / hand asm kernel before building** (research rev 11): `npu-research/tools/asmlint.py` (in npu-iron, CPU
  only; `--prj build/X.prj` for a build's GEMM + EW elfs, or the generator's `.s`; `--rules` lists the 22 rules and what cannot be
  checked statically). It flags all 18 historic bugs kept in `npu-research/r16/hist/` (`tools/asmlint_hist.py`) and is clean on
  full12r15m. `npu-research/tools/try.sh NAME --ew-add .. --k-add .. --env K=V [--sweep]` = build (fresh name) + lint + regress +
  time in one command (`--print-only` shows the commands).
- **ZOL setup distance is in BYTES to the loop END** (rev 11: the bad and the good zero2 both have 10 bundles from the lc write to
  le, 52 vs 132 B). Peano writes lc 2 bundles before ls when the body is long. Keep >= 80 B from the last lc / ls / le write to le
  (every deployed loop; <= 52 B ran once; the gap is unmeasured). Peano's loops start with compact 10 / 12 B bundles: only the
  LAST body bundle must be full form and 16 B aligned.
- Immediate ranges come from the .td operand types (cNs_stepK: signed N bits, multiples of K; 256-bit w loads c9s_step32 =
  [-256, 224]); the assembler aborts with a crash-looking "error in backend" on any violation.
- Peano itself relies on the mv -> mv bypass and on scalar conveyor reads (reading r / p while a newer lda is in flight): do it
  in generated code only on purpose. Scalar `lda` has no itinerary in the .td; Peano uses its result 4-6 cycles later.
- [UNVERIFIED scope] the r14 "scalar result read 2 cycles on" failure was alu -> alu INSIDE a ZOL body; Peano does alu -> alu at +1
  outside ZOL bodies everywhere, and deployed loops do alu -> mv at +1 inside them. Keep alu -> alu >= 2 apart in ZOL bodies.
- Linker-script `bankN` regions are the space for `.aie.bankN` sections, not free space: free = LENGTH minus the sections there
  (`llvm-nm -S`). Per-call scratch fits on the stack (`paddxm [sp], #N` in a leaf asm kernel): the GEMM cores' deepest path uses
  832 of 2,560 B (stackaudit; its ASM_FRAMES table lacks the r13-r15 kernels' own frames).
- Parallel design8 builds: serialise them (`npu-research/tools/asmfit_tmp.patch`: a flock on build/.design8.lock). Per-build
  object dirs are NOT a fix: mm8.cc includes the generated headers as "build/<k>.h", which resolves next to mm8.cc first.
- **A dense 576-bit FIFO pop stream underflows at its 9th pop (round 16, npu-tapnext r16/popprobe.s probes 6 / 7)**: each pop
  brings one 64 B line but consumes 72 B; without an extra `vldb.fill.512 [p, lf, r]` (or vlda.fill) at the SAME position (no
  pointer / position reset) after every 8 pops, pops 8-12 read zeros / half exponents and the stream then resyncs. Peano inserts
  that fill (gate_gemm); streams restarted every <= 8 pops (attv, attkb) never hit it. A generated loop with one long pop stream
  needs the fill as a scheduled op (r16/gen_ggk.py FB0 / FB1). Symptom: whole k-terms missing in EVERY block, identical across
  schedules; find it by tapping the operand register at the MAC's cycle (a vst into a free st slot of the emitted body).
  (Round 17: popprobes 2 / 3 had checked only the lf0 stream; their lf1 stream of 16 dense pops is a true FIFO-POP9 positive.)
- **The GEMM k-step's largest static item is the accumulator swap** (research rev 13, `npu-research/tools/ksplan.py`): mm_ks runs
  32 MACs per 47.75 bundles, of which 12.75 are the store / reload of the 4 accumulators, which all end together (459 cycles per
  k-step). dm4 is unused. With the swap gone, the next bound is the B bank: 8 pops + 1 fill per 8 MACs in one bank. The A port
  floor is 10,368 B / 7.95 = 1,304 cycles per k-step.
- **ROT, the rotating-accumulator k-step (research pass 14, `npu-research/r19/gen_rot.py`; [CP + whole-call replay + asmlint;
  NOT yet run on the device])**: a job = one accumulator over the 8 k-panels (8 dependent vmacs, kp order = mm_ks's: bit-identical),
  jobs every 9 cycles so 2-3 chains interleave and one dm finishes every 9 cycles (dm[cb % 4]); B is popped in MAC order, so B
  must be REPACKED on the host (stream position -> pair4 panel, `mm_rot_border`). Bound: the B bank, 72 line reads per row block
  + the restart gap g: R = 72 + g per row block (CP OPTIMAL for g = 0 / 1 / 2); whole call 1,384 (g 2, deployed restart) vs mm_ks
  1,824, 0 bank stalls, text -338 B. Recipe and risks: research PROPOSALS rev 14.
- **Identical job templates at fixed offsets can be structurally infeasible** (pass 14, CP INFEASIBLE + proof): with jobs 9 apart
  and a 74-cycle row block, the B pass's exact cover wraps a job's pops by 74 mod 9 = 2 and forces its 8 pops into consecutive
  cycles, which a MAC chain (distance >= 3, 4 B registers) cannot follow. When the loop body is one row block, give every job its
  own issue cycles (gen_rot `--free`): feasible at the bound.
- **A skewed pop stream starts mid-pass in the prologue** (pass 14 replay): row block 0's earliest pops sit in pass -1, so the
  prologue restarts the B stream at the last 576 B boundary before them (`movxm m3, #576 i; padda [p1], m3` in setup = a fresh
  stream at a line) and runs every B-stream op from there, including the previous row block's pops nobody reads. [UNVERIFIED on
  hardware; the rotb bench covers it.]
- The .p2align pad before a modulo body broke ROT's replay at pad 1 (a prologue pop landed a cycle early and clobbered a B register):
  gen_rot `fit` expands prologue bundles to the full form until the prologue ends 16 B aligned and asserts on the disassembly that
  no pad bundle precedes the body (the round-12 rule, now automated).
- `rel.cond #id, rV, r26` (REL_COND, alu slot) releases only when r26 != 0: deployed in mm_rot since round 18 (bit-identical).
  **`vldb.pop.576.2d .., [p1, lf1, r25, dN]` with dnN = 0 ADDS mN to the 64 B line step** (round 19 probe npu-tapnext
  r19/wrapb.py, MEASURED: m2 = -4608 wraps a 4,608 B object; the pre-registered -4544 gave a wrong pass 2). `padda [p1], mN` works
  ONE bundle before the next fill, not in the fill's own bundle (MEASURED). Deployed as ROT g0a (round 19).
- **DMA writes into the bank the core is using cost the core ~1 cycle per 32 B** (main round 18 DMAB, measured: +58-59 per 256
  accesses at 8 B/cycle vs the model's 65; other banks 0) and the k-step's in-situ excess is these collisions (KS-P38: MEMORY_STALL
  173.7, LOCK_STALL 13.3 per k-step vs the model's 221): price layout and kernel changes with ksplan's model x 0.79-0.9.
- **2D modifiers are periodic** (dn steps of dj, then m, then again): for exactly one jump inside k steps set dn = k - 1 and start
  the counter dc at k - n1 (round 16 ring wrap in r16/gen_n1p.py); dn = n1 - 1 jumps every n1 steps.
- **A flag can be #undef-ed by an included header** (round 16: -DPIN compiled the old path, same elf md5): after adding a flag,
  check the elf changed (md5 / llvm-nm sizes) before timing; prefer long, specific flag names.
- 64-lane bf16 MACs read register PAIRS (`vmac.f dm, dm, y4, y5, r` with y_n = x_2n : x_2n+1): per-lane fp32 accumulation in ONE
  dm instead of two 32-lane MACs writing two whole dm (round 16 n1k statistics; msched: a generator-level "y" class, gen_n1k.py).
- **msched loop-carried uses (round 16)**: Op(cuses=[(v, part, stage)]) reads the previous iteration's value (an accumulator);
  U = 1, the accumulating op in the LAST stage (drain iterations must not add), the initial value loaded into the allocated
  register by the preamble, stored after the loop (+7 cycles). cpsched takes the carried edges and lifetimes.
- **Register-resident state across C++ calls (round 16 n1k _begin / _step / _end)**: if the C++ between the calls is scalar only
  (check the elf: in_need / in_done use r0-r4, p0, a lock), vector constants, dm accumulators and dn / dj survive; m and r
  registers do not (Peano steps pointers with m0): set those per call.
- Per-call overhead decides short loops: preamble + (stages - 1) x II + ret ~ 70 cycles bought little for an 8-iteration loop
  (n1p 133 vs 211 C++); fold more work per call or keep state across calls.
- Timing twins overstate once an op nears its I/O bound (round 16 norms: twin -1.98 ms, real asm -0.2; in_need waits 105 cycles
  per unit by NPROF stamps): stamp the waits before pricing a compute cut. **Research rev 12:** N1K's -0.2 was its correct
  whole-call static price (the miss was rev 11's loop-only price); the twins saved 0.17 / 0.5 ms MORE than their own static
  compute, which no I/O mechanism explains. UNVERIFIED cause: twins with wrong outputs change downstream data-dependent work (the
  attention's lazy-max fallback). Read a twin's per-op walls from `step8.py timeline`, never the step delta.
- **EW input = ONE shim MM2S channel per column, ~3.8 B/cycle (round 19 MEASURED, r19d walls; supersedes research rev 13's FITTED
  5.75)**: 3-input norms run 801 / 821 cycles per unit = 3,072 B at 3.84 / 3.74 B/cycle, while 2-input norms with the same compute
  run 567 (RMS) / 693 (LN). `Ein{c}` shares shim MM2S 0 with `Ain{c}`; MM2S 1 carries B on shims 0-2 only, and the EW core's S2MM
  1 is unused. Output ring not binding (MEASURED out_begin ~5 cycles per unit). The rev-13 fit's free per-op constant and pass-2
  excess absorbed the n3 input time: a fit with a free constant can hide a bound, so check it against per-kind walls.
- **DDR read cap ~55-59 GB/s aggregate (round 19 DDRCAP, MEASURED, npu-tapnext r19/ddrcap.py; refutes rev 15's "84 GB/s, cap
  unmeasured")**: one shim -> core stream alone reaches ~7 B/cycle (12.6 GB/s); 8 or 16 concurrent streams (1 or 2 per column) cap
  at ~55-59 GB/s of reads (63-74 GB/s reads + writes with drains). The 3-input norms read 3,072 B x 8 per 801 cycles = 55.5 GB/s:
  they sit at the DDR cap, so a second input channel per column (NIN2) cannot help; only fewer DDR bytes move them.
- **A stamp interval includes the call it brackets** (rev 13): NPROF's in_need / out_begin numbers contain ~44 / ~40 cycles of
  static call (jl + delay slots + body + ret + the stamp); net wait = stamped - static. Subtract before fitting.
- **A fit with a nuisance parameter fixed by assumption mis-assigns the bound** (rev 12 -> 13): R_out 2.35 with the bench's pass-2
  cost and R_out >= 3.5 with ~4x that cost reproduce r15i / NRMK / N1K equally; only a stamp on the hypothesised wait separated
  them. Before pricing levers off a fitted bound, stamp the wait itself.
- (Round 18: the DMA-collision model above is MEASURED by DMAB and KS-P38; research rev 13 had it as unverified.)
- **All 24 GEMM cores share one L1 placement** (r17d input_with_addresses.mlir: C object 0x78900, A slots 0x77B80 / 0x78000 /
  0x78480): per-core tables built at run time from those addresses (mm_kstep's ks_records, ~280 B of text) can be build-time
  data, guarded by a post-link check.
- **IRON lowers worker loop counters to i64** (opted_main_core_*.ll: `phi i64`, `icmp slt i64`; the endless loop runs to i64 max):
  each loop pays ~6 extra scalar ops plus hi-word spills in the core's control code. [UNVERIFIED lever] i32 bounds on `range_`.
- **The same C++ loop can be serial in situ and pipelined in a bench** (rev 13): the SSM z loop is 27 bundles per 64 lanes inside
  ew_ssm (aliasing pointers, a big function) and ran 97 cycles for all 8 in its one-core bench. Read the deployed disassembly's
  ZOL body (tcall --func) before trusting a bench of C++.
- **Price with tcall** (npu-research tools/tcall.py): whole-call cycles from the `-z` disassembly, natural loops, delay
  slots, callees, a per-step call census; `--seg` for phases, `branches` in the spec to pin a mode. It reproduced r15h's norm
  row blocks within 3 % and the r15h -> r15i delta exactly. A part of a 1 KB-unit stream straddles the 15 KB EW ring end
  whenever the op params leave the ring position off a 1 KB boundary (SSM: 10,752 B per head, ~47 % of chunks pay a copy).
- **An EW input fill group must total a multiple of the 768 B ring slot (round 17)**: the shim -> core packet BD completes per
  slot, so padding the SSM params to 11,264 B (1 KB multiple) left a slot unfilled and the step hung (timeout). Pad to multiples
  of lcm(slot, alignment) (AL1: 3 KB). Same for any new fill size on the EW rings.
- **Loop state through memory in a modulo loop (round 17 S2)**: iteration i loads what i - 1 stored (by offset, no post-modify) and
  stores with post-modify (vst #64, vst.2d dN): maxdist(load, last store) <= II - 4 (data land +4, pointer +3) and the stores in
  the last stage (drain garbage iterations must not write). msched's spill / fill model does not cover it: assert it in the generator.
- cpsched + the .p2align pad: a CP schedule can fail msched.check only with the pad; re-solve at the same II with the colliding op's
  time forbidden (`forbid=[(op, t)]`, r14/gen_attkb.py loops it); when allocation fails at maxlive = budget, `reserve={"x": 1}`.
- Rotating state instead of copying it (round 17 X1): per-frame runtime sequences (one .bin per phase, swap the instruction bo per
  dispatch like set_depth) + a phase word in the params header the kernel maps by. A GEMM drain can split its C objects into two
  regions in one TAP when they arrive in a linear order ([2, half, RB, XB], first stride = the gap).
- Dispatch-time noise under a foreign CPU load is bimodal (~0.17 ms clusters): compare p10 / median / mean over 200 iterations
  interleaved with the baseline (step8 SPREAD), never one 60-iteration median.
- Compute placement (user rule, round 17): every CP-SAT / generator / python run on testbox (`nice -n 10`, <= 12 solver workers;
  ortools via `~/.local/bin/uv run --no-project --with ortools --with numpy`), never on the laptop.
- **`add rX, rY, #imm` takes a 7-bit signed immediate (round 18)**: `#64` assembled SILENTLY as `#-64` (a hang). Pass such
  constants from C++ and read the disassembly of every new preamble. asmlint IMM-SRC (rev 15) checks every immediate of a .s
  against its .td operand type on the SOURCE text (the disassembly of a wrapped value looks legal).
- **A lock acquire needs distance before the load it guards (round 18 ROT)**: research's g2f had `acq.cond` one bundle before the
  A fill and raced in situ (wrong, run-to-run varying tokens) while the one-core bench passed (it pre-releases every slot); re-solved
  with 6 bundles (as mm_ks; npu-tapnext r18/gen_rot.py ROT_ACQ_GAP=6) it is bit-identical. A bench of a lock-taking kernel says
  nothing about acquire timing unless a DMA feeds the ring (npu-research r20/rotdb.py does). asmlint LOCK-ACQ-GAP (rev 15) flags a
  fill of a freshly `lda`-loaded pointer < 6 bundles after an acquire (fires on g2f, clean on g2a / g0a / full12r19b); the gap
  between 2 and 5 bundles is unmeasured.
- **Core performance counters (round 18 KS-P38, measured)**: program Performance_Control0 / 1 (0x31500 / 0x31504: counters 0-1 /
  2-3, start event bits [6:0] / [22:16], stop [14:8] / [30:24]) from the runtime sequence with `npu_write32`; start = stop = an
  event counts its occurrences (event 1 TRUE = cycles, matches the timer). The core reads Counter n at 0x31520 + 4n by `read_tm`
  after the bus enable. 23 MEMORY_STALL, 26 LOCK_STALL. Lock values at 0x1F000 + 16 id read 0 from the core (not usable).
- **L1 addresses as compile-time constants (round 18 KSC / P35)**: constexpr record tables from -D addresses, verified after the
  link (design8 ksc_check: llvm-nm for statics, input_with_addresses.mlir for buffers and lock ids, all 24 cores) and the build
  fails if anything moved (acc_hi moves whenever .data grows). A kernel may take an IRON ObjectFifo's locks by id and its buffers
  by address when each call consumes an even count of the 2-deep fifo (IRON's own index stays right; record the buffer IRON hands
  out elsewhere, e.g. the bias object, to know the parity).
- Linked-but-unused entry points stay in the elf (no --gc-sections): leave unused asm objects out of the merge (design8 drops
  mm_ks.o under MM_ROT; an unbound mm_kstep still dragged mm_ks in).
- A/B timing (round 18): both builds in ONE process, interleaved blocks (npu-tapnext r18/ab.py), 300-600 kept dispatches each.
- **GEMM k-step static cuts land at ~0.75 of their static price** (round 19 MEASURED: ROT g0a -0.18 ms for 26 cycles, g1a -0.11
  for 15): price k-step changes x 0.75.
- **Time-resolved DMA-collision model (research pass 15, npu-research `r20/collide.py`) [MODEL, calibrated]**: replaying WHEN
  each 32 B DMA write lands against the banks mm_rot touches per cycle reproduces KS-P38's 167.5 with alpha = 0.72 core cycles per
  colliding write, the per-write loss DMAB measured directly (0.68-0.75). Split: A 75, B 92 per k-step. B's is structural (the
  popped B bank is busy 0.97); no feasible A-slot placement saves more than ~15 cycles (every row block's pieces cover two of
  banks 0-2). **B objects in two banks is infeasible under ROT**: a bank holding a popped B object can hold nothing the k-step
  touches, and two such banks need 23 KB of k-step-cold data (the GEMM core has 14.6 KB).
- **Op boundaries (r18m timeline, MEASURED by main)**: a GEMM whose A is the previous op's output starts ~33-38 us late (non-
  dependent ops: ~0); boundaries cost ~5 ms per step (GEMM 3.3, attention 1.2, norms 0.8). **The command processor costs ~0.06
  us per npu_write32 and ~0.11 us per npu_maskpoll** (round 19 CPRATE, MEASURED: 400 extra per block on mlp3_a's path), but
  moving post-sync work before the producer wait did NOT shorten the boundaries (round 19 SEQPRE: slot loads first -0.04 ms,
  drains first +0.15, memtile sends first +1.1): the boundary is not the CP instruction count. Never push a memtile MM2S task
  before the same op's fills: the 4-deep queue fills and the blocked push waits on work the CP has not issued yet.
- **What a dependent op boundary IS (round 20 TPROF + CPST, MEASURED by main; decomposition research rev 16)**: for mlp0 -> mlp3_a
  the CP passes the token wait 12.2 us after the producer's last epilogue (the "tail": its last C drain + completion token),
  the 8-tct run takes ~1 us, then 8 A fills + the memtile sends reach column 0's first k-step in ~4.2 us (+0.85 us per further
  column for K = 1536: 5 send pushes x 0.17), and the first tile is then paced by the A load at the DDR read cap (k-step
  2,630-2,980 cycles instead of 1,576; 6.5-8.5 us per K = 768 op, 13.3 for K = 1536; 0.87 ms per step). A BD write + patch
  costs only ~0.10 us (INFERRED from CPST), so hoisting BD writes is worth <= 0.1 ms. Boundaries total ~3 ms per step, not 5:
  the timeline "gap" of a GEMM contains its whole first tile. **DDRW (main round 20, npu-tapnext r20/ddrw.py, MEASURED)**: one
  shim S2MM stream writes ~7 B/cycle (12.1 GB/s), 4 columns 45 GB/s, 8 columns ~50-60 GB/s: the tail is the last 442 KB of C
  objects (store-and-forward joins) at the AGGREGATE DDR write cap (= the read cap), not token latency or a per-channel limit.
- **Host-poll timeline medians are quantised** (rev 16, verified on r18m / r19d JSONs): polls come every ~16.5-17.5 us, so a
  per-op median gap is 0 / 16.5 / 33 / 49.5 us. Use the MEAN over blocks 1-11 (SE ~1-2 us with 3 runs); norm gaps are 8.6-11 us,
  not "16.5 every op".
- **A_RT chunk locks are identity-free** (rev 16, from design8): the slot loads release ONE lock F for both halves and the t0
  sends acquire F per chunk, so loading half 1 before half 0 would let half-0 sends go before their data. Any reordered slot load
  (DAPRE, MLPRR) needs per-half F locks.
- **DDR bytes (rev 16, `npu-research/r21/ddrbudget.py`, plan8 TAPs vs r19d shares)**: a step reads 914 MB and writes 676 MB;
  every EW op reads at 48-55 GB/s inside its span (norms, SSM: at or near the cap), GEMM steady tiles 13-24 GB/s. APAN outputs
  (gated / gelu_pan) drain whole 18,432 B objects with 10,368 B of data: 74 MB of junk writes per step.
- **An acquire-only DMA BD is legal hardware-wise** (DOC aie-rt BD layout, as txn_trace decodes it): lock acquire enable /
  value and release id / value are independent fields; release value 0 = no release. The MLIR lowering of a BD block with
  only an acquire is UNVERIFIED (check the lowered BD words).
- **Boundary cures (round 20, all MEASURED and bit-identical, -2.07 ms together)**: keep the dependent A on chip (NRES / SRES: the EW
  producer streams the panels into the memtile A slot through its MM2S 1), or load the next op's A into the slot half the current op
  leaves free before the producer ends (IL / DAPRE / AQPRE / AKVPRE: per-half F locks, h1first fills, drain groups split so the
  consumer waits only for the chunks it needs), and issue everything independent before the wait (NSEQ). The CP's serial per-column
  issue staggers the columns and the B multicast holds every column to the last one (NRES alone gave -0.09; with NSEQ -0.19).
- **IRON 1.4.4 pathfinder: two packet flows from two DMA channels of ONE tile merging onto one port find no route** (even in one
  column; from one channel they route). Declare both from MM2S 0, give the second DMA channel by index (`DmaChannel(MM2S, 1, ...)`,
  `Bd(..., packet=(0, id))`), and patch the lowered switchbox (`aie.packet_rules(DMA : 1) { aie.rule(31, id, %amsel) }`,
  npu-tapnext r20/nresfix.py) before aiecc re-runs: the unpatched xclbin must never run. The core tile's 4 south ports are full
  (C x 3 + the EW output): a fifth southbound flow must share a packet port.
- A memtile S2MM runtime task may scatter a stream with a 3D TAP (`[18 rb][12 q][576 B]` -> chunk-major); keep the A slot layout the
  sends expect, or the last tile's chunk-wise release lets the next op overwrite rows still being read (round 20: an rb-major NRES
  half gave wrong G chunks 31 / 39 / 47).
- **Probing the CP (round 20 CPST)**: `npu_write32(address=<core-local> - 0x70000, column=c, row=r)` writes a core's data memory
  from the runtime sequence; a core spinning on that word stamps when the CP passed a point. The 24 GEMM cores' timers agree within
  ~1k cycles. A .bin is a TXN stream: `aie.utils.txn_trace.decode` + re-serialising (header op count / size) moves CP instructions
  (r20/binpatch.py). TPROF-style records under CSPLIT + EPLR go into C part 2 (part 1 may drain first).
- `vlda.pop.576` / `vlda.fill.512` on lf1 work (POPLF1, one core, 128 / 128; E7'' is not blocked by the pops). The EW core's deepest
  stack is ~3.2 KB (stack_size 0x1300 left room for a 1,152 B ring in bank 0).
- Dead (round 20): memtile sends before the wait after EW ops (SENDPRE +0.16); out_proj's A loaded per attention head group during
  pass 1 (OPPRE +0.18, cause not measured).
- **GEMM tiles wait for their own C drain (round 19 TPROF, mm8 -DTPROF + npu-tapnext r19/tprof.py)**: the C object (18,432 B)
  drains whole at 7.9 B/cycle and 12,288 B of the accumulators live in it, so each next tile waited ~2,330 cycles (~1.2 ms per
  step) plus acc_zero (~600). Zero acc_hi after the release (ZHI), drain the accumulator part first (CSPLIT), release it as soon
  as the epilogue has written it (EPLR); together -0.5 ms. L1 has no room to move the accumulators out (research rev 15).
- **Patch the lowered MLIR when IRON cannot express a DMA / lock shape (round 19 CSPLIT, npu-tapnext r19/csplit.py / csplit.sh)**:
  `aiecc in.mlir --unified --peano=<llvm-aie> --get-npu-insts --npu-insts-name=X.bin --get-xclbin --xclbin-name=X.xclbin
  --tmpdir=D --output-dir=D --get-input-with-addresses --dynamic-objFifos [--reclaim-runtime-bds]` on a build's own
  input_with_addresses.mlir reproduces its elfs, CDO and .bin byte for byte, so a patched copy (extra locks, a split BD chain, a
  core use_lock value) rebuilds in seconds. Re-run it right after the build (aiecc relinks build/*.o), check the .bin unchanged, and
  never dispatch the unpatched xclbin of a kernel that takes the new lock (a hang). A core MM2S object split over 2 BDs (each
  acquiring the cons lock; the first releasing prod, the second a new lock) reaches the memtile join as one stream: bit-identical.
- Peano: `#pragma unroll` over a loop with a conditional vector reference stayed rolled (selects + branches): unroll by macro. Put
  flag stores BEFORE a final call so it becomes a tail call (no callee-saved frame: ~-100 B per small wrapper).
- A one-core bench that #includes build/<k>.h gets whatever the last design build left there: copy the variant's header first.
- Every new/changed kernel: one dispatch with a short timeout first (`run.wait(3_000_000)` us), then a
  known-good design; after any timeout check `journalctl -k | grep -i xdna` and run the known-good test; max 3
  timeouts per phase; stop if the device does not recover. Never loop retries on a failing design.

## 4. The asm route (when a loop must beat Peano)

- `clang++ -O2 -std=c++20 -D__AIE_API_AIE_ADF_HPP__ -Wno-attributes --target=aie2p-none-unknown-elf -I<aie_api>
  -I<aie_kernels> ... -S k.cc` emits the schedule; `npu-moondream/p10/asmfix.py` fixes the three dialect gaps
  (`##sym` -> `#sym`, `.word` -> `.4byte`, `event #n` is unassemblable - compile events out) and
  `clang --target=aie2p-none-unknown-elf -c k.s` (Peano's integrated assembler; no llvm-mc, no inline asm)
  reproduces the object byte for byte (`p10/objcmp.sh`). Hand-written `.s` is assembled alone and merged with the
  normally compiled C++ by `ld.lld -r cpp.o asm.o -o merged.o`; link via `Kernel(...)`. **Never inject raw
  `.byte` bundles** (hung the core twice).
- Machine model: `p10/tdparse.py` turned llvm-aie's `AIE2PGenInstrInfo.td` + `AIE2PGenSchedule.td` (fetched
  read-only from Xilinx/llvm-aie branch aie-public into `p10/td/`) into `p10/aie2p_lat.json` (879 instructions:
  slot class lda/ldb/st/mv/vec/alu/lng - one instruction per slot per bundle - and per-operand latencies; apply
  the round-11 visibility rule, section 3, yourself). `p10/asmcheck.py file.s func --lat aie2p_lat.json` flags reads
  whose producer has not landed (info lines for conveyor reads of an older value); generators must additionally
  assert which pop a conveyor read sees (`p11/gen_gemm.py` does). Calling convention: pointers p0-p5, ints
  r0-r7, callee-saved r8-r15/p6/p7, no vector register preserved; fifo-pop pointer field is p0/p1 only;
  `vmax_lt.bf16` names r16; `vst`/`padda` immediates are 10-bit; the mmul config is 780 (`mova r3, #780`).
- Hardware trace: `Program.enable_trace(trace_size, workers=[w], coretile_events=[...])` (trace buffer = last
  kernel arg; one-column builds only - the trace unit could not be routed beside 8 broadcast columns),
  `aie.utils.trace.parse.parse_trace(buf, input_with_addresses.mlir)` -> chrome JSON; `p9/tracesum.py`,
  `tracekv.py` summarise cycles between `event0/event1` and the stall classes.

## 5. Dispatching from the host (pyxrt)

```
dev = pyxrt.device(0); xb = pyxrt.xclbin(p + ".xclbin"); dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid()); kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
instr = np.fromfile(p + ".bin", dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1)); bo_x = pyxrt.bo(dev, nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo.write(bytes, 0); bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
run = kern(3, bo_i, len(instr), bo_a, bo_w, bo_c); st = run.wait(timeout_us); assert str(st) == "ert_cmd_state.ERT_CMD_STATE_COMPLETED"
bo_c.sync(...BO_FROM_DEVICE); out = np.frombuffer(bo_c.read(n, 0), dtype=...)
```
Buffer order after `(3, bo_i, n_instr)` = the design's `Runtime(sequence, [types..., prods, conses])` inputs then
outputs (a trace buffer, if any, last). One xclbin per frame/step: a hw-context switch between xclbins costs
~2.7 ms and each dispatch ~80 us (pyxrt) - fuse ops into one runtime sequence. Weights/constants: write their
bo once, reuse across dispatches. Warm up ~20 dispatches (DPM ramp); "slow mode" (sub-ms dispatches finishing on
a ~1 ms timer) exists - `run.py` flags it.

## 6. bfp16ebs8 on the host

`aie.utils.bfp.encode(x_f32, rounding="conv_even")` -> 9 bytes per 8 values along the last axis: `[E][8 x int8]`,
value = m * 2^(E - 133) (E = the block's max biased fp32 exponent; mantissas round-to-nearest-even and saturate at
127); `decode` inverts. **The device's own conversion (`to_vector<bfp16ebs8>()`, `vst.push.conv`) differs: a
mantissa that would round to +128 bumps the shared exponent and re-rounds every mantissa from the exact values,
and -128 is kept** - ~2.5% of panels differ from the host codec by one mantissa step. Reference anything a kernel
converts with `npu-tapnext/pack7.py:encode_hw` (bit-exact against the device on 13,824 panels); host-encoded GEMM
operands stay with the codec (the mmul consumes either). An 8x8
panel = 8 rows of 9 B = 72 B; the GEMM program's operand orders are `pack.py` (`pack_a(..., half=True)` =
`[c][K/k][rho][m_a/16][2][k/8][8 rows][9B]`, `pack_w` = `[j][K/k][n/16][k/8][2][8 cols][9B]`); the attention
program's are `qk_obj`/`v_obj` in `runattn*.py` (K natural `[keys][dims]`, V transposed per 8x8 block). Accuracy:
bfp16 operands with an fp32 accumulator over the whole K are fine (moondream 99.6% vs 99.4%; TAPNext DAVIS
70.68 vs 70.54); bf16 C every 64 K cost 1.4 points; Hadamard rotation hurts with per-8 scales. GPU emulation of
the numerics: `moondream-opt/bfpemu.py`, `npu-tapnext/davis_npu.py` (`bfp_quantize`).

## 7. Performance facts

- Peak ~46 TOPS; the deployed GEMM loop runs 63% of the MAC slot (1.59 bundles/MAC; the ldb slot bound is
  1.125; a hand schedule reached 1.83 correct / 1.32 broken by the stage-3 rule - not worth more). Large frames
  are DDR-bound: NPU reads 37-39 GB/s under load (~55-60 peak); design6 re-reads A ~2x per N tile; C writes
  dominate when outputs are wide. Memtile -> core stream: 7.95 B/cycle measured (64-bit; r10/rtm.py); the slot load from DDR is ~4 B/cycle per column with all 8 loading.
- Measured moondream kernels (Turbo): GEMM frame 34 ms/27 blocks (DDR-bound), LN 116 us / 768x1152 (bit-exact),
  LN+residual 146 us, GELU epilogue 492 bundles/64 elements (bound ~114; the biggest remaining win), attention
  v15 0.675 ms per 729x729x16x72 (one pass, online softmax, bfp operands, 2 ms/frame of its budget is the exp).
- Elementwise work beside a GEMM frame: give each column one elementwise core (row 3) with a packet input ring and
  a circuit output ring (design7); it is input-port-bound (4 B/cycle): a 1152 x 768 bf16 norm pass costs ~70 us
  per op incl. the dependency bubble. Dependent ops through DDR in one sequence cost ~50-80 us of fixed time per
  op (slot load + drain tail): small GEMMs (1152 x 768 x 768) take 166 us, so fuse or prefetch rather than add ops.
- **Clock: 1.81 GHz (Turbo), settled in round 7** (section 1). Convert cycles with it.
- In-situ GEMM profile (round 7; round 8 for the asm call: bank stalls, section 3): the C++ mm_acc runs at its static schedule; ~400 cycles per k-step go to
  the 9 calls + IRON lock code; tiles wait mostly on dependencies (first tiles after EW ops, K = 1536 A reloads).
- Host side (pyxrt, measured): launch floor ~97 us median / 116 p99; frame upload 192 KB ~27 us (write + sync);
  read-back of a few hundred bytes < 1 us; mapped bos + numpy copy save ~30% on the write; cacheable bos only help
  at MB sizes. Resize/normalise a camera frame on the CPU (0.5-1 ms single-threaded) rather than uploading it.
- GPU jobs share DDR: NPU timings under GPU load are +10% or worse - record `gpu_busy_percent` with every number
  (`/sys/class/drm/card*/device/gpu_busy_percent`) and label contended runs.

## 8. Verifying and measuring

- GEMM programs: `run.py build/<name> --iters N --warmup 20 --check 1` prints `TIME` (kernel us median/min,
  TOPS, ddr_MB, GB/s, gpu_busy, SLOW-MODE flag) and per-op `CHECK` lines (bit-exactness vs the fp64 reference
  rounded to bf16: `mismatch_vs_bf16(ref) 0.0000%` is the pass; `max_err/ulp`); `M_REAL` env sets the real rows.
  `DUMP=<file>` saves one op's output for diagnosis (`gemmhyp.py` fits outputs to K subsets, `gemmmap.py` maps
  wrong 8x8 blocks - both found real bugs). Attention: `runattn*.py build/<name> <iters>` (rms vs the bf16emu
  reference, vs the bfp-operand reference, by tile/head/dim; `--save` for offline models).
- Elementwise kernels: compare to `moondream-opt/bf16emu.py` numerics (every op in fp32 then bf16 RNE); aim for
  bit-exact or <= 1 ulp and say which.
- Tracking accuracy: TAP-Vid DAVIS 'first' AJ (`tapnet/davis.py` metrics; `npu-tapnext/davis_npu.py` for
  emulated numerics on the GPU - only when GPU use is allowed; `npu-tapnext/npu_step.py davis N` runs the real
  NPU kernels inside the vendored model on the CPU, 10 videos ~1 h). References: bf16 70.54 (10 videos) / 66.35
  (30), fp32 66.3; stay within ~1 point. Check tracked POSITIONS against the frame (the heads had x/y swapped once).
- **Host-polled op timeline** (`npu-tapnext/step8.py timeline`): while one dispatch runs, poll the first and last
  word each op writes (64 B `bo.sync(FROM_DEVICE, 64, off)` per probe - host_only bos are CPU-cached, mapped reads
  alone never see device writes) and assign the k-th change of an address to the k-th op writing it: a per-op
  wall-clock timeline of the real multi-column program at ~17 us resolution, no trace hardware. It found the
  stride serialisation, the -Oz cost and the SSM in one afternoon; use it before any differential build.
- **Differential builds are cheap now** (design8 builds take ~10 s): stub one piece with a `-D` flag (`--kdefs=a,b`
  or `EW_DEFS`), take the timeline, compare the op's wall time. Round 5 split attention (exp / rescale / PV / K
  phase), SSM (transcendentals / ring I/O) and the GELU chain this way in minutes. Outputs are wrong: timing only.
- **Emulate a numeric formulation before building it**: a numpy model of the exact op sequence (fp32 accumulate,
  bf16 round at every conversion) against the current kernel's accuracy (npu-tapnext/r5/gelufit3.py) rejected a
  "cheap" GELU in minutes (26-37% bf16 mismatches vs 15%).
- DAVIS 10 videos on the NPU takes ~7 min at ~90 ms per step: run it for every accuracy-relevant change.
- CPU timings: say the torch thread count (torch's default in the container is 16; step8 uses 8) and median over
  many runs; a "0.2 ms" conv was 0.28 ms at 8 threads and 0.93 ms on one.
- Random-weight tests miss large-value bugs (a GELU that clamped its multiplicand passed every one): compare a
  depth-1 build with real weights against the torch block region by region (`step8.py blocks`).
- **Bit-level regression between builds** (`npu-tapnext/step8.py regress --save NAME` / `--ref NAME`): final tokens of
  all 1152 rows + tracks over 4 frames; a refactor must print BIT-IDENTICAL. Changing a numeric then shows its size
  against the NPU-vs-CPU noise (smoke).
- **Compute- vs data-movement-bound: twin builds** (npu-research P1): the kernel compiled with no MACs (`-DMM_NOCOMPUTE`,
  locks and DMA untouched) and with 2x MACs (`-DMM_TWICE`); full-step timelines of both. Round 6: -16.1 / +24.7 ms of
  84.0: the GEMMs are compute-bound (data movement ~1.0 us per k-step vs ~1.56 us of compute).
- **Numerics on real activations** (`npu-tapnext/r6/capture.py` + `numh.py` + `emu.py`): judge an approximation by
  the bfp ratio rms(approx - exact) / rms(bfp16(exact) - exact) of the tensor it lands in, not by bf16 ulps alone,
  and use each site's true reference (torch's ViT MLP is the erf GELU; everything else the tanh form).
- `npu-tapnext/r6/tlview.py tl_A.json [tl_B.json] --html out.html`: per-op roofline (MAC / loop / DDR / A-port
  models against the measured share) and a Gantt; `step8.py timeline` writes the JSON.
- Timing method: median and min over >= 20 timed dispatches after warm-up, kernel time (`run.wait`) separate
  from host sync; per-op cost by differential builds (`-DPROF_NO...`) only as a first look - the hardware trace is
  the truth (differentials misattributed the moondream scores block).

## 9. Safety rules (non-negotiable)

- NPU work only inside Docker; no host installs, no sudo, no driver/firmware/power-mode changes, no systemd, no
  listening ports. Remove only containers you created (`--rm` or `docker rm -f <yours>`), leave none running.
- No GPU use unless the coordinator explicitly allows it; then only via
  `~/git/app/.scratch/moondream-opt/gpu_safe.sh <log> <cmd>` (venv `venv-rocm714`), one GPU process at a time,
  short jobs (a GPU emulation job overlapped another agent's load when the GPU hung on 2026-10-09 00:01).
- Never touch the live app server (tmux `app`), voice model (:PORT), LLM server, gpu_guard, the user's phone, or
  other tenants' containers/processes. `pkill`/`pgrep` only with the bracket trick (`pgrep -f '[n]ame'`).
- Never read or print `.env`. Clean `/tmp` and your scratchpad when done; keep work under `.scratch/`.
- Long steps run in the background waiting on process exit with timeouts (not on log markers); one dispatch +
  short timeout for every new design; separate measured from extrapolated in every report and label the power
  mode and GPU contention.

## 10. Where things are (testbox, `~/git/app/.scratch/`)

- `npu-moondream/`: NOTES.md + RESULTS.md per phase (1-12), `p6/design6.py` (the GEMM frame program: memtile A
  ring, static BD chains; its constraints: `[K, nt]` = N tiles per CORE (4 cores split N, 8 columns split M),
  `wa == 288` words so m_a = 16 (m = 16 * rho, rho % a_ring == 0), `RA_K` = the cached A depth must equal the
  small K and other K's must be multiples of 2*RA_K, lock values 2*nt <= 63, <= 22 A-chain BDs), `pack.py`,
  `run.py`; `p7/ln.py` + `ln.cc` (bf16 LayerNorm, bit-exact), `ln_res.py`; `p8/mm_f32acc.cc` (GEMM kernel with
  `-DGELU_BF16` sigmoid-form GELU epilogue), `expsplit.h`; `p9/attn15.cc` + `attn13.py` + `runattn8.py`
  (attention v15: bfp operands, online softmax, ones-panel row sums, K/V broadcast, trace tooling); `p10/`
  (asm route: asmfix, asmcheck, tdparse, aie2p_lat.json, td/, gen_scores.py, buildmerge.sh); `p11/`
  (gen_gemm.py, the stage-3 finding); `p12/` (GELU staged experiments); `mlir-aie/aie_kernels` headers.
- `moondream-opt/`: GPU side (bench.py, gpu_safe.sh, bf16emu.py, bfpemu.py).
- `npu-tapnext/step8.py` (round 4, 5): the real-weights runner of a design8 build (`Step8(build, depth)`: W from the
  safetensors, ACT resident, `step(frame, queries)`, `set_depth(d)` swaps the runtime sequence on one xclbin;
  smoke / blocks / davis / time / timeline / sweep). HANDOFF.md there (state + ordered next actions), REDESIGN.md
  (the activation-stationary design, estimated, not built), `r4src/` (round-4 sources), `r5/` (round-5 probes).
- `npu-tapnext/r7/` (round 7 tools): clk.py/.cc (clock + timer probe), prof.py (in-situ GEMM cycles), t4var.sh,
  expemu.py / explut.py (attention exp by table on real scores), gen_mm.py (asm GEMM call); twins -DATTN_KTWICE /
  VTWICE / ETWICE. A host layout switch (pack8 B_PAIR4) must reach only the consumers that changed: `step8.py blocks`
  localises a wrong region in one run.
- `npu-tapnext/r8/` (round 8 tools): mmb.py/.cc (GEMM-call bank bench), gprof.py (GEMM tile + attention stamps of
  a -DPROF build: run8 tests via `PROF_OUT=X.npy`, or `--step BUILD`), ewprof.py (SSM phase stamps, EW_DEFS=-DPROF),
  cmpact.py (bit-compare two run8 ACT images). design8 env: `A_BANKS` (A ring banks), `ONE_SEG` (one op loop).
- `npu-tapnext/r9/` (round 9): ssmemu.py (RG-LRU numerics on real gates vs torch), ssmlut.py + ssm9.h (table sigmoid /
  exact sqrt / exp passes), lprobe.py (primitive probe), cpuemu.py (CPU model with emulated numerics: track noise of a
  numeric change without a build), gen_ks.py (pipelined asm GEMM k-step + checker + `--banks` model), ksb.py (one-core
  k-step bench). Flags: ew8 SSM_LUT / SSM_EXP3, mm8 MM_KSTEP / MM_PIECES / MM_KS; run8 MISMAP=1.
- `npu-tapnext/r11/` (round 11): msched.py (+ aie2p_lat.json) the modulo-schedule generator, gen_gelu.py, apb.py/.cc (one-core
  asm bench), asmx.py (Peano functions -> probes), mk_copy.py, mk_haz.py + haz.py / rd.py / mid.py (register-timing probes),
  stackaudit.py (bb.sh runs it; aiecc's measured_stack_sizes warnings agree), hostprof.py, q16bench.py, tlcmp.py (timeline
  per-op diff). Flags: mm8 GELU_MS / GELU_II, design8 P43, step8 HOST_FAST / HEAD_THREADS.
- `npu-tapnext/r13/` (round 13): gen_attk.py (the fused attention K loop: QK^T + exp(s - stored max) + push + flag, CP
  schedule attk.json), gen_attqk.py / gen_aexp.py (the split loops: superseded), attkb.cc / attkb.py (one-core bench of
  chunk_k vs the fused driver, bit-exact, skip modes), lazycount*.py (the lazy-max fallback rate on captured / model scores),
  attkcount.py (+ mm8 `-DATTK_COUNT`: in-situ slow-unit counts over the attention output), textmap.py, push.sh. Flags: mm8
  `-DATTKF` (fused K phase), `-DATTK_FORCE=bits` (timing twins), design8 env `ASM_FIT=1` (asmfit on the asm kernels), `--l2-depth`.
- `npu-tapnext/r14/` (round 14): gen_attkb.py (batched fused K loop, attkb.json), gen_attv.py (V phase pairs, attv.json; `--sums`
  the row sums, attvs.json), gen_vt.py (qkv V^T epilogue, vt.json), zero2.s, csize.sh (per-function GEMM text of mm8.cc),
  deal.py / deal2.py (P60 TAP feasibility), ssm_exp_deployed.s (the SSM exp chain as deployed). Flags: mm8 ATTKE, ATTKB, KFOLD,
  ZASM, ATTV, VTK (+ twins ATTK_FORCE=128, KRB_STAMP); r13/attkb.py `--defs / --hname / --extra-asm / --krb / --vphase / --zero`;
  msched: per-op 2D / 3D modifiers and pointer-only ("pa") stream ops in check().
- `npu-tapnext/r15/` (round 15): generators gen_ssx (SSM log_a + exp + 1 - a^2, constants by reload), gen_cpy64 (II-1 copy),
  gen_conv (SSM conv), gen_rec (SSM recurrence, U 4), gen_resc (attention rescale O update), gen_nrm (norm pass 2 RMS / LN with
  direct ring stores), gen_zk; one-core benches ssxb / cpyb / convb / recb / rescb / nrmb / zkb (.py + .cc vs verbatim C++);
  endrun.sh (depth builds + sweep + DAVIS for a suffix). Flags: EW_DEFS SSX, CPY64, CONVK, RECK, NRMK, ZRING (+ twin NORM_TWIN);
  kdefs RESCK; plan P60G=1 (query half-tile deal in the gathers, research rev 10 P60g). r8/ewprof.py has 14 phases.
- `npu-tapnext/r16/` (round 16): generators gen_sqk (exact sqrt, exponent folded into the table), gen_ggk (both SSM gate GEMMs,
  B fills every 8 pops), gen_ssq (ssx + sqk fused), gen_n1p / gen_n1k (norm pass 1 + loop-carried statistics, _begin / _step /
  _end), sig10.json (r12/gen_sig.py --umax 4); benches sqkb / ggkb (+ ggkref.py host bfp reference) / ssqb / n1pb / n1kb;
  popprobe.s + popb.py (FIFO pop probes); nrun.sh (python + NPU in npu-iron-torch), b16.sh, try16.sh (research try.sh + the
  current best's flags), commit.sh (commit with a message file). Flags: EW_DEFS SQK INPART_INLINE XONE GGK SSQ (replaces SSX)
  SIG10 N1K, NPROF (norm-step stamps; PROF no longer fits), twins NORM_TWIN=2 / 4.
- `npu-tapnext/r17/` (round 17): try17.sh (r16z base), xtry.sh (XROT=4 phase builds + regress / time / sweep), xrotsim.py (state
  flow of the rotation), cp_attks.sh (CP on testbox), attks.json (S2 schedule). Flags: mm8 ATTKS2MAX (gen_attkb --s2); EW_DEFS XROT
  (+ plan XROT=4), AL1 (+ plan AL1=1), NPIPE / NPIPE_G, CPROF (conv split stamps); NPROF stamps out_begin too (record word 19).
- `npu-tapnext/r18/` (round 18): try18.sh (current best's flags), ab.py (one-process A/B), dmab.py/.cc/.s (DMA-vs-core bank bench),
  ksp38.py (+ mm8 -DKSP38), gen_rot.py (research's ROT generator + ROT_ACQ_GAP / ROT_REL_GAP), mm_rot_g2a.s/.h (deployed ROT),
  mm8_rot.inc, rotbench.py, fsize.sh, convkp.json, zk.json. Flags: EW HRING PTRG ZK NIOC CGF NIO2 (norm-unit stamps); mm8 KSC
  (+ KS_* addresses) P35 MM_ROT (+ ROT_V env) V2T KSP38; pack8 B_ROT.
- `npu-tapnext/r19/` (round 19): try19.sh (current best: g0a + SEQPRE=1 + ZHI + CSPLIT + EPLR + KPREP, then csplit + regress /
  sweep), csplit.py / csplit.sh (lowered-MLIR patch + aiecc rebuild), tlsum.py (timeline per kind + k-step model), tprof.py (mm8
  -DTPROF tile stamps), wrapgen.py / wrapb.* (pop-wrap probes), build_rot.sh (mm_rot_g0a / g1a with the lock gaps), kcc.sh (mm8.cc
  alone with a build's kdefs + one function's disassembly), ddrcap.py (DDR rate probe), cbytes.py. Flags: mm8 ZHI CSPLIT (+ CS_Z)
  EPLR (+ CS_CONS) KPREP TPROF; design8 env SEQPRE (bits) CPRATE / CPRATE_POLL; design8 / pack8 take ROT variants from r19/.
- `npu-tapnext/r20/` (round 20): try20.sh (current best: r19z + IL DAPRE NSEQ NRES x4 SRES AQPRE AKVPRE, csplit20.sh with the
  NRESFIX route patch), tprof20.py / tpan.py / tpgap.py (TPROF records + core tile counter: first-tile pace, gaps), cpst.py / cpst2.py
  (mm8 -DCPST + design8 CPST_OP / CPST_ADDR), binpatch.py (.bin re-ordering), cpdump.py (CP instructions per sync segment), ddrw.py
  (DDR write cap), nresfix.py, poplf1gen.py / poplf1b.* (lf1 pop probe), atprof.py (mm8 -DATPROF attention stamps), actdiff.py (two
  builds' ACT images diffed by region / column / k-step). design8 env: IL DAPRE NSEQ NRES (list) SRES AQPRE AKVPRE EOUTPKT CPST_*.
- `npu-tapnext/r12/` (round 12): cpsched.py (CP-SAT scheduler), asmfit.py, lutflat.py (flat-edge GELU tables), generators
  gen_gup / gen_pan / gen_copy / gen_bgelu / gen_sig (+ saved schedules *.json) with one-core benches gupb / panb / cpyb / bgb /
  sigb (r11/apb.py `--cc`), gen_pushprobe + pushcheck. Flags: mm8 GUP, CPY32, PAN, GELU_FLAT, BGELU (+ timing GUP_NO*, PAN_NOPUSH);
  EW_DEFS SIGK (design8 `ew_object` then builds the EW object itself: ExternalFunction cannot link asm).
- `npu-tapnext/r10/` (round 10): gdist.py (per-tile medians of PROF records), rtm.py/.cc (runtime memtile task bench +
  memtile->core rate), pf*.mlir (router multicast bug), gx.cc / gxb.py (GELU loop bundles / one-core cycles), mmasm.sh
  (mm8.cc -> asm with the deployed flags). design8 / plan8 flags: A_HALF (plan: single A fills, reuse_a), A_RT (memtile
  A slot by runtime tasks), P23 (round-robin tiles for yx / qkv); builds record them in build/<name>.json and step8 /
  run8 apply them (`stale build` asserts otherwise). `PROF_OUT=name.npy` lands in build/.
- `npu-tapnext/r6/` (round 6 tools): asmstat.py, ccs.sh, tlview.py, kprobe.py/.cc, emu.py, capture.py, numh.py,
  lutgen.py; `npu-research/RESULTS_FROM_MAIN.md` holds the measured verdicts on the research proposals.
- `npu-research/PROPOSALS.md` (once it exists): experiments and tools proposed by a research agent; read each round.
  Rev 9 tools: `tools/attsim.py` (attention multicast-lockstep event sim), `tools/sim16.py` (block model on r13f with it),
  `r14/lazymap.py` (lazy-max flag map on the NPU layout, CPU container), `r14/attcal.py` / `attblk.py` (twin fit, per-block check).
  Rev 10: `tools/attsim17.py` + `tools/sim17.py` (r14f, kappa 0.6), `r15/deal4.py` (P60g gather deal, checks single-TAP both
  sides), `r15/slots.py` (per-slot op counts of the zero-overhead loops in a -z disassembly), `r15/prereg17.py`.
  Rev 11: `tools/asmlint.py` (+ `--zolstats`, `--calib`), `tools/asmlint_hist.py` + `r16/hist/` (historic bad / fixed kernels),
  `tools/try.sh` (one-command experiment), `tools/asmfit_tmp.patch` (build lock), `tools/sim18.py` (attention levers on r15m).
  Rev 12: `tools/tcall.py` (whole-call static cycles + per-step census from a `-z` disassembly), `tools/ewio.py` (EW ring I/O model).
  Rev 13: `ewio.py fit13 / levers13` (the refit on round-17 stamps; the rev-12 `fit` kept), `tools/ksplan.py` (GEMM k-step budget,
  variants, collision model), `r18/` (r17d / r18a disassemblies, fit and lever outputs).
  Pass 14 (`r19/`): `gen_rot.py` (ROT: CP-SAT `solve [--free] [--place ..]`, `fit` (emit + prologue alignment, npu-iron),
  `check [--place ..]` (whole-call value-level replay of any emitted .s on any L1 placement: correctness, bank stalls, busy per
  bank)), mm_rot_{g2f,g1f,g0f,g0u}.s / .h / .dis + rot_*.json, `build_all.sh`, `price.py`, `rotb.py` / `rotb.cc` (one-core ks vs rot
  bench), `mm8_rot.inc` (the mm8.cc side: constexpr records under KSC).
  Pass 15 (`r20/`): `gen_rot.py` (lock gaps ACQ_GAP 6 / REL_GAP 8 as hard floors, replay asserts the acquire -> A fill distance,
  per-cycle bank trace), `collide.py` (time-resolved DMA-collision model, `--scan`), `tilefit.py` (per-tile period / overhead per
  GEMM kind from timelines), `cpseq.py` (CP instructions on each op boundary's critical path, now vs SEQPRE), `rotdb.py` / `rotdb.cc`
  (one-core k-step bench with a DMA-FED A ring: IRON Buffers + locks 4 / 5 + a shim -> core flow, so acquires really wait).
  asmlint rev 15: LOCK-ACQ-GAP, LOCK-REL-GAP, IMM-SRC (immediates checked on the .s source; the hist suite has 23 cases).
  Pass 16 (`r21/`): `tlhist21.py` (per-op gap means + poll-quantum histograms), `bexcess.py` (per-op excess over steady tiles),
  `ddrbudget.py` (DDR bytes per op from plan8 TAPs vs the timeline), `cpsig.py` / `cpdump.py` / `bduse.py` (.bin segment
  signatures, instruction dumps, shim BD usage; npu-iron), `bndmodel.py` (boundary decomposition against main's TPROF npz).
- `npu-tapnext/` design8 (round 3, the fused step): `plan8.py` (memory plan + op list: Q16 / APAN / QKV layouts,
  deps), `design8.py` (the frame: table-driven GEMM cores, attention op, EW ops, BD accounting), `mm8.cc` (GEMM +
  flag-driven epilogues + attention), `ew8.cc` (norm / patch / ssm on the row-3 cores), `nk8.h` (exp32, recip,
  sigmoid, gelu, sqrt), `pack8.py` (host packing + numpy references), `run8.py` (tests t1..t5, g11, full12*),
  `attn_dbg.py`; builds `build/t*.xclbin`, `g11*.xclbin`, `full12r.xclbin` (the whole 12-block step).
- `npu-tapnext/` design7 (round 2): NOTES.md, RESULTS.md, HANDOFF.md (the plan), **`design7.py` + `mm7.cc` + `ew7.cc` + `pack7.py` +
  `run7.py`** (the fused-step frame: 3 GEMM rows + an elementwise core per column, packet-shared shim channel,
  hand-built rings, `encode_hw`), `pkt_test.py` (packet/BD-iteration probes), `xfer*.py` (transfer costs),
  `davis_cpu.py`, `design6.py` (+ `RA_K` env), `attn_tn.*` (attention for 12 heads x 1072 x 64), `npu_step.py`
  (the old prototype), `davis_npu.py`, `build/` (s7_*.xclbin, op_nt*.xclbin, attn_tn.xclbin, tn_gemm_A).
- `tapnet/`: the TAPNext++ exploration (tnpp.py, davis.py + TAP-Vid DAVIS data, bench/prof scripts, RESULTS.md).
- Memory notes: `npu-kernel-learnings`, `npu-moondream`, `tapnext-tracker`, `strix-box-facts`.
