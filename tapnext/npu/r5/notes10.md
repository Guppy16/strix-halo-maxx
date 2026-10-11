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
