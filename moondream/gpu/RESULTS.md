# moondream2 latency optimisation on Strix Halo (gfx1151) -- results log

Box: Radeon 8060S (gfx1151), torch 2.14.0+rocm7.2, transformers 4.57.6, moondream2 rev 2025-06-21, bf16.
Test set: 493 camera frames (720x1280) from a private set of camera frames (not included; `$MD_FRAMES/*/judged.jsonl`), tasks mug (91 frames, 7 found) and tv (402 frames, 46 found).
Question: "Is there a <obj> in this image? Answer yes or no."  Label yes := VLM-judge status "found".
Latency: median over the whole test set (a different frame every run), torch.cuda.synchronize() around encode and query.

## Baseline (stock HF bf16, 378x378 squash, model.encode_image + model.query temp=0)
tv: acc=97.3% n=402 tp=46 fn=0 fp=11 tn=345 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.8% n=493
median encode 239.3 ms, query 99.1 ms, total 338.4 ms
raw answers: {'no': 424, 'yes': 52, 'Yes': 12, 'No': 5}; sanity 0060 mug -> yes, 0000 -> no (verified in verify.py below)
Profile (stock): encode = 3068 kernel launches, 238 ms GPU; 78% in aten::addmm (Tensile MT128x128x32 bf16 kernels), attention 35 ms.
query = 4400 launches, 94 ms GPU, 87% addmm: 98 GEMMs at M~14 (prompt prefill) run 400-560 us each (<1 TFLOPS), 96 GEMMs at M=1 (one decode step) 160-400 us each (~85 GB/s) -> the small-M GEMM kernels are terrible.
Accuracy gate for every experiment: >= 95.8% overall (baseline - 2pp) and sanity images correct.

## fast-bf16-eager
config: dtype=bf16 graph=False lin=Lin 
tv: acc=97.0% n=402 tp=46 fn=0 fp=12 tn=344 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.6% n=493 | median encode 201.4 ms, query 49.8 ms, total 251.3 ms | agree w/ baseline 99.4% (n=493) | emb cos min 0.99860 | sanity 0060=True 0000=False

## fast-bf16-graph
config: dtype=bf16 graph=True lin=Lin 
tv: acc=97.0% n=402 tp=46 fn=0 fp=12 tn=344 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.6% n=493 | median encode 196.9 ms, query 50.7 ms, total 247.7 ms | agree w/ baseline 99.4% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-fp16-graph
config: dtype=fp16 graph=True lin=Lin blas=default tunable=False compile= 
tv: acc=95.3% n=402 tp=46 fn=0 fp=19 tn=337 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=96.1% n=493 | median encode 165.1 ms, query 37.8 ms, total 202.9 ms | agree w/ baseline 98.4% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-bf16-tunable-eager
config: dtype=bf16 graph=False lin=Lin tlin= blas=default tunable=True compile= pool=0 
tv: acc=98.0% n=150 tp=10 fn=0 fp=3 tn=137 | all: acc=98.0% n=150 | median encode 152.0 ms, query 46.4 ms, total 198.4 ms | agree w/ baseline 99.3% (n=150) | emb cos min 0.99933 | sanity 0060=True 0000=False

## Notes on the fast path (md2fast.py) -- KEEP
Re-implemented single-tile path: vision encoder on B=1 (global crop == local crop for a 378x378 input, verified bit-identical in image_crops.py),
730-token image prefill without mask over exactly 730 keys (flash attention instead of masked efficient-attention over a 2048 KV cache),
no KV-cache clone/reload, question answered from the yes/no logits of a single 32-token padded prompt prefill (no decode loop),
static shapes captured in HIP graphs. verify.py: top-1 token identical to stock on all checked frames, logits cos>0.9997, img_emb cos>0.999.
bf16: encode 201 -> 197 ms (graph), query 99 -> 50 ms. Graphs give ~2% because the pipeline is GPU-bound on GEMM kernels.
Accuracy 97.6% (one extra tv false positive vs stock; agreement 99.4%).

## GEMM microbenchmarks (gemm_bench.py; hot-cache, repeated same shape -- in-situ numbers are worse)
Big shapes (M=729/730): rocBLAS bf16 19-31 TFLOPS (text fc2 K=8192 only 11.9), rocBLAS fp16 up to 21-31, hipBLASLt similar, TunableOp best-of 21-32,
inductor Triton 23-30. Nothing gets above ~32 TFLOPS on this GPU (theoretical WMMA peak ~57 TFLOPS at the observed 2.8 GHz).
Small shapes (M=14/32, query prefill): rocBLAS 1.4-2.2 TFLOPS / ~100-160 GB/s; TunableOp (hipBLASLt) 2.3-5 TFLOPS in the microbench,
but in situ (cold weights) the same kernels take 2.5x longer (359 us for 2048x8192, ~93 GB/s) -> weight streaming bound at poor bandwidth utilisation.
torch._int_mm works on gfx1151 but runs at bf16 speed (28-32 TOPS, no int8 WMMA benefit); torch._scaled_mm (fp8) is unsupported on this arch.
Hand-written Triton int8 GEMM (tl.dot int8->int32) reaches 34-38 TOPS vs 25-28 for Triton bf16 -> int8 WMMA is used but only ~1.4x.

## fast-fp16-graph: encode 165 / query 38 / total 203 ms, acc 96.1% (tv fp 19 vs 11) -- faster rocBLAS fp16 kernels but numerics drift; borderline on the gate -> not chosen.
## fast-bf16-tunable-eager (150 frames): encode 152 / query 46 / total 198 ms, acc 98.0% on subset -- TunableOp + HIP graph capture DEADLOCKS (process hangs in futex, GPU idle) so tunable can only be used eagerly.

## fast-bf16-triton-graph
config: dtype=bf16 graph=True lin=TLin tlin= blas=default tunable=False compile= pool=0 
tv: acc=97.3% n=402 tp=46 fn=0 fp=11 tn=345 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.8% n=493 | median encode 116.8 ms, query 19.2 ms, total 136.0 ms | agree w/ baseline 99.6% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-bf16-vis-rocblas-txt-int8-graph
config: dtype=bf16 graph=True lin=Lin tlin=I8Lin blas=default tunable=False compile= pool=0 
tv: acc=96.8% n=402 tp=46 fn=0 fp=13 tn=343 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.4% n=493 | median encode 106.5 ms, query 13.1 ms, total 119.6 ms | agree w/ baseline 98.8% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-bf16-vis-triton-txt-int8-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=0 
tv: acc=97.3% n=402 tp=46 fn=0 fp=11 tn=345 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.8% n=493 | median encode 103.5 ms, query 12.0 ms, total 115.6 ms | agree w/ baseline 99.2% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-all-int8-graph
config: dtype=bf16 graph=True lin=I8Lin tlin=I8Lin blas=default tunable=False compile= pool=0 
tv: acc=86.8% n=402 tp=46 fn=0 fp=53 tn=303 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=89.2% n=493 | median encode 101.5 ms, query 11.5 ms, total 113.0 ms | agree w/ baseline 91.5% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-int8txt-pool19-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=19 fuse=True 
tv: acc=97.8% n=402 tp=42 fn=4 fp=5 tn=351 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=98.2% n=493 | median encode 63.8 ms, query 9.7 ms, total 73.5 ms | agree w/ baseline 98.0% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fast-int8txt-pool14-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=14 fuse=True 
tv: acc=97.8% n=402 tp=41 fn=5 fp=4 tn=352 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=98.2% n=493 | median encode 52.0 ms, query 9.5 ms, total 61.5 ms | agree w/ baseline 97.6% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## Triton GEMMs (tgemm.py) -- KEEP
Hand-written Triton GEMM (autotuned tiles, fused bias / gelu / residual epilogues). bf16 in situ: encode GEMM time 172 -> 94 ms, query GEMM 42 -> 15 ms.
fast-bf16-triton-graph: encode 116.8 / query 19.2 / total 136.0 ms, acc 97.8% (== baseline), agreement 99.6%.
W8A8 int8 (per-channel weights, per-token dynamic activations, int8 WMMA via tl.dot) for the TEXT model only:
fast-bf16-vis-triton-txt-int8-graph: encode 103.5 / query 12.0 / total 115.6 ms, acc 97.8%, agreement 99.2%.  <- best exact-architecture config so far
int8 for the VISION encoder too (fast-all-int8-graph): 113 ms but accuracy collapses to 89.2% (53 tv false positives) -> DISCARD; SigLIP activations do not survive per-token int8.
torch.compile (inductor default) around the fast path: no gain (106 vs 116 ms eager on 100 frames, and it cannot be combined with TunableOp) -> DISCARD.
Bandwidth (bw.py): device copy ~200 GB/s, read ~224 GB/s; 32 MB working sets read at 460 GB/s (MALL) which is why hot-cache microbenchmarks overstate small-M GEMM speed.

## Image-token pooling (workload shaping) -- the biggest single lever, but changes the model input
The 729 projected image tokens are average-pooled (27x27 -> kxk) before the LLM prefill (the vision encoder still runs at full 378px).
pool 19 (361 tokens): encode 63.8 / query 9.7 / total 73.5 ms, acc 98.2% (tv tp=42 fn=4 fp=5), agreement 98.0%
pool 14 (196 tokens): encode 52.0 / query 9.5 / total 61.5 ms, acc 98.2% (tv tp=41 fn=5 fp=4), agreement 97.6%
Both pass the gate; they trade the baseline's false positives for a few false negatives (4-5 of 46 found frames missed) -> report as an option, not the default.

## fused-int8txt-pool0-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=0 fuse=True 
tv: acc=96.3% n=402 tp=46 fn=0 fp=15 tn=341 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.0% n=493 | median encode 97.0 ms, query 10.2 ms, total 107.2 ms | agree w/ baseline 98.4% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fused-int8txt-pool23-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=23 fuse=True 
tv: acc=97.5% n=402 tp=42 fn=4 fp=6 tn=350 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=98.0% n=493 | median encode 77.6 ms, query 9.3 ms, total 86.9 ms | agree w/ baseline 97.8% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fused-int8txt-pool16-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=16 fuse=True fold=False fusedres=True nq=1 
tv: acc=97.8% n=402 tp=42 fn=4 fp=5 tn=351 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=98.2% n=493 | median encode 57.9 ms, query 8.6 ms, total 66.4 ms | agree w/ baseline 98.0% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fused-int8txt-pool12-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=12 fuse=True fold=False fusedres=True nq=1 
tv: acc=98.8% n=402 tp=42 fn=4 fp=1 tn=355 | mug: acc=97.8% n=91 tp=5 fn=2 fp=0 tn=84 | all: acc=98.6% n=493 | median encode 47.2 ms, query 8.4 ms, total 55.6 ms | agree w/ baseline 96.8% (n=493) | emb cos min nan | sanity 0060=False 0000=False

## fused-int8txt-pool9-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=9 fuse=True fold=False fusedres=True nq=1 
tv: acc=97.5% n=402 tp=40 fn=6 fp=4 tn=352 | mug: acc=96.7% n=91 tp=7 fn=0 fp=3 tn=81 | all: acc=97.4% n=493 | median encode 42.6 ms, query 8.4 ms, total 51.0 ms | agree w/ baseline 96.8% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fused-nofold-int8txt-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=0 fuse=True fold=False fusedres=True nq=1 
tv: acc=96.3% n=402 tp=46 fn=0 fp=15 tn=341 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.0% n=493 | median encode 97.5 ms, query 9.2 ms, total 106.7 ms | agree w/ baseline 98.8% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## fused-nofold-nofusedres-int8txt-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=0 fuse=True fold=False fusedres=False nq=1 
tv: acc=97.5% n=402 tp=46 fn=0 fp=10 tn=346 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=98.0% n=493 | median encode 95.4 ms, query 9.4 ms, total 104.8 ms | agree w/ baseline 99.4% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## Kernel fusion round (fused.py, tgemm.py epilogues)
* fused rotary + KV-cache write Triton kernel (one program per token, all heads): replaces ~10 small aten kernels per layer (split/rope mul-add/stack/cat/index_put), ~1 ms total in encode.
* yes/no-only lm_head (8 rows instead of 51200): query -1 ms.
* fused residual add in the GEMM epilogue: SLOWER (autotuner picks worse tiles with the extra load) and drifts numerically (agreement 98.8% vs 99.4%) -> DISCARD (default off).
* folding the two K-halves of the projection fc1 (input is cat([x,x])): exact in real arithmetic but bf16 rounding of the summed weight changes decisions (agreement 98.4%) -> DISCARD (default off).
* wider Triton autotune space (BK=128/256, num_stages 1-3, waves_per_eu): no gain. (N,K) weight layout with in-register transpose: 8-25% slower than the (K,N) layout in use.
Result of the round: fused-nofold-nofusedres-int8txt-graph: encode 95.4 / query 9.4 / total 104.8 ms, acc 98.0%, agreement 99.4%.  <- DEFAULT of bench.py
In-situ breakdown of the 95 ms encode: int8 text-prefill GEMMs 56 ms (31 TOPS), bf16 vision GEMMs 22 ms (~28 TFLOPS), flash attention 10 ms, quant/LN/GELU/adds ~6 ms.

## Pooling sweep (fused build): pool23 86.9 ms / 98.0% / agree 97.8%; pool16 66.4 ms / 98.2% / 98.0%; pool14 61.5 ms / 98.2% / 97.6%;
pool12 55.6 ms FAILS the sanity gate (0060 mug -> no, mug fn=2); pool9 51.0 ms / 97.4% (mug fp=3). -> pool16 is the recommended fast mode.

## Batched questions: n_q=2 (mug + tv in one prompt prefill, shared image KV): query 9.5 -> 11.2 ms, i.e. a second question costs ~2 ms instead of ~10.

## final-sharedquant-int8txt-graph
config: dtype=bf16 graph=True lin=TLin tlin=I8Lin blas=default tunable=False compile= pool=0 fuse=True fold=False fusedres=True nq=1 
tv: acc=96.3% n=402 tp=46 fn=0 fp=15 tn=341 | mug: acc=100.0% n=91 tp=7 fn=0 fp=0 tn=84 | all: acc=97.0% n=493 | median encode 97.4 ms, query 9.2 ms, total 106.5 ms | agree w/ baseline 98.8% (n=493) | emb cos min nan | sanity 0060=True 0000=False

## Reproducibility / noise
Same code, same process: repeated runs are bit-identical (max logit diff 0.0). Across processes the Triton autotuner may pick different tiles
for the bf16 vision GEMMs -> different fp32 accumulation order -> 48 of 493 frames have |yes-no margin| < 1 and a few flip,
which is why identical configs scored 97.0-98.0% in different runs. Fixed by `cache_results=True` on the autotuners (choices persisted in the Triton cache).
Sharing the int8 activation quantisation between qkv and fc1 (SHARE_QUANT): bitwise identical, -24 kernels per encode.

## Preprocessing (outside the timed region, but part of frame-to-answer latency)
PIL LANCZOS squash of a 720x1280 JPEG to 378x378 costs ~8 ms on this CPU. Decoding the JPEG at 1/2 scale first (`Image.draft`, DCT-domain)
then LANCZOS is BIT-IDENTICAL on all 99 checked frames and ~2.5x faster -> adopted in common.prep(). A GPU resize would differ from LANCZOS numerically (not adopted).

## Toolchain limits
No hipcc/amdclang on the box (torch's ROCm wheels only ship runtime libraries), so custom kernels are Triton-only; rocBLAS/hipBLASLt/CK cannot be rebuilt or tuned offline.
torch._scaled_mm (fp8) unsupported on gfx1151; torch._int_mm exists but is not faster than bf16; TunableOp + HIP graph capture deadlocks.

## FINAL (bench.py, full 493-frame test set, same process; Triton autotune results cached)
| config | encode | query | total | accuracy | agreement w/ stock |
|---|---|---|---|---|---|
| stock HF bf16 | 245.4 | 99.6 | 344.9 ms | 97.8% (tv fp=11) | - |
| fast exact (TLin vision + I8Lin text, fused, graphs) | 95.1 | 9.3 | 104.4 ms | 98.0% (tv fp=10) | 99.4% |
| fast + pool 16 | 58.1 | 8.7 | 66.8 ms | 98.4% (tv fn=3 fp=5) | 98.2% |
Sanity gate passes for both (judged_0060 mug=yes, judged_0000 mug=no).

What mattered most (exact path, 345 -> 104 ms):
1. Re-implementing the pipeline with static shapes: single vision crop (-40 ms), no decode loop / yes-no logits (-50 ms), no masked 2048-wide attention, no cache cloning.
2. Replacing rocBLAS/Tensile GEMMs with Triton kernels (+fused bias/gelu) -> encode 197 -> 117 ms, query 50 -> 19 ms.
3. int8 W8A8 (int8 WMMA via Triton) for the 1.2B-param text model only -> 104 ms; vision int8 breaks accuracy.
4. Fused rope+KV-cache kernel, yes/no-only lm_head, shared quantisation: a few ms.
5. HIP graphs: only ~2-4% here (GPU-bound), but make the CPU side free.
What did not work: fp16 (faster rocBLAS kernels, numerics drift to 96.1%), hipBLASLt, TunableOp (deadlocks with graphs; eager only 198 ms),
torch.compile/inductor (no gain), torch._int_mm (bf16 speed), fp8 (unsupported), int8 vision (89%), fused residual epilogue (slower + drift),
folded projection weights (drift), wider autotune space / (N,K) weight layout (no gain), pool<=12 (fails sanity gate).
To go further: the text prefill (730 tokens x 1.2B params, 1.76 TOP) is 56 of the 95 encode ms at ~31 TOPS int8 -- a better int8 WMMA kernel
(hand-written HIP/CK, needs a HIP compiler which the box lacks) could maybe reach 50-60 TOPS; int4 weights would only help the query;
otherwise fewer LLM tokens (pooling, or a moondream build fine-tuned for fewer tokens) is the remaining big lever.

# ===== Round 3 (2026-10-01/02): hand-written HIP kernels, ROCm 7.14 (host venv-rocm714 = torch 2.14.1+rocm7.14, HIP 7.14.60850) =====
Compiler: Docker image rocm/dev-ubuntu-24.04:7.14.1-full (same HIP build as the host torch wheel). Sources in hip/ (i8gemm.hip = all kernels,
i8test.cpp / bf16test.cpp = C++ harnesses, hblt_i8.cpp = hipBLASLt probe, probe_layout.cpp = WMMA lane-layout probe), Python side hgemm.py, test_hgemm.py.
Build (inside the container, see README): hipcc -O3 --offload-arch=gfx1151 -fPIC -shared i8gemm.hip -o libmdgemm.so
The .so only needs libamdhip64.so.7, which torch's rocm wheel (_rocm_sdk_core) already loads -> ctypes-loaded in the HOST venv-rocm714 and HIP-graph captured fine.

## Hardware facts measured on this gfx1151 (hip/i8gemm.hip wmma_peak_* probes, register-only WMMA loops)
* int8 WMMA (v_wmma_i32_16x16x16_iu8) peak = 57 TOPS; bf16 WMMA (v_wmma_f32_16x16x16_bf16) peak = 55 TFLOPS -> int8 is NOT 2x bf16 on this chip
  (same issue rate, 512 ops/clk/CU at 2.9 GHz x 40 CU). int8 only halves operand bandwidth / LDS traffic. The "2x int8 peak" assumption was wrong.
* WMMA wave32 layout (verified): A/B operand lane l holds row (l%16) with 16 consecutive K bytes (lanes 16-31 duplicate 0-15); accumulator element e of
  lane l is C[e*2 + l/16][l%16]. So W in the natural nn.Linear (N,K) layout and A (M,K) row-major are directly the operand layouts (no transpose needed).

## hipBLASLt int8 on gfx1151 (hip/hblt_i8.cpp, all algos enumerated via hipblaslt_ext::getAllAlgos, best of each)
Only 6 (i32 out) / 8 (i8 out) int8 Tensile kernels exist for gfx1151; no fp16/bf16/fp32 output for int8 inputs, no OUTER_VEC_32F (per-row/col) scaling.
730x2048x6144: i32-out 33 TOPS; 730x2048x2048: 33 / i8-out 39; 730x2048x8192: 35 / 37; 730x8192x2048: 34 / 41; M=32: 9-16 TOPS.
bf16 hipBLASLt best-of-341 algos: 730x2048x6144 41.3 TFLOPS (MT64x96x32), 730x2048x2048 41.4, 730x2048x8192 31.7, 730x8192x2048 35.9, M=32 15-17.
-> hipBLASLt int8 is no better than the Triton int8 (and needs a separate dequant pass) -> DISCARD. (hipBLASLt bf16 is good on N=6144 but
   on the vision shapes torch's default Tensile path already gives the same ~29-34 TFLOPS as my kernel -> not pursued.)

## HIP int8 W8A8 GEMM kernel (hip/i8gemm.hip i8gemm2_kernel): C[M,N] bf16 = (A[M,K] i8 * W[N,K]^T i8) * sa[M] * sw[N] + bias (+gelu)
Structure: 16-B global loads -> registers -> padded LDS (row stride BK+16 B -> conflict-free ds_load_b128) -> WMMA fragments; register prefetch of the
next K tile during compute; configurable LDS buffering / prefetch distance; grid order "swap" (M tiles fastest so consecutive WGs share a W tile in L2).
Microbench (C++ harness, hipEvent timing, 20 reps, container), best config = cfg17: 128x128 tile, BK=128, 8 waves of 32x64, single LDS buffer (36 KB):
  730x2048x6144 39.3 TOPS | 730x2048x2048 35.0 | 730x2048x8192 40.6 | 730x8192x2048 37.0   (Triton _gemm_i8 in situ ~31, hot microbench 27-32)
  L2-resident control (W = 2 MB): 41-43 TOPS -> the kernel is core-bound (barriers / LDS latency), not memory-bound; ~70% of the 57 TOPS WMMA peak,
  which is also where hipBLASLt's best bf16 kernels land (41 TFLOPS). BK=32 configs are 2x slower (48-B LDS stride conflicts); s_setprio no gain;
  double-buffered LDS only helps at BK=64 (40 KB -> 1 WG/CU); 64x128 / 128x256 / 256x128 tiles all worse.
bf16 variant (bf16gemm_kernel, same structure, K-tail at 8-element granularity, N-tail masked): vision shapes 729x1152x3456 36.4 TFLOPS,
  729x1152x1152 36.8, 729x1152x4304 33.4, 729x4304x1152 36.8, 729x2304x8192 33.2, 729x8192x2048 23.4 (Triton TLin 26-30, rocBLAS 18-34).
Small-M (query, M=32/64) first attempt (i8gemm_small_kernel: direct global->register fragments, split-K, int32 partials + epilogue kernel):
  only 57-167 GB/s of weight streaming (uncoalesced 16-B fragment loads: 16 cache lines per instruction) -> worse than Triton on K=8192; replaced by
  the LDS-staged split-K variant of the main kernel (cfg 3-6 of i8gemm_small_launch), see below.

## test_hgemm.py on the HOST venv-rocm714 (python-level timing incl. launch+sync overhead; gpu_busy 13% from other users at the time)
int8: HIP == Triton bit-identical on all 9 shapes (same int32 sums, same epilogue), rel err vs bf16 F.linear 0.8-1.2% (= the quantisation itself).
  730x2048x6144: torch 595 / triton-i8 628 / HIP-i8 551 us; x2048: 200 / 226 / 219; x8192: 874 / 824 / 673; 730x8192x2048: 1237 / 768 / 789.
bf16: HIP == Triton bit-identical (same fp32 WMMA accumulation order), 729x1152x3456: torch 197 / triton 221 / HIP 195 us; 729x1152x1152: 107/105/80;
  x4304: 246/257/251; 729x4304x1152: 249/252/223; 729x2304x8192: 803/975/878; 729x8192x2048: 1185/800/1051 (K=8192 is the weak shape of my kernel).
HIP graph capture of the ctypes launches: replay == eager bit-exact.

## In situ the first HIP build was SLOWER than Triton (bench 150 frames: HLin/HI8Lin 97.2/20.5/117.7 ms vs TLin/I8Lin 88.3/9.2/97.6) -- root cause
insitu_time.py (CUDA events around every linear, eager): HI8Lin 730x8192x2048 = 1022 us in situ vs 644 us in the hot microbench; M=32 shapes
3.5-5 TOPS in situ vs Triton 9-12. coldbench.py (graph-captured loop over 8 copies of the weights = 96-128 MB, i.e. more than the 32 MB MALL,
like the real pipeline where 1.2 GB of int8 weights stream through per frame) reproduces it exactly:
  730x8192x2048: HIP 1166-1298 us cold vs 673-688 hot (Triton 677 cold == hot); 32x8192x2048: HIP 268-449 cold vs 75-158 hot (Triton 78 cold).
  Prefetch distance 2 (cfg 11/23) does not help -> not a latency problem. L2-resident test earlier showed the kernel itself is fine.
Cause: the (N,K) weight layout with K = 2048/8192 bytes per row -> the chunk at a given k offset of every row maps to the same DRAM channel/bank
(power-of-two stride), so the concurrent WGs (all at similar k) serialise on one channel; Triton's (K,N) layout walks N-contiguous rows instead.
MALL-hot microbenchmarks hide this completely -> ALL earlier "TOPS" numbers for weight-streaming kernels on this box must be taken cold.
Fix 1 (HGEMM_WPAD=64): pad every int8 weight row by 64 bytes (row stride K+64): 730x2048x6144 486 us cold (Triton 545), 730x2048x8192 635 (753),
  730x2048x2048 177 (171), 730x8192x2048 739 (669); M=32: 73-97 us (Triton 60-126). Activations (quant_rows output) padded the same way -> 730x8192x2048 687.
Fix 2 (HGEMM_PACK=1, default): tiled weight layout [N/128][K/TK][128][TK] (TK=128 int8 / 64 bf16) so every WG streams one contiguous 16 KB block per
  K step; same cold numbers as the padding (both remove the aliasing), no padding needed, used for int8 and bf16.
Fix 3 (small M): split-K only while the grid has < 48 WGs (I8_SPLIT_MIN), otherwise the plain kernel with fused epilogue (no int32 partial round trip,
  no second launch): M=32 cold: 2048x6144 61 us (Triton 60), 2048x2048 18 (19), 2048x8192 78-83 (125), 8192x2048 83 (85).
In situ after the fixes (insitu_time.py, eager, per frame): text int8 GEMMs 48.4 ms (Triton 58.1), vision bf16 GEMMs 20.7 ms (Triton 23.6),
query GEMMs 7.8 ms (Triton 7.4); total GEMM time 78.0 ms vs 87.2 ms.
bench.py 150 frames, same session: HLin/HI8Lin 82.5 / 9.0 / 91.5 ms vs TLin/I8Lin 88.3 / 9.2 / 97.6 ms (-6%); acc/agreement unchanged.
Still open: N=2048 shapes (proj, fc2) only reach 31-34 TOPS because 96 work-groups on 40 CUs leave the last wave 60% idle (a stream-K style
persistent kernel would fix the tail); the WMMA core ceiling of this kernel structure is ~43 TOPS (barriers + LDS->fragment latency).
verify.py HLin HI8Lin: top-1 token identical to stock on all 8 checked frames (eager and graph), logits cos 0.994-0.999, img_emb cos > 0.9989.
Tail-effect configs for the N=2048 shapes (cold bench, cfg27 vs 128x64 / 64x128 tiles with 2 WGs per CU): cfg35 (128x64) 186/733 us, cfg36 (128x64, 64x32 waves)
  179/691, cfg37 (64x128) 175/699, cfg38 (64x128, 32x32 waves) 188/740 vs cfg27 177/681 us (730x2048x2048 / 730x8192x2048): the smaller tiles lose more
  per-WG efficiency than they gain in balance -> cfg27 (128x128x128, 8 waves of 32x64) stays the single config for all M=730 shapes. DISCARD.
bf16 fused GELU: the Triton path applies F.gelu to the bf16-rounded fc1 output; the HIP epilogue now rounds to bf16 before the tanh-GELU, so
  HLin(x, gelu=True) == F.gelu(TLin(x)) bit-exactly on all 6 vision shapes (test_hgemm.py). (First 493-frame run before this change: 97.6%, 2 tv flips.)

## FINAL round 3 (bench.py default = HLin/HI8Lin, full 493-frame test set; run_docker_vs_host.sh, logs_dvh_host.txt / logs_dvh_docker.txt)
NOTE on the environment: during this run another GPU process (an LLM server) kept gpu_busy_percent at 93-100%; the earlier 150-frame runs were at 0-5%. Host and container were measured
back-to-back under the same load, so the comparison holds; absolute numbers may be a few % pessimistic (150-frame run at idle GPU: 82.5 / 9.0 / 91.5).
| where | code | accuracy | agreement w/ stock | encode | query | total | throughput nq=1 | throughput nq=2 |
|---|---|---|---|---|---|---|---|---|
| host venv-rocm714 (torch 2.14.1+rocm7.14, HIP 7.14.60850) | HLin/HI8Lin | 98.0% (tv fp=10, mug 100%) | 99.4% | 79.6 ms | 9.0 ms | 88.7 ms | 10.80-10.94 frames/s | 10.59-10.68 frames/s = 21.2-21.4 answers/s |
| docker rocm/dev-ubuntu-24.04:7.14.1-full, venv-c714 (same torch wheel, python 3.12) | same | 98.0% (identical) | 99.4% | 79.4 ms | 9.0 ms | 88.4 ms | 10.51-10.88 frames/s | 10.55-10.61 frames/s = 21.1-21.2 answers/s |
Docker vs host gap: < 1% on every metric (same HIP build, same kernels; the container adds nothing on the GPU path -- seccomp=unconfined, /dev/kfd + /dev/dri passthrough).
Sanity gate passes in both. CPU prep (JPEG decode + LANCZOS) 8.1-8.5 ms/frame outside the loop (slower than the 3 ms measured earlier: CPU shared with another process).
Previous best (Triton GEMMs, same session, idle GPU, 150 frames): 88.3 / 9.2 / 97.6 ms; stored FINAL of round 2: 95.1 / 9.3 / 104.4 ms (493 frames).
=> HIP WMMA GEMMs: encode -9%, total -6..-9%, numerics unchanged (bit-identical GEMM outputs to the Triton path, same 98.0% / 99.4% agreement).

## Flash-attention prefill kernel (hip/attn.hip, hattn.py; bench.py --hattn 1) -- marginal, OPTIONAL (default off)
The two mask-free prefills (vision: 27 layers x 16 heads x 729 x hd 72; text: 24 x 32 x 730 x hd 64) cost ~9.5 ms per frame in aotriton SDPA
(~18 TFLOPS). Hand-written WMMA bf16 flash attention, 64-key blocks staged in LDS (K natural layout, V transposed by a small HIP kernel into a
zero-padded (H, DP, TP) buffer, hd 72 padded to 80), online softmax, output written directly in the (T, H*hd) layout (no transpose/copy kernel).
v1 (S = Q K^T, P transposed through LDS): text 263 / vision 182 us per call vs SDPA 231-240 / 166-175 -> slower.
v2 (S^T = K Q^T: each lane then holds 8 keys of ONE query, so the row softmax is in-lane + one half-wave exchange, and the P fragment for P*V is
   assembled in registers with 4 shuffles per tile): text 200 / vision 128 us (kernel) + 25 us V transpose = 225 / 152 us -> 7-9% faster than SDPA.
   Numerics: rel error vs fp32 SDPA 0.0022 = identical to torch's bf16 SDPA (not bit-identical to it; yes/no logits move by ~0.3).
v3 (32 query rows per wave to halve LDS fragment traffic): 256 VGPRs + 217 spilled -> 346 / 224 us, DISCARD (would need Q in LDS).
In situ (ab_attn.py, same process, alternating graph replays, 40 reps): encode min 78.6 vs 79.1 ms, median 82.4 vs 83.0 ms (sdpa) -> -0.5 ms (<1%),
within the noise of the shared GPU; 150-frame bench flipped 2 borderline tv frames (97.3 vs 98.7% on that subset). -> left OFF by default.

## ROUND 3 SUMMARY (hand-written HIP kernels; host venv-rocm714 = torch 2.14.1+rocm7.14)
| config | encode | query | total | accuracy (493) | agreement | frames/s (nq=1 / nq=2) |
|---|---|---|---|---|---|---|
| stock HF bf16 | 223-245 | 100 | ~340 ms | 97.8% | - | - |
| round-2 best: Triton GEMMs (`bench.py --vlin TLin --tlin I8Lin`) | 88-95 | 9.2 | 97.6-104 ms | 98.0% | 99.4% | - |
| round-3 best: HIP WMMA GEMMs (`bench.py`, default) -- host | 79.6 | 9.0 | 88.7 ms | 98.0% | 99.4% | 10.9 / 10.6 (21.3 answers/s) |
| same code inside the ROCm 7.14 container | 79.4 | 9.0 | 88.4 ms | 98.0% | 99.4% | 10.9 / 10.6 |
What mattered: (1) a 128x128x128 int8 WMMA GEMM with register prefetch + padded single LDS buffer + fused dequant/bias/gelu epilogue (~40 TOPS cold
vs Triton's 31-33); (2) discovering that the (N,K) weight layout with power-of-two K aliases DRAM channels -> tiled weight layout (fixed a 2x in-situ
slowdown that MALL-hot microbenchmarks could not see); (3) split-K only for tiny grids, plain fused kernel otherwise (query GEMMs == Triton);
(4) bf16 variant for the vision encoder (+10%), GELU rounded like the Triton path so the pipeline numerics are bit-identical to round 2.
What did not: hipBLASLt int8 (6-8 kernels, 33-41 TOPS, no fused scaling); the int8-is-2x assumption (int8 and bf16 WMMA both peak at ~56 T on gfx1151);
tiles other than 128x128 (tail balance vs per-WG efficiency), prefetch distance 2, s_setprio, 16-wave WGs, A-direct variant; the WMMA flash attention
(correct, 7-9% faster per call than SDPA, but only -0.5 ms in situ; 32-rows-per-wave variant spills) -> optional.
Repro: see README.md "Commands" (bench.py / test_hgemm.py / verify.py HLin HI8Lin / test_throughput.py / run_docker_vs_host.sh) and "Building the kernels".
Shared-GPU caveat: on 2026-10-02 another GPU process (an LLM server) kept gpu_busy_percent at 93-100%;
all final numbers were taken under that load (host and container back-to-back); idle-GPU 150-frame runs earlier gave 82.5 / 9.0 / 91.5 ms.
