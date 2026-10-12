# bf16-only elementwise ops in the vision tower (XDNA2 NPU numerics), 2026-10-08

Question: the NPU vector unit has no fp32 arithmetic. If LayerNorm, tanh-GELU, attention softmax and residual adds of
moondream2's vision tower (27 blocks + post_ln + projection MLP; text model untouched) round every intermediate to bf16,
with fp32 only for reductions, how much accuracy is lost on the hand-labelled frames?

## Method
`bf16emu.py` replaces `FastMD2._vision` (`frame_preds.py --bf16 all|ln,gelu,softmax`). Every op computes in fp32 and rounds
to bf16 straight away; constants are bf16; exp is correctly rounded from its bf16 input (best case for a bf16 exp unit).
* LayerNorm: fp32 mean -> bf16; d = x - mean; fp32 variance of the bf16 d; rstd = rsqrt in fp32 -> bf16; d*rstd, *gamma, +beta each rounded.
* tanh-GELU: x*x, *x, *c1, +x, *c0, exp(2u), +1, 1/., 1-2r, 1+th, 0.5x, product: each rounded.
  `gelusig` variant: same u, then x * (1 / (1 + exp(-2u))) (identical function, avoids the 1 - 2/(e+1) and 1 + th cancellations).
* Softmax: QK^T fp32-accumulated -> bf16, *scale (bf16) -> bf16, bf16 row max, exp(s - max) bf16, fp32 row sum, bf16
  reciprocal, p * r -> bf16, PV with bf16 inputs fp32 accumulation -> bf16.
* Residual adds: not emulated because the GPU path already does them in bf16 (bf16 + bf16 tensors, sum rounded to bf16 =
  a bf16 vector add). Today's GPU path computes LN/GELU/softmax in fp32 internally with one rounding of the output
  (HLin fuses GELU into the GEMM epilogue in fp32).
* `npu_full*` add `--vlin BfpAcc32Lin` (bfp16ebs8 operands, fp32 accumulation; bfpemu.py).

Op check (`bf16emu_check.py`, frame session A/0060, blocks 0/13/26, rel rms error vs fp32; runs-1008/check2.log):

| op | GPU path | bf16 emu |
|---|---|---|
| LayerNorm | 1.6e-3 | 3.7-4.4e-3 |
| softmax attention | 1.7e-3 | 4.2-7.6e-3 |
| tanh-GELU (tanh form) | 0.4-1.8e-3 | **1.4-2.1e-2** |
| GELU, sigmoid form | 0.4-1.8e-3 | 0.5-4.5e-3 |

Encoder output vs GPU path (rel rms / min token cosine): ln 7.4% / 0.88, gelu 17% / 0.48, gelusig 7.7% / 0.75,
softmax 5.8% / 0.93, all 18% / 0.49, all with gelusig 9.0% / 0.81.

## Results (`python3 rescore.py`, 462 frames, 24 positives; runs-1008/rescore.txt, margins.txt)

| config | accuracy | fp | fn | flips vs ref | mean / max abs margin shift |
|---|---|---|---|---|---|
| exact (GPU path) | 98.3% | 8 | 0 | - | - |
| npu_bfp16 (GEMMs only) | 97.8% | 10 | 0 | 6 | 0.24 / 0.93 |
| bf16_ln | 97.6% | 11 | 0 | 5 | 0.19 / 0.75 |
| bf16_gelu | 97.0% | 14 | 0 | 10 | 0.28 / 2.22 |
| bf16_softmax | 97.2% | 13 | 0 | 7 | 0.19 / 0.71 |
| bf16_all | 97.4% | 12 | 0 | 8 | 0.29 / 1.94 |
| npu_full (all + bfp16 GEMMs) | 97.2% | 13 | 0 | 9 | 0.28 / 1.74 |
| bf16_gelusig | 97.2% | 13 | 0 | 5 | 0.18 / 0.97 |
| bf16_all_sig | 97.8% | 10 | 0 | 6 | 0.19 / 0.88 |
| npu_full_sig | 97.8% | 10 | 0 | 6 | 0.24 / 1.42 |
| pool16 (GPU path) | 99.4% | 3 | 0 | - | - |
| **npu_full_pool16** | **99.6%** | 2 | 0 | 1 | 0.19 / 1.03 |
| npu_full_sig_pool16 | 99.6% | 2 | 0 | 1 | 0.19 / 0.75 |

Flips/shifts are vs exact (729-token rows) or vs pool16 (pool rows). Nearly every flip is a tv frame of session
B that the GPU path already scores within +-0.8 of zero (a band of ~15 borderline negatives),
so the per-config fp counts move up and down by noise (bf16_all has fewer fp than bf16_gelu alone). No config misses a
positive: the smallest true-positive margin stays +4.7..+5.1 (exact +4.8) and +3.5..+3.7 (pool16 +3.2).

## Verdict
bf16 elementwise is acceptable for the NPU encoder in the target config: npu_full_pool16 makes 2 errors vs pool16's 3
(one borderline negative flipped, +0.6 -> -0.1), no misses. At 729 tokens the full NPU numerics cost +5 fp (13 vs 8),
just outside the ~2-frame target, all on the borderline band, no misses.
The costly op is GELU in the tanh form 1 - 2/(exp(2u)+1): ~2% error per call from cancellation (10x the other ops), max
margin shift 2.2. It does not need more precision, only the sigmoid form x / (1 + exp(-2u)) in the same bf16 ops
(0.5% error); with it the full NPU numerics at 729 tokens match the bfp16-GEMM-only result (10 fp, +2 vs exact).
LayerNorm (fp32 mean/var) and softmax (fp32 row sum) in bf16 are fine as specified.

Files: bf16emu.py, bf16emu_check.py, bf16_margins.py, frame_preds.py (--bf16), runs-1008/ (fp_*.log, check*.log,
rescore.txt, margins.txt, run_all.sh, run_sig.sh).

## Follow-up: the NPU attention kernel (npu-moondream p9/attn15.cc), op `attnbfp`
`bf16emu.attn_bfp`: Q, K (blocks of 8 along head dim), V and P (blocks of 8 along keys) as bfp16ebs8, fp32 accumulation;
online softmax over 48-key chunks, bf16 running max (start -64), bf(exp(bf(s - m))) (correctly rounded, clamped at -40),
O and sums rescaled by bf(exp(bf(m_old - m_new))) in fp32, row sum = fp32 sum of the bfp P that PV multiplies,
o = bf(O * bf(1/bf(sum))), scale 1/sqrt(72) as a bf16 constant. Check (runs-1008/check3.log), rel rms vs fp32:
1.0-1.6e-2 (bf16 `attn` 0.4-0.8e-2; attn_bfp vs `attn` 1.0-1.7e-2, matching the kernel's ~1e-2 vs bf16emu on random data).
Encoder output vs GPU path: attnbfp alone 7.7% / min token cosine 0.77; ln,gelusig,attnbfp 9.2% / 0.85 (with bf16 attn: 9.0% / 0.81).

| config | accuracy | fp | fn | flips vs ref | mean / max abs margin shift |
|---|---|---|---|---|---|
| exact (GPU path) | 98.3% | 8 | 0 | - | - |
| npu_full_sig (bf16 attn) | 97.8% | 10 | 0 | 6 | 0.24 / 1.42 |
| **npu_full_sig_bfpattn** | 97.6% | 11 | 0 | 5 | 0.25 / 1.44 |
| pool16 (GPU path) | 99.4% | 3 | 0 | - | - |
| **npu_full_sig_bfpattn_pool16** | 99.6% | 2 | 0 | 1 | 0.19 / 1.03 |

At 729 tokens the flips are all tv negatives with GPU margins in -1.25..+0.2 (session B 0114/0115/0124/0147,
session C 0047); the pooled flip is session B/0181 (+0.1 -> -0.3, now correct). Smallest true-positive margin
+4.8 (exact +4.8) and +3.3 (pool16 +3.2): no misses.
Verdict: bfp16 attention operands are acceptable. The target pool16 config is unchanged (2 errors vs the GPU path's 3);
at 729 tokens it is +3 fp vs exact, 1 more than with bf16 attention, all on the borderline band.
