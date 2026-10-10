# moondream 0.5B as a yes/no gate -- evaluation log (vs moondream2 1.9B)

Box: testbox (Strix Halo, Ryzen AI MAX+ 395 / Radeon 8060S gfx1151). Test set and metric identical to RESULTS.md:
493 frames from ~/git/app/data/sessions/*/judged.jsonl (mug: 91 frames / 7 found, tv: 402 frames / 46 found), question
"Is there a <obj> in this image? Answer yes or no.", label yes := Qwen3-VL judge "found". Baselines on this set: stock moondream2 bf16
97.8% / 345 ms; optimised moondream2 98.0% / 105 ms (bench.py). Sanity gate: sessA/judged_0060 mug -> yes, judged_0000 (black) -> no.

## Step 1: which "moondream 0.5B" exists (checked on the Hub 2026-10-01)
* `vikhyatk/moondream-0.5b` does NOT exist (HF API returns 401/not-found); no repo under vikhyatk/* or moondream/* has a 0.5B transformers checkpoint.
  A Hub-wide search for "moondream" finds only two third-party uploads: `whistleroosh/moondream-0.5B` (the unpacked ONNX graphs, Mar 2026)
  and `grifttown/moondream-0.5b` (a re-upload of the official .mf file, May 2025).
* The official 0.5B release (vikhyat/moondream README, Jan 2025; HF commit 9dddae84d54db4ac56fe37817aeaeb502ed083e2 = branch `onnx`
  of vikhyatk/moondream2, dated 2024-12-05) ships ONLY as ONNX ".mf" bundles for the `moondream` pip client (0.0.5/0.0.6, onnxruntime, CPU only):
  moondream-0_5b-int8.mf.gz 621,619,051 bytes (593 MiB) and moondream-0_5b-int4.mf.gz 442,376,060 bytes. No bf16/fp16 weights, no safetensors,
  no HF `trust_remote_code` model class -> the requested "stock HF path (trust_remote_code, bf16, GPU)" is not possible for 0.5B.
  The current `moondream` pypi package (2.x, Sep 2026) replaced this with the closed "kestrel"/Photon engine (CUDA/MPS/CPU only; no ROCm).
* Consequence for this evaluation: run the official int8 .mf with the official client (onnxruntime, CPU) in a separate venv
  (`~/moondream-opt/venv05`, python 3.12 -- moondream==0.0.6 pins pillow<11 and tokenizers<0.21 so it cannot go into the main venv).
  Accuracy is the gating question; GPU latency on this box would need a torch port of the ONNX weights (only worth it if accuracy passes).

## Step 2a: what is inside moondream-0_5b-int8.mf (models/, 692,802,047 bytes unpacked)
ONNX graphs (com.microsoft contrib ops): vision_encoder 156 MiB (27 SigLIP-style blocks, dim 720, ff 2690, 729 tokens for 378x378),
vision_projection 19 MiB (1440 -> 1024 GELU MLP; global and local tile features concatenated), text_encoder 100 MiB (embedding 51200x1024 fp16),
text_decoder 340 MiB (24 layers, dim 1024, 16 heads x 64, ff 4096, lm_head 1024x51200; KV cache (24,2,1,16,seq,64) fp16 with a 1-token BOS prefix),
coord/size encoders+decoders (region model, unused here). Every linear is `DynamicQuantizeMatMul` with uint8 weights (W8A8, dynamic per-tensor activation
quantisation); activations fp16. ~0.56B params total (text ~0.40B incl. embeddings, vision ~0.16B). There is no higher-precision release of these weights.
Prompt template: BOS + image(729 tokens) + "\n\nQuestion: <q>\n\nAnswer:"; decoding is greedy argmax (== temperature 0).
Harness (eval05.py, venv05): common.prep (378x378 LANCZOS squash, same as the moondream2 harness) -> client create_patches picks the (1,1) template
(single tile, 730-token image prefill); accuracy both from the first-step yes/no logits (max over {Yes,yes, Yes, yes} vs {No,...}, like md2fast)
and from the client greedy text (max_tokens=4). onnxruntime 1.30 CPUExecutionProvider, 16 intra-op threads (shared box; 16 cores).
Probe on the sanity frames (3 runs each): encode ~570-610 ms, yes/no logits 37 ms, greedy query 86 ms per frame (CPU).
  judged_0060 (mug) -> ' Yes' margin +1.86 (ok); judged_0000 (black frame) -> ' Yes' margin +0.73  <-- SANITY GATE FAILS on the stock path.

## md05b-int8-onnx-cpu-lanczos378
logits yes/no: tv: acc=36.8% n=402 tp=46 fn=0 fp=254 tn=102 | mug: acc=46.2% n=91 tp=7 fn=0 fp=49 tn=35 | all: acc=38.5% n=493 | median encode 640.0 ms, query 41.4 ms, total 681.4 ms | agree w/ stock md2 39.1% (n=493), w/ optimised md2 38.9%
greedy text (max_tokens=4): tv: acc=36.8% n=402 tp=46 fn=0 fp=254 tn=102 | mug: acc=46.2% n=91 tp=7 fn=0 fp=49 tn=35 | all: acc=38.5% n=493 | median encode 640.0 ms, query 110.8 ms, total 780.1 ms | agree w/ stock md2 39.1% | logits-vs-greedy agreement 100.0% | raw answers {' No': 137, ' Yes': 356}
sanity (logits) 0060 mug=True 0000 mug=True | threads=16 prep=lanczos378

## Step 2b: full test set (493 frames), official 0.5B int8 .mf, onnxruntime CPU 16 threads, 378px single tile, logits yes/no (== greedy text, 100% agreement)
| model | tv acc (n=402) | tv tp/fn/fp/tn | mug acc (n=91) | mug tp/fn/fp/tn | all acc | agreement w/ stock md2 | encode | query | total |
|---|---|---|---|---|---|---|---|---|---|
| moondream2 stock bf16 (GPU) | 97.3% | 46/0/11/345 | 100% | 7/0/0/84 | 97.8% | - | 245 ms | 100 ms | 345 ms |
| moondream2 optimised (GPU, bench.py) | 97.5% | 46/0/10/346 | 100% | 7/0/0/84 | 98.0% | 99.4% | 95 ms | 9 ms | 105 ms |
| moondream 0.5B int8 (official ONNX, CPU) | 36.8% | 46/0/254/102 | 46.2% | 7/0/49/35 | 38.5% | 39.1% | 640 ms | 41 ms (logits) / 111 ms (greedy, 4 tok) | 681 ms |
Raw greedy answers: {' Yes': 356, ' No': 137}. Recall is 100% on both tasks (every judge-'found' frame is a yes) but 303 of 440 negative frames are also 'Yes'.
Sanity gate: judged_0060 mug -> yes (ok), judged_0000 black frame mug -> yes (FAIL). Agreement with the optimised moondream2 predictions: 38.9%.
Per-frame CPU latency: median encode 640 ms (vision 27 blocks + 730-token text prefill via W8A8 DynamicQuantizeMatMul), yes/no logits 41 ms, greedy 4-token query 111 ms.

## Step 2c: is the yes-bias a prompt/preprocessing artefact? (probe05.py)
* Captions are sensible (judged_0060 -> 'A [caption removed], filled with white tea, sits on a [removed]'; black frame -> 'A dark blue gradient ...'),
  so the model, tokenizer and preprocessing are working; the failure is acquiescence on closed questions.
* Four wordings on 18 frames (2 sanity + 8 found + 8 not found): 'Is there a X in this image? Answer yes or no.' 11/18 correct, 'Is there a X in this image?' 9/18,
  'Does this image contain a X? Answer yes or no.' 9/18, 'Can you see a X? Answer yes or no.' 9/18 -> every wording says yes on the black frame; the harness prompt is the least biased.
* Control object ('Is there a giraffe ...') is 'No' on all 18 frames (margins -0.8 .. -4.3): the bias is specific to objects that are plausible in the scene, not a global yes-prior.
* The client's own preprocessing on the raw 720x1280 frame (2x1 multi-crop, encode 1.4 s) gives the same decisions (0060 +1.64, black +0.47).
* Margin analysis (yes-logit minus no-logit, first decode step) over the 493 frames:
  - tv: positives +2.44/+4.97/+7.25 (min/med/max), negatives -2.94/+0.18/+4.44, AUC 0.998; best in-sample threshold margin>+2.609: acc=99.0% tp=45 fn=1 fp=3 tn=353; lowest threshold keeping recall 100% (margin>=+2.438): acc=99.0% tp=46 fn=0 fp=4 tn=352
  - mug: positives +0.06/+0.80/+2.25 (min/med/max), negatives -1.81/+0.14/+0.73, AUC 0.908; best in-sample threshold margin>+0.734: acc=96.7% tp=4 fn=3 fp=0 tn=84; lowest threshold keeping recall 100% (margin>=+0.062): acc=48.4% tp=7 fn=0 fp=47 tn=37
  - Per-task thresholds fitted on this same test set would reach tv 99.0% / mug 96.7% (overall 98.6%), i.e. the tv signal is usable with a ~+2.6 offset,
    but mug (7 positives, margins +0.06..+2.25 overlapping the negatives) cannot be made both sensitive and specific, and these thresholds are fitted in-sample
    on 53 positives from a few sessions -- not a deployable rule without a held-out set and per-object calibration.
  - 10 tv negatives have a margin of exactly 0.00 (fp16 logit ties between ' Yes' and ' No'); the decision there is arbitrary.

## Step 3: speedups (HIP graphs / logits yes-no / Triton) -- NOT pursued
Accuracy is ~60 points below moondream2 (gate was 'within ~2 points'), so no GPU port was made. The logits-based yes/no was already used above (it is the
natural first decode step; it agrees 100% with the greedy text and saves 70 ms on CPU). GPU latency can only be estimated: the 0.5B needs ~0.23 TFLOP for the
vision encoder (0.16B params x 729 tokens) + ~0.44 TFLOP for the 730-token text prefill (0.30B non-embedding params), ~0.67 TFLOP vs ~2.3 TFLOP for moondream2.
At the 25-31 TFLOPS the moondream2 Triton/int8 kernels achieve on this GPU that is ~25-35 ms per frame with an equivalent optimised path (vs 105 ms), or roughly
100 ms with stock-quality kernels -- i.e. a ~3x speedup ceiling, which would require porting the uint8 ONNX weights into a torch/Triton re-implementation.
onnxruntime on GPU is not an option here either: the official client supports CPU and CUDA only, and there is no ROCm onnxruntime wheel for this torch/ROCm 7.2 stack.

## Verdict
moondream 0.5B (the only release: int8 ONNX, Jan 2025) is NOT a viable replacement for moondream2 as a yes/no presence gate on this data: 38.5% accuracy
(tv 36.8%, mug 46.2%) vs 97.8-98.0%, because it answers 'Yes' to almost any 'is there a <object>' question about a scene-plausible object, including on a black frame.
It would catch every true positive (recall 100%) but pass 69% of negatives, which defeats the purpose of a gate. The only salvage is a calibrated logit-margin
threshold, which works for tv (AUC 0.998) but not for mug (AUC 0.908), and would need a held-out calibration set per object. Recommendation: keep the optimised
moondream2 (105 ms exact / 67 ms pooled); if a smaller gate is wanted, candidates are the moondream2 pooled mode already measured, or a different small VLM
(e.g. a 0.5-1B Qwen-VL-class model with bf16 weights that can run on the GPU), not moondream 0.5B.

Files: eval05.py (evaluation, venv05), probe05.py (prompt/caption probe), models/moondream-0_5b-int8.mf (official weights), ref/md05b-int8-onnx-cpu-lanczos378.json
(per-frame predictions, margins, raw answers), logs_05b_full.txt. venv05 = ~/.local/bin/uv venv --python 3.12 venv05 && uv pip install moondream==0.0.6 (onnxruntime 1.30).
