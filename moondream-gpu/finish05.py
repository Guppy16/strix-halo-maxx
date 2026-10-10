import json, sys, types, os, statistics
HERE = os.path.expanduser("~/moondream-opt"); sys.path.insert(0, HERE)
_t = types.ModuleType("torch"); _nn = types.ModuleType("torch.nn"); _F = types.ModuleType("torch.nn.functional"); _t.nn=_nn; _nn.functional=_F; _t.bfloat16=None; _t.cuda=None
sys.modules.update({"torch": _t, "torch.nn": _nn, "torch.nn.functional": _F})
from common import load_testset
r = json.load(open(f"{HERE}/ref/md05b-int8-onnx-cpu-lanczos378.json")); mg = r["margins"]; s = r["summary"]; sg = r["summary_greedy"]
items = load_testset()
def conf(obj, t):
    sel = [it for it in items if it["obj"] == obj]
    tp = sum(1 for it in sel if it["label"] and mg[it["path"]] > t); fn = sum(1 for it in sel if it["label"] and mg[it["path"]] <= t)
    fp = sum(1 for it in sel if not it["label"] and mg[it["path"]] > t); tn = sum(1 for it in sel if not it["label"] and mg[it["path"]] <= t)
    return f"t>{t:+.2f}: acc={(tp+tn)/len(sel)*100:.1f}% tp={tp} fn={fn} fp={fp} tn={tn}"
out = []
out.append("\n## Step 2b: full test set (493 frames), official 0.5B int8 .mf, onnxruntime CPU 16 threads, 378px single tile, logits yes/no (== greedy text, 100% agreement)")
out.append("| model | tv acc (n=402) | tv tp/fn/fp/tn | mug acc (n=91) | mug tp/fn/fp/tn | all acc | agreement w/ stock md2 | encode | query | total |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
out.append("| moondream2 stock bf16 (GPU) | 97.3% | 46/0/11/345 | 100% | 7/0/0/84 | 97.8% | - | 245 ms | 100 ms | 345 ms |")
out.append("| moondream2 optimised (GPU, bench.py) | 97.5% | 46/0/10/346 | 100% | 7/0/0/84 | 98.0% | 99.4% | 95 ms | 9 ms | 105 ms |")
tv, mug = s["tv"], s["mug"]
out.append(f"| moondream 0.5B int8 (official ONNX, CPU) | {tv['acc']*100:.1f}% | {tv['tp']}/{tv['fn']}/{tv['fp']}/{tv['tn']} | {mug['acc']*100:.1f}% | {mug['tp']}/{mug['fn']}/{mug['fp']}/{mug['tn']} | {s['all']['acc']*100:.1f}% | 39.1% | {s['enc_ms']:.0f} ms | {s['qry_ms']:.0f} ms (logits) / {sg['qry_ms']:.0f} ms (greedy, 4 tok) | {s['tot_ms']:.0f} ms |")
hist = {k: list(r["raw"].values()).count(k) for k in set(r["raw"].values())}
out.append(f"Raw greedy answers: {hist}. Recall is 100% on both tasks (every judge-'found' frame is a yes) but 303 of 440 negative frames are also 'Yes'.")
out.append("Sanity gate: judged_0060 mug -> yes (ok), judged_0000 black frame mug -> yes (FAIL). Agreement with the optimised moondream2 predictions: 38.9%.")
out.append("Per-frame CPU latency: median encode 640 ms (vision 27 blocks + 730-token text prefill via W8A8 DynamicQuantizeMatMul), yes/no logits 41 ms, greedy 4-token query 111 ms.")
out.append("\n## Step 2c: is the yes-bias a prompt/preprocessing artefact? (probe05.py)")
out.append("* Captions are sensible (judged_0060 -> 'A [caption removed], filled with white tea, sits on a [removed]'; black frame -> 'A dark blue gradient ...'),")
out.append("  so the model, tokenizer and preprocessing are working; the failure is acquiescence on closed questions.")
out.append("* Four wordings on 18 frames (2 sanity + 8 found + 8 not found): 'Is there a X in this image? Answer yes or no.' 11/18 correct, 'Is there a X in this image?' 9/18,")
out.append("  'Does this image contain a X? Answer yes or no.' 9/18, 'Can you see a X? Answer yes or no.' 9/18 -> every wording says yes on the black frame; the harness prompt is the least biased.")
out.append("* Control object ('Is there a giraffe ...') is 'No' on all 18 frames (margins -0.8 .. -4.3): the bias is specific to objects that are plausible in the scene, not a global yes-prior.")
out.append("* The client's own preprocessing on the raw 720x1280 frame (2x1 multi-crop, encode 1.4 s) gives the same decisions (0060 +1.64, black +0.47).")
out.append("* Margin analysis (yes-logit minus no-logit, first decode step) over the 493 frames:")
for obj, auc, best, keep in (("tv", 0.998, 2.61, 2.44), ("mug", 0.908, 0.73, 0.06)):
    sel = [it for it in items if it["obj"] == obj]; pos = sorted(mg[it["path"]] for it in sel if it["label"]); neg = sorted(mg[it["path"]] for it in sel if not it["label"])
    out.append(f"  - {obj}: positives {pos[0]:+.2f}/{statistics.median(pos):+.2f}/{pos[-1]:+.2f} (min/med/max), negatives {neg[0]:+.2f}/{statistics.median(neg):+.2f}/{neg[-1]:+.2f}, AUC {auc:.3f}; "
               f"best in-sample threshold {conf(obj, best)}; threshold keeping recall 100% {conf(obj, keep - 0.001)}")
out.append("  - A per-task threshold fitted on this same test set would reach tv 99.0% / mug 96.7% (overall ~98.6%), i.e. the tv signal is usable with a +2.6 offset,")
out.append("    but mug (7 positives, margins +0.06..+2.25 overlapping the negatives) cannot be made both sensitive and specific, and these thresholds are fitted in-sample")
out.append("    on 53 positives from a few sessions -- not a deployable rule without a held-out set and per-object calibration.")
out.append("  - 10 tv negatives have a margin of exactly 0.00 (fp16 logit ties between ' Yes' and ' No'); the decision there is arbitrary.")
out.append("\n## Step 3: speedups (HIP graphs / logits yes-no / Triton) -- NOT pursued")
out.append("Accuracy is ~60 points below moondream2 (gate was 'within ~2 points'), so no GPU port was made. The logits-based yes/no was already used above (it is the")
out.append("natural first decode step; it agrees 100% with the greedy text and saves 70 ms on CPU). GPU latency can only be estimated: the 0.5B needs ~0.23 TFLOP for the")
out.append("vision encoder (0.16B params x 729 tokens) + ~0.44 TFLOP for the 730-token text prefill (0.30B non-embedding params), ~0.67 TFLOP vs ~2.3 TFLOP for moondream2.")
out.append("At the 25-31 TFLOPS the moondream2 Triton/int8 kernels achieve on this GPU that is ~25-35 ms per frame with an equivalent optimised path (vs 105 ms), or roughly")
out.append("100 ms with stock-quality kernels -- i.e. a ~3x speedup ceiling, which would require porting the uint8 ONNX weights into a torch/Triton re-implementation.")
out.append("onnxruntime on GPU is not an option here either: the official client supports CPU and CUDA only, and there is no ROCm onnxruntime wheel for this torch/ROCm 7.2 stack.")
out.append("\n## Verdict")
out.append("moondream 0.5B (the only release: int8 ONNX, Jan 2025) is NOT a viable replacement for moondream2 as a yes/no presence gate on this data: 38.5% accuracy")
out.append("(tv 36.8%, mug 46.2%) vs 97.8-98.0%, because it answers 'Yes' to almost any 'is there a <object>' question about a scene-plausible object, including on a black frame.")
out.append("It would catch every true positive (recall 100%) but pass 69% of negatives, which defeats the purpose of a gate. The only salvage is a calibrated logit-margin")
out.append("threshold, which works for tv (AUC 0.998) but not for mug (AUC 0.908), and would need a held-out calibration set per object. Recommendation: keep the optimised")
out.append("moondream2 (105 ms exact / 67 ms pooled); if a smaller gate is wanted, candidates are the moondream2 pooled mode already measured, or a different small VLM")
out.append("(e.g. a 0.5-1B Qwen-VL-class model with bf16 weights that can run on the GPU), not moondream 0.5B.")
out.append("\nFiles: eval05.py (evaluation, venv05), probe05.py (prompt/caption probe), models/moondream-0_5b-int8.mf (official weights), ref/md05b-int8-onnx-cpu-lanczos378.json")
out.append("(per-frame predictions, margins, raw answers), logs_05b_full.txt. venv05 = ~/.local/bin/uv venv --python 3.12 venv05 && uv pip install moondream==0.0.6 (onnxruntime 1.30).")
open(f"{HERE}/RESULTS-0.5b.md", "a").write("\n".join(out) + "\n")
print("\n".join(out))
