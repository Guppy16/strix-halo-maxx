# moondream3-preview vs moondream2: grounding benchmark (2026-10-09)

Everything is in `~/git/app/.scratch/moondream3/` on testbox. Nothing in the app package or
the live code changed.

## Download (pinned, safetensors only)
- `moondream/moondream3-preview` @ `5112966d1a723413b1c9a1e8bea272b72e647b35`, into `hf/` (cache_dir).
  - First pass: only `*.py *.json *.md .gitattributes`. I read every .py before running any of it.
  - Second pass (`download.py`): only `modelv2-0000{1..4}-of-00004.safetensors`, the shards that
    `model.safetensors.index.json` maps to (18.5 GB bf16, 9.27 B params).
  - **Not downloaded:** `model_fp8.pt` (10.5 GB pickle) and the older `model-*` shards (not in the
    index), nor the PNGs.
- `moondream/starmie-v1` (the tokenizer) @ `35192e10a54e36eabe0a7cc57a2c1aab371cafc5`: json/md only.
- sha256 of every file: `SHA256SUMS`. The four safetensors hashes match the LFS oids the HF API
  lists for that commit.
- Licence: BSL 1.1 (M87 Labs), Apache-2.0 two years after release. Non-production / internal use
  is fine; offering it as a competing paid hosted API is not.

## Code review (all 11 .py files: hf_moondream, moondream, config, text, vision, layers, region,
## image_crops, rope, utils, lora)
- **Network:**
  - `moondream.py:103` `Tokenizer.from_pretrained("moondream/starmie-v1")` fetches the tokenizer
    from HF `main` (unpinned) on every model construction. `bench.py` replaces it with
    `Tokenizer.from_file(<pinned local tokenizer.json>)` and runs with `HF_HUB_OFFLINE=1`.
  - `lora.py` `_fetch_presigned_url` / `cached_adapter_path`: `urllib.urlopen` to
    `https://api.moondream.ai` (or `$MOONDREAM_ENDPOINT`) with `$MOONDREAM_API_KEY`, then downloads
    `adapter.pt` to `$HF_HOME/hub/md_finetunes/...` and `torch.load`s it (`weights_only=True`, but
    it falls back to an unrestricted `torch.load` on TypeError). **Only reached when a caller passes
    `settings={"adapter": ...}` or `{"model": "...@..."}`.** We never do; without an API key it
    raises before any request.
- **Pickle:** only that adapter `torch.load` path. The base weights load via transformers'
  safetensors loader.
- **subprocess / os.system / eval / exec / importlib / sockets:** none.
- **File writes:** only the adapter cache above (unused). transformers copies the remote .py into
  `HF_MODULES_CACHE`; bench.py points that, plus the inductor/triton caches, into this folder.
- Other: `image_crops.py` uses pyvips if installed (it isn't, so PIL). `layers.py` imports torchao
  only for int4 `QuantizedLinear`, which is unused because `group_size=None`. `HfMoondream` has no
  `compile()` of its own, so `model.compile()` is torch's `nn.Module.compile` (a no-op here).
  Use `model.model.compile()`.

## Segment
**Not in the open weights.** The local code has only query/caption/detect/point (+ detect_gaze).
There's no `segment` method, no segment template in `TokenizerConfig.templates`, and no
mask/SVG head in the weight map. Segmentation is cloud-only (moondream.ai API). Not tried.

## Runs (all through `../moondream-opt/gpu_safe.sh`, venv `../moondream-opt/venv-rocm714`:
## torch 2.14.1+rocm7.14, transformers 4.57.6; no new packages, no new venv)
`bench.py md2|md3 [--compile]` runs detect, point, then encode once + detect-on-encoded, for the
50 queries in `data/box_queue/grounding-2026-10-09.json` (full 720x1280 frame, leading
"the/a/an" stripped as in ground._noun), then caption/query on 5 frames.
`score.py` grades results the way replay_hybrid.py does (first box, IoU>0.5 with a labelled answer;
points: inside the smallest labelled box). On the old md2 results it gives 44 correct / 4 wrong /
1 unlabelled, which matches the 44/48 baseline. `compare.py` lists disagreements. `overlay.py`
writes `out/<id>.jpg` (orange md2, cyan md3, green/red labelled boxes).
- `results/md3_modulecompile_noop_*`: a run where `model.compile()` was the no-op nn.Module one.
  Same boxes as eager, but detect was ~2.2 s. The slowdown wasn't explained (the GPU was idle
  before and after), so ignore it.

## Results (GPU otherwise idle; latencies are medians over 50 queries)
| | md2 (2025-06-21) | md3 eager | md3 compiled (`model.model.compile()`) |
|---|---|---|---|
| load (warm page cache) | 2.7 s | 6.1 s | 6.1 s + 39 s compile + 19 s warm-up |
| GTT after load / peak | 3.8 / 4.9 GB | 17.9 / 19.3 GB | 19.1 / 19.5 GB |
| detect (encode + decode) | 464 ms (p90 769) | 1290 ms (p90 1325) | 1204 ms (p90 1227) |
| point | 440 ms | 1186 ms | 1130 ms |
| encode_image alone | 370 ms | 841 ms | 842 ms |
| detect on a pre-encoded image | 92 ms | 450 ms | 365 ms |
| detect: correct / wrong / unlabelled (49 graded) | 44 / 4 / 1 | 42 / 3 / 4 | same boxes as eager on 43/50; same grades |
| detect after my look at the unlabelled | 44 / 4 (+q36 ambiguous) | 44 / 5 | |
| point: correct / wrong / partial / unlabelled | 41 / 7 / 0 / 1 | 43 / 4 / 1 / 1 | 43 / 4 / 1 / 1 |
| object present (12), detect | 12 | 11 (missed q18) | |
| absent (10): nothing returned, detect / point | 8 / 4 | 8 / 6 | |

Unlabelled md3 detect boxes, judged by eye (`out/`): q06 and q09 (green LED) are on the LED,
tighter than the labelled boxes, so correct (IoU < 0.5 only because the target is tiny). q18 boxes a
blurred strip at the bottom edge; the organiser is on the desk, so wrong. q11 boxes the white
extension block, not the object, so wrong (md3's *point* on q11 is right).

Detect disagreements, md2 -> md3:
- q36 water bottle: unlabelled -> correct (md3 picks the left bottle like Qwen).
- q47 monitor on the black frame: wrong (whole-frame box) -> correct (nothing).
- q18 tiny far object: correct -> wrong (bottom-edge strip).
- q49 object, absent: correct -> wrong (boxed the object's base).
- q11 object: wrong -> wrong (a different object).
- q06, q09 LED: correct -> correct by eye (unlabelled by IoU).

VQA latency (5 frames q01 q08 q15 q36 q47, default sampling, medians; latency only):
md2 caption short 1.39 s / normal 2.58 s / query 1.08 s.
md3 eager: caption short 2.95 s / normal 6.17 s / query (reasoning=False) 2.91 s / query with
reasoning 7.6 s. md3 compiled: 2.27 / 4.26 / 1.91 / 6.16 s.

## Verdict
Not better at grounding on this set: the same 44 correct by eye, one more wrong, and it loses the
tiny-organiser case that motivated moondream. It is ~2.6x slower (1.2-1.3 s against 0.46 s) and
needs ~4.5x the GPU memory (19 GB against 4.4 GB). It is somewhat better at `point` as a presence
test (6/10 absent against 4/10), but that isn't how we use it. So no optimisation plan beyond the
note in the report.
