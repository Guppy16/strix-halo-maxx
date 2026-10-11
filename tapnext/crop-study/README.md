# Crop study: TAPNext++ on a small, moving crop (CPU, 2026-10-11)

Question: can TAPNext++ track a compact group of points from a W x W crop of the 256 frame (fewer image tokens)
without losing accuracy? CPU bf16 reference, depth 12, TAP-Vid DAVIS "first", the 10-video subset the NPU rounds use.
Everything ran in `npu-iron-torch` on testbox, `nice -n 19`, 8 threads. No NPU, no GPU. Code: `crop_study.py`, `run.sh`.

## Setup

- **Grouping rule.** A video's queries are grouped by the 64 x 64 cell (in 256 model space) that holds the point's
  query-frame (y, x). That gives 53 groups over 10 videos, 1 to 15 points each (16 singletons, median 2).
  Spread: the query-frame bbox side is median 20 px, max 54 px. The per-frame bbox side of the visible GT points is median 12 px.
  Its per-group max over time is median 26 px, max 150 px (points drift apart).
- Each group is tracked on its own: its points are the only query tokens, with no 48-slot padding.
  Metrics are computed per video over all points (the groups' predictions merged), then averaged over the videos, as in davis_cpu.py.
- **A: moving crop at native scale.** A W x W window on the 8 px patch grid gives (W/8)^2 image tokens, each with its own
  slice of the learned 32 x 32 pos-emb, so absolute positions are kept. Queries stay in 256 space.
  The readout is the coordinate head: 2 x 256 logits from the query token, soft-argmax, so predictions are always
  full-frame 256 coordinates. There is no heatmap over image tokens, and no remapping is needed.
  - Crop policy, each frame: the anchors are the active points at their t-1 prediction, plus points queried this frame
    at their query position. Before the first query, the anchors are the next queries. The window is centred on the anchors'
    bbox, rounded to whole patches and clamped to the frame. If bbox + 2 x 16 px margin does not fit, W grows to the next of
    160 / 192 / 256 (fallback) for that frame and shrinks again when the points come back together.
  - State: per block, the RG-LRU state and the 3-frame conv cache live in a 32 x 32 buffer.
    - **s** (shift): the buffer is indexed by absolute patch. State moves with the content, tokens that leave drop
      their state, and entering tokens start from zero.
    - **n** (no shift): the buffer is indexed by crop-local slot, so the state stays put while the crop moves.
- **B: zoomed fixed crop.** The pickle holds the 480p DAVIS frames (480 x 854), so real extra pixels exist.
  Per group, a square of side S in 256 space (smallest of 128 / 160 / 192 that fits the query-frame bbox + 2 x 16 px; all groups
  fit at S) is fixed for the whole video. It is sampled from the 480p frames straight to 256 x 256 (grid_sample bilinear, same
  convention as preprocess) and run with the full pos-emb. Queries are mapped into the crop and predictions mapped back.
- **Correctness check.** `c256` is the crop path with W = 256. Its tracks and vis are **bit-identical to `base` (model.step)
  on all 53 groups**, so its AJ is the same 69.67.

## Results (10 videos, AJ / OA / delta_avg in %)

| variant | AJ | OA | delta_avg | frames with enlarged W | frames at full 256 |
|---|---|---|---|---|---|
| base, grouped, full 256 frame | **69.67** | 91.96 | 82.84 | - | - |
| c256 (crop path, W = 256) | 69.67 | 91.96 | 82.84 | - | - |
| A W=192, shift | **69.72** | 92.10 | 82.63 | 0.0 % | 0.0 % |
| A W=192, no shift | 27.39 | 85.74 | 38.94 | 0.1 % | 0.1 % |
| A W=160, shift | **69.46** | 92.20 | 82.30 | 0.15 % | 0.0 % |
| A W=160, no shift | 22.40 | 83.12 | 32.96 | 0.25 % | 0.02 % |
| A W=128, shift | **68.72** | 91.42 | 82.13 | 2.7 % | 0.0 % |
| A W=128, no shift | 21.18 | 77.60 | 31.36 | 1.1 % | 0.05 % |
| B S=192 (zoom 1.33x, fixed) | 67.03 | 87.75 | 79.33 | - | - |
| B S=160 (zoom 1.6x, fixed) | 64.67 | 84.88 | 76.88 | - | - |
| B S=128 (zoom 2x, fixed) | 64.09 | 83.93 | 75.29 | - | - |

Reference: the ungrouped CPU bf16 run with all queries in one pass and 48 slots gives AJ 70.98. Grouping alone costs 1.3 AJ:
points lose their co-tracked neighbours. That matches the support-point effect seen before.

- **A with the state shift is effectively lossless at W = 192 and 160, and costs about 1 AJ at W = 128.**
  Per group, the a128s drop is median +0.6 AJ (10th to 90th percentile -2.8 .. +8.2, worst +18).
  Per video it ranges -1.3 .. +3.8 (dance-twirl -3.8, camel -2.3, car-shadow -2.0).
- **The state shift is essential.** Without it AJ collapses to 21 to 27. The recurrent state is spatially tied to its patch.
- **Fallback is rare.** The crop grew on 2.7 % of group-frames at W = 128, 0.15 % at 160, 0 % at 192. It never had to go to
  the full frame with the shift on.
- **B (fixed zoom) loses overall**, because points leave a fixed crop: car-shadow 81 -> 55, dance-twirl 71 -> 32.
  Scored only on frames where the point's GT is inside its crop, zoom helps:

| inside-crop frames only | base AJ | B AJ | base delta_avg | B delta_avg |
|---|---|---|---|---|
| S=192 | 73.99 | 75.31 (+1.3) | 82.99 | 84.33 |
| S=160 | 73.81 | 75.29 (+1.5) | 82.69 | 84.41 |
| S=128 | 74.08 | 76.38 (+2.3) | 83.00 | 85.14 |

  So the extra pixel density is worth about 1.3 to 2.3 AJ while the points stay in view. A zoom crop would have to move,
  which needs the pos-emb and state treatment worked out at a non-native scale. Untried here.

Per-video AJ: `results_per_video_aj.csv`. Per-group rows (spread, attention, drops): `results_groups.csv`. Everything: `results.json`.

## Attention outside the window (base runs)

On the full-frame base runs, each frame's would-be A window (same policy, nominal W, no enlarging) is computed from the
previous prediction. The softmax attention mass is averaged over heads, active query rows, and frames.

| W | area outside | query->image mass outside (mean / min / max over groups) | in-window image->outside | top-8 outside keys' share of the query's outside mass |
|---|---|---|---|---|
| 128 | 75 % | 20.5 % / 13.4 / 35.5 | 36.6 % | 42 % |
| 160 | 61 % | 15.4 % / 10.6 / 28.1 | 30.6 % | 41 % |
| 192 | 44 % | 11.2 % / 7.7 / 22.5 | 24.3 % | 43 % |

Per block, query->outside at W = 128, blocks 0..11:
`.35 .20 .15 .08 .09 .15 .14 .11 .14 .11 .28 .67`. The middle blocks are local (8 to 15 %).
Block 0 and the last two blocks look far: block 11 puts 67 % outside, close to the 75 % uniform share.
About 40 % of the outside mass sits on 8 keys, which looks like sink/register tokens.

- The far mass is **not small**: 20 % at W = 128. It is concentrated about 3.7x relative to area, but it is not negligible.
- It does **not predict the AJ drop**:
  - Per group (n = 53) the correlation is about 0: Pearson / Spearman a128s 0.10 / 0.04, a160s 0.05 / 0.01, a192s -0.21 / -0.10.
  - Per video (n = 10) a128s gives 0.59 / 0.62 and a160s 0.43 / 0.59, a weak trend on 10 points. a192s gives -0.28 / 0.27.
- The model loses little when the far keys are dropped even though it attends to them. That mass is probably
  context or sink-like, and the softmax renormalises over what is left.

## Speed

CPU, the same session and process, 8 threads, 8 queries, interleaved sizes, 3 x 8 timed frames after warm-up. The box was
loaded (load average 11 to 14, the NPU optimiser running), so the ratios mean more than the absolute ms.
FLOPs are the analytic count per frame: patch embed, 12 x (SSM + ViT linears + attention), heads.

| W | image tokens | GFLOP / frame | CPU ms / frame (median, min) | NPU estimate ms |
|---|---|---|---|---|
| 256 | 1024 | 436 | 260 (243) | 33.9 (measured, 48 slots) |
| 192 | 576 | 237 | 140 (133) | ~17 to 20 |
| 160 | 400 | 163 | 100 (92) | ~13 to 16 |
| 128 | 256 | 104 | 71 (69) | ~8 to 12 |

The NPU numbers are an **estimate, not a build**. They start from the round-19 timeline of the 33.9 ms step: GEMM k-steps
+ epilogues + drains about 15.4, attention 5.5, norms 4.1, SSM 3.35, op-boundary bubbles about 5.
Each part is scaled by r = tokens / 1072 (48 slots kept): GEMM 15.4 r but never below the ~4 ms weight floor (~230 MB
of weights from DDR at ~57 GB/s each frame), attention 5.5 r^2, norms and SSM linear in r.
The bubbles are held fixed for the upper end of the range and scaled with r for the lower end.
At W = 128, r = 0.28 and the GEMMs sit at the weight floor (4.4 ms vs 4.0), so going below W = 128 buys little.
Fewer than 48 query slots helps a little more. The current design's tiling (rows per column, 8 x 8 panels) would have to
be rebuilt for a smaller M, and fixed per-op costs may not shrink in proportion.

## Caveats

- 10 videos and 53 groups: per-group AJ is noisy, especially for 1-point groups. Differences under about 1 AJ are within noise.
  a192s and a160s sit 0.05 to 0.2 from base in both directions.
- The grid grouping can split close points across a cell edge. Groups are spatially compact at query time, but DAVIS points
  on one object drift up to 150 px apart. A real use case of a few points on one object should behave like these groups.
- Entering tokens start from a zero RG-LRU state, so they get the normalised update, not the first-frame "state = input"
  path. Using that path for new tokens is untried.
- The crop is chosen from the previous frame's prediction, so a fast-moving group can be partly cut for one frame.
  The 16 px margin covered this on DAVIS.
- Attention was measured with fp32 softmax over the bf16 q and k (SDPA computes it internally). It is averaged over heads,
  so individual heads can be far more local or more global.
- B was run with a fixed crop only, as specified. A moving zoomed crop is the obvious follow-up.
- Speed was measured on a shared, loaded box. The NPU figure is extrapolated, not measured.

## Reproduce

    crop-study/run.sh main run --variants base,a128s,a128n,a160s,a160n,a192s,a192n,b128,b160,b192,c256   # ~2.6 h on the loaded box
    crop-study/run.sh speed speed --q 8 --reps 3 --iters 8
    crop-study/run.sh rep report        # results.json + results_*.csv need out/runs.jsonl and out/pred/*.npz

Run time: about 2.6 h wall for all variants on 10 videos (4,026 group-frames per variant). Base with attention recording runs
at ~0.5 s/frame, the full-frame crop path at ~0.26, W=128 at ~0.07, B at ~0.26.
Predictions (`out/pred/*.npz`, 7 MB) are not committed; `out/runs.jsonl` is.
