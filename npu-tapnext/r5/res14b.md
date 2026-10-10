| **DAVIS 10 videos, full12m depth 8** (shared xclbin) | **AJ 68.28** (OA 91.05) vs CPU bf16 68.22; 62.5 ms/frame |
### 14.5 The Pareto front after round 5 (latency MEASURED; AJ = NPU DAVIS 10 videos where marked, else CPU bf16)
| depth | NPU ms per frame (dispatch) | fps | DAVIS AJ |
|---|---|---|---|
| 12 | 91.9 (89.4) | 10.9 | **71.00 NPU** (CPU 70.98) |
| 10 | 77.0 (74.6) | 13.0 | **69.87 NPU** (CPU 70.07) |
| 8 | 62.0 (59.7) | 16.1 | **68.28 NPU** (CPU 68.22) |
| 6 | 47.2 (44.9) | 21.2 | 64.77 (CPU) |
All from one xclbin (full12m) with Step8.set_depth. GPU compiled bf16: 22 ms at depth 12.
