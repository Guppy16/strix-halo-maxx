
## 14. Round 5 measurements (MEASURED, Turbo, GPU idle (gpu_busy 0), step8.py, one dispatch per frame)
### 14.1 Speed, change by change (depth 12; block 5 of the host-polled timeline)
| build | change | dispatch ms | per block |
|---|---|---|---|
| full12f (round 4) | | 106.8 | attn 1.77 wall, SSM 1.45, norms 0.31-0.36 each |
| full12h | new norm kernel, ew_op dispatch, cores one block per worker iteration | 96.4 (time, 20 iters) | norms 0.10-0.13 each |
| full12i | attention: lazy max (TAU 8) + expd_i | 91.6 | attention 1.77 -> 1.35 (qkv end -> attn end) |
| full12j | `--lookahead 2` | 90.4 | gaps before up_b / down_a / down_b / mlp3_b 0.08-0.12 -> 0.05-0.08 |
| full12k | SSM: e2 = a^2 from the fp32 a | 89.5 | SSM 1.45 -> 1.38 |
| **full12m** | the same sources, built for depths 12/10/8/6/1 (one xclbin serves all) | **89.40** | 7.4 |
### 14.2 Differentials (timing only, wrong outputs; block 5 wall time)
| build | removed | result |
|---|---|---|
| full12h{ne,nr,nv,nk} | attention exp / rescale / PV (chunk_v) / whole K phase | attention 1.77 -> 1.31 / 1.35 / 1.24 / 0.47 ms |
| full12jng / full12jcp | SSM transcendentals / SSM math (ring copy only) | SSM 1.45 -> 0.75 / 0.38 ms |
| full12kng | the GELU chain in every epilogue (`-DNO_GELU`) | step 89.5 -> 80.3 ms; up_a 0.54 -> 0.43, mlp0 0.73 -> 0.46 |
### 14.3 Correctness
| test | result |
|---|---|
| t1 (random): RMS(2 in) panels / LN(2 in) panels | bit-exact / 0.0148% of mantissas, <= 2.56 ulp (round 3: 0.09% of panels) |
| t2 (random): LN(3 in) panels | 0.0125%, <= 2.37 ulp |
| t4 (random): attention O | rms_rel 7.16e-3 (round 4 7.2e-3) |
| t3 (random): SSM h / z panels | h rms_rel 4.05e-3, z 10.2% of mantissas (round 3/4: 4e-3, 10%) |
| `blocks --build full1h` (depth 1, real weights) | every region corr >= 0.9997, rms_rel 0.3-2.2% (round 4: 0.6-3%) |
| smoke full12k (4 frames vs CPU bf16) | tracks within 0.021-0.027 px |
| smoke depth 10 on the depth-12 xclbin (full12m + full10m.bin) | tracks within 0.025-0.042 px |
| **DAVIS 10 videos, full12m depth 12** | **AJ 71.00** (OA 92.70) vs CPU bf16 70.98, round-4 NPU 70.81; 92.6 ms/frame |
| **DAVIS 10 videos, full12m depth 10** (shared xclbin) | **AJ 69.87** (OA 91.11) vs CPU bf16 70.07; 77.1 ms/frame |
Timeouts this round: 0.
### 14.4 Depth at run time (`step8.py sweep --build full12m --depths 12,10,8,6,12`, one process, one xclbin)
| depth | dispatch median / min | whole step incl. host | fps |
|---|---|---|---|
| 12 | 89.40 / 89.38 | 91.86 | 10.9 |
| 10 | 74.64 / 74.62 | 76.95 | 13.0 |
| 8 | 59.73 / 59.72 | 62.04 | 16.1 |
| 6 | 44.87 / 44.86 | 47.23 | 21.2 |
| 12 (again) | 89.40 / 89.38 | 91.96 | |
The elfs and CDO of full{12,10,8,6,1}m are byte-identical (cmp); the xclbins differ only in their headers.
