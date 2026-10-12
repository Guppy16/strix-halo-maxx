# moondream2 yes/no latency optimisation on Strix Halo (gfx1151)

Everything lives in `$SHM_SCRATCH/moondream-opt` (models, venvs and runs are not in the repo). `RESULTS.md` is the experiment log (every run, keep/discard, root causes).
Runtime: `venv-rocm714` (torch 2.14.1+rocm7.14, HIP 7.14.60850). Compiler for the HIP kernels: Docker image `rocm/dev-ubuntu-24.04:7.14.1-full`
(same HIP build), see "Building the kernels".

## Result (493-frame test set, median per frame, host venv-rocm714)

| config | encode | query | total | accuracy | notes |
|---|---|---|---|---|---|
| stock HF bf16 (`model.encode_image` + `model.query`) | 223-245 ms | 100 ms | **~340 ms** | 97.8% | baseline (`bench.py --stock`) |
| fast path, Triton GEMMs (previous best): `bench.py --vlin TLin --tlin I8Lin` | 88-95 ms | 9.2 ms | **98-104 ms** | 98.0% | exact model (729 image tokens) |
| fast path, HIP WMMA GEMMs (this round): `bench.py` (default) | 79.6 ms | 9.0 ms | **88.7 ms** (3.8x) | 98.0%, 99.4% agreement with stock | same numerics as the Triton int8/bf16 path (bit-identical GEMM outputs) |
| same + image tokens avg-pooled 27x27 -> 16x16 before the LLM | ~55 ms | 9 ms | **~65 ms** | 98.4% | `--pool 16`; changes the model input (misses 3/46 "found" tv frames) |

Throughput (`test_throughput.py`, 200 distinct frames, tight loop, CPU prep excluded): 10.9 frames/s with one question, 10.6 frames/s (21.3 answers/s) with two
questions per frame. Docker (same image, same code) vs host: 88.4 vs 88.7 ms, 10.9 vs 10.9 frames/s, identical accuracy (< 1% gap) -- see RESULTS.md "FINAL round 3".
Add ~3 ms CPU for JPEG decode + LANCZOS squash (`common.prep`, outside the timed region).

See RESULTS.md for the full ladder, the hardware facts (int8 WMMA is NOT faster than bf16 WMMA on this chip: 57 vs 55 T peak), the discarded
avenues (hipBLASLt int8, fp16, TunableOp+graphs deadlock, all-int8 vision, torch.compile, ...) and the cold-cache / DRAM-channel-aliasing story.

## Files
* `md2fast.py` -- re-implementation of the single-tile encode + yes/no query with static shapes (`FastMD2`).
* `hgemm.py` -- ctypes wrapper for the hand-written HIP WMMA GEMMs (`HLin` bf16, `HI8Lin` int8 W8A8), tiled weight layout, config table.
* `hip/i8gemm.hip` -- the kernels (int8 128x128x128 WMMA GEMM with fused dequant/bias/gelu epilogue, split-K small-M variant, bf16 variant,
  WMMA peak probes); `hip/i8test.cpp`, `hip/bf16test.cpp` C++ harnesses; `hip/hblt_i8.cpp` hipBLASLt probe; `hip/libmdgemm.so` the built library.
* `hip/attn.hip`, `hattn.py` -- WMMA flash-attention prefill kernel (optional, `bench.py --hattn 1`; only ~0.5 ms faster than torch SDPA in situ, see RESULTS.md).
* `tgemm.py` -- Triton GEMMs (previous best; still used for the per-token int8 quantisation kernel), `fused.py` -- fused rope + KV-cache kernel.
* `common.py` -- model loading, test set, metrics, timing. `bench.py` -- reproducible end-to-end benchmark (accuracy + latency, optional stock run).
* Regression tests: `test_hgemm.py` (kernel correctness vs Triton/torch + timing + graph capture), `verify.py` (fast path vs stock model),
  `test_throughput.py` (frames/s), `run_docker_vs_host.sh` (same code inside the container and on the host), `coldbench.py` (cold-cache GEMM bench),
  `insitu_time.py` (per-GEMM in-situ timing).
* `ref/` -- stored baseline predictions / embeddings used for agreement and cosine checks.

## Commands
```bash
cd $SHM_SCRATCH/moondream-opt
venv-rocm714/bin/python bench.py                          # best exact config (HIP kernels), full test set -> accuracy + latency + sanity gate
venv-rocm714/bin/python bench.py --vlin TLin --tlin I8Lin # previous best (Triton GEMMs)
venv-rocm714/bin/python bench.py --stock                  # also the stock HF path in the same process
venv-rocm714/bin/python bench.py --pool 16                # fast mode (pooled image tokens)
venv-rocm714/bin/python bench.py --image $MD_FRAMES/sessA/judged_0060.jpg --question "Is there a mug in this image? Answer yes or no."
venv-rocm714/bin/python test_hgemm.py                     # regression 1: kernels vs Triton (bit-identical) and torch, timing, HIP graph capture
venv-rocm714/bin/python verify.py HLin HI8Lin             # regression 1: fast path vs stock model (top-1 token, logits cosine, embedding cosine)
venv-rocm714/bin/python test_throughput.py --nq 1         # regression 3: frames/s over 200 frames (--nq 2: two questions per frame)
bash run_docker_vs_host.sh                                # regression 4: bench + throughput on the host and inside the ROCm container
```

## Building the kernels (only needed after editing hip/i8gemm.hip)
```bash
docker run --rm -it --device /dev/kfd --device /dev/dri --group-add $(getent group render | cut -d: -f3) --security-opt seccomp=unconfined \
  -v $SHM_SCRATCH:/work rocm/dev-ubuntu-24.04:7.14.1-full \
  bash -c "cd /work/moondream-opt/hip && hipcc -O3 --offload-arch=gfx1151 -fPIC -shared i8gemm.hip -o libmdgemm.so && hipcc -O3 --offload-arch=gfx1151 -fPIC -shared attn.hip -o libmdattn.so && hipcc -O3 --offload-arch=gfx1151 i8gemm.hip i8test.cpp -o i8test && chown 1000:1000 *"
```
The library only depends on `libamdhip64.so.7`, which torch's ROCm wheel already loads, so the host venv can `ctypes.CDLL` it directly.
The container venv used for the Docker comparison was made with `uv venv --python 3.12 venv-c714` and the same torch wheel
(`--index-url https://download.pytorch.org/whl/rocm7.14 torch==2.14.1+rocm7.14 triton-rocm "transformers<5" einops accelerate pillow numpy`).
The host venv: `uv venv --python 3.14 venv-rocm714` + the same packages.

## Using it from an application
```python
from md2fast import FastMD2; import hgemm
fast = FastMD2(model, lin_cls=hgemm.HLin, tlin_cls=hgemm.HI8Lin, n_q=2)   # model = AutoModelForCausalLM.from_pretrained(..., revision="2025-06-21")
fast.set_image(pil_image); fast.set_questions(["Is there a mug in this image? Answer yes or no.", "Is there a tv in this image? Answer yes or no."])
fast.encode(); fast.query()   # warm-up (Triton autotune of the small helper kernels ~10 s), then fast.capture() to freeze into HIP graphs
for pred_yes, _, margin in fast.decide_all(fast.query()): ...
```
`set_image` accepts any PIL image (resized/squashed to 378x378 with LANCZOS, like the stock single-tile path).
