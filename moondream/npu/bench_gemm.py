"""NPU GEMM benchmark at moondream2 vision-encoder shapes using mlir-aie's whole_array design.

Per shape: pad M/K/N to the design's tile constraints, run warmup+iters, and report
  kernel  = submit + wait of one dispatch (aie runtime's npu_time; includes command latency)
  sync_in = host->device cache sync of A (B = weights would stay resident in a real pipeline)
  sync_out= device->host sync of C
Numerics vs a float64 CPU reference on the same (bf16/int8) inputs.
  python bench_gemm.py --dtype bf16 --out f32 --tiles 64,64,32 [--shapes name ...]
"""

import argparse
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, "/work/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array")
import aie.iron as iron  # noqa: E402
from aie.iron.device import from_name  # noqa: E402
from aie.utils.hostruntime import set_current_device  # noqa: E402
from whole_array import whole_array  # noqa: E402

# (name, M, K, N, calls per encode) -- moondream2 rev 2025-06-21 vision tower + projection, one 378x378 crop
SHAPES = [
    ("patch_emb", 729, 588, 1152, 1),
    ("qkv", 729, 1152, 3456, 27),
    ("attn_out", 729, 1152, 1152, 27),
    ("fc1", 729, 1152, 4304, 27),
    ("fc2", 729, 4304, 1152, 27),
    ("proj_fc1", 729, 2304, 8192, 1),
    ("proj_fc2", 729, 8192, 2048, 1),
    ("sq2048", 2048, 2048, 2048, 0),
    ("sq4096", 4096, 4096, 4096, 0),
]


def up(x, a):
    return (x + a - 1) // a * a


def stats(xs):
    return statistics.median(xs), min(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "i8"])
    ap.add_argument("--out", default="f32")
    ap.add_argument("--tiles", default="64,64,32")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--bfp16", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", nargs="*")
    args = ap.parse_args()
    m, k, n = (int(v) for v in args.tiles.split(","))
    din = bfloat16 if args.dtype == "bf16" else np.int8
    dout = {"f32": np.float32, "bf16": bfloat16, "i32": np.int32, "i16": np.int16, "i8": np.int8}[args.out]
    set_current_device(from_name("npu2", n_cols=None))
    rng = np.random.default_rng(0)
    print(f"# dtype={args.dtype}->{args.out} tiles m,k,n={m},{k},{n} cols={args.cols} bfp16={args.bfp16}")
    for name, M, K, N, calls in SHAPES:
        if (args.shapes and name not in args.shapes) or (not args.shapes and calls == 0):
            continue
        Mp, Kp, Np = up(M, 4 * m), up(K, k), up(N, n * args.cols)
        if din is np.int8:
            a = rng.integers(-127, 128, (M, K)).astype(np.int8)
            b = rng.integers(-127, 128, (K, N)).astype(np.int8)
        else:  # activation-like and weight-like magnitudes
            a = rng.standard_normal((M, K)).astype(bfloat16)
            b = (rng.standard_normal((K, N)) * 0.03).astype(bfloat16)
        ap_ = np.zeros((Mp, Kp), din)
        bp = np.zeros((Kp, Np), din)
        ap_[:M, :K] = a
        bp[:K, :N] = b
        t0 = time.perf_counter()
        try:
            design = whole_array.specialize(
                A=np.ndarray[(Mp, Kp), np.dtype[din]],
                B=np.ndarray[(Kp, Np), np.dtype[din]],
                C=np.ndarray[(Mp, Np), np.dtype[dout]],
                M=Mp, K=Kp, N=Np, m=m, k=k, n=n, n_aie_cols=args.cols,
                emulate_bf16_mmul_with_bfp16=bool(args.bfp16),
            )
            A = iron.tensor(ap_, dtype=din, device="npu")
            B = iron.tensor(bp, dtype=din, device="npu")
            C = iron.zeros((Mp, Np), dtype=dout, device="npu")
            design(A, B, C)  # compile + first run
        except Exception as e:  # report and continue with the next shape
            print(f"{name:9s} {Mp}x{Kp}x{Np}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        tcomp = time.perf_counter() - t0
        kern, e2e, sin, sout = [], [], [], []
        for i in range(3 + args.iters):
            t0 = time.perf_counter_ns()
            A._sync_to_device()
            t1 = time.perf_counter_ns()
            r = design(A, B, C)
            t2 = time.perf_counter_ns()
            C._sync_from_device()
            t3 = time.perf_counter_ns()
            if i >= 3:
                sin.append((t1 - t0) / 1e3)
                kern.append(r[1].npu_time / 1e3)
                e2e.append((t2 - t1) / 1e3)
                sout.append((t3 - t2) / 1e3)
        out = np.asarray(C.numpy()).reshape(Mp, Np)[:M, :N].astype(np.float64)
        ref = a.astype(np.float64) @ b.astype(np.float64)
        err = np.abs(out - ref)
        scale = np.abs(ref).max()
        relmax = err.max() / scale
        relrms = np.sqrt((err**2).mean() / (ref**2).mean())
        kmed, kmin = stats(kern)
        flops = 2 * M * K * N
        print(
            f"{name:9s} {M}x{K}x{N} pad {Mp}x{Kp}x{Np} x{calls}: kernel med {kmed:8.1f} min {kmin:8.1f} us "
            f"| call {statistics.median(e2e):8.1f} us | sync_in {statistics.median(sin):6.1f} sync_out {statistics.median(sout):6.1f} us "
            f"| {flops / kmed / 1e6:6.2f} TOPS useful ({2 * Mp * Kp * Np / kmed / 1e6:6.2f} padded) "
            f"| err max_abs {err.max():.3g} max/|ref|max {relmax:.2e} rms_rel {relrms:.2e} | compile {tcomp:.1f}s "
            f"| per-encode {kmed * calls / 1e3:6.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
