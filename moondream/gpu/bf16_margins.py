"""Margin shifts of the bf16-elementwise configs vs the GPU path, on rescore.py's 462 frames.

    python3 bf16_margins.py
"""

import contextlib
import io
import json
import os
import runpy
import statistics
from pathlib import Path

P = json.loads((Path(os.environ.get("MD_FRAMES", "frames/sessions")).resolve().parent / "calibration/frame_preds.json").read_text())
with contextlib.redirect_stdout(io.StringIO()):
    truth = runpy.run_path("rescore.py")["truth"]  # key -> (obj, present), same frames as rescore

PAIRS = [(n, "exact") for n in ("npu_bfp16", "bf16_ln", "bf16_gelu", "bf16_softmax", "bf16_all",
                                "bf16_gelusig", "bf16_all_sig", "npu_full", "npu_full_sig", "npu_full_sig_bfpattn")]
PAIRS += [(n, "pool16") for n in ("npu_full_pool16", "npu_full_sig_pool16", "npu_full_sig_bfpattn_pool16")]

print(f"{'config':20} {'ref':6} {'mean|dm|':>8} {'p95':>5} {'max':>5} {'flips':>5} {'near0':>5} {'min pos m':>9}")
for name, ref in PAIRS:
    d = sorted(abs(P[name][k]["margin"] - P[ref][k]["margin"]) for k in truth)
    flips = [k for k in truth if P[name][k]["yes"] != P[ref][k]["yes"]]
    near0 = sum(abs(P[ref][k]["margin"]) < 0.8 for k in flips)  # flips of frames the GPU path already called within 0.8
    pos = min(P[name][k]["margin"] for k, (_, t) in truth.items() if t)
    print(f"{name:20} {ref:6} {statistics.mean(d):8.2f} {d[int(0.95 * len(d))]:5.2f} {d[-1]:5.2f} {len(flips):5} {near0:5} {pos:+9.1f}")
for ref in ("exact", "pool16"):
    print(f"{ref}: min margin on a true positive {min(P[ref][k]['margin'] for k, (_, t) in truth.items() if t):+.1f}")
for name in ("npu_full_pool16", "npu_full_sig_pool16", "npu_full_sig_bfpattn_pool16"):
    for k in truth:
        if P[name][k]["yes"] != P["pool16"][k]["yes"]:
            print(f"  {name} flips {k} truth={truth[k][1]} {P['pool16'][k]['margin']:+.1f} -> {P[name][k]['margin']:+.1f}")
