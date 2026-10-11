"""Round 19: summarise host-polled timelines (step8 timeline JSONs) per op kind, median over runs.

Per kind: ops, sum of span (first..last word), sum of gap (first - previous op's last), share (last - previous last).
GEMMs also get the compute model: nt tiles x K/64 k-steps x CYC per k-step (+ GAP per tile) at 1.81 GHz, against their share.
usage: python3 r19/tlsum.py tl_a.json [tl_b.json ...] [--cyc 1603] [--build full12r18m]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/work/npu-tapnext")

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--cyc", type=float, default=1613.0, help="in-situ timer cycles per k-step (KS-P38 r18r6K: 1,613)")
ap.add_argument("--build", default="full12r18m")
o = ap.parse_args()

import plan8  # noqa: E402

bj = json.load(open(f"/work/npu-tapnext/build/{o.build}.json"))
plan8.A_HALF, plan8.P23, plan8.P60G = bool(bj.get("a_half")), bool(bj.get("p23")), bool(bj.get("p60g"))
plan8.XROT, plan8.XPH, plan8.AL1 = int(bj.get("xrot", 0)), 0, bool(bj.get("al1"))
p, _ = plan8.full_step(12)
geo = {op["name"]: op for op in p.ops}

runs = [json.load(open(f))["ops"] for f in o.files]
names = [r["name"] for r in runs[0]]
F = 1.81e6  # cycles per ms


def kind(name, k):
    return f"gemm:{name.split('_', 1)[1]}" if k == "gemm" else k


rows = {}
for r in runs:
    last = 0.0
    acc = {}
    for op in r:
        k = kind(op["name"], op["kind"])
        a = acc.setdefault(k, [0, 0.0, 0.0, 0.0])
        a[0] += 1
        if op["first"] is not None and op["last"] is not None:
            a[1] += op["last"] - op["first"]
            a[2] += op["first"] - last
        if op["last"] is not None:
            a[3] += op["last"] - last
            last = op["last"]
    for k, a in acc.items():
        rows.setdefault(k, []).append(a)
model = {}
for n in names:
    op = geo.get(n)
    if op is None or op["kind"] != "gemm":
        continue
    k = kind(n, "gemm")
    ks = op["nt"] * op["K"] // 64
    model[k] = model.get(k, 0) + ks
tot = 0.0
print(f"{'kind':14s} {'ops':>4s} {'span':>7s} {'gap':>7s} {'share':>7s} {'k-steps':>8s} {'model':>7s} {'share-model':>11s}  (ms, median of {len(runs)})")
for k, v in sorted(rows.items(), key=lambda kv: -np.median([x[3] for x in kv[1]])):
    v = np.array(v)
    med = np.median(v, axis=0)
    tot += med[3]
    m = model.get(k)
    ms = f"{m:8d} {m * o.cyc / F:7.2f} {med[3] - m * o.cyc / F:11.2f}" if m else ""
    print(f"{k:14s} {int(med[0]):4d} {med[1]:7.2f} {med[2]:7.2f} {med[3]:7.2f} {ms}")
print(f"total share {tot:.2f} ms")
