"""Round 19: bytes each GEMM op's drains move to DDR per C object (vs the 18,432 B the core DMA sends), block 0."""
import json
import sys

sys.path.insert(0, "/work/npu-tapnext")
import plan8  # noqa: E402

bj = json.load(open("/work/npu-tapnext/build/full12r18m.json"))
plan8.A_HALF, plan8.P23, plan8.P60G = bool(bj.get("a_half")), bool(bj.get("p23")), bool(bj.get("p60g"))
plan8.XROT, plan8.XPH, plan8.AL1 = int(bj.get("xrot", 0)), 0, bool(bj.get("al1"))
p, _ = plan8.full_step(12)


def tbytes(t):
    n = 1
    for s in t["sizes"]:
        n *= s
    return 4 * n


for op in p.ops:
    if op["kind"] != "gemm" or not op["name"].startswith("b0_"):
        continue
    c0 = op["c"][0]
    taps = c0 if isinstance(c0, list) else [c0]
    tot = sum(tbytes(t) for t in taps)
    nobj = 3 * op["nt"] // (2 if op["epi"] == "gated" else 1)
    print(f"{op['name']:12s} epi {op['epi']:9s} nt {op['nt']:3d} column bytes {tot:8d} per object {tot / nobj:8.0f} ({len(taps)} taps)")
