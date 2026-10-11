"""Round 25: does plan8 (as edited now) still give a build's recorded runtime plans? usage (npu-iron): python3 r25/schedchk.py NAME"""
import json
import sys

sys.path.insert(0, ".")
import plan8  # noqa: E402

name = sys.argv[1]
bj = json.load(open(f"build/{name}.json"))
plan8.A_HALF, plan8.P23, plan8.P60G = bool(bj.get("a_half")), bool(bj.get("p23")), bool(bj.get("p60g"))
plan8.XROT = int(bj.get("xrot", 0))
for k in ("il", "dapre", "a3", "yxr", "yxh", "zdrop", "rrtwin", "sres", "al1"):
    setattr(plan8, k.upper(), bool(bj.get(k, False)))
plan8.MSPLIT, plan8.NRES = tuple(bj.get("msplit", [])), tuple(bj.get("nres", []))
for ph in range(max(plan8.XROT, 1)):
    p, _ = plan8.full_step(12, phase=ph)
    f = f"build/{name}" + (f"_p{ph}" if ph else "") + ".sched"
    print(f, "SAME" if open(f).read() == p.sched_json() else "DIFFERENT")
