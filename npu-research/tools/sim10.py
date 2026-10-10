"""sim10: the per-op cost model of design8, rebuilt on the round-7/8 in-situ cycle profiles (npu-research rev 3).

What changed from sim9 (ANALYSIS rev 3 section 1):
  - The GEMM k-step is no longer a free fitted constant. It is composed from the deployed binaries (llvm-objdump of the
    linked core elfs, r8/core02_*.dis) plus at most two dynamic terms:
        t_ks = (9 * (CALL + s) + IRON_KS + d) / F
    CALL = bundles per GEMM call (C++ mm_acc 321; asm: 34-bundle C++ wrapper + 225 mm_core = 259), IRON_KS = 89 bundles
    of IRON control code per k-step (identical in r7a/f/g/p), s = in-call stall cycles per call (0 for the C++ loop:
    the r7p stamps measured it at its static schedule), d = dynamic cycles per k-step outside the calls (lock latency,
    ring waits; the r7p stamps say ~15).
  - Every GEMM op pays L_dep (the round-8 g11 stamps show a 27-50 us first-tile wait in EVERY op, "prefetchable" or not:
    the memtile A slot is single-buffered, so op i+1's slot load starts only after op i's last send).
  - yx and qkv carry their measured extra A-ring waits (r7p full step: +5.04 / +1.23 us per tile over the steady
    2.66 us per 12 k-steps).
  - Two builds are fitted jointly: shared GEMM / EW constants, per-build attention, and the asm stall s for r7g.
    The fitted s is then checked against the g11 PROF stamps and the bank microbenchmark (mmb).

usage: python3 sim10.py            # joint fit of full12r7a + full12r7g, the gap decomposition, what-ifs, P14 re-price
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent.parent / "npu-tapnext" / "build"
F = 1.81e9  # MEASURED (npu-tapnext r7/clk.py, Turbo)
IRON_KS = 89  # DOC: bundles 0x170-0x35f of core_0_2 (r7a, r7f, r7g, r7p identical)
CALL = {"full12r7a": 321, "full12r7g": 34 + 225}  # DOC: C++ mm_acc dynamic path; asm = wrapper + mm_core
EXTRA_TILE = {"yx": 5.04, "qkv": 1.23}  # MEAS r7p (us per tile): median k-wait above the steady 2.66 us

# name, kind, K, nt (N-tiles per core), gelu tiles per core
BLOCK = [
    ("norm0", "norm"), ("yx", "gemm", 768, 8, 4), ("ssm", "ssm"), ("out", "gemm", 768, 4, 0),
    ("norm1", "norm"), ("up_a", "gemm", 768, 16, 8), ("up_b", "gemm", 768, 16, 8),
    ("down_a", "gemm", 1536, 4, 0), ("down_b", "gemm", 1536, 4, 0), ("norm2", "norm"),
    ("qkv", "gemm", 768, 12, 0), ("attn", "attn"), ("out_proj", "gemm", 768, 4, 0), ("norm3", "norm"),
    ("mlp0", "gemm", 768, 16, 16), ("mlp3_a", "gemm", 1536, 4, 0), ("mlp3_b", "gemm", 1536, 4, 0),
]
KS_BLOCK = sum(o[3] * o[2] // 64 for o in BLOCK if o[1] == "gemm")  # 1296 tile-k-steps per core per block
TILES_BLOCK = sum(o[3] for o in BLOCK if o[1] == "gemm")  # 92
SHARED = ["L_dep", "t_tile", "t_gelu", "d", "norm0", "norm1", "norm2", "norm3", "ssm"]


def load(path):
    d = json.loads(Path(path).read_text())
    return d, [(o["name"], o["last"]) for o in d["ops"]]


def op_terms(o, build):
    """(constant us, {param: coefficient}) of one op for one build."""
    if o[1] == "gemm":
        _, _, K, nt, g = o
        ks = nt * K // 64
        const = ks * (9 * CALL[build] + IRON_KS) / F * 1e6 + nt * EXTRA_TILE.get(o[0], 0.0)
        coef = {"L_dep": 1.0, "t_tile": nt, "t_gelu": g, "d": ks / F * 1e6}
        if build != "full12r7a":
            coef["s_" + build] = 9 * ks / F * 1e6
        return const, coef
    if o[1] == "norm":
        return 0.0, {o[0]: 1.0}
    if o[1] == "attn":
        return 0.0, {"attn_" + build: 1.0}
    return 0.0, {"ssm": 1.0}


def spans(ops, build, blocks=range(1, 11)):
    """(const, coef, measured us, label) per span between visible last writes (sim9's segment method)."""
    kinds = {o[0]: o for o in BLOCK}
    out, const, coef, lab, prev = [], 0.0, {}, [], None
    for name, last in ops:
        if name == "final":
            break
        b, op = name.split("_", 1)
        b = int(b[1:])
        if b < blocks.start - 1 or b >= blocks.stop:
            prev = None
            continue
        if b == blocks.start - 1:
            prev = last if op == "mlp3_b" else None
            continue
        c, k = op_terms(kinds[op], build)
        const += c
        for p, v in k.items():
            coef[p] = coef.get(p, 0.0) + v
        lab.append(op)
        if last is not None and prev is not None:
            out.append((const, dict(coef), (last - prev) * 1e3, "+".join(lab)))
        if last is not None:
            const, coef, lab, prev = 0.0, {}, [], last
    return out


def fit(builds):
    rows = []
    for b in builds:
        d, ops = load(BUILD / f"tl_{b}_d12.json") if (BUILD / f"tl_{b}_d12.json").exists() else load(
            next(BUILD.glob(f"tl_{b}_d12*.json")))
        rows += [(b,) + r for r in spans(ops, b)]
    params = SHARED + [f"attn_{b}" for b in builds] + [f"s_{b}" for b in builds if b != "full12r7a"]
    A = np.array([[r[2].get(p, 0.0) for p in params] for r in rows])
    y = np.array([r[3] - r[1] for r in rows])
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = A @ x - y
    return dict(zip(params, x)), rows, res


def block_parts(p, build, over=None):
    """per-block us by part for one build (over: parameter overrides, plus 'call', 'iron', 'extra' keys)."""
    q = dict(p, **(over or {}))
    call = q.get("call", CALL[build])
    s = q.get("s_" + build, 0.0) if build != "full12r7a" else q.get("s_full12r7a", 0.0)
    iron = q.get("iron", IRON_KS)
    parts = dict(mac=0.0, stall=0.0, callovh=0.0, dyn=0.0, ldep=0.0, tile=0.0, gelu=0.0, ringx=0.0)
    for o in BLOCK:
        if o[1] != "gemm":
            continue
        _, _, K, nt, g = o
        ks = nt * K // 64
        mac_cyc = q.get("mac_cyc", 225 - 0)  # cycles of the call that are the MAC schedule (asm) - split below
        parts["mac"] += ks * 9 * min(call, mac_cyc) / F * 1e6
        parts["callovh"] += ks * (9 * max(call - mac_cyc, 0) + iron) / F * 1e6
        parts["stall"] += ks * 9 * s / F * 1e6
        parts["dyn"] += ks * q["d"] / F * 1e6
        parts["ldep"] += q["L_dep"]
        parts["tile"] += nt * q["t_tile"]
        parts["gelu"] += g * q["t_gelu"]
        parts["ringx"] += nt * q.get("extra", EXTRA_TILE).get(o[0], 0.0)
    parts["attn"] = q["attn_" + build]
    parts["ssm"] = q["ssm"]
    parts["norms"] = sum(q[f"norm{i}"] for i in range(4))
    return parts


def main():
    builds = ["full12r7a", "full12r7g"]
    p, rows, res = fit(builds)
    print("sim10 joint fit, blocks 1-10 of " + " + ".join(builds))
    for b in builds:
        r = np.array([x for x, row in zip(res, rows) if row[0] == b])
        print(f"  {b}: {len(r)} spans, span rms {np.sqrt(np.mean(r ** 2)):.1f} us, mean residual {r.mean():+.2f} us")
    print(f"  all: span rms {np.sqrt(np.mean(res ** 2)):.1f} us")
    for k, v in p.items():
        print(f"    {k:16s} {v:9.2f}")
    s = p["s_full12r7g"]
    print(f"\nasm in-call stall s = {s:.1f} cycles per call (C++ loop: 0, measured); "
          f"dynamic per k-step d = {p['d']:.0f} cycles")
    for b in builds:
        t = (9 * (CALL[b] + (s if b == "full12r7g" else 0)) + IRON_KS + p["d"]) / F * 1e6
        parts = block_parts(p, b, {"mac_cyc": 225 if b == "full12r7g" else 302})
        tot = sum(parts.values())
        print(f"\n{b}: t_ks {t:.3f} us; block {tot / 1e3:.3f} ms -> depth 12 {(12 * tot + 150) / 1e3:.1f} ms (+0.15 tail)")
        for k, v in parts.items():
            print(f"    {k:8s} {v:7.1f} us/block  {12 * v / 1e3:6.2f} ms/step  {100 * v / tot:5.1f}%")
    # the gap: -7.4 extrapolated vs -1.9 measured
    n = KS_BLOCK * 12
    print(f"\nGap (depth 12, {n} k-steps per core):")
    print(f"  extrapolated (96 cyc/call saved):            {-n * 9 * 96 / F * 1e3:6.2f} ms")
    print(f"  static, with the 34-bundle wrapper (62/call): {-n * 9 * 62 / F * 1e3:6.2f} ms")
    print(f"  + fitted in-call stall s = {s:.0f}/call:          {n * 9 * s / F * 1e3:+6.2f} ms")
    print(f"  = model {-n * 9 * (62 - s) / F * 1e3:6.2f} ms; measured (step8 time, r7f -> r7g) -1.91 ms")
    what_ifs(p)


# ---- scenarios on the r7g fit: per-op costs, serial and two frames in flight (sim9's list scheduler) ----------------
def op_cost(q, o):
    """us of one op. q: k-step cycles 'ks_cyc', 'L_dep', 't_tile', 't_gelu', 'ringx' (dict), EW / attention times."""
    if o[1] == "gemm":
        _, _, K, nt, g = o
        return q["L_dep"] + nt * (K // 64) * q["ks_cyc"] / F * 1e6 + nt * (q["t_tile"] + q["ringx"].get(o[0], 0.0)) \
            + g * q["t_gelu"]
    if o[1] == "norm":
        return q[o[0]]
    return q[o[1]]


def base_q(p):
    """r7g as fitted: k-step = 9 x (259 + s) + 89 + d."""
    q = dict(p)
    q["attn"] = p["attn_full12r7g"]
    q["call"], q["s"], q["iron"] = 259, p["s_full12r7g"], IRON_KS
    q["ringx"] = dict(EXTRA_TILE)
    return q


def finish(q):
    q = dict(q)
    q["ks_cyc"] = 9 * (q["call"] + q["s"]) + q["iron"] + q["d"]
    return q


def step_ms(q, depth=12):
    q = finish(q)
    return (depth * sum(op_cost(q, o) for o in BLOCK) + 150) / 1e3


def schedule(q, frames=6, depth=12, inflight=2):
    """Two resources: G = the 24 GEMM cores (GEMM ops + attention), E = the 8 EW cores (norms + SSM). Frame f's SSM of
    block b waits for frame f-1's (the state); at most `inflight` frames started. Returns (us per frame, latency us)."""
    q = finish(q)
    names = [o[0] for o in BLOCK]
    jobs = []
    for fr in range(frames):
        for b in range(depth):
            for o in BLOCK:
                jobs.append(dict(fr=fr, b=b, op=o[0], res="E" if o[1] in ("norm", "ssm") else "G", cost=op_cost(q, o)))
    idx = {(j["fr"], j["b"], j["op"]): i for i, j in enumerate(jobs)}
    deps = []
    for j in jobs:
        k = names.index(j["op"])
        d = []
        if k > 0:
            d.append(idx[(j["fr"], j["b"], names[k - 1])])
        elif j["b"] > 0:
            d.append(idx[(j["fr"], j["b"] - 1, names[-1])])
        if j["op"] == "ssm" and j["fr"] > 0:
            d.append(idx[(j["fr"] - 1, j["b"], "ssm")])
        if j["fr"] >= inflight and j["b"] == 0 and k == 0:
            d.append(idx[(j["fr"] - inflight, depth - 1, names[-1])])
        deps.append(d)
    free, start, end = {"G": 0.0, "E": 0.0}, [None] * len(jobs), [None] * len(jobs)
    for _ in range(len(jobs)):
        best = None
        for i, j in enumerate(jobs):
            if start[i] is not None or any(end[d] is None for d in deps[i]):
                continue
            t = max([end[d] for d in deps[i]] + [free[j["res"]]])
            key = (t, j["fr"], j["b"])
            if best is None or key < best[0]:
                best = (key, i)
        (t, *_), i = best
        start[i], end[i] = t, t + jobs[i]["cost"]
        free[jobs[i]["res"]] = end[i]
    lat = [max(end[i] for i, j in enumerate(jobs) if j["fr"] == f) - min(start[i] for i, j in enumerate(jobs) if j["fr"] == f)
           for f in range(frames)]
    per = (max(end) - lat[0]) / (frames - 1)
    return per, max(lat[1:])


LEVERS = [  # name, overrides on the r7g q
    ("SSM 1.32 -> 0.5 ms per block (P13)", dict(ssm=500.0)),
    ("one call per k-step, stripes pipelined (P20)", dict(iron=40, call=225 - 25)),
    ("t_tile halved (C handoff, acc_zero folded; P11)", "tile_half"),
    ("bank-aware mm_core: in-call stall 44 -> 5 (P21)", dict(s=5.0)),
    ("norms folded into the GEMMs, 40 us each (P7)", dict(norm0=40.0, norm1=40.0, norm2=40.0, norm3=40.0)),
    ("attention 0.94 -> 0.6 ms per block (P12/P19)", dict(attn=600.0)),
    ("2x2 register blocking: mm_core 225 -> 150 (P18 full)", dict(call=150 + 34)),
    ("A slot chunk-streamed: L_dep -> 5 us (P5)", dict(L_dep=5.0)),
    ("yx / qkv extra A-ring waits removed (P23)", dict(ringx={})),
]


def apply(q, ov):
    q = dict(q)
    if ov == "tile_half":
        q["t_tile"] = q["t_tile"] / 2
    else:
        q.update(ov)
    return q


def what_ifs(p):
    q0 = base_q(p)
    base = step_ms(q0)
    print(f"\nLevers, each alone on the r7g fit (depth 12 serial; base {base:.1f} ms):")
    rows = sorted((step_ms(apply(q0, ov)) - base, name) for name, ov in LEVERS)
    for dlt, name in rows:
        print(f"  {dlt:+6.1f} ms  {name}")
    allq = q0
    for name, ov in LEVERS:
        if "2x2" not in name and "one call" not in name:
            allq = apply(allq, ov)
    gemm3 = dict(s=5.0, iron=40, call=200)  # bank-aware + one call per k-step, 4-chain mm_core
    gemm4 = dict(s=5.0, iron=40, call=150)  # ... with the 2x2 schedule
    print("\nCombined (serial | two frames in flight: ms per frame, fps, latency):")
    for name, q in (("today r7g", q0),
                    ("GEMM: bank + one call/k-step", apply(q0, gemm3)),
                    ("GEMM: + 2x2", apply(q0, gemm4)),
                    ("all levers, 4-chain GEMM", apply(allq, gemm3)),
                    ("all levers + 2x2 GEMM", apply(allq, gemm4))):
        for depth in (12, 10):
            ser = step_ms(q, depth)
            per, lat = schedule(q, depth=depth)
            print(f"  {name:30s} d{depth:2d}: serial {ser:5.1f} ms ({1e3 / ser:4.1f} fps) | 2 in flight {per / 1e3:5.1f} ms "
                  f"({1e6 / per:4.1f} fps), latency {lat / 1e3:5.1f} ms")


if __name__ == "__main__":
    main()
