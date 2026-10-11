"""Per-op cost model of design8, refitted to the round-6 host-polled timelines (step8.py timeline JSON).

Successor of sim8.py (round 5). Pure python + numpy, no device. Usage:
    python3 sim9.py                      # fit every round-6 build found in ../../tapnext/npu/build
    python3 sim9.py tl_a.json tl_b.json  # fit the given timelines

What changed from sim8:
  - Reads the JSON timelines (first / last write per op) instead of the text output.
  - Segment fit: when an op's last write is invisible (the SSM always; many ops in the garbage-output twin builds),
    the time between two VISIBLE last writes is the sum of the ops in between. Every unknown (GEMM law, norms,
    attention, SSM) is fitted jointly by least squares over blocks 1-10, so twin builds with holes fit too.
  - The clock is 1.8 GHz (amdxdna npu4/npu5 DPM table, top level, MACHINE.md section 1), not 1.43.
  - Static cycle counts of mm_acc (r6/mm8o2.s, asmstat) give the compute-only k-step to compare against the fit.

GEMM op law (unchanged): L_dep [producer is the op right before it, or an EW op] + nt (K/64 t_ks + t_tile) + g t_gelu.
"""

import json
import sys
from pathlib import Path

import numpy as np

F = 1.8e9  # AIE clock at the top DPM level (Default with an open context, or Turbo)
# mm_acc static schedule (r6/mm8o2.s): entry 13 + 2 row blocks x (prologue 48 + 3 x 28-bundle loop + tail 19) + ret 6,
# plus ~6 bundles of IRON control code per call (acq A / jl / rel A, objdump of full12r6e core_0_2), 9 calls per k-step
# and ~20 bundles per k-step for the B object and the loop.
CALL = 13 + 2 * (48 + 3 * 28 + 19) + 6
KSTEP_STATIC = 9 * (CALL + 6) + 20
TWICE_EXTRA = 9 * 2 * (48 + 3 * 28 + 19)  # MM_TWICE runs both row blocks twice
KSTEPS_PER_BLOCK = 1296  # tile-k-steps per GEMM core per block (sum over the 11 GEMM ops of nt x K/64)

# name, kind, K, nt (N-tiles per core), gelu tiles per core, prefetchable (producer is not the op right before it)
BLOCK = [
    ("norm0", "norm", 3), ("yx", "gemm", 768, 8, 4, False), ("ssm", "ssm"), ("out", "gemm", 768, 4, 0, False),
    ("norm1", "norm", 2), ("up_a", "gemm", 768, 16, 8, False), ("up_b", "gemm", 768, 16, 8, True),
    ("down_a", "gemm", 1536, 4, 0, True), ("down_b", "gemm", 1536, 4, 0, True), ("norm2", "norm", 3),
    ("qkv", "gemm", 768, 12, 0, False), ("attn", "attn"), ("out_proj", "gemm", 768, 4, 0, False), ("norm3", "norm", 2),
    ("mlp0", "gemm", 768, 16, 16, False), ("mlp3_a", "gemm", 1536, 4, 0, False), ("mlp3_b", "gemm", 1536, 4, 0, True),
]
GEMM_P = ["L_dep", "t_tile", "t_ks", "t_gelu"]
OTHER_P = ["norm0", "norm1", "norm2", "norm3", "attn", "ssm"]
PARAMS = GEMM_P + OTHER_P


def load(path):
    d = json.loads(Path(path).read_text())
    return d, [(o["name"], o["last"]) for o in d["ops"]]


def op_row(o):
    """Coefficients of one op on PARAMS."""
    r = dict.fromkeys(PARAMS, 0.0)
    if o[1] == "gemm":
        _, _, K, nt, g, pre = o
        r["L_dep"] = 0.0 if pre else 1.0
        r["t_tile"] = nt
        r["t_ks"] = nt * (K // 64)
        r["t_gelu"] = g
    elif o[1] == "norm":
        r[o[0]] = 1.0
    else:
        r[o[1]] = 1.0
    return [r[p] for p in PARAMS]


def segments(ops, blocks=range(1, 11)):
    """(coefficient row, measured us, label) per span between two visible last writes inside the fitted blocks."""
    kinds = {o[0]: o for o in BLOCK}
    out, acc, lab, prev = [], None, [], None
    for name, last in ops:
        if name == "final":
            break
        b, op = name.split("_", 1)
        b = int(b[1:])
        if b < blocks.start - 1 or b >= blocks.stop:
            prev = None
            continue
        if b == blocks.start - 1:  # the last op of the previous block anchors the first span
            prev = last if op == "mlp3_b" else None
            continue
        row = op_row(kinds[op])
        acc = row if acc is None else [a + x for a, x in zip(acc, row)]
        lab.append(op)
        if last is not None and prev is not None:
            out.append((acc, (last - prev) * 1e3, "+".join(lab)))
        if last is not None:
            acc, lab, prev = None, [], last
    return out


def fit(ops, fixed=None):
    """Least squares on all spans. `fixed`: params held at given values (e.g. t_gelu when no build separates it)."""
    seg = segments(ops)
    fixed = fixed or {}
    free = [p for p in PARAMS if p not in fixed]
    A = np.array([[r[PARAMS.index(p)] for p in free] for r, _, _ in seg])
    y = np.array([m - sum(r[PARAMS.index(p)] * v for p, v in fixed.items()) for r, m, _ in seg])
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    p = dict(fixed, **dict(zip(free, x)))
    res = A @ x - y
    return p, float(np.sqrt(np.mean(res**2))), seg, res


def block_us(p, over=None):
    q = dict(p, **(over or {}))
    return sum(float(np.dot(op_row(o), [q[k] for k in PARAMS])) for o in BLOCK)


def measured_block_us(ops):
    """Mean block time over b1-b10: from the last visible write of block 0 to the last visible write of block 10
    (the twin builds hide some ops; the same op is used at both ends)."""
    last = dict(ops)
    for op in ("mlp3_b", "mlp3_a", "mlp0", "norm3", "out_proj"):
        a, b = last.get(f"b0_{op}"), last.get(f"b10_{op}")
        if a is not None and b is not None:
            return (b - a) / 10 * 1e3
    return float("nan")


def report(path, fixed=None):
    d, ops = load(path)
    p, rms, seg, res = fit(ops, fixed)
    name = d["build"]
    mb, pb = measured_block_us(ops), block_us(p)
    print(f"\n{name}: dispatch {d['dispatch_ms']:.2f} ms, {len(seg)} spans, span rms {rms:.1f} us")
    print("  t_ks {t_ks:.3f} us ({cyc:.0f} cycles at 1.8 GHz; static compute {st} cycles = {sus:.3f} us)".format(
        cyc=p["t_ks"] * F / 1e6, st=KSTEP_STATIC, sus=KSTEP_STATIC / F * 1e6, **p))
    print("  L_dep {L_dep:.1f}  t_tile {t_tile:.2f}  t_gelu {t_gelu:.2f} us/tile   norms {n0:.0f}/{n1:.0f}/{n2:.0f}/{n3:.0f}"
          "  attn {attn:.0f}  ssm {ssm:.0f} us".format(n0=p["norm0"], n1=p["norm1"], n2=p["norm2"], n3=p["norm3"], **p))
    print(f"  block: model {pb / 1e3:.3f} ms vs measured mean (b1-b10) {mb / 1e3:.3f} ms ({(pb - mb) / mb * 100:+.1f}%)")
    worst = sorted(zip(res, seg), key=lambda t: -abs(t[0]))[:3]
    print("  worst spans: " + "; ".join(f"{s[2]} {r:+.0f} us" for r, s in worst))
    return name, p, rms, mb


def main():
    if sys.argv[1:2] == ["gantt"]:  # steady-state window of the 2-frames-in-flight schedule, 2x2-loop scenario
        d, ops = load(Path(__file__).resolve().parent.parent.parent / "tapnext/npu/build/tl_full12r6e_d12.json")
        p = fit(ops)[0]
        for name in ("today (r6e)", "serial fixes, 2x2 loop"):
            q = dict(p, **scenarios(p)[name])
            span, lat = schedule(q, frames=6, inflight=2)
            per = (span - lat[0]) / 5
            t0 = 3 * per
            print(f"{name}: 2 frames in flight, window {t0 / 1e3:.1f}-{(t0 + 2 * block_us(q)) / 1e3:.1f} ms")
            gantt(q, t0, t0 + 2 * block_us(q), frames=6, inflight=2)
        return
    here = Path(__file__).resolve().parent
    bdir = here.parent.parent / "tapnext" / "npu" / "build"
    files = sys.argv[1:] or [str(bdir / f"tl_full12{b}_d12.json") for b in ("m", "r6a", "r6b", "r6e", "p1n", "p1t")]
    fits = {}
    for f in files:
        if Path(f).exists():
            n, p, rms, mb = report(f)
            fits[n] = p
    base, nomac, twice = fits.get("full12r6a"), fits.get("full12p1n"), fits.get("full12p1t")
    if base and nomac and twice:
        print("\nTwins (per tile-k-step, us):")
        print(f"  base {base['t_ks']:.3f}  no-MAC {nomac['t_ks']:.3f}  2x-MAC {twice['t_ks']:.3f}")
        add = twice["t_ks"] - base["t_ks"]
        print(f"  2x adds {add:.3f} us = {add * F / 1e6:.0f} cycles at 1.8 GHz vs {TWICE_EXTRA} static bundles "
              f"-> implied clock {TWICE_EXTRA / add / 1e3:.2f} GHz if the added loop never stalls (a lower bound)")
        print(f"  whole step: {KSTEPS_PER_BLOCK * 12} k-steps x {TWICE_EXTRA} bundles = "
              f"{KSTEPS_PER_BLOCK * 12 * TWICE_EXTRA / F * 1e3:.1f} ms at 1.8 GHz, "
              f"{KSTEPS_PER_BLOCK * 12 * TWICE_EXTRA / 1.43e9 * 1e3:.1f} ms at 1.43 GHz (measured +24.7)")
        print(f"  data movement per k-step (no-MAC t_ks) {nomac['t_ks']:.3f} us: 10,368 B of A -> "
              f"{10368 / (nomac['t_ks'] * 1e-6) / 1e9:.1f} GB/s = {10368 / (nomac['t_ks'] * F / 1e6):.1f} B/cycle per core")
    p = fits.get("full12r6e")
    if p:
        what_ifs(p)


# Hand-asm GEMM loop estimates (static cycles per mm_acc call, see ANALYSIS section 6): the stall factor measured
# today (fitted t_ks / static) is applied to them.
CALL_18 = 13 + 2 * (30 + 3 * 18 + 12) + 6  # today's data flow, loop at the ldb bound (16 B pops + 2 fills)
CALL_2X2 = 4 * 34 + 24  # 2x2 register blocking: A pops on lda, B pops on ldb, 4 accumulators, pipelined across passes


def t_ks_for(p, call):
    return p["t_ks"] * (9 * (call + 6) + 20) / KSTEP_STATIC


def scenarios(p):
    s = {"today (r6e)": {}}
    serial = dict(L_dep=5.0, t_tile=p["t_tile"] / 2, attn=600.0, ssm=500.0, norm0=40.0, norm1=40.0, norm2=40.0,
                  norm3=40.0)
    s["serial fixes, 18-bundle loop"] = dict(serial, t_ks=t_ks_for(p, CALL_18))
    s["serial fixes, 2x2 loop"] = dict(serial, t_ks=max(t_ks_for(p, CALL_2X2), 0.753))
    return s


def what_ifs(p):
    """Serial what-ifs on the r6e fit (depth 12, ms; final norm + host-side tail ~0.15 ms)."""
    final = 150.0
    base = 12 * block_us(p) + final
    print(f"\nwhat-if on full12r6e (serial schedule, depth 12): base {base / 1e3:.1f} ms")
    cases = [
        ("GEMM loop 28 -> 18 bundles, today's data flow (hand asm)", dict(t_ks=t_ks_for(p, CALL_18))),
        ("GEMM loop 2x2-blocked (hand asm), floor 0.753 us", dict(t_ks=max(t_ks_for(p, CALL_2X2), 0.753))),
        ("L_dep -> 5 us (P5 k-chunk A slot)", dict(L_dep=5.0)),
        ("t_tile halved", dict(t_tile=p["t_tile"] / 2)),
        ("attention -> 600 us", dict(attn=600.0)),
        ("SSM -> 500 us", dict(ssm=500.0)),
        ("norms folded (stats pass only, 40 us each)", dict(norm0=40.0, norm1=40.0, norm2=40.0, norm3=40.0)),
    ]
    for label, ov in cases:
        t = 12 * block_us(p, ov) + final
        print(f"  {t / 1e3:5.1f} ({(t - base) / 1e3:+5.1f})  {label}  [t_ks {dict(p, **ov)['t_ks']:.3f}]")
    for name, ov in scenarios(p).items():
        q = dict(p, **ov)
        g = sum(float(np.dot(op_row(o), [q[k] for k in PARAMS])) for o in BLOCK if o[1] == "gemm")
        e = block_us(q) - g - q["attn"]
        print(f"  {name}: serial {(12 * block_us(q) + final) / 1e3:.1f} ms; per block GEMM ops {g / 1e3:.2f}, "
              f"attention {q['attn'] / 1e3:.2f} (GEMM cores), EW ops {e / 1e3:.2f} ms")
    overlap(p)


# ---- overlap schedules: two resources (G = the 24 GEMM cores, which also run attention; E = the 8 EW cores) ----
def op_cost(q, o, f):
    """Cost (us) of op o on a row fraction f of the 1152 rows. GEMM: the k-step loop and GELU scale with rows, the
    per-tile and dependent-load costs do not (each chunk runs every N-tile); EW ops scale with rows + 10 us fixed."""
    if o[1] == "gemm":
        _, _, K, nt, g, pre = o
        return (0 if pre else q["L_dep"]) + nt * (K // 64 * q["t_ks"] * f + q["t_tile"]) + g * q["t_gelu"] * f
    c = q[o[0]] if o[1] == "norm" else q[o[1]]
    return c if o[1] == "attn" else 10.0 + (c - 10.0) * f


def schedule(q, chunks=(1.0,), frames=1, depth=12, inflight=2, jobs_out=None):
    """List-schedule `frames` frames of `depth` blocks, each frame split into row chunks (fractions). Attention is a
    join over the chunks of its frame. Frame i's SSM of block b waits for frame i-1's SSM of block b (the state);
    at most `inflight` frames are started at once. Returns (makespan, per-frame latency list) in us."""
    jobs, deps = [], []
    for fr in range(frames):
        for b in range(depth):
            for o in BLOCK:
                cs = [None] if o[1] == "attn" else range(len(chunks))
                for c in cs:
                    f = 1.0 if c is None else chunks[c]
                    jobs.append(dict(fr=fr, b=b, op=o[0], c=c, res="G" if o[1] in ("gemm", "attn") else "E",
                                     cost=op_cost(q, o, f), start=None, end=None))
    idx = {(j["fr"], j["b"], j["op"], j["c"]): i for i, j in enumerate(jobs)}
    names = [o[0] for o in BLOCK]
    for i, j in enumerate(jobs):
        d = []
        k = names.index(j["op"])
        prev_ops = [(j["b"], names[k - 1])] if k > 0 else ([(j["b"] - 1, names[-1])] if j["b"] > 0 else [])
        for (b, op) in prev_ops:
            if op == "attn":
                d.append(idx[(j["fr"], b, op, None)])
            elif j["c"] is None:
                d += [idx[(j["fr"], b, op, c)] for c in range(len(chunks))]
            else:
                d.append(idx[(j["fr"], b, op, j["c"])])
        if j["op"] == "ssm" and j["fr"] > 0:
            d.append(idx[(j["fr"] - 1, j["b"], "ssm", j["c"])])
        if j["fr"] >= inflight and j["b"] == 0 and j["op"] == names[0]:
            d.append(idx[(j["fr"] - inflight, depth - 1, names[-1], len(chunks) - 1)])
        deps.append(d)
    free = {"G": 0.0, "E": 0.0}
    done = 0
    while done < len(jobs):
        best = None
        for i, j in enumerate(jobs):
            if j["start"] is not None or any(jobs[d]["end"] is None for d in deps[i]):
                continue
            ready = max([jobs[d]["end"] for d in deps[i]] + [0.0])
            t = max(ready, free[j["res"]])
            key = (t, j["fr"], j["b"], j["c"] or 0)
            if best is None or key < best[0]:
                best = (key, i)
        (t, *_), i = best
        jobs[i]["start"], jobs[i]["end"] = t, t + jobs[i]["cost"]
        free[jobs[i]["res"]] = jobs[i]["end"]
        done += 1
    lat = []
    for fr in range(frames):
        js = [j for j in jobs if j["fr"] == fr]
        lat.append(max(j["end"] for j in js) - min(j["start"] for j in js))
    if jobs_out is not None:
        jobs_out.extend(jobs)
    return max(j["end"] for j in jobs), lat


def gantt(q, t0, t1, **kw):
    """Print the scheduled ops that overlap [t0, t1] us, per resource (python3 sim9.py gantt)."""
    jobs = []
    schedule(q, jobs_out=jobs, **kw)
    for res in ("G", "E"):
        row = [j for j in sorted(jobs, key=lambda j: j["start"]) if j["res"] == res and j["end"] > t0 and j["start"] < t1]
        print(f"  {res}: " + " | ".join(f"{j['start'] / 1e3:.2f}-{j['end'] / 1e3:.2f} f{j['fr']} b{j['b']} {j['op']}"
                                        + ("" if j["c"] is None else f" c{j['c']}") for j in row))


def overlap(p):
    print("\noverlap schedules (depth 12, list-scheduled on G = GEMM cores + attention, E = EW cores; +0.15 ms tail):")
    for name, ov in scenarios(p).items():
        q = dict(p, **ov)
        one, _ = schedule(q)
        two, _ = schedule(q, chunks=(80 / 144, 64 / 144))
        three, _ = schedule(q, chunks=(48 / 144,) * 3)
        n = 6
        span, lat = schedule(q, frames=n, inflight=2)
        per = (span - lat[0]) / (n - 1)
        span2, lat2 = schedule(q, chunks=(80 / 144, 64 / 144), frames=n, inflight=2)
        per2 = (span2 - lat2[0]) / (n - 1)
        print(f"  {name}:")
        print(f"    serial {one / 1e3 + 0.15:.1f} ms | 2 row chunks (80+64) {two / 1e3 + 0.15:.1f} ms | "
              f"3 chunks (48) {three / 1e3 + 0.15:.1f} ms")
        print(f"    2 frames in flight: {per / 1e3:.1f} ms/frame = {1e6 / per:.1f} fps, latency "
              f"{max(lat[1:]) / 1e3:.1f} ms | + 2 row chunks: {per2 / 1e3:.1f} ms/frame = {1e6 / per2:.1f} fps, "
              f"latency {max(lat2[1:]) / 1e3:.1f} ms")


if __name__ == "__main__":
    main()
