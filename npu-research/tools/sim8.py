"""Per-op cost model of design8 (round-5 builds), fitted to the host-polled timelines and used for what-if questions.

Pure python (no numpy, no device). Usage:
    python3 sim8.py [timeline.out ...]      # fit + validate on the given step8 timeline outputs (default: r5 k / kng)

Model (one block; the step is depth x block + the final norm):
  GEMM op  = L_dep (if its producer is the op right before it, or an EW op) + nt * (K/64 * t_ks + t_tile) + g * t_gelu
             nt = N-tiles per core (24 GEMM cores, 3 per column split N), g = GELU tiles per core
  norm op  = n_in * 221,184 B / port + c_norm      (input-port bound on the row-5 core)
  SSM, attention = measured constants (nothing inside them is modelled yet)
t_ks is the time of one k-step of one N-tile on one core: 9 mm_acc calls over 16-row stripes. Two hypotheses give
nearly the same number: A-port bound (10,368 B at 4 B/cycle = 1.813 us at 1.43 GHz) or compute + call overhead
(9 x (128 MACs x 1.59 bundles + ~115 cycles)). The free fit prefers 2.0 us. PROPOSALS P1 separates them.
"""

import re
import sys
from pathlib import Path

F = 1.43e9
A_KSTEP_US = 144 * 64 * 9 / 8 / 4 / F * 1e6  # 1.813

# name, kind, K, nt (tiles per core), gelu tiles per core, prefetchable (producer is not the op right before it)
BLOCK = [
    ("norm0", "norm", 3), ("yx", "gemm", 768, 8, 4, False), ("ssm", "ssm"), ("out", "gemm", 768, 4, 0, False),
    ("norm1", "norm", 2), ("up_a", "gemm", 768, 16, 8, False), ("up_b", "gemm", 768, 16, 8, True),
    ("down_a", "gemm", 1536, 4, 0, True), ("down_b", "gemm", 1536, 4, 0, True), ("norm2", "norm", 3),
    ("qkv", "gemm", 768, 12, 0, False), ("attn", "attn"), ("out_proj", "gemm", 768, 4, 0, False), ("norm3", "norm", 2),
    ("mlp0", "gemm", 768, 16, 16, False), ("mlp3_a", "gemm", 1536, 4, 0, False), ("mlp3_b", "gemm", 1536, 4, 0, True),
]


def parse_timeline(path):
    """step8 timeline output -> {op name: (first, last)} in ms."""
    ops = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^(b\d+_\w+|final)\s+(\w+)\s+([\d.]+|-)\s+([\d.]+|-)", line)
        if m:
            f = None if m.group(3) == "-" else float(m.group(3))
            last = None if m.group(4) == "-" else float(m.group(4))
            ops[m.group(1)] = (f, last)
    return ops


def block_costs(tl, b):
    """Measured cost of every op of block b = its last write - the previous op's last write (the SSM has no last
    write in the timeline: its cost is folded into `out`, so out's own cost is taken from the model below)."""
    names = [f"b{b}_{o[0]}" for o in BLOCK]
    prev = tl.get(f"b{b - 1}_mlp3_b", (None, None))[1]
    out = {}
    for n in names:
        last = tl.get(n, (None, None))[1]
        if last is None or prev is None:
            out[n.split("_", 1)[1]] = None
            continue
        out[n.split("_", 1)[1]] = last - prev
        prev = last
    return out


def solve3(rows, ys):
    """Least squares for 3 unknowns via the normal equations (no numpy on the host)."""
    n = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    m = [ata[i] + [aty[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [a - f * c for a, c in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def gemm_us(o, p, gelu=True):
    _, _, K, nt, g, pre = o
    return (0 if pre else p["L_dep"]) + nt * (K // 64 * p["t_ks"] + p["t_tile"]) + (g * p["t_gelu"] if gelu else 0)


def fit(tl_gelu, tl_nogelu, blocks=range(1, 11)):
    """Fit L_dep, t_tile, t_ks on the no-GELU build's GEMMs (out excluded: it carries the SSM), then t_gelu from the
    GELU build's extra time on the GEMMs with GELU tiles; EW / attention constants as medians."""
    rows, ys, gd, gn = [], [], [], []
    for b in blocks:
        c0, c1 = block_costs(tl_nogelu, b), block_costs(tl_gelu, b)
        for o in BLOCK:
            if o[1] != "gemm" or o[0] in ("out", "yx"):
                continue  # yx: its first write is missing in some builds; out: carries the SSM
            if c0.get(o[0]) is None:
                continue
            rows.append([0 if o[5] else 1, o[3], o[3] * (o[2] // 64)])
            ys.append(c0[o[0]] * 1e3)
            if o[4] and c1.get(o[0]) is not None:
                gd.append((c1[o[0]] - c0[o[0]]) * 1e3)
                gn.append(o[4])
    L, tt, tks = solve3(rows, ys)
    p = dict(L_dep=L, t_tile=tt, t_ks=tks, t_gelu=sum(gd) / sum(gn))
    pred = [r[0] * L + r[1] * tt + r[2] * tks for r in rows]
    p["fit_rms_us"] = (sum((a - y) ** 2 for a, y in zip(pred, ys)) / len(ys)) ** 0.5
    # EW / attention constants (medians over blocks of the GELU build)
    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2]
    cs = [block_costs(tl_gelu, b) for b in blocks]
    p["norm"] = {o[0]: med([c[o[0]] for c in cs]) * 1e3 for o in BLOCK if o[1] == "norm"}
    p["attn"] = med([c["attn"] for c in cs]) * 1e3
    p["ssm"] = med([c["out"] for c in cs]) * 1e3 - gemm_us(BLOCK[3], p)  # out's cost includes the SSM
    return p


def block_us(p, gelu=True, over=None):
    """Predicted block time and its parts (us). `over`: dict of overrides for what-if questions."""
    q = dict(p, **(over or {}))
    parts = {}
    for o in BLOCK:
        if o[1] == "gemm":
            parts[o[0]] = gemm_us(o, q, gelu)
        elif o[1] == "norm":
            parts[o[0]] = q["norm"][o[0]] if "norm_scale" not in q else q["norm"][o[0]] * q["norm_scale"]
        else:
            parts[o[0]] = q[o[1]]
    return sum(parts.values()), parts


def main():
    here = Path(__file__).resolve().parent
    files = sys.argv[1:] or [str(here / "r5_tl12k.out"), str(here / "r5_tl12kng.out")]
    tl, tln = parse_timeline(files[0]), parse_timeline(files[1])
    p = fit(tl, tln)
    print(f"fit: t_ks {p['t_ks']:.3f} us (A-port bound would be {A_KSTEP_US:.3f}), L_dep {p['L_dep']:.1f} us, "
          f"t_tile {p['t_tile']:.2f} us, t_gelu {p['t_gelu']:.2f} us/tile, GEMM fit rms {p['fit_rms_us']:.1f} us")
    print(f"     SSM {p['ssm']:.0f} us, attention {p['attn']:.0f} us, norms " + ", ".join(f"{k} {v:.0f}" for k, v in p["norm"].items()))
    final = 150.0
    for label, f, gelu in (("GELU build", files[0], True), ("no-GELU build", files[1], False)):
        meas = re.search(r"dispatch ([\d.]+) ms", Path(f).read_text()).group(1)
        tot = 12 * block_us(p, gelu)[0] + final
        print(f"{label}: predicted step {tot / 1e3:.1f} ms, measured dispatch {meas} ms")
    # validation on other depths (r5_sweep_m: depth 10 / 8 / 6 = 74.64 / 59.73 / 44.87 ms)
    for d, m in ((10, 74.64), (8, 59.73), (6, 44.87)):
        print(f"depth {d}: predicted {(d * block_us(p)[0] + final) / 1e3:.1f} ms, measured {m}")
    tot, parts = block_us(p)
    print(f"\nblock {tot / 1e3:.2f} ms; per-op (us): " + ", ".join(f"{k} {v:.0f}" for k, v in parts.items()))
    g = sum(v for k, v in parts.items() if dict((o[0], o[1]) for o in BLOCK)[k] == "gemm")
    print(f"  GEMM ops {g / 1e3:.2f} ms: k-step loop {1296 * p['t_ks'] / 1e3:.2f}, per-tile {92 * p['t_tile'] / 1e3:.2f}, "
          f"GELU {36 * p['t_gelu'] / 1e3:.2f}, dependent-load latency {7 * p['L_dep'] / 1e3:.2f}")
    print("\nwhat-if (depth-12 step, ms; each alone):")
    base = 12 * tot + final
    whatifs = [
        ("t_ks -> 1.813 (A-port floor; e.g. one call per k-step if call-overhead bound)", dict(t_ks=A_KSTEP_US)),
        ("t_ks -> 1.28 (compute bound at 1.59 bundles/MAC, port-balanced tiles)", dict(t_ks=1832 / F * 1e6)),
        ("L_dep -> 5 us (k-chunk streamed A slot)", dict(L_dep=5.0)),
        ("t_tile -> 3 us (C drain overlapped)", dict(t_tile=3.0)),
        ("t_gelu halved", dict(t_gelu=p["t_gelu"] / 2)),
        ("norms 4x faster (4 cores per column, or folded)", dict(norm_scale=0.25)),
        ("SSM -> 500 us", dict(ssm=500.0)),
        ("attention -> 700 us", dict(attn=700.0)),
    ]
    for label, ov in whatifs:
        t = 12 * block_us(p, over=ov)[0] + final
        print(f"  {base / 1e3:5.1f} -> {t / 1e3:5.1f}  ({(t - base) / 1e3:+.1f})  {label}")
    allov = {k: v for _, ov in whatifs[1:] for k, v in ov.items()}
    print(f"  all of the above together: {(12 * block_us(p, over=allov)[0] + final) / 1e3:.1f} ms (serial schedule, no overlap)")


if __name__ == "__main__":
    main()
