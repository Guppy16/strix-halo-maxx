"""Timeline viewer + per-op roofline for design8 steps (round 6).

usage: python3 r6/tlview.py build/tl_A.json [build/tl_B.json] [--block 5] [--html out.html]

Input: the JSON that `step8.py timeline` writes (per op: first / last change of its column-0 output, ms).
Per op of the chosen block it prints
  share   wall time owned by the op: its last write minus the previous op's last write (ms, MEASURED)
  span    last - first write of the op's output (ms, MEASURED)
  work    MACs (GEMM / attention mmul work) or elements (EW ops)
  bytes   DDR bytes the op's runtime transfers move (A / B / weight / C taps of plan8, all columns)
  t_mac   time at 100% of the MAC slot of the cores that run it (24 GEMM cores x 512 MAC/cycle)
  t_loop  time at the deployed GEMM loop's rate (1.59 bundles per mmul, i.e. 63% of the slot)
  t_ddr   time at DDR_GBS (the NPU's measured read peak under load)
  t_port  GEMM: time to stream A into one core through its input port at PORT_B_PER_CYCLE (nt tiles x K/64 k-steps x
          10,368 B; npu-research ANALYSIS section 2: the fitted k-step is 2.4% above this floor)
  bound   the largest of t_loop / t_ddr / t_port, and x = share / that = how far the op is from its own limit
With two timelines it also prints the per-op difference (B - A). --html writes a self-contained page: a Gantt of
the block (ops on a time axis, coloured by kind), the roofline table and, with two inputs, the diff.
All bounds are MODELS (clock, rates below); only share / span are measurements.
"""

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan8 import COLS, M, full_step  # noqa: E402

F_HZ = 1.43e9  # AIE clock, Turbo (skill section 1)
GEMM_CORES = 24
MAC_PER_CYCLE = 512  # bfp16 8x8x8 mmul per core
LOOP_BUNDLES_PER_MMUL = 1.59  # deployed GEMM loop (skill section 7)
DDR_GBS = 57.0  # NPU DDR rate the norms reach (RESULTS 14), ~55-60 peak
PORT_B_PER_CYCLE = 4.0  # MACHINE.md section 3 model; P1 (round 6) measured A arriving faster (~1.0 us per k-step): t_port is a model, not a measured limit
KSTEP_A_BYTES = 10_368  # 144 rows x 64 k of bfp16
ROWS = COLS * M
KEYS, HEADS, HD = 1072, 12, 64


def tap_bytes(t):
    n = 1
    for s in t["sizes"]:
        n *= s
    return n


def flat_taps(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, (list, tuple)):
        return [t for e in x for t in flat_taps(e)]
    return []


def op_model(op):
    """(work, unit, bytes, t_mac_ms, t_loop_ms, t_ddr_ms, t_port_ms) for one plan8 op."""
    taps = flat_taps(op.get("a", [])) + flat_taps(op.get("b", [])) + flat_taps(op.get("c", [])) + flat_taps(op.get("fills", []))
    nbytes = sum(tap_bytes(t) for t in taps)
    if op["kind"] == "gemm":
        nbytes += 3 * op["nt"] * op["lw"]  # weights + bias objects, read once (broadcast through memtiles 0-2)
        work = ROWS * op["K"] * op["N"]
        unit = "MAC"
    elif op["kind"] == "attn":
        # executed mmul work: every query row against the valid keys, QK^T (64) + PV (64) + the ones-panel sums (8)
        work = HEADS * ROWS * KEYS * (HD + HD + 8)
        unit = "MAC"
    else:
        work = ROWS * 768 * (op.get("n_in", 1) if op.get("fn") == "norm" else 1)
        unit = "elem"
    t_ddr = nbytes / (DDR_GBS * 1e9) * 1e3
    if unit == "MAC":
        cyc = work / GEMM_CORES / MAC_PER_CYCLE
        t_mac = cyc / F_HZ * 1e3
        t_loop = t_mac * LOOP_BUNDLES_PER_MMUL
    else:
        t_mac = t_loop = None
    t_port = op["nt"] * op["K"] // 64 * KSTEP_A_BYTES / PORT_B_PER_CYCLE / F_HZ * 1e3 if op["kind"] == "gemm" else None
    return work, unit, nbytes, t_mac, t_loop, t_ddr, t_port


def load(path):
    d = json.loads(Path(path).read_text())
    plan, _ = full_step(d["depth"])
    byname = {op["name"]: op for op in plan.ops}
    prev = 0.0
    for r in d["ops"]:
        r["share"] = None if r["last"] is None else r["last"] - prev
        r["span"] = None if (r["first"] is None or r["last"] is None) else r["last"] - r["first"]
        if r["last"] is not None:
            prev = r["last"]
        r["model"] = op_model(byname[r["name"]])
    return d


def fmt(v, w=7, p=2):
    return f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"


def block_rows(d, b):
    pre = f"b{b}_"
    return [r for r in d["ops"] if r["name"].startswith(pre)]


def limit(m):
    """(time, name) of the binding model bound of an op_model tuple"""
    names = ("loop", "ddr", "port")
    vals = (m[4], m[5], m[6])
    t, nm = max((v, n) for v, n in zip(vals, names) if v is not None)
    return t, nm


def text_report(ds, b):
    a = ds[0]
    print(f"{a['build']} depth {a['depth']}: dispatch {a['dispatch_ms']:.1f} ms (host-polled)"
          + "".join(f"; {x['build']}: {x['dispatch_ms']:.1f} ms" for x in ds[1:]))
    hdr = f"{'op':14s} {'kind':5s} {'share':>7s} {'span':>7s} {'work':>9s} {'MB':>6s} {'t_mac':>7s} {'t_loop':>7s} {'t_ddr':>7s} {'t_port':>7s}  bound  x"
    if len(ds) > 1:
        hdr += f"   {'B share':>7s} {'B-A':>7s}"
    print(hdr)
    rows_a = block_rows(a, b)
    rows_b = {r["name"].split("_", 1)[1]: r for r in block_rows(ds[1], b)} if len(ds) > 1 else {}
    tot = [0.0, 0.0]
    for r in rows_a:
        work, unit, nb, tm, tl, td, tp = r["model"]
        lim, which = limit(r["model"])
        x = r["share"] / lim if r["share"] else None
        ws = f"{work / 1e9:7.2f}G" if unit == "MAC" else f"{work / 1e6:7.2f}M"
        line = (f"{r['name'].split('_', 1)[1]:14s} {r['kind']:5s} {fmt(r['share'])} {fmt(r['span'])} {ws:>9s} {nb / 1e6:6.1f} "
                f"{fmt(tm, 7, 3)} {fmt(tl, 7, 3)} {fmt(td, 7, 3)} {fmt(tp, 7, 3)}  {which:5s} {fmt(x, 4, 1)}")
        if r["share"]:
            tot[0] += r["share"]
        rb = rows_b.get(r["name"].split("_", 1)[1])
        if rb is not None:
            line += f"   {fmt(rb['share'])} {fmt(rb['share'] - r['share'] if rb['share'] and r['share'] else None)}"
            if rb["share"]:
                tot[1] += rb["share"]
        print(line)
    print(f"{'block total':20s} {tot[0]:7.2f}" + (f"   B {tot[1]:.2f}  B-A {tot[1] - tot[0]:+.2f}" if len(ds) > 1 else ""))
    print("share = MEASURED wall time owned by the op (the SSM's last write is invisible: its time lands on `out`); "
          "t_* = MODEL bounds; x = share / max(t_loop, t_ddr, t_port)")


KIND_SLOT = dict(gemm=1, attn=2, norm=3, ssm=4)


def gantt_svg(d, b, width=900):
    rows = [r for r in block_rows(d, b) if r["first"] is not None or r["last"] is not None]
    t0 = min(r["first"] if r["first"] is not None else r["last"] for r in rows)
    t1 = max(r["last"] for r in rows if r["last"] is not None)
    # an op's bar: from the previous op's last write to its own last write (its share), the span drawn darker
    lane, left, top = 18, 120, 24
    h = top + lane * len(rows) + 30
    sc = (width - left - 20) / (t1 - t0)
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" aria-label="block {b} timeline">']
    # recessive grid every 0.5 ms
    t = 0.0
    while t0 + t <= t1 + 1e-9:
        x = left + t * sc
        out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{h - 26}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 10}" class="tick" text-anchor="middle">{t:.1f}</text>')
        t += 0.5
    prev = t0
    for i, r in enumerate(rows):
        y = top + i * lane
        name = r["name"].split("_", 1)[1]
        out.append(f'<text x="{left - 8}" y="{y + 12}" class="lab" text-anchor="end">{html.escape(name)}</text>')
        if r["last"] is None:
            continue
        s = KIND_SLOT.get(r["kind"], 1)
        xs, xe = left + (prev - t0) * sc, left + (r["last"] - t0) * sc
        tip = f"{name} ({r['kind']}): share {r['share']:.3f} ms" + (f", span {r['span']:.3f} ms" if r["span"] is not None else "")
        out.append(f'<rect x="{xs:.1f}" y="{y + 2}" width="{max(xe - xs, 1.5):.1f}" height="{lane - 6}" rx="3" class="s{s} share"><title>{html.escape(tip)}</title></rect>')
        if r["first"] is not None:
            xf = left + (r["first"] - t0) * sc
            out.append(f'<rect x="{xf:.1f}" y="{y + 2}" width="{max(xe - xf, 1.5):.1f}" height="{lane - 6}" rx="3" class="s{s}"><title>{html.escape(tip)}</title></rect>')
        prev = r["last"]
    out.append(f'<text x="{width - 20}" y="{h - 10}" class="tick" text-anchor="end">ms from b{b} start</text>')
    out.append("</svg>")
    return "\n".join(out)


def html_report(ds, b, path):
    a = ds[0]
    rows_b = {r["name"].split("_", 1)[1]: r for r in block_rows(ds[1], b)} if len(ds) > 1 else {}
    trs = []
    for r in block_rows(a, b):
        work, unit, nb, tm, tl, td, tp = r["model"]
        lim, _ = limit(r["model"])
        x = r["share"] / lim if r["share"] else None
        name = r["name"].split("_", 1)[1]
        rb = rows_b.get(name)
        bar = f'<span class="bar" style="width:{min(x or 0, 12) / 12 * 100:.0f}%"></span>' if x else ""
        diff = ""
        if len(ds) > 1:
            dv = rb["share"] - r["share"] if rb and rb["share"] and r["share"] else None
            diff = f"<td>{fmt(rb['share'] if rb else None)}</td><td>{fmt(dv)}</td>"
        trs.append(f"<tr><td>{html.escape(name)}</td><td>{r['kind']}</td><td>{fmt(r['share'], 1, 3)}</td><td>{fmt(r['span'], 1, 3)}</td>"
                   f"<td>{work / (1e9 if unit == 'MAC' else 1e6):.2f} {'G' if unit == 'MAC' else 'M'}{unit}</td><td>{nb / 1e6:.1f}</td>"
                   f"<td>{fmt(tm, 1, 3)}</td><td>{fmt(tl, 1, 3)}</td><td>{fmt(td, 1, 3)}</td><td>{fmt(tp, 1, 3)}</td>"
                   f"<td class='x'>{fmt(x, 1, 1)}{bar}</td>{diff}</tr>")
    dh = "<th>B share</th><th>B&minus;A</th>" if len(ds) > 1 else ""
    gantts = "".join(f"<h2>{html.escape(d['build'])}: block {b}</h2>{gantt_svg(d, b)}" for d in ds)
    legend = "".join(f'<span class="key"><span class="sw s{s}"></span>{k}</span>' for k, s in KIND_SLOT.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step timeline</title><style>
:root{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
body{{background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:28px}} .note{{color:var(--ink2)}}
svg text{{fill:var(--ink2);font-size:11px}} .grid{{stroke:var(--grid)}} .s1{{fill:var(--s1)}} .s2{{fill:var(--s2)}} .s3{{fill:var(--s3)}} .s4{{fill:var(--s4)}}
.share{{opacity:.35}} .key{{margin-right:14px;color:var(--ink2)}} .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.sw.s1{{background:var(--s1)}} .sw.s2{{background:var(--s2)}} .sw.s3{{background:var(--s3)}} .sw.s4{{background:var(--s4)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-variant-numeric:tabular-nums}} td,th{{padding:3px 8px;text-align:right;border-bottom:1px solid var(--grid)}}
td:first-child,th:first-child,td:nth-child(2){{text-align:left}} .x{{position:relative;min-width:90px}} .bar{{position:absolute;left:0;bottom:2px;height:3px;background:var(--s1);border-radius:2px}}
</style></head><body>
<h1>Step timeline and roofline</h1>
<p class="note">{html.escape(', '.join(f"{d['build']} (depth {d['depth']}, dispatch {d['dispatch_ms']:.1f} ms)" for d in ds))}. Host-polled timeline: an op's
pale bar is its share (previous op's last write to its own last write), the solid part its span (first to last write of
its column-0 output). Measured; ~17 us resolution.</p>
<p>{legend}</p>{gantts}
<h2>Roofline, block {b}</h2>
<p class="note">share/span MEASURED (ms). t_mac = 100% of the MAC slot on 24 cores at {F_HZ / 1e9:.2f} GHz; t_loop = the deployed
loop rate ({LOOP_BUNDLES_PER_MMUL} bundles per mmul); t_ddr = bytes at {DDR_GBS:.0f} GB/s; t_port = A into one GEMM core at {PORT_B_PER_CYCLE:.0f} B/cycle. x = share / max(t_loop, t_ddr, t_port): how far the op is from its own limit (models).</p>
<div class="wrap"><table><tr><th>op</th><th>kind</th><th>share</th><th>span</th><th>work</th><th>MB</th><th>t_mac</th><th>t_loop</th><th>t_ddr</th><th>t_port</th><th>x</th>{dh}</tr>
{''.join(trs)}</table></div></body></html>"""
    Path(path).write_text(page)
    print(f"html -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tl", nargs="+")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--html", default=None)
    o = ap.parse_args()
    ds = [load(p) for p in o.tl]
    text_report(ds, o.block)
    if o.html:
        html_report(ds, o.block, o.html)


if __name__ == "__main__":
    main()
