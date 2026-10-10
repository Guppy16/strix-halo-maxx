# summarise an IRON chrome-format trace: chunk windows (INSTR_EVENT_0 -> next INSTR_EVENT_1) and the cycles each
# traced event is active inside them
import json, sys
import numpy as np
ev = json.load(open(sys.argv[1]))
names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
procs = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
print("processes:", procs)
print("threads:", sorted(set(names.values())))
# intervals per thread name (within the core tile process)
iv = {}
open_ts = {}
for e in ev:
    if "ph" not in e or e["ph"] not in ("B", "E"): continue
    k = (e["pid"], e["tid"]); n = f'{procs.get(e["pid"], e["pid"])}:{names.get(k, k)}'
    if e["ph"] == "B": open_ts[k] = e["ts"]
    elif k in open_ts: iv.setdefault(n, []).append((open_ts.pop(k), e["ts"]))
for n in sorted(iv):
    a = np.array(iv[n]); d = a[:, 1] - a[:, 0]
    print(f"  {n:50s} n={len(d):6d} total={d.sum():10.0f} median={np.median(d):7.1f} max={d.max():8.0f}")
core = [n for n in iv if "core" in n.lower() or "Core" in n]
e0 = sorted(s for s, _ in iv.get(next((n for n in iv if n.endswith("INSTR_EVENT_0")), ""), []))
e1 = sorted(s for s, _ in iv.get(next((n for n in iv if n.endswith("INSTR_EVENT_1")), ""), []))
if e0 and e1:
    e1a = np.array(e1); wins = []
    for s in e0:
        j = np.searchsorted(e1a, s)
        if j < len(e1a): wins.append((s, e1a[j]))
    w = np.array(wins); d = w[:, 1] - w[:, 0]
    print(f"\nchunk windows (event0 -> event1): n={len(d)} median {np.median(d):.0f} cycles, min {d.min():.0f}, max {d.max():.0f}, mean {d.mean():.0f}")
    gaps = w[1:, 0] - w[:-1, 1]
    print(f"gaps between chunks (event1 -> next event0): median {np.median(gaps):.0f} max {gaps.max():.0f} (includes item begin/end every {len(d)//8} chunks)")
    # stall / vector cycles inside the windows
    ws, we = w[:, 0], w[:, 1]
    for n in sorted(iv):
        if "INSTR_EVENT" in n: continue
        a = np.array(iv[n]); tot = 0.0
        for s, e in a:  # clip to windows
            j = np.searchsorted(ws, s, side="right") - 1
            if j >= 0 and s < we[j]: tot += min(e, we[j]) - s
        print(f"  inside chunks: {n:48s} {tot:10.0f} cycles = {100 * tot / d.sum():5.1f}% of chunk time")
