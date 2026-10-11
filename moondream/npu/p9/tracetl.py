# timeline check: when does DMA-in (PORT_RUNNING_0) run relative to the chunk windows?
import json, sys
import numpy as np
ev = json.load(open(sys.argv[1]))
names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
iv, op = {}, {}
for e in ev:
    if e.get("ph") not in ("B", "E"): continue
    k = (e["pid"], e["tid"]); n = names.get(k, str(k))
    if e["ph"] == "B": op[k] = e["ts"]
    elif k in op: iv.setdefault(n, []).append((op.pop(k), e["ts"]))
e0 = np.array(sorted(s for s, _ in iv["INSTR_EVENT_0"])); e1 = np.array(sorted(s for s, _ in iv["INSTR_EVENT_1"]))
dma = np.array(sorted(iv["PORT_RUNNING_0"]))
t0 = e0[40]
print("chunks 40..46 (event0, event1) relative to chunk 40's start:")
for i in range(40, 47): print(f"  chunk {i}: start {e0[i]-t0:7.0f} end {e1[np.searchsorted(e1, e0[i])]-t0:7.0f}")
sel = dma[(dma[:, 0] > t0 - 3000) & (dma[:, 0] < e0[46])]
print("DMA-in port running intervals in that span:")
for s, e in sel: print(f"  {s-t0:7.0f} .. {e-t0:7.0f}  ({e-s:.0f} cycles)")
