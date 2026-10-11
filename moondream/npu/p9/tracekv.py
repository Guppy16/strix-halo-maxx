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
w = []
for s in e0:
    j = np.searchsorted(e1, s)
    if j < len(e1): w.append((s, e1[j]))
w = np.array(w); d = w[:, 1] - w[:, 0]
# windows alternate K-phase, V-phase; find the phase by the first window after a long gap (item start = K)
gaps = np.concatenate([[1e9], w[1:, 0] - w[:-1, 1]])
phase = np.zeros(len(d), int); p = 0
for i in range(len(d)):
    if gaps[i] > 5000: p = 0
    phase[i] = p; p ^= 1
print(f"K-phase windows: n={int((phase==0).sum())} median {np.median(d[phase==0]):.0f} mean {d[phase==0].mean():.0f}")
print(f"V-phase windows: n={int((phase==1).sum())} median {np.median(d[phase==1]):.0f} mean {d[phase==1].mean():.0f}")
big = gaps[gaps > 5000]; print(f"item boundaries: n={len(big)} median {np.median(big):.0f} min {big.min():.0f} max {big.max():.0f} cycles")
chunks = 16; per_item = d.reshape(-1)[:].sum() / max(1, len(big))
print(f"per item: compute {d.sum()/len(big):.0f} cycles + boundary {np.median(big):.0f} -> boundary share {100*np.median(big)/(d.sum()/len(big)+np.median(big)):.1f}%")
