"""Round 11: decode hazard-probe dumps (r11/mk_haz.py): per writer and distance d, did the store see OLD (1.0) or NEW?"""
import sys

import numpy as np

base = sys.argv[1]
names = {1: "vlda", 2: "vldb.4x64", 3: "vadd.32", 4: "vconv", 5: "vfloor", 6: "vshuffle", 7: "vmin_ge"}
if len(sys.argv) > 2:  # WAW probes: old = 1.0 (the younger vmov's value survived: the older load's write was lost)
    names = {1: "vlda then vmov", 2: "vldb.4x64 then vmov", 3: "vldb then vmov"}
for k, nm in names.items():
    v = np.load(f"{base}_p{k}_l0_n144_i0.npy").reshape(-1, 32)
    row = []
    for d in range(9):
        u = v[d]
        row.append(("vmov" if len(sys.argv) > 2 else "old") if np.all(u == 0x3F80) else (("load" if len(sys.argv) > 2 else "NEW") if not np.any(u == 0x3F80) else "mix"))
    print(f"{nm:10s} " + " ".join(f"d{d}:{r}" for d, r in enumerate(row)))
