"""decode r11/mk_haz.py rd probes: per reader, per d, the first stored 32-bit word (old vs new outcomes differ)"""
import sys

import numpy as np

names = {1: "vshuffle", 2: "vadd.32", 3: "vmin_ge", 4: "vmul.f", 5: "vfloor", 6: "vmov w", 7: "vst"}
for k, nm in names.items():
    v = np.load(f"{sys.argv[1]}_p{k}_l0_n144_i0.npy").reshape(-1, 32)
    d0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"{nm:9s}", " ".join(f"d{d + d0}:{int(v[d, 1]):04x}{int(v[d, 0]):04x}" for d in range(7)))
