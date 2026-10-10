"""decode r11/mk_haz.py mid probes: per probe, per d: the stored x1 lanes 0-15 / 16-31 as values"""
import sys

import numpy as np

for k in (1, 2, 3):
    v = np.load(f"{sys.argv[1]}_p{k}_l0_n144_i0.npy").reshape(-1, 32)
    f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)  # noqa: E731
    print(k, [(float(f(v[d, :1])[0]), float(f(v[d, 16:17])[0])) for d in range(6)])
