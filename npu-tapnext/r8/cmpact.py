"""Bit-level comparison of two ACT images saved by run8.py (PROF_OUT=...): python3 r8/cmpact.py A.npy B.npy"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.nonzero(a != b)[0]
print(f"{len(a)} words, {len(d)} differ" + (f" (first at word {d[0]}, last {d[-1]})" if len(d) else ": BIT-IDENTICAL"))
