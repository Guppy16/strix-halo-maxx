"""Rev 5: median measured span per span kind in the twin and base timelines (us). No device."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim9  # noqa: E402

B = Path(__file__).resolve().parents[2] / "npu-tapnext" / "build"
for build in ("r6a", "p1n", "p1t"):
    _, ops = sim9.load(B / f"tl_full12{build}_d12.json")
    d = defaultdict(list)
    for row, us, lab in sim9.segments(ops):
        d[lab].append((us, row))
    print(f"\nfull12{build}")
    for lab, v in sorted(d.items()):
        row = v[0][1]
        coef = dict(zip(sim9.PARAMS, row))
        print(f"  {lab:22s} n={len(v):2d} median {np.median([x for x, _ in v]):7.1f} us  "
              f"[ks {coef['t_ks']:.0f} tiles {coef['t_tile']:.0f} gelu {coef['t_gelu']:.0f} dep {coef['L_dep']:.0f}]")
