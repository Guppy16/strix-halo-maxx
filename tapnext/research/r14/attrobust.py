"""rev 9: robustness of the lockstep / layout prices to the map's column imbalance. The CPU map's per-core slow rates span
8.8-13.6 %, the in-situ count 8.6-10.9 %: thin the heavy cores to 10.9 %, recalibrate on the twins, re-price.
    python3 attrobust.py"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "tools"))
import attsim as A  # noqa: E402
import attlev as L  # noqa: E402  (prints the baseline table on import)

FIT = ["F3", "F2", "F16", "r13c", "F8"]
meas = {k: m for k, (_, _, m) in A.V_TWINS.items()}
for cap in (0.109, 0.10):
    at = A.default_att().thinned(cap)
    cnts = {k: at.counts(v) for k, (v, _, _) in A.V_TWINS.items()}

    def ev(p):
        raw = {k: 12 * at.block_cycles(cnts[k], replace(p, D=A.V_TWINS[k][1])) / A.F / 1e3 for k in A.V_TWINS}
        base = np.mean([meas[k] - raw[k] for k in FIT])
        pred = {k: base + raw[k] for k in raw}
        return np.sqrt(np.mean([(pred[k] - meas[k]) ** 2 for k in FIT])), pred

    best = min((ev(replace(L.pb, T=T, R=R, L=Lc))[0], T, R, Lc) for T in (66, 100, 130) for R in (0, 150, 300, 450)
               for Lc in (390, 400, 410))
    _, T, R, Lc = best
    pb = replace(L.pb, T=T, R=R, L=Lc)
    rms, pred = ev(pb)
    print(f"\ncap {cap}: refit T {T} R {R} L {Lc}: rms {rms:.3f}; r13f model {pred['r13f']:.2f} (49.21), r13e {pred['r13e']:.2f}")
    b = L.ms(at, prm=pb)
    print(f"  lockstep cost {b - L.ms(at, prm=replace(pb, LOCK=False)):.2f} ms")
    print(f"  rb interleave {L.ms(at.remap_rows(L.assign_rb_interleave()), prm=pb) - b:+.2f} ms")
    print(f"  chunk 22 fast {L.ms(at, L.c22, prm=pb) - b:+.2f} ms")
    print(f"  asm aexp + cheaper rescale {L.ms(at, dict(L.R13F, e_scale=250 / 400, r_scale=1 / 3), prm=pb) - b:+.2f} ms")
    print(f"  36-iteration {L.ms(at, dict(L.R13F, l_scale=285 / 400), prm=pb) - b:+.2f} ms")
