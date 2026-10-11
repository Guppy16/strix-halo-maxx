"""rev 10: pre-registered per-block attention walls (blocks 1-11, us) for P60g and P60g + S1 + S2 vs r14f: the model's
per-block deltas added to the measured r14f walls (kappa 0.6; range kappa 0.5-0.7).
    python3 prereg17.py"""
from dataclasses import replace

import numpy as np

import attsim17 as A
from blk17 import walls
from deal4 import deal_g6, layout_of
from fit17 import kappa_att

raw = A.load()
P = replace(A.P17(), VRB=136.0)
G = layout_of(deal_g6(0, 3, 1, (2, 5)))
meas = walls("r14f")
for name, deal, o in (("P60g", True, {}), ("P60g + S1 + S2", True, dict(R=200.0, FP=0.0, T=70.0, B=268.0)),
                      ("S1 alone", False, dict(R=200.0))):
    rows = {}
    for kappa in (0.5, 0.6, 0.7):
        d = []
        for s in (0, 1, 2, 3):
            a = kappa_att(raw, kappa, s)
            t0, _ = a.sim(P, detail=True)
            t1, _ = (A.remap(a, G) if deal else a).sim(replace(P, **o), detail=True)
            d.append((t1 - t0).reshape(-1, 12).mean(0)[1:12] / A.F)
        rows[kappa] = np.mean(d, 0)
    pred = meas + rows[0.6]
    print(f"{name}: mean {pred.mean():.0f} us/block (r14f {meas.mean():.0f}; delta {rows[0.5].mean():+.0f} / {rows[0.6].mean():+.0f} / "
          f"{rows[0.7].mean():+.0f}); spread {pred.max() - pred.min():.0f} (r14f {meas.max() - meas.min():.0f})")
    print("   r14f  " + " ".join(f"{x:4.0f}" for x in meas))
    print("   pred  " + " ".join(f"{x:4.0f}" for x in pred))
