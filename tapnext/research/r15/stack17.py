"""rev 10: the attention stack on r14f with sim17 (kappa 0.5 / 0.6 / 0.7, 4 seeds each): each row adds to the previous.
    python3 stack17.py"""
from dataclasses import replace

import numpy as np

import attsim17 as A
from deal4 import deal_g6, layout_of
from fit17 import kappa_att

raw = A.load()
P = replace(A.P17(), VRB=136.0)
G = layout_of(deal_g6(0, 3, 1, (2, 5)))
ROWS = [
    ("P60g single-TAP deal G{0,3,6}+{1,4,7} / pair {2,5}", True, {}),
    ("+ S1 rescale_rb asm (R 578 -> 200 +- 50)", True, dict(R=200.0)),
    ("+ S2 running row max in attkb (FP 393 -> 0, T 164 -> 70; B +7 per rb for II 34 -> 35)", True,
     dict(R=200.0, FP=0.0, T=70.0, B=268.0)),
    ("+ S3 exact test + rescale in asm in one call (T + R call overhead -60)", True,
     dict(R=170.0, FP=0.0, T=40.0, B=268.0)),
    ("+ V sums folded into attv, glue (VRB 136 -> 110)", True, dict(R=170.0, FP=0.0, T=40.0, B=268.0, VRB=110.0)),
]
ALONE = [
    ("S1 alone (no deal)", False, dict(R=200.0)),
    ("S1 + S2 alone (no deal)", False, dict(R=200.0, FP=0.0, T=70.0, B=268.0)),
    ("S2 alone (no deal)", False, dict(FP=0.0, T=70.0, B=268.0)),
    ("P60g + no slow path at all (floor)", True, dict(ignore_flags=True)),
]
res = {}
for kappa in (0.5, 0.6, 0.7):
    atts = [kappa_att(raw, kappa, s) for s in (0, 1, 2, 3)]
    base = np.mean([A.ms(a, P) for a in atts])
    out = []
    for name, deal, o in ROWS + ALONE:
        v = np.mean([A.ms(A.remap(a, G) if deal else a, replace(P, **o)) for a in atts]) - base
        out.append(v)
    res[kappa] = out
print("delta vs r14f, ms per step (kappa 0.5 / 0.6 / 0.7); rows 1-5 cumulative")
for i, (name, _, _) in enumerate(ROWS + ALONE):
    print(f"  {res[0.5][i]:+6.2f} / {res[0.6][i]:+6.2f} / {res[0.7][i]:+6.2f}   {name}")
