"""rev 10: main's round-15 note (global host row permutation at TILE granularity + the K / V fill in natural key order) priced
against P60g (half-tile gathers), and the two composed (tile permutation on the host, then the G6 gathers on the device).
Layout arrays: [dealt col, tile, rb] -> natural (col, tile, rb).
    python3 deal5.py"""
from dataclasses import replace

import numpy as np

import attsim17 as A
from deal4 import deal_g6, layout_of
from fit17 import kappa_att


def tile_deal(f):
    """f(n) -> (col, pos) for natural tile n = 3 c + t; returns the layout"""
    a = np.zeros((8, 3, 6, 3), int)
    for n in range(24):
        c, p = f(n)
        for r in range(6):
            a[c, p, r] = (n // 3, n % 3, r)
    return a


def compose(outer, inner):
    """inner deal applied on the device to rows that the host already permuted by `outer`"""
    a = np.zeros_like(inner)
    for c in range(8):
        for t in range(3):
            for r in range(6):
                dc, dt, dr = inner[c, t, r]
                a[c, t, r] = outer[dc, dt, dr]
    return a


rr = tile_deal(lambda n: (n % 8, n // 8))
rr3 = tile_deal(lambda n: ((3 * n) % 8, n // 8))
G = layout_of(deal_g6(0, 3, 1, (2, 5)))
C = {"tile rr (main's note)": rr, "tile rr, columns x3 (0,3,6,1,4,7,2,5)": rr3, "P60g (G6 gathers)": G,
     "tile rr + G6 gathers": compose(rr, G), "tile rr x3 + G6 gathers": compose(rr3, G)}
raw = A.load()
P = replace(A.P17(), VRB=136.0)
for kappa in (0.5, 0.6, 0.7):
    atts = [kappa_att(raw, kappa, s) for s in (0, 1, 2, 3)]
    base = np.mean([A.ms(a, P) for a in atts])
    line = []
    for name, L in C.items():
        v = np.mean([A.ms(A.remap(a, L), P) for a in atts]) - base
        line.append(f"{v:+.2f} {name}")
    print(f"kappa {kappa}: " + " | ".join(line))
