"""rev 9: query-row layouts that a few DMA pieces can express (the attention A fill / O drain TAPs), priced on attsim.
Source = today's layout (column c' = rows 144c' .., tile i' = 48 rows, rb m' = 8 rows); a layout deals pieces of g row blocks
round robin to the columns: piece n (natural order) -> column n mod 8. g = 6 (tile), 3 (half tile), 2, 1 (row block).
Fills per column per head = 18 / g pieces (today 1 TAP covers all heads and tiles of a column).
    python3 attlayout.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "tools"))
import attlev as L  # noqa: E402

def assign(g):
    a = np.zeros((8, 3, 6, 3), int)
    per = 18 // g  # pieces per column
    for c in range(8):
        for t in range(3):
            for m in range(6):
                pos = 6 * t + m           # position in the destination column
                n = c + 8 * (pos // g)    # piece index (natural), dealt round robin
                k = n * g + pos % g       # natural global row block
                a[c, t, m] = (k // 18, (k % 18) // 6, k % 6)
    assert len({tuple(x) for x in a.reshape(-1, 3)}) == 144
    return a

b = L.ms(L.att)
print(f"\nlayouts (bit-identical), delta vs today's model {b:.2f} ms:")
for g in (6, 3, 2, 1):
    print(f"  pieces of {g} rb ({18 // g:2d} pieces per column per head): {L.ms(L.att.remap_rows(assign(g))) - b:+.2f} ms")
at = L.A.default_att().thinned(0.109)
pb = L.replace(L.pb, R=450)
b2 = L.ms(at, prm=pb)
print("with the map thinned to the in-situ range (cap 10.9 %, R 450):")
for g in (6, 3, 2, 1):
    print(f"  pieces of {g} rb: {L.ms(at.remap_rows(assign(g)), prm=pb) - b2:+.2f} ms")
