"""rev 10: attention levers on r14f with sim17 (attsim17, map thinned to kappa, 4 seeds averaged).
    python3 lev17.py [kappa]
Deltas in ms per step at depth 12 against the modelled r14f (VRB 136)."""
from dataclasses import replace
import sys

import numpy as np

import attsim17 as A
from fit17 import kappa_att

KAPPA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
SEEDS = (0, 1, 2, 3)
raw = A.load()
atts = [kappa_att(raw, KAPPA, s) for s in SEEDS]
P = replace(A.P17(), VRB=136.0)


def layout(g, order="rr"):
    """[8 cols, 3 tiles, 6 rb] -> natural (col, tile, rb): pieces of g row blocks dealt round robin"""
    a = np.zeros((8, 3, 6, 3), int)
    for c in range(8):
        for t in range(3):
            for m in range(6):
                pos = 6 * t + m
                n = c + 8 * (pos // g)
                k = n * g + pos % g
                a[c, t, m] = (k // 18, (k % 18) // 6, k % 6)
    return a


def layout_pairs(beta=4):
    """the affine-feasible deal: dealt column c' takes half 0 of each tile from natural c' and half 1 from c' ^ beta"""
    a = np.zeros((8, 3, 6, 3), int)
    for c in range(8):
        for t in range(3):
            for m in range(6):
                src = c if m < 3 else (c ^ beta)
                a[c, t, m] = (src, t, m)
    return a


def msm(fn, p, **kw):
    return np.mean([A.ms(fn(a), p, **kw) for a in atts])


def own_floor(p):
    """K + V cycles of the mean core (no waits, no imbalance): ms per step"""
    return np.mean([A.ms(a, replace(p, LOCK=False, D=10_000)) for a in atts])


ident = lambda a: a  # noqa: E731
base = msm(ident, P)
f128 = msm(ident, replace(P, ignore_flags=True))
print(f"kappa {KAPPA}: r14f model attention {base:.2f} ms/step; F128-like (no slow path in chunks 1-21) {f128 - base:+.2f}")
nolock = msm(ident, replace(P, LOCK=False))
print(f"  columns independent (memtile depth 8 still couples): {nolock - base:+.2f}; D=24 {msm(ident, replace(P, LOCK=False, D=24)) - base:+.2f}"
      f"; D=48 {msm(ident, replace(P, LOCK=False, D=48)) - base:+.2f}")


def row(name, fn, p):
    v = msm(fn, p) - base
    print(f"  {v:+6.2f}  {name}")
    return v


print("\nlockstep levers (bit-identical):")
for g in (1, 2, 3, 6):
    row(f"deal round robin, pieces of {g} rb", lambda a, g=g: A.remap(a, layout(g)), P)
for b in (4, 2, 1):
    row(f"pair deal (half 1 from column c ^ {b})", lambda a, b=b: A.remap(a, layout_pairs(b)), P)
row("deal rr 3 rb + columns independent D 8", lambda a: A.remap(a, layout(3)), replace(P, LOCK=False))
row("deal rr 3 rb + columns independent D 24", lambda a: A.remap(a, layout(3)), replace(P, LOCK=False, D=24))
row("3 L1 slots", ident, replace(P, SLOTS=3))
row("4 L1 slots", ident, replace(P, SLOTS=4))

print("\nslow-path levers (bit-identical), alone / after deal rr 3:")
SL = {
    "asm rescale (578 -> 200)": dict(R=200.0),
    "exact test -O2 (164 -> 80)": dict(T=80.0),
    "no first-pass rerun (row max kept by attkb; FP 393 -> 40)": dict(FP=40.0),
    "all three": dict(R=200.0, T=80.0, FP=40.0),
    "all three + exact rerun as P-only fix (XR 393 -> 393)": dict(R=200.0, T=80.0, FP=40.0),
}
d3 = lambda a: A.remap(a, layout(3))  # noqa: E731
b3 = msm(d3, P)
for n, o in SL.items():
    v1 = msm(ident, replace(P, **o)) - base
    v2 = msm(d3, replace(P, **o)) - b3
    print(f"  {v1:+6.2f} / {v2:+6.2f}  {n}")
print(f"\n  deal rr 3 alone {b3 - base:+.2f}; floor of the slow path after deal (F128 with deal) {msm(d3, replace(P, ignore_flags=True)) - base:+.2f}")
print("\nuniform levers:")
row("V phase 136 -> 100 per rb (sums folded into attv)", ident, replace(P, VRB=100.0))
row("batched K 261 -> 230 per rb (per-(tile,chunk) prep / flag scan in asm)", ident, replace(P, B=230.0))
row("chunk 0 1005 -> 600 per rb (QK-only first pass + test, no fast P)", ident, replace(P, C0=600.0))
row("head-group fixed H0/H1 -30 %", ident, replace(P, H0=P.H0 * 0.7, H1=P.H1 * 0.7))
