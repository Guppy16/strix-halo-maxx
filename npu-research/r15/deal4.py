"""rev 10: P60g, the gather-form query deal that keeps every A fill ONE head-major TAP per column (deal3.py's constraint).
Construction (verified below piece by piece):
  group G = {x + j d + H s : j 0..2, H 0..1} (two arithmetic triples, s apart); dealt column phys(T, e) = x + T d + e s takes
  half e of tile T from all six natural columns of G, natural column (j, H) sitting at slot position p = 2 j + H.
  pair {a, b} (the two columns outside G): dealt a takes half 0 of every tile of a (positions 0-2) and of b (3-5); dealt b
  the halves 1.
Checks: bijection; per dealt column the attention gather sources are one affine family with non-negative strides in the order
of the slot positions (linear memtile write); per natural column the out_proj gather sources likewise in natural order.
Prices with sim17 (kappa 0.6, 4 seeds), also with the map's column clustering moved (cyclic shifts of the natural columns).
    python3 deal4.py"""
from dataclasses import replace
import itertools

import numpy as np

import attsim17 as A
from fit17 import kappa_att

QPH, COLQ, TQ, HQ = 442368, 55296, 13824, 1728  # QKV: head, column, tile slot, half (bytes)
COLA, PA = 10 ** 7, 1728  # APAN: a column stride (any large value) and one 3-rb position


def deal_g6(x, d, s, pair):
    """dict (dealt col, pos) -> (natural col, tile, half)"""
    m = {}
    for T in range(3):
        for e in range(2):
            phys = x + T * d + e * s
            for j in range(3):
                for H in range(2):
                    m[(phys, 2 * j + H)] = (x + j * d + H * s, T, e)
    a, b = pair
    for T in range(3):
        m[(a, T)], m[(a, 3 + T)] = (a, T, 0), (b, T, 0)
        m[(b, T)], m[(b, 3 + T)] = (a, T, 1), (b, T, 1)
    return m


def affine_ok(seq):
    """seq: list of 6 integers in stream order; one TAP = [n1][n2] affine with non-negative strides"""
    for n1, n2 in ((6, 1), (3, 2), (2, 3)):
        a = seq[n2] - seq[0] if n1 > 1 else 0
        b = seq[1] - seq[0] if n2 > 1 else 0
        if a < 0 or b < 0:
            continue
        if all(seq[i * n2 + j] == seq[0] + i * a + j * b for i in range(n1) for j in range(n2)):
            return (n1, n2, a, b)
    return None


def check(m):
    assert len(m) == 48 and len(set(m.values())) == 48, "not a bijection"
    att = {}
    for c in range(8):
        src = [m[(c, p)] for p in range(6)]
        att[c] = affine_ok([n * COLQ + T * TQ + h * HQ for n, T, h in src])
        assert att[c], (c, src)
    inv = {v: k for k, v in m.items()}
    out = {}
    for n in range(8):
        loc = [inv[(n, T, h)] for T in range(3) for h in range(2)]
        out[n] = affine_ok([c * COLA + p * PA for c, p in loc])
        assert out[n], (n, loc)
    return att, out


def layout_of(m):
    a = np.zeros((8, 3, 6, 3), int)
    for (c, p), (n, T, h) in m.items():
        t, hh = divmod(p, 2)
        for r in range(3):
            a[c, t, 3 * hh + r] = (n, T, 3 * h + r)
    return a


DEALS = {
    "G {0,3,6}+{1,4,7}, pair {2,5}": (0, 3, 1, (2, 5)),
    "G {1,2,3}+{4,5,6}, pair {0,7}": (1, 1, 3, (0, 7)),
    "G {1,3,5}+{2,4,6}, pair {0,7}": (1, 2, 1, (0, 7)),
    "G {0,1,2}+{5,6,7}, pair {3,4}": (0, 1, 5, (3, 4)),
    "G {0,2,4}+{3,5,7}, pair {1,6}": (0, 2, 3, (1, 6)),
}
if __name__ == "__main__":  # noqa
    for name, (x, d, s, pr) in DEALS.items():
        att, out = check(deal_g6(x, d, s, pr))
        print(f"{name}: single-TAP both sides OK; attention TAPs {sorted(set(att.values()))}; out_proj TAPs {sorted(set(out.values()))}")
    raw = A.load()
    P = replace(A.P17(), VRB=136.0)
    from lev17 import layout  # noqa: E402

    def price(fn, shift=0):
        vals = []
        for sd in (0, 1, 2, 3):
            at = kappa_att(raw, 0.6, sd)
            if shift:  # move the clustering: natural column n takes the map of column (n + shift) mod 8
                perm = np.zeros((8, 3, 6, 3), int)
                for c in range(8):
                    for t in range(3):
                        for r in range(6):
                            perm[c, t, r] = ((c + shift) % 8, t, r)
                at = A.remap(at, perm)
            b0 = A.ms(at, P)
            vals.append(A.ms(fn(at), P) - b0)
        return np.mean(vals)

    print("\nprice vs r14f (ms/step; sim17 kappa 0.6), map as measured / clustering moved by +2, +4, +6 columns (mean):")
    cands = {"identity (today)": lambda a: a, "rr 3-rb pieces (P60, multi-TAP)": lambda a: A.remap(a, layout(3))}
    for name, (x, d, s, pr) in DEALS.items():
        L = layout_of(deal_g6(x, d, s, pr))
        cands[name] = lambda a, L=L: A.remap(a, L)
    for name, fn in cands.items():
        v0 = price(fn)
        vs = [price(fn, sh) for sh in (2, 4, 6)]
        print(f"  {v0:+6.2f}  {np.mean(vs):+6.2f}   {name}")
