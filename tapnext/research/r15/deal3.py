"""rev 10: which query deals can the two GATHERS express with ONE head-major TAP per column?

Gather form (P60g): the qkv drains and the attention O drains stay as they are (every core writes its own slots); the deal is
done by the two consumers' A fills, which today are one shim TAP per column [12 heads][3 tiles][3456 B] into the memtile A slot
(loaded in per-head chunks by A_RT: BdIteration(size 12, stride CH) with a lock per chunk, so the stream must stay
HEAD-MAJOR: all 6 pieces of head h before head h + 1).
  attention A fill, dealt column c':  stream (head, i, j) <- natural Q piece u = 32 c + 8 T + H   (units of 1,728 B:
                                     QKV slot stride 3 x 4608 = 8 units, column stride 12 x 4608 = 32 units)
                                     -> slot position p = 2 t + h (units of 1,728 B inside the head's 10,368 B chunk)
  out_proj A fill, natural column c:  stream (head, i, j) <- dealt O piece v = (c', p) (APAN column stride, rb 3 p)
                                     -> slot position k = 2 T + H
One TAP = 4 dims: head, i, j, words. So per column the 6 pieces must be a 2D affine family in (i, j) (sizes 6x1, 3x2, 2x3) with
NON-NEGATIVE strides, both on the source side and on the destination side (the memtile S2MM may write with its own 2D pattern
inside a chunk: [unverified] that a memtile BD with BdIteration also takes a 3D in-chunk pattern).
Exact cover search over the 8 dealt columns; then the inverse side is checked; deals are scored by the distinct natural columns
per dealt column (balance) and by sim17 in lev17.py.
"""
import itertools

VALID = [32 * c + 8 * T + H for c in range(8) for T in range(3) for H in range(2)]
VS = set(VALID)
SHAPES = [(6, 1), (3, 2), (2, 3)]


def dest_maps(n1, n2):
    """affine non-negative bijections (i, j) -> p in 0..5"""
    out = []
    for p0, a, b in itertools.product(range(6), range(0, 6), range(0, 6)):
        ps = [p0 + a * i + b * j for i in range(n1) for j in range(n2)]
        if sorted(ps) == list(range(6)):
            out.append((p0, a, b))
    return out


def lines():
    """(shape, u0, a, b, dest) families of 6 natural pieces"""
    res = []
    for n1, n2 in SHAPES:
        dm = dest_maps(n1, n2)
        for u0 in VALID:
            for a in range(0, 256):
                if n1 > 1 and a == 0:
                    continue
                if n1 == 1 and a:
                    continue
                for b in (range(0, 256) if n2 > 1 else (0,)):
                    if n2 > 1 and b == 0:
                        continue
                    us = [u0 + a * i + b * j for i in range(n1) for j in range(n2)]
                    if len(set(us)) == 6 and all(u in VS for u in us):
                        res.append(((n1, n2), tuple(us), dm))
                        if n2 == 1:
                            break
    return res


L = lines()
print(len(L), "single-TAP source families of 6 natural pieces")
by_first = {}
for sh, us, dm in L:
    by_first.setdefault(min(us), []).append((sh, us, dm))

sols = []


def cover(used, chosen):
    if len(sols) >= 20000:
        return
    if len(chosen) == 8:
        sols.append(list(chosen))
        return
    first = min(u for u in VALID if u not in used)
    for sh, us, dm in by_first.get(first, []):
        if used.isdisjoint(us):
            chosen.append((sh, us, dm))
            cover(used | set(us), chosen)
            chosen.pop()


cover(set(), [])
print(len(sols), "exact covers (capped at 20000)")


def natcol(u):
    return u // 32


def inverse_ok(sol):
    """assign the 8 families to dealt columns (any order) and slot positions by the family's dest map; the out_proj side must
    be single-TAP for every natural column: its 6 pieces' (c', p) a 2D affine family over v = 1000 c' + p (APAN column
    stride >> 6 positions) with non-negative strides and an affine dest k = 2T + H."""
    best = None
    for perm in [range(8)]:  # dealt column order does not change affinity of v up to the column labels: try identity + sorted
        for dmc in itertools.product(*[f[2][:4] for f in sol]):  # first few dest maps per family
            loc = {}
            for cprime, (fam, d) in enumerate(zip(sol, dmc)):
                (n1, n2), us, _ = fam
                p0, a, b = d
                for idx, u in enumerate(us):
                    i, j = divmod(idx, n2)
                    loc[u] = (cprime, p0 + a * i + b * j)
            ok = True
            for c in range(8):
                pcs = sorted(loc[32 * c + 8 * T + H] + (2 * T + H,) for T in range(3) for H in range(2))
                vs = {(cp, p): k for cp, p, k in pcs}
                if not single_tap(vs):
                    ok = False
                    break
            if ok:
                return dmc
    return best


def single_tap(vs):
    pts = list(vs)
    for n1, n2 in SHAPES:
        for start in pts:
            for s1 in pts:
                for s2 in (pts if n2 > 1 else [start]):
                    d1 = (s1[0] - start[0], s1[1] - start[1])
                    d2 = (s2[0] - start[0], s2[1] - start[1])
                    if n1 > 1 and d1 == (0, 0) or n2 > 1 and d2 == (0, 0):
                        continue
                    if any(1000 * d[0] + d[1] < 0 for d in (d1, d2)):
                        continue
                    fam = [(start[0] + i * d1[0] + j * d2[0], start[1] + i * d1[1] + j * d2[1]) for i in range(n1) for j in range(n2)]
                    if sorted(fam) != sorted(pts):
                        continue
                    ks = [vs[f] for f in fam]
                    k0 = ks[0]
                    ka = ks[n2] - k0 if n1 > 1 else 0
                    kb = ks[1] - k0 if n2 > 1 else 0
                    if ka >= 0 and kb >= 0 and all(ks[i * n2 + j] == k0 + i * ka + j * kb for i in range(n1) for j in range(n2)):
                        return True
    return False


good = []
for s in sols:
    spread = sorted(len({natcol(u) for u in f[1]}) for f in s)
    good.append((sum(spread), spread, s))
good.sort(key=lambda g: -g[0])
seen = set()
shown = 0
for sc, spread, s in good:
    key = tuple(spread)
    if key in seen:
        continue
    seen.add(key)
    inv = inverse_ok(s)
    print(f"spread {spread} (sum {sc}); out_proj side single-TAP: {'yes' if inv else 'no'}")
    for f in s:
        print("    ", f[0], [(u // 32, (u % 32) // 8, u % 8) for u in f[1]])
    shown += 1
    if shown >= 6:
        break
