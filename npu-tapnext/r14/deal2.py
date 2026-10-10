"""Round 14 (research P60, single-TAP form): natural tile (c, i) -> (destination column s0 + i d, position p0 + i e). Both the
qkv Q drain and out_proj's O gather stay one TAP per natural column iff these are arithmetic in i, and the gather's stride
d x QKV/APAN column stride + e x object must be positive (d >= 1, or d = 0 with e = +1). Each (column, position) once."""
import sys

opts = []
for s0 in range(8):
    for d in (0, 1, 2, 3):
        for p0, e in ((0, 1), (2, -1)):
            if d == 0 and e == -1:
                continue
            if 0 <= s0 + 2 * d <= 7:
                opts.append((s0, d, p0, e))
sols = []


def rec(c, used, cur):
    if len(sols) > 200000:
        return
    if c == 8:
        sols.append(dict(cur))
        return
    for s0, d, p0, e in opts:
        cells = [(s0 + i * d, p0 + i * e) for i in range(3)]
        if any(x in used for x in cells):
            continue
        used.update(cells)
        cur[c] = (s0, d, p0, e)
        rec(c + 1, used, cur)
        del cur[c]
        used.difference_update(cells)


rec(0, set(), {})
print(len(sols), "deals")
good = []
for s in sols:
    dest = {(c, i): s[c][0] + i * s[c][1] for c in range(8) for i in range(3)}
    slow = [dest[(c, i)] for c in (5, 6) for i in range(3)]
    if len(set(slow)) == 6:
        good.append(s)
print(len(good), "spread the six slow tiles over six columns")
for s in good[:5]:
    print(s)
