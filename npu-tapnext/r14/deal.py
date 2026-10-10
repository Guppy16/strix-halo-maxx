"""Round 14 (research P60, the form that keeps every DMA a single TAP): deal query tiles to columns so that natural column c's
tile i is processed by column s0(c) + i d(c) (an arithmetic progression per column: the qkv epilogue's Q drain and out_proj's
A gather stay one TAP per column; the qkv drain's junk object i = 3 must land in a real column too). Each position i must be a
permutation of the 8 columns. Score: the slow tiles (rows 720-1007 = natural columns 5 and 6) on 6 distinct columns, and the
spread of all tiles' natural columns per destination column."""
import itertools

opts = {c: [(s0, d) for s0 in range(8) for d in (-2, -1, 0, 1, 2) if 0 <= s0 + 2 * d <= 7] for c in range(8)}
best = []


def rec(c, used, cur):
    if c == 8:
        dest = {(cc, i): s0 + i * d for cc, (s0, d) in cur.items() for i in range(3)}
        slow = [dest[(cc, i)] for cc in (5, 6) for i in range(3)]
        if len(set(slow)) == 6:
            # per destination column: the natural columns it serves (prefer far apart)
            spread = sum(max(cc for cc in range(8) for i in range(3) if dest[(cc, i)] == k) -
                         min(cc for cc in range(8) for i in range(3) if dest[(cc, i)] == k) for k in range(8))
            best.append((spread, dict(cur)))
        return
    for s0, d in opts[c]:
        cols = [s0 + i * d for i in range(3)]
        if any(cols[i] in used[i] for i in range(3)):
            continue
        for i in range(3):
            used[i].add(cols[i])
        cur[c] = (s0, d)
        rec(c + 1, used, cur)
        del cur[c]
        for i in range(3):
            used[i].discard(cols[i])


rec(0, [set(), set(), set()], {})
best.sort(key=lambda b: -b[0])
print(len(best), "solutions")
for sp, cur in best[:3]:
    print(sp, cur, {k: [(cc, i) for cc in range(8) for i in range(3) if cur[cc][0] + i * cur[cc][1] == k] for k in range(8)})
