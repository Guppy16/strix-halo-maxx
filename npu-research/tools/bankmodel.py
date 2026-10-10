"""bankmodel: the L1 bank-conflict rule of AIE2P, fitted to the round-8 mmb sweep (npu-research rev 4).

Rule (fits all 256 mmb placements exactly, see --check):
    a bundle costs max(1, max over banks b of n_b) cycles, n_b = number of memory accesses of that bundle in bank b.
I.e. same-bank accesses issued in ONE bundle serialise, one extra cycle per extra access; accesses in different
bundles never conflict (no adjacent-bundle effect, no read/write asymmetry: load+load, load+store, store+pop all cost 1).

Classes of memory op are recognised by their pointer register (mm_core / mm_core_split conventions):
    A  = vlda.pop / vlda.fill on p0      B  = vldb.pop / vldb.fill on p1
    Cl = vlda on p3 (accumulator loads)  Cs = vst on p4 (accumulator stores)  (mm_core: p3/p4 both point into C)

usage:
    python3 bankmodel.py S.s --check MMB_RUN.out      # predict every placement of the sweep, compare
    python3 bankmodel.py S.s --pairs                  # co-issue counts per class pair (the conflict cost per pair)
    python3 bankmodel.py S.s --banks A=1,B=3,Cl=2,Cs=2
"""

import argparse
import itertools
import re
from collections import Counter


def classify(op):
    op = op.strip()
    if op.startswith("vlda.pop") or op.startswith("vlda.fill"):
        return "A" if "[p0" in op else "B"
    if op.startswith("vldb.pop") or op.startswith("vldb.fill"):
        return "B" if "[p1" in op else "A"
    if op.startswith("vlda ") or op.startswith("vldb "):
        return "Cl"
    if op.startswith("vst"):
        return "Cs"
    return None


def bundles(path):
    out = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith(".") or s.endswith(":"):
            continue
        out.append([c for c in (classify(o) for o in s.split(";")) if c])
    return out


def cycles(bs, bank):
    tot = 0
    for b in bs:
        n = Counter(bank[c] for c in b)
        tot += max([1] + list(n.values()))
    return tot


def pairs(bs):
    p = Counter()
    for b in bs:
        for x, y in itertools.combinations(sorted(b), 2):
            p[x + "-" + y] += 1
        if len(b) >= 3:
            p["triple:" + "".join(sorted(b))] += 1
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s")
    ap.add_argument("--check")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--banks")
    o = ap.parse_args()
    bs = bundles(o.s)
    free = cycles(bs, {"A": 0, "B": 1, "Cl": 2, "Cs": 3})
    print(f"{o.s}: {len(bs)} bundles, conflict-free {free} cycles")
    if o.pairs:
        for k, v in sorted(pairs(bs).items()):
            print(f"  {k:12s} {v:4d} bundles")
    if o.banks:
        bank = dict(kv.split("=") for kv in o.banks.split(","))
        print(f"  {o.banks}: {cycles(bs, bank)} cycles (+{cycles(bs, bank) - free})")
    if o.check:
        rows = re.findall(r"\((\d), (\d), (\d), (\d)\)\s+([\d.]+)", open(o.check).read())
        base = None
        worst = 0
        for r in rows:
            bank = dict(zip(["A", "B", "Cl", "Cs"], r[:4]))
            pred = cycles(bs, bank)
            meas = float(r[4])
            if base is None:
                base = meas - free
            worst = max(worst, abs(pred + base - meas))
        print(f"  {len(rows)} placements, call/loop constant {base:.0f} cycles, max |pred - meas| = {worst:.1f}")


if __name__ == "__main__":
    main()
