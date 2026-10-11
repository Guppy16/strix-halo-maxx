import re, sys
cur, n, sp, out = None, 0, 0, []
for line in open(sys.argv[1]):
    m = re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
    if m and not m.group(1).startswith(".L"):
        if cur: out.append((cur, n, sp))
        cur, n, sp = m.group(1), 0, 0
        continue
    if re.match(r"^\s+[0-9a-f]+:", line):
        n += 1; sp += line.count("[sp")
if cur: out.append((cur, n, sp))
for c, n, sp in out: print(f"  {c:40s} {n:5d} bundles  {sp:3d} sp refs")
