"""Rev 16: BD ids written / pushed per tile in a .bin (which shim BDs are never used)."""
import collections
import sys

from aie.utils import txn_trace as T

w = T.load(sys.argv[1])
ops = T.decode(w)[1:]
used = collections.defaultdict(set)
pushed = collections.defaultdict(collections.Counter)
for o in ops:
    if o.kind in ("blockwrite", "patch"):
        a = o.addr
        col, row, reg = (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF
        if row == 0 and 0x1D000 <= reg < 0x1D200:
            used[(col, row)].add((reg - 0x1D000) // 0x20)
    if o.kind == "write32":
        a = o.addr
        col, row, reg = (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF
        if row == 0 and reg in (0x1D204, 0x1D20C, 0x1D214, 0x1D21C):
            pushed[(col, row)][(reg, o.value & 0xF)] += 1
for k in sorted(used):
    print(k, "BD ids written:", sorted(used[k]), "free:", sorted(set(range(16)) - used[k]))
for k in sorted(pushed):
    print(k, dict(sorted(pushed[k].items())))
