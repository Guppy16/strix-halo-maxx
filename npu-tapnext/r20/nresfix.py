"""Round 20 NRES: IRON 1.4.4's pathfinder finds no route for two packet sources on one tile, so design8 declares the panel flow (packet
5) from the EW core's MM2S 0 (beside X', packet 3) and the panels really leave on MM2S 1. This patch gives each EW tile's switchbox a
packet_rules(DMA : 1) block that sends packet 5 to the same arbiter / master as the DMA 0 rule that matches it.
usage: python3 r20/nresfix.py IN.mlir OUT.mlir [--row 5]"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--row", type=int, default=5)
ap.add_argument("--pkt", type=int, default=5)
o = ap.parse_args()
lines = open(o.inp).read().split("\n")
n = 0
for c in range(8):
    k = next(i for i, ln in enumerate(lines) if re.search(rf"%switchbox_{c}_{o.row} = aie\.switchbox\(%tile_{c}_{o.row}\)", ln))
    e = next(i for i in range(k, len(lines)) if lines[i].startswith("    }"))
    j = next(i for i in range(k, e) if "aie.packet_rules(DMA : 0)" in lines[i])
    assert not any("aie.packet_rules(DMA : 1)" in lines[i] for i in range(k, e)), (c, "DMA 1 rules exist")
    rules = []
    i = j + 1
    while "aie.rule(" in lines[i]:
        m = re.search(r"aie\.rule\((\d+), (\d+), (%\w+)\)", lines[i])
        rules.append((int(m.group(1)), int(m.group(2)), m.group(3)))
        i += 1
    hit = [r for r in rules if (o.pkt & r[0]) == (r[1] & r[0])]
    assert len(hit) == 1, (c, rules)
    end = i  # the closing line of the DMA 0 block
    ind = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
    loc = re.search(r"( loc\(.*\))$", lines[j])
    loc = loc.group(1) if loc else ""
    lines[end + 1:end + 1] = [f"{ind}aie.packet_rules(DMA : 1) {{", f"{ind}  aie.rule(31, {o.pkt}, {hit[0][2]}){loc}", f"{ind}}}{loc}"]
    n += 1
open(o.out, "w").write("\n".join(lines))
print(f"nresfix: {n} EW switchboxes route packet {o.pkt} from DMA 1")
