"""Round 19: one-core probes of the B pop stream's wrap (research PROPOSALS rev 14, PWRAP / PADDA) -> r19/wrapprobe.s.

Two passes over one 4,608 B B object (8 x (fill + 8 pops)); every popped panel (64 B x + 8 B e) is stored, pass 2 must equal pass
1 (r19/wrapb.py checks both against the source). The wrap itself runs at ROT's density: the last pop of pass 1, the restart and
pass 2's fill + 4 pops in consecutive bundles.
  probe0: control - plain last pop, then 12 bundles later mov p1, p5 + mova r25, #0, nops, fill (a fresh stream)
  probe1: PWRAP, last pop = vldb.pop.576.2d .. d2 with m2 = -4544, dn2 0, dj2 64, dc2 0, mov r25, #0 in its bundle; fill next
  probe2: PWRAP with m2 = -4608 (in case the 2D step adds to the line step)
  probe3: PADDA, plain last pop + mov r25, #0; padda [p1], m2 (-4608) next bundle; fill the bundle after (g1f's sequence)
  probe4: PADDA with the fill in the SAME bundle as padda (negative control: should fail if the padda lands late)
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "wrapprobe.s"


def stores(regs):
    s = []
    for r in regs:
        s += [f"\tvst\tx{r}, [p3], #64", f"\tst\tel{r}, [p4], #4", f"\tst\teh{r}, [p4], #4"]
    return s


def probe(k):
    s = [f"\t.section\t.text.probe{k},\"ax\",@progbits", f"\t.globl\tprobe{k}", "\t.p2align\t4", f"\t.type\tprobe{k},@function", f"probe{k}:",
         "\tmov\tp3, p1", "\tmov\tp4, p1", "\tmov\tp1, p0", "\tmov\tp5, p0", "\tmova\tr25, #0"]
    s += ["\tpadda\t[p4], #448"] * 18 + ["\tpadda\t[p4], #320"]  # e words at out + 8,384 (x: 128 x 64 B first)
    m2 = {1: -4544, 2: -4608, 3: -4608, 4: -4608}.get(k)
    if m2 is not None:
        s += [f"\tmovxm\tm2, #{m2}"]
    if k in (1, 2):
        s += ["\tmova\tdn2, #0", "\tmova\tdj2, #64", "\tmova\tdc2, #0"]
    s += ["\tnop"] * 6
    pop = "\tvldb.pop.576\tex{r}, [p1, lf1, r25]"
    fill = "\tvldb.fill.512\t[p1, lf1, r25]"
    wait = ["\tnop"] * 9
    # pass 1, batches 0..6: fill + 8 dense pops, store
    for b in range(7):
        s += [fill] + [pop.format(r=r) for r in range(8)] + wait + stores(range(8))
    # pass 1 batch 7 + the wrap + pass 2 batch 0's fill and first 4 pops, dense
    s += [fill] + [pop.format(r=r) for r in range(7)]
    if k in (1, 2):
        s += ["\tvldb.pop.576.2d\tex7, [p1, lf1, r25, d2];\tmov\tr25, #0", fill]
    elif k == 3:
        s += [pop.format(r=7) + ";\tmov\tr25, #0", "\tpadda\t[p1], m2", fill]
    elif k == 4:
        s += [pop.format(r=7) + ";\tmov\tr25, #0", "\tpadda\t[p1], m2;\tvldb.fill.512\t[p1, lf1, r25]"]
    else:
        s += [pop.format(r=7)] + wait + stores(range(8)) + ["\tmov\tp1, p5", "\tmova\tr25, #0"] + ["\tnop"] * 6 + [fill]
    s += [pop.format(r=r) for r in range(8, 12)] + wait + (stores(range(8)) if k else []) + stores(range(8, 12))
    s += [pop.format(r=r) for r in range(4)] + wait + stores(range(4))  # pass 2 batch 0, pops 4..7 (same position, no fill)
    for b in range(1, 8):
        s += [fill] + [pop.format(r=r) for r in range(8)] + wait + stores(range(8))
    s += ["\tret\tlr"] + ["\tnop"] * 5 + [f".Lfunc_end_probe{k}:", f"\t.size\tprobe{k}, .Lfunc_end_probe{k}-probe{k}"]
    return s


OUT.write_text("\t.text\n" + "\n".join(line for k in range(5) for line in probe(k)) + "\n")
print("wrote", OUT)
