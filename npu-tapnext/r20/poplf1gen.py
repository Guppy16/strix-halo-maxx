"""Round 20 POPLF1 (research rev 16): can the A load slot pop the lf1 FIFO (vlda.pop.576 .. [p1, lf1, r25])? (E7'' gate)
probe0 control (vldb pops), probe1 every pop vlda on lf1, probe2 vlda / vldb alternating, probe3 vlda.fill + vlda pops, probe4
vlda pops on lf1 while vldb pops lf0 from the same object (two live streams).
Adapted from round 19: one-core probes of the B pop stream's wrap (research PROPOSALS rev 14, PWRAP / PADDA) -> r19/wrapprobe.s.

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

OUT = Path(__file__).resolve().parent / "poplf1.s"


def stores(regs):
    s = []
    for r in regs:
        s += [f"\tvst\tx{r}, [p3], #64", f"\tst\tel{r}, [p4], #4", f"\tst\teh{r}, [p4], #4"]
    return s



def probe(k):
    s = [f"\t.section\t.text.probe{k},\"ax\",@progbits", f"\t.globl\tprobe{k}", "\t.p2align\t4", f"\t.type\tprobe{k},@function", f"probe{k}:",
         "\tmov\tp3, p1", "\tmov\tp4, p1", "\tmov\tp1, p0", "\tmov\tp5, p0", "\tmova\tr25, #0", "\tmova\tr24, #0"]
    s += ["\tpadda\t[p4], #448"] * 18 + ["\tpadda\t[p4], #320"]
    s += ["\tnop"] * 6
    def pop(r, i):
        if k == 0 or (k == 2 and i % 2):
            return f"\tvldb.pop.576\tex{r}, [p1, lf1, r25]"
        return f"\tvlda.pop.576\tex{r}, [p1, lf1, r25]"
    fill = "\tvlda.fill.512\t[p1, lf1, r25]" if k == 3 else "\tvldb.fill.512\t[p1, lf1, r25]"
    wait = ["\tnop"] * 9
    for ps in range(2):
        if ps:
            s += ["\tmov\tp1, p5", "\tmova\tr25, #0"] + ["\tnop"] * 6
        for b in range(8):
            s += [fill] + [pop(r, 8 * b + r) for r in range(8)] + wait + stores(range(8))
    s += ["\tret\tlr"] + ["\tnop"] * 5 + [f".Lfunc_end_probe{k}:", f"\t.size\tprobe{k}, .Lfunc_end_probe{k}-probe{k}"]
    return s


OUT.write_text("\t.text\n" + "\n".join(line for k in range(4) for line in probe(k)) + "\n")
print("wrote", OUT)
