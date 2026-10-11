"""Round 21 POPLF0 (E7'' gate, research pass 17: 'lf0 on ldb is the same encoding class, UNVERIFIED'): can the B load slot pop and fill
the lf0 FIFO (vldb.pop.576 .. [p0, lf0, r24])? Same harness as r20/poplf1gen.py (two passes of 8 x (fill + 8 pops) over one 4,608 B
object, every popped panel stored; pass 2 must equal pass 1 and the source).
  probe0: control (vldb fill / pops on lf1, the deployed form)
  probe1: vldb.fill + vldb.pop on lf0
  probe2: vlda.fill + vlda.pop on lf0 (the deployed A form)
  probe3: vldb.fill on lf0, pops alternating vlda / vldb
  probe4: vlda.fill on lf0, all pops vldb
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "poplf0.s"


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
    lf = "[p1, lf1, r25]" if k == 0 else "[p0, lf0, r24]"

    def pop(r, i):
        slot = {0: "b", 1: "b", 2: "a", 3: "ab"[i % 2], 4: "b"}[k]
        return f"\tvld{slot}.pop.576\tex{r}, {lf}"
    fill = f"\tvld{'a' if k in (2, 4) else 'b'}.fill.512\t{lf}"
    restart = ["\tmov\tp1, p5", "\tmova\tr25, #0"] if k == 0 else ["\tmov\tp0, p5", "\tmova\tr24, #0"]
    wait = ["\tnop"] * 9
    for ps in range(2):
        if ps:
            s += restart + ["\tnop"] * 6
        for b in range(8):
            s += [fill] + [pop(r, 8 * b + r) for r in range(8)] + wait + stores(range(8))
    s += ["\tret\tlr"] + ["\tnop"] * 5 + [f".Lfunc_end_probe{k}:", f"\t.size\tprobe{k}, .Lfunc_end_probe{k}-probe{k}"]
    return s


OUT.write_text("\t.text\n" + "\n".join(line for k in range(5) for line in probe(k)) + "\n")
print("wrote", OUT)
