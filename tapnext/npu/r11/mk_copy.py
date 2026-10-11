"""Round 11 debug: copy-loop probes (in -> out, n vectors) to isolate addressing / loop / stack-ring semantics."""
import sys

HEAD = """	.section	.text.{f},"ax",@progbits
	.globl	{f}
	.p2align	4
	.type	{f},@function
{f}:"""
TAIL = """	ret	lr
	nop
	nop
	nop
	{x}
	nop
.Lfunc_end_{f}:
	.size	{f}, .Lfunc_end_{f}-{f}
"""
FULL = "\tnopa;\t\tnopb;\t\tnops;\t\tnopx;\t\tnopm;\t\tnopv"


def probe(f, ld, st, pre=(), ring=False):
    L = [HEAD.format(f=f)]
    L += ["\t" + p for p in pre]
    if ring:
        L += ["\tpaddxm\t[sp], #256", "\tmov\tp2, sp", "\tpadda\t[p2], #-256", "\tmov\tp3, p2",
              "\tmova\tdn3, #3", "\tmova\tdj3, #64", "\tmova\tm3, #-192", "\tmova\tdc3, #0",
              "\tmova\tdn4, #3", "\tmova\tdj4, #64", "\tmova\tm4, #-192", "\tmova\tdc4, #0"]
    L += [f"\tmovxm\tls, #{f}_start", f"\tmovxm\tle, #{f}_last", "\tadd.nc\tlc, r0, #0"] + ["\tnop"] * 6
    L += ["\t.p2align\t4", f"{f}_start:", "\t" + ld + ";\t\tnopb;\t\tnops;\t\tnopx;\t\tnopm;\t\tnopv"]
    L += [FULL] * 7
    if ring:
        L += ["\tnopa;\t\tnopb;\t\tvst.2d\tx0, [p2], d3;\t\tnopx;\t\tnopm;\t\tnopv"] + [FULL] * 7
        L += ["\tvlda.2d\tx0, [p3], d4;\t\tnopb;\t\tnops;\t\tnopx;\t\tnopm;\t\tnopv"] + [FULL] * 7
    L += [f"{f}_last:", "\tnopa;\t\tnopb;\t\t" + st + ";\t\tnopx;\t\tnopm;\t\tnopv"]
    L += ["\tnop"] * 4
    L.append(TAIL.format(f=f, x="paddxm\t[sp], #-256" if ring else "nop"))
    return "\n".join(L)


D2 = ["mov\tm0, r1", "mov\tm1, r1", "mova\tdn0, #7", "mova\tdj0, #64", "mova\tdc0, #0", "mova\tdn1, #7", "mova\tdj1, #64", "mova\tdc1, #0"]
out = ["\t.text",
       probe("probe1", "vlda\tx0, [p0], #64", "vst\tx0, [p1], #64"),
       probe("probe2", "vlda.2d\tx0, [p0], d0", "vst.2d\tx0, [p1], d1", pre=D2),
       probe("probe3", "vlda.2d\tx0, [p0], d0", "vst.2d\tx0, [p1], d1", pre=D2, ring=True)]
open(sys.argv[1], "w").write("\n".join(out) + "\n")
