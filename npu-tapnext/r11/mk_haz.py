"""Round 11: register hazard probes. Each probe tests one WRITER op: per trial d = 0..8, x1 := 1.0 (bf16 broadcast),
the writer into x1, then `vst x1` d bundles later (d = 0: the same bundle) to out vector d. Stored 1.0 = the read saw
the OLD value; anything else = the new one. Static bundle counts are printed so stalls show in the cycle count.
    python3 r11/mk_haz.py OUT.s   (probes 1..7; run with apb.py --n 144 --dump, then r11/haz.py decodes)"""
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
	nop
	nop
.Lfunc_end_{f}:
	.size	{f}, .Lfunc_end_{f}-{f}
"""
SETUP = ["movxm\tr16, #16256", "vbcst.16\tx6, r16",  # x6 = 1.0 (the old value)
         "movxm\tr17, #16384", "vbcst.16\tx4, r17",  # x4 = 2.0
         "movxm\tr17, #16448", "vbcst.16\tx5, r17",  # x5 = 3.0
         "mov\tr17, p2", "vbcst.32\tx3, r17",  # x3 = the LUT table address in every lane
         "mova\tr1, #828", "mova\tr2, #24", "mov\ts0, #0", "mov\tp5, p1",
         "nop", "nop", "nop", "nop", "nop", "nop", "nop", "nop",
         "vmul.f\tdm0, x4, x5, r1", "nop", "nop", "nop", "nop", "nop", "nop", "nop"]  # cml0 = 6.0
WRITERS = {
    1: ("vlda\tx1, [p0, #0]", "lda"),
    2: ("vldb.4x64.lo\twl1, wl3", "ldb"),
    3: ("vadd.32\tx1, x4, x5", "mv"),
    4: ("vconv.bf16.fp32\tx1, cml0", "st"),
    5: ("vfloor.s32.bf16\tx1, wl4, s0", "st"),
    6: ("vshuffle\tx1, x4, x5, r2", "mv"),
    7: ("vmin_ge.bf16\tx1, r16, x4, x5", "mv"),
}


def probe(k):
    w, slot = WRITERS[k]
    f = f"probe{k}"
    L = [HEAD.format(f=f)] + ["\t" + s for s in SETUP]
    nb = len(SETUP)
    for d in range(9):
        L += ["\tvmov\tx1, x6", "\tnop", "\tnop", "\tnop", "\tnop"]  # x1 = 1.0, settled
        if d == 0 and slot != "st":
            L.append(f"\t{w};\t\tvst\tx1, [p5], #64")
            nb += 6
        elif d == 0:  # the writer is on the st slot: the store cannot share its bundle
            L += [f"\t{w}", "\tvst\tx1, [p5], #64"]
            nb += 7
        else:
            L += [f"\t{w}"] + ["\tnop"] * (d - 1) + ["\tvst\tx1, [p5], #64"]
            nb += 5 + d + 1
        L += ["\tnop"] * 8  # let everything land before the next trial
        nb += 8
    L.append(TAIL.format(f=f))
    return "\n".join(L), nb


def waw(k, load, slot2):
    """WAW: an older long-latency load into x1, then d = 1..8 bundles later a younger vmov x1 <- x6 (1.0) that lands
    FIRST; read x1 16 bundles later. Stored load value = the older write landed last (in-order completion is NOT
    enforced, but nothing is lost); 1.0 = the in-flight load's write was dropped / overtaken."""
    f = f"probe{k}"
    L = [HEAD.format(f=f)] + ["\t" + s for s in SETUP]
    for d in range(9):
        L += ["\tvmov\tx1, x5", "\tnop", "\tnop", "\tnop", "\tnop", "\t" + load]
        L += ["\tnop"] * max(d - 1, 0)
        if d == 0:
            L[-1] = L[-1] + ";\t\tvmov\tx1, x6"
        else:
            L += ["\tvmov\tx1, x6"]
        L += ["\tnop"] * 16 + ["\tvst\tx1, [p5], #64"] + ["\tnop"] * 4
    L.append(TAIL.format(f=f))
    return "\n".join(L)


def mid(k, load, wr="vmov\tx1, x6", wlat=2):
    """H5: an older load into x1 in flight (lands at +8); 1 bundle later a younger write to x1 (lands at +wlat+2);
    read x1 (vst) d = 0..5 bundles after the younger write landed (still before the load lands when d is small).
    Stored 1.0 in the loaded half = the younger write was seen (no suppression); 3.0 (the pre-load value) = suppressed."""
    f = f"probe{k}"
    L = [HEAD.format(f=f)] + ["\t" + s for s in SETUP]
    for d in range(6):
        L += ["\tvmov\tx1, x5", "\tnop", "\tnop", "\tnop", "\tnop", "\t" + load, "\t" + wr]
        L += ["\tnop"] * (wlat + d) + ["\tvst\tx1, [p5], #64"] + ["\tnop"] * 16
    L.append(TAIL.format(f=f))
    return "\n".join(L)


READERS = {  # reader op of x1 (old 3.0, new 1.0) -> its result register, then how it is stored
    1: ("vshuffle\tx2, x1, x1, r2", "vst\tx2, [p5], #64"),
    2: ("vadd.32\tx2, x1, x7", "vst\tx2, [p5], #64"),
    3: ("vmin_ge.bf16\tx2, r16, x1, x1", "vst\tx2, [p5], #64"),
    4: ("vmul.f\tdm1, x1, x4, r1", "vst.conv.bf16.fp32\tcml1, [p5], #64"),
    5: ("vfloor.s32.bf16\tx2, wl1, s0", "vst\tx2, [p5], #64"),
    6: ("vmov\twl2, wh1", "vst\tx2, [p5], #64"),
    7: ("vst\tx1, [p5], #64", None),
}


def rd(k, writer="vmov\tx1, x6", d0=0, nd=7):
    """reader stage: x1 := 3.0, then `vmov x1, x6` (1.0, lat 2), the reader d = 0..6 bundles later; stored result shows
    whether the reader saw 3.0 (old) or 1.0 (new): the first d that sees new = 2 - r + 1 for a read at stage r"""
    r, st = READERS[k]
    f = f"probe{k}"
    L = [HEAD.format(f=f)] + ["\t" + s for s in SETUP] + ["\tvbcst.32\tx7, r0", "\tmova\tr0, #0", "\tvbcst.32\tx7, r0", "\tnop", "\tnop"]
    for d in range(d0, d0 + nd):
        L += ["\tvmov\tx1, x5", "\tvmov\tx2, x5", "\tnop", "\tnop", "\tnop", "\tnop"]
        if d == 0:
            L += [f"\t{writer};\t\t" + r] if not r.startswith(("vshuffle", "vadd", "vmin", "vmov")) else [f"\t{writer}", "\t" + r]
        else:
            L += [f"\t{writer}"] + ["\tnop"] * (d - 1) + ["\t" + r]
        L += ["\tnop"] * 10 + (["\t" + st] if st else []) + ["\tnop"] * 4
    L.append(TAIL.format(f=f))
    return "\n".join(L)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] in ("rd", "rdl", "rdc"):
    w = {"rd": ("vmov\tx1, x6", 0), "rdl": ("vlda\tx1, [p0, #0]", 4), "rdc": ("vconv.bf16.fp32\tx1, cml0", 0)}[sys.argv[2]]
    open(sys.argv[1], "w").write("\t.text\n" + "\n".join(rd(k, w[0], w[1]) for k in READERS) + "\n")
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "mid":
    out = ["\t.text", mid(1, "vldb.4x64.lo\twl1, wl3"), mid(2, "vlda\tx1, [p0, #0]"), mid(3, "vldb.4x64.lo\twl1, wl3", "vmax_lt.bf16\tx1, r16, x6, x6")]
    open(sys.argv[1], "w").write("\n".join(out) + "\n")
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "waw":
    out = ["\t.text", waw(1, "vlda\tx1, [p0, #0]", "lda"), waw(2, "vldb.4x64.lo\twl1, wl3", "ldb"), waw(3, "vldb\tx1, [p0, #0]", "ldb")]
    open(sys.argv[1], "w").write("\n".join(out) + "\n")
    sys.exit(0)

if __name__ == "__main__":
    out, counts = ["\t.text"], {}
    for k in WRITERS:
        t, nb = probe(k)
        out.append(t)
        counts[k] = nb
    open(sys.argv[1], "w").write("\n".join(out) + "\n")
    print("static bundles per probe (excl. call / return):", counts)
