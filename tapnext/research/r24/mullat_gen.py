"""pass 19 MULLAT: when does a scalar result become visible to a scalar reader, inside a zero-overhead-loop body and in straight
code? KTL's lock conditions are `mul r26, ..` two bundles before acq.cond / rel.cond (the .td: mul latency 2, acq reads at stage 1).
Each probe: r26 = 1 (old), then `mul r26, r20, r21` (5 x 7 = 35, new) or `add r26, r20, #1` (6), then `st r26` d bundles later.
ZOL probes run the body 2 trips (two words); straight probes one word. A word = the value the store saw: old (1) or new.
probe0-2: ZOL, mul, d = 1 / 2 / 3; probe3-5: straight, mul, d = 1 / 2 / 3; probe6 / 7: ZOL, add, d = 1 / 2.
usage: python3 r24/mullat_gen.py > r24/mullat.s"""

FULL = ["nopa", "nopb", "nops", "nopx", "nopm", "nopv"]
SLOT = dict(alu=3, st=2, mv=4)


def full(**ops):
    b = list(FULL)
    for sl, txt in ops.items():
        b[SLOT[sl]] = txt
    return "\t" + ";\t\t".join(b)


def head(f):
    return [f"\t.section\t.text.{f},\"ax\",@progbits", f"\t.globl\t{f}", "\t.p2align\t4", f"\t.type\t{f},@function", f"{f}:",
            "\tmova\tr20, #5", "\tmova\tr21, #7", "\tmova\tr22, #1"]


def tail(f):
    return ["\tret\tlr"] + ["\tnop"] * 5 + [f".Lfunc_end_{f}:", f"\t.size\t{f}, .Lfunc_end_{f}-{f}"]


def op(kind):
    return "mul r26, r20, r21" if kind == "mul" else "add r26, r20, #1"


def zol(f, kind, d):
    L = head(f) + [f"\tmovxm\tls, #.L{f}_body", f"\tmovxm\tle, #.L{f}_end", "\tmova\tr0, #2", "\tadd.nc\tlc, r0, #0"]
    L += ["\tnop"] * 4 + ["\t.p2align\t4"] + [full() for _ in range(6)]  # >= 80 B between the lc write and le
    body = [full() for _ in range(16)]
    body[0] = full(mv="mov r26, r22")
    body[9] = full(alu=op(kind))
    body[9 + d] = full(st="st r26, [p1], #4")
    L += ["\t.p2align\t4", f".L{f}_body:"] + body[:-1] + ["\t.p2align\t4", f".L{f}_end:", body[-1]]
    return L + tail(f)


def straight(f, kind, d):
    L = head(f) + ["\tmov\tr26, r22"] + ["\tnop"] * 8 + [f"\t{op(kind)}"] + ["\tnop"] * (d - 1) + ["\tst\tr26, [p1], #4"]
    return L + ["\tnop"] * 8 + tail(f)


out = ["\t.text"]
for k, (fn, kind, d) in enumerate([(zol, "mul", 1), (zol, "mul", 2), (zol, "mul", 3), (straight, "mul", 1), (straight, "mul", 2),
                                   (straight, "mul", 3), (zol, "add", 1), (zol, "add", 2)]):
    out += fn(f"probe{k}", kind, d)
print("\n".join(out))
