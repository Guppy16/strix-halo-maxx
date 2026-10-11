"""Round 12: probes of the block-vector output stream in asm (vst.push.576.conv.bfp16ebs8.fp32 + vst.flush.512.conv on
[p2, sf, r26]), for r11/apb.py (probeK(in p0, out p1, n r0, ab p2, cd p3, jump r1); n and the tables are ignored).

Each probe converts 64-lane blocks of the input (bf16 -> fp32 by vlda.conv into cml1 / cmh1) and pushes them to `out`:
  probe0  8 blocks, push + flush after every push (Peano's idiom)              -> out[0, 576)
  probe1  8 blocks, pushes only, one flush at the end
  probe2  8 blocks, pushes only, two flushes at the end
  probe3  4 blocks (push + flush), restart (p2 = out + 1024, r26 = 0), 4 more   -> [0, 288) and [1024, 1312)
  probe4  4 blocks (push + flush), no restart, 4 more after `padda [p2], #32`
  probe5  8 blocks with push / flush in consecutive bundles, blocks preloaded in dm1..dm4 (timing: back to back)
  probe6  like probe5 but push, push, ... with ONE flush per two pushes (8 pushes, 4 flushes)
  probe7  probe0 with the flush issued 3 bundles after its push (push -> flush distance)
r12/pushcheck.py decodes the dumps against pack7.encode_hw.

    python3 r12/gen_pushprobe.py OUT.s
"""

import sys

NOPS = 8


def head(name):
    return [f"\t.section\t.text.{name},\"ax\",@progbits", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]


def tail(name):
    return ["\tret\tlr", "\tnop", "\tnop", "\tnop", "\tnop", "\tnop", f".Lfunc_end_{name}:", f"\t.size\t{name}, .Lfunc_end_{name}-{name}"]


def load(dm="dm1"):
    k = dm[2:]
    return [f"\tvlda.conv.fp32.bf16\tcml{k}, [p0, #0]", f"\tvlda.conv.fp32.bf16\tcmh{k}, [p0, #64]", "\tpadda\t[p0], #128"] + ["\tnop"] * NOPS


def push(dm="dm1", flush=True, gap=0):
    out = [f"\tvst.push.576.conv.bfp16ebs8.fp32\t{dm}, [p2, sf, r26]"] + ["\tnop"] * gap
    if flush:
        out.append("\tvst.flush.512.conv\t[p2, sf, r26]")
    return out + ["\tnop"] * 4


def probe(k):
    n = f"probe{k}"
    L = head(n) + ["\tmov\tp2, p1", "\tmova\tr26, #0", "\tnop", "\tnop"]
    if k in (0, 1, 2, 7):
        for i in range(8):
            L += load() + push(flush=(k in (0, 7)), gap=3 if k == 7 else 0)
        if k == 1:
            L += ["\tvst.flush.512.conv\t[p2, sf, r26]"] + ["\tnop"] * 4
        if k == 2:
            L += ["\tvst.flush.512.conv\t[p2, sf, r26]", "\tnop", "\tnop", "\tnop", "\tnop", "\tvst.flush.512.conv\t[p2, sf, r26]"] + ["\tnop"] * 4
    elif k in (3, 4):
        for i in range(4):
            L += load() + push()
        if k == 3:
            L += ["\tmov\tp2, p1", "\tmovxm\tr1, #1024", "\tmov\tm0, r1", "\tnop", "\tpadda\t[p2], m0", "\tmova\tr26, #0", "\tnop", "\tnop", "\tnop"]
        else:
            L += ["\tmova\tr1, #32", "\tmov\tm0, r1", "\tnop", "\tpadda\t[p2], m0", "\tnop", "\tnop", "\tnop"]
        for i in range(4):
            L += load() + push()
    elif k in (5, 6):
        for half in range(2):
            for dm in ("dm1", "dm2", "dm3", "dm4"):
                L += load(dm)[:3]
            L += ["\tnop"] * NOPS
            for j, dm in enumerate(("dm1", "dm2", "dm3", "dm4")):
                L.append(f"\tvst.push.576.conv.bfp16ebs8.fp32\t{dm}, [p2, sf, r26]")
                if k == 5 or j % 2 == 1:
                    L.append("\tvst.flush.512.conv\t[p2, sf, r26]")
            L += ["\tnop"] * 6
    return L + tail(n)


def main():
    out = ["\t.text"]
    for k in range(8):
        out += probe(k)
    open(sys.argv[1], "w").write("\n".join(out) + "\n")
    print("wrote", sys.argv[1])




def probe2d(k, m, dj=0, every=4, n=12):
    """round 12 second set: push + vst.flush.512.conv.2d with d0 = (m, dn = every - 1, dj): the flush's post-modify adds dj
    after flushes 1 .. every-1 of a group and m after the last"""
    nm = f"probe{k}"
    L = head(nm) + ["\tmov\tp2, p1", "\tmova\tr26, #0", f"\tmova\tdn0, #{every - 1}", f"\tmova\tdj0, #{dj}", f"\tmovxm\tm0, #{m}",
                    "\tmova\tdc0, #0", "\tnop", "\tnop"]
    for i in range(n):
        L += load()[:3] + ["\tnop"] * NOPS + ["\tvst.push.576.conv.bfp16ebs8.fp32\tdm1, [p2, sf, r26]",
                                              "\tvst.flush.512.conv.2d\t[p2, sf, r26, d0]"] + ["\tnop"] * 4
    return L + tail(nm)


def main2():
    out = ["\t.text"] + probe2d(0, 32) + probe2d(1, 288) + probe2d(2, 0) + probe2d(3, 64, every=2)
    open(sys.argv[2], "w").write("\n".join(out) + "\n")
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main2() if len(sys.argv) > 2 else main()
