"""Rewrite Peano's emitted AIE2P assembly into the dialect its integrated assembler accepts (the ##sym immediates,
event #n, .word), so a .s can be edited and reassembled. Usage: asmfix.py in.s out.s"""
import re, sys
src = open(sys.argv[1]).read().split("\n")
out = []
for line in src:
    l = line
    l = re.sub(r"##([A-Za-z_.$][A-Za-z0-9_.$]*)", r"#\1", l)      # movxm p2, ##sym -> #sym
    m = re.search(r"\bevent\s+#([01])\b", l)
    if m:  # the integrated assembler cannot parse `event #n`: split it out of its bundle as the raw standalone encoding
        parts = [p for p in l.split(";") if "event" not in p]
        out.append("\t.byte 0x18, 0x00, 0x10, 0x1" + ("0" if m.group(1) == "0" else "2") + "   // event #" + m.group(1))
        l = ";".join(parts) if parts else "\tnop"
    l = re.sub(r"^(\s*)\.word\b", r"\1.4byte", l)                  # .word -> .4byte
    out.append(l)
open(sys.argv[2], "w").write("\n".join(out))
