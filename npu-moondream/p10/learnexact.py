"""Minimum producer->consumer bundle distances in Peano's output, matching exact register names only (no aliasing),
for the load/pop/conversion pairs a hand schedule needs. Usage: learnexact.py file.s func..."""
import re, sys
sys.path.insert(0, "/work/p10"); from asmcheck import parse_bundles, split_instr, REG
res = {}
for func in sys.argv[2:]:
    last = {}  # exact reg -> (cycle, mn)
    cyc = 0
    for lineno, instrs in parse_bundles(sys.argv[1], func):
        if instrs and instrs[0] == "label": continue
        cyc += 1
        writes = []
        for ins in instrs:
            mn, ops = split_instr(ins)
            if mn.startswith("nop"): continue
            regs = REG.findall(ops)
            if not regs: continue
            # first register is the destination for these mnemonics; the rest are sources
            dst_first = not (mn.startswith("vst") or mn in ("st", "st.s16", "st.s8", "padda", "jnz", "jz", "jl", "movs") or mn.startswith("vst"))
            dst, srcs = (regs[0], regs[1:]) if dst_first else (None, regs)
            for s in srcs:
                if s in last:
                    c, pm = last[s]
                    key = (pm, mn)
                    if s[0] in "xye" or s.startswith("ex") or s.startswith("dm") or s.startswith("cm") or s.startswith("bm"):
                        d = cyc - c
                        if key not in res or d < res[key][0]: res[key] = (d, s, lineno)
            if dst: writes.append((dst, mn))
        for d, m in writes: last[d] = (cyc, m)
for (p, c), (d, s, l) in sorted(res.items()):
    if any(k in p for k in ("vld", "vconv", "vmac", "vmul", "vshuffle", "vmax", "vadd", "vups", "vsrs", "vbcst", "vshift", "vmov")):
        print(f"{p:34s} -> {c:34s} min distance {d:3d}  (e.g. {s} line {l})")
