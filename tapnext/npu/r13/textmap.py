"""Round 13: where an asm kernel's text goes. For each function in an object: bytes before the loop body (preamble +
prologue), the body (<f>_start .. <f>_last inclusive: bundles, how many are the full 16-byte form, what the compact form
would cost), and the tail (drain + ret). Run in npu-iron (needs Peano's llvm-objdump).

    python3 r13/textmap.py build/gup.o [build/pan.o ...]
"""

import re
import subprocess
import sys

B = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin"


def bundles(obj):
    """{section: [(offset, size, text)]} and {symbol: offset}"""
    dis = subprocess.run([f"{B}/llvm-objdump", "-d", obj], capture_output=True, text=True, check=True).stdout
    syms = subprocess.run([f"{B}/llvm-objdump", "-t", obj], capture_output=True, text=True, check=True).stdout
    sym = {}
    for m in re.finditer(r"^([0-9a-f]+)\s+(\w)?\s*\w*\s+(\S+)\s+([0-9a-f]+)\s+(\S+)$", syms, re.M):
        sym[m.group(5)] = (m.group(3), int(m.group(1), 16), int(m.group(4), 16))
    sec, out = None, {}
    for line in dis.splitlines():
        m = re.match(r"Disassembly of section (\S+):", line)
        if m:
            sec = m.group(1)
            out[sec] = []
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)$", line)
        if m and sec:
            out[sec].append((int(m.group(1), 16), len(m.group(2).split()), m.group(3).strip()))
    return out, sym


def main():
    for obj in sys.argv[1:]:
        secs, sym = bundles(obj)
        for name, (sec, off, size) in sorted(sym.items(), key=lambda kv: kv[1][1]):
            if size == 0 or name.startswith(".") or sec not in secs:
                continue
            bl = [b for b in secs[sec] if off <= b[0] < off + size]
            st = sym.get(f"{name}_start")
            la = sym.get(f"{name}_last")
            if st and la:
                s0, l0 = st[1], la[1]
                body = [b for b in bl if s0 <= b[0] <= l0]
                pre = sum(b[1] for b in bl if b[0] < s0)
                tail = off + size - (l0 + 16)
                full = sum(1 for b in body if b[1] == 16)
                print(f"{obj}:{name}: {size} B = before body {pre} + body {len(body)} bundles ({full} full-form) "
                      f"{l0 + 16 - s0} B + tail {tail}")
            else:
                print(f"{obj}:{name}: {size} B, {len(bl)} bundles (no _start / _last labels)")


if __name__ == "__main__":
    main()
