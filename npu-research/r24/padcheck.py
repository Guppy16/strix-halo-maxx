"""pass 19: assert that a (modified) generated kernel has no .p2align pad bundle between its prologue and its loop body (a pad
shifts the prologue a cycle against the body: rows 0-7 lost, SKILL round 21). usage (npu-iron): python3 r24/padcheck.py X.s FUNC"""
import re
import subprocess
import sys
from pathlib import Path

P = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin"
s, f = sys.argv[1], sys.argv[2]
o = Path(s).with_suffix(".pc.o")
subprocess.run([f"{P}/clang", "--target=aie2p-none-unknown-elf", "-c", s, "-o", str(o)], check=True)
dis = subprocess.run([f"{P}/llvm-objdump", "-d", "-z", "--no-show-raw-insn", str(o)], capture_output=True, text=True).stdout
syms = subprocess.run([f"{P}/llvm-objdump", "-t", str(o)], capture_output=True, text=True).stdout
addr = {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+)\s.*\s(\S+)$", syms, re.M)}
offs = [int(m.group(1), 16) for m in re.finditer(r"^\s*([0-9a-f]+):", dis, re.M)]
text = open(s).read().splitlines()
a, b = text.index(f".L{f}_pro:"), text.index(f".L{f}_body:")
npro = sum(1 for x in text[a + 1:b] if x.strip() and not x.strip().startswith(".") and not x.strip().endswith(":"))
start = addr[f"{f}_start"]
i = offs.index(start)
pro_first = offs[i - npro]
# the prologue label is 16 B aligned: its first bundle must be the bundle npro before the body
ok = pro_first % 16 == 0 and addr[f"{f}_last"] % 16 == 0
print(f"{s}: prologue {npro} bundles, first at {pro_first:#x} ({'aligned' if pro_first % 16 == 0 else 'NOT aligned: a pad sits inside'}),"
      f" body {start:#x}, last {addr[f'{f}_last']:#x}: {'OK' if ok else 'BAD'}")
o.unlink()
sys.exit(0 if ok else 1)
