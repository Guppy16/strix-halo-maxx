"""Round 11 (research P42): stack audit of a build's core elfs. Per core: the deepest call path (llvm-readelf --stack-sizes
+ the call graph from llvm-objdump's jl targets), against the stack the linker script reserves. Exits 1 when a core's
path + MARGIN exceeds its reservation (bb.sh runs it after every build).
    python3 r11/stackaudit.py build/NAME.prj [--all]     (npu-iron: needs Peano's llvm tools)"""

import re
import subprocess
import sys
from pathlib import Path

B = "/opt/ironenv/lib/python3.13/site-packages/llvm-aie/bin"
MARGIN = 128
ASM_FRAMES = {"gelu_ms": 256, "gup": 0, "cpy32": 0, "pan": 0, "bgelu": 512, "sigk": 0, "mm_ks": 0, "mm_core_p": 0, "__start": 0, "__modsi3": 0, "__divsi3": 0, "__udivsi3": 0}  # hand-asm functions (no .stack_sizes entry): their own frames


def audit(elf, script):
    sizes = {}
    for line in subprocess.run([f"{B}/llvm-readelf", "--stack-sizes", elf], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+(.+)$", line)
        if m:
            for nm in m.group(2).split(", "):
                sizes[nm.strip()] = int(m.group(1))
    dis = subprocess.run([f"{B}/llvm-objdump", "-d", elf], capture_output=True, text=True).stdout
    addr2fn, calls, cur = {}, {}, None
    for line in dis.splitlines():
        m = re.match(r"([0-9a-f]+) <([^>]+)>:", line)
        if m:
            if not m.group(2).startswith(".L"):
                cur = m.group(2)
                addr2fn[int(m.group(1), 16)] = cur
                calls.setdefault(cur, set())
            continue
        for t in re.findall(r"\bjl\s+#0x([0-9a-f]+)", line):
            calls[cur].add(int(t, 16))
        for t in re.findall(r"\bj\s+#0x([0-9a-f]+)", line):  # a jump to another function's start = a tail call (counted
            calls[cur].add(-int(t, 16) - 1)  # as a call: conservative, the caller's frame is already gone)
    calls = {f: {addr2fn.get(a, f"?{a:x}") if a >= 0 else addr2fn.get(-a - 1) for a in ts} - {None, f} for f, ts in calls.items()}
    unknown = set()

    def frame(f):
        if f in sizes:
            return sizes[f]
        if f in ASM_FRAMES:
            return ASM_FRAMES[f]
        unknown.add(f)
        return 0

    def deepest(f, seen=()):
        if f in seen:
            raise ValueError(f"recursion through {f}")
        sub = [deepest(g, seen + (f,)) for g in calls.get(f, ())]
        best = max(sub, key=lambda s: s[0], default=(0, []))
        return frame(f) + best[0], [f] + best[1]

    depth, path = deepest("__start")
    txt = Path(script).read_text()
    reserved = int(re.search(r"\.\s*\+=\s*(0x[0-9a-fA-F]+|\d+);\s*/\*\s*stack\s*\*/", txt).group(1), 0)
    return depth, path, reserved, unknown


def main():
    prj = Path(sys.argv[1])
    bad = 0
    seen = {}
    for elf in sorted(prj.glob("elfs_main_core_*/elfs_main_core_*.elf")):
        core = elf.stem.replace("elfs_main_core_", "")
        depth, path, reserved, unknown = audit(str(elf), prj / f"ldScripts_main_core_{core}.ld.script")
        key = (depth, reserved, tuple(path))
        ok = depth + MARGIN <= reserved
        bad += not ok
        if key in seen and "--all" not in sys.argv:
            seen[key].append(core)
            continue
        seen[key] = [core]
        names = [re.sub(r"^_ZL\d+|^_ZN\w*?\d+(?=[a-z_])", "", p)[:28] for p in path]
        print(f"core {core}: deepest {depth} B + {MARGIN} vs stack {reserved} B ({reserved - depth} free) {'OK' if ok else 'OVERFLOW'}"
              f"  path {' > '.join(names)}" + (f"  (no size for {sorted(unknown)})" if unknown else ""))
    for (depth, reserved, _), cores in seen.items():
        if len(cores) > 1:
            print(f"  same as above: {len(cores)} cores ({', '.join(cores[:6])}{'...' if len(cores) > 6 else ''}) at {depth} / {reserved}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
