"""Round 11: extract functions from Peano's .s as renamed probes (local labels suffixed), in the dialect the integrated
assembler accepts (p10/asmfix.py's fixes).
    python3 r11/asmx.py IN.s OUT.s gelu_ii16=probe0 gelu_v1=probe1 ..."""
import re
import sys


def fix(line):
    line = re.sub(r"##([A-Za-z_.$][A-Za-z0-9_.$]*)", r"#\1", line)
    return re.sub(r"^(\s*)\.word\b", r"\1.4byte", line)


def extract(src, func):
    """the lines from the function's .section directive to its .size line"""
    lines = src.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"{func}:"))
    while not lines[start].lstrip().startswith(".section"):
        start -= 1
    end = next(i for i in range(start, len(lines)) if lines[i].lstrip().startswith(f".size\t{func},") or lines[i].lstrip().startswith(f".size {func},"))
    return lines[start:end + 1]


def rename(lines, old, new):
    out = []
    for l in lines:
        l = re.sub(rf"\b{re.escape(old)}\b", new, l)
        l = re.sub(r"(\.L[A-Za-z_]*\d+(?:_\d+)?)\b", rf"\1_{new}", l)
        out.append(fix(l))
    return out


def main():
    src = open(sys.argv[1]).read()
    out = []
    for pair in sys.argv[3:]:
        old, new = pair.split("=")
        out += rename(extract(src, old), old, new)
    open(sys.argv[2], "w").write("\t.text\n" + "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
