#!/usr/bin/env python3
"""Check the book for rot. Run from the repo root: python3 common/tools/bookcheck.py

Checks (exit 1 on any problem):
  - every relative link and #anchor in the book's .md files resolves (rNN/ folders are skipped);
  - every `<!-- nr: ID -->` marker names an id in common/book/negative-results.md;
  - every index row has a valid class, a body entry with the same class, and a chapter link
    (or a chapter cell starting with "none:" that gives the reason);
  - the class counts in the index header match the rows.
Standard library only.
"""

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
INDEX = ROOT / "common/book/negative-results.md"
CLASSES = ("fundamental", "measured-here", "unfinished", "revived", "method")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "mlir-aie"}  # mlir-aie: vendored upstream docs

problems: list[str] = []


def bad(where, msg):
    problems.append(f"{where}: {msg}")


def md_files():
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS or re.fullmatch(r"r\d+", part) for part in rel.parts[:-1]):
            continue
        yield p


def strip_code(text):
    """Blank out fenced blocks and inline code so example links are not checked."""
    text = re.sub(r"^(```|~~~).*?^\1", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", text)


def slug(heading):
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    h = re.sub(r"[`*]|<[^>]+>", "", h).strip().lower()
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


_anchor_cache: dict[Path, set[str]] = {}


def anchors(path):
    if path not in _anchor_cache:
        text = path.read_text(errors="replace")
        found, seen = set(), Counter()
        for m in re.finditer(r"^#{1,6}\s+(.*?)\s*#*\s*$", strip_code(text), flags=re.M):
            s = slug(m.group(1))
            found.add(s if not seen[s] else f"{s}-{seen[s]}")
            seen[s] += 1
        found.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
        _anchor_cache[path] = found
    return _anchor_cache[path]


LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def check_links(path):
    text = strip_code(path.read_text(errors="replace"))
    rel = path.relative_to(ROOT)
    for m in LINK.finditer(text):
        target = m.group(1)
        line = text.count("\n", 0, m.start()) + 1
        if re.match(r"[a-z][a-z0-9+.-]*:", target, re.I):
            continue
        file_part, _, frag = target.partition("#")
        dest = path if not file_part else (path.parent / file_part).resolve()
        if not dest.exists():
            bad(f"{rel}:{line}", f"missing file {target}")
            continue
        if frag and dest.is_file() and dest.suffix == ".md" and frag not in anchors(dest):
            if not re.fullmatch(r"L\d+(-L?\d+)?", frag):
                bad(f"{rel}:{line}", f"missing anchor {target}")


def check_index():
    text = INDEX.read_text()
    rows = {}
    for m in re.finditer(r"^\| \[(\w+)\]\(#(\w+)\) \| (.*?) \| (\w+) \| (.*?) \| (.*?) \| (.*?)\s*\|\s*$", text, flags=re.M):
        rid, anchor, _name, _proj, cls, _why, chap = m.groups()
        if anchor != rid.lower():
            bad("index", f"{rid}: link #{anchor} is not #{rid.lower()}")
        if rid in rows:
            bad("index", f"{rid}: duplicate row")
        if cls not in CLASSES:
            bad("index", f"{rid}: class '{cls}' is not one of {CLASSES}")
        if not re.search(r"\]\([^)]+\)", chap) and not chap.strip().lower().startswith("none:"):
            bad("index", f"{rid}: no chapter link and no 'none: reason'")
        rows[rid] = cls
    # every table line should have parsed
    n_table = len(re.findall(r"^\| \[\w+\]\(", text, flags=re.M))
    if n_table != len(rows):
        bad("index", f"{n_table} table rows but {len(rows)} parsed")
    # body entries
    for rid, cls in rows.items():
        m = re.search(rf'<a id="{rid.lower()}"></a>\n(.*?)(?=\n<a id="|\n## |\Z)', text, flags=re.S)
        if not m:
            bad("index", f"{rid}: no body entry")
            continue
        c = re.search(r"^\*\*Class:\*\* ([\w-]+)", m.group(1), flags=re.M)
        if not c:
            bad("index", f"{rid}: body has no **Class:** line")
        elif c.group(1) != cls:
            bad("index", f"{rid}: body class {c.group(1)} != row class {cls}")
        elif not re.search(r"Would have to change|What would settle it", m.group(1)) and cls in ("measured-here", "unfinished"):
            bad("index", f"{rid}: {cls} body lacks 'Would have to change' / 'What would settle it'")
    # header counts
    h = re.search(r"Counts \((\d+)\): (.*)", text)
    if not h:
        bad("index", "no 'Counts (N): ...' header line")
    else:
        stated = {k: int(v) for k, v in re.findall(r"([\w-]+) (\d+)", h.group(2))}
        actual = Counter(rows.values())
        if int(h.group(1)) != len(rows):
            bad("index", f"header says {h.group(1)} entries, table has {len(rows)}")
        for c in CLASSES:
            if stated.get(c, 0) != actual.get(c, 0):
                bad("index", f"header says {c} {stated.get(c, 0)}, table has {actual.get(c, 0)}")
    for m in re.finditer(r"(\d+) entries", text):
        if int(m.group(1)) != len(rows):
            bad("index", f"text says '{m.group(0)}', table has {len(rows)}")
    return rows


def main():
    rows = check_index()
    files = list(md_files())
    for p in files:
        check_links(p)
        for m in re.finditer(r"<!--\s*nr:\s*([^>]*?)\s*-->", strip_code(p.read_text(errors="replace"))):
            for rid in re.split(r"[,\s]+", m.group(1)):
                if rid and rid not in rows:
                    bad(str(p.relative_to(ROOT)), f"nr marker {rid} is not in the index")
    for p in problems:
        print(p)
    print(f"bookcheck: {len(files)} files, {len(rows)} index rows, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
