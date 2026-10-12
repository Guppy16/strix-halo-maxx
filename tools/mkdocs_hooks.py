"""MkDocs hooks: render the repo's GitHub-flavoured Markdown faithfully on the site.

The Markdown is written for GitHub, which is lenient where Python-Markdown is strict. Before parsing, each page gets:

1. a blank line before a list or table that directly follows a paragraph (otherwise it renders as running text);
2. nested-list indentation doubled where a list nests by 2 spaces (Python-Markdown needs 4);
3. links to repo files that are not site pages (code, data, directories, and Markdown kept out of the site such as
   the skills and CLAUDE.md) rewritten to their GitHub URL (``repo_url`` in mkdocs.yml).

The source files are not changed, so GitHub keeps rendering them as before.
"""

import os
import re
from urllib.parse import unquote

LINK = re.compile(r"(\]\()([^)\s]+)(\))")  # [text](target) and ![alt](target)
SKIP_PREFIX = ("http:", "https:", "mailto:", "#", "/")
FENCE = re.compile(r"^\s*(```|~~~)")
ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s")
TABLE = re.compile(r"^\s*\|")
HEADING = re.compile(r"^\s{0,3}#")

_repo_url = None
_branch = "main"


def on_config(config):
    global _repo_url
    _repo_url = (config.get("repo_url") or "").rstrip("/") or None
    return config


def _blank_before_blocks(lines):
    out, fence = [], False
    for line in lines:
        if FENCE.match(line):
            fence = not fence
        if not fence and out and (ITEM.match(line) or TABLE.match(line)):
            prev = out[-1]
            prev_is_text = (prev.strip() and not ITEM.match(prev) and not TABLE.match(prev)
                            and not HEADING.match(prev) and not prev.startswith((" ", "\t")))
            if prev_is_text:
                out.append("")
        out.append(line)
    return out


def _double_list_indent(lines):
    """In each list block whose nesting step is 2 spaces, double the indentation of its lines."""
    out, block = [], []

    def flush():
        indents = [len(m.group(1)) for m in (ITEM.match(b) for b in block) if m and m.group(1)]
        if indents and min(indents) == 2:
            for b in block:
                n = len(b) - len(b.lstrip(" "))
                out.append(" " * (2 * n) + b[n:] if b.strip() else b)
        else:
            out.extend(block)
        block.clear()

    fence = False
    for line in lines:
        starts_item = bool(ITEM.match(line)) and not line.startswith(" ")
        in_block = bool(block)
        if FENCE.match(line):
            fence = not fence
        if not fence and starts_item and not in_block:
            block.append(line)
            continue
        if in_block:
            if fence or line.strip() == "" or line.startswith((" ", "\t")) or ITEM.match(line):
                block.append(line)
                continue
            flush()
        out.append(line)
    if block:
        flush()
    return out


def on_page_markdown(markdown, page, config, files):
    lines = markdown.split("\n")
    markdown = "\n".join(_double_list_indent(_blank_before_blocks(lines)))

    docs_dir = config["docs_dir"]
    src_dir = os.path.dirname(page.file.src_path)

    def fix(m):
        target = m.group(2)
        if target.startswith(SKIP_PREFIX):
            return m.group(0)
        path, _, frag = target.partition("#")
        if not path:
            return m.group(0)
        rel = os.path.normpath(os.path.join(src_dir, unquote(path)))
        full = os.path.join(docs_dir, rel)
        is_dir = os.path.isdir(full)
        if not is_dir and not os.path.exists(full):
            return m.group(0)  # genuinely missing: strict mode reports it
        if path.endswith(".md"):
            f = files.get_file_from_path(rel)
            if f is not None and f.inclusion.is_included():
                return m.group(0)  # a site page: MkDocs resolves it
        if not _repo_url:
            return m.group(0)
        kind = "tree" if is_dir else "blob"
        anchor = f"#{frag}" if frag else ""
        return f"{m.group(1)}{_repo_url}/{kind}/{_branch}/{rel}{anchor}{m.group(3)}"

    return LINK.sub(fix, markdown)
