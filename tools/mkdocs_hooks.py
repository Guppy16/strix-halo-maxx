"""MkDocs hook: make links to repo files that are not pages work.

The book links to code (.py, .sh, .cc), data and directories. With
``repo_url`` set (in mkdocs.yml) they become GitHub URLs; until then they stay
relative, and links to directories (which MkDocs cannot resolve) fall back to
the directory's README.md when it has one, else are left as they are (INFO, not a warning).
"""

import os
import re
from urllib.parse import unquote

# [text](target) and ![alt](target); targets without spaces or nested parens
LINK = re.compile(r"(\]\()([^)\s]+)(\))")
SKIP_PREFIX = ("http:", "https:", "mailto:", "#", "/")

_repo_url = None
_branch = "main"


def on_config(config):
    global _repo_url
    _repo_url = (config.get("repo_url") or "").rstrip("/") or None
    return config


def on_page_markdown(markdown, page, config, files):
    docs_dir = config["docs_dir"]
    src_dir = os.path.dirname(page.file.src_path)

    def fix(m):
        target = m.group(2)
        if target.startswith(SKIP_PREFIX):
            return m.group(0)
        path, sep, frag = target.partition("#")
        if not path or path.endswith(".md"):
            return m.group(0)
        rel = os.path.normpath(os.path.join(src_dir, unquote(path)))
        full = os.path.join(docs_dir, rel)
        is_dir = os.path.isdir(full)
        if not is_dir and not os.path.exists(full):
            return m.group(0)  # genuinely missing: let strict mode report it
        if _repo_url:
            kind = "tree" if is_dir else "blob"
            return f"{m.group(1)}{_repo_url}/{kind}/{_branch}/{rel}{m.group(3)}"
        if is_dir:
            readme = os.path.join(rel, "README.md")
            if os.path.exists(os.path.join(docs_dir, readme)):
                new = os.path.relpath(readme, src_dir or ".")
                return f"{m.group(1)}{new}{m.group(3)}"
            return m.group(0)  # MkDocs logs it at INFO and keeps the relative link
        return m.group(0)

    return LINK.sub(fix, markdown)
