#!/usr/bin/env python3
"""Every repo-relative path named in the documentation must exist.

A backlog item pointing at a file that is not there costs whoever follows it the time it takes to
work out whether they are confused or the document is. This repository's documents cross-reference
each other heavily and several of them are generated, so a path can go wrong in the generator and
appear in fourteen files at once -- which is how `docs/CONTRIBUTING.md` got written into an issue
when the file has always been `CONTRIBUTING.md` at the root.

Scope: paths that look like repo paths -- a slash or a known extension, no scheme, no spaces --
whether they appear in a markdown link or in backticks in running prose, which is how most of
them appear here. A trailing directory slash is honoured. Anything with a URL scheme is somebody
else's problem and is skipped.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = sorted(
    p for p in list(ROOT.glob("*.md")) + list(ROOT.rglob("docs/*.md"))
    + list(ROOT.rglob("issues/*.md")) + list(ROOT.rglob("examples/*.md"))
    if ".git" not in p.parts)

# `path/to/thing` in backticks, or the target of a [text](target) markdown link.
BACKTICK = re.compile(r"`([^`\s]+)`")
MDLINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

# Paths that belong to somebody else's repository, as {(document, path): reason}. A checker cannot
# tell "our docs/X.md" from another project's, so an exception is listed here with its reason
# rather than the sentence being bent to suit the tool. An entry is a promise that the path is
# genuinely external.
EXTERNAL = {}

KNOWN_EXT = {".md", ".py", ".sh", ".json", ".cpp", ".cu", ".h", ".txt", ".yml", ".npy"}
SCHEMES = ("http://", "https://", "mailto:", "#")


def looks_like_a_path(tok: str) -> bool:
    """A path CLAIM, not any mention of a file.

    Prose names bare filenames constantly -- "`bench.py` read the whole of /dev/urandom" is about
    a script, not a location, and demanding it be spelled `eval/bench.py` would make the prose
    worse to satisfy a checker. So a token has to contain a slash to be treated as a claim about
    where something lives. That is the form that misleads: a reader follows `docs/THING.md`
    literally and a bare `THING.md` by searching.
    """
    if tok.startswith(SCHEMES) or " " in tok or tok.startswith("-"):
        return False
    # A placeholder, not a path: `eval/cells/<new>/`, `--ledger <dir>`. Documentation has to be
    # able to show the SHAPE of a path without a file of that name existing, and demanding a
    # real example everywhere would make the prose worse to satisfy a checker.
    if "<" in tok or ">" in tok:
        return False
    if "/" not in tok:
        return False
    return tok.endswith("/") or Path(tok).suffix in KNOWN_EXT


def main() -> int:
    bad = []
    for doc in DOCS:
        text = doc.read_text()
        for tok in set(BACKTICK.findall(text)) | set(MDLINK.findall(text)):
            if not looks_like_a_path(tok):
                continue
            if (str(doc.relative_to(ROOT)), tok) in EXTERNAL:
                continue
            target = tok.rstrip("/")
            # A markdown link inside issues/ is relative to that directory.
            # A glob is a claim that something matches it, not that the literal path exists.
            if any(ch in target for ch in "*?["):
                if any(ROOT.glob(target)) or any(doc.parent.glob(target)):
                    continue
            elif any((base / target).exists() for base in (ROOT, doc.parent)):
                continue
            bad.append((doc.relative_to(ROOT), tok))
    if bad:
        print("!! documentation points at paths that do not exist:")
        for doc, tok in sorted(bad):
            print(f"   {doc}: {tok}")
        print("\n   If the document is generated (anything under issues/, or")
        print("   docs/ROOFLINE.md), fix the generator rather than the file it wrote -- a")
        print("   generator repeats a wrong path everywhere at once. A placeholder path is")
        print("   spelled with angle brackets and is skipped.")
        return 1
    print(f"ok: every repo path named across {len(DOCS)} documents exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
