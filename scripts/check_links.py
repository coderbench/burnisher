#!/usr/bin/env python3
"""Every repo-relative path named in the documentation must exist, and every link must land.

A backlog item pointing at a file that is not there costs whoever follows it the time it takes to
work out whether they are confused or the document is. This repository's documents cross-reference
each other heavily and several of them are generated, so a path can go wrong in the generator and
appear in fourteen files at once -- which is how `docs/CONTRIBUTING.md` got written into an issue
when the file has always been `CONTRIBUTING.md` at the root.

Scope:
- Paths in backticks that look like repo paths -- a slash and a known extension, no scheme, no
  spaces -- must exist, relative to the root or to the document.
- Markdown links `[text](target)` must resolve the way GitHub resolves them: relative to the
  document's own directory, never the root. A link that only works from the root is a dead link
  on GitHub.
- A `#fragment`, on its own or after a markdown file, must name a heading in that file.
Anything with a URL scheme is somebody else's problem and is skipped.
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
FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")

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


def prose(text: str) -> str:
    """The document without its fenced code blocks, where brackets are code, not links."""
    out, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: link text kept, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "").lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(doc: Path) -> set[str]:
    found, seen = set(), {}
    for line in prose(doc.read_text()).splitlines():
        m = HEADING.match(line)
        if not m:
            continue
        s = slug(m.group(1))
        n = seen.get(s, 0)
        seen[s] = n + 1
        found.add(s if n == 0 else f"{s}-{n}")
    return found


def broken_link(doc: Path, link: str) -> bool:
    if link.startswith(("http://", "https://", "mailto:")):
        return False
    target, _, fragment = link.partition("#")
    if "<" in target or ">" in target:
        return False
    path = doc.parent / target if target else doc
    if not path.exists():
        return True
    return bool(fragment) and path.suffix == ".md" and fragment not in anchors(path)


def main() -> int:
    bad = []
    for doc in DOCS:
        text = doc.read_text()
        name = str(doc.relative_to(ROOT))
        for tok in set(BACKTICK.findall(text)):
            if not looks_like_a_path(tok) or (name, tok) in EXTERNAL:
                continue
            target = tok.rstrip("/")
            # A glob is a claim that something matches it, not that the literal path exists.
            if any(ch in target for ch in "*?["):
                if any(ROOT.glob(target)) or any(doc.parent.glob(target)):
                    continue
            elif any((base / target).exists() for base in (ROOT, doc.parent)):
                continue
            bad.append((doc.relative_to(ROOT), tok))
        for link in set(MDLINK.findall(prose(text))):
            if (name, link) not in EXTERNAL and broken_link(doc, link):
                bad.append((doc.relative_to(ROOT), link))
    if bad:
        print("!! documentation points at paths or headings that do not exist:")
        for doc, tok in sorted(bad):
            print(f"   {doc}: {tok}")
        print("\n   A markdown link is relative to the document's own directory, and a #fragment")
        print("   must match a heading. If the document is generated (anything under issues/,")
        print("   or docs/ROOFLINE.md), fix the generator rather than the file it wrote -- a")
        print("   generator repeats a wrong path everywhere at once. A placeholder path is")
        print("   spelled with angle brackets and is skipped.")
        return 1
    print(f"ok: every repo path, link and anchor across {len(DOCS)} documents exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
