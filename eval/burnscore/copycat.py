"""Is this submission somebody else's work?

A copy earns what its original earned, and a guard against it is only worth having if it is
right: blocking the author of the original for being copied is worse than no guard at all.

Comparing the literal added lines of one pull request with another's has four blind spots, and
each one shaped a decision here:

* **Renaming and reformatting defeat literal lines.** Here code is compared as a stream of
  normalized tokens -- identifiers, numbers and strings collapsed, comments dropped -- hashed as
  overlapping k-grams and winnowed, so a renamed, re-indented copy has the same fingerprint.
* **Merged work needs its own question.** The references here are the pull requests open now,
  as the repository decided. A kernel already on main registered again under a new name is the
  copy this runtime would pay twice, because the base arm never runs it, so it has its own guard
  (`eval/burnscore/reregistration.py`). Code on main that no submission put there -- the baseline
  kernel a contributor is told to copy and register beside the old one -- is never evidence.
* **Shared code is not evidence.** Launch boilerplate, a helper already on main, a three-line
  build fix everyone needs -- each looks like a copy line by line. Here a fingerprint
  counts only if it is NEW: not in the diff's own context, not on main, and not common to several
  submissions. And a verdict needs enough new code to mean anything; a tiny identical change
  goes to review, never to a copy verdict.
* **The lower pull-request number is not the original.** A pull request opened early and
  force-pushed with copied code later looks like the original. Here the original is whoever the
  evaluator OBSERVED first, from a record the submission cannot write to.

A copy embedded inside a larger submission, and a partial match, are REVIEW: measured, but not
paid until a maintainer looks. A branch STACKED on somebody else's open pull request -- its head
commit is in this branch's history -- carries that pull request's code in its diff against main
by construction, so it is never a copy of it: it is REVIEW, because the measurement includes
work that is not the author's.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

K = 16                  # tokens per k-gram: long enough that two loops do not match by accident
W = 4                   # winnowing window
MIN_MASS = 20           # new-code fingerprints a submission needs before it can be called a copy
T_COPY = 0.70           # share of a submission's new code found in one earlier submission
T_REVIEW = 0.40
TINY_MIN = 2            # a smaller identical change than this is not even worth a review
EMBED_MIN = 60          # shared fingerprints for "contains most of an earlier submission's work"
BOILERPLATE_DF = 3      # a fingerprint in more submissions than this is shared infrastructure

CODE_EXT = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".cu", ".cuh", ".py", ".sh",
            ".cmake")
_HASH_COMMENTS = (".py", ".sh", ".cmake")

KEYWORDS = frozenset("""
if else for while do switch case default break continue return goto struct class enum union typedef
using namespace template typename const constexpr static inline extern volatile auto void int float
double char bool long short unsigned signed size_t true false nullptr new delete sizeof operator
public private protected virtual override final this def elif in not and or is None True False
import from as with try except finally raise yield lambda pass global nonlocal assert del async await
""".split())

_TOKEN = re.compile(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(\d[\w.]*)|([A-Za-z_]\w*)|'
                    r'(<<=|>>=|<<<|>>>|->|::|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||'
                    r'[-+*/%&|^!~<>=?:;,.(){}\[\]#])')


def is_code(path: str) -> bool:
    return path.endswith(CODE_EXT) or path.endswith("CMakeLists.txt")


def parse_diff(text: str) -> dict:
    """{path: {"added": [[line, ...] per contiguous run], "base": [line, ...]}}, code files only.

    `base` is every context and removed line: code that already existed where the change was
    made. Moving code shows up as removed-then-added, so it is never mistaken for duplication.
    """
    files, cur, run = {}, None, []

    def flush():
        nonlocal run
        if cur is not None and run:
            files[cur]["added"].append(run)
        run = []

    for line in text.splitlines():
        if line.startswith("diff --git "):
            flush(); cur = None
        elif line.startswith("+++ "):
            flush()
            p = line[4:].strip()
            p = p[2:] if p.startswith("b/") else p
            cur = p if p != "/dev/null" and is_code(p) else None
            if cur is not None:
                files.setdefault(cur, {"added": [], "base": []})
        elif cur is None or line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            flush()
        elif line.startswith("+"):
            run.append(line[1:])
        elif line.startswith("-") or line.startswith(" "):
            flush()
            files[cur]["base"].append(line[1:])
    flush()
    return files


def strip_comments(lines, path):
    out, in_block = [], False
    hash_comments = path.endswith(_HASH_COMMENTS) or path.endswith("CMakeLists.txt")
    for s in lines:
        if in_block:
            end = s.find("*/")
            if end < 0:
                out.append(""); continue
            s, in_block = s[end + 2:], False
        while "/*" in s:
            a = s.find("/*"); b = s.find("*/", a + 2)
            if b < 0:
                s, in_block = s[:a], True
                break
            s = s[:a] + " " + s[b + 2:]
        if "//" in s and not path.endswith(".py"):
            s = s[:s.find("//")]
        if hash_comments and s.lstrip().startswith("#"):
            s = ""
        out.append(s)
    return out


def canonical_tokens(lines, path):
    """Tokens with every identifier, number and string collapsed, and the line each came from."""
    toks, owners = [], []
    for li, line in enumerate(strip_comments(lines, path)):
        found = list(_TOKEN.finditer(line))
        for m in found:
            string, number, ident, op = m.groups()
            if string is not None:
                toks.append("S")
            elif number is not None:
                toks.append("N")
            elif ident is not None:
                toks.append(ident if ident in KEYWORDS or ident.startswith("__") else "I")
            else:
                toks.append(op)
            owners.append(li)
    return toks, owners


def _kgrams(tokens):
    return [hashlib.blake2b(" ".join(tokens[i:i + K]).encode(), digest_size=8).hexdigest()
            for i in range(len(tokens) - K + 1)]


def _winnow(hashes):
    if len(hashes) <= W:
        return set(hashes)
    return {min(hashes[i:i + W]) for i in range(len(hashes) - W + 1)}


def fingerprint_diff(text: str) -> dict:
    """What a submission adds (winnowed) and what it touches (every k-gram of its context)."""
    added, base = set(), set()
    files = parse_diff(text)
    for path, f in files.items():
        for run in f["added"]:
            added |= _winnow(_kgrams(canonical_tokens(run, path)[0]))
        base |= set(_kgrams(canonical_tokens(f["base"], path)[0]))
    return {"added": added, "base": base, "paths": sorted(files)}


def fingerprint_sources(files: dict) -> set:
    """Every k-gram of a tree's source files: what is already on main, anywhere."""
    out = set()
    for path, text in files.items():
        if is_code(path):
            out |= set(_kgrams(canonical_tokens(text.splitlines(), path)[0]))
    return out


def boilerplate(added_sets, limit=BOILERPLATE_DF) -> set:
    """Fingerprints that more than `limit` submissions add: shared infrastructure, not evidence."""
    df = Counter(h for s in added_sets for h in s)
    return {h for h, n in df.items() if n > limit}


def judge(candidate: dict, references: list, *, on_main=frozenset(), boiler=frozenset(),
          stacked=frozenset()) -> dict:
    """COPY, REVIEW or CLEAR for one submission.

    `candidate` and each reference: {"pr", "author", "first_seen", "added", "base"}. Only
    references by a different author that the evaluator observed EARLIER are considered: a
    self-resubmission is iteration, and a later submission cannot be the original. `stacked` names
    the references whose head commit is in the candidate's history; matching one of those is
    REVIEW, never COPY.
    """
    me = (candidate["author"] or "").lower()
    refs = [r for r in references
            if (r["author"] or "").lower() != me and r["pr"] != candidate["pr"]
            and r["first_seen"] < candidate["first_seen"]]

    # Code on main is not evidence -- unless an earlier submission put it there. A submission has
    # landed when most of its new code is now on main, and what it added stays attributable to
    # it. Everything else on main is the baseline every contributor starts from: CONTRIBUTING.md
    # tells them to copy the old kernel and register the new one beside it, so treating that as
    # copying would flag the recommended workflow.
    landed = set()
    for r in refs:
        theirs = r["added"] - boiler - r["base"]
        if theirs and len(theirs & on_main) / len(theirs) >= T_COPY:
            landed |= theirs
    infra = on_main - landed

    novel = candidate["added"] - boiler - candidate["base"] - infra
    clear = {"outcome": "CLEAR", "kind": None, "original": None, "new_code": len(novel),
             "shared": set()}
    if not novel:
        return clear

    slots = {"copy": [], "contains-earlier": [], "stacked": [], "partial": [],
             "tiny-identical": []}
    for ref in refs:
        mine = novel - ref["base"]
        theirs = ref["added"] - boiler - ref["base"] - candidate["base"] - infra
        shared = mine & ref["added"]
        if not mine or not shared:
            continue
        a = len(shared) / len(mine)
        b = len(theirs & candidate["added"]) / len(theirs) if theirs else 0.0
        row = {"ref": ref, "containment": a, "contains": b, "new_code": len(mine),
               "shared": shared}
        if ref["pr"] in stacked:
            if ((len(mine) >= MIN_MASS and a >= T_REVIEW)
                    or (len(theirs) >= EMBED_MIN and b >= T_COPY)
                    or (TINY_MIN <= len(mine) < MIN_MASS and a >= 0.99)):
                slots["stacked"].append(row)
        elif len(mine) >= MIN_MASS and a >= T_COPY:
            slots["copy"].append(row)
        elif len(theirs) >= EMBED_MIN and b >= T_COPY and len(shared) >= EMBED_MIN:
            slots["contains-earlier"].append(row)
        elif len(mine) >= MIN_MASS and a >= T_REVIEW:
            slots["partial"].append(row)
        elif TINY_MIN <= len(mine) < MIN_MASS and a >= 0.99:
            slots["tiny-identical"].append(row)

    for kind in ("copy", "contains-earlier", "stacked", "partial", "tiny-identical"):
        rows = slots[kind]
        if not rows:
            continue
        key = "contains" if kind in ("contains-earlier", "stacked") else "containment"
        top = max(r[key] for r in rows)
        # The original is whoever was observed FIRST among the references that match as well as
        # the best one does -- not whichever pull request happens to have the lower number.
        best = min((r for r in rows if r[key] >= top - 0.05), key=lambda r: r["ref"]["first_seen"])
        ref = best["ref"]
        reason = {"copy": f"{best['containment']:.0%} of its new code is in #{ref['pr']}",
                  "contains-earlier": f"it contains {best['contains']:.0%} of #{ref['pr']}'s new code",
                  "stacked": f"it is built on #{ref['pr']}'s unmerged commits, so its diff "
                             f"carries {best['contains']:.0%} of #{ref['pr']}'s new code",
                  "partial": f"{best['containment']:.0%} of its new code is in #{ref['pr']}",
                  "tiny-identical": f"a small change identical to #{ref['pr']}"}[kind]
        return {"outcome": "COPY" if kind == "copy" else "REVIEW", "kind": kind,
                "original": {"pr": ref["pr"], "author": ref["author"],
                             "first_seen": ref["first_seen"]},
                "containment": best["containment"], "contains": best["contains"],
                "new_code": best["new_code"], "shared": best["shared"], "reason": reason}
    return clear


def evidence(diff_text: str, shared: set, limit=12) -> list:
    """The submission's added lines that the shared fingerprints come from, to quote as evidence."""
    hits = []
    for path, f in parse_diff(diff_text).items():
        for run in f["added"]:
            toks, owners = canonical_tokens(run, path)
            lines = set()
            for i, h in enumerate(_kgrams(toks)):
                if h in shared:
                    lines.update(owners[i:i + K])
            hits += [{"path": path, "line": run[li].strip()} for li in sorted(lines)
                     if run[li].strip()]
            if len(hits) >= limit:
                return hits[:limit]
    return hits
