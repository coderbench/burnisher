#!/usr/bin/env python3
"""Is this submission a copy of another pull request that is open right now?

    scripts/copycat_guard.py --repo <worktree> --base origin/main --pr 184 --author alice \
        --open-prs 180,181,184 --corpus <ledger>/copycat --json verdict.json
    scripts/copycat_guard.py --corpus <ledger>/copycat --unblock alice --reason "independent work"

Answered before any GPU time is spent. Every run RECORDS what it observed -- the pull request, its
head commit, the time the evaluator first saw that head, and the fingerprints of its new code --
in an append-only corpus under the ledger, outside any submission's reach. That record decides
who had the code first: a pull request opened early and force-pushed later with copied code gets
the time its copied head was first seen, not the time the pull request was opened.

The references are the pull requests OPEN now, by a different author, observed earlier. A copy
of a merged kernel is a different question with its own guard (`scripts/reregistration_guard.py`).

A branch stacked on another open pull request -- that pull request's head commit is in its
history -- is never a copy of it; matching it is REVIEW.

A COPY verdict BLOCKS the author: the block is appended to `blocked.jsonl` in the corpus, and every
later submission from that author comes back BLOCKED without being judged. Maintainers -- the
owners .github/CODEOWNERS names, passed as --maintainers -- are exempt. A block is lifted only by
an explicit, recorded --unblock. See `eval/burnscore/copycat.py` for how a copy is decided.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import copycat as CC  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout


def submission_diff(repo, base):
    r = subprocess.run(["git", "-C", str(repo), "diff", f"{base}...HEAD"], capture_output=True,
                       text=True)
    return r.stdout if r.returncode == 0 else git(repo, "diff", base)


def main_sources(repo, base):
    files = {}
    for path in git(repo, "ls-tree", "-r", "--name-only", base).splitlines():
        if CC.is_code(path):
            r = subprocess.run(["git", "-C", str(repo), "show", f"{base}:{path}"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                files[path] = r.stdout
    return files


def stacked_on(repo, entries) -> set:
    """Open pull requests whose recorded head commit is in this branch's history.

    Git decides, not the code: a branch built on somebody's unmerged pull request contains that
    pull request's commits, and a copy does not. A head this checkout has never fetched is not
    evidence either way, so it counts as not stacked.
    """
    out = set()
    for e in entries:
        if e["pr"] in out:
            continue
        r = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", e["head"], "HEAD"],
                           capture_output=True)
        if r.returncode == 0:
            out.add(e["pr"])
    return out


def load_corpus(corpus: Path) -> list:
    out = []
    for p in sorted((corpus / "entries").glob("*.json")) if corpus.is_dir() else []:
        d = json.loads(p.read_text())
        d["added"], d["base"] = set(d["added"]), set(d["base"])
        out.append(d)
    return out


def record(corpus: Path, entry: dict) -> None:
    """Write this observation once. A head already on record keeps its original first-seen time."""
    path = corpus / "entries" / f"pr-{entry['pr']:06d}-{entry['head'][:12]}.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(entry, added=sorted(entry["added"]), base=sorted(entry["base"])),
                               sort_keys=True) + "\n")


def block_log(corpus: Path) -> list:
    p = corpus / "blocked.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def current_block(corpus: Path, author: str):
    """The block in force for this author, or None. The latest action for an author decides."""
    last = None
    for rec in block_log(corpus):
        if rec["author"].lower() == author.lower():
            last = rec
    return last if last and last["action"] == "block" else None


def append_block(corpus: Path, rec: dict) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    with (corpus / "blocked.jsonl").open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="the submission's worktree")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--author")
    ap.add_argument("--open-prs", help="comma-separated numbers of the pull requests open now; "
                                       "only these are references")
    ap.add_argument("--maintainers", default="", help="comma-separated logins that are exempt")
    ap.add_argument("--corpus", required=True, help="append-only observation record and block log, "
                                                     "outside the worktree (normally <ledger>/copycat)")
    ap.add_argument("--cleared", action="store_true",
                    help="a maintainer has cleared this pull request; record it, flag nothing")
    ap.add_argument("--no-record", action="store_true",
                    help="judge without writing the corpus or the block log")
    ap.add_argument("--unblock", metavar="LOGIN", help="lift a block, recorded with --reason")
    ap.add_argument("--reason", default="")
    ap.add_argument("--now", help="observation time (ISO 8601 UTC); defaults to now")
    ap.add_argument("--json")
    a = ap.parse_args()
    corpus = Path(a.corpus)
    now = a.now or utcnow()

    if a.unblock:
        if not a.reason.strip():
            print("!! an unblock needs --reason; the block log is a record, not a switch.",
                  file=sys.stderr)
            return 2
        append_block(corpus, {"action": "unblock", "author": a.unblock, "reason": a.reason, "at": now})
        print(f">> unblocked {a.unblock}: {a.reason}")
        return 0

    missing = [f for f, v in (("--repo", a.repo), ("--pr", a.pr), ("--author", a.author),
                              ("--open-prs", a.open_prs)) if v is None]
    if missing:
        print(f"!! {', '.join(missing)} required", file=sys.stderr)
        return 2
    repo = Path(a.repo)
    if corpus.resolve().is_relative_to(repo.resolve()):
        print("!! the corpus is inside the submission worktree; a submission that can rewrite who "
              "had it first, or who is blocked, is not being judged.", file=sys.stderr)
        return 2

    diff = submission_diff(repo, a.base)
    head = git(repo, "rev-parse", "HEAD").strip()
    fp = CC.fingerprint_diff(diff)
    observed = load_corpus(corpus)
    mine = next((e for e in observed if e["pr"] == a.pr and e["head"] == head), None)
    entry = {"pr": a.pr, "author": a.author, "head": head,
             "first_seen": mine["first_seen"] if mine else now,
             "added": fp["added"], "base": fp["base"], "paths": fp["paths"]}
    if not a.no_record:
        record(corpus, entry)

    maintainers = {m.strip().lower() for m in a.maintainers.split(",") if m.strip()}
    open_now = {int(n) for n in a.open_prs.split(",") if n.strip()}
    block = current_block(corpus, a.author)
    doc = {"pr": a.pr, "author": a.author, "head": head, "first_seen": entry["first_seen"],
           "kind": None, "reason": None, "original": None, "containment": None, "new_code": None,
           "evidence": [], "blocked": False, "block": None,
           "parameters": {"k": CC.K, "window": CC.W, "min_mass": CC.MIN_MASS,
                          "copy": CC.T_COPY, "review": CC.T_REVIEW}}

    if a.author.lower() in maintainers:
        doc.update(outcome="EXEMPT", reason="a maintainer named in .github/CODEOWNERS")
    elif a.cleared:
        doc.update(outcome="CLEARED", reason="cleared by a maintainer")
    elif block:
        doc.update(outcome="BLOCKED", blocked=True, block=block,
                   reason=f"the account was blocked for #{block['pr']}")
    else:
        others = [e for e in observed if e["pr"] != a.pr and e["pr"] in open_now]
        latest = {}
        for e in others:
            if e["pr"] not in latest or e["first_seen"] > latest[e["pr"]]["first_seen"]:
                latest[e["pr"]] = e
        stacked = stacked_on(repo, others)
        verdict = CC.judge(entry, others, on_main=CC.fingerprint_sources(main_sources(repo, a.base)),
                           boiler=CC.boilerplate([e["added"] for e in latest.values()]),
                           stacked=stacked)
        doc["stacked_on"] = sorted(stacked)
        doc.update(outcome=verdict["outcome"], kind=verdict.get("kind"), reason=verdict.get("reason"),
                   original=verdict.get("original"), containment=verdict.get("containment"),
                   new_code=verdict.get("new_code"), observations_compared=len(others))
        if verdict["outcome"] in ("COPY", "REVIEW"):
            doc["evidence"] = CC.evidence(diff, verdict["shared"])
        if verdict["outcome"] == "COPY" and not a.no_record:
            rec = {"action": "block", "author": a.author, "pr": a.pr, "head": head,
                   "original": verdict["original"], "reason": verdict["reason"], "at": now}
            append_block(corpus, rec)
            doc.update(blocked=True, block=rec)

    print(f">> copycat guard: #{a.pr} by {a.author}: {doc['outcome']}"
          + (f" -- {doc['reason']}" if doc["reason"] else "")
          + (" (account blocked)" if doc["blocked"] else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
