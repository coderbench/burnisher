#!/usr/bin/env python3
"""Does this submission register a kernel that is already on main under a new name?

    scripts/reregistration_guard.py --repo <worktree> --base origin/main --json verdict.json

Answered before any GPU time is spent. See `eval/burnscore/reregistration.py` for what counts and
why constants are kept. The verdict is written to --json for the bot to act on.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import copycat as CC  # noqa: E402
from burnscore import reregistration as RR  # noqa: E402

RUNTIME = ("src/", "include/", "tools/")


def tree(repo, ref):
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref],
                         capture_output=True, text=True, check=True).stdout
    files = {}
    for path in out.splitlines():
        if path.startswith(RUNTIME) and CC.is_code(path):
            r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                files[path] = r.stdout
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--cleared", action="store_true",
                    help="a maintainer has cleared this pull request; flag nothing")
    ap.add_argument("--json")
    a = ap.parse_args()
    verdict = RR.judge(tree(a.repo, "HEAD"), tree(a.repo, a.base))
    if a.cleared and verdict["outcome"] == "REREGISTERED":
        verdict = dict(verdict, outcome="CLEARED")
    print(f">> reregistration guard: {verdict['outcome']}; new kernel names: "
          f"{', '.join(verdict['candidate_names']) or 'none'}"
          + "".join(f"\n   {f['registration']['op']}/{f['registration']['name']} = "
                    f"{f['matches']['op']}/{f['matches']['name']} ({f['kind']}, {f['similarity']:.0%})"
                    for f in verdict["findings"]))
    verdict["parameters"] = {"similar": RR.SIMILAR, "min_tokens": RR.MIN_TOKENS}
    if a.json:
        Path(a.json).write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
