#!/usr/bin/env python3
"""Publish the ledger: commit what the evaluator appended to it, and push. Never forced.

    eval/publish_ledger.py --ledger DIR --remote https://github.com/owner/burnisher-ledger.git

Why. The ledger -- receipts, raw measurements, which commit each outcome was for, the copycat
record and the block list -- lived only on the evaluation box, and boxes are rented and returned.
A returned box took the history with it, and `burnish challenge --ledger` named a public ledger
that did not exist anywhere.

So after every round the ledger directory is committed and pushed to a repository of its own.

  - A push is never forced. The ledger is append-only on disk (burnscore/ledger.py refuses a
    rewrite); a published history that could be force-pushed would not be. Protect the branch
    against force pushes as well.
  - The token comes from BURNISH_LEDGER_TOKEN and reaches git through its environment, never its
    command line or a file: command lines are readable by every account on the box, the sandbox
    account included.
  - A failed publish does not fail the round. The commit stays local and goes out with the next.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOKEN_ENV = "BURNISH_LEDGER_TOKEN"
REMOTE_ENV = "BURNISH_LEDGER_REMOTE"
AUTHOR = ("burnish evaluator", "evaluator@burnisher.invalid")
_CREDENTIALS = re.compile(r"^https?://[^/\s@]+@")


class PublishError(RuntimeError):
    """The ledger could not be published. Its local history is untouched."""


def git_env(token=None) -> dict:
    """git's environment: a fixed identity, no prompts, and the token as a request header."""
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME=AUTHOR[0], GIT_AUTHOR_EMAIL=AUTHOR[1],
               GIT_COMMITTER_NAME=AUTHOR[0], GIT_COMMITTER_EMAIL=AUTHOR[1],
               GIT_TERMINAL_PROMPT="0")
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraHeader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}")
    return env


def _git(ledger, *args, env, check=True):
    r = subprocess.run(["git", "-C", str(ledger), *args], capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise PublishError(f"git {args[0]} failed: {r.stderr.strip()[-400:]}")
    return r


def publish(ledger, remote, *, token=None, branch="main", message=None) -> dict:
    """Commit everything under `ledger` and push it to `remote`. Returns what happened."""
    ledger = Path(ledger)
    if not ledger.is_dir():
        raise PublishError(f"there is no ledger at {ledger}")
    if _CREDENTIALS.match(remote):
        raise PublishError(f"the remote URL carries credentials; pass the token in {TOKEN_ENV}")
    env = git_env(token)
    if not (ledger / ".git").exists():
        _git(ledger, "init", "-q", "-b", branch, env=env)
    if "origin" in _git(ledger, "remote", env=env).stdout.split():
        _git(ledger, "remote", "set-url", "origin", remote, env=env)
    else:
        _git(ledger, "remote", "add", "origin", remote, env=env)
    _git(ledger, "add", "-A", env=env)
    committed = _git(ledger, "diff", "--cached", "--quiet", env=env, check=False).returncode != 0
    if committed:
        _git(ledger, "commit", "-q", "-m",
             message or f"ledger at {time.strftime('%FT%TZ', time.gmtime())}", env=env)
    head = _git(ledger, "rev-parse", "--verify", "-q", "HEAD", env=env, check=False)
    if head.returncode != 0:
        return {"committed": False, "pushed": False, "head": None}
    push = _git(ledger, "push", "-q", "origin", f"HEAD:refs/heads/{branch}", env=env, check=False)
    if push.returncode != 0:
        raise PublishError("the push was refused and was not forced; the commit stays local and "
                           f"goes out with the next round. {push.stderr.strip()[-400:]}")
    return {"committed": committed, "pushed": True, "head": head.stdout.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=os.environ.get("BURNISH_LEDGER", ""))
    ap.add_argument("--remote", default=os.environ.get(REMOTE_ENV, ""))
    ap.add_argument("--branch", default="main")
    a = ap.parse_args()
    if not a.ledger or not a.remote:
        print(f"!! --ledger and --remote (or BURNISH_LEDGER and {REMOTE_ENV}) are required",
              file=sys.stderr)
        return 2
    try:
        out = publish(a.ledger, a.remote, token=os.environ.get(TOKEN_ENV), branch=a.branch)
    except PublishError as exc:
        print(f"!! could not publish the ledger: {exc}", file=sys.stderr)
        return 1
    print(f">> ledger published at {out['head']}" if out["pushed"]
          else ">> the ledger is empty; nothing to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
