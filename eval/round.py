#!/usr/bin/env python3
"""One evaluation round: pick a few open pull requests, score them, merge at most one.

    eval/round.py --repo owner/name --slots 3 --ledger DIR --weights DIR --noise FILE
    eval/round.py --repo owner/name --dry-run

Why rounds, rather than scoring each pull request as it arrives
---------------------------------------------------------------

Two benchmarks cannot run at once -- they race for VRAM and both results are worthless -- so
throughput is a hard constraint rather than a tuning question. A submission costs about 24
measured GPU-minutes, so a two-hour round fits three comfortably and five not at all.

But the deeper reason is that **gains do not compose**, and that is what forces "merge one, rebase
the rest":

  - The ledger compounds toward the ceiling. Two submissions each closing 20% of the remaining
    gap close 36% together, not 40%.
  - Worse, two wins can overlap entirely. Fused AdaLN and CUDA-graph capture both attack launch
    overhead; the second one, measured against the old `main`, may be worth nothing once the
    first has landed.

So at most one submission per round can be credited against a known baseline. Scoring two against
the same `main` and merging both would pay twice for one improvement. Everything else in the round
is asked to rebase and be re-measured -- not because its result was wrong, but because it was a
measurement of a baseline that no longer exists.

What a round freezes
--------------------

The head commit of every selected pull request, at the moment the round starts. Later pushes
cannot reach the measurement: the worktree is built from that exact SHA. The receipt records it
and every comment names it, so a label that has been overtaken by a push is visibly stale rather
than quietly wrong.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pr_bot as B
from burnscore import verdict as V

ROOT = Path(__file__).resolve().parent.parent

LOCK_PATH = os.environ.get("BURNISH_ROUND_LOCK", "/tmp/burnish-round.lock")

# Applied to every scored submission in a round except the winner.
NEEDS_REBASE = f"{V.PREFIX}:needs-rebase"
# The winner, when merging is not enabled. A label rather than a merge, so a human can look.
MERGE_FIRST = f"{V.PREFIX}:merge-first"

def _round_cost():
    """What a submission costs, from the artifact rather than a constant in this file.

    Used only to warn when a slot budget cannot fit the interval it is being run on -- never to
    predict a score. It was a pair of hardcoded numbers whose provenance was a file mtime
    somebody had read off a directory listing; a figure this evaluator ACTS on has to come from
    something a reader can check.
    """
    p = ROOT / "eval" / "cells" / "BG-1" / "round-cost.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return d["total_minutes_warm_cache"], d["stages_minutes"]["gate_base"]


MEASURED_MINUTES_PER_PR, COLD_GATE_MINUTES = _round_cost()


class RoundBusy(RuntimeError):
    """Another round is already running. Two benches at once make both results worthless."""


class Lock:
    """Exclusive right to run a round. Non-blocking: a second round exits rather than queueing.

    Queueing would be worse than skipping. A round that waits an hour for the lock then runs
    with a stale PR list, against a `main` that moved while it waited, produces measurements of
    a baseline nobody asked about. Skipping costs one round; the next cron tick is two hours
    away and the work is still there.
    """

    def __init__(self, path=LOCK_PATH):
        self.path, self.fh = path, None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise RoundBusy(
                f"another round holds {self.path}. Not queueing behind it: a round that waits "
                f"then runs would measure against a `main` that moved while it waited.")
        self.fh.write(f"{os.getpid()} {time.strftime('%FT%TZ', time.gmtime())}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False


def open_prs_fifo(repo) -> list:
    """Open pull requests, oldest first. First in, first out.

    FIFO rather than "most promising first" on purpose. Any ordering that reads the submission
    to decide whether to measure it is an ordering somebody can game, and it makes the wait a
    function of the evaluator's opinion rather than of the queue. Oldest-first is the only
    ordering that needs no judgement.
    """
    r = B.gh(["pr", "list", "-R", repo, "--state", "open", "--limit", "100",
              "--json", "number,headRefOid,headRefName,labels,title,author,createdAt"])
    prs = json.loads(r.stdout)
    return sorted(prs, key=lambda p: p["createdAt"])


def select(prs, slots) -> tuple:
    """Which pull requests this round takes, and which wait for the next one.

    A submission that already carries an outcome is not re-measured -- the label is keyed to a
    commit, and re-scoring an unchanged one spends GPU time to re-derive the same number.

    Slots count only submissions that will be MEASURED. A pull request the guard skips costs
    nothing, so letting it consume a slot would mean a round of five instrument edits displaced
    five real submissions for no measurement at all.
    """
    fresh = [p for p in prs if not B.already_labelled(p)]
    return fresh[:slots], fresh[slots:]


def decide_winner(results):
    """The one submission this round may merge, or None.

    Highest credited gap-closed, and only if it actually pays. A round of four null results must
    merge nothing: giving away a merge for an unresolved measurement is paying for a number
    nobody could distinguish from a quiet afternoon.
    """
    paying = [r for r in results if r.get("payout_fraction", 0) > 0]
    if not paying:
        return None
    return max(paying, key=lambda r: r["payout_fraction"])


def merge(repo, num, *, dry_run=False) -> bool:
    if dry_run:
        print(f"   [dry-run] would merge #{num}")
        return True
    r = B.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--delete-branch"], check=False)
    if r.returncode != 0:
        print(f"   !! could not merge #{num}: {r.stderr.strip()[:200]}")
        return False
    return True


def _rebase_note(num, winner, results) -> str:
    w = next(r for r in results if r["pr"] == winner)
    return (
        f"### `{NEEDS_REBASE}`\n\n"
        f"This was measured in the same round as #{winner}, which closed a larger fraction of "
        f"the remaining gap (`{w['payout_fraction']:+.4f}` against your "
        f"`{next(r['payout_fraction'] for r in results if r['pr'] == num):+.4f}`) and has been "
        f"merged.\n\n"
        f"**Your result is not wrong — it is a measurement of a baseline that no longer "
        f"exists.** Gains do not compose: the ledger compounds toward the ceiling, so two "
        f"submissions each closing 20% of the remaining gap close 36% together rather than 40%. "
        f"And two wins can overlap entirely — fused AdaLN and CUDA-graph capture both attack "
        f"launch overhead, so a gain measured against the old `main` can be worth nothing once "
        f"another has landed.\n\n"
        f"Rebase onto `main` and it will be re-measured in a later round. Nothing is lost; the "
        f"number simply has to be against the baseline that now exists.")


def run(args) -> dict:
    started = time.strftime("%FT%TZ", time.gmtime())
    budget = COLD_GATE_MINUTES + args.slots * MEASURED_MINUTES_PER_PR
    print(f">> round starting {started}  ({args.slots} slots)")
    if args.interval and budget > args.interval:
        print(f"   !! {args.slots} slots is about {budget:.0f} measured minutes against a "
              f"{args.interval:.0f}-minute interval.\n"
              f"      Rounds will overlap, and the lock will make the next one skip. Lower "
              f"--slots.")

    prs = open_prs_fifo(args.repo)
    args.open_prs = [p["number"] for p in prs]          # the copycat guard's references
    taken, waiting = select(prs, args.slots)
    print(f"   {len(prs)} open, {len(taken)} taken this round, {len(waiting)} waiting")
    for p in taken:
        # Printed because this is the freeze: later pushes cannot reach this measurement.
        print(f"     #{p['number']:<5} {p['headRefOid'][:12]}  {p['title'][:52]}")
    if not taken:
        return {"started": started, "evaluated": [], "winner": None,
                "waiting": [p["number"] for p in waiting]}

    results, scored = [], []
    for p in taken:
        try:
            r = B.evaluate(args.repo, p, args)
        except Exception as exc:                     # one bad PR must not end the round
            print(f"   !! #{p['number']}: {exc}", file=sys.stderr)
            r = {"pr": p["number"], "outcome": "EVAL_ERROR", "error": str(exc)}
        r["head"] = p["headRefOid"]
        results.append(r)
        if r.get("payout_fraction") is not None:
            scored.append(r)

    winner = decide_winner(scored)
    merged = None
    if winner is None:
        print("\n>> nothing to merge: no submission in this round resolved a gain.")
    else:
        print(f"\n>> best of round: #{winner['pr']}  "
              f"{winner['label']}  pays {winner['payout_fraction']:.4f}")
        if args.merge:
            merged = winner["pr"] if merge(args.repo, winner["pr"],
                                           dry_run=args.dry_run) else None
        else:
            B.set_label(args.repo, winner["pr"], MERGE_FIRST, color=V.GREEN,
                        description="best verified gain of its round", dry_run=args.dry_run)
            print("   labelled, not merged (--merge is off)")

        # Everyone else measured against a baseline that is about to move, or already has.
        for r in scored:
            if r["pr"] == winner["pr"]:
                continue
            B.set_label(args.repo, r["pr"], NEEDS_REBASE, color=V.AMBER,
                        description="measured against a baseline that has since moved",
                        dry_run=args.dry_run)
            B.comment(args.repo, r["pr"], _rebase_note(r["pr"], winner["pr"], scored),
                      dry_run=args.dry_run)

    return {"started": started, "finished": time.strftime("%FT%TZ", time.gmtime()),
            "slots": args.slots, "evaluated": results,
            "winner": winner["pr"] if winner else None, "merged": merged,
            "waiting": [p["number"] for p in waiting]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--slots", type=int, default=3,
                    help="how many submissions to MEASURE this round. Three fits a two-hour "
                         "interval with slack at the measured cost of about 24 minutes each; "
                         "five does not fit at all.")
    ap.add_argument("--interval", type=float, default=120,
                    help="minutes between rounds, used only to warn about an unfittable budget")
    ap.add_argument("--merge", action="store_true",
                    help="merge the winner. Off by default: a round that merges unattended is "
                         "an outward-facing action, so it is opted into rather than assumed.")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--ledger", default=os.environ.get("BURNISH_LEDGER", ""))
    ap.add_argument("--weights", default=os.environ.get("BURNISH_WEIGHTS", ""))
    ap.add_argument("--noise", default=os.environ.get("BURNISH_NOISE", ""))
    ap.add_argument("--generation", default=os.environ.get("BURNISH_GENERATION", "BG-1"),
                    help="the frozen generation this round scores; --noise must be that "
                         "generation's pinned noise")
    ap.add_argument("--calibration", default=os.environ.get("BURNISH_CALIBRATION", ""))
    ap.add_argument("--copycat-corpus", default=os.environ.get("BURNISH_COPYCAT_CORPUS", ""),
                    help="append-only copycat observation record; defaults to <ledger>/copycat")
    ap.add_argument("--impl-base", default="cuda")
    ap.add_argument("--impl-candidate", default="cuda")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    a.pr = None
    a.once = True

    try:
        with Lock():
            out = run(a)
    except RoundBusy as exc:
        print(f">> skipping this round: {exc}")
        return 0
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
