#!/usr/bin/env python3
"""One evaluation round: FIFO selection, one merge, everyone else rebases.

The rule that shapes all of this is that **gains do not compose**. The ledger compounds toward
the ceiling, so two submissions each closing 20% of the remaining gap close 36% together rather
than 40% -- and worse, two wins can overlap entirely, because fused AdaLN and CUDA-graph capture
both attack launch overhead. A gain measured against the old `main` can be worth nothing once
another has landed.

So at most one submission per round may be credited against a known baseline. Scoring two
against the same `main` and merging both would pay twice for one improvement.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import round as RD
from burnscore import verdict as V


def pr(num, *, created, labels=(), head=None):
    return {"number": num, "createdAt": created, "headRefOid": head or f"{num:040d}",
            "headRefName": f"b{num}", "title": f"pr {num}",
            "labels": [{"name": n} for n in labels]}


class TestSelection(unittest.TestCase):
    def test_oldest_first(self):
        """FIFO, because any ordering that reads the submission can be gamed, and it makes the
        wait a function of the evaluator's opinion rather than of the queue."""
        prs = [pr(3, created="2026-09-03"), pr(1, created="2026-09-01"),
               pr(2, created="2026-09-02")]
        due = RD.select(sorted(prs, key=lambda p: p["createdAt"]))
        self.assertEqual([p["number"] for p in due], [1, 2, 3])

    def test_a_submission_that_already_has_an_outcome_is_not_remeasured(self):
        """An unchanged commit with an outcome: re-scoring it re-derives the same number."""
        prs = [pr(1, created="2026-09-01", labels=["burnish:unresolved"]),
               pr(2, created="2026-09-02")]
        last = {1: {"head": pr(1, created="")["headRefOid"]}}.get
        self.assertEqual([p["number"] for p in RD.select(prs, last)], [2])

    def test_a_push_after_an_outcome_makes_it_due_again(self):
        """`needs-rebase` says rebase and it is measured again; that has to be true."""
        prs = [pr(1, created="2026-09-01", labels=["burnish:needs-rebase"], head="b" * 40)]
        last = {1: {"head": "a" * 40}}.get
        self.assertEqual([p["number"] for p in RD.select(prs, last)], [1])


class _FakeRound:
    """`run` with the bot's GitHub and GPU halves replaced by recorded outcomes."""

    def __init__(self, prs, outcomes):
        import argparse
        import pr_bot as B
        self.B, self.calls, self.labels = B, [], []
        self._saved = (RD.open_prs_fifo, B.evaluate, B.set_label, B.add_label, B.comment)
        RD.open_prs_fifo = lambda repo: prs
        B.evaluate = lambda repo, p, args: (self.calls.append(p["number"])
                                            or dict(outcomes[p["number"]], pr=p["number"]))
        B.set_label = lambda repo, num, label, **kw: self.labels.append(("set", num, label))
        B.add_label = lambda repo, num, label, **kw: self.labels.append(("add", num, label))
        B.comment = lambda *a, **kw: None
        self.args = argparse.Namespace(slots=2, interval=0, merge=False, dry_run=True, ledger="",
                                       repo="o/r", generation="BG-1", calibration="")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        RD.open_prs_fifo, self.B.evaluate, self.B.set_label, self.B.add_label, self.B.comment = \
            self._saved


class TestSlots(unittest.TestCase):
    def test_slots_count_only_submissions_that_are_measured(self):
        """A round of guard-skipped pull requests must not displace real submissions."""
        prs = [pr(i, created=f"2026-09-{i:02d}") for i in range(1, 6)]
        outcomes = {1: {"outcome": "SKIPPED"}, 2: {"outcome": "UNRESOLVED", "payout_fraction": 0.0},
                    3: {"outcome": "COPYCAT"}, 4: {"outcome": "NO_GAIN", "payout_fraction": 0.0},
                    5: {"outcome": "UNRESOLVED", "payout_fraction": 0.0}}
        with _FakeRound(prs, outcomes) as f:
            out = RD.run(f.args)
        self.assertEqual(f.calls, [1, 2, 3, 4])
        self.assertEqual(out["waiting"], [5])


class TestTheRoundLabels(unittest.TestCase):
    def test_only_other_gains_are_asked_to_rebase_and_the_winner_keeps_its_number(self):
        prs = [pr(i, created=f"2026-09-{i:02d}") for i in range(1, 4)]
        outcomes = {1: {"outcome": "FRONTIER_EXPANDED", "label": "burnish:gap+0.0100",
                        "payout_fraction": 0.01},
                    2: {"outcome": "FRONTIER_EXPANDED", "label": "burnish:gap+0.0300",
                        "payout_fraction": 0.03},
                    3: {"outcome": "NO_GAIN", "payout_fraction": 0.0}}
        with _FakeRound(prs, outcomes) as f:
            f.args.slots = 3
            RD.run(f.args)
        self.assertIn(("add", 2, RD.MERGE_FIRST), f.labels, "merge-first must not replace the number")
        self.assertIn(("set", 1, RD.NEEDS_REBASE), f.labels)
        self.assertNotIn(("set", 3, RD.NEEDS_REBASE), f.labels,
                         "a result that did not pay was relabelled needs-rebase")
        self.assertNotIn(("set", 2, RD.MERGE_FIRST), f.labels)


class TestHolds(unittest.TestCase):
    def test_the_latest_receipt_per_pull_request_decides_a_hold(self):
        history = [{"pr": 7, "held": True, "receipt": "a"}, {"pr": 7, "held": False, "receipt": "b"},
                   {"pr": 8, "held": True, "receipt": "c"}, {"pr": 9, "held": False, "receipt": "d"}]
        hold, release, _ = RD.hold_changes(history, labelled={7, 9})
        self.assertEqual(hold, [8])
        self.assertEqual(release, [7, 9])


class TestWhatGetsMerged(unittest.TestCase):
    def test_the_largest_credited_gain_wins(self):
        results = [{"pr": 1, "payout_fraction": 0.004}, {"pr": 2, "payout_fraction": 0.031},
                   {"pr": 3, "payout_fraction": 0.012}]
        self.assertEqual(RD.decide_winner(results)["pr"], 2)

    def test_a_round_of_null_results_merges_nothing(self):
        """Giving away a merge for an unresolved measurement is paying for a number nobody could
        distinguish from a quiet afternoon."""
        results = [{"pr": 1, "payout_fraction": 0.0}, {"pr": 2, "payout_fraction": 0.0}]
        self.assertIsNone(RD.decide_winner(results))

    def test_a_regression_never_wins(self):
        results = [{"pr": 1, "payout_fraction": 0.0}, {"pr": 2, "payout_fraction": 0.0}]
        self.assertIsNone(RD.decide_winner(results))

    def test_one_paying_submission_wins_alone(self):
        results = [{"pr": 1, "payout_fraction": 0.0}, {"pr": 7, "payout_fraction": 0.0009}]
        self.assertEqual(RD.decide_winner(results)["pr"], 7)


class TestTheLock(unittest.TestCase):
    def test_a_second_round_skips_rather_than_queueing(self):
        """Queueing is worse than skipping: a round that waits an hour then runs measures
        against a `main` that moved while it waited. The next tick is two hours away and the
        work is still there."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "round.lock")
            with RD.Lock(path):
                with self.assertRaises(RD.RoundBusy):
                    with RD.Lock(path):
                        pass
            # released, so it can be taken again
            with RD.Lock(path):
                pass

    def test_the_cron_wrapper_does_not_queue_either(self):
        """`exec`s the round, which takes the lock itself and exits 0 when it is held."""
        src = (ROOT / "eval" / "run_round_cron.sh").read_text()
        self.assertIn("exec python3 -u eval/round.py", src)
        self.assertIn("git merge --quiet --ff-only origin/main", src)
        self.assertIn("./scripts/build_cuda.sh", src)

    def test_the_cron_wrapper_defaults_to_twelve_slots(self):
        src = (ROOT / "eval" / "run_round_cron.sh").read_text()
        self.assertIn("BURNISH_SLOTS:-12", src)


class TestTheBudgetIsChecked(unittest.TestCase):
    def test_twelve_slots_fit_two_hours_and_thirty_do_not(self):
        """From the measured cost, so this fails if the cost is ever re-measured upward without
        the slot count being revisited."""
        per = RD.MEASURED_MINUTES_PER_PR
        cold = RD.COLD_GATE_MINUTES
        self.assertLess(cold + 12 * per, 120, "twelve slots no longer fit a two-hour round")
        self.assertGreater(cold + 30 * per, 120, "thirty slots now fit; the warning is stale")

    def test_the_default_is_twelve(self):
        src = (ROOT / "eval" / "round.py").read_text()
        self.assertIn('"--slots", type=int, default=12', src)

    def test_merging_is_opted_into_not_assumed(self):
        """A round that merges unattended is an outward-facing action."""
        src = (ROOT / "eval" / "round.py").read_text()
        self.assertIn('"--merge", action="store_true"', src)


class TestWhatTheAuthorIsTold(unittest.TestCase):
    def test_the_rebase_note_says_it_is_not_a_rejection(self):
        results = [{"pr": 4, "payout_fraction": 0.0031},
                   {"pr": 9, "payout_fraction": 0.0210}]
        note = RD._rebase_note(4, 9, results)
        self.assertIn("not wrong", note)
        self.assertIn("Gains do not compose", note)
        self.assertIn("#9", note)
        self.assertIn("+0.0210", note)
        self.assertIn("+0.0031", note)
        self.assertIn("Nothing is lost", note)

    def test_every_comment_names_the_commit_it_measured(self):
        """A round freezes the head, but GitHub does not remove a label when you push -- so
        without this the author sees a verdict that looks like it describes their new head."""
        import pr_bot as B
        rec = json.loads((ROOT / "examples" / "BG-1-pr-000001-receipt.json").read_text())
        body = B.report(V.verdict(rec), rec, raw_name="r.json", receipt_name="x.json")
        commit = rec["provenance"]["candidate_commit"]
        self.assertIn(commit[:12], body)
        self.assertIn("freezes the head commit", body)

    def test_both_round_labels_are_in_the_taxonomy_with_meanings(self):
        for label in (RD.NEEDS_REBASE, RD.MERGE_FIRST):
            self.assertIn(label, V.all_labels(), f"{label} is applied but not declared")
        self.assertIn(V.NEEDS_REBASE, V.ALL_OUTCOMES)
        self.assertIn(V.MERGE_FIRST, V.ALL_OUTCOMES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
