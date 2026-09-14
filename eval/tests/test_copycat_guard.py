#!/usr/bin/env python3
"""The copycat guard end to end on a throwaway git repository, and the bot's use of its verdict."""
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
sys.path.insert(0, str(HERE))

from test_copycat import KERNEL, RENAMED, OTHER

GUARD = ROOT / "scripts" / "copycat_guard.py"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t",
                    *args], check=True, capture_output=True, text=True)


class TestTheGuardEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.corpus = Path(self.tmp.name) / "corpus"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "src" / "cuda").mkdir(parents=True)
        (self.repo / "src" / "cuda" / "ops_cuda.cu").write_text(OTHER + "\n")
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-q", "-m", "baseline")

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, branch, code):
        git(self.repo, "checkout", "-q", "-B", branch, "main")
        (self.repo / "src" / "cuda" / f"{branch}.cu").write_text(code + "\n")
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-q", "-m", branch)

    def run_guard(self, pr, author, now, *extra, open_prs="10,11,12", corpus=None, args=None):
        out = Path(self.tmp.name) / f"v-{pr}-{now}.json"
        cmd = args or ["--repo", str(self.repo), "--base", "main", "--pr", str(pr),
                       "--author", author, "--open-prs", open_prs, "--now", now,
                       "--json", str(out), *extra]
        r = subprocess.run([sys.executable, str(GUARD), "--corpus", str(corpus or self.corpus), *cmd],
                           capture_output=True, text=True)
        return r, (json.loads(out.read_text()) if out.exists() else None)

    def copy_scenario(self):
        self.submit("alice", KERNEL)
        self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        self.submit("bob", RENAMED)
        return self.run_guard(11, "bob", "2026-09-02T00:00:00Z")

    def test_the_first_submission_is_clear_and_recorded(self):
        self.submit("alice", KERNEL)
        r, v = self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(v["outcome"], "CLEAR")
        self.assertEqual(len(list((self.corpus / "entries").glob("*.json"))), 1)

    def test_a_renamed_copy_of_an_open_pull_request_is_a_copy_and_blocks_its_author(self):
        r, v = self.copy_scenario()
        self.assertEqual(v["outcome"], "COPY", r.stdout + r.stderr)
        self.assertEqual((v["original"]["pr"], v["original"]["author"]), (10, "alice"))
        self.assertTrue(v["evidence"])
        self.assertTrue(v["blocked"])

    def test_a_branch_stacked_on_another_open_pull_request_is_review_not_a_block(self):
        """Its diff against main carries the other pull request by construction."""
        self.submit("alice", KERNEL)
        self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        git(self.repo, "checkout", "-q", "-B", "bob", "alice")
        (self.repo / "src" / "cuda" / "bob.cu").write_text("int tiny() { return 1; }\n")
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-q", "-m", "bob on alice")
        r, v = self.run_guard(11, "bob", "2026-09-02T00:00:00Z")
        self.assertEqual((v["outcome"], v["kind"]), ("REVIEW", "stacked"), r.stdout + r.stderr)
        self.assertEqual(v["stacked_on"], [10])
        self.assertFalse(v["blocked"])
        self.assertFalse((self.corpus / "blocked.jsonl").exists())

    def test_a_copy_of_a_pull_request_that_is_no_longer_open_is_not_this_guards_question(self):
        self.submit("alice", KERNEL)
        self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        self.submit("bob", RENAMED)
        _, v = self.run_guard(11, "bob", "2026-09-02T00:00:00Z", open_prs="11")
        self.assertEqual(v["outcome"], "CLEAR")

    def test_a_blocked_author_is_blocked_on_every_later_submission(self):
        self.copy_scenario()
        self.submit("bob2", OTHER.replace("acc +=", "acc -="))
        _, v = self.run_guard(12, "bob", "2026-09-03T00:00:00Z")
        self.assertEqual(v["outcome"], "BLOCKED")
        self.assertEqual(v["block"]["pr"], 11)

    def test_an_unblock_is_recorded_and_lifts_the_block(self):
        self.copy_scenario()
        r, _ = self.run_guard(0, "", "", args=["--unblock", "bob"])
        self.assertEqual(r.returncode, 2, "an unblock without a reason must be refused")
        r, _ = self.run_guard(0, "", "", args=["--unblock", "bob", "--reason", "independent work"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.submit("bob2", OTHER.replace("acc +=", "acc -="))
        _, v = self.run_guard(12, "bob", "2026-09-03T00:00:00Z")
        self.assertEqual(v["outcome"], "CLEAR")
        self.assertEqual([rec["action"] for rec in map(json.loads,
                          (self.corpus / "blocked.jsonl").read_text().splitlines())], ["block", "unblock"])

    def test_a_maintainer_is_exempt(self):
        self.submit("alice", KERNEL)
        self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        self.submit("bob", RENAMED)
        _, v = self.run_guard(11, "bob", "2026-09-02T00:00:00Z", "--maintainers", "Bob")
        self.assertEqual(v["outcome"], "EXEMPT")
        self.assertFalse((self.corpus / "blocked.jsonl").exists())

    def test_a_cleared_submission_is_neither_flagged_nor_blocked(self):
        self.submit("alice", KERNEL)
        self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        self.submit("bob", RENAMED)
        _, v = self.run_guard(11, "bob", "2026-09-02T00:00:00Z", "--cleared")
        self.assertEqual(v["outcome"], "CLEARED")
        self.assertFalse((self.corpus / "blocked.jsonl").exists())

    def test_a_head_keeps_the_time_it_was_first_observed(self):
        self.submit("alice", KERNEL)
        self.run_guard(10, "alice", "2026-09-01T00:00:00Z")
        _, v = self.run_guard(10, "alice", "2026-09-05T00:00:00Z")
        self.assertEqual(v["first_seen"], "2026-09-01T00:00:00Z")

    def test_starting_from_the_baseline_on_main_is_not_copying(self):
        self.submit("bob", OTHER.replace("acc +=", "acc -="))
        _, v = self.run_guard(11, "bob", "2026-09-02T00:00:00Z")
        self.assertEqual(v["outcome"], "CLEAR")

    def test_the_corpus_may_not_live_inside_the_worktree(self):
        self.submit("alice", KERNEL)
        r, _ = self.run_guard(10, "alice", "2026-09-01T00:00:00Z", corpus=self.repo / "corpus")
        self.assertEqual(r.returncode, 2)


class TestTheBotActsOnTheVerdicts(unittest.TestCase):
    SRC = (ROOT / "eval" / "pr_bot.py").read_text()

    def test_blocked_and_copied_submissions_are_answered_before_the_instrument_guard(self):
        self.assertLess(self.SRC.index("cc = copycat(repo, wt, pr, args)"),
                        self.SRC.index("g = guard(wt, args.base)"))

    def test_the_reregistration_guard_runs_before_anything_is_built(self):
        self.assertLess(self.SRC.index("rr = reregistration(wt, pr, args)"),
                        self.SRC.index("build_cuda.sh"))

    def test_a_copy_and_a_blocked_account_close_the_pull_request(self):
        for outcome in ('if cc["outcome"] == "BLOCKED":', 'if cc["outcome"] == "COPY":'):
            start = self.SRC.index(outcome, self.SRC.index("def _evaluate("))
            self.assertIn("close_pr(repo, num", self.SRC[start:self.SRC.index("return", start)])

    def test_a_review_is_measured_but_pays_nothing(self):
        start = self.SRC.index('if cc["outcome"] == "REVIEW" and v["pays"]:')
        block = self.SRC[start:self.SRC.index('set_label(repo, num, v["label"]', start)]
        self.assertIn('"payout_fraction": 0.0', block)

    def test_a_failed_guard_is_an_evaluator_error_not_a_pass(self):
        self.assertIn('if cc["outcome"] == "ERROR":', self.SRC)
        self.assertIn('if rr["outcome"] == "ERROR":', self.SRC)

    def test_the_bot_passes_the_open_pull_requests_and_the_maintainers(self):
        self.assertIn('"--open-prs"', self.SRC)
        self.assertIn('"--maintainers"', self.SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
