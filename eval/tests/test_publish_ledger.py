#!/usr/bin/env python3
"""The ledger outlives the rented box: it is committed and pushed after every round, never forced.

Pushed to a local bare repository here, which is the same git path as a GitHub remote apart from
the request header that carries the token.
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

import publish_ledger as P


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
                          env=P.git_env()).stdout


class TestPublishingTheLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.remote = t / "ledger.git"
        git("init", "-q", "--bare", "-b", "main", str(self.remote))
        self.ledger = t / "ledger"
        receipts = self.ledger / "BG-1" / "receipts"
        receipts.mkdir(parents=True)
        (receipts / "pr-000001.json").write_text(json.dumps({"pr": 1}))
        (self.ledger / "copycat").mkdir()
        (self.ledger / "copycat" / "blocked.jsonl").write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def published(self):
        return (git("--git-dir", str(self.remote), "log", "--format=%s", "main").splitlines(),
                git("--git-dir", str(self.remote), "ls-tree", "-r", "--name-only", "main").split())

    def test_every_file_is_published_and_publishing_again_changes_nothing(self):
        first = P.publish(self.ledger, str(self.remote))
        self.assertTrue(first["committed"] and first["pushed"])
        again = P.publish(self.ledger, str(self.remote))
        self.assertFalse(again["committed"])
        commits, files = self.published()
        self.assertEqual(len(commits), 1)
        self.assertEqual(files, ["BG-1/receipts/pr-000001.json", "copycat/blocked.jsonl"])

    def test_a_later_round_adds_a_commit(self):
        P.publish(self.ledger, str(self.remote))
        (self.ledger / "BG-1" / "receipts" / "pr-000002.json").write_text("{}")
        P.publish(self.ledger, str(self.remote))
        commits, files = self.published()
        self.assertEqual(len(commits), 2)
        self.assertIn("BG-1/receipts/pr-000002.json", files)

    def test_a_diverged_remote_is_refused_not_overwritten(self):
        """A force push would erase history somebody else already has."""
        P.publish(self.ledger, str(self.remote))
        other = Path(self.tmp.name) / "other"
        git("clone", "-q", str(self.remote), str(other))
        (other / "elsewhere.json").write_text("{}")
        git("add", "-A", cwd=other)
        git("commit", "-q", "-m", "written elsewhere", cwd=other)
        git("push", "-q", "origin", "main", cwd=other)
        (self.ledger / "BG-1" / "receipts" / "pr-000003.json").write_text("{}")
        with self.assertRaises(P.PublishError) as cm:
            P.publish(self.ledger, str(self.remote))
        self.assertIn("not forced", str(cm.exception))
        commits, _ = self.published()
        self.assertEqual(commits[0], "written elsewhere")

    def test_an_empty_ledger_publishes_nothing(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        self.assertEqual(P.publish(empty, str(self.remote)),
                         {"committed": False, "pushed": False, "head": None})


class TestTheTokenStaysOffTheCommandLine(unittest.TestCase):
    def test_it_travels_as_a_header_in_gits_environment(self):
        seen = []
        real = P.subprocess.run

        def spy(cmd, **kw):
            seen.append((cmd, kw.get("env") or {}))
            return real(cmd, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            remote, ledger = Path(tmp, "r.git"), Path(tmp, "l")
            git("init", "-q", "--bare", "-b", "main", str(remote))
            ledger.mkdir()
            (ledger / "x.json").write_text("{}")
            P.subprocess.run = spy
            try:
                P.publish(ledger, str(remote), token="ghp_ledger_secret")
            finally:
                P.subprocess.run = real
        self.assertTrue(seen)
        for cmd, env in seen:
            self.assertNotIn("ghp_ledger_secret", " ".join(cmd))
            self.assertEqual(env["GIT_CONFIG_KEY_0"], "http.extraHeader")
            self.assertTrue(env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic "))

    def test_a_remote_url_with_credentials_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(P.PublishError):
                P.publish(tmp, "https://someone:ghp_x@github.com/o/ledger.git")


class TestTheRoundPublishes(unittest.TestCase):
    def test_the_round_publishes_inside_its_lock(self):
        src = (ROOT / "eval" / "round.py").read_text()
        block = src[src.index("with Lock():"):src.index("except RoundBusy")]
        self.assertIn("publish_ledger(a)", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
