#!/usr/bin/env python3
"""eval/run_from_base.sh must tell the instrument what it is scoring.

The overlay is the guard that stops a submission grading its own homework: `eval/`, `configs/`,
`schemas/` and `tools/burnish` are taken from the base ref, so a one-line edit to a noise floor,
a ceiling, a tolerance or the model revision cannot reach the scorer. None of those look like
cheating in a diff; several look like tidying.

The side effect is that the instrument then runs from a staging directory extracted with `git
archive` -- files, no history. So the evaluator cannot find out which commit it is scoring by
asking git about its own location: it gets nothing. It did get nothing, for the first two
receipts this harness ever produced, both of which named the instrument they were scored with and
left the candidate blank.

Only this script knows where the submission is. These tests pin that it says so.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "eval" / "run_from_base.sh"


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True).stdout.strip()


@unittest.skipUnless((ROOT / ".git").exists(), "needs a git checkout")
class TestTheOverlayNamesWhatItScores(unittest.TestCase):
    """Runs the real script against a real worktree. `roofline --help` needs no GPU."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wt = Path(cls.tmp.name) / "wt"
        head = _git("rev-parse", "HEAD")
        r = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q",
                            str(cls.wt), head], capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"could not make a worktree: {r.stderr}")
        cls.head = head
        env = dict(os.environ, BURNISH_ENTRY="roofline")
        cls.proc = subprocess.run(
            ["bash", str(SCRIPT), head, str(cls.wt), "--", "--help"],
            capture_output=True, text=True, env=env, timeout=300)

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(cls.wt)],
                       capture_output=True)
        cls.tmp.cleanup()

    def test_it_runs(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_it_reports_the_instrument_it_overlaid(self):
        self.assertIn(">> instrument:", self.proc.stdout)
        for path in ("eval", "configs", "schemas", "tools/burnish"):
            self.assertIn(path, self.proc.stdout,
                          f"{path} is part of what decides WHAT IS MEASURED and must be "
                          f"overlaid from the base ref, and reported as overlaid")

    def test_it_reports_the_commit_under_test(self):
        """Without this line the receipt comes back unable to name the code it scored."""
        self.assertIn(self.head, self.proc.stdout,
                      "the overlay does not say which commit it is scoring. The instrument runs "
                      "from a staging directory with no git history, so if this script does not "
                      "pass the commit down, nothing can.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
