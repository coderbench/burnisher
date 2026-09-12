#!/usr/bin/env python3
"""The one distinction this repository has to make and most benchmarks do not.

A benchmark that forbids contributors touching its harness cannot pay for cartography. A
benchmark that lets them touch it can be won by editing the ruler. Burnisher wants both: opening
a new cell is a scored contribution, and changing an existing one is not allowed.

The line is ADD versus MODIFY, and these tests pin both sides of it. They exercise the
classifier directly rather than shelling out to git, so they run in CI without a repository
history to fabricate.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import instrument_guard as G


class TestTheLineBetweenOpeningACellAndEditingTheRuler(unittest.TestCase):
    def c(self, rows, base_generations=("BG-1",)):
        return G.classify(rows, set(base_generations))

    def test_a_kernel_change_is_contributor_surface(self):
        r = self.c([("M", "src/cuda/ops_cuda.cu"), ("M", "include/burnisher/ops.h"),
                    ("A", "tests/test_ops.cpp")])
        self.assertEqual(r["blocked"], [])
        self.assertEqual(r["cartography"], [])
        self.assertEqual(len(r["contributor"]), 3)

    def test_opening_a_new_generation_is_cartography(self):
        """Allowed, because it cannot change what any existing receipt meant."""
        r = self.c([("A", "eval/cells/BG-2/generation.json"),
                    ("A", "eval/cells/BG-2/reference.json"),
                    ("A", "eval/cells/BG-2/reference-latents/manifest.json")])
        self.assertEqual(r["blocked"], [])
        self.assertEqual(len(r["cartography"]), 3)

    def test_editing_an_existing_generation_is_blocked_and_named(self):
        """The attack this guard exists for: quietly widening what counts as resolved."""
        r = self.c([("M", "eval/cells/BG-1/reference.json")])
        self.assertEqual(len(r["blocked"]), 1)
        path, why = r["blocked"][0]
        self.assertEqual(path, "eval/cells/BG-1/reference.json")
        self.assertIn("frozen generation BG-1", why)
        self.assertIn("re-scores history", why)

    def test_editing_the_scorer_is_blocked(self):
        for path in ("eval/burnscore/compute.py", "eval/burnscore/floor.py",
                     "configs/devices.json", "configs/axes.json", "schemas/receipt.schema.json",
                     "tools/burnish"):
            r = self.c([("M", path)])
            self.assertEqual(len(r["blocked"]), 1, f"{path} was not blocked")

    def test_adding_a_file_to_the_instrument_is_blocked(self):
        """A second scorer beside the first is a modification wearing a hat.

        Without this, `eval/burnscore/compute2.py` plus one import is a rewritten scorer that
        never shows up as a modified file.
        """
        r = self.c([("A", "eval/burnscore/compute2.py")])
        self.assertEqual(len(r["blocked"]), 1)
        self.assertIn("wearing a hat", r["blocked"][0][1])

    def test_deleting_any_of_it_is_blocked(self):
        r = self.c([("D", "eval/burnscore/floor.py")])
        self.assertEqual(len(r["blocked"]), 1)
        self.assertIn("deletes", r["blocked"][0][1])

    def test_a_cartography_pr_may_also_carry_the_kernel_that_needs_the_cell(self):
        """Opening a cell usually means implementing the thing it measures."""
        r = self.c([("A", "eval/cells/BG-2/generation.json"),
                    ("M", "src/cuda/ops_cuda.cu")])
        self.assertEqual(r["blocked"], [])
        self.assertEqual(len(r["cartography"]), 1)
        self.assertEqual(len(r["contributor"]), 1)

    def test_the_guarded_paths_are_the_overlaid_paths(self):
        """A path guarded here but not overlaid by run_from_base.sh is a hole.

        The overlay is what stops an edit affecting its own author's score; the guard is what
        stops it affecting everyone after. They have to cover the same ground or one of them is
        protecting something the other is not.
        """
        script = (ROOT / "eval" / "run_from_base.sh").read_text()
        body = script.split("INSTRUMENT=(", 1)[1].split(")", 1)[0]
        overlaid = {line.strip() for line in body.splitlines() if line.strip()}
        for path in G.INSTRUMENT:
            self.assertIn(path.rstrip("/"), overlaid,
                          f"{path} is guarded but not overlaid from the base ref")


class TestTheGuardDiffsTheTreeItWasPointedAt(unittest.TestCase):
    """The bug that let the first pull request this bot ever saw slip past the guard.

    `instrument_guard.py` resolves its own repository from `__file__`, which is right when a
    contributor runs it by hand in their checkout. The bot runs it against a temporary worktree,
    and passing `cwd=` does not change where `git -C` looks -- so the guard diffed ITSELF,
    found a clean tree, and reported the submission clean. It had edited `floor.py`.

    Nothing about the classifier was wrong. The guard was simply looking at the wrong repo, which
    is the failure mode a guard cannot report on itself.
    """

    def test_changed_accepts_the_repository_to_diff(self):
        import inspect
        for fn in (G.changed, G.existing_generations):
            self.assertIn("repo", inspect.signature(fn).parameters,
                          f"{fn.__name__} cannot be pointed at another checkout, so the bot "
                          f"would diff its own tree instead of the submission's")

    def test_the_bot_points_the_guard_at_the_worktree(self):
        bot = (ROOT / "eval" / "pr_bot.py").read_text()
        self.assertIn('"--repo", str(worktree)', bot,
                      "the bot runs the guard without telling it which tree to diff")


class TestTheGuardRunsAsACommand(unittest.TestCase):
    def test_it_reports_a_clean_tree_against_itself(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "instrument_guard.py"),
                            "--base", "HEAD"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
