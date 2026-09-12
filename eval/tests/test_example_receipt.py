#!/usr/bin/env python3
"""The committed example must still be what the scorer produces from the committed raw file.

Every other test in this directory scores measurements that the test itself invented. Synthetic
records are the right tool for asking whether a guard fires -- you can put the effect exactly
where you want it -- but they share one weakness: they are shaped by the same understanding that
shaped the scorer, so a wrong assumption held in both places cancels out and looks like a pass.

These are real measurements from real hardware, and they are pinned here for two reasons.

The first is that scoring is supposed to be deterministic and hardware-free: the GPU's entire job
is to produce the raw file, and everything after that is arithmetic. If that is true, this test
reproduces a receipt taken off a Blackwell box on a CI runner with no GPU at all. If it ever
stops being true, something in the scoring path has started depending on where it runs.

The second is drift. examples/README.md quotes figures from this receipt. A generated document
that quotes a number nobody regenerates becomes wrong quietly, and this repository's rule is that
a figure is never typed by hand -- so the example is checked against the scorer that claims to
produce it, on every run.
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

from burnscore import cells as C
from burnscore import receipt as R

EXAMPLES = ROOT / "examples"
RAW = EXAMPLES / "BG-1-pr-000001-raw.json"
RECEIPT = EXAMPLES / "BG-1-pr-000001-receipt.json"

# Two fields cannot match and must not be asserted on: a receipt is stamped with the wall clock
# at the moment it is built, and its digest covers that stamp. Everything else is a function of
# the raw measurements alone -- which is the property under test.
VOLATILE = ("timestamp_utc", "content_digest")


class TestTheCommittedExampleStillReproduces(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(RAW.read_text())
        self.want = json.loads(RECEIPT.read_text())
        self.gen = C.load(ROOT / "eval" / "cells" / "BG-1" / "generation.json")

    def _rescore(self):
        """Through `burnish score`, not through a reimplementation of it.

        Rebuilding the call to build_receipt() here would be a second copy of the CLI's wiring,
        and a test that agrees with its own copy of the wiring proves nothing about the command
        anybody actually runs. (The first draft of this test did exactly that and passed the
        wrong thing for held-out records.)
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "receipt.json"
            r = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "burnish"), "score", str(RAW),
                 "--generation", "BG-1", "--output", str(out),
                 "--ledger", str(Path(tmp) / "ledger"), "--pr", str(self.want["pr"])],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return json.loads(out.read_text())

    def test_the_receipt_reproduces_field_for_field(self):
        got = self._rescore()
        self.assertEqual(set(got), set(self.want), "the receipt's shape changed")
        for key in sorted(set(got) - set(VOLATILE)):
            self.assertEqual(got[key], self.want[key],
                             f"scoring the committed raw measurements no longer reproduces the "
                             f"committed receipt at {key!r}. Either the scorer changed meaning "
                             f"-- in which case past receipts no longer mean what they said, "
                             f"and that is a new generation, not an edit -- or this is a bug.")

    def test_the_receipt_verifies_against_itself_and_the_generation(self):
        R.verify_receipt(self.want, self.gen)

    def test_the_raw_file_is_paired_interleaved_and_complete(self):
        """The properties that make these records a comparison rather than two measurements."""
        recs = self.raw["records"]
        arms = {}
        for r in recs:
            arms.setdefault((r["cell"], r["variant"]), set()).add(r["repeat"])
        cells = sorted({c for c, _ in arms})
        self.assertEqual(cells, sorted(c.id for c in self.gen.scorable_cells()),
                         "a partial matrix credits nothing; this example must not be one")
        for cell in cells:
            base, cand = arms[(cell, "base")], arms[(cell, "candidate")]
            self.assertEqual(base, cand,
                             f"{cell}: the arms do not share repeat indices, so the pairing the "
                             f"bootstrap depends on does not exist")
            self.assertGreaterEqual(len(base), 2, f"{cell}: a spread needs repeats")
        self.assertTrue(self.raw["held_out"])
        self.assertTrue(self.raw["held_out_shape"])

    def test_the_readmes_claims_about_this_run_are_true_of_this_receipt(self):
        """Prose drifts. The specific claims examples/README.md makes are checked here."""
        r = self.want
        # "It lost, and the receipt says so" -- and a loss credits zero rather than negative.
        self.assertLess(r["score"]["gap_closed"], 0.0)
        self.assertEqual(r["score"]["credited_gap_closed"], 0.0)
        self.assertEqual(r["status"], "UNRESOLVED")
        # "Two cells resolved a regression and are named."
        resolved = {c for c, v in r["per_cell"].items() if v["resolved"]}
        self.assertEqual(resolved, {"dit-step/1024/bf16", "t5-encode/1024/bf16"})
        # "vae-decode did not resolve ... contributes zero and does not block."
        vae = r["per_cell"]["vae-decode/1024/bf16"]
        self.assertFalse(vae["resolved"])
        self.assertTrue(r["coverage"]["complete"],
                        "an unresolved cell must not drop out of the matrix")
        # "not admissible" -- the example is explicitly unprovenanced, and says so.
        self.assertFalse(r["provenance"]["code_provenance_complete"])

    def test_scoring_does_not_touch_the_committed_tree(self):
        """It writes a ledger; the ledger is written outside the worktree, never into it."""
        before = sorted(p.name for p in (ROOT / "eval" / "cells").iterdir())
        self._rescore()
        self.assertEqual(before, sorted(p.name for p in (ROOT / "eval" / "cells").iterdir()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
