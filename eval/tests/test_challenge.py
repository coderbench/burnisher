#!/usr/bin/env python3
"""Independent re-measurement, and what the ledger does when two boxes disagree.

`burnish audit` is cheap and catches everything arithmetic can catch. What it cannot catch is a
measurement that never happened. The only thing that settles that is somebody else measuring the
same submission on their own hardware -- so this checks that doing so actually changes the
outcome, that it is refused when it would not be evidence, and that agreement is not mistaken
for disagreement.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from burnscore import cells as C
from burnscore import challenge as CH
from burnscore import ledger as L
from burnscore import receipt as R
from tests import fixtures

RECEIPT = ROOT / "examples" / "BG-1-pr-000001-receipt.json"


def _rebuild(rec):
    """Re-stamp a mutated receipt so it verifies, the way a real second run would."""
    rec = copy.deepcopy(rec)
    rec.pop("content_digest", None)
    rec["content_digest"] = R.content_digest(rec)
    return rec


class TestChallenges(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ledger"
        self.gen = C.load(ROOT / "eval" / "cells" / "BG-1" / "generation.json")
        self.canonical = json.loads(RECEIPT.read_text())
        # A paying receipt, so there is credit for a challenge to hold.
        self.canonical["status"] = "FRONTIER_EXPANDED"
        self.canonical["score"]["gap_closed"] = 0.20
        self.canonical["score"]["credited_gap_closed"] = 0.20
        self.canonical["score"]["resolved"] = True
        for cid in self.canonical["per_cell"]:
            self.canonical["per_cell"][cid]["gap_closed"] = 0.20
        self.canonical = _rebuild(self.canonical)
        L.append_receipt(self.root, self.canonical, generation=self.gen)
        self.rid = L.default_receipt_id(self.canonical)

    def tearDown(self):
        self.tmp.cleanup()

    def _counter(self, *, gap=None, floor_fraction=None,
                 uuid="GPU-ffffffff-0000-0000-0000-000000000000"):
        """A second box's receipt for the same submission.

        `floor_fraction` offsets each cell by a fraction of ITS OWN floor, which is the only way
        to express "agrees" across three cells whose floors differ by two orders of magnitude.
        The first version of this helper used one cell's floor for all three and vae-decode --
        the tightest -- correctly reported a disagreement.
        """
        c = copy.deepcopy(self.canonical)
        c["provenance"] = dict(c["provenance"])
        c["provenance"]["device"] = dict(c["provenance"]["device"], uuid=uuid)
        if floor_fraction is not None:
            from burnscore.floor import floor_as_gap_closed
            per = {}
            for k, v in c["per_cell"].items():
                cell = self.gen.cells[k]
                f = floor_as_gap_closed(cell.floor_pct, cell.achieved)
                per[k] = dict(v, gap_closed=v["gap_closed"] + f * floor_fraction)
            c["per_cell"] = per
        else:
            c["score"] = dict(c["score"], gap_closed=gap, credited_gap_closed=gap)
            c["per_cell"] = {k: dict(v, gap_closed=gap) for k, v in c["per_cell"].items()}
        return _rebuild(c)

    def test_an_agreeing_remeasurement_confirms_and_does_not_hold(self):
        """Two boxes measuring the same thing land inside the floor. That is confirmation."""
        CH.attach(self.root, self.canonical, self._counter(floor_fraction=0.5),
                  receipt_id=self.rid)
        st = CH.status_for(self.root, "BG-1", self.rid, self.canonical, self.gen)
        self.assertFalse(st["held"])
        self.assertEqual(st["confirmations"], 1)

        cur = L.update_current(self.root, "BG-1", self.gen)
        self.assertEqual(cur["crediting_receipts"], 1)
        self.assertEqual(cur["independently_confirmed"], 1)
        self.assertAlmostEqual(cur["gap_closed_cumulative"], 0.20, places=9)

    def test_a_disagreeing_remeasurement_holds_the_credit(self):
        """The submission is not rejected and not paid. It is held, and the ledger says why."""
        CH.attach(self.root, self.canonical, self._counter(gap=0.02), receipt_id=self.rid)
        st = CH.status_for(self.root, "BG-1", self.rid, self.canonical, self.gen)
        self.assertTrue(st["held"])
        self.assertIn("does not know which", st["_why_held"])

        cur = L.update_current(self.root, "BG-1", self.gen)
        self.assertEqual(cur["crediting_receipts"], 0, "held credit was paid anyway")
        self.assertEqual(cur["gap_closed_cumulative"], 0.0)
        self.assertEqual(cur["held_receipts"], [self.rid])
        self.assertFalse(cur["history"][0]["paid"])
        self.assertTrue(cur["history"][0]["held"])

    def test_the_threshold_is_the_cells_own_measured_floor(self):
        """Not a tolerance anybody chose.

        The floor is what the calibration measured running the unmodified base against itself,
        so a disagreement inside it is two measurements of the same thing.
        """
        d = CH.disagreement(self.canonical, self._counter(gap=0.02), self.gen)
        row = next(r for r in d["per_cell"] if r["cell"] == "dit-step/1024/bf16")
        from burnscore.floor import floor_as_gap_closed
        cell = self.gen.cells["dit-step/1024/bf16"]
        self.assertAlmostEqual(row["floor_as_gap_closed"],
                               floor_as_gap_closed(cell.floor_pct, cell.achieved), places=12)
        self.assertFalse(row["within_floor"])
        self.assertGreater(row["floors_apart"], 1.0)

    def test_a_rerun_on_the_same_card_is_refused_as_evidence(self):
        """Same silicon, same thermal regime. A repeat, not an independent measurement.

        Two RTX 5090s differ by 3% on achievable GEMM -- larger than most cells' floors -- and
        that spread is exactly what a challenge exists to surface. Re-running on the same card
        cannot surface it.
        """
        same = self._counter(gap=0.02,
                             uuid=self.canonical["provenance"]["device"]["uuid"])
        with self.assertRaises(CH.ChallengeError) as cm:
            CH.attach(self.root, self.canonical, same, receipt_id=self.rid)
        self.assertIn("same physical device", str(cm.exception))

    def test_a_counter_receipt_for_a_different_experiment_is_refused(self):
        """Otherwise a submission could be held hostage by an unrelated result."""
        other = self._counter(gap=0.02)
        other["provenance"] = dict(other["provenance"], candidate_commit="b" * 40)
        other = _rebuild(other)
        with self.assertRaises(CH.ChallengeError) as cm:
            CH.attach(self.root, self.canonical, other, receipt_id=self.rid)
        self.assertIn("same thing", str(cm.exception))

    def test_challenges_are_append_only_like_everything_else(self):
        c = self._counter(gap=0.02)
        CH.attach(self.root, self.canonical, c, receipt_id=self.rid)
        CH.attach(self.root, self.canonical, c, receipt_id=self.rid)      # idempotent
        d = self._counter(gap=0.03)
        with self.assertRaises(CH.ChallengeError):
            CH.attach(self.root, self.canonical, d, receipt_id=self.rid)

    def test_a_ledger_with_no_challenges_behaves_exactly_as_before(self):
        """The mechanism must be invisible until somebody uses it."""
        cur = L.update_current(self.root, "BG-1", self.gen)
        self.assertEqual(cur["held_receipts"], [])
        self.assertEqual(cur["independently_confirmed"], 0)
        self.assertEqual(cur["crediting_receipts"], 1)
        self.assertAlmostEqual(cur["gap_closed_cumulative"], 0.20, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
