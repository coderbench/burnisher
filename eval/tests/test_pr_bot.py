#!/usr/bin/env python3
"""The bot's decisions, tested without GitHub and without a GPU.

Everything the bot decides is a pure function of things already on disk: a guard verdict, a
receipt, a label. Those are what is tested here. What is deliberately NOT tested is the network
-- `gh` calls are the one part that cannot be exercised offline, and mocking them would test the
mock.

The property worth protecting is that the bot adds no judgement of its own. If it ever starts
deciding something that is not derivable from a published artifact, the derivation stops being
checkable by anyone else, and the whole point of publishing the measurements goes with it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import pr_bot as B
from burnscore import verdict as V

RECEIPT = ROOT / "examples" / "BG-1-pr-000001-receipt.json"


class TestTheBotDecidesNothingItself(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(RECEIPT.read_text())

    def test_the_label_it_applies_is_the_one_burnish_verdict_derives(self):
        """Not 'the bot computes a label'. The bot READS one off a pure function.

        A bot that computed its own label would be the only thing that knew how the label was
        reached, and every contributor without a GPU would be asked to trust it.
        """
        self.assertEqual(V.verdict(self.receipt)["label"], V.label_for(self.receipt))

    def test_the_comment_tells_the_reader_how_to_check_it(self):
        v = V.verdict(self.receipt)
        body = B.report(v, self.receipt, raw_name="pr-000042-raw.json",
                        receipt_name="pr-000042.json")
        self.assertIn("burnish audit pr-000042-raw.json pr-000042.json", body)
        self.assertIn("no GPU", body)
        self.assertIn(v["label"], body)
        # The honest boundary is stated on every PR, not buried in a document nobody opens.
        self.assertIn("cannot prove the measurements describe", body)

    def test_the_comment_carries_the_number_and_the_interval(self):
        rec = json.loads(RECEIPT.read_text())
        rec["status"] = "FRONTIER_EXPANDED"
        rec["score"]["credited_gap_closed"] = 0.0342
        rec["score"]["resolved"] = True
        body = B.report(V.verdict(rec), rec, raw_name="r.json", receipt_name="x.json")
        self.assertIn("+0.0342", body)
        self.assertIn("interval", body)

    def test_an_unprovenanced_receipt_is_flagged_on_the_pr(self):
        rec = json.loads(RECEIPT.read_text())
        rec["provenance"] = dict(rec["provenance"], code_provenance_complete=False)
        body = B.report(V.verdict(rec), rec, raw_name="r.json", receipt_name="x.json")
        self.assertIn("not evidence about a particular commit", body)


class TestTheInstrumentSkipNote(unittest.TestCase):
    def test_it_says_skipped_not_rejected_and_points_at_cartography(self):
        """A contributor who reads 'rejected' stops. One who reads 'separated' resubmits.

        The distinction matters more here than in most benchmarks, because one of the two things
        this repository pays for -- opening a cell -- looks superficially like the thing it
        forbids.
        """
        note = B._skip_note({"blocked": [
            {"path": "eval/burnscore/floor.py", "why": "modifies the measuring instrument."}]})
        self.assertIn("not a rejection", note.lower())
        self.assertIn("no GPU time was spent", note)
        self.assertIn("cartography", note.lower())
        self.assertIn("eval/burnscore/floor.py", note)

    def test_the_skip_label_is_not_an_outcome_label(self):
        """Nothing was measured, so it must not sit in the same namespace as a measurement."""
        self.assertNotIn(B.SKIPPED, V.all_labels())
        self.assertTrue(B.SKIPPED.startswith(f"{V.PREFIX}:"))


class TestLabelSetupCoversWhatTheBotCanApply(unittest.TestCase):
    def test_every_label_in_the_code_is_created_by_the_setup_script(self):
        """The setup script reads the list from the code rather than repeating it.

        A hand-maintained second list is how a bot comes to apply a label that does not exist,
        which on GitHub is a silent no-op -- and a silently dropped label costs a contributor
        their credit.
        """
        script = (ROOT / "eval" / "setup_labels.sh").read_text()
        self.assertIn("from burnscore import verdict as V", script)
        self.assertIn("V.all_labels()", script)
        # The one label not in all_labels() must be added explicitly, and is.
        self.assertIn("skipped-instrument", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
